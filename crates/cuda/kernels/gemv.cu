// Skinny GEMM ("GEMV") for decode-sized batches: Y[M, N] = X[M, K] . W[N, K]^T with M <= 16.
//
// Weights are streamed once with 16-byte loads; each warp owns output rows and walks its K range,
// every lane accumulating a fixed strided subset in FP32, followed by a fixed-order warp reduction.
// When N is small the K range is split across CTAs; partial sums [split, M, N] are then reduced in
// split order. The result for a token never depends on M or on which other tokens share the batch.
//
// Weight formats (fmt): 0 BF16 [N, K]; 1 FP8 E4M3 [N, K] and 2 INT8 [N, K], each with an FP32 scale per row;
// 3 INT8 [N, K] with an FP32 scale per 64 consecutive weights of a row.

#include <cuda_fp8.h>

#include "variants.cuh"
#include "wfmt.cuh"

namespace {

constexpr int WARPS = 8, THREADS = WARPS * 32;

// grid (ceil(N / (NW R)), splits). Each split covers k_per_split columns (a multiple of 512). A warp owns
// R rows; lane l handles the 16-weight chunks at columns k_begin + 16 l + 512 c. The first PRE chunks
// of every row (all of them for K <= 2560 per split) are loaded raw before the PDL wait, so the weight
// stream overlaps the previous kernel; activations are read only after it. NW warps per CTA. Neither
// R, PRE nor NW changes any output's arithmetic.
template <WFmt F, int M, typename XT, int R, int PRE = 5, int NW = WARPS>
__global__ void __launch_bounds__(NW * 32) gemv_kernel(const XT* __restrict__ X, int ldx,
                                                       const uint8_t* __restrict__ W, const float* __restrict__ row_scale,
                                                       float* Y, int ldy, float* partial, int ldp, int N,
                                                       int K, int k_per_split, float alpha, int accumulate) {
    pdl_launch_dependents();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int k_begin = blockIdx.y * k_per_split, k_end = min(K, k_begin + k_per_split);
    const size_t row_bytes = (size_t)K * elem_bytes<F>();
    const int n0 = (blockIdx.x * NW + warp) * R;
    const uint8_t* rows[R];
    bool row_live[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        row_live[r] = n0 + r < N;
        rows[r] = W + (size_t)min(n0 + r, N - 1) * row_bytes;
    }
    Raw16<F> pre[R][PRE];
    float pre_gs[R][PRE];
#pragma unroll
    for (int c = 0; c < PRE; ++c) {
        const int k = k_begin + lane * 16 + c * 512;
        if (k < k_end)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                pre[r][c] = load_raw16<F>(rows[r], k);
                if constexpr (F == WFmt::INT8G)
                    pre_gs[r][c] = __ldg(row_scale + (size_t)min(n0 + r, N - 1) * (K / GROUP) + k / GROUP);
                if constexpr (F == WFmt::MX8)
                    pre_gs[r][c] = ue8m0_value(reinterpret_cast<const uint8_t*>(row_scale)[(size_t)min(n0 + r, N - 1) * (K / 32) + k / 32]);
            }
    }
    pdl_wait();

    float acc[R][M] = {};
    auto consume = [&](int k, float (&w)[R][16]) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float xf[16];
            load_x8(X + (size_t)m * ldx + k, xf);
            load_x8(X + (size_t)m * ldx + k + 8, xf + 8);
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int i = 0; i < 16; ++i) acc[r][m] = fmaf(w[r][i], xf[i], acc[r][m]);
        }
    };
#pragma unroll
    for (int c = 0; c < PRE; ++c) {
        const int k = k_begin + lane * 16 + c * 512;
        if (k >= k_end) break;
        float w[R][16];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            decode_raw16<F>(pre[r][c], w[r]);
            if constexpr (F == WFmt::INT8G || F == WFmt::MX8)
#pragma unroll
                for (int i = 0; i < 16; ++i) w[r][i] *= pre_gs[r][c];
        }
        consume(k, w);
    }
    for (int k = k_begin + lane * 16 + PRE * 512; k < k_end; k += 512) {
        float w[R][16];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            decode_raw16<F>(load_raw16<F>(rows[r], k), w[r]);
            if constexpr (F == WFmt::INT8G) {
                const float gs = __ldg(row_scale + (size_t)min(n0 + r, N - 1) * (K / GROUP) + k / GROUP);
#pragma unroll
                for (int i = 0; i < 16; ++i) w[r][i] *= gs;
            }
            if constexpr (F == WFmt::MX8) {
                const float gs = ue8m0_value(reinterpret_cast<const uint8_t*>(row_scale)[(size_t)min(n0 + r, N - 1) * (K / 32) + k / 32]);
#pragma unroll
                for (int i = 0; i < 16; ++i) w[r][i] *= gs;
            }
        }
        consume(k, w);
    }
    auto store = [&](int r, int m, float v) {
        const int n = n0 + r;
        const float scale = alpha * (F == WFmt::FP8 || F == WFmt::INT8 ? row_scale[n] : 1.0f);
        if (gridDim.y == 1) {
            float* y = &Y[(size_t)m * ldy + n];
            *y = accumulate ? *y + v * scale : v * scale;
        } else {
            partial[((size_t)blockIdx.y * M + m) * ldp + n] = v * scale;
        }
    };
    constexpr int NS = R * M;
    if constexpr ((NS & (NS - 1)) == 0 && NS <= 32) {
        // All R x M sums at once (slot r * M + m), each bitwise as its own warp_sum.
        float v[NS];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int m = 0; m < M; ++m) v[r * M + m] = acc[r][m];
        const float sum = warp_sum_slots(v);
        const int slot = lane / (32 / NS), r = slot / M, m = slot % M;
        if (lane % (32 / NS) == 0 && (r == 0 ? row_live[0] : row_live[R - 1])) store(r, m, sum);
    } else {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int m = 0; m < M; ++m) acc[r][m] = warp_sum(acc[r][m]);
            if (lane == 0 && row_live[r])
#pragma unroll
                for (int m = 0; m < M; ++m) store(r, m, acc[r][m]);
        }
    }
}

// Y[m, n] (+)= sum over splits of partial[split, m, n], in split order.
__global__ void reduce_kernel(const float* __restrict__ partial, int ldp, int splits, int M, int N, float* Y, int ldy,
                              int accumulate) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    const int m = i / N, n = i % N;
    float acc = 0;
    for (int s = 0; s < splits; ++s) acc += partial[((size_t)s * M + m) * ldp + n];
    float* y = &Y[(size_t)m * ldy + n];
    *y = accumulate ? *y + acc : acc;
}

template <WFmt F, int M, int R, int PRE, int NW>
cudaError_t launch_one(const __nv_bfloat16* X, int ldx, const uint8_t* W, const float* scale, float* Y, int ldy, float* partial,
                       int N, int K, int splits, int k_per_split, float alpha, int accumulate, cudaStream_t st) {
    const dim3 grid((N + NW * R - 1) / (NW * R), splits);
    SQ_CHECK_LAUNCH(sq_launch(gemv_kernel<F, M, __nv_bfloat16, R, PRE, NW>, grid, NW * 32, 0, st, X, ldx, W, scale, Y, ldy, partial, N,
                              N, K, k_per_split, alpha, accumulate));
    return cudaSuccess;
}

// Variants (kbench, dev/KBENCH.md) for decode (M = 1) and verify rounds (M = 4): gemv.pre in {3, 5, 8},
// gemv.warps in {4, 8, 16}, gemv.rows in {1, 2, 4} (0: the default choice below).
template <WFmt F, int M, int R>
cudaError_t launch_r(int pre, int nw, const __nv_bfloat16* X, int ldx, const uint8_t* W, const float* scale, float* Y, int ldy,
                     float* partial, int N, int K, int splits, int k_per_split, float alpha, int accumulate, cudaStream_t st) {
#define SQ_ARGS X, ldx, W, scale, Y, ldy, partial, N, K, splits, k_per_split, alpha, accumulate, st
    if constexpr (M == 1 || M == 4) {
        switch (pre * 100 + nw) {
            case 304: return launch_one<F, M, R, 3, 4>(SQ_ARGS);
            case 308: return launch_one<F, M, R, 3, 8>(SQ_ARGS);
            case 316: return launch_one<F, M, R, 3, 16>(SQ_ARGS);
            case 504: return launch_one<F, M, R, 5, 4>(SQ_ARGS);
            case 508: return launch_one<F, M, R, 5, 8>(SQ_ARGS);
            case 516: return launch_one<F, M, R, 5, 16>(SQ_ARGS);
            case 804: return launch_one<F, M, R, 8, 4>(SQ_ARGS);
            case 808: return launch_one<F, M, R, 8, 8>(SQ_ARGS);
            case 816: return launch_one<F, M, R, 8, 16>(SQ_ARGS);
            default: return cudaErrorInvalidValue;
        }
    } else {
        if (pre != 5 || nw != WARPS) return cudaErrorInvalidValue;
        return launch_one<F, M, R, 5, WARPS>(SQ_ARGS);
    }
#undef SQ_ARGS
}

template <WFmt F, int M>
cudaError_t launch_m(const __nv_bfloat16* X, int ldx, const uint8_t* W, const float* scale, float* Y, int ldy,
                     float* partial, int N, int K, int splits, float alpha, int accumulate, cudaStream_t st) {
    const int k_per_split = ((K + splits - 1) / splits + 511) / 512 * 512;
    splits = (K + k_per_split - 1) / k_per_split;
    const int variant = M == 1 || M == 4;
    const int pre = variant ? sq_variant("gemv.pre", 5) : 5, nw = variant ? sq_variant("gemv.warps", WARPS) : WARPS;
    int rows = variant ? sq_variant("gemv.rows", 0) : 0;
    // Default: two rows per warp when that still gives at least one CTA per SM, otherwise one; four-row
    // steps (MTP verify rounds) always take two (kbench: 1.1% faster per verify step).
    if (rows == 0) rows = M == 4 || (N + 2 * nw - 1) / (2 * nw) * splits >= 188 ? 2 : 1;
    cudaError_t e;
    if (rows == 1) e = launch_r<F, M, 1>(pre, nw, X, ldx, W, scale, Y, ldy, partial, N, K, splits, k_per_split, alpha, accumulate, st);
    else if (rows == 2) e = launch_r<F, M, 2>(pre, nw, X, ldx, W, scale, Y, ldy, partial, N, K, splits, k_per_split, alpha, accumulate, st);
    else if (rows == 4 && variant) e = launch_r<F, M, 4>(pre, nw, X, ldx, W, scale, Y, ldy, partial, N, K, splits, k_per_split, alpha, accumulate, st);
    else e = cudaErrorInvalidValue;
    if (e != cudaSuccess) return e;
    if (splits > 1)
        SQ_CHECK_LAUNCH(sq_launch(reduce_kernel, (M * N + 255) / 256, 256, 0, st, partial, N, splits, M, N, Y, ldy, accumulate));
    return cudaGetLastError();
}

// M in 1..16.
template <WFmt F>
cudaError_t launch_rows(int M, const __nv_bfloat16* X, int ldx, const uint8_t* W, const float* scale, float* Y, int ldy,
                        float* partial, int N, int K, int splits, float alpha, int accumulate, cudaStream_t st) {
    switch (M) {
#define SQ_CASE(m) \
    case m:        \
        return launch_m<F, m>(X, ldx, W, scale, Y, ldy, partial, N, K, splits, alpha, accumulate, st);
        SQ_CASE(1) SQ_CASE(2) SQ_CASE(3) SQ_CASE(4) SQ_CASE(5) SQ_CASE(6) SQ_CASE(7) SQ_CASE(8)
        SQ_CASE(9) SQ_CASE(10) SQ_CASE(11) SQ_CASE(12) SQ_CASE(13) SQ_CASE(14) SQ_CASE(15) SQ_CASE(16)
#undef SQ_CASE
        default:
            return cudaErrorInvalidValue;
    }
}

template <WFmt F>
cudaError_t launch(int M, const void* Xv, int ldx, const void* Wv, const float* scale, float* Y, int ldy, float* partial,
                   int N, int K, int splits, float alpha, int accumulate, cudaStream_t st) {
    if (K % 16 != 0 || ldx % 8 != 0) return cudaErrorInvalidValue;
    const auto* X = static_cast<const __nv_bfloat16*>(Xv);
    const auto* W = static_cast<const uint8_t*>(Wv);
    return launch_rows<F>(M, X, ldx, W, scale, Y, ldy, partial, N, K, splits, alpha, accumulate, st);
}

}  // namespace

extern "C" int quench6k_gemv_max_m() { return 16; }

// fmt: 0 BF16, 3 INT8 with one scale per 64 weights. With splits > 1 the K range is split across CTAs;
// `partial` holds splits * M * N floats, reduced in split order.
extern "C" cudaError_t quench6k_gemv(int fmt, int M, const void* X, int ldx, const void* W, const float* scale, float* Y,
                                   int ldy, float* partial, int N, int K, int splits, float alpha, int accumulate,
                                   cudaStream_t st) {
    if (M == 0) return cudaSuccess;
    if (fmt == 0) return launch<WFmt::BF16>(M, X, ldx, W, scale, Y, ldy, partial, N, K, splits, alpha, accumulate, st);
    if (fmt == 3 && K % GROUP == 0)
        return launch<WFmt::INT8G>(M, X, ldx, W, scale, Y, ldy, partial, N, K, splits, alpha, accumulate, st);
    if (fmt == 4 && K % 32 == 0)
        return launch<WFmt::MX8>(M, X, ldx, W, scale, Y, ldy, partial, N, K, splits, alpha, accumulate, st);
    return cudaErrorInvalidValue;
}

namespace {

// BF16 [N, K] -> INT8 [N, K] with scale[n, g] = absmax(row n, group g) / 127. One warp per (row, group of 128).
__global__ void quantize_int8_groups_kernel(const __nv_bfloat16* __restrict__ w, int K, int8_t* q, float* scale) {
    pdl_launch_dependents();
    pdl_wait();
    const int groups = K / GROUP;
    const int item = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    const int n = item / groups, g = item % groups;
    const __nv_bfloat16* src = w + (size_t)n * K + g * GROUP;
    float v[GROUP / 32], mx = 0;
#pragma unroll
    for (int i = 0; i < GROUP / 32; ++i) {
        v[i] = bf2f(src[i * 32 + lane]);
        mx = fmaxf(mx, fabsf(v[i]));
    }
    mx = warp_max(mx);
    const float s = mx > 0 ? mx / 127.0f : 1.0f;
    if (lane == 0) scale[(size_t)n * groups + g] = s;
#pragma unroll
    for (int i = 0; i < GROUP / 32; ++i)
        q[(size_t)n * K + g * GROUP + i * 32 + lane] = (int8_t)fmaxf(-127.0f, fminf(127.0f, rintf(v[i] / s)));
}

}  // namespace

extern "C" int quench6k_int8_group() { return GROUP; }

extern "C" cudaError_t quench6k_quantize_int8_groups(const void* w, int N, int K, void* q, float* scale, cudaStream_t st) {
    if (N == 0) return cudaSuccess;
    if (K % GROUP != 0) return cudaErrorInvalidValue;
    const int items = N * (K / GROUP);
    SQ_CHECK_LAUNCH(sq_launch(quantize_int8_groups_kernel, (items + 7) / 8, 256, 0, st, static_cast<const __nv_bfloat16*>(w), K,
                                                                  static_cast<int8_t*>(q), scale));
    SQ_RETURN_LAUNCH();
}

namespace {

// out[n, k] = bf16(q[n, k] * scale[n, k / GROUP]): the BF16 values the GEMM stages from INT8-group
// weights (same product and rounding), so a BF16 GEMM on them gives identical results. 8 per thread.
__global__ void dequant_int8_groups_kernel(const int8_t* __restrict__ q, const float* __restrict__ scale, int K, size_t count,
                                           __nv_bfloat16* out) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * 8;
    if (i >= count) return;
    const float s = scale[i / GROUP];  // rows are whole groups: flat index / GROUP is (n, k / GROUP)
    const uint2 raw = *reinterpret_cast<const uint2*>(q + i);
    const int8_t* p = reinterpret_cast<const int8_t*>(&raw);
    uint32_t packed[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __nv_bfloat162 v = __floats2bfloat162_rn((float)p[2 * j] * s, (float)p[2 * j + 1] * s);
        packed[j] = *reinterpret_cast<const uint32_t*>(&v);
    }
    *reinterpret_cast<uint4*>(out + i) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
}

}  // namespace

extern "C" cudaError_t quench6k_dequant_int8_groups(const void* q, const float* scale, int N, int K, void* out, cudaStream_t st) {
    if (N == 0) return cudaSuccess;
    if (K % GROUP != 0) return cudaErrorInvalidValue;
    const size_t count = (size_t)N * K;
    SQ_CHECK_LAUNCH(sq_launch(dequant_int8_groups_kernel, (unsigned)((count / 8 + 255) / 256), 256, 0, st, static_cast<const int8_t*>(q), scale, K,
                              count, static_cast<__nv_bfloat16*>(out)));
    SQ_RETURN_LAUNCH();
}

namespace {

// W[n, k] = bf16(W[n, k] * (1 + g[k])): folds a (1 + w)-style norm weight into a projection's columns.
__global__ void fold_norm_kernel(__nv_bfloat16* w, const __nv_bfloat16* __restrict__ g, int K, size_t count) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) w[i] = f2bf(bf2f(w[i]) * (1.0f + bf2f(g[i % K])));
}

}  // namespace

extern "C" cudaError_t quench6k_fold_norm(void* w, const void* g, int N, int K, cudaStream_t st) {
    const size_t count = (size_t)N * K;
    if (count == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(fold_norm_kernel, (unsigned)((count + 255) / 256), 256, 0, st, static_cast<__nv_bfloat16*>(w),
                                                                      static_cast<const __nv_bfloat16*>(g), K, count));
    SQ_RETURN_LAUNCH();
}
