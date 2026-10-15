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

#include "wfmt.cuh"

namespace {

constexpr int WARPS = 8, THREADS = WARPS * 32;

// grid (ceil(N / 16), splits). Each split covers k_per_split columns (a multiple of 512). A warp owns
// ROWS_PER_WARP rows and walks them together: per step every lane loads two 16-weight chunks of each
// row (all in flight at once) and one activation chunk per token, reused across the rows. When `xsq`
// is set, blocks with blockIdx.x == 0 also write each token's sum of squared activations over their
// K range to xsq[split * M + m].
template <WFmt F, int M, typename XT, int R>
__global__ void __launch_bounds__(THREADS) gemv_kernel(const XT* __restrict__ X, int ldx,
                                                       const uint8_t* __restrict__ W, const float* __restrict__ row_scale,
                                                       float* Y, int ldy, float* partial, int ldp, float* xsq, int N,
                                                       int K, int k_per_split, float alpha, int accumulate) {
    pdl_launch_dependents();
    constexpr int U = 2;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int k_begin = blockIdx.y * k_per_split, k_end = min(K, k_begin + k_per_split);
    const size_t row_bytes = (size_t)K * elem_bytes<F>();
    const int n0 = (blockIdx.x * WARPS + warp) * R;
    const uint8_t* rows[R];
    bool row_live[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        row_live[r] = n0 + r < N;
        rows[r] = W + (size_t)min(n0 + r, N - 1) * row_bytes;
    }
    pdl_wait();
    float acc[R][M] = {};
    float sq[M] = {};
    for (int k0 = k_begin + lane * 16; k0 < k_end; k0 += U * 512) {
        float w[R][U][16];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int k = k0 + u * 512;
            if (k < k_end)
#pragma unroll
                for (int r = 0; r < R; ++r) load16<F>(rows[r], k, w[r][u]);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int k = k0 + u * 512;
            if (k >= k_end) continue;
            if constexpr (F == WFmt::INT8G) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float gs = __ldg(row_scale + (size_t)min(n0 + r, N - 1) * (K / GROUP) + k / GROUP);
#pragma unroll
                    for (int i = 0; i < 16; ++i) w[r][u][i] *= gs;
                }
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float xf[16];
                load_x8(X + (size_t)m * ldx + k, xf);
                load_x8(X + (size_t)m * ldx + k + 8, xf + 8);
                if (xsq) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) sq[m] = fmaf(xf[i], xf[i], sq[m]);
                }
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int i = 0; i < 16; ++i) acc[r][m] = fmaf(w[r][u][i], xf[i], acc[r][m]);
            }
        }
    }
    if (xsq && blockIdx.x == 0 && warp == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float v = warp_sum(sq[m]);
            if (lane == 0) xsq[blockIdx.y * M + m] = v;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int n = n0 + r;
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = warp_sum(acc[r][m]);
        if (lane == 0 && row_live[r]) {
            const float scale = alpha * (F == WFmt::FP8 || F == WFmt::INT8 ? row_scale[n] : 1.0f);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                if (gridDim.y == 1) {
                    float* y = &Y[(size_t)m * ldy + n];
                    *y = accumulate ? *y + acc[r][m] * scale : acc[r][m] * scale;
                } else {
                    partial[((size_t)blockIdx.y * M + m) * ldp + n] = acc[r][m] * scale;
                }
            }
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

// x_f32: X holds FP32 activations. reduce = 0 leaves split partials in `partial` for the caller.
template <WFmt F, int M>
cudaError_t launch_m(const void* X, int x_f32, int ldx, const void* W, const float* scale, float* Y, int ldy,
                     float* partial, int ldp, float* xsq, int N, int K, int splits, float alpha, int accumulate,
                     int reduce, cudaStream_t st) {
    const int k_per_split = ((K + splits - 1) / splits + 511) / 512 * 512;
    splits = (K + k_per_split - 1) / k_per_split;
    // Two rows per warp when that still gives at least one CTA per SM; otherwise one.
    const bool two = (N + 2 * WARPS - 1) / (2 * WARPS) * splits >= 188;
    const int rows_per_cta = WARPS * (two ? 2 : 1);
    dim3 grid((N + rows_per_cta - 1) / rows_per_cta, splits);
    const auto* w8 = static_cast<const uint8_t*>(W);
#define SQ_GEMV(XT, R) \
    SQ_CHECK_LAUNCH(sq_launch(gemv_kernel<F, M, XT, R>, grid, THREADS, 0, st, static_cast<const XT*>(X), ldx, w8, scale, Y, ldy, partial, ldp, \
                                                        xsq, N, K, k_per_split, alpha, accumulate))
    if (x_f32) {
        if (two) SQ_GEMV(float, 2);
        else SQ_GEMV(float, 1);
    } else {
        if (two) SQ_GEMV(__nv_bfloat16, 2);
        else SQ_GEMV(__nv_bfloat16, 1);
    }
#undef SQ_GEMV
    if (splits > 1 && reduce)
        SQ_CHECK_LAUNCH(sq_launch(reduce_kernel, (M * N + 255) / 256, 256, 0, st, partial, ldp, splits, M, N, Y, ldy, accumulate));
    return cudaGetLastError();
}

template <WFmt F>
cudaError_t launch(int M, const void* X, int x_f32, int ldx, const void* W, const float* scale, float* Y, int ldy,
                   float* partial, int ldp, float* xsq, int N, int K, int splits, float alpha, int accumulate, int reduce,
                   cudaStream_t st) {
    if (K % 8 != 0 || ldx % 8 != 0) return cudaErrorInvalidValue;
    switch (M) {
#define SQ_CASE(m) \
    case m:        \
        return launch_m<F, m>(X, x_f32, ldx, W, scale, Y, ldy, partial, ldp, xsq, N, K, splits, alpha, accumulate, reduce, st);
        SQ_CASE(1) SQ_CASE(2) SQ_CASE(3) SQ_CASE(4) SQ_CASE(5) SQ_CASE(6) SQ_CASE(7) SQ_CASE(8)
        SQ_CASE(9) SQ_CASE(10) SQ_CASE(11) SQ_CASE(12) SQ_CASE(13) SQ_CASE(14) SQ_CASE(15) SQ_CASE(16)
#undef SQ_CASE
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace

extern "C" int squash_gemv_max_m() { return 16; }

// With splits > 1, partial sums go to partial[(split * M + m) * ldp + n] (ldp >= N). With reduce = 0 they
// are left for the caller (Y untouched); the effective split count then must be exactly `splits`.
// xsq (optional): per-split sums of squared activations, [splits, M].
extern "C" cudaError_t squash_gemv(int fmt, int M, const void* X, int x_f32, int ldx, const void* W, const float* scale,
                                   float* Y, int ldy, float* partial, int ldp, float* xsq, int N, int K, int splits,
                                   float alpha, int accumulate, int reduce, cudaStream_t st) {
    if (M == 0) return cudaSuccess;
#define SQ_ARGS M, X, x_f32, ldx, W, scale, Y, ldy, partial, ldp, xsq, N, K, splits, alpha, accumulate, reduce, st
    if (fmt == 0) return launch<WFmt::BF16>(SQ_ARGS);
    if (fmt == 1) return launch<WFmt::FP8>(SQ_ARGS);
    if (fmt == 2) return launch<WFmt::INT8>(SQ_ARGS);
    if (K % GROUP != 0) return cudaErrorInvalidValue;
    return launch<WFmt::INT8G>(SQ_ARGS);
#undef SQ_ARGS
}

namespace {

// BF16 [N, K] -> FP8 E4M3 [N, K] with scale[n] = absmax(row) / 448. One block per row.
__global__ void quantize_fp8_rows_kernel(const __nv_bfloat16* __restrict__ w, int K, uint8_t* q, float* scale) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const __nv_bfloat16* row = w + (size_t)blockIdx.x * K;
    float mx = 0;
    for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(bf2f(row[k])));
    mx = block_max(mx, scratch);
    const float s = mx > 0 ? mx / 448.0f : 1.0f;
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        q[(size_t)blockIdx.x * K + k] = __nv_cvt_float_to_fp8(bf2f(row[k]) / s, __NV_SATFINITE, __NV_E4M3);
}

}  // namespace

extern "C" cudaError_t squash_quantize_fp8_rows(const void* w, int N, int K, void* q, float* scale, cudaStream_t st) {
    if (N == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(quantize_fp8_rows_kernel, N, 256, 0, st, static_cast<const __nv_bfloat16*>(w), K, static_cast<uint8_t*>(q), scale));
    SQ_RETURN_LAUNCH();
}

namespace {

// BF16 [N, K] -> INT8 [N, K] with scale[n] = absmax(row) / 127, round to nearest. One block per row.
__global__ void quantize_int8_rows_kernel(const __nv_bfloat16* __restrict__ w, int K, int8_t* q, float* scale) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const __nv_bfloat16* row = w + (size_t)blockIdx.x * K;
    float mx = 0;
    for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(bf2f(row[k])));
    mx = block_max(mx, scratch);
    const float s = mx > 0 ? mx / 127.0f : 1.0f;
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        q[(size_t)blockIdx.x * K + k] = (int8_t)fmaxf(-127.0f, fminf(127.0f, rintf(bf2f(row[k]) / s)));
}

}  // namespace

extern "C" cudaError_t squash_quantize_int8_rows(const void* w, int N, int K, void* q, float* scale, cudaStream_t st) {
    if (N == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(quantize_int8_rows_kernel, N, 256, 0, st, static_cast<const __nv_bfloat16*>(w), K, static_cast<int8_t*>(q), scale));
    SQ_RETURN_LAUNCH();
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

extern "C" int squash_int8_group() { return GROUP; }

extern "C" cudaError_t squash_quantize_int8_groups(const void* w, int N, int K, void* q, float* scale, cudaStream_t st) {
    if (N == 0) return cudaSuccess;
    if (K % GROUP != 0) return cudaErrorInvalidValue;
    const int items = N * (K / GROUP);
    SQ_CHECK_LAUNCH(sq_launch(quantize_int8_groups_kernel, (items + 7) / 8, 256, 0, st, static_cast<const __nv_bfloat16*>(w), K,
                                                                  static_cast<int8_t*>(q), scale));
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

extern "C" cudaError_t squash_fold_norm(void* w, const void* g, int N, int K, cudaStream_t st) {
    const size_t count = (size_t)N * K;
    if (count == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(fold_norm_kernel, (unsigned)((count + 255) / 256), 256, 0, st, static_cast<__nv_bfloat16*>(w),
                                                                      static_cast<const __nv_bfloat16*>(g), K, count));
    SQ_RETURN_LAUNCH();
}
