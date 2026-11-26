// Nemotron-H small steps of several rows: Y[m, n] (+)= alpha * X[m, :] . W[n, :]^T for any number of rows,
// bitwise as the shared GEMV (gemv.cu, no K split) computes each row alone, at a fraction of its cost when
// there are several rows. The kernel is Hy4's wide GEMV (hy4_gemv.cu) with the INT8-group format and the
// shared GEMV's output factor and accumulation added.
//
// The arithmetic of a row is the GEMV's: lane l accumulates in FP32 the weights at columns 16 l + 512 c
// (c ascending, the 16 columns of a chunk in order; FP8 block-scaled and INT8 group-scaled weights decoded
// and multiplied by their scale first), then a butterfly warp sum, then y = sum * alpha (+ y). What differs
// is the work per instruction: a warp takes R weight rows and MT input rows at once, decodes each weight once
// for MT inputs and converts each input once for R weights (half a chunk at a time, which keeps every
// accumulator's order), and the CTAs of one weight tile (one per group of MT input rows) are adjacent, so the
// weights stream from DRAM about once per step.

#include <cuda_fp8.h>

#include "wfmt.cuh"

namespace {

constexpr int NW = 4;  // warps per CTA

// Raw weights of one row's chunk (16 columns): 16 bytes of an 8-bit format, 32 of BF16.
template <WFmt F>
struct Chunk {
    uint4 v[F == WFmt::BF16 ? 2 : 1];
};

template <WFmt F>
SQ_DEVICE Chunk<F> load_chunk(const uint8_t* row, int k) {
    Chunk<F> c;
    if constexpr (F == WFmt::BF16) {
        c.v[0] = ld_stream(row + (size_t)k * 2);
        c.v[1] = ld_stream(row + (size_t)k * 2 + 16);
    } else {
        c.v[0] = ld_stream(row + k);
    }
    return c;
}

// Columns 8 h .. 8 h + 7 of a chunk as floats, as decode_raw16 gives them.
template <WFmt F>
SQ_DEVICE void decode_half(const Chunk<F>& c, int h, float* w) {
    if constexpr (F == WFmt::BF16) {
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&c.v[h]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(p[i]);
            w[2 * i] = f.x;
            w[2 * i + 1] = f.y;
        }
    } else if constexpr (F == WFmt::FP8B) {
        const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&c.v[0]) + 4 * h;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3);
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
            w[2 * i] = f.x;
            w[2 * i + 1] = f.y;
        }
    } else {
        const int8_t* p = reinterpret_cast<const int8_t*>(&c.v[0]) + 8 * h;
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = (float)p[i];
    }
}

// Grid: (tiles of NW * R weight rows) x groups of MT input rows, the group index fastest. `mrows` input
// rows in all (the last group may hold fewer). BF16, FP8B or INT8G weights.
template <WFmt F, int MT, int R, int PRE = (MT >= 16 ? 1 : 8 / R)>
__global__ void __launch_bounds__(NW * 32) wide_kernel(const __nv_bfloat16* __restrict__ X, int ldx, const uint8_t* __restrict__ W,
                                                       const float* __restrict__ scale, float* Y, int ldy, int N, int K, int mrows, float alpha,
                                                       int accumulate) {
    pdl_launch_dependents();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int groups = (mrows + MT - 1) / MT, g = blockIdx.x % groups;
    const int mv = min(MT, mrows - g * MT);
    X += (size_t)g * MT * ldx;
    Y += (size_t)g * MT * ldy;
    const int n0 = (blockIdx.x / groups * NW + warp) * R;
    const size_t row_bytes = (size_t)K * elem_bytes<F>();
    const uint8_t* rows[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rows[r] = W + (size_t)min(n0 + r, N - 1) * row_bytes;

    float acc[R][MT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[r][m] = 0.0f;

    // One chunk (16 columns at k) of the R weight rows into the MT rows' sums.
    auto consume = [&](const Chunk<F> (&c)[R], int k) {
        float gs[R];
        if constexpr (F == WFmt::FP8B)
#pragma unroll
            for (int r = 0; r < R; ++r) gs[r] = fp8b_scale(scale, min(n0 + r, N - 1), k, K);
        if constexpr (F == WFmt::INT8G)
#pragma unroll
            for (int r = 0; r < R; ++r) gs[r] = ld_stream_f32(scale + (size_t)min(n0 + r, N - 1) * (K / GROUP) + k / GROUP);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float w[R][8];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                decode_half<F>(c[r], h, w[r]);
                if constexpr (F != WFmt::BF16)
#pragma unroll
                    for (int i = 0; i < 8; ++i) w[r][i] *= gs[r];
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                float xf[8];
                load_x8(X + (size_t)min(m, mv - 1) * ldx + k + 8 * h, xf);
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[r][m] = fmaf(w[r][i], xf[i], acc[r][m]);
            }
        }
    };
    // The first PRE chunks load before the dependency wait (the weight stream overlaps the previous
    // kernel), the rest one chunk ahead.
    Chunk<F> pre[PRE][R];
#pragma unroll
    for (int c = 0; c < PRE; ++c)
        if (lane * 16 + c * 512 < K)
#pragma unroll
            for (int r = 0; r < R; ++r) pre[c][r] = load_chunk<F>(rows[r], lane * 16 + c * 512);
    pdl_wait();
    int k = lane * 16 + PRE * 512;
    Chunk<F> cur[R];
    if (k < K)
#pragma unroll
        for (int r = 0; r < R; ++r) cur[r] = load_chunk<F>(rows[r], k);
#pragma unroll
    for (int c = 0; c < PRE; ++c)
        if (lane * 16 + c * 512 < K) consume(pre[c], lane * 16 + c * 512);
    for (; k < K; k += 512) {
        // The next chunk's weights load while this one is consumed.
        Chunk<F> nxt[R];
        const bool more = k + 512 < K;
        if (more)
#pragma unroll
            for (int r = 0; r < R; ++r) nxt[r] = load_chunk<F>(rows[r], k + 512);
        consume(cur, k);
        if (more)
#pragma unroll
            for (int r = 0; r < R; ++r) cur[r] = nxt[r];
    }
    // The shared GEMV's store: y = sum * alpha (+ y).
    auto store = [&](int m, int n, float v) {
        float* y = &Y[(size_t)m * ldy + n];
        *y = accumulate ? *y + v * alpha : v * alpha;
    };
    // Sums 32 at a time (slot j of a batch: r = j / MT, m = j % MT within the batch), each bitwise as its
    // own warp_sum; fewer than 32 sums (a power of two) at once, slot j in lanes j * 32 / NS ...
    constexpr int NS = R * MT;
    static_assert(NS % 32 == 0 || (NS & (NS - 1)) == 0, "R * MT must be a power of two or a multiple of 32");
    if constexpr (NS < 32) {
        float v[NS];
#pragma unroll
        for (int j = 0; j < NS; ++j) v[j] = acc[j / MT][j % MT];
        const float sum = warp_sum_slots(v);
        const int j = lane / (32 / NS), r = j / MT, m = j % MT;
        if (lane % (32 / NS) == 0 && n0 + r < N && m < mv) store(m, n0 + r, sum);
    } else {
#pragma unroll
        for (int b = 0; b < NS / 32; ++b) {
            float v[32];
#pragma unroll
            for (int j = 0; j < 32; ++j) v[j] = acc[(b * 32 + j) / MT][(b * 32 + j) % MT];
            const float sum = warp_sum_slots(v);
            const int j = b * 32 + lane, r = j / MT, m = j % MT;
            if (n0 + r < N && m < mv) store(m, n0 + r, sum);
        }
    }
}

template <WFmt F, int MT, int R>
cudaError_t launch(int M, const void* X, int ldx, const void* W, const float* scale, float* Y, int ldy, int N, int K, float alpha, int accumulate,
                   cudaStream_t st) {
    const int groups = (M + MT - 1) / MT;
    const int tiles = (N + NW * R - 1) / (NW * R);
    SQ_CHECK_LAUNCH(sq_launch(wide_kernel<F, MT, R>, (unsigned)(tiles * groups), NW * 32, 0, st, static_cast<const __nv_bfloat16*>(X), ldx,
                              static_cast<const uint8_t*>(W), scale, Y, ldy, N, K, M, alpha, accumulate));
    return cudaSuccess;
}

template <WFmt F>
cudaError_t launch_f(int mt, int r, int M, const void* X, int ldx, const void* W, const float* scale, float* Y, int ldy, int N, int K, float alpha,
                     int accumulate, cudaStream_t st) {
#define SQ_W(mt_, r_) \
    if (mt == mt_ && r == r_) return launch<F, mt_, r_>(M, X, ldx, W, scale, Y, ldy, N, K, alpha, accumulate, st);
    SQ_W(16, 4) SQ_W(16, 2) SQ_W(8, 4) SQ_W(8, 2) SQ_W(8, 1) SQ_W(4, 4) SQ_W(4, 2) SQ_W(4, 1) SQ_W(2, 4) SQ_W(2, 2)
#undef SQ_W
    return cudaErrorInvalidValue;
}

}  // namespace

// fmt 0 BF16 [N, K], 3 INT8 with an FP32 scale per 64 weights of a row, 5 FP8 E4M3 with an FP32 scale per
// 128 x 128 block; mt x r in {16 x 4, 16 x 2, 8 x 4, 8 x 2, 8 x 1, 4 x 4, 4 x 2, 4 x 1, 2 x 4, 2 x 2}. K a
// multiple of 16 (FP8B: of 128, INT8G: of 64), ldx a multiple of 8.
extern "C" cudaError_t quench6k_nem_gemv_wide(int fmt, int M, int mt, int r, const void* X, int ldx, const void* W, const float* scale, float* Y,
                                            int ldy, int N, int K, float alpha, int accumulate, cudaStream_t st) {
    if (M == 0) return cudaSuccess;
    if (K % 16 != 0 || ldx % 8 != 0) return cudaErrorInvalidValue;
    cudaError_t e = cudaErrorInvalidValue;
    if (fmt == 0) e = launch_f<WFmt::BF16>(mt, r, M, X, ldx, W, scale, Y, ldy, N, K, alpha, accumulate, st);
    else if (fmt == 3 && K % GROUP == 0) e = launch_f<WFmt::INT8G>(mt, r, M, X, ldx, W, scale, Y, ldy, N, K, alpha, accumulate, st);
    else if (fmt == 5 && K % 128 == 0) e = launch_f<WFmt::FP8B>(mt, r, M, X, ldx, W, scale, Y, ldy, N, K, alpha, accumulate, st);
    if (e != cudaSuccess) return e;
    return cudaGetLastError();
}
