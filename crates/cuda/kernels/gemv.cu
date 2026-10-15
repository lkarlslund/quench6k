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

#include "common.cuh"

namespace {

constexpr int WARPS = 8, THREADS = WARPS * 32, ROWS_PER_WARP = 2, ROWS_PER_CTA = WARPS * ROWS_PER_WARP;

enum class WFmt { BF16 = 0, FP8 = 1, INT8 = 2, INT8G = 3 };

// INT8G: INT8 with one FP32 scale per GROUP consecutive weights of a row (scales [N, K / GROUP]).
constexpr int GROUP = 64;

// Loads 8 consecutive weights (16 bytes of BF16 or 8 bytes of FP8) as floats.
template <WFmt F>
SQ_DEVICE void load8(const uint8_t* row, int k, float* w);

template <>
SQ_DEVICE void load8<WFmt::BF16>(const uint8_t* row, int k, float* w) {
    const uint4 v = __ldg(reinterpret_cast<const uint4*>(row + (size_t)k * 2));
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(p[i]);
        w[2 * i] = f.x;
        w[2 * i + 1] = f.y;
    }
}

template <>
SQ_DEVICE void load8<WFmt::FP8>(const uint8_t* row, int k, float* w) {
    const uint2 v = __ldg(reinterpret_cast<const uint2*>(row + k));
    const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3);
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
        w[2 * i] = f.x;
        w[2 * i + 1] = f.y;
    }
}

template <>
SQ_DEVICE void load8<WFmt::INT8>(const uint8_t* row, int k, float* w) {
    const uint2 v = __ldg(reinterpret_cast<const uint2*>(row + k));
    const int8_t* p = reinterpret_cast<const int8_t*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = (float)p[i];
}

template <>
SQ_DEVICE void load8<WFmt::INT8G>(const uint8_t* row, int k, float* w) {
    load8<WFmt::INT8>(row, k, w);
}

template <WFmt F>
constexpr int elem_bytes() {
    return F == WFmt::BF16 ? 2 : 1;
}

// Loads 8 consecutive activations as floats from BF16 or FP32 storage.
SQ_DEVICE void load_x8(const __nv_bfloat16* x, float* out) {
    const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x));
    const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(xp[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}
SQ_DEVICE void load_x8(const float* x, float* out) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(x)), b = __ldg(reinterpret_cast<const float4*>(x + 4));
    out[0] = a.x, out[1] = a.y, out[2] = a.z, out[3] = a.w, out[4] = b.x, out[5] = b.y, out[6] = b.z, out[7] = b.w;
}

// grid (ceil(N / 16), splits). Each split covers k_per_split columns (a multiple of 256).
template <WFmt F, int M, typename XT>
__global__ void __launch_bounds__(THREADS) gemv_kernel(const XT* __restrict__ X, int ldx,
                                                       const uint8_t* __restrict__ W, const float* __restrict__ row_scale,
                                                       float* Y, int ldy, float* partial, int ldp, int N, int K,
                                                       int k_per_split, float alpha, int accumulate) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int k_begin = blockIdx.y * k_per_split, k_end = min(K, k_begin + k_per_split);
    const size_t row_bytes = (size_t)K * elem_bytes<F>();
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int n = blockIdx.x * ROWS_PER_CTA + warp * ROWS_PER_WARP + r;
        if (n >= N) break;
        const uint8_t* row = W + n * row_bytes;
        float acc[M] = {};
        for (int k = k_begin + lane * 8; k < k_end; k += 256) {
            float w[8];
            load8<F>(row, k, w);
            if constexpr (F == WFmt::INT8G) {
                const float gs = __ldg(row_scale + (size_t)n * (K / GROUP) + k / GROUP);
#pragma unroll
                for (int i = 0; i < 8; ++i) w[i] *= gs;
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float xf[8];
                load_x8(X + (size_t)m * ldx + k, xf);
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[m] = fmaf(w[i], xf[i], acc[m]);
            }
        }
#pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = warp_sum(acc[m]);
        if (lane == 0) {
            const float scale = alpha * (F == WFmt::FP8 || F == WFmt::INT8 ? row_scale[n] : 1.0f);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                if (gridDim.y == 1) {
                    float* y = &Y[(size_t)m * ldy + n];
                    *y = accumulate ? *y + acc[m] * scale : acc[m] * scale;
                } else {
                    partial[((size_t)blockIdx.y * M + m) * ldp + n] = acc[m] * scale;
                }
            }
        }
    }
}

// Y[m, n] (+)= sum over splits of partial[split, m, n], in split order.
__global__ void reduce_kernel(const float* __restrict__ partial, int ldp, int splits, int M, int N, float* Y, int ldy,
                              int accumulate) {
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
                     float* partial, int ldp, int N, int K, int splits, float alpha, int accumulate, int reduce,
                     cudaStream_t st) {
    const int k_per_split = ((K + splits - 1) / splits + 255) / 256 * 256;
    splits = (K + k_per_split - 1) / k_per_split;
    dim3 grid((N + ROWS_PER_CTA - 1) / ROWS_PER_CTA, splits);
    if (x_f32)
        gemv_kernel<F, M, float><<<grid, THREADS, 0, st>>>(static_cast<const float*>(X), ldx, static_cast<const uint8_t*>(W),
                                                           scale, Y, ldy, partial, ldp, N, K, k_per_split, alpha, accumulate);
    else
        gemv_kernel<F, M, __nv_bfloat16><<<grid, THREADS, 0, st>>>(static_cast<const __nv_bfloat16*>(X), ldx,
                                                                   static_cast<const uint8_t*>(W), scale, Y, ldy,
                                                                   partial, ldp, N, K, k_per_split, alpha, accumulate);
    if (splits > 1 && reduce)
        reduce_kernel<<<(M * N + 255) / 256, 256, 0, st>>>(partial, ldp, splits, M, N, Y, ldy, accumulate);
    return cudaGetLastError();
}

template <WFmt F>
cudaError_t launch(int M, const void* X, int x_f32, int ldx, const void* W, const float* scale, float* Y, int ldy,
                   float* partial, int ldp, int N, int K, int splits, float alpha, int accumulate, int reduce,
                   cudaStream_t st) {
    if (K % 8 != 0 || ldx % 8 != 0) return cudaErrorInvalidValue;
    switch (M) {
#define SQ_CASE(m) \
    case m:        \
        return launch_m<F, m>(X, x_f32, ldx, W, scale, Y, ldy, partial, ldp, N, K, splits, alpha, accumulate, reduce, st);
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
extern "C" cudaError_t squash_gemv(int fmt, int M, const void* X, int x_f32, int ldx, const void* W, const float* scale,
                                   float* Y, int ldy, float* partial, int ldp, int N, int K, int splits, float alpha,
                                   int accumulate, int reduce, cudaStream_t st) {
    if (M == 0) return cudaSuccess;
#define SQ_ARGS M, X, x_f32, ldx, W, scale, Y, ldy, partial, ldp, N, K, splits, alpha, accumulate, reduce, st
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
    quantize_fp8_rows_kernel<<<N, 256, 0, st>>>(static_cast<const __nv_bfloat16*>(w), K, static_cast<uint8_t*>(q), scale);
    SQ_RETURN_LAUNCH();
}

namespace {

// BF16 [N, K] -> INT8 [N, K] with scale[n] = absmax(row) / 127, round to nearest. One block per row.
__global__ void quantize_int8_rows_kernel(const __nv_bfloat16* __restrict__ w, int K, int8_t* q, float* scale) {
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
    quantize_int8_rows_kernel<<<N, 256, 0, st>>>(static_cast<const __nv_bfloat16*>(w), K, static_cast<int8_t*>(q), scale);
    SQ_RETURN_LAUNCH();
}

namespace {

// BF16 [N, K] -> INT8 [N, K] with scale[n, g] = absmax(row n, group g) / 127. One warp per (row, group of 128).
__global__ void quantize_int8_groups_kernel(const __nv_bfloat16* __restrict__ w, int K, int8_t* q, float* scale) {
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
    quantize_int8_groups_kernel<<<(items + 7) / 8, 256, 0, st>>>(static_cast<const __nv_bfloat16*>(w), K,
                                                                  static_cast<int8_t*>(q), scale);
    SQ_RETURN_LAUNCH();
}

namespace {

// W[n, k] = bf16(W[n, k] * (1 + g[k])): folds a (1 + w)-style norm weight into a projection's columns.
__global__ void fold_norm_kernel(__nv_bfloat16* w, const __nv_bfloat16* __restrict__ g, int K, size_t count) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) w[i] = f2bf(bf2f(w[i]) * (1.0f + bf2f(g[i % K])));
}

}  // namespace

extern "C" cudaError_t squash_fold_norm(void* w, const void* g, int N, int K, cudaStream_t st) {
    const size_t count = (size_t)N * K;
    if (count == 0) return cudaSuccess;
    fold_norm_kernel<<<(unsigned)((count + 255) / 256), 256, 0, st>>>(static_cast<__nv_bfloat16*>(w),
                                                                      static_cast<const __nv_bfloat16*>(g), K, count);
    SQ_RETURN_LAUNCH();
}
