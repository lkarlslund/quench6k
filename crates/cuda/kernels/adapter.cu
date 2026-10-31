// Rank-1 weight adapter on a block's output projection: y_t += s * (a . x_t) b, with x [T, K] BF16 (the
// projection's input, row stride ldx), y [T, 2560] FP32 (its output), a [K] and b [2560] FP32. The scale
// s is device data (the sequence's variant); a zero scale writes nothing.

#include "common.cuh"

namespace {

constexpr int D = 2560, THREADS = 256, PER = D / THREADS;

// grid T.
__global__ void __launch_bounds__(THREADS) adapter_rank1_kernel(const __nv_bfloat16* __restrict__ x, int ldx, int k,
                                                                float* y, const float* __restrict__ a,
                                                                const float* __restrict__ b, const float* __restrict__ scale_ptr) {
    pdl_launch_dependents();
    pdl_wait();
    const float scale = *scale_ptr;
    if (scale == 0.0f) return;
    __shared__ float scratch[32];
    const __nv_bfloat16* row = x + (size_t)blockIdx.x * ldx;
    float dot = 0.0f;
    for (int j = threadIdx.x; j < k; j += THREADS) dot = fmaf(__bfloat162float(row[j]), a[j], dot);
    dot = block_sum(dot, scratch) * scale;
    float* out = y + (size_t)blockIdx.x * D;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int c = threadIdx.x + j * THREADS;
        out[c] = fmaf(dot, b[c], out[c]);
    }
}

}  // namespace

extern "C" cudaError_t quench6k_adapter_rank1(const __nv_bfloat16* x, int ldx, int k, int T, float* y, const float* a,
                                            const float* b, const float* scale, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(adapter_rank1_kernel, dim3(T), THREADS, 0, st, x, ldx, k, y, a, b, scale));
    SQ_RETURN_LAUNCH();
}
