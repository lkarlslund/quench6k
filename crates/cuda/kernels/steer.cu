// Direction-removal steering of the residual lanes at a layer's attention-input pre-mix point:
// for every steered lane s of a token row, y_s = x_s - a * sum_r (x_s . d_{r,s}) d_{r,s} with all dot
// products taken from x_s, optionally rescaled to the norm of x_s. Strength, rank and lane mask are
// device data; a zero strength, zero rank or masked lane writes nothing.

#include "common.cuh"

namespace {

constexpr int D = 2560, S = 4, H = S * D, MAX_RANK = 4, THREADS = 256, PER = D / THREADS;

// grid (T, 4 lanes). directions [48][MAX_RANK][4][2560]; ranks, masks [48].
__global__ void __launch_bounds__(THREADS) steer_kernel(float* h, const float* __restrict__ directions,
                                                        const int* __restrict__ ranks, const int* __restrict__ masks,
                                                        int norm_preserve, const float* __restrict__ strength_ptr, int layer) {
    pdl_launch_dependents();
    pdl_wait();
    const float strength = *strength_ptr;
    const int rank = ranks[layer], lane = blockIdx.y;
    if (strength == 0.0f || rank == 0 || !(masks[layer] >> lane & 1)) return;
    __shared__ float scratch[32];
    float* row = h + (size_t)blockIdx.x * H + lane * D;
    float x[PER], delta[PER] = {}, before = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        x[j] = row[threadIdx.x + j * THREADS];
        before = fmaf(x[j], x[j], before);
    }
    for (int r = 0; r < rank; ++r) {
        const float* d = directions + (((size_t)layer * MAX_RANK + r) * S + lane) * D;
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) dot = fmaf(x[j], d[threadIdx.x + j * THREADS], dot);
        dot = block_sum(dot, scratch);
#pragma unroll
        for (int j = 0; j < PER; ++j) delta[j] = fmaf(dot, d[threadIdx.x + j * THREADS], delta[j]);
    }
    float y[PER], after = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        y[j] = x[j] - strength * delta[j];
        after = fmaf(y[j], y[j], after);
    }
    float scale = 1.0f;
    if (norm_preserve) {
        before = block_sum(before, scratch);
        after = block_sum(after, scratch);
        if (after > 0.0f) scale = sqrtf(before / after);  // a zero row stays zero
    }
#pragma unroll
    for (int j = 0; j < PER; ++j) row[threadIdx.x + j * THREADS] = y[j] * scale;
}

}  // namespace

extern "C" int squash_steer_max_rank() { return MAX_RANK; }

extern "C" cudaError_t squash_steer(float* h, int T, const float* directions, const int* ranks, const int* masks,
                                    int norm_preserve, const float* strength, int layer, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(steer_kernel, dim3(T, S), THREADS, 0, st, h, directions, ranks, masks, norm_preserve, strength, layer));
    SQ_RETURN_LAUNCH();
}
