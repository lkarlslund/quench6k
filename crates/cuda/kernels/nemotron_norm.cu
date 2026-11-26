// Nemotron-H block RMSNorm over the FP32 residual stream: FP32 sum of squares in a fixed order, FP32 scale and
// weight multiply, one rounding at the end (BF16 output) or none (FP32 output). One block per row.

#include "nemotron.cuh"

using namespace nem;

namespace {

constexpr int RMS_THREADS = 256, RMS_PER = HIDDEN / RMS_THREADS;

template <bool F32OUT>
__global__ void __launch_bounds__(RMS_THREADS) rms_kernel(const float* x, int ldx, const bf16* w, float eps, void* out, int ldo) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const int r = blockIdx.x, tid = threadIdx.x;
    float v[RMS_PER];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < RMS_PER; ++k) {
        v[k] = __ldcg(x + (size_t)r * ldx + k * RMS_THREADS + tid);
        ss = fmaf(v[k], v[k], ss);
    }
    ss = block_sum(ss, scratch);
    const float rs = 1.0f / sqrtf(ss / (float)HIDDEN + eps);
#pragma unroll
    for (int k = 0; k < RMS_PER; ++k) {
        const int c = k * RMS_THREADS + tid;
        const float o = bf2f(w[c]) * (v[k] * rs);
        if constexpr (F32OUT) static_cast<float*>(out)[(size_t)r * ldo + c] = o;
        else static_cast<bf16*>(out)[(size_t)r * ldo + c] = f2bf(o);
    }
}

}  // namespace

// out[r] = w * (x[r] / sqrt(mean(x[r]²) + eps)) over HIDDEN columns; f32_out: FP32 output, else BF16.
extern "C" cudaError_t quench6k_nem_rms(const float* x, int ldx, const void* w, float eps, int rows, void* out, int ldo, int f32_out,
                                        cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    const bf16* wb = static_cast<const bf16*>(w);
    if (f32_out) SQ_CHECK_LAUNCH(sq_launch(rms_kernel<true>, rows, RMS_THREADS, 0, st, x, ldx, wb, eps, out, ldo));
    else SQ_CHECK_LAUNCH(sq_launch(rms_kernel<false>, rows, RMS_THREADS, 0, st, x, ldx, wb, eps, out, ldo));
    SQ_RETURN_LAUNCH();
}
