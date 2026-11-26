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

// Row copies: table[2 i] = destination, table[2 i + 1] = source of HIDDEN floats; one block per row.
__global__ void __launch_bounds__(RMS_THREADS) copy_rows_kernel(const unsigned long long* table) {
    pdl_launch_dependents();
    pdl_wait();
    const float4* src = reinterpret_cast<const float4*>(__ldcg(table + 2 * blockIdx.x + 1));
    float4* dst = reinterpret_cast<float4*>(__ldcg(table + 2 * blockIdx.x));
    for (int i = threadIdx.x; i < HIDDEN / 4; i += RMS_THREADS) dst[i] = __ldcg(src + i);
}

}  // namespace

// rows copies of HIDDEN floats from table[2 i + 1] to table[2 i] (device addresses, u64 pairs in device memory).
extern "C" cudaError_t quench6k_nem_copy_rows(const void* table, int rows, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(copy_rows_kernel, rows, RMS_THREADS, 0, st, static_cast<const unsigned long long*>(table)));
    SQ_RETURN_LAUNCH();
}

// out[r] = w * (x[r] / sqrt(mean(x[r]²) + eps)) over HIDDEN columns; f32_out: FP32 output, else BF16.
extern "C" cudaError_t quench6k_nem_rms(const float* x, int ldx, const void* w, float eps, int rows, void* out, int ldo, int f32_out,
                                        cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    const bf16* wb = static_cast<const bf16*>(w);
    if (f32_out) SQ_CHECK_LAUNCH(sq_launch(rms_kernel<true>, rows, RMS_THREADS, 0, st, x, ldx, wb, eps, out, ldo));
    else SQ_CHECK_LAUNCH(sq_launch(rms_kernel<false>, rows, RMS_THREADS, 0, st, x, ldx, wb, eps, out, ldo));
    SQ_RETURN_LAUNCH();
}

namespace {

__global__ void bf16_to_f32_kernel(const uint4* x, float4* out, size_t n8) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const uint4 v = __ldcg(x + i);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    float f[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        f[2 * j] = __uint_as_float(w[j] << 16);
        f[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
    }
    out[2 * i] = make_float4(f[0], f[1], f[2], f[3]);
    out[2 * i + 1] = make_float4(f[4], f[5], f[6], f[7]);
}

}  // namespace

// out[i] = float(x[i]) for n BF16 values (n a multiple of 8, both 16-byte aligned): prefill embedding rows
// uploaded as BF16.
extern "C" cudaError_t quench6k_nem_bf16_to_f32(const void* x, size_t n, float* out, cudaStream_t st) {
    if (n % 8 != 0) return cudaErrorInvalidValue;
    const size_t n8 = n / 8;
    if (n8 == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(bf16_to_f32_kernel, (unsigned)((n8 + 255) / 256), 256, 0, st, static_cast<const uint4*>(x), reinterpret_cast<float4*>(out), n8));
    SQ_RETURN_LAUNCH();
}
