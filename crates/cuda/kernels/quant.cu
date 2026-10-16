// NVFP4 quantization of BF16 weights at load (used for the MTP layer's experts, whose checkpoint copy
// is BF16). Matches the checkpoint's NVFP4 layout: codes U8 [N, K/2] (low nibble first), block scales
// FP8 E4M3 [N, K/16], and one FP32 global scale per matrix:
//   scale2 = amax / (6 * 448);  block scale s_b = fp8(amax_b / 6 / scale2);  code = e2m1(w / (s_b * scale2)).

#include <cuda_fp8.h>

#include "common.cuh"

namespace {

// Round-to-nearest-even onto the E2M1 grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}, saturating; returns the 4-bit code.
SQ_DEVICE uint8_t e2m1_code(float x) {
    const float a = fminf(fabsf(x), 6.0f);
    uint8_t c;
    if (a < 0.25f) c = 0;
    else if (a <= 0.75f) c = a == 0.25f ? 0 : (a < 0.75f ? 1 : 2);
    else if (a < 1.25f) c = 2;
    else if (a <= 1.75f) c = a == 1.25f ? 2 : (a < 1.75f ? 3 : 4);
    else if (a < 2.5f) c = 4;
    else if (a <= 3.5f) c = a == 2.5f ? 4 : (a < 3.5f ? 5 : 6);
    else if (a < 5.0f) c = 6;
    else c = a == 5.0f ? 6 : 7;
    return c | (x < 0 && c != 0 ? 8 : 0);
}

// Per-matrix absolute maximum: matrices [count, N * K] BF16; amax[count] must start at 0.
__global__ void amax_kernel(const __nv_bfloat16* __restrict__ w, size_t per, unsigned* amax) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const size_t base = blockIdx.y * per;
    float m = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < per; i += (size_t)gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(bf2f(w[base + i])));
    m = block_max(m, scratch);
    if (threadIdx.x == 0) atomicMax(&amax[blockIdx.y], __float_as_uint(m));
}

// One thread per 16-element block of one matrix row: grid (blocks, count).
__global__ void quantize_kernel(const __nv_bfloat16* __restrict__ w, size_t per, int K, const unsigned* amax,
                                uint8_t* codes, uint8_t* scales, float* scale2) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x, blocks = per / 16;
    const int m = blockIdx.y;
    const float s2 = fmaxf(__uint_as_float(amax[m]), 1e-30f) / (6.0f * 448.0f);
    if (b == 0) scale2[m] = s2;
    if (b >= blocks) return;
    const __nv_bfloat16* src = w + m * per + b * 16;
    float v[16], bmax = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        v[i] = bf2f(src[i]);
        bmax = fmaxf(bmax, fabsf(v[i]));
    }
    const __nv_fp8_storage_t sc = __nv_cvt_float_to_fp8(bmax / 6.0f / s2, __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 scf;
    scf.__x = sc;
    const float denom = float(scf) * s2;
    scales[m * blocks + b] = sc;
    uint8_t* dst = codes + m * (per / 2) + b * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint8_t lo = denom > 0 ? e2m1_code(v[2 * i] / denom) : 0;
        const uint8_t hi = denom > 0 ? e2m1_code(v[2 * i + 1] / denom) : 0;
        dst[i] = lo | (hi << 4);
    }
    (void)K;
}

}  // namespace

// `count` consecutive BF16 matrices of n x k (k a multiple of 16) -> codes [count, n, k/2],
// scales [count, n, k/16], scale2 [count]. `amax` is scratch for `count` u32.
extern "C" cudaError_t squash_quantize_nvfp4(const void* w, int count, int n, int k, void* codes, void* scales,
                                             float* scale2, unsigned* amax, cudaStream_t st) {
    if (count == 0) return cudaSuccess;
    if (k % 16 != 0) return cudaErrorInvalidValue;
    const size_t per = (size_t)n * k;
    cudaError_t e = cudaMemsetAsync(amax, 0, count * sizeof(unsigned), st);
    if (e != cudaSuccess) return e;
    const auto* wb = static_cast<const __nv_bfloat16*>(w);
    SQ_CHECK_LAUNCH(sq_launch(amax_kernel, dim3(64, count), 256, 0, st, wb, per, amax));
    SQ_CHECK_LAUNCH(sq_launch(quantize_kernel, dim3((unsigned)((per / 16 + 255) / 256), count), 256, 0, st, wb, per, k,
                              static_cast<const unsigned*>(amax), static_cast<uint8_t*>(codes),
                              static_cast<uint8_t*>(scales), scale2));
    SQ_RETURN_LAUNCH();
}

namespace {

SQ_DEVICE float e2m1_value(uint8_t c) {
    constexpr float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (c & 8 ? -1.0f : 1.0f) * grid[c & 7];
}

SQ_DEVICE float ue4m3_value(int code) {
    __nv_fp8_e4m3 v;
    v.__x = (__nv_fp8_storage_t)code;
    return float(v);
}

// Squared error of one 16-value block quantized with block scale `bs` (values already divided by the
// row scale).
SQ_DEVICE float block_error(const float* v, float bs) {
    if (bs <= 0.0f) return 3.4e38f;
    float e = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float d = v[i] - e2m1_value(e2m1_code(v[i] / bs)) * bs;
        e = fmaf(d, d, e);
    }
    return e;
}

// NVFP4 activation quantization with a per-row scale (amax / (6 * 448)) and, per block of 16, the
// E4M3 block scale among the nearest code to amax_b / 6 and its +-1, +-2 neighbours with the least
// squared error. `fake`: writes the dequantized values back into x (BF16 [rows, cols], in place).
// One warp per row.
__global__ void fp4_act_kernel(__nv_bfloat16* x, int rows, int cols) {
    pdl_launch_dependents();
    pdl_wait();
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    __nv_bfloat16* xr = x + (size_t)row * cols;
    float amax = 0.0f;
    for (int i = lane; i < cols; i += 32) amax = fmaxf(amax, fabsf(bf2f(xr[i])));
#pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float rs = amax / (6.0f * 448.0f);
    if (rs == 0.0f) return;
    for (int b = lane; b < cols / 16; b += 32) {
        float v[16], am = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            v[i] = bf2f(xr[16 * b + i]) / rs;
            am = fmaxf(am, fabsf(v[i]));
        }
        const int first = (int)__nv_fp8_e4m3(am / 6.0f).__x;
        int best = first;
        float best_e = block_error(v, ue4m3_value(first));
        constexpr int offs[4] = {-1, 1, -2, 2};
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int c = first + offs[t];
            if (c < 0 || c > 0x7e) continue;
            const float e = block_error(v, ue4m3_value(c));
            if (e < best_e) best_e = e, best = c;
        }
        const float bs = ue4m3_value(best);
#pragma unroll
        for (int i = 0; i < 16; ++i) xr[16 * b + i] = f2bf(bs > 0.0f ? e2m1_value(e2m1_code(v[i] / bs)) * bs * rs : 0.0f);
    }
}

}  // namespace

extern "C" cudaError_t squash_fake_fp4_act(void* x, int rows, int cols, cudaStream_t st) {
    if (cols % 16 != 0) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(fp4_act_kernel, (rows + 7) / 8, 256, 0, st, (__nv_bfloat16*)x, rows, cols));
    SQ_RETURN_LAUNCH();
}
