// NVFP4 quantization of BF16 weights at load (used for the MTP layer's experts, whose checkpoint copy
// is BF16). Matches the checkpoint's NVFP4 layout: codes U8 [N, K/2] (low nibble first), block scales
// FP8 E4M3 [N, K/16], and one FP32 global scale per matrix:
//   scale2 = amax / (6 * 448);  block scale s_b = fp8(amax_b / 6 / scale2);  code = e2m1(w / (s_b * scale2)).

#include <cuda_fp8.h>

#include "fp4.cuh"

namespace {

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
extern "C" cudaError_t quench6k_quantize_nvfp4(const void* w, int count, int n, int k, void* codes, void* scales,
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
        float v[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = bf2f(xr[16 * b + i]) / rs;
        const float bs = ue4m3_value(fp4_block_scale(v));
#pragma unroll
        for (int i = 0; i < 16; ++i) xr[16 * b + i] = f2bf(bs > 0.0f ? e2m1_value(e2m1_code(v[i] / bs)) * bs * rs : 0.0f);
    }
}

}  // namespace

extern "C" cudaError_t quench6k_fake_fp4_act(void* x, int rows, int cols, cudaStream_t st) {
    if (cols % 16 != 0) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(fp4_act_kernel, (rows + 7) / 8, 256, 0, st, (__nv_bfloat16*)x, rows, cols));
    SQ_RETURN_LAUNCH();
}

namespace {

// MXFP8 quantize-dequantize in place (experiments): per 32 consecutive values of a row, a power-of-two
// scale 2^ceil(log2(amax / 448)) and E4M3 values (round to nearest, saturating). One thread per block.
__global__ void fake_mxfp8_kernel(__nv_bfloat16* x, int rows, int cols, int ld) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const int per = cols / 32;
    if (i >= (size_t)rows * per) return;
    __nv_bfloat16* p = x + (i / per) * (size_t)ld + (i % per) * 32;
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(bf2f(p[j])));
    if (amax == 0.0f) return;
    const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
#pragma unroll
    for (int j = 0; j < 32; ++j) p[j] = f2bf(float(__nv_fp8_e4m3(bf2f(p[j]) / scale)) * scale);
}

}  // namespace

extern "C" cudaError_t quench6k_fake_mxfp8(void* x, int rows, int cols, int ld, cudaStream_t st) {
    if (cols % 32 != 0) return cudaErrorInvalidValue;
    const size_t n = (size_t)rows * (cols / 32);
    if (n == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(fake_mxfp8_kernel, (unsigned)((n + 255) / 256), 256, 0, st, (__nv_bfloat16*)x, rows, cols, ld));
    SQ_RETURN_LAUNCH();
}

namespace {

// Symmetric integer quantize-dequantize in place (experiments): per `group` consecutive values of a
// row, scale amax / qmax (qmax = 2^(bits-1) - 1), round to nearest. One thread per group.
__global__ void fake_int_kernel(__nv_bfloat16* x, size_t groups, int group, float qmax) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= groups) return;
    __nv_bfloat16* p = x + i * group;
    float amax = 0.0f;
    for (int j = 0; j < group; ++j) amax = fmaxf(amax, fabsf(bf2f(p[j])));
    if (amax == 0.0f) return;
    const float s = amax / qmax;
    for (int j = 0; j < group; ++j) p[j] = f2bf(rintf(bf2f(p[j]) / s) * s);
}

}  // namespace

extern "C" cudaError_t quench6k_fake_int(void* x, int rows, int cols, int group, int bits, cudaStream_t st) {
    if (cols % group != 0 || bits < 2 || bits > 8) return cudaErrorInvalidValue;
    const size_t n = (size_t)rows * (cols / group);
    if (n == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(fake_int_kernel, (unsigned)((n + 255) / 256), 256, 0, st, (__nv_bfloat16*)x, n, group, (float)((1 << (bits - 1)) - 1)));
    SQ_RETURN_LAUNCH();
}
