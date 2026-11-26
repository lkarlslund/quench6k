// Nemotron-H prefill: 8-bit dense weights expanded to BF16 once per large step, so the projection runs the
// BF16 GEMM. The values are those the GEMM's 8-bit paths stage (gemm.cu: bf16(fp8 or int8 code x its
// block or group scale), round to nearest), so the outputs keep their bits.

#include <cuda_fp8.h>

#include "nemotron.cuh"

using namespace nem;

namespace {

// One thread per 8 weights of a row (k % 8 == 0). fmt 5: E4M3 with a scale per 128 x 128 block
// ([ceil(n / 128), k / 128]); fmt 3: INT8 with a scale per 64 weights of a row ([n, k / 64]).
template <int FMT>
__global__ void expand_kernel(const uint8_t* w, const float* scale, int n, int k, bf16* out) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * k / 8) return;
    const int row = (int)(i / (k / 8)), c = (int)(i % (k / 8)) * 8;
    const uint2 raw = __ldg(reinterpret_cast<const uint2*>(w + (size_t)row * k + c));
    const float s = FMT == 5 ? __ldg(scale + (size_t)(row >> 7) * (k >> 7) + (c >> 7)) : __ldg(scale + (size_t)row * (k / 64) + c / 64);
    uint32_t packed[4];
    if constexpr (FMT == 5) {
        const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(p[j], __NV_E4M3);
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
            const __nv_bfloat162 v = __floats2bfloat162_rn(f.x * s, f.y * s);
            packed[j] = *reinterpret_cast<const uint32_t*>(&v);
        }
    } else {
        const int8_t* p = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __nv_bfloat162 v = __floats2bfloat162_rn((float)p[2 * j] * s, (float)p[2 * j + 1] * s);
            packed[j] = *reinterpret_cast<const uint32_t*>(&v);
        }
    }
    *reinterpret_cast<uint4*>(out + (size_t)row * k + c) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
}

}  // namespace

// out BF16 [n, k] = the weights of an FP8-block (fmt 5) or INT8-group (fmt 3) projection as the GEMM stages them.
extern "C" cudaError_t quench6k_nem_expand_bf16(int fmt, const void* w, const float* scale, int n, int k, void* out, cudaStream_t st) {
    if (k % 128 != 0 || (fmt != 3 && fmt != 5)) return cudaErrorInvalidValue;
    const size_t threads = (size_t)n * k / 8;
    if (threads == 0) return cudaSuccess;
    const unsigned blocks = (unsigned)((threads + 255) / 256);
    if (fmt == 5) SQ_CHECK_LAUNCH(sq_launch(expand_kernel<5>, blocks, 256, 0, st, static_cast<const uint8_t*>(w), scale, n, k, static_cast<bf16*>(out)));
    else SQ_CHECK_LAUNCH(sq_launch(expand_kernel<3>, blocks, 256, 0, st, static_cast<const uint8_t*>(w), scale, n, k, static_cast<bf16*>(out)));
    SQ_RETURN_LAUNCH();
}
