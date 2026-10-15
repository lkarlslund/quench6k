// Shared device helpers.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define SQ_DEVICE __device__ __forceinline__

SQ_DEVICE float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
SQ_DEVICE __nv_bfloat16 f2bf(float x) { return __float2bfloat16_rn(x); }

SQ_DEVICE float silu(float x) { return x / (1.0f + __expf(-x)); }
SQ_DEVICE float sigmoidf(float x) { return 1.0f / (1.0f + __expf(-x)); }

SQ_DEVICE float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

SQ_DEVICE float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// Sum over the whole block; `scratch` holds at least 32 floats. All threads get the result.
SQ_DEVICE float block_sum(float v, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    v = lane < warps ? scratch[lane] : 0.0f;
    return warp_sum(v);
}

SQ_DEVICE float block_max(float v, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    v = lane < warps ? scratch[lane] : -INFINITY;
    return warp_max(v);
}

#define SQ_RETURN_LAUNCH() return cudaGetLastError()
