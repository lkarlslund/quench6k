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

// Sums N slots (N a power of two <= 32) over the warp at once: returns, in every lane, the warp sum of
// v[lane >> (5 - log2 N)]. Each slot is added over the same xor 16, 8, 4, 2, 1 pairing as warp_sum
// (and addition is commutative), so every result is bitwise equal to warp_sum of that slot; it takes
// N - 1 + (5 - log2 N) shuffles instead of 5 N. v is clobbered.
template <int N>
SQ_DEVICE float warp_sum_slots(float (&v)[N]) {
    const int lane = threadIdx.x & 31;
    int off = 16;
#pragma unroll
    for (int n = N; n > 1; n >>= 1, off >>= 1) {
        const bool hi = lane & off;
#pragma unroll
        for (int i = 0; i < n / 2; ++i) {
            const float send = hi ? v[i] : v[i + n / 2];
            const float keep = hi ? v[i + n / 2] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffff, send, off);
        }
    }
#pragma unroll
    for (; off > 0; off >>= 1) v[0] += __shfl_xor_sync(0xffffffff, v[0], off);
    return v[0];
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
#define SQ_CHECK_LAUNCH(call)              \
    do {                                   \
        const cudaError_t sq_err_ = (call); \
        if (sq_err_ != cudaSuccess) return sq_err_; \
    } while (0)

// Programmatic dependent launch (PDL). Every kernel lets its successor start launching immediately and
// waits for its predecessor's completion (and memory) before reading anything an earlier kernel wrote.
// Every kernel waits at some point, so completion stays ordered transitively along the stream.
SQ_DEVICE void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
SQ_DEVICE void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// Launches with the PDL attribute so the successor's blocks may start while this kernel finishes.
template <typename... Exp, typename... Act>
cudaError_t sq_launch(void (*kernel)(Exp...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Act&&... args) {
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = st;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, static_cast<Exp>(args)...);
}

// Sampling parameters and generator state of sampled speculative rounds (40 bytes). `draft_sampled`
// > 0: drafts are sampled from the drafter's distribution (and verified against it) instead of greedy,
// at draft_sampled percent of the temperature.
struct SampleParams {
    float temperature, top_p;
    int k;  // candidates (a prefix of the sorted top list)
    int draft_sampled;
    unsigned long long seed, counter;
    float min_p;  // drop candidates below min_p times the most likely one's probability
    int pad;
};

SQ_DEVICE double sample_uniform(SampleParams* sp) {
    unsigned long long z = sp->seed + 0x9e3779b97f4a7c15ull * (++sp->counter);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return (double)(z >> 11) / 9007199254740992.0;
}

// A sampled draft's distribution: `count` (id, probability) pairs, most likely first.
constexpr int DRAFT_Q_MAX = 64;
struct DraftQ {
    int count;
    int ids[DRAFT_Q_MAX];
    float probs[DRAFT_Q_MAX];
};
