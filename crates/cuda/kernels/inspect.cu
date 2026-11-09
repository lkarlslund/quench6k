// Inspection and model edits. Per-neuron readouts of routed experts: For each (expert, readout direction) pair p,
// out[p, j] = scale2[e] * sum_n W_e[n, j] r[p, n] over the expert's NVFP4 down projection [2560, 640]
// (codes U8 [N, K/2], low nibble first; scales F8E4M3 [N, K/16]), each weight decoded as the
// decoding kernels decode it (the code times its block scale, rounded to half).

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "common.cuh"

namespace {

constexpr int D = 2560, K = 640;

__global__ void __launch_bounds__(K) nvfp4_column_dots_kernel(const uint64_t* __restrict__ dw,
                                                             const uint64_t* __restrict__ ds,
                                                             const float* __restrict__ d2,
                                                             const int* __restrict__ experts,
                                                             const float* __restrict__ r, float* out) {
    pdl_launch_dependents();
    pdl_wait();
    const int p = blockIdx.x, j = threadIdx.x, e = experts[p];
    const uint8_t* codes = reinterpret_cast<const uint8_t*>(dw[e]);
    const uint8_t* scales = reinterpret_cast<const uint8_t*>(ds[e]);
    const float* rp = r + (size_t)p * D;
    float acc = 0;
    for (int n = 0; n < D; ++n) {
        const __half2_raw two = __nv_cvt_fp4x2_to_halfraw2(codes[(size_t)n * (K / 2) + j / 2], __NV_E2M1);
        const __half2 h2 = *reinterpret_cast<const __half2*>(&two);
        __nv_fp8_e4m3 sc;
        sc.__x = scales[(size_t)n * (K / 16) + j / 16];
        const __half w = __hmul((j & 1) ? __high2half(h2) : __low2half(h2), __float2half(float(sc)));
        acc = fmaf(__half2float(w), rp[n], acc);
    }
    out[(size_t)p * K + j] = acc * d2[e];
}

// Neuron edits: act[row, unit] = act * scale + add for each edit k whose expert is the pair's routed
// expert (row = pair_row[pair]); without top_i, every row 0..pairs (the shared expert's activations).
__global__ void edit_act_kernel(__nv_bfloat16* act, const int* __restrict__ top_i, const int* __restrict__ pair_row,
                                int pairs, const int* __restrict__ experts, const int* __restrict__ units,
                                const float* __restrict__ scale, const float* __restrict__ add, int n) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs * n) return;
    const int p = i / n, k = i % n;
    int row = p;
    if (top_i) {
        if (top_i[p] != experts[k]) return;
        row = pair_row[p];
    }
    __nv_bfloat16* a = act + (size_t)row * K + units[k];
    *a = __float2bfloat16(__bfloat162float(*a) * scale[k] + add[k]);
}

// Router edits: logits[r, expert] += bias for every row (a bias of -inf bans the expert).
__global__ void router_bias_kernel(float* logits, int ld, int T, const int* __restrict__ experts,
                                   const float* __restrict__ bias, int n) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * n) return;
    logits[(size_t)(i / n) * ld + experts[i % n]] += bias[i % n];
}

}  // namespace

extern "C" cudaError_t quench6k_nvfp4_column_dots(const uint64_t* dw, const uint64_t* ds, const float* d2,
                                                const int* experts, int pairs, const float* r, float* out,
                                                cudaStream_t st) {
    if (pairs == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(nvfp4_column_dots_kernel, pairs, K, 0, st, dw, ds, d2, experts, r, out));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_edit_act(void* act, const int* top_i, const int* pair_row, int pairs, const int* experts,
                                       const int* units, const float* scale, const float* add, int n,
                                       cudaStream_t st) {
    if (pairs == 0 || n == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(edit_act_kernel, (pairs * n + 255) / 256, 256, 0, st, static_cast<__nv_bfloat16*>(act), top_i,
                              pair_row, pairs, experts, units, scale, add, n));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_router_bias(float* logits, int ld, int T, const int* experts, const float* bias, int n,
                                          cudaStream_t st) {
    if (T == 0 || n == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(router_bias_kernel, (T * n + 255) / 256, 256, 0, st, logits, ld, T, experts, bias, n));
    SQ_RETURN_LAUNCH();
}
