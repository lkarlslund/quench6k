// Embedding, HyperConnection mix/injection/combine (SPEC §4) and small elementwise helpers.
//
// Layouts: the residual h is FP32 [T, 4*2560] (stream-major within a token); n is the BF16
// grouped-normalized residual; m (block input) is BF16 [T, 2560].

#include "common.cuh"

namespace {

constexpr int D = 2560, S = 4, H = S * D, RANK = 320;
constexpr float EPS = 1e-6f;

__global__ void embed_kernel(const int* __restrict__ tokens, const __nv_bfloat16* __restrict__ table, float* h) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x;
    const __nv_bfloat16* row = table + (size_t)tokens[t] * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float v = bf2f(row[d]);
#pragma unroll
        for (int s = 0; s < S; ++s) h[(size_t)t * H + s * D + d] = v;
    }
}

// n[t, s, :] = rms(h[t, s, :]) * (1 + w[s, :]); one block per (token, stream).
__global__ void grouped_rmsnorm_kernel(const float* __restrict__ h, const __nv_bfloat16* __restrict__ w,
                                       __nv_bfloat16* __restrict__ n) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const int t = blockIdx.x / S, s = blockIdx.x % S;
    const float* x = h + (size_t)t * H + s * D;
    float ss = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) ss = __fmaf_rn(x[d], x[d], ss);
    const float r = rsqrtf(block_sum(ss, scratch) / D + EPS);
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        n[(size_t)t * H + s * D + d] = f2bf(__fmul_rn(__fmul_rn(x[d], r), w ? 1.0f + bf2f(w[s * D + d]) : 1.0f));
}

// hc_combine then grouped_rmsnorm (no weight) in one pass: h[t, s, :] += y[t, :] * inj[t, s], then
// n = rmsnorm(h[t, s, :]); the same roundings as the two kernels. One block of 256 per (t, s).
__global__ void __launch_bounds__(256) hc_combine_norm_kernel(float* h, const float* __restrict__ y,
                                                              const float* __restrict__ inj, __nv_bfloat16* n) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const int t = blockIdx.x / S, s = blockIdx.x % S;
    float* x = h + (size_t)t * H + s * D;
    const float a = inj[t * S + s];
    constexpr int PER = D / 256;
    float v[PER];
    float ss = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = threadIdx.x + i * 256;
        v[i] = __fmaf_rn(y[(size_t)t * D + d], a, x[d]);
        x[d] = v[i];
        ss = __fmaf_rn(v[i], v[i], ss);
    }
    const float r = rsqrtf(block_sum(ss, scratch) / D + EPS);
#pragma unroll
    for (int i = 0; i < PER; ++i) n[(size_t)t * H + s * D + threadIdx.x + i * 256] = f2bf(__fmul_rn(__fmul_rn(v[i], r), 1.0f));
}

// inj[t, j] = 2 * sigmoid(dot(n[t], W[j]) / 4) for j < 4; one block per token.
__global__ void hc_inject_kernel(const __nv_bfloat16* __restrict__ n, const __nv_bfloat16* __restrict__ w, float* inj) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const int t = blockIdx.x;
    float acc[S] = {};
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float x = bf2f(n[(size_t)t * H + i]);
#pragma unroll
        for (int j = 0; j < S; ++j) acc[j] += x * bf2f(w[(size_t)j * H + i]);
    }
#pragma unroll
    for (int j = 0; j < S; ++j) {
        const float v = block_sum(acc[j], scratch);
        if (threadIdx.x == 0) inj[t * S + j] = 2.0f * sigmoidf(v / S);
    }
}

// From the fused projection x [T, 320 + 4] (row stride ld): z = bf16(silu(x[:, :320] / 4)) and, when
// inj is set, inj = 2 * sigmoid(x[:, 320:324] / 4).
__global__ void hc_act_kernel(const float* __restrict__ x, int ld, __nv_bfloat16* z, float* inj, int T) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * (RANK + S)) return;
    const int t = i / (RANK + S), c = i % (RANK + S);
    if (c >= RANK && !inj) return;
    const float v = x[(size_t)t * ld + c] / S;
    if (c < RANK) z[t * RANK + c] = f2bf(silu(v));
    else if (inj) inj[t * S + c - RANK] = 2.0f * sigmoidf(v);
}

// m[t, d] = mean_s sigmoid(u[t, s, d]) * n[t, s, d] * (1 + g[s, d]) (g optional)
__global__ void hc_mix_kernel(const float* __restrict__ u, const __nv_bfloat16* __restrict__ n,
                              const __nv_bfloat16* __restrict__ g, __nv_bfloat16* m, int T) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int t = i / D, d = i % D;
    float acc = 0;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const size_t j = (size_t)t * H + s * D + d;
        // Explicit roundings (no FMA contraction), as the fused up+mix GEMM epilogue.
        acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(sigmoidf(u[j]), bf2f(n[j])), g ? 1.0f + bf2f(g[s * D + d]) : 1.0f));
    }
    m[i] = f2bf(acc / S);
}

// h[t, s, d] += y[t, d] * inj[t, s]
__global__ void hc_combine_kernel(float* h, const float* __restrict__ y, const float* __restrict__ inj, int T) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const int t = i / H, s = (i % H) / D, d = i % D;
    h[i] = __fmaf_rn(y[(size_t)t * D + d], inj[t * S + s], h[i]);
}

__global__ void f32_to_bf16_kernel(const float* __restrict__ x, __nv_bfloat16* y, size_t count) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) y[i] = f2bf(x[i]);
}

__global__ void add_f32_kernel(float* y, const float* __restrict__ x, size_t count) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) y[i] += x[i];
}

// out[r, :] = bf16(rms(x[r, :D]) * (1 + w)); x rows at stride ldx. One block per row.
__global__ void rmsnorm_rows_kernel(const float* __restrict__ x, int ldx, const __nv_bfloat16* __restrict__ w,
                                    __nv_bfloat16* out) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const float* row = x + (size_t)blockIdx.x * ldx;
    float ss = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) ss += row[d] * row[d];
    const float r = rsqrtf(block_sum(ss, scratch) / D + EPS);
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        out[(size_t)blockIdx.x * D + d] = f2bf(row[d] * r * (1.0f + bf2f(w[d])));
}

int blocks(size_t n, int threads) { return (int)((n + threads - 1) / threads); }

}  // namespace

extern "C" cudaError_t squash_embed(const int* tokens, int T, const void* table, float* h, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(embed_kernel, T, 256, 0, st, tokens, static_cast<const __nv_bfloat16*>(table), h));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_grouped_rmsnorm(const float* h, const void* w, void* n, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(grouped_rmsnorm_kernel, T * S, 256, 0, st, h, static_cast<const __nv_bfloat16*>(w), static_cast<__nv_bfloat16*>(n)));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_hc_inject(const void* n, const void* w, float* inj, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(hc_inject_kernel, T, 256, 0, st, static_cast<const __nv_bfloat16*>(n), static_cast<const __nv_bfloat16*>(w), inj));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_hc_act(const float* x, int ld, void* z, float* inj, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(hc_act_kernel, blocks((size_t)T * (RANK + S), 256), 256, 0, st, x, ld, static_cast<__nv_bfloat16*>(z), inj, T));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_hc_mix(const float* u, const void* n, const void* g, void* m, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(hc_mix_kernel, blocks((size_t)T * D, 256), 256, 0, st, u, static_cast<const __nv_bfloat16*>(n),
                                                               static_cast<const __nv_bfloat16*>(g),
                                                               static_cast<__nv_bfloat16*>(m), T));
    SQ_RETURN_LAUNCH();
}

namespace {

// The MoE combine (as combine_kernel with BF16 expert outputs) and hc_combine_norm in one pass: one
// block of 256 per token computes y[t, :] = sum_k w[t, k] yd[row(t, k), :] + sigmoid(sg[t]) sh[t, :] in
// registers, then for each stream h[t, s, :] += y * inj[t, s] and n = rmsnorm(h[t, s, :]). The same
// arithmetic as the three kernels.
constexpr int MOE_TOPK = 10;
__global__ void __launch_bounds__(256) hc_moe_combine_norm_kernel(float* h, const __nv_bfloat16* __restrict__ yd,
                                                                  const int* __restrict__ pair_row, const float* __restrict__ top_w,
                                                                  const float* __restrict__ sg, int ld_sg, const float* __restrict__ sh,
                                                                  const float* __restrict__ inj, __nv_bfloat16* n) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const int t = blockIdx.x;
    constexpr int PER = D / 256;
    const float g = sigmoidf(sg[(size_t)t * ld_sg]);
    float y[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = threadIdx.x + i * 256;
        float acc = 0;
#pragma unroll
        for (int k = 0; k < MOE_TOPK; ++k) acc += top_w[t * MOE_TOPK + k] * (float)yd[(size_t)pair_row[t * MOE_TOPK + k] * D + d];
        y[i] = acc + g * sh[(size_t)t * D + d];
    }
    for (int s = 0; s < S; ++s) {
        float* x = h + (size_t)t * H + s * D;
        const float a = inj[t * S + s];
        float v[PER];
        float ss = 0;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int d = threadIdx.x + i * 256;
            v[i] = __fmaf_rn(y[i], a, x[d]);
            x[d] = v[i];
            ss = __fmaf_rn(v[i], v[i], ss);
        }
        const float r = rsqrtf(block_sum(ss, scratch) / D + EPS);
#pragma unroll
        for (int i = 0; i < PER; ++i) n[(size_t)t * H + s * D + threadIdx.x + i * 256] = f2bf(__fmul_rn(__fmul_rn(v[i], r), 1.0f));
    }
}

}  // namespace

extern "C" cudaError_t squash_hc_moe_combine_norm(float* h, const void* yd, const int* pair_row, const float* top_w, const float* sg,
                                                  int ld_sg, const float* sh, const float* inj, void* n, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(hc_moe_combine_norm_kernel, T, 256, 0, st, h, static_cast<const __nv_bfloat16*>(yd), pair_row, top_w, sg, ld_sg,
                              sh, inj, static_cast<__nv_bfloat16*>(n)));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_hc_combine_norm(float* h, const float* y, const float* inj, void* n, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    static_assert(D % 256 == 0, "D must be a multiple of 256");
    SQ_CHECK_LAUNCH(sq_launch(hc_combine_norm_kernel, T * S, 256, 0, st, h, y, inj, static_cast<__nv_bfloat16*>(n)));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_hc_combine(float* h, const float* y, const float* inj, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(hc_combine_kernel, blocks((size_t)T * H, 256), 256, 0, st, h, y, inj, T));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_f32_to_bf16(const float* x, void* y, size_t count, cudaStream_t st) {
    if (count == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(f32_to_bf16_kernel, blocks(count, 256), 256, 0, st, x, static_cast<__nv_bfloat16*>(y), count));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_add_f32(float* y, const float* x, size_t count, cudaStream_t st) {
    if (count == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(add_f32_kernel, blocks(count, 256), 256, 0, st, y, x, count));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_rmsnorm_rows(const float* x, int ldx, const void* w, void* out, int rows, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(rmsnorm_rows_kernel, rows, 256, 0, st, x, ldx, static_cast<const __nv_bfloat16*>(w),
                              static_cast<__nv_bfloat16*>(out)));
    SQ_RETURN_LAUNCH();
}
