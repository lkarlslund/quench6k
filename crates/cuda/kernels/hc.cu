// Embedding, HyperConnection mix/injection/combine (SPEC §4) and small elementwise helpers.
//
// Layouts: the residual h is FP32 [T, 4*2560] (stream-major within a token); n is the BF16
// grouped-normalized residual; m (block input) is BF16 [T, 2560].

#include "common.cuh"

namespace {

constexpr int D = 2560, S = 4, H = S * D, RANK = 320;
constexpr float EPS = 1e-6f;

__global__ void embed_kernel(const int* __restrict__ tokens, const __nv_bfloat16* __restrict__ table, float* h) {
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
    __shared__ float scratch[32];
    const int t = blockIdx.x / S, s = blockIdx.x % S;
    const float* x = h + (size_t)t * H + s * D;
    float ss = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) ss += x[d] * x[d];
    const float r = rsqrtf(block_sum(ss, scratch) / D + EPS);
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        n[(size_t)t * H + s * D + d] = f2bf(x[d] * r * (w ? 1.0f + bf2f(w[s * D + d]) : 1.0f));
}

// inj[t, j] = 2 * sigmoid(dot(n[t], W[j]) / 4) for j < 4; one block per token.
__global__ void hc_inject_kernel(const __nv_bfloat16* __restrict__ n, const __nv_bfloat16* __restrict__ w, float* inj) {
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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int t = i / D, d = i % D;
    float acc = 0;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const size_t j = (size_t)t * H + s * D + d;
        acc += sigmoidf(u[j]) * bf2f(n[j]) * (g ? 1.0f + bf2f(g[s * D + d]) : 1.0f);
    }
    m[i] = f2bf(acc / S);
}

// Small batches, after the down GEMV over the raw residual with norm-folded weights left `splits`
// partials [splits, T, N] (splits / 4 per stream, in stream order) and per-split sums of squared
// residual values xsq [splits, T]: r[t, s] = rsqrt(mean h[t, s]^2 + eps); x = sum_split r * partial;
// z = bf16(silu(x[:320] / 4)); inj = 2 sigmoid(x[320:324] / 4) when N = 324. grid (T, ceil(N / 64)).
__global__ void hc_finish_kernel(const float* __restrict__ partial, const float* __restrict__ xsq, int splits, int T,
                                 int N, __nv_bfloat16* z, float* inj, float* r) {
    const int t = blockIdx.x, per_stream = splits / S;
    float rs[S];
#pragma unroll
    for (int s = 0; s < S; ++s) {
        float ss = 0;
        for (int j = 0; j < per_stream; ++j) ss += xsq[(s * per_stream + j) * T + t];
        rs[s] = rsqrtf(ss / D + EPS);
    }
    if (blockIdx.y == 0 && threadIdx.x < S) r[t * S + threadIdx.x] = rs[threadIdx.x];
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float x = 0;
    for (int sp = 0; sp < splits; ++sp) x += rs[sp / per_stream] * partial[((size_t)sp * T + t) * N + j];
    x /= S;
    if (j < RANK) z[t * RANK + j] = f2bf(silu(x));
    else if (inj) inj[t * S + j - RANK] = 2.0f * sigmoidf(x);
}

// m[t, d] = mean_s sigmoid(u[t, s, d]) * h[t, s, d] * r[t, s] * (1 + g[s, d]), FP32 residual input.
__global__ void hc_mix_small_kernel(const float* __restrict__ u, const float* __restrict__ h,
                                    const float* __restrict__ r, const __nv_bfloat16* __restrict__ g,
                                    __nv_bfloat16* m, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int t = i / D, d = i % D;
    float acc = 0;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const size_t j = (size_t)t * H + s * D + d;
        acc += sigmoidf(u[j]) * h[j] * r[t * S + s] * (1.0f + bf2f(g[s * D + d]));
    }
    m[i] = f2bf(acc / S);
}

// h[t, s, d] += y[t, d] * inj[t, s]
__global__ void hc_combine_kernel(float* h, const float* __restrict__ y, const float* __restrict__ inj, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const int t = i / H, s = (i % H) / D, d = i % D;
    h[i] += y[(size_t)t * D + d] * inj[t * S + s];
}

__global__ void f32_to_bf16_kernel(const float* __restrict__ x, __nv_bfloat16* y, size_t count) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) y[i] = f2bf(x[i]);
}

__global__ void add_f32_kernel(float* y, const float* __restrict__ x, size_t count) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) y[i] += x[i];
}

int blocks(size_t n, int threads) { return (int)((n + threads - 1) / threads); }

}  // namespace

extern "C" cudaError_t squash_embed(const int* tokens, int T, const void* table, float* h, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    embed_kernel<<<T, 256, 0, st>>>(tokens, static_cast<const __nv_bfloat16*>(table), h);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_grouped_rmsnorm(const float* h, const void* w, void* n, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    grouped_rmsnorm_kernel<<<T * S, 256, 0, st>>>(h, static_cast<const __nv_bfloat16*>(w), static_cast<__nv_bfloat16*>(n));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_hc_inject(const void* n, const void* w, float* inj, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    hc_inject_kernel<<<T, 256, 0, st>>>(static_cast<const __nv_bfloat16*>(n), static_cast<const __nv_bfloat16*>(w), inj);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_hc_act(const float* x, int ld, void* z, float* inj, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    hc_act_kernel<<<blocks((size_t)T * (RANK + S), 256), 256, 0, st>>>(x, ld, static_cast<__nv_bfloat16*>(z), inj, T);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_hc_mix(const float* u, const void* n, const void* g, void* m, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    hc_mix_kernel<<<blocks((size_t)T * D, 256), 256, 0, st>>>(u, static_cast<const __nv_bfloat16*>(n),
                                                               static_cast<const __nv_bfloat16*>(g),
                                                               static_cast<__nv_bfloat16*>(m), T);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_hc_finish(const float* partial, const float* xsq, int splits, int T, int N, void* z,
                                        float* inj, float* r, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    if (splits % S != 0) return cudaErrorInvalidValue;
    hc_finish_kernel<<<dim3(T, (N + 63) / 64), 64, 0, st>>>(partial, xsq, splits, T, N, static_cast<__nv_bfloat16*>(z), inj, r);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_hc_mix_small(const float* u, const float* h, const float* r, const void* g, void* m, int T,
                                           cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    hc_mix_small_kernel<<<blocks((size_t)T * D, 256), 256, 0, st>>>(u, h, r, static_cast<const __nv_bfloat16*>(g),
                                                                     static_cast<__nv_bfloat16*>(m), T);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_hc_combine(float* h, const float* y, const float* inj, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    hc_combine_kernel<<<blocks((size_t)T * H, 256), 256, 0, st>>>(h, y, inj, T);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_f32_to_bf16(const float* x, void* y, size_t count, cudaStream_t st) {
    if (count == 0) return cudaSuccess;
    f32_to_bf16_kernel<<<blocks(count, 256), 256, 0, st>>>(x, static_cast<__nv_bfloat16*>(y), count);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_add_f32(float* y, const float* x, size_t count, cudaStream_t st) {
    if (count == 0) return cudaSuccess;
    add_f32_kernel<<<blocks(count, 256), 256, 0, st>>>(y, x, count);
    SQ_RETURN_LAUNCH();
}
