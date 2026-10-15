// Gated DeltaNet (SPEC §5): causal conv, gating, the delta-rule recurrence and the gated norm.
//
// Per-sequence state: conv [3, 10240] FP32 (oldest first) and recurrent [48, 128(k), 128(v)] FP32.

#include "common.cuh"

namespace {

constexpr int C = 10240, KH = 16, VH = 48, DK = 128, DV = 128, TAPS = 4;
constexpr float EPS = 1e-6f;

// out[t, c] = silu(sum_j w[c, j] * x[t + j - 3, c]); history from `state`, which is then advanced.
__global__ void conv_kernel(const float* __restrict__ x, float* out, float* state, const __nv_bfloat16* __restrict__ w,
                            int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float w0 = bf2f(w[c * TAPS]), w1 = bf2f(w[c * TAPS + 1]), w2 = bf2f(w[c * TAPS + 2]), w3 = bf2f(w[c * TAPS + 3]);
    float p0 = state[c], p1 = state[C + c], p2 = state[2 * C + c];  // x[t-3], x[t-2], x[t-1]
    for (int t = 0; t < T; ++t) {
        const float cur = x[(size_t)t * C + c];
        out[(size_t)t * C + c] = silu(w0 * p0 + w1 * p1 + w2 * p2 + w3 * cur);
        p0 = p1;
        p1 = p2;
        p2 = cur;
    }
    state[c] = p0;
    state[C + c] = p1;
    state[2 * C + c] = p2;
}

// In place on the conv output: l2-normalize q and k per key head (q also / sqrt(128)); and
// beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias). One block of 128 threads per (t, key head).
__global__ void prep_kernel(float* qkv, const float* __restrict__ a, const float* __restrict__ b,
                            const __nv_bfloat16* __restrict__ a_log, const __nv_bfloat16* __restrict__ dt_bias,
                            float* beta, float* g) {
    __shared__ float scratch[32];
    const int t = blockIdx.x / KH, h = blockIdx.x % KH, i = threadIdx.x;
    float* q = qkv + (size_t)t * C + h * DK;
    float* k = qkv + (size_t)t * C + KH * DK + h * DK;
    const float qv = q[i], kv = k[i];
    const float qn = rsqrtf(block_sum(qv * qv, scratch) + EPS);
    const float kn = rsqrtf(block_sum(kv * kv, scratch) + EPS);
    q[i] = qv * qn * rsqrtf((float)DK);
    k[i] = kv * kn;
    if (i < 3) {
        const int vh = h * 3 + i;
        const float x = a[t * VH + vh] + bf2f(dt_bias[vh]);
        const float softplus = x > 20.0f ? x : log1pf(__expf(x));
        g[t * VH + vh] = -__expf(bf2f(a_log[vh])) * softplus;
        beta[t * VH + vh] = sigmoidf(b[t * VH + vh]);
    }
}

// Token-sequential delta rule. Each thread owns one value column of one head's state in registers;
// grid (48 heads, 4 column groups) x 32 threads.
__global__ void __launch_bounds__(32) recurrent_kernel(const float* __restrict__ qkv, const float* __restrict__ beta,
                                                       const float* __restrict__ g, float* state, float* out, int T) {
    __shared__ float sq[DK], sk[DK];
    const int vh = blockIdx.x, col = blockIdx.y * 32 + threadIdx.x, kh = vh / 3;
    float s[DK];
    float* st = state + (size_t)vh * DK * DV;
#pragma unroll
    for (int j = 0; j < DK; ++j) s[j] = st[j * DV + col];
    for (int t = 0; t < T; ++t) {
        const float* row = qkv + (size_t)t * C;
        __syncwarp();
#pragma unroll
        for (int j = threadIdx.x; j < DK; j += 32) {
            sq[j] = row[kh * DK + j];
            sk[j] = row[KH * DK + kh * DK + j];
        }
        __syncwarp();
        const float decay = __expf(g[t * VH + vh]);
        float kvm = 0;
#pragma unroll
        for (int j = 0; j < DK; ++j) {
            s[j] *= decay;
            kvm += s[j] * sk[j];
        }
        const float v = row[2 * KH * DK + vh * DV + col];
        const float delta = beta[t * VH + vh] * (v - kvm);
        float o = 0;
#pragma unroll
        for (int j = 0; j < DK; ++j) {
            s[j] += sk[j] * delta;
            o += s[j] * sq[j];
        }
        out[(size_t)t * VH * DV + vh * DV + col] = o;
    }
#pragma unroll
    for (int j = 0; j < DK; ++j) st[j * DV + col] = s[j];
}

// y[t, h, :] = bf16(rms(o[t, h, :]) * w * sigmoid(z[t, h, :])); one block of 128 per (t, head).
__global__ void gated_norm_kernel(const float* __restrict__ o, const float* __restrict__ z,
                                  const __nv_bfloat16* __restrict__ w, __nv_bfloat16* y) {
    __shared__ float scratch[32];
    const size_t base = (size_t)blockIdx.x * DV;
    const int i = threadIdx.x;
    const float x = o[base + i];
    const float r = rsqrtf(block_sum(x * x, scratch) / DV + EPS);
    y[base + i] = f2bf(x * r * bf2f(w[i]) * sigmoidf(z[base + i]));
}

}  // namespace

extern "C" cudaError_t squash_gdn_conv(const float* x, float* out, float* state, const void* w, int T, cudaStream_t st) {
    conv_kernel<<<C / 256, 256, 0, st>>>(x, out, state, static_cast<const __nv_bfloat16*>(w), T);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_gdn_prep(float* qkv, const float* a, const float* b, const void* a_log, const void* dt_bias,
                                       float* beta, float* g, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    prep_kernel<<<T * KH, DK, 0, st>>>(qkv, a, b, static_cast<const __nv_bfloat16*>(a_log),
                                       static_cast<const __nv_bfloat16*>(dt_bias), beta, g);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_gdn_recurrent(const float* qkv, const float* beta, const float* g, float* state, float* out,
                                            int T, cudaStream_t st) {
    recurrent_kernel<<<dim3(VH, DV / 32), 32, 0, st>>>(qkv, beta, g, state, out, T);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_gdn_gated_norm(const float* o, const float* z, const void* w, void* y, int T,
                                             cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    gated_norm_kernel<<<T * VH, DV, 0, st>>>(o, z, static_cast<const __nv_bfloat16*>(w), static_cast<__nv_bfloat16*>(y));
    SQ_RETURN_LAUNCH();
}
