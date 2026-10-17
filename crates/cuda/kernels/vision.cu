// Vision tower (SPEC §14): LayerNorm, bias/activation epilogues, 2D-RoPE QKV preparation and
// bidirectional attention for the 27-block ViT and its patch merger. Projections themselves run on
// the dense BF16 GEMM. Width 1152 = 16 heads x 72; activations FP32, GEMM inputs BF16.

#include "common.cuh"

namespace {

constexpr int VH = 16, VD = 72, VW = VH * VD;

// out[t] = LayerNorm(x[t]) * w + b over d columns (two-pass mean / variance, eps 1e-6), BF16.
__global__ void layernorm_kernel(const float* __restrict__ x, int d, const __nv_bfloat16* __restrict__ w,
                                 const __nv_bfloat16* __restrict__ b, __nv_bfloat16* out) {
    __shared__ float scratch[32];
    const float* row = x + (size_t)blockIdx.x * d;
    float s = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s += row[i];
    const float mean = block_sum(s, scratch) / d;
    float v = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float c = row[i] - mean;
        v += c * c;
    }
    const float r = rsqrtf(block_sum(v, scratch) / d + 1e-6f);
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        out[(size_t)blockIdx.x * d + i] = f2bf((row[i] - mean) * r * bf2f(w[i]) + bf2f(b[i]));
}

SQ_DEVICE float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

SQ_DEVICE float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)); }

// out[t, j] = act(y[t, j] + bias[j]) for j < n, BF16 with row stride ldo (act 0: none, 1: GELU tanh,
// 2: GELU erf).
__global__ void bias_act_kernel(const float* __restrict__ y, int n, const __nv_bfloat16* __restrict__ bias, int act,
                                __nv_bfloat16* out, int ldo) {
    const int t = blockIdx.y;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        float v = y[(size_t)t * n + j] + bf2f(bias[j]);
        if (act == 1) v = gelu_tanh(v);
        else if (act == 2) v = gelu_erf(v);
        out[(size_t)t * ldo + j] = f2bf(v);
    }
}

// h[t, j] (+)= y[t, j] + bias[j] (+ extra[t, j]); `assign` overwrites h instead of adding.
__global__ void residual_kernel(float* h, const float* __restrict__ y, const __nv_bfloat16* __restrict__ bias,
                                const float* __restrict__ extra, int n, int assign) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)gridDim.y * n;
    const size_t idx = (size_t)blockIdx.y * n + i;
    if (i >= (size_t)n || idx >= total) return;
    float v = y[idx] + bf2f(bias[i]);
    if (extra) v += extra[idx];
    h[idx] = assign ? v : h[idx] + v;
}

// qkv [t, 3 * 1152] (+ bias) -> q, k rotated by the patch's 2D RoPE (cos / sin [t, 72]) and v, each
// FP32 [t, 16, 72]. One block of 384 threads per token.
__global__ void qkv_rope_kernel(const float* __restrict__ qkv, const __nv_bfloat16* __restrict__ bias,
                                const float* __restrict__ cos, const float* __restrict__ sin, float* q, float* k, float* v) {
    const int t = blockIdx.x;
    const float* row = qkv + (size_t)t * 3 * VW;
    for (int i = threadIdx.x; i < 3 * VW; i += blockDim.x) {
        const int part = i / VW, j = i % VW, d = j % VD;
        const float x = row[i] + bf2f(bias[i]);
        if (part == 2) {
            v[(size_t)t * VW + j] = x;
            continue;
        }
        // rotate_half over the head: partner d +- 36.
        const int pd = d < VD / 2 ? d + VD / 2 : d - VD / 2;
        const float xp = row[i - d + pd] + bf2f(bias[i - d + pd]);
        const float rot = d < VD / 2 ? -xp : xp;
        const float r = x * cos[(size_t)t * VD + d] + rot * sin[(size_t)t * VD + d];
        (part == 0 ? q : k)[(size_t)t * VW + j] = r;
    }
}

// Bidirectional attention within segments of `seg` consecutive patches (one image frame each).
// Block (8 queries, head): warp w owns query q0 + w; key tiles of 32 are staged in shared memory,
// lane l scores key l of the tile, then each lane accumulates output dims l, l + 32, l + 64.
constexpr int AQ = 8, AK = 32, KPAD = VD + 1;

__global__ void __launch_bounds__(AQ * 32) attention_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                                            const float* __restrict__ v, int T, int seg, __nv_bfloat16* out) {
    __shared__ float ks[AK][KPAD], vs[AK][KPAD];
    __shared__ float qs[AQ][VD];
    const int head = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int q0 = blockIdx.x * AQ, qi = q0 + warp;
    // All queries of a block share a segment when seg is a multiple of AQ; otherwise per-warp bounds.
    const int s_begin = (q0 / seg) * seg, s_end = min(T, ((q0 + AQ - 1) / seg + 1) * seg);
    const int my_begin = (min(qi, T - 1) / seg) * seg, my_end = min(T, my_begin + seg);
    const bool active = qi < T;
    for (int i = threadIdx.x; i < AQ * VD; i += blockDim.x) {
        const int r = i / VD, d = i % VD;
        qs[r][d] = q0 + r < T ? q[(size_t)(q0 + r) * VW + head * VD + d] * 0.11785113019775792f : 0.0f;  // 72^-0.5
    }
    float m = -INFINITY, l = 0.0f, acc[3] = {0.0f, 0.0f, 0.0f};
    for (int k0 = s_begin; k0 < s_end; k0 += AK) {
        __syncthreads();
        for (int i = threadIdx.x; i < AK * VD; i += blockDim.x) {
            const int r = i / VD, d = i % VD, key = k0 + r;
            const bool ok = key < s_end;
            ks[r][d] = ok ? k[(size_t)key * VW + head * VD + d] : 0.0f;
            vs[r][d] = ok ? v[(size_t)key * VW + head * VD + d] : 0.0f;
        }
        __syncthreads();
        if (!active) continue;
        const int key = k0 + lane;
        float sc = -INFINITY;
        if (key >= my_begin && key < my_end) {
            sc = 0.0f;
#pragma unroll 8
            for (int d = 0; d < VD; ++d) sc += qs[warp][d] * ks[lane][d];
        }
        const float tile_max = warp_max(sc);
        if (tile_max == -INFINITY) continue;
        const float m_new = fmaxf(m, tile_max);
        const float corr = m == -INFINITY ? 0.0f : __expf(m - m_new);
        const float p = sc == -INFINITY ? 0.0f : __expf(sc - m_new);
        l = l * corr + warp_sum(p);
#pragma unroll
        for (int c = 0; c < 3; ++c) acc[c] *= corr;
        for (int j = 0; j < AK; ++j) {
            const float pj = __shfl_sync(0xffffffff, p, j);
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                const int d = lane + 32 * c;
                if (d < VD) acc[c] += pj * vs[j][d];
            }
        }
        m = m_new;
    }
    if (!active) return;
    const float inv = 1.0f / l;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const int d = lane + 32 * c;
        if (d < VD) out[(size_t)qi * VW + head * VD + d] = f2bf(acc[c] * inv);
    }
}

__global__ void to_bf16_kernel(const float* __restrict__ x, __nv_bfloat16* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2bf(x[i]);
}

// dst [n, ldd] (zeroed beyond k) <- src [n, k], BF16.
__global__ void pad_cols_kernel(const __nv_bfloat16* __restrict__ src, int k, __nv_bfloat16* dst, int ldd) {
    const int r = blockIdx.y;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < ldd; j += gridDim.x * blockDim.x)
        dst[(size_t)r * ldd + j] = j < k ? src[(size_t)r * k + j] : f2bf(0.0f);
}

// h[r, s, :] = BF16(emb[r, :]) for the 4 residual streams (vision tokens' input embedding).
__global__ void embed_rows_kernel(const float* __restrict__ emb, float* h) {
    constexpr int D = 2560, S = 4;
    const int r = blockIdx.x;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float v = bf2f(f2bf(emb[(size_t)r * D + d]));
#pragma unroll
        for (int s = 0; s < S; ++s) h[(size_t)r * S * D + s * D + d] = v;
    }
}

}  // namespace

extern "C" cudaError_t quench6k_embed_rows(const float* emb, int rows, float* h, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    embed_rows_kernel<<<rows, 256, 0, st>>>(emb, h);
    return cudaGetLastError();
}

extern "C" cudaError_t quench6k_vit_layernorm(const float* x, int t, int d, const void* w, const void* b, void* out, cudaStream_t st) {
    if (t == 0) return cudaSuccess;
    layernorm_kernel<<<t, 256, 0, st>>>(x, d, static_cast<const __nv_bfloat16*>(w), static_cast<const __nv_bfloat16*>(b),
                                        static_cast<__nv_bfloat16*>(out));
    return cudaGetLastError();
}

extern "C" cudaError_t quench6k_vit_bias_act(const float* y, int t, int n, const void* bias, int act, void* out, int ldo, cudaStream_t st) {
    if (t == 0) return cudaSuccess;
    bias_act_kernel<<<dim3((n + 255) / 256, t), 256, 0, st>>>(y, n, static_cast<const __nv_bfloat16*>(bias), act,
                                                               static_cast<__nv_bfloat16*>(out), ldo);
    return cudaGetLastError();
}

extern "C" cudaError_t quench6k_vit_residual(float* h, const float* y, const void* bias, const float* extra, int t, int n, int assign,
                                           cudaStream_t st) {
    if (t == 0) return cudaSuccess;
    residual_kernel<<<dim3((n + 255) / 256, t), 256, 0, st>>>(h, y, static_cast<const __nv_bfloat16*>(bias), extra, n, assign);
    return cudaGetLastError();
}

extern "C" cudaError_t quench6k_vit_qkv_rope(const float* qkv, const void* bias, const float* cos, const float* sin, int t, float* q,
                                           float* k, float* v, cudaStream_t st) {
    if (t == 0) return cudaSuccess;
    qkv_rope_kernel<<<t, 384, 0, st>>>(qkv, static_cast<const __nv_bfloat16*>(bias), cos, sin, q, k, v);
    return cudaGetLastError();
}

extern "C" cudaError_t quench6k_vit_attention(const float* q, const float* k, const float* v, int t, int seg, void* out, cudaStream_t st) {
    if (t == 0) return cudaSuccess;
    attention_kernel<<<dim3((t + AQ - 1) / AQ, VH), AQ * 32, 0, st>>>(q, k, v, t, seg, static_cast<__nv_bfloat16*>(out));
    return cudaGetLastError();
}

extern "C" cudaError_t quench6k_to_bf16(const float* x, size_t n, void* out, cudaStream_t st) {
    if (n == 0) return cudaSuccess;
    to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, static_cast<__nv_bfloat16*>(out), n);
    return cudaGetLastError();
}

extern "C" cudaError_t quench6k_pad_cols(const void* src, int n, int k, void* dst, int ldd, cudaStream_t st) {
    pad_cols_kernel<<<dim3((ldd + 255) / 256, n), 256, 0, st>>>(static_cast<const __nv_bfloat16*>(src), k, static_cast<__nv_bfloat16*>(dst), ldd);
    return cudaGetLastError();
}
