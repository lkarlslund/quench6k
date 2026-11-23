// Hy4-flash (hy_v4_flash, tencent/Hy4-flash): kernels of the text model that the shared GEMV/GEMM, top-k
// and LSE kernels do not cover.
//
// - Hidden state: 4 hyper-connection lanes of FP32 [T, 4, 3072] (BF16 lanes, as the transformers model keeps
//   them, are measurably farther from an FP32 forward on generated text). Each sublayer mixes the lanes into its
//   input (sigmoid weights over a normalized FP32 projection of all lanes), and adds its output back to
//   every lane with a per-lane weight.
// - Attention: gated MLA (q_lora 1024, kv_lora 512, nope 192 + rope 64 query/key dims, value 256) with
//   learned sinks, run absorbed: the cache holds one 576-wide row per position ([512 normalized latent |
//   64 rotated key]); a query head is [q_nope . W_uk (512) | rotated q (64)] and its output latent is
//   expanded by W_uv afterwards. Sliding layers (64 heads) see the last 512 positions; indexed layers (32
//   heads) see everything until the context exceeds 2048 positions, then the 511 four-position groups a
//   compressed indexer scores highest plus their own group.
// - MoE: sigmoid router with a selection bias, 16 of 512 experts on a 1536-wide latent (NVFP4, one FP32
//   scale per expert), a shared expert, and an RMSNorm over the sum.
// - Engram: hashed bigram/trigram embeddings (FP8 rows looked up on the host) gated into the lanes.
//
// Rounding follows the transformers model (BF16 activations): `rb` marks where it rounds to BF16.

#include <algorithm>

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "mma.cuh"
#include "variants.cuh"

// One kind of per-sequence cache of a layer: the row of position p of sequence s starts at element
// base[s] + off + (ring > 0 ? p % ring : p) * ld (ld: elements per position; the layers of a kind are
// interleaved per position). A step's rows carry their own position pos[t] and sequence seq[t].
struct SeqRows {
    const unsigned long long* base;
    long long off;
    int ld;
    int ring;
};

namespace {

using bf16 = __nv_bfloat16;

constexpr int D = 3072, HC = 4, KVL = 512, ROT = 64, NOPE = 192, QKD = NOPE + ROT;
constexpr int KROW = KVL + ROT;  // 576: one cache row
constexpr int LAT = 1536, E = 512, TOPK = 16;
// SEL: width of a query's key list (2048 selected positions, plus the drafter's query block).
constexpr int IH = 16, ID = 128, GRP = 4, TOPG = 512, SEL = TOPG * GRP + 16;
constexpr int EH = 16, EDIM = 96;
constexpr float TORCH_BF16_EPS = 0.0078125f;

SQ_DEVICE float rb(float x) { return bf2f(f2bf(x)); }

template <typename T>
SQ_DEVICE T* seq_row(const SeqRows& c, int s, int p) {
    return reinterpret_cast<T*>(c.base[s]) + c.off + (size_t)(c.ring > 0 ? p % c.ring : p) * c.ld;
}

SQ_DEVICE float2 bf2x2(const bf16* p) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
}

SQ_DEVICE float e4m3f(uint8_t v) {
    __nv_fp8_e4m3 f;
    f.__x = v;
    return float(f);
}

// RoPE angle of channel pair i of a 64-wide rotary slice at position p, as torch computes it in FP32
// (inv_freq = 1 / theta^(2i / 64)); cos and sin are rounded to BF16 like the model's tables.
SQ_DEVICE void rope_cs(int p, int i, float theta, float& c, float& s) {
    const float inv_freq = 1.0f / powf(theta, (float)(2 * i) / (float)ROT);
    const float angle = (float)p * inv_freq;
    float sn, cs;
    sincosf(angle, &sn, &cs);
    c = rb(cs);
    s = rb(sn);
}

// ---- Hyper-connections -----------------------------------------------------------------------------

// Hyper-connection pre-mix in two kernels. hc_mix_partial: block (slice, token) sums, over its slice of
// the 4 * 3072 lane values, flat . fn[j] for the NM mix rows and flat^2; partial [T, HC_SLICES, NM + 1].
constexpr int HC_SLICES = 24, HC_SLICE = HC * D / HC_SLICES;  // 512 values per slice
template <int NM>
__global__ void __launch_bounds__(256) hc_mix_partial_kernel(const float* __restrict__ lanes, const float* __restrict__ fn,
                                                             float* partial) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float red[8][NM + 1];
    const int slice = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int i = slice * HC_SLICE + tid * 2;
    const float2 v = *reinterpret_cast<const float2*>(lanes + (size_t)t * HC * D + i);
    float acc[NM + 1];
#pragma unroll
    for (int j = 0; j < NM; ++j) {
        const float2 w = *reinterpret_cast<const float2*>(fn + (size_t)j * HC * D + i);
        acc[j] = w.x * v.x + w.y * v.y;
    }
    acc[NM] = v.x * v.x + v.y * v.y;
#pragma unroll
    for (int j = 0; j <= NM; ++j) acc[j] = warp_sum(acc[j]);
    if (lane == 0)
#pragma unroll
        for (int j = 0; j <= NM; ++j) red[warp][j] = acc[j];
    __syncthreads();
    if (tid <= NM) {
        float s = 0.0f;
        for (int w = 0; w < 8; ++w) s += red[w][tid];
        partial[((size_t)t * HC_SLICES + slice) * (NM + 1) + tid] = s;
    }
}

// One block per token: mixes[j] = (flat . fn[j]) * rsqrt(mean(flat^2) + eps) from the slices' partial sums
// (in slice order); NM = 8 for a sublayer (4 pre + 4 post weights), 4 for the final head (pre only). The
// mixed input x = bf16(sum_c pre_c lane_c) is then normalized with `norm_w` (eps): out = bf16(w * bf16(x * r)).
template <int NM>
__global__ void __launch_bounds__(512) hc_pre_kernel(const float* __restrict__ lanes, const float* __restrict__ partial,
                                                     const float* __restrict__ base, const float* __restrict__ scale,
                                                     const bf16* __restrict__ norm_w, float hc_eps, float mag, float eps,
                                                     bf16* out, float* post) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float coef[HC];
    __shared__ float xs[D];
    __shared__ float scratch[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float* x = lanes + (size_t)t * HC * D;
    if (tid < 32) {
        // Lane j sums mix row j (lane NM: flat^2) over the slices, in slice order.
        float m = 0.0f;
        if (tid <= NM)
            for (int sl = 0; sl < HC_SLICES; ++sl) m += __ldcg(partial + ((size_t)t * HC_SLICES + sl) * (NM + 1) + tid);
        const float r = rsqrtf(__shfl_sync(0xffffffff, m, NM) / (float)(HC * D) + eps);
        if (tid < HC) coef[tid] = 1.0f / (1.0f + expf(-(m * r * scale[0] + base[tid]))) + hc_eps;
        if (NM == 2 * HC && tid >= HC && tid < 2 * HC)
            post[t * HC + tid - HC] = mag / (1.0f + expf(-(m * r * scale[1] + base[tid]))) + hc_eps;
    }
    __syncthreads();
    float ss2 = 0.0f;
    for (int d = tid; d < D; d += 512) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += coef[c] * x[c * D + d];
        s = rb(s);
        xs[d] = s;
        ss2 += s * s;
    }
    const float r2 = rsqrtf(block_sum(ss2, scratch) / (float)D + eps);
    for (int d = tid; d < D; d += 512) out[(size_t)t * D + d] = f2bf(bf2f(norm_w[d]) * rb(xs[d] * r2));
}

// lanes[t, c, d] += post[t, c] * y[t, d]; y BF16 rows of stride ldy.
// A block of 256 lies within one token (HC * D is a multiple of 256): its 4 post weights are read once,
// through L2 (hc_pre rewrote them just before).
__global__ void hc_post_kernel(float* lanes, const float* post, const bf16* __restrict__ y, int ldy, int T) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float pw[HC];
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int t = (int)((size_t)blockIdx.x * blockDim.x / (HC * D));
    if (threadIdx.x < HC && t < T) pw[threadIdx.x] = __ldcg(post + t * HC + threadIdx.x);
    __syncthreads();
    if (i >= (size_t)T * HC * D) return;
    const int c = (int)(i / D % HC), d = (int)(i % D);
    lanes[i] = pw[c] * bf2f(y[(size_t)t * ldy + d]) + lanes[i];
}

// lanes[t, c, :] = emb[t, :] for every lane.
__global__ void hc_expand_kernel(const bf16* __restrict__ emb, float* lanes, int T) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * HC * D) return;
    lanes[i] = bf2f(emb[(i / (HC * D)) * D + i % D]);
}

// ---- Norms and elementwise ---------------------------------------------------------------------------

// out[r, :n] = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))) with x = bf16(in[r, :n]) (a BF16 linear output).
// One block of 256 threads per row; n <= 4096.
__global__ void __launch_bounds__(256) rms_rows_kernel(const float* __restrict__ in, int ldi, const bf16* __restrict__ w,
                                                       int n, float eps, bf16* out, int ldo) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const float* x = in + (size_t)blockIdx.x * ldi;
    float v[16], ss = 0.0f;
    int c = 0;
    for (int i = threadIdx.x; i < n; i += 256, ++c) {
        v[c] = rb(x[i]);
        ss += v[c] * v[c];
    }
    const float r = rsqrtf(block_sum(ss, scratch) / (float)n + eps);
    c = 0;
    for (int i = threadIdx.x; i < n; i += 256, ++c) out[(size_t)blockIdx.x * ldo + i] = f2bf(bf2f(w[i]) * rb(v[c] * r));
}

// act[r, i] = bf16(bf16(silu(g)) * u) with g = bf16(gu[r, i]) and u = bf16(gu[r, ldu_off + i]); with `limit`
// > 0 the routed experts' clamps (g <= limit, |u| <= limit) apply.
__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, int ld, int width, float limit,
                              bf16* act, int lda, int rows) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * width) return;
    const size_t r = i / width, c = i % width;
    float gv = rb(g[r * ld + c]), uv = rb(u[r * ld + c]);
    if (limit > 0) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    act[r * lda + c] = f2bf(rb(silu(gv)) * uv);
}

// out[i] = bf16(bf16(a[i]) + bf16(b[i])) over rows of `n` (strides lda, ldb, ldo).
__global__ void add_rows_kernel(const float* __restrict__ a, int lda, const float* __restrict__ b, int ldb, int n, int rows,
                                float* out, int ldo) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * n) return;
    const size_t r = i / n, c = i % n;
    out[r * ldo + c] = rb(rb(a[r * lda + c]) + rb(b[r * ldb + c]));
}

// y = bf16(x); triggers its dependents only after its own dependency wait (see experts_gate_up_kernel).
__global__ void to_bf16_late_kernel(const float* __restrict__ x, bf16* y, size_t count) {
    pdl_wait();
    pdl_launch_dependents();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) y[i] = f2bf(x[i]);
}

// BF16 [N, K] -> FP8 E4M3 [N, K] with an FP32 scale per 128 x 128 block (amax / 448; [ceil(N / 128), K / 128]),
// values rounded to nearest. One block of 256 threads per 128 x 128 tile.
__global__ void __launch_bounds__(256) quantize_fp8b_kernel(const bf16* __restrict__ w, int N, int K, uint8_t* q, float* scale) {
    __shared__ float scratch[32];
    const int bn = blockIdx.y, bk = blockIdx.x, tid = threadIdx.x;
    const int rows = min(128, N - bn * 128);
    float amax = 0.0f;
    for (int i = tid; i < rows * 128; i += 256) {
        const int r = i / 128, c = i % 128;
        amax = fmaxf(amax, fabsf(bf2f(w[(size_t)(bn * 128 + r) * K + bk * 128 + c])));
    }
    amax = block_max(amax, scratch);
    const float sc = amax > 0.0f ? amax / 448.0f : 1.0f;
    if (tid == 0) scale[(size_t)bn * (K / 128) + bk] = sc;
    for (int i = tid; i < rows * 128; i += 256) {
        const int r = i / 128, c = i % 128;
        const size_t at = (size_t)(bn * 128 + r) * K + bk * 128 + c;
        const __nv_fp8_e4m3 v(bf2f(w[at]) / sc);
        q[at] = v.__x;
    }
}

// ---- MLA -------------------------------------------------------------------------------------------

// One block (64 threads) per (token, head) for the query and one per token for the key (blockIdx.y == H).
// q: q_b output F32 [T, H * 256] (ldq); ckv: kv_a output F32 [T, 576] (ldc). Query: qn[t, h] = bf16(q nope)
// [T, H, 192]; qa[t, h, 512..576) = rotated bf16(q rope). Key: the cache row of position pos[t] of sequence
// seq[t] gets the RMS-normalized latent (kv_a_layernorm) and the rotated key.
__global__ void __launch_bounds__(64) mla_prep_kernel(const float* __restrict__ q, int ldq, const float* __restrict__ ckv, int ldc,
                                                      const bf16* __restrict__ kv_norm, float eps, float theta, int H,
                                                      const int* pos, const int* seq, SeqRows cache, bf16* qn, bf16* qa) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const int p = pos[t];
    if (h < H) {
        const float* src = q + (size_t)t * ldq + h * QKD;
        bf16* dn = qn + ((size_t)t * H + h) * NOPE;
        for (int j = i; j < NOPE; j += 64) dn[j] = f2bf(src[j]);
        if (i < ROT / 2) {
            float c, s;
            rope_cs(p, i, theta, c, s);
            const float x0 = rb(src[NOPE + 2 * i]), x1 = rb(src[NOPE + 2 * i + 1]);
            bf16* dq = qa + ((size_t)t * H + h) * KROW + KVL;
            dq[2 * i] = f2bf(x0 * c - x1 * s);
            dq[2 * i + 1] = f2bf(x1 * c + x0 * s);
        }
        return;
    }
    const float* src = ckv + (size_t)t * ldc;
    bf16* row = seq_row<bf16>(cache, seq[t], p);
    float v[KVL / 64], ss = 0.0f;
#pragma unroll
    for (int j = 0; j < KVL / 64; ++j) {
        v[j] = rb(src[i + 64 * j]);
        ss += v[j] * v[j];
    }
    const float r = rsqrtf(block_sum(ss, scratch) / (float)KVL + eps);
#pragma unroll
    for (int j = 0; j < KVL / 64; ++j) row[i + 64 * j] = f2bf(bf2f(kv_norm[i + 64 * j]) * rb(v[j] * r));
    if (i < ROT / 2) {
        float c, s;
        rope_cs(p, i, theta, c, s);
        const float x0 = rb(src[KVL + 2 * i]), x1 = rb(src[KVL + 2 * i + 1]);
        row[KVL + 2 * i] = f2bf(x0 * c - x1 * s);
        row[KVL + 2 * i + 1] = f2bf(x1 * c + x0 * s);
    }
}

// Per-head small-batch GEMV: y[t, h, n] = sum_k W[h][n, k] x[t, h, k] for T <= 16 tokens. W BF16 per head
// [N, K] at stride wh (elements) between heads; x BF16 with ldx per token and xh per head; y BF16 with ldy
// per token and yh per head. With `gate` (F32 [T, H * N], ldg) the output is bf16(bf16(y) * bf16(sigmoid(bf16(g)))).
// Grid (H, N / 32); 8 warps, 4 rows each.
template <int MAXT>
__global__ void __launch_bounds__(256) heads_gemv_kernel(const bf16* __restrict__ x, int ldx, int xh, const bf16* __restrict__ W,
                                                         size_t wh, int N, int K, int T, const float* __restrict__ gate, int ldg,
                                                         bf16* y, int ldy, int yh) {
    pdl_launch_dependents();
    pdl_wait();
    // blockIdx.z: rows MAXT z .. of the step.
    x += (size_t)blockIdx.z * MAXT * ldx;
    y += (size_t)blockIdx.z * MAXT * ldy;
    if (gate) gate += (size_t)blockIdx.z * MAXT * ldg;
    T = min(MAXT, T - (int)blockIdx.z * MAXT);
    const int h = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n0 = blockIdx.y * 32 + warp * 4;
    const bf16* w = W + (size_t)h * wh;
    float acc[4][MAXT] = {};
    for (int k = lane * 8; k < K; k += 256) {
        float wf[4][8];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const uint4 raw = *reinterpret_cast<const uint4*>(w + (size_t)(n0 + r) * K + k);
            const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(p[j]);
                wf[r][2 * j] = f.x;
                wf[r][2 * j + 1] = f.y;
            }
        }
#pragma unroll
        for (int m = 0; m < MAXT; ++m) {
            if (m >= T) break;
            const uint4 raw = *reinterpret_cast<const uint4*>(x + (size_t)m * ldx + (size_t)h * xh + k);
            const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
            float xf[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(p[j]);
                xf[2 * j] = f.x;
                xf[2 * j + 1] = f.y;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[r][m] = fmaf(wf[r][j], xf[j], acc[r][m]);
        }
    }
    auto put = [&](int r, int m, float v) {
        const int n = n0 + r;
        float o = rb(v);
        if (gate) o = o * rb(1.0f / (1.0f + expf(-rb(gate[(size_t)m * ldg + h * N + n]))));
        y[(size_t)m * ldy + (size_t)h * yh + n] = f2bf(o);
    };
    if constexpr (MAXT >= 4) {
        // NS sums at a time, each bitwise as its own warp_sum; slot j (r = j / MAXT, m = j % MAXT of the
        // batch) ends in lanes j * 32 / NS ...
        constexpr int NS = 4 * MAXT < 32 ? 4 * MAXT : 32;
#pragma unroll
        for (int b = 0; b < 4 * MAXT / NS; ++b) {
            float v[NS];
#pragma unroll
            for (int j = 0; j < NS; ++j) v[j] = acc[(b * NS + j) / MAXT][(b * NS + j) % MAXT];
            const float sum = warp_sum_slots(v);
            const int j = b * NS + lane / (32 / NS), r = j / MAXT, m = j % MAXT;
            if (lane % (32 / NS) == 0 && m < T) put(r, m, sum);
        }
    } else {
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int m = 0; m < MAXT; ++m) {
                if (m >= T) break;
                const float v = warp_sum(acc[r][m]);
                if (lane == 0) put(r, m, v);
            }
    }
}

// v[t, h, :] *= bf16(sigmoid(bf16(gate[t, h, :]))) in place (BF16), for the prefill path.
__global__ void gate_mul_kernel(bf16* v, const float* __restrict__ gate, size_t count) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    v[i] = f2bf(bf2f(v[i]) * rb(1.0f / (1.0f + expf(-rb(gate[i])))));
}

// Absorbed MLA attention. Block (t, G head groups of 16, split) with 4 warps per group; key tiles of 32 cache
// rows of the query's sequence shared by the block's groups. Keys of query t (position p = pos[t]): MODE 0 sliding
// (max(0, p - 511) .. p), 1 dense (0 .. p), 2 selected (sel[t, 0 .. sel_n[t]) positions). Scores =
// qa . row / 16; the head's sink adds exp(sink) to the softmax denominator. splits == 1: out BF16
// [T, H, 512]; otherwise unnormalized partial sums part_o [split, T, H, 512] and (max, sum) part_ml
// [split, T, H, 2]. Warp w of a group: S over keys 8w .. 8w + 7, then output dims 128w .. 128w + 127.
constexpr int AT_KT = 32, AT_HG = 16, AT_LD = KROW + 8;
constexpr size_t at_smem(int G) {
    return (size_t)(G * AT_HG * AT_LD + 2 * AT_KT * AT_LD + G * AT_HG * (AT_KT + 8)) * 2 + (size_t)G * AT_HG * (AT_KT + 1) * 4 +
           G * AT_HG * 3 * 4;
}

template <int MODE, int G>
__global__ void __launch_bounds__(128 * G) mla_attn_kernel(const bf16* __restrict__ qa, int H, SeqRows cache,
                                                          const int* __restrict__ sel, const int* __restrict__ sel_n,
                                                          const float* __restrict__ sinks, const int* pos, const int* seq, int T,
                                                          bf16* out, float* part_o, float* part_ml) {
    pdl_launch_dependents();
    pdl_wait();
    constexpr int NT = 128 * G, R = G * AT_HG;  // threads; query rows (heads) of the block
    extern __shared__ __align__(16) unsigned char smem[];
    bf16* Qs = reinterpret_cast<bf16*>(smem);                          // [R][AT_LD]
    bf16* Ks = Qs + R * AT_LD;                                         // 2 stages of [AT_KT][AT_LD]
    bf16* Ps = Ks + 2 * AT_KT * AT_LD;                                 // [R][AT_KT + 8]
    float* Ss = reinterpret_cast<float*>(Ps + R * (AT_KT + 8));        // [R][AT_KT + 1]
    float* row_m = Ss + R * (AT_KT + 1);
    float* row_l = row_m + R;
    float* row_a = row_l + R;
    const int blocks_per_t = H / R;
    const int t = blockIdx.x / blocks_per_t, h0 = blockIdx.x % blocks_per_t * R, split = blockIdx.y, splits = gridDim.y;
    const int tid = threadIdx.x, lane = tid & 31, warp_all = tid >> 5;
    const int grp = warp_all >> 2, warp = warp_all & 3;  // head group of the block; warp within it
    const int p = pos[t];
    const bf16* cbase = reinterpret_cast<const bf16*>(cache.base[seq[t]]) + cache.off;
    int n_keys, k_first = 0;
    if (MODE == 0) {
        k_first = max(0, p - 511);
        n_keys = p - k_first + 1;
    } else if (MODE == 1) {
        n_keys = p + 1;
    } else {
        n_keys = __ldcg(sel_n + t);
    }
    const int tiles = (n_keys + AT_KT - 1) / AT_KT;
    const int per = (tiles + splits - 1) / splits;
    const int tile0 = split * per, tile1 = min(tiles, tile0 + per);
    // Key lists and positions are rewritten in place every layer and step: read through L2.
    auto key_pos = [&](int j) { return MODE == 2 ? __ldcg(sel + (size_t)t * SEL + j) : k_first + j; };
    auto load_tile = [&](int tile, int stage) {
        bf16* dst = Ks + stage * AT_KT * AT_LD;
        for (int c = tid; c < AT_KT * (KROW / 8); c += NT) {
            const int r = c / (KROW / 8), col = (c % (KROW / 8)) * 8;
            const int j = tile * AT_KT + r;
            const bool ok = j < n_keys;
            const int kp = ok ? key_pos(j) : 0;
            const size_t row = cache.ring > 0 ? (size_t)(kp % cache.ring) : (size_t)kp;
            cp_async16(dst + r * AT_LD + col, cbase + row * cache.ld + col, ok);
        }
    };
    for (int c = tid; c < R * (KROW / 8); c += NT) {
        const int r = c / (KROW / 8), col = (c % (KROW / 8)) * 8;
        cp_async16(Qs + r * AT_LD + col, qa + ((size_t)t * H + h0 + r) * KROW + col, true);
    }
    if (tid < R) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
    }
    if (tile0 < tile1) load_tile(tile0, 0);
    cp_async_commit();
    float o[16][4] = {};  // this warp's 16 x 128 slice of its group's output
    const float scale = 1.0f / 16.0f;  // 256^-0.5
    const bf16* Qg = Qs + grp * AT_HG * AT_LD;
    bf16* Pg = Ps + grp * AT_HG * (AT_KT + 8);
    float* Sg = Ss + grp * AT_HG * (AT_KT + 1);
    float *mg = row_m + grp * AT_HG, *lg = row_l + grp * AT_HG, *ag = row_a + grp * AT_HG;
    for (int tile = tile0; tile < tile1; ++tile) {
        const int stage = (tile - tile0) & 1;
        if (tile + 1 < tile1) load_tile(tile + 1, stage ^ 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const bf16* Kt = Ks + stage * AT_KT * AT_LD;
        float sacc[4] = {};
#pragma unroll 4
        for (int kk = 0; kk < KROW; kk += 16) {
            uint32_t a[4], b[2];
            ldmatrix_x4(a, Qg + (lane & 15) * AT_LD + kk + (lane >> 4) * 8);
            ldmatrix_x2(b, Kt + (warp * 8 + (lane & 7)) * AT_LD + kk + ((lane >> 3) & 1) * 8);
            mma_bf16(sacc, a, b);
        }
        {
            const int g = lane >> 2, c0 = warp * 8 + (lane & 3) * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = g + (e >> 1) * 8, c = c0 + (e & 1);
                const int j = tile * AT_KT + c;
                Sg[r * (AT_KT + 1) + c] = j < n_keys ? rb(rb(sacc[e]) * scale) : -INFINITY;
            }
        }
        __syncthreads();
        // Online softmax: 8 threads per head row of the group, 4 keys each.
        {
            const int r = (tid & 127) >> 3, q = tid & 7;
            float v[4], mx = -INFINITY;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                v[e] = Sg[r * (AT_KT + 1) + q * 4 + e];
                mx = fmaxf(mx, v[e]);
            }
#pragma unroll
            for (int off = 4; off >= 1; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            const float m_old = mg[r], m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float pr = m_new == -INFINITY ? 0.0f : __expf(v[e] - m_new);
                sum += pr;
                Pg[r * (AT_KT + 8) + q * 4 + e] = f2bf(pr);
            }
#pragma unroll
            for (int off = 4; off >= 1; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
            __syncwarp();
            if (q == 0) {
                const float alpha = m_old == -INFINITY ? 0.0f : __expf(m_old - m_new);
                ag[r] = alpha;
                lg[r] = lg[r] * alpha + sum;
                mg[r] = m_new;
            }
        }
        __syncthreads();
        {
            const int g = lane >> 2;
            const float a0 = ag[g], a1 = ag[g + 8];
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                o[nt][0] *= a0;
                o[nt][1] *= a0;
                o[nt][2] *= a1;
                o[nt][3] *= a1;
            }
#pragma unroll
            for (int kk = 0; kk < AT_KT; kk += 16) {
                uint32_t a[4];
                ldmatrix_x4(a, Pg + (lane & 15) * (AT_KT + 8) + kk + (lane >> 4) * 8);
#pragma unroll
                for (int nt = 0; nt < 16; nt += 2) {
                    uint32_t b[4];
                    const int m = lane >> 3, rr = lane & 7;
                    ldmatrix_x4_trans(b, Kt + (kk + (m & 1) * 8 + rr) * AT_LD + warp * 128 + nt * 8 + (m >> 1) * 8);
                    mma_bf16(o[nt], a, &b[0]);
                    mma_bf16(o[nt + 1], a, &b[2]);
                }
            }
        }
        __syncthreads();
    }
    cp_async_wait<0>();
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        const int r = g + hh * 8, h = h0 + grp * AT_HG + r;
        const float m = mg[r], l = lg[r];
        if (splits == 1) {
            const float denom = l + (m == -INFINITY ? __expf(sinks[h]) : __expf(sinks[h] - m));
            const float inv = 1.0f / denom;
            bf16* dst = out + ((size_t)t * H + h) * KVL + warp * 128;
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                const int c = nt * 8 + t4 * 2;
                *reinterpret_cast<__nv_bfloat162*>(dst + c) = __floats2bfloat162_rn(o[nt][hh * 2] * inv, o[nt][hh * 2 + 1] * inv);
            }
        } else {
            float* dst = part_o + (((size_t)split * T + t) * H + h) * KVL + warp * 128;
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                const int c = nt * 8 + t4 * 2;
                dst[c] = o[nt][hh * 2];
                dst[c + 1] = o[nt][hh * 2 + 1];
            }
            if (warp == 0 && t4 == 0) {
                float* ml = part_ml + (((size_t)split * T + t) * H + h) * 2;
                ml[0] = m;
                ml[1] = l;
            }
        }
    }
}

// out[t, h, :] = sum_s e^(m_s - m) O_s / (sum_s e^(m_s - m) l_s + e^(sink_h - m)). Block (t, h), 128 threads.
__global__ void __launch_bounds__(128) mla_merge_kernel(const float* __restrict__ part_o, const float* __restrict__ part_ml, int splits,
                                                       const float* __restrict__ sinks, int T, int H, bf16* out) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x / H, h = blockIdx.x % H;
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, __ldcg(part_ml + (((size_t)s * T + t) * H + h) * 2));
    float l = 0.0f, acc[4] = {};
    for (int s = 0; s < splits; ++s) {
        const float* ml = part_ml + (((size_t)s * T + t) * H + h) * 2;
        const float m_s = __ldcg(ml);
        if (m_s == -INFINITY) continue;
        const float w = __expf(m_s - m);
        l += w * __ldcg(ml + 1);
        const float* po = part_o + (((size_t)s * T + t) * H + h) * KVL;
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] += w * __ldcg(po + threadIdx.x + 128 * j);
    }
    const float inv = 1.0f / (l + (m == -INFINITY ? __expf(sinks[h]) : __expf(sinks[h] - m)));
#pragma unroll
    for (int j = 0; j < 4; ++j) out[((size_t)t * H + h) * KVL + threadIdx.x + 128 * j] = f2bf(acc[j] * inv);
}

// ---- Compressed indexer --------------------------------------------------------------------------------

// One block (128 threads) per token. tk: token_key_projection F32 [T, 128] (ldk) -> RMS-normalized BF16 into
// the token-key cache of sequence seq[t] at position pos[t]; iq: query_projection F32 [T, 16 * 128] (ldq) -> BF16 with the
// last 64 dims of each head rotated (theta) into iq_out [T, 16, 128]; hw: head_weight_projection F32
// [T, 16] (ldw) -> bf16 value * 16^-0.5 * 128^-0.5 into hw_out [T, 16].
__global__ void __launch_bounds__(128) idx_prep_kernel(const float* __restrict__ tk, int ldk, const bf16* __restrict__ tk_norm, float eps,
                                                      const float* __restrict__ iq, int ldq, const float* __restrict__ hw, int ldw,
                                                      float theta, const int* pos, const int* seq, SeqRows tk_cache, bf16* iq_out,
                                                      float* hw_out) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const int t = blockIdx.x, i = threadIdx.x;
    const int p = pos[t];
    const float v = rb(tk[(size_t)t * ldk + i]);
    const float r = rsqrtf(block_sum(v * v, scratch) / (float)ID + eps);
    seq_row<bf16>(tk_cache, seq[t], p)[i] = f2bf(bf2f(tk_norm[i]) * rb(v * r));
    for (int j = i; j < IH * ID; j += 128) {
        const int h = j / ID, d = j % ID;
        const float* src = iq + (size_t)t * ldq + h * ID;
        float o;
        if (d < ID - ROT) {
            o = src[d];
        } else {
            const int pr = (d - (ID - ROT)) / 2;
            float c, s;
            rope_cs(p, pr, theta, c, s);
            const float x0 = rb(src[ID - ROT + 2 * pr]), x1 = rb(src[ID - ROT + 2 * pr + 1]);
            o = (d & 1) ? x1 * c + x0 * s : x0 * c - x1 * s;
        }
        iq_out[(size_t)t * IH * ID + j] = f2bf(o);
    }
    if (i < IH) hw_out[t * IH + i] = rb(hw[(size_t)t * ldw + i]) * (0.25f * 0.08838834764831845f);
}

// Pools the group each row completes (one block of 128 threads per row; rows at positions 4g + 3 complete
// group g of their sequence) from the token-key cache: per dim d,
// value_s = Wv[d] . k_s and gate_s = Wg[d] . k_s + ape[s, d] over the group's 4 keys k_s (FP32), pooled =
// sum_s value_s softmax_s(gate_s), RMS-normalized with the FP32 key_norm weight, rounded to BF16 and its
// last 64 dims rotated at the group's first position.
__global__ void __launch_bounds__(128) idx_pool_kernel(SeqRows tk_cache, const float* __restrict__ wv,
                                                      const float* __restrict__ wg, const float* __restrict__ ape,
                                                      const float* __restrict__ key_norm, float eps, float theta,
                                                      const int* pos, const int* seq, SeqRows pooled) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float ks[GRP][ID];
    __shared__ float ps[ID];
    __shared__ float scratch[32];
    const int p = pos[blockIdx.x], sq = seq[blockIdx.x], d = threadIdx.x;
    if ((p + 1) % GRP != 0) return;
    const int g = p / GRP;
#pragma unroll
    for (int s = 0; s < GRP; ++s) ks[s][d] = bf2f(__ldcg(seq_row<const bf16>(tk_cache, sq, g * GRP + s) + d));
    __syncthreads();
    float val[GRP] = {}, gate[GRP] = {};
    for (int k = 0; k < ID; ++k) {
        const float a = wv[d * ID + k], b = wg[d * ID + k];
#pragma unroll
        for (int s = 0; s < GRP; ++s) {
            val[s] = fmaf(a, ks[s][k], val[s]);
            gate[s] = fmaf(b, ks[s][k], gate[s]);
        }
    }
    float mx = -INFINITY;
#pragma unroll
    for (int s = 0; s < GRP; ++s) {
        gate[s] += ape[s * ID + d];
        mx = fmaxf(mx, gate[s]);
    }
    float den = 0.0f;
#pragma unroll
    for (int s = 0; s < GRP; ++s) den += expf(gate[s] - mx);
    // softmax then weighted sum, as (values * softmax).sum
    float pooled_d = 0.0f;
#pragma unroll
    for (int s = 0; s < GRP; ++s) pooled_d += val[s] * (expf(gate[s] - mx) / den);
    const float r = rsqrtf(block_sum(pooled_d * pooled_d, scratch) / (float)ID + eps);
    ps[d] = rb(pooled_d * r * key_norm[d]);
    __syncthreads();
    float o = ps[d];
    if (d >= ID - ROT) {
        const int pr = (d - (ID - ROT)) / 2;
        float c, s;
        rope_cs(g * GRP, pr, theta, c, s);
        const float x0 = ps[ID - ROT + 2 * pr], x1 = ps[ID - ROT + 2 * pr + 1];
        o = (d & 1) ? x1 * c + x0 * s : x0 * c - x1 * s;
    }
    seq_row<bf16>(pooled, sq, g)[d] = f2bf(o);
}

// scores[t, g] = sum_h hw[t, h] relu(iq[t, h] . pooled[g]) for g < pos[t] / 4 (the row's sequence's groups),
// else -inf; g < G. Block (64 groups, t), 8 warps; lane l takes dims 4l .. 4l + 3.
__global__ void __launch_bounds__(256) idx_scores_kernel(const bf16* __restrict__ iq, const float* __restrict__ hw, SeqRows pooled,
                                                        const int* pos, const int* seq, int G, float* scores, int ld) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float qs[IH][ID];
    __shared__ float ws[IH];
    const int t = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int qg = pos[t] / GRP;
    // Queries within the first 2048 positions do not select (idx_expand lists their keys densely).
    if (qg <= TOPG - 1) return;
    const int sq = seq[t];
    for (int j = threadIdx.x; j < IH * ID; j += 256) qs[j / ID][j % ID] = bf2f(__ldcg(iq + (size_t)t * IH * ID + j));
    if (threadIdx.x < IH) ws[threadIdx.x] = __ldcg(hw + t * IH + threadIdx.x);
    __syncthreads();
    for (int gi = warp; gi < 64; gi += 8) {
        const int g = blockIdx.x * 64 + gi;
        if (g >= G) break;
        if (g >= qg) {
            if (lane == 0) scores[(size_t)t * ld + g] = -INFINITY;
            continue;
        }
        const uint2 raw = __ldcg(reinterpret_cast<const uint2*>(seq_row<const bf16>(pooled, sq, g) + lane * 4));
        const __nv_bfloat162* pk = reinterpret_cast<const __nv_bfloat162*>(&raw);
        const float2 k01 = __bfloat1622float2(pk[0]), k23 = __bfloat1622float2(pk[1]);
        float total = 0.0f;
#pragma unroll
        for (int h = 0; h < IH; ++h) {
            const float* q = &qs[h][lane * 4];
            float s = q[0] * k01.x + q[1] * k01.y + q[2] * k23.x + q[3] * k23.y;
            s = warp_sum(s);
            total += ws[h] * fmaxf(s, 0.0f);
        }
        if (lane == 0) scores[(size_t)t * ld + g] = total;
    }
}

// Selected positions of query t: groups picked[t, 0 .. 511) that close before its own group (all of them
// when fewer than 511 do), expanded to their 4 positions, then its own group's positions up to itself.
// One warp per token.
__global__ void idx_expand_kernel(const int* __restrict__ picked, int npick, const int* pos, int T, int* sel, int* sel_n) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (t >= T) return;
    const int p = pos[t], qg = p / GRP;
    int* dst = sel + (size_t)t * SEL;
    int n = 0;
    if (qg <= TOPG - 1) {
        // Every earlier group.
        for (int j = lane; j < qg * GRP; j += 32) dst[j] = j;
        n = qg * GRP;
    } else {
        for (int base = 0; base < npick; base += 32) {
            const int j = base + lane;
            const int g = j < npick ? __ldcg(picked + (size_t)t * npick + j) : -1;
            const bool ok = g >= 0 && g < qg;
            const unsigned mask = __ballot_sync(0xffffffff, ok);
            const int before = __popc(mask & ((1u << lane) - 1));
            if (ok)
#pragma unroll
                for (int s = 0; s < GRP; ++s) dst[n + before * GRP + s] = g * GRP + s;
            n += __popc(mask) * GRP;
        }
    }
    for (int j = qg * GRP + lane; j <= p; j += 32) dst[n + j - qg * GRP] = j;
    if (lane == 0) sel_n[t] = n + p - qg * GRP + 1;
}

// ---- MoE -------------------------------------------------------------------------------------------

// One block (512 threads, one per expert) per token: scores = sigmoid(logits); the 16 experts of highest
// score + bias (ties: lower index), returned in ascending expert order with weights
// score / (sum + 1e-20) * scaling.
__global__ void __launch_bounds__(E) route_kernel(const float* __restrict__ logits, int ld, const float* __restrict__ bias,
                                                  float scaling, int* sel_e, float* sel_w) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float cv[E / 32];
    __shared__ int ci[E / 32];
    __shared__ int picked[TOPK];
    const int t = blockIdx.x, e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const float score = 1.0f / (1.0f + expf(-__ldcg(logits + (size_t)t * ld + e)));
    float choice = score + bias[e];
    bool taken = false;
    for (int k = 0; k < TOPK; ++k) {
        float v = taken ? -INFINITY : choice;
        int idx = e;
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, v, off);
            const int oi = __shfl_xor_sync(0xffffffff, idx, off);
            if (ov > v || (ov == v && oi < idx)) v = ov, idx = oi;
        }
        if (lane == 0) cv[warp] = v, ci[warp] = idx;
        __syncthreads();
        if (warp == 0) {
            v = lane < E / 32 ? cv[lane] : -INFINITY;
            idx = lane < E / 32 ? ci[lane] : E;
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffff, v, off);
                const int oi = __shfl_xor_sync(0xffffffff, idx, off);
                if (ov > v || (ov == v && oi < idx)) v = ov, idx = oi;
            }
            if (lane == 0) picked[k] = idx;
        }
        __syncthreads();
        if (e == picked[k]) taken = true;
    }
    // Ascending order: rank among taken experts.
    const unsigned m = __ballot_sync(0xffffffff, taken);
    __shared__ int wcount[E / 32];
    __shared__ float ws[TOPK];
    __shared__ int es[TOPK];
    if (lane == 0) wcount[warp] = __popc(m);
    __syncthreads();
    if (taken) {
        int rank = __popc(m & ((1u << lane) - 1));
        for (int w = 0; w < warp; ++w) rank += wcount[w];
        es[rank] = e;
        ws[rank] = score;
    }
    __syncthreads();
    if (e < TOPK) {
        float s = 0.0f;
        for (int k = 0; k < TOPK; ++k) s += ws[k];
        sel_e[t * TOPK + e] = es[e];
        sel_w[t * TOPK + e] = ws[e] / (s + 1e-20f) * scaling;
    }
}

// NVFP4 weight tables of one projection kind: codes U8 [512][1536][768], scales E4M3 [512][1536][96],
// FP32 scale per expert.
struct Fp4Experts {
    const uint8_t* codes;
    const uint8_t* scales;
    const float* s2;
};

// Small-batch experts. A row of 1536 NVFP4 values is 96 chunks of 8 code bytes (16 values) and 1 scale
// byte; lane l takes chunks l, l + 32 and l + 64. Each warp loads the raw chunks of all its rows before
// decoding any, so 3 * ROWS (gate/up: 6 * ROWS) loads per lane are in flight.
constexpr int EX_ROWS = 4, EX_CPL = 3;

SQ_DEVICE float fp4_dot16(uint2 codes, uint8_t scale, const float* x) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&codes);
    float part = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2(b[i], __NV_E2M1);
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
        part = fmaf(f.x, x[2 * i], part);
        part = fmaf(f.y, x[2 * i + 1], part);
    }
    return part * e4m3f(scale);
}

template <int R>
struct Fp4Rows {
    uint2 c[R][EX_CPL];
    uint8_t sc[R][EX_CPL];
};

template <int R>
SQ_DEVICE void fp4_rows_load(const uint8_t* codes, const uint8_t* scales, size_t row0, int lane, Fp4Rows<R>& w) {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < EX_CPL; ++j) {
            const int chunk = lane + 32 * j;
            w.c[r][j] = ld_stream_u2(codes + (row0 + r) * (LAT / 2) + chunk * 8);
            w.sc[r][j] = scales[(row0 + r) * (LAT / 16) + chunk];
        }
}

template <int R>
SQ_DEVICE void fp4_rows_dot(const Fp4Rows<R>& w, const float* xs, int lane, float (&out)[R]) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < EX_CPL; ++j) acc += fp4_dot16(w.c[r][j], w.sc[r][j], xs + (lane + 32 * j) * 16);
        out[r] = acc;
    }
}

// The lane's 48 activations of a row (chunks lane, lane + 32, lane + 64 of 16 values), BF16 -> F32.
SQ_DEVICE void lane_x(const bf16* x, int lane, float* xs) {
#pragma unroll
    for (int j = 0; j < EX_CPL; ++j) {
        const uint4* src = reinterpret_cast<const uint4*>(x + (lane + 32 * j) * 16);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 v = src[h];
            const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __bfloat1622float2(p2[i]);
                xs[j * 16 + h * 8 + 2 * i] = f.x;
                xs[j * 16 + h * 8 + 2 * i + 1] = f.y;
            }
        }
    }
}

template <int R>
SQ_DEVICE void fp4_rows_dot_lane(const Fp4Rows<R>& w, const float* xl, float (&out)[R]) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < EX_CPL; ++j) acc += fp4_dot16(w.c[r][j], w.sc[r][j], xl + j * 16);
        out[r] = acc;
    }
}

// Order of a small step's pairs by expert (ties by pair): order[rank] = pair. One block with a thread per
// pair (at most 1024).
__global__ void pair_order_kernel(const int* __restrict__ sel_e, int pairs, int* order) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ int es[1024];
    for (int q = threadIdx.x; q < pairs; q += blockDim.x) es[q] = __ldcg(sel_e + q);
    __syncthreads();
    const int p = threadIdx.x;
    if (p >= pairs) return;
    const int e = es[p];
    int rank = 0;
    for (int q = 0; q < pairs; ++q) {
        const int f = es[q];
        rank += f < e || (f == e && q < p);
    }
    order[rank] = p;
}

// Warp-granular persistent kernels: work item = (4 rows, pair in expert order); each warp strides over
// items, so the last wave is one item, not one block. Items of the pairs of one expert are adjacent, so
// their weight rows are read from DRAM once and from L2 by the others. Gate/up: x routed input BF16 [T, 1536]; act[pair, n] = swiglu
// with the experts' clamps.
// Gate/up triggers its dependents only after its dependency wait, so experts_down (which runs after it) may
// read the routing and load its first item's weights before its own wait.
__global__ void __launch_bounds__(256) experts_gate_up_kernel(const bf16* __restrict__ x, const int* sel_e, const int* order,
                                                             Fp4Experts gate, Fp4Experts up, float limit, int pairs, bf16* act) {
    pdl_wait();
    pdl_launch_dependents();
    const int lane = threadIdx.x & 31;
    const int warps = gridDim.x * 8, items = pairs * (LAT / EX_ROWS);
    for (int item = blockIdx.x * 8 + (threadIdx.x >> 5); item < items; item += warps) {
        const int pair = __ldcg(order + item % pairs), n0 = item / pairs * EX_ROWS;
        const int t = pair / TOPK, e = __ldcg(sel_e + pair);
        const size_t row0 = (size_t)e * LAT + n0;
        Fp4Rows<EX_ROWS> wg, wu;
        fp4_rows_load(gate.codes, gate.scales, row0, lane, wg);
        fp4_rows_load(up.codes, up.scales, row0, lane, wu);
        float xl[EX_CPL * 16];
        lane_x(x + (size_t)t * LAT, lane, xl);
        float g[EX_ROWS], u[EX_ROWS];
        fp4_rows_dot_lane(wg, xl, g);
        fp4_rows_dot_lane(wu, xl, u);
        const float sg = gate.s2[e], su = up.s2[e];
#pragma unroll
        for (int r = 0; r < EX_ROWS; ++r) {
            const float gs = warp_sum(g[r]), us = warp_sum(u[r]);
            if (lane == 0) {
                const float gv = fminf(rb(gs * sg), limit), uv = fminf(fmaxf(rb(us * su), -limit), limit);
                act[(size_t)pair * LAT + n0 + r] = f2bf(rb(silu(gv)) * uv);
            }
        }
    }
}

// Down: y[pair, n] = s2[e] * W_e[n] . act[pair] (F32 [pairs, 1536]); items of 8 rows.
__global__ void __launch_bounds__(256) experts_down_kernel(const bf16* act, const int* sel_e, const int* order, Fp4Experts down, int pairs,
                                                           float* y) {
    pdl_launch_dependents();
    constexpr int R = 2 * EX_ROWS;
    const int lane = threadIdx.x & 31;
    const int warps = gridDim.x * 8, items = pairs * (LAT / R);
    bool first = true;
    for (int item = blockIdx.x * 8 + (threadIdx.x >> 5); item < items; item += warps) {
        const int pair = __ldcg(order + item % pairs), n0 = item / pairs * R;
        const int e = __ldcg(sel_e + pair);
        Fp4Rows<R> wd;
        fp4_rows_load(down.codes, down.scales, (size_t)e * LAT + n0, lane, wd);
        if (first) {
            pdl_wait();
            first = false;
        }
        float xl[EX_CPL * 16];
        lane_x(act + (size_t)pair * LAT, lane, xl);
        float v[R];
        fp4_rows_dot_lane(wd, xl, v);
        const float sd = down.s2[e];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float vs = warp_sum(v[r]);
            if (lane == 0) y[(size_t)pair * LAT + n0 + r] = vs * sd;
        }
    }
    if (first) pdl_wait();
}

// Blocks of a persistent launch: the resident capacity (blocks per SM x SMs), at most `needed`.
template <typename K>
int resident_blocks(K kernel, int needed) {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    int per_sm = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, 256, 0);
    return std::max(1, std::min(needed, per_sm * sms));
}

// out[t, n] = BF16 running sum over the token's experts in ascending order of bf16(bf16(y) * w), as the
// transformers loop accumulates; y rows at pair_row[pair] (or pair itself when pair_row is null).
__global__ void experts_combine_kernel(const float* __restrict__ y, const int* __restrict__ pair_row, const float* __restrict__ sel_w,
                                       int T, bf16* out) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * LAT) return;
    const int t = (int)(i / LAT), n = (int)(i % LAT);
    float acc = 0.0f;
    for (int k = 0; k < TOPK; ++k) {
        const int pair = t * TOPK + k;
        const int row = pair_row ? __ldcg(pair_row + pair) : pair;
        acc = rb(acc + rb(rb(__ldcg(y + (size_t)row * LAT + n)) * __ldcg(sel_w + pair)));
    }
    out[i] = f2bf(acc);
}

// Grouping for the prefill GEMMs: counts per expert, tile plan (64-row tiles), and each pair's row.
__global__ void moe_count_kernel(const int* __restrict__ sel_e, int pairs, int* counts) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) atomicAdd(&counts[__ldcg(sel_e + i)], 1);
}

__global__ void __launch_bounds__(E) moe_plan_kernel(const int* __restrict__ counts, int bm, int* offsets, int* cursor,
                                                     int* tile_expert, int* tile_row, int* num_tiles) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ int off[E + 1];
    __shared__ int toff[E + 1];
    if (threadIdx.x == 0) {
        int o = 0, to = 0;
        for (int e = 0; e < E; ++e) {
            off[e] = o;
            toff[e] = to;
            const int c = __ldcg(counts + e);
            o += c;
            to += (c + bm - 1) / bm;
        }
        off[E] = o;
        toff[E] = to;
        *num_tiles = to;
    }
    __syncthreads();
    const int e = threadIdx.x;
    offsets[e] = off[e];
    cursor[e] = off[e];
    if (e == 0) offsets[E] = off[E];
    for (int j = 0; j < toff[e + 1] - toff[e]; ++j) {
        tile_expert[toff[e] + j] = e;
        tile_row[toff[e] + j] = off[e] + j * bm;
    }
}

__global__ void moe_scatter_kernel(const int* __restrict__ sel_e, int pairs, int* cursor, int* pair_row) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) pair_row[i] = atomicAdd(&cursor[__ldcg(sel_e + i)], 1);
}

// xs[pair_row[pair]] = x[pair / 16] (BF16 rows of 1536).
__global__ void moe_gather_kernel(const bf16* __restrict__ x, const int* __restrict__ pair_row, bf16* xs) {
    pdl_launch_dependents();
    pdl_wait();
    const int pair = blockIdx.x;
    const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)(pair / TOPK) * LAT);
    uint4* dst = reinterpret_cast<uint4*>(xs + (size_t)__ldcg(pair_row + pair) * LAT);
    for (int i = threadIdx.x; i < LAT / 8; i += blockDim.x) dst[i] = src[i];
}

// ---- Engram ----------------------------------------------------------------------------------------

// One warp per (token, head): the FP8 row (q8 * scale) as BF16, the FP8 round trip the model was trained
// with, then nn.RMSNorm (BF16 epsilon) with the head's weight. Output e BF16 [T, 1536].
__global__ void engram_embed_kernel(const uint8_t* __restrict__ q8, const float* __restrict__ sc, const bf16* __restrict__ head_norm,
                                    int T, bf16* e) {
    pdl_launch_dependents();
    pdl_wait();
    const int item = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (item >= T * EH) return;
    const int h = item % EH;
    const uint8_t* row = q8 + (size_t)item * EDIM;
    const float s = sc[item];
    float v[3], amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        v[j] = rb(e4m3f(row[lane + 32 * j]) * s);
        amax = fmaxf(amax, fabsf(v[j]));
    }
    amax = warp_max(amax);
    const float scale = rb(fmaxf(amax, 1e-12f) / 448.0f);
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        float y = rb(v[j] / scale);
        y = fminf(fmaxf(y, -448.0f), 448.0f);
        __nv_fp8_e4m3 f(y);
        v[j] = rb(float(f) * scale);
        ss += v[j] * v[j];
    }
    const float r = rsqrtf(warp_sum(ss) / (float)EDIM + TORCH_BF16_EPS);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int d = lane + 32 * j;
        e[(size_t)item * EDIM + d] = f2bf(v[j] * r * bf2f(head_norm[h * EDIM + d]));
    }
}

// One block (256 threads) per token. kv: [T, 5 * 3072] F32 from e . [key_0 .. key_3, value]^T, plus
// bias [5 * 3072]. For lane c: key = rms(bf16(kv_c + b), norm_key_c) (BF16), query = rms(lane_c, norm_query_c)
// (FP32, as the lanes), gate = sigmoid(signed sqrt(sum key * query * 3072^-0.5)); main = gamma *
// rms(bf16(value + b)); lane_c += gate * main + bf16(scale_c * main). Positions with mask 0 are left unchanged.
__global__ void __launch_bounds__(256) engram_apply_kernel(const float* __restrict__ kv, int ldkv, const bf16* __restrict__ bias,
                                                          const bf16* __restrict__ norm_key, const bf16* __restrict__ norm_query,
                                                          const bf16* __restrict__ gamma, const bf16* __restrict__ hc_scale,
                                                          const uint8_t* __restrict__ mask, float* lanes) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    __shared__ float main_s[D];
    __shared__ float gates[HC];
    const int t = blockIdx.x, tid = threadIdx.x;
    if (!mask[t]) return;
    const float* row = kv + (size_t)t * ldkv;
    float* ln = lanes + (size_t)t * HC * D;
    constexpr int PER = D / 256;
    // value -> main
    {
        float v[PER], ss = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int d = tid + 256 * j;
            v[j] = rb(row[HC * D + d] + bf2f(bias[HC * D + d]));
            ss += v[j] * v[j];
        }
        const float r = rsqrtf(block_sum(ss, scratch) / (float)D + TORCH_BF16_EPS);
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int d = tid + 256 * j;
            main_s[d] = rb(bf2f(gamma[d]) * rb(v[j] * r));
        }
    }
    for (int c = 0; c < HC; ++c) {
        float k[PER], q[PER], sk = 0.0f, sq = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int d = tid + 256 * j;
            k[j] = rb(row[c * D + d] + bf2f(bias[c * D + d]));
            q[j] = ln[c * D + d];
            sk += k[j] * k[j];
            sq += q[j] * q[j];
        }
        const float rk = rsqrtf(block_sum(sk, scratch) / (float)D + TORCH_BF16_EPS);
        const float rq = rsqrtf(block_sum(sq, scratch) / (float)D + TORCH_BF16_EPS);
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int d = tid + 256 * j;
            const float kk = rb(k[j] * rk * bf2f(norm_key[c * D + d]));
            const float qq = q[j] * rq * bf2f(norm_query[c * D + d]);
            dot += kk * qq;
        }
        dot = block_sum(dot, scratch);
        if (tid == 0) {
            float g = dot * 0.018042195912175804f;
            const float mag = sqrtf(fmaxf(fabsf(g), 1e-6f));
            g = g > 0 ? mag : (g < 0 ? -mag : 0.0f);
            gates[c] = 1.0f / (1.0f + expf(-g));
        }
        __syncthreads();
    }
    for (int c = 0; c < HC; ++c)
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int d = tid + 256 * j;
            const float m = main_s[d];
            ln[c * D + d] += gates[c] * m + rb(bf2f(hc_scale[c * D + d]) * m);
        }
}

}  // namespace

// ---- Launchers -------------------------------------------------------------------------------------

extern "C" {

size_t quench6k_hy4_hc_partial_floats(int T) { return (size_t)T * HC_SLICES * (2 * HC + 1); }

// partial: quench6k_hy4_hc_partial_floats(T) floats of scratch.
cudaError_t quench6k_hy4_hc_pre(const void* lanes, const float* fn, const float* base, const float* scale, const void* norm_w, float hc_eps,
                              float mag, float eps, int head, int T, float* partial, void* out, float* post, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    auto* o = static_cast<bf16*>(out);
    const auto* l = static_cast<const float*>(lanes);
    const auto* w = static_cast<const bf16*>(norm_w);
    const dim3 grid(HC_SLICES, T);
    if (head) {
        SQ_CHECK_LAUNCH(sq_launch(hc_mix_partial_kernel<HC>, grid, 256, 0, st, l, fn, partial));
        SQ_CHECK_LAUNCH(sq_launch(hc_pre_kernel<HC>, T, 512, 0, st, l, (const float*)partial, base, scale, w, hc_eps, mag, eps, o, post));
    } else {
        SQ_CHECK_LAUNCH(sq_launch(hc_mix_partial_kernel<2 * HC>, grid, 256, 0, st, l, fn, partial));
        SQ_CHECK_LAUNCH(sq_launch(hc_pre_kernel<2 * HC>, T, 512, 0, st, l, (const float*)partial, base, scale, w, hc_eps, mag, eps, o, post));
    }
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_hc_post(void* lanes, const float* post, const void* y, int ldy, int T, cudaStream_t st) {
    const size_t n = (size_t)T * HC * D;
    if (n == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(hc_post_kernel, (unsigned)((n + 255) / 256), 256, 0, st, static_cast<float*>(lanes), post,
                              static_cast<const bf16*>(y), ldy, T));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_hc_expand(const void* emb, void* lanes, int T, cudaStream_t st) {
    const size_t n = (size_t)T * HC * D;
    if (n == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(hc_expand_kernel, (unsigned)((n + 255) / 256), 256, 0, st, static_cast<const bf16*>(emb),
                              static_cast<float*>(lanes), T));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_rms_rows(const float* in, int ldi, const void* w, int n, float eps, void* out, int ldo, int rows,
                                cudaStream_t st) {
    if (n > 4096) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(rms_rows_kernel, rows, 256, 0, st, in, ldi, static_cast<const bf16*>(w), n, eps, static_cast<bf16*>(out), ldo));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_swiglu(const float* g, const float* u, int ld, int width, float limit, void* act, int lda, int rows,
                              cudaStream_t st) {
    const size_t n = (size_t)rows * width;
    if (n == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(swiglu_kernel, (unsigned)((n + 255) / 256), 256, 0, st, g, u, ld, width, limit, static_cast<bf16*>(act), lda,
                              rows));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_add_rows(const float* a, int lda, const float* b, int ldb, int n, int rows, float* out, int ldo, cudaStream_t st) {
    const size_t c = (size_t)rows * n;
    if (c == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(add_rows_kernel, (unsigned)((c + 255) / 256), 256, 0, st, a, lda, b, ldb, n, rows, out, ldo));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_mla_prep(const float* q, int ldq, const float* ckv, int ldc, const void* kv_norm, float eps, float theta, int H,
                                const int* pos, const int* seq, SeqRows cache, int T, void* qn, void* qa, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(mla_prep_kernel, dim3(T, H + 1), 64, 0, st, q, ldq, ckv, ldc, static_cast<const bf16*>(kv_norm), eps, theta, H,
                              pos, seq, cache, static_cast<bf16*>(qn), static_cast<bf16*>(qa)));
    SQ_RETURN_LAUNCH();
}

int quench6k_hy4_heads_gemv_max_t() { return 16; }

cudaError_t quench6k_hy4_heads_gemv(const void* x, int ldx, int xh, const void* W, size_t wh, int N, int K, int H, int T, const float* gate,
                                  int ldg, void* y, int ldy, int yh, cudaStream_t st) {
    if (N % 32 != 0 || K % 8 != 0) return cudaErrorInvalidValue;
    if (T == 0) return cudaSuccess;
    // More than 16 rows: blocks of 16 (grid z).
    const dim3 grid(H, N / 32, (T + 15) / 16);
    const auto* xp = static_cast<const bf16*>(x);
    const auto* wp = static_cast<const bf16*>(W);
    auto* yp = static_cast<bf16*>(y);
    if (T <= 1)
        SQ_CHECK_LAUNCH(sq_launch(heads_gemv_kernel<1>, grid, 256, 0, st, xp, ldx, xh, wp, wh, N, K, T, gate, ldg, yp, ldy, yh));
    else if (T <= 4)
        SQ_CHECK_LAUNCH(sq_launch(heads_gemv_kernel<4>, grid, 256, 0, st, xp, ldx, xh, wp, wh, N, K, T, gate, ldg, yp, ldy, yh));
    else if (T <= 8)
        SQ_CHECK_LAUNCH(sq_launch(heads_gemv_kernel<8>, grid, 256, 0, st, xp, ldx, xh, wp, wh, N, K, T, gate, ldg, yp, ldy, yh));
    else
        SQ_CHECK_LAUNCH(sq_launch(heads_gemv_kernel<16>, grid, 256, 0, st, xp, ldx, xh, wp, wh, N, K, T, gate, ldg, yp, ldy, yh));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_gate_mul(void* v, const float* gate, size_t count, cudaStream_t st) {
    if (count == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(gate_mul_kernel, (unsigned)((count + 255) / 256), 256, 0, st, static_cast<bf16*>(v), gate, count));
    SQ_RETURN_LAUNCH();
}

size_t quench6k_hy4_attn_part_floats(int T, int H, int splits) { return (size_t)splits * T * H * (KVL + 2); }

// mode 0 sliding, 1 dense, 2 selected. splits > 1 needs part (quench6k_hy4_attn_part_floats floats). Blocks
// take `heads_per_block` heads (16, 32 or 64; 0: all of a token's heads, up to 64) sharing each key tile.
}  // extern "C"

namespace {

template <int G>
cudaError_t launch_mla(int mode, dim3 grid, cudaStream_t st, const bf16* q, int H, SeqRows c, const int* sel, const int* sel_n,
                       const float* sinks, const int* pos, const int* seq, int T, bf16* o, float* part_o, float* part_ml) {
    static bool configured = false;
    if (!configured) {
        for (const void* k : {(const void*)mla_attn_kernel<0, G>, (const void*)mla_attn_kernel<1, G>, (const void*)mla_attn_kernel<2, G>}) {
            const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)at_smem(G));
            if (e != cudaSuccess) return e;
        }
        configured = true;
    }
#define SQ_AT(M) sq_launch(mla_attn_kernel<M, G>, grid, 128 * G, at_smem(G), st, q, H, c, sel, sel_n, sinks, pos, seq, T, o, part_o, part_ml)
    if (mode == 0) SQ_CHECK_LAUNCH(SQ_AT(0));
    else if (mode == 1) SQ_CHECK_LAUNCH(SQ_AT(1));
    else SQ_CHECK_LAUNCH(SQ_AT(2));
#undef SQ_AT
    return cudaSuccess;
}

}  // namespace

extern "C" {

cudaError_t quench6k_hy4_mla_attention(int mode, const void* qa, int H, SeqRows cache, const int* sel, const int* sel_n,
                                     const float* sinks, const int* pos, const int* seq, int T, int splits, int heads_per_block, float* part,
                                     void* out, cudaStream_t st) {
    if (heads_per_block == 0) heads_per_block = H < 64 ? H : 64;
    if (H % AT_HG != 0 || splits < 1 || H % heads_per_block != 0) return cudaErrorInvalidValue;
    if (T == 0) return cudaSuccess;
    const dim3 grid(T * (H / heads_per_block), splits);
    const auto* q = static_cast<const bf16*>(qa);
    const SeqRows c = cache;
    auto* o = static_cast<bf16*>(out);
    float* part_o = part;
    float* part_ml = part ? part + (size_t)splits * T * H * KVL : nullptr;
    cudaError_t e;
    switch (heads_per_block / AT_HG) {
        case 1: e = launch_mla<1>(mode, grid, st, q, H, c, sel, sel_n, sinks, pos, seq, T, o, part_o, part_ml); break;
        case 2: e = launch_mla<2>(mode, grid, st, q, H, c, sel, sel_n, sinks, pos, seq, T, o, part_o, part_ml); break;
        case 4: e = launch_mla<4>(mode, grid, st, q, H, c, sel, sel_n, sinks, pos, seq, T, o, part_o, part_ml); break;
        default: return cudaErrorInvalidValue;
    }
    if (e != cudaSuccess) return e;
    if (splits > 1)
        SQ_CHECK_LAUNCH(sq_launch(mla_merge_kernel, T * H, 128, 0, st, (const float*)part_o, (const float*)part_ml, splits, sinks, T, H, o));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_idx_prep(const float* tk, int ldk, const void* tk_norm, float eps, const float* iq, int ldq, const float* hw,
                                int ldw, float theta, const int* pos, const int* seq, int T, SeqRows tk_cache, void* iq_out, float* hw_out,
                                cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(idx_prep_kernel, T, 128, 0, st, tk, ldk, static_cast<const bf16*>(tk_norm), eps, iq, ldq, hw, ldw, theta, pos,
                              seq, tk_cache, static_cast<bf16*>(iq_out), hw_out));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_idx_pool(SeqRows tk_cache, const float* wv, const float* wg, const float* ape, const float* key_norm, float eps,
                                float theta, const int* pos, const int* seq, int T, SeqRows pooled, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(idx_pool_kernel, T, 128, 0, st, tk_cache, wv, wg, ape, key_norm, eps, theta, pos, seq, pooled));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_idx_scores(const void* iq, const float* hw, SeqRows pooled, const int* pos, const int* seq, int T, int G,
                                  float* scores, int ld, cudaStream_t st) {
    if (T == 0 || G == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(idx_scores_kernel, dim3((G + 63) / 64, T), 256, 0, st, static_cast<const bf16*>(iq), hw, pooled, pos, seq, G,
                              scores, ld));
    SQ_RETURN_LAUNCH();
}

int quench6k_hy4_sel_width() { return SEL; }

cudaError_t quench6k_hy4_idx_expand(const int* picked, int npick, const int* pos, int T, int* sel, int* sel_n, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(idx_expand_kernel, (T + 3) / 4, 128, 0, st, picked, npick, pos, T, sel, sel_n));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_route(const float* logits, int ld, const float* bias, float scaling, int T, int* sel_e, float* sel_w, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(route_kernel, T, E, 0, st, logits, ld, bias, scaling, sel_e, sel_w));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_experts_small(const void* x, const int* sel_e, const uint8_t* gc, const uint8_t* gs, const float* g2,
                                     const uint8_t* uc, const uint8_t* us, const float* u2, const uint8_t* dc, const uint8_t* ds,
                                     const float* d2, float limit, int T, int* order, void* act, float* y, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    const int pairs = T * TOPK;
    if (pairs > 1024) return cudaErrorInvalidValue;  // pair_order_kernel's block
    SQ_CHECK_LAUNCH(sq_launch(pair_order_kernel, 1, std::max(128, (pairs + 127) / 128 * 128), 0, st, sel_e, pairs, order));
    const int gu = resident_blocks(experts_gate_up_kernel, (pairs * (LAT / EX_ROWS) + 7) / 8);
    const int dn = resident_blocks(experts_down_kernel, (pairs * (LAT / (2 * EX_ROWS)) + 7) / 8);
    SQ_CHECK_LAUNCH(sq_launch(experts_gate_up_kernel, gu, 256, 0, st, static_cast<const bf16*>(x), sel_e, (const int*)order, Fp4Experts{gc, gs, g2},
                              Fp4Experts{uc, us, u2}, limit, pairs, static_cast<bf16*>(act)));
    SQ_CHECK_LAUNCH(sq_launch(experts_down_kernel, dn, 256, 0, st, static_cast<const bf16*>(act), sel_e, (const int*)order, Fp4Experts{dc, ds, d2}, pairs, y));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_quantize_fp8b(const void* w, int N, int K, uint8_t* q, float* scale, cudaStream_t st) {
    if (K % 128 != 0) return cudaErrorInvalidValue;
    quantize_fp8b_kernel<<<dim3(K / 128, (N + 127) / 128), 256, 0, st>>>(static_cast<const bf16*>(w), N, K, q, scale);
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_to_bf16(const float* x, void* y, size_t count, cudaStream_t st) {
    if (count == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(to_bf16_late_kernel, (unsigned)((count + 255) / 256), 256, 0, st, x, static_cast<bf16*>(y), count));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_experts_combine(const float* y, const int* pair_row, const float* sel_w, int T, void* out, cudaStream_t st) {
    const size_t n = (size_t)T * LAT;
    if (n == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(experts_combine_kernel, (unsigned)((n + 255) / 256), 256, 0, st, y, pair_row, sel_w, T, static_cast<bf16*>(out)));
    SQ_RETURN_LAUNCH();
}

// counts must hold E ints and be zeroed by the caller; tile arrays hold at least pairs / bm + E entries.
cudaError_t quench6k_hy4_moe_plan(const int* sel_e, int T, int bm, int* counts, int* offsets, int* cursor, int* tile_expert, int* tile_row,
                                int* num_tiles, int* pair_row, cudaStream_t st) {
    const int pairs = T * TOPK;
    if (pairs == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(moe_count_kernel, (pairs + 255) / 256, 256, 0, st, sel_e, pairs, counts));
    SQ_CHECK_LAUNCH(sq_launch(moe_plan_kernel, 1, E, 0, st, (const int*)counts, bm, offsets, cursor, tile_expert, tile_row, num_tiles));
    SQ_CHECK_LAUNCH(sq_launch(moe_scatter_kernel, (pairs + 255) / 256, 256, 0, st, sel_e, pairs, cursor, pair_row));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_moe_gather(const void* x, const int* pair_row, int T, void* xs, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(moe_gather_kernel, T * TOPK, 128, 0, st, static_cast<const bf16*>(x), pair_row, static_cast<bf16*>(xs)));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_engram_embed(const uint8_t* q8, const float* sc, const void* head_norm, int T, void* e, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(engram_embed_kernel, (T * EH + 7) / 8, 256, 0, st, q8, sc, static_cast<const bf16*>(head_norm), T,
                              static_cast<bf16*>(e)));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_engram_apply(const float* kv, int ldkv, const void* bias, const void* norm_key, const void* norm_query,
                                    const void* gamma, const void* hc_scale, const uint8_t* mask, int T, void* lanes, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(engram_apply_kernel, T, 256, 0, st, kv, ldkv, static_cast<const bf16*>(bias),
                              static_cast<const bf16*>(norm_key), static_cast<const bf16*>(norm_query), static_cast<const bf16*>(gamma),
                              static_cast<const bf16*>(hc_scale), mask, static_cast<float*>(lanes)));
    SQ_RETURN_LAUNCH();
}

}  // extern "C"

// ---- DFly drafter ----------------------------------------------------------------------------------

// out[r, :D] = bf16(w * bf16(bf16(x) * rsqrt(mean + eps))) for BF16 rows x = src[(idx ? idx[r] : r) * lds] (embedding or
// hidden rows picked by a device index); with src_tab, row r's src is src_tab[r] instead.
__global__ void __launch_bounds__(256) dfly_norm_kernel(const bf16* src, const unsigned long long* src_tab, size_t lds, const int* idx,
                                                       const bf16* __restrict__ w, float eps, bf16* out, int ldo) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    __shared__ int row;
    const int r = blockIdx.x;
    // idx may have been written by the kernel just before (PDL): read it through L2, once per block.
    if (threadIdx.x == 0) row = idx ? __ldcg(idx + r) : r;
    __syncthreads();
    const bf16* base = src_tab ? reinterpret_cast<const bf16*>(src_tab[r]) : src;
    const bf16* x = base + (size_t)row * lds;
    float v[D / 256], ss = 0.0f;
#pragma unroll
    for (int c = 0; c < D / 256; ++c) {
        v[c] = bf2f(x[threadIdx.x + 256 * c]);
        ss += v[c] * v[c];
    }
    const float rs = rsqrtf(block_sum(ss, scratch) / (float)D + eps);
#pragma unroll
    for (int c = 0; c < D / 256; ++c) {
        const int i = threadIdx.x + 256 * c;
        out[(size_t)r * ldo + i] = f2bf(bf2f(w[i]) * rb(v[c] * rs));
    }
}

// x[r, :D] (F32) = embedding rows table[idx[r]].
__global__ void dfly_embed_kernel(const bf16* __restrict__ table, const int* __restrict__ idx, float* x) {
    pdl_launch_dependents();
    pdl_wait();
    const int r = blockIdx.x;
    const bf16* row = table + (size_t)idx[r] * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) x[(size_t)r * D + i] = bf2f(row[i]);
}

// x[r * xs, :D] += (gate ? bf16(gate * y) : y) with y row r * ys of BF16 (yb) or F32 (yf); out (optional) row r = bf16(x).
__global__ void dfly_add_kernel(float* x, int xs, const bf16* __restrict__ yb, const float* __restrict__ yf, int ys,
                                const bf16* __restrict__ gate, bf16* out, int rows) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * D) return;
    const int r = i / D, c = i % D;
    const size_t xi = (size_t)r * xs * D + c, yi = (size_t)r * ys * D + c;
    float y = yb ? bf2f(yb[yi]) : yf[yi];
    if (gate) y = rb(bf2f(gate[c]) * rb(y));
    const float v = x[xi] + y;
    x[xi] = v;
    if (out) out[i] = f2bf(v);
}

// Key lists of the drafter's query blocks (block s: rows s * W + j, j < W, at positions anchor[s] + j): the
// target's list for its source row (keep[krow[s]] of sequence s's key lists at keep_tab[s] + keep_off, lengths
// at + keep_n_off, in ints) then the block's own positions anchor .. anchor + j; while the source row's own group
// is among the first 512 (dense attention), every position 0 .. anchor + j.
__global__ void dfly_sel_kernel(const unsigned long long* keep_tab, long long keep_off, long long keep_n_off, const int* anchors,
                                const int* krows, int W, int* sel, int* sel_n) {
    pdl_launch_dependents();
    pdl_wait();
    const int s = blockIdx.x / W, j = blockIdx.x % W;
    const int anchor = anchors[s], krow = krows[s];
    const int* keep = reinterpret_cast<const int*>(keep_tab[s]) + keep_off;
    const int* keep_n = reinterpret_cast<const int*>(keep_tab[s]) + keep_n_off;
    int* dst = sel + (size_t)blockIdx.x * SEL;
    if ((anchor - 1) / GRP <= TOPG - 1) {
        for (int i = threadIdx.x; i <= anchor + j; i += blockDim.x) dst[i] = i;
        if (threadIdx.x == 0) sel_n[blockIdx.x] = anchor + j + 1;
        return;
    }
    const int n = __ldcg(keep_n + krow);
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = __ldcg(keep + (size_t)krow * SEL + i);
    for (int i = threadIdx.x; i <= j; i += blockDim.x) dst[n + i] = anchor + i;
    if (threadIdx.x == 0) sel_n[blockIdx.x] = n + j + 1;
}

// Copies rows of a step into per-sequence buffers: row t with slot[t] >= 0 goes to seq_tab[seq[t]] + off +
// slot[t] * ld (bytes): `bytes` bytes of src + t * lds, or 4 * count[t] bytes when count is given.
__global__ void seq_scatter_kernel(const uint8_t* src, size_t lds, int bytes, const int* count, const int* slot, const int* seq,
                                   const unsigned long long* seq_tab, long long off, long long ld) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x, k = slot[t];
    if (k < 0) return;
    const int n = count ? 4 * __ldcg(count + t) : bytes;
    const uint8_t* from = src + (size_t)t * lds;
    uint8_t* to = reinterpret_cast<uint8_t*>(seq_tab[seq[t]]) + off + (size_t)k * ld;
    for (int i = threadIdx.x * 4; i < n; i += blockDim.x * 4) *reinterpret_cast<uint32_t*>(to + i) = __ldcg(reinterpret_cast<const unsigned*>(from + i));
}

// NVFP4 [N, K] (codes [N, K / 2], E4M3 scales [N, K / 16], global scale s2) -> BF16 [N, K].
__global__ void nvfp4_to_bf16_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, float s2, size_t pairs,
                                     int K, bf16* out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    const size_t e = 2 * i, row = e / K, col = e % K;
    const float sc = e4m3f(scales[row * (K / 16) + col / 16]) * s2;
    const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2(codes[i], __NV_E2M1);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
    out[e] = f2bf(f.x * sc);
    out[e + 1] = f2bf(f.y * sc);
}

extern "C" {

cudaError_t quench6k_hy4_dfly_norm(const void* src, const unsigned long long* src_tab, size_t lds, const int* idx, const void* w, float eps,
                                   void* out, int ldo, int rows, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(dfly_norm_kernel, rows, 256, 0, st, static_cast<const bf16*>(src), src_tab, lds, idx, static_cast<const bf16*>(w),
                              eps, static_cast<bf16*>(out), ldo));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_dfly_embed(const void* table, const int* idx, int rows, float* x, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(dfly_embed_kernel, rows, 256, 0, st, static_cast<const bf16*>(table), idx, x));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_dfly_add(float* x, int xs, const void* yb, const float* yf, int ys, const void* gate, void* out, int rows,
                                  cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(dfly_add_kernel, (rows * D + 255) / 256, 256, 0, st, x, xs, static_cast<const bf16*>(yb), yf, ys,
                              static_cast<const bf16*>(gate), static_cast<bf16*>(out), rows));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_dfly_sel(const unsigned long long* keep_tab, long long keep_off, long long keep_n_off, const int* anchors,
                                  const int* krows, int S, int W, int* sel, int* sel_n, cudaStream_t st) {
    if (S == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(dfly_sel_kernel, S * W, 256, 0, st, keep_tab, keep_off, keep_n_off, anchors, krows, W, sel, sel_n));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_seq_scatter(const void* src, size_t lds, int bytes, const int* count, const int* slot, const int* seq, int T,
                                     const unsigned long long* seq_tab, long long off, long long ld, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    if (bytes % 4 != 0 || lds % 4 != 0 || off % 4 != 0 || ld % 4 != 0) return cudaErrorInvalidValue;
    SQ_CHECK_LAUNCH(sq_launch(seq_scatter_kernel, T, 256, 0, st, static_cast<const uint8_t*>(src), lds, bytes, count, slot, seq, seq_tab, off, ld));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_hy4_nvfp4_to_bf16(const uint8_t* codes, const uint8_t* scales, float s2, int N, int K, void* out, cudaStream_t st) {
    if (K % 16 != 0) return cudaErrorInvalidValue;
    const size_t pairs = (size_t)N * K / 2;
    nvfp4_to_bf16_kernel<<<(unsigned)((pairs + 255) / 256), 256, 0, st>>>(codes, scales, s2, pairs, K, static_cast<bf16*>(out));
    SQ_RETURN_LAUNCH();
}

}  // extern "C"
