// Inkling 2 Mini (inkling2_loop, thinkingmachines/Inkling-2-Mini): kernels of the text model that the shared
// GEMV/GEMM, top-k and LSE kernels do not cover.
//
// - Residual stream: per decoder layer a partial sum of the current block of 6 layers; each sublayer's input
//   is a softmax mixture (one learned query per input) over the RMS-normalized earlier block sums and the
//   partial ("block attention residual"). Layers 6..17 run three times; passes after the first restart from
//   the state before the loop and mix messages from the previous pass into their inputs (message gates).
//   The engine keeps the stream and the block sums in FP32 (the transformers model keeps them in BF16; the
//   model amplifies that rounding, so FP32 stays closer to an FP32 forward).
// - Attention: no rotary embedding. Queries are scaled and L2-clipped per head; keys and values come from a
//   linear projection, a causal depthwise convolution of width 4 (static taps; the full-attention layers add
//   per-token taps per head) and an L2 clip (keys) or RMS normalization (values). Scores get a relative
//   position bias (per token and head: 16 relative states times a [16, 1024] profile bank, by backward
//   distance < 1024) and the softmax a per-token sink logit. Sliding layers see the last 1024 positions.
//   The output is gated by a sigmoid of the layer input before the output projection.
// - Every sublayer output passes a causal depthwise convolution of width 4 (with a residual and a 1/sqrt(5)
//   scale) before it joins the stream.
// - MoE: sigmoid router with a selection bias, 8 of 128 experts (NVFP4 or MXFP8), softcapped SwiGLU.
//
// Activations are rounded to BF16 only where they are stored as BF16 (projection inputs, caches, rings);
// the transformers model rounds after every operation.

#include <algorithm>

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "mma.cuh"

namespace {

using bf16 = __nv_bfloat16;

constexpr int D = 2304, H = 24, KVH = 8, HD = 128, QD = H * HD, KVD = KVH * HD, GQ = H / KVH;
constexpr int DREL = 16, EXT = 1024, WIN = 1024, CONVK = 4;
constexpr int E = 128, TOPK = 8, FF = 1152;
constexpr float EPS = 1e-6f;
constexpr int MAX_SRC = 8, MAX_Q = 4;

SQ_DEVICE float rb(float x) { return bf2f(f2bf(x)); }

SQ_DEVICE float e4m3_value(uint8_t b) {
    __nv_fp8_e4m3 v;
    v.__x = b;
    return float(v);
}

SQ_DEVICE float ue8m0(uint8_t b) { return b == 0 ? 5.877471754e-39f : __uint_as_float((uint32_t)b << 23); }

// Sums the values v[i] with live(i) (block-uniform) over the block; every thread gets the sums. `red`
// holds 32 * N floats.
template <int N, typename Live>
SQ_DEVICE void block_sums(float (&v)[N], Live live, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
#pragma unroll
    for (int i = 0; i < N; ++i)
        if (live(i)) v[i] = warp_sum(v[i]);
    if (lane == 0)
#pragma unroll
        for (int i = 0; i < N; ++i)
            if (live(i)) red[i * 32 + warp] = v[i];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; ++i)
        if (live(i)) v[i] = warp_sum(lane < warps ? red[i * 32 + lane] : 0.0f);
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------------------
// Embedding: out[t] = rb(w * rmsnorm(table[token t])) (F32 [T, D]).
constexpr int RT = 256, RPT = D / RT;  // 9 values per thread

__global__ void __launch_bounds__(RT) embed_kernel(const int* __restrict__ tokens, const bf16* __restrict__ table, const bf16* __restrict__ w,
                                                    float* out) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float red[32];
    const int t = blockIdx.x;
    const bf16* row = table + (size_t)tokens[t] * D;
    float v[RPT], ss = 0.0f;
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
        v[i] = bf2f(row[threadIdx.x + RT * i]);
        ss += v[i] * v[i];
    }
    const float r = rsqrtf(block_sum(ss, red) / D + EPS);
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
        const int c = threadIdx.x + RT * i;
        out[(size_t)t * D + c] = bf2f(w[c]) * (v[i] * r);
    }
}

// RMSNorm of F32 rows to BF16, for up to MAX_SETS sets (blockIdx.y): out[t] = bf16(w * x * rsqrt(mean x^2 +
// eps)) (w null: unweighted); x rows of D (stride D), out rows at stride ldo.
constexpr int MAX_SETS = 8;
struct RmsSets {
    const float* x[MAX_SETS];
    const bf16* w[MAX_SETS];
    bf16* out[MAX_SETS];
    int ldo[MAX_SETS];
};

__global__ void __launch_bounds__(RT) rms_kernel(RmsSets sets) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float red[32];
    const int t = blockIdx.x, g = blockIdx.y;
    const float* x = sets.x[g];
    const bf16* w = sets.w[g];
    float v[RPT], ss = 0.0f;
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
        v[i] = x[(size_t)t * D + threadIdx.x + RT * i];
        ss += v[i] * v[i];
    }
    const float r = rsqrtf(block_sum(ss, red) / D + EPS);
    bf16* out = sets.out[g] + (size_t)t * sets.ldo[g];
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
        const int c = threadIdx.x + RT * i;
        out[c] = f2bf(w ? bf2f(w[c]) * (v[i] * r) : v[i] * r);
    }
}

// ---------------------------------------------------------------------------------------------------------
// Block attention residual: for each of nq queries, out[q][t] = sum_s softmax_s(Q_q . n_s / sqrt(D)) n_s,
// n_s = x_s * rsqrt(mean x_s^2 + eps), over the sources x_0 .. x_{ns-1} (F32 [T, D] rows; the last one is
// the partial sum). Outputs F32 [nq][T, D] at out + q * qstride.
struct Srcs {
    const float* p[MAX_SRC];
};

constexpr int ART = 768, ARPT = D / ART;  // 3 values per thread

// Optional RMSNorm outputs of the mixtures: o[q] (BF16 rows at stride ldo) = w[q] * rmsnorm(out q).
struct ResNorms {
    const bf16* w[MAX_Q];
    bf16* o[MAX_Q];
    int ldo;
};

__global__ void __launch_bounds__(ART) attn_res_kernel(Srcs src, int ns, const float* __restrict__ queries, int nq, float* out, size_t qstride,
                                                       ResNorms norms) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float red[32 * (MAX_SRC + MAX_SRC * MAX_Q)];
    const int t = blockIdx.x;
    float x[MAX_SRC][ARPT];
    float acc[MAX_SRC + MAX_SRC * MAX_Q];
#pragma unroll
    for (int i = 0; i < MAX_SRC + MAX_SRC * MAX_Q; ++i) acc[i] = 0.0f;
#pragma unroll
    for (int s = 0; s < MAX_SRC; ++s)
#pragma unroll
        for (int i = 0; i < ARPT; ++i) x[s][i] = s < ns ? src.p[s][(size_t)t * D + threadIdx.x + ART * i] : 0.0f;
#pragma unroll
    for (int i = 0; i < ARPT; ++i) {
        const int c = threadIdx.x + ART * i;
        float qv[MAX_Q];
#pragma unroll
        for (int q = 0; q < MAX_Q; ++q) qv[q] = q < nq ? queries[q * D + c] : 0.0f;
#pragma unroll
        for (int s = 0; s < MAX_SRC; ++s) {
            acc[s] += x[s][i] * x[s][i];
#pragma unroll
            for (int q = 0; q < MAX_Q; ++q) acc[MAX_SRC + s * MAX_Q + q] += qv[q] * x[s][i];
        }
    }
    block_sums(acc, [&](int i) { return i < MAX_SRC ? i < ns : ((i - MAX_SRC) / MAX_Q < ns && (i - MAX_SRC) % MAX_Q < nq); }, red);
    float r[MAX_SRC];
#pragma unroll
    for (int s = 0; s < MAX_SRC; ++s) r[s] = s < ns ? rsqrtf(acc[s] / D + EPS) : 0.0f;
    const float scale = rsqrtf((float)D);
    float o[MAX_Q][ARPT], ss[MAX_Q];
#pragma unroll
    for (int q = 0; q < MAX_Q; ++q) {
        ss[q] = 0.0f;
        float l[MAX_SRC], m = -INFINITY;
#pragma unroll
        for (int s = 0; s < MAX_SRC; ++s) {
            l[s] = s < ns && q < nq ? acc[MAX_SRC + s * MAX_Q + q] * r[s] * scale : -INFINITY;
            m = fmaxf(m, l[s]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < MAX_SRC; ++s) {
            l[s] = s < ns && q < nq ? __expf(l[s] - m) : 0.0f;
            sum += l[s];
        }
        const float inv = q < nq ? 1.0f / sum : 0.0f;
#pragma unroll
        for (int i = 0; i < ARPT; ++i) {
            float v = 0.0f;
#pragma unroll
            for (int s = 0; s < MAX_SRC; ++s)
                if (s < ns) v += l[s] * inv * (x[s][i] * r[s]);
            o[q][i] = v;
            ss[q] += v * v;
            if (out && q < nq) out[q * qstride + (size_t)t * D + threadIdx.x + ART * i] = v;
        }
    }
    block_sums(ss, [&](int q) { return q < nq && norms.o[q] != nullptr; }, red);
#pragma unroll
    for (int q = 0; q < MAX_Q; ++q) {
        if (q >= nq || !norms.o[q]) continue;
        const float rn = rsqrtf(ss[q] / D + EPS);
#pragma unroll
        for (int i = 0; i < ARPT; ++i) {
            const int c = threadIdx.x + ART * i;
            norms.o[q][(size_t)t * norms.ldo + c] = f2bf(bf2f(norms.w[q][c]) * (o[q][i] * rn));
        }
    }
}

// Message gates (up to MAX_Q, blockIdx of the group): out_g = (1 - tanh(F_g z)) hs_g + tanh(G_g z) m_g with
// z = bf16(down_g [rms(hs_g), rms(m_g)]). Step 1 (rms_kernel) writes cat [G][T][2D]; step 2 the BF16 z
// [G][T][64]; step 3 the outputs.
struct MsgGates {
    const float* hs[MAX_Q];
    const float* m[MAX_Q];
    const bf16* down[MAX_Q];  // [64, 2D]
    const bf16* fg[MAX_Q];    // [2D, 64]: forget rows, then gain rows
    float* out[MAX_Q];
};
constexpr int RANK = 64;

// Block (gate, 8 rows of down, token): warp r computes row r.
__global__ void __launch_bounds__(256) msg_down_kernel(MsgGates g, const bf16* __restrict__ cat, int T, bf16* z) {
    pdl_launch_dependents();
    pdl_wait();
    const int gi = blockIdx.x, lane = threadIdx.x & 31, r = blockIdx.y * 8 + (threadIdx.x >> 5);
    const bf16* w = g.down[gi] + (size_t)r * 2 * D;
    {
        const int t = blockIdx.z;
        const bf16* x = cat + ((size_t)gi * T + t) * 2 * D;
        float acc = 0.0f;
        for (int c = lane * 8; c < 2 * D; c += 256) {
            const uint4 wv = *reinterpret_cast<const uint4*>(w + c);
            const uint4 xv = *reinterpret_cast<const uint4*>(x + c);
            const bf16* wp = reinterpret_cast<const bf16*>(&wv);
            const bf16* xp = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc += bf2f(wp[e]) * bf2f(xp[e]);
        }
        acc = warp_sum(acc);
        if (lane == 0) z[((size_t)gi * T + t) * RANK + r] = f2bf(acc);
    }
}

// Block (gate * T + t, 256 channels).
__global__ void __launch_bounds__(256) msg_apply_kernel(MsgGates g, const bf16* __restrict__ z, int T) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float zs[RANK];
    const int gi = blockIdx.x / T, t = blockIdx.x % T, c = blockIdx.y * 256 + threadIdx.x;
    if (threadIdx.x < RANK) zs[threadIdx.x] = bf2f(z[((size_t)gi * T + t) * RANK + threadIdx.x]);
    __syncthreads();
    if (c >= D) return;
    const bf16* wf = g.fg[gi] + (size_t)c * RANK;
    const bf16* wg = g.fg[gi] + (size_t)(D + c) * RANK;
    float f = 0.0f, gg = 0.0f;
#pragma unroll
    for (int r = 0; r < RANK; r += 8) {
        const uint4 a = *reinterpret_cast<const uint4*>(wf + r);
        const uint4 b = *reinterpret_cast<const uint4*>(wg + r);
        const bf16* ap = reinterpret_cast<const bf16*>(&a);
        const bf16* bp = reinterpret_cast<const bf16*>(&b);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            f += bf2f(ap[e]) * zs[r + e];
            gg += bf2f(bp[e]) * zs[r + e];
        }
    }
    const size_t i = (size_t)t * D + c;
    g.out[gi][i] = (1.0f - tanhf(f)) * g.hs[gi][i] + tanhf(gg) * g.m[gi][i];
}

// ---------------------------------------------------------------------------------------------------------
// Query side of an attention layer from the fused projection lin [T, ld] (q | gate | relative states |
// sinks): q BF16 [T, 24, 128] scaled, L2-clipped (and log-scaled in full layers); rel F32 [T, 24, 16]
// likewise; sink F32 [T, 24].
SQ_DEVICE float log_tau(int pos) {
    return 1.0f + 0.1f * logf(fmaxf((float)(pos + 1) / 8192.0f, 1.0f));
}

// Keys (blockIdx.y 0) or values (1): the causal convolution of the projection rows (this step's from
// `lin`, earlier positions' from the ring, both as BF16), then per head the L2 clip (keys) or RMS
// normalization (values); stored into the cache (row pos % cache_rows); the step's rows join the ring.
// Static taps conv [KVD, 4] (tap 3 = current position) or, with `dyn`, static_w [4, KVD] (tap 0 = current)
// plus the token's per-head taps dyn[t, h * 4 + lag]. blockIdx.y 2: the query side (q_prep).
struct KvConv {
    const float* lin;
    int ld;
    bf16* ring;
    const bf16* conv;
    const bf16* static_w;
    const float* dyn;
    bf16* cache;
};

struct QPrep {
    const float* lin;
    int ld;
    const bf16* qscale;
    const bf16* rscale;
    float qclip, rclip;
    int full;
    bf16* q;
    float* rel;
    float* sink;
};

__global__ void __launch_bounds__(256) qkv_prep_kernel(QPrep qp, KvConv k, KvConv v, int ring_rows, int cache_rows, float clip,
                                                       const int* __restrict__ start_ptr) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x, lane = threadIdx.x & 31, h = threadIdx.x >> 5;
    const int pos = *start_ptr + t;
    if (blockIdx.y == 2) {
        const float tau = qp.full ? log_tau(pos) : 1.0f;
        const float* row = qp.lin + (size_t)t * qp.ld;
        for (int hh = h; hh < H; hh += 8) {
            float vq[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int d = lane * 4 + i;
                vq[i] = row[hh * HD + d] * bf2f(qp.qscale[hh * HD + d]);
                ss += vq[i] * vq[i];
            }
            const float c = fminf(qp.qclip / (sqrtf(warp_sum(ss)) + 1e-6f), 1.0f);
#pragma unroll
            for (int i = 0; i < 4; ++i) qp.q[(size_t)t * QD + hh * HD + lane * 4 + i] = f2bf(vq[i] * c * tau);
            float r = lane < DREL ? row[2 * QD + hh * DREL + lane] * bf2f(qp.rscale[hh * DREL + lane]) : 0.0f;
            const float rc = fminf(qp.rclip / (sqrtf(warp_sum(r * r)) + 1e-6f), 1.0f);
            if (lane < DREL) qp.rel[((size_t)t * H + hh) * DREL + lane] = r * rc * tau;
            if (lane == 0) qp.sink[(size_t)t * H + hh] = row[2 * QD + H * DREL + hh];
        }
        return;
    }
    const KvConv& a = blockIdx.y == 0 ? k : v;
    float y[4], ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = h * HD + lane * 4 + i;
        float xs[CONVK];
#pragma unroll
        for (int lag = 0; lag < CONVK; ++lag) {
            if (t - lag >= 0)
                xs[lag] = rb(a.lin[(size_t)(t - lag) * a.ld + c]);
            else
                xs[lag] = pos - lag >= 0 ? bf2f(a.ring[(size_t)((pos - lag) % ring_rows) * KVD + c]) : 0.0f;
        }
        a.ring[(size_t)(pos % ring_rows) * KVD + c] = f2bf(xs[0]);
        float acc = 0.0f, norm;
        if (a.dyn) {
#pragma unroll
            for (int lag = 0; lag < CONVK; ++lag) {
                const float wl = bf2f(a.static_w[lag * KVD + c]) + a.dyn[(size_t)t * a.ld + h * CONVK + lag];
                acc += wl * xs[lag];
            }
            norm = 3.0f;  // sqrt(2 * 4 + 1)
        } else {
#pragma unroll
            for (int j = 0; j < CONVK; ++j) acc += bf2f(a.conv[c * CONVK + j]) * xs[CONVK - 1 - j];
            norm = 2.2360679775f;  // sqrt(4 + 1)
        }
        y[i] = (acc + xs[0]) / norm;
        ss += y[i] * y[i];
    }
    ss = warp_sum(ss);
    const float f = blockIdx.y == 0 ? fminf(clip / (sqrtf(ss) + 1e-6f), 1.0f) : rsqrtf(ss / HD + EPS);
    bf16* dst = a.cache + (size_t)(pos % cache_rows) * KVD + h * HD + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = f2bf(y[i] * f);
}

// Hidden-width causal convolution of a sublayer output x (this step's rows from `src`, or with `cy` the
// experts' combine sum_k w[t, k] cy[t * 8 + k]; earlier positions' from the ring; all as BF16, conv
// [D, 4]): out = base + (conv + x) / sqrt(5) (base null: 0); the step's rows join the ring.
SQ_DEVICE float hidden_x(const float* src, const float* cy, const float* cw, int t, int c) {
    if (!cy) return rb(src[(size_t)t * D + c]);
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) acc += cw[t * TOPK + k] * cy[((size_t)t * TOPK + k) * D + c];
    return rb(acc);
}

__global__ void hidden_conv_kernel(const float* __restrict__ src, const float* __restrict__ cy, const float* __restrict__ cw, bf16* ring,
                                   int ring_rows, const bf16* __restrict__ conv, const float* base, const int* __restrict__ start_ptr, int T,
                                   float* out) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * D) return;
    const int t = (int)(i / D), c = (int)(i % D);
    const int pos = *start_ptr + t;
    float acc = 0.0f, cur = 0.0f;
#pragma unroll
    for (int lag = 0; lag < CONVK; ++lag) {
        float x;
        if (t - lag >= 0)
            x = hidden_x(src, cy, cw, t - lag, c);
        else
            x = pos - lag >= 0 ? bf2f(ring[(size_t)((pos - lag) % ring_rows) * D + c]) : 0.0f;
        if (lag == 0) cur = x;
        acc += bf2f(conv[c * CONVK + CONVK - 1 - lag]) * x;
    }
    ring[(size_t)(pos % ring_rows) * D + c] = f2bf(cur);
    const float y = (acc + cur) / 2.2360679775f;
    out[i] = (base ? base[i] : 0.0f) + y;
}

// ---------------------------------------------------------------------------------------------------------
// Attention partials of one (query t, KV head g, key split): its 3 query heads over the split's keys.
// Keys of query position p = start + t: up to hi = p (or, with `block`, the block's last position
// start + block - 1 for every query), from max(0, hi - window + 1) (window 0: from 0); split into gridDim.z
// ranges of at least `min_keys`. Scores get the relative position bias when `rel` is set.
// part[(t * H + h) * splits + split] = {max, sum, out[128]} (unnormalized).
constexpr int AT_T = 32, AT_THREADS = 128;

struct AttnArgs {
    const bf16* q;
    const float* rel;
    const bf16* kc;
    const bf16* vc;
    int cache_rows;
    const bf16* proj;
    int window;
    int block;
    int min_keys;
    const int* start;
    float* part;
};

__global__ void __launch_bounds__(AT_THREADS) attn_kernel(AttnArgs a) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float qs[GQ][HD];
    __shared__ float rs[GQ][DREL];
    __shared__ float ks[AT_T][HD + 1];
    __shared__ float vs[AT_T][HD];
    __shared__ float ps[GQ][AT_T];
    __shared__ float scl[GQ];
    const int t = blockIdx.x, g = blockIdx.y, split = blockIdx.z, splits = gridDim.z, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int p = *a.start + t;
    const int hi = a.block ? *a.start + a.block - 1 : p;
    const int lo = a.window ? max(0, hi - a.window + 1) : 0;
    const int n = hi + 1 - lo;
    const int per = max(a.min_keys, (n + splits - 1) / splits);
    const int k0 = lo + split * per, k1 = min(hi + 1, k0 + per);
    for (int i = tid; i < GQ * HD; i += AT_THREADS) qs[i / HD][i % HD] = bf2f(a.q[(size_t)t * QD + (g * GQ + i / HD) * HD + i % HD]);
    if (tid < GQ * DREL) rs[tid / DREL][tid % DREL] = a.rel ? a.rel[((size_t)t * H + g * GQ + tid / DREL) * DREL + tid % DREL] : 0.0f;
    float m = -INFINITY, l = 0.0f;  // per head (warps 0..2, lane-uniform)
    float o[GQ] = {0.0f, 0.0f, 0.0f};
    const float scale = rsqrtf((float)HD);
    for (int kb = k0; kb < k1; kb += AT_T) {
        const int nk = min(AT_T, k1 - kb);
        __syncthreads();
        for (int i = tid; i < AT_T * (HD / 8); i += AT_THREADS) {
            const int j = i / (HD / 8), c = (i % (HD / 8)) * 8;
            float kv[8], vv[8];
            if (j < nk) {
                const size_t row = (size_t)((kb + j) % a.cache_rows) * KVD + g * HD + c;
                const uint4 kr = *reinterpret_cast<const uint4*>(a.kc + row);
                const uint4 vr = *reinterpret_cast<const uint4*>(a.vc + row);
                const bf16* kp = reinterpret_cast<const bf16*>(&kr);
                const bf16* vp = reinterpret_cast<const bf16*>(&vr);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    kv[e] = bf2f(kp[e]);
                    vv[e] = bf2f(vp[e]);
                }
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) kv[e] = vv[e] = 0.0f;
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                ks[j][c + e] = kv[e];
                vs[j][c + e] = vv[e];
            }
        }
        __syncthreads();
        if (warp < GQ) {
            const int j = lane, key = kb + j;
            float s = -INFINITY;
            if (j < nk) {
                float acc = 0.0f;
#pragma unroll 16
                for (int d = 0; d < HD; ++d) acc += qs[warp][d] * ks[j][d];
                s = acc * scale;
                const int dist = p - key;
                if (a.rel && dist >= 0 && dist < EXT) {
                    float b = 0.0f;
#pragma unroll
                    for (int r = 0; r < DREL; ++r) b += rs[warp][r] * bf2f(a.proj[r * EXT + dist]);
                    s += b;
                }
            }
            const float mx = fmaxf(m, warp_max(s));
            const float pr = j < nk ? __expf(s - mx) : 0.0f;
            const float c = __expf(m - mx);
            l = l * c + warp_sum(pr);
            m = mx;
            ps[warp][j] = pr;
            if (lane == 0) scl[warp] = c;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < GQ; ++i) {
            float acc = o[i] * scl[i];
            for (int j = 0; j < nk; ++j) acc += ps[i][j] * vs[j][tid];
            o[i] = acc;
        }
    }
    __syncthreads();
    // Warps 0..2 hold their head's (m, l); share them.
    __shared__ float ml[GQ][2];
    if (warp < GQ && lane == 0) {
        ml[warp][0] = m;
        ml[warp][1] = l;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < GQ; ++i) {
        float* dst = a.part + (((size_t)t * H + g * GQ + i) * splits + split) * (2 + HD);
        if (tid == 0) {
            dst[0] = ml[i][0];
            dst[1] = ml[i][1];
        }
        dst[2 + tid] = o[i];
    }
}

// Tensor-core attention for prefill chunks: block (16 query positions t0 .., KV head g), warp w = query head
// g * 3 + w over its 16 positions (one m16 tile). Keys in tiles of 64 through shared memory; S = Q K^T and
// O += P V on BF16 MMAs (P from the S accumulators); the relative bias (keys within 1024 positions) and
// the masks per element; online softmax; then the sink and the output gate. Same arguments as attn_kernel
// (rel required; no block mode).
constexpr int TC_Q = 16, TC_K = 64, TC_W = GQ, TC_LD = HD + 8, TC_B2 = TC_Q + TC_K;
constexpr size_t TC_SMEM = (size_t)(GQ * TC_Q * TC_LD + 2 * TC_K * TC_LD) * 2 + (size_t)TC_W * TC_Q * TC_B2 * 4 + (size_t)GQ * TC_Q * DREL * 4;

__global__ void __launch_bounds__(TC_W * 32) attn_tc_kernel(AttnArgs a, int T, const float* __restrict__ sink, const float* __restrict__ lin, int ld,
                                                            int gate_off, bf16* out) {
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* Qs = reinterpret_cast<bf16*>(smem_raw);  // [48][TC_LD]
    bf16* Ks = Qs + GQ * TC_Q * TC_LD;               // [64][TC_LD]
    bf16* Vs = Ks + TC_K * TC_LD;                    // [64][TC_LD]
    float* B2 = reinterpret_cast<float*>(Vs + TC_K * TC_LD);  // [3][16][80]
    float* Rs = B2 + TC_W * TC_Q * TC_B2;                     // [3][16][16]
    const int t0 = blockIdx.x * TC_Q, g = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int gq = lane >> 2, t4 = lane & 3;
    const int start = *a.start;
    const int rows = min(TC_Q, T - t0);
    // Q rows (head-major) and relative states.
    for (int i = tid; i < GQ * TC_Q * (HD / 8); i += TC_W * 32) {
        const int r = i / (HD / 8), c = (i % (HD / 8)) * 8, hh = r / TC_Q, qi = r % TC_Q;
        const bool ok = qi < rows;
        cp_async16(Qs + r * TC_LD + c, a.q + (size_t)(t0 + (ok ? qi : 0)) * QD + (g * GQ + hh) * HD + c, ok);
    }
    cp_async_commit();
    for (int i = tid; i < GQ * TC_Q * DREL; i += TC_W * 32) {
        const int hh = i / (TC_Q * DREL), qi = (i / DREL) % TC_Q, r = i % DREL;
        Rs[i] = qi < rows ? a.rel[((size_t)(t0 + qi) * H + g * GQ + hh) * DREL + r] : 0.0f;
    }
    const int p_first = start + t0, p_last = start + t0 + rows - 1;
    const int lo = a.window ? max(0, p_first - a.window + 1) : 0;
    const int h = g * GQ + w;
    // Rows of this thread: gq and gq + 8 (positions p_first + gq, + 8).
    const int pr[2] = {p_first + gq, p_first + gq + 8};
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    float o[HD / 8][4];
#pragma unroll
    for (int j = 0; j < HD / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    const float scale = rsqrtf((float)HD);
    float* b2 = B2 + w * TC_Q * TC_B2;
    const float* rs = Rs + w * TC_Q * DREL;
    for (int kb = lo; kb <= p_last; kb += TC_K) {
        __syncthreads();
        for (int i = tid; i < TC_K * (HD / 8); i += TC_W * 32) {
            const int j = i / (HD / 8), c = (i % (HD / 8)) * 8;
            const bool ok = kb + j <= p_last;
            const size_t row = (size_t)((ok ? kb + j : kb) % a.cache_rows) * KVD + g * HD + c;
            cp_async16(Ks + j * TC_LD + c, a.kc + row, ok);
            cp_async16(Vs + j * TC_LD + c, a.vc + row, ok);
        }
        cp_async_commit();
        // Relative bias by distance, for the tile's distances dmin .. dmin + 78 (row i, key kb + j:
        // d = p_first + i - kb - j, index i + 63 - j).
        const int dmin = p_first - (kb + TC_K - 1);
        const bool bias = dmin < EXT;
        if (bias) {
            for (int e = lane; e < TC_Q * (TC_B2 - 1); e += 32) {
                const int i = e / (TC_B2 - 1), x = e % (TC_B2 - 1), d = dmin + x;
                float v = 0.0f;
                if (d >= 0 && d < EXT) {
#pragma unroll
                    for (int r = 0; r < DREL; ++r) v += rs[i * DREL + r] * bf2f(a.proj[r * EXT + d]);
                }
                b2[i * TC_B2 + x] = v;
            }
        }
        cp_async_wait<0>();
        __syncthreads();
        float sc[TC_K / 8][4];
#pragma unroll
        for (int j = 0; j < TC_K / 8; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < HD; kk += 16) {
            uint32_t af[4];
            ldmatrix_x4(af, Qs + (w * TC_Q + (lane & 15)) * TC_LD + kk + (lane >> 4) * 8);
#pragma unroll
            for (int j = 0; j < TC_K / 16; ++j) {
                uint32_t bf[4];
                ldmatrix_x4(bf, Ks + (j * 16 + (lane & 7) + ((lane >> 4) << 3)) * TC_LD + kk + ((lane >> 3) & 1) * 8);
                mma_bf16(sc[2 * j], af, bf);
                mma_bf16(sc[2 * j + 1], af, bf + 2);
            }
        }
        // Scale, bias, masks; row maxima.
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < TC_K / 8; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int ri = e >> 1, key = kb + j * 8 + t4 * 2 + (e & 1);
                const int i = gq + ri * 8, d = pr[ri] - key;
                float v = sc[j][e] * scale;
                if (bias && d < EXT && d >= 0) v += b2[i * TC_B2 + (i + TC_K - 1 - (key - kb))];
                if (d < 0 || (a.window && d >= a.window) || i >= rows) v = -INFINITY;
                sc[j][e] = v;
                mx[ri] = fmaxf(mx[ri], v);
            }
        float cf[2];
#pragma unroll
        for (int ri = 0; ri < 2; ++ri) {
            mx[ri] = fmaxf(mx[ri], __shfl_xor_sync(0xffffffff, mx[ri], 1));
            mx[ri] = fmaxf(mx[ri], __shfl_xor_sync(0xffffffff, mx[ri], 2));
            const float mn = fmaxf(m[ri], mx[ri]);
            cf[ri] = mn == -INFINITY ? 1.0f : __expf(m[ri] - mn);
            m[ri] = mn;
        }
        float rsum[2] = {0.0f, 0.0f};
        uint32_t pf[TC_K / 16][4];
#pragma unroll
        for (int j = 0; j < TC_K / 8; ++j) {
            float pv[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int ri = e >> 1;
                pv[e] = m[ri] == -INFINITY ? 0.0f : __expf(sc[j][e] - m[ri]);
                rsum[ri] += pv[e];
            }
            const __nv_bfloat162 lo2 = __floats2bfloat162_rn(pv[0], pv[1]), hi2 = __floats2bfloat162_rn(pv[2], pv[3]);
            pf[j / 2][(j & 1) * 2] = *reinterpret_cast<const uint32_t*>(&lo2);
            pf[j / 2][(j & 1) * 2 + 1] = *reinterpret_cast<const uint32_t*>(&hi2);
        }
#pragma unroll
        for (int ri = 0; ri < 2; ++ri) {
            rsum[ri] += __shfl_xor_sync(0xffffffff, rsum[ri], 1);
            rsum[ri] += __shfl_xor_sync(0xffffffff, rsum[ri], 2);
            l[ri] = l[ri] * cf[ri] + rsum[ri];
        }
#pragma unroll
        for (int j = 0; j < HD / 8; ++j) {
            o[j][0] *= cf[0];
            o[j][1] *= cf[0];
            o[j][2] *= cf[1];
            o[j][3] *= cf[1];
        }
        // O += P V: k-steps of 16 keys; P's A fragment from S tiles 2s, 2s + 1.
#pragma unroll
        for (int ks = 0; ks < TC_K / 16; ++ks) {
            uint32_t af[4] = {pf[ks][0], pf[ks][1], pf[ks][2], pf[ks][3]};
#pragma unroll
            for (int nj = 0; nj < HD / 16; ++nj) {
                uint32_t bf[4];
                ldmatrix_x4_trans(bf, Vs + (ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * TC_LD + nj * 16 + (lane >> 4) * 8);
                mma_bf16(o[2 * nj], af, bf);
                mma_bf16(o[2 * nj + 1], af, bf + 2);
            }
        }
    }
    // Sink, normalization, gate.
#pragma unroll
    for (int ri = 0; ri < 2; ++ri) {
        const int i = gq + ri * 8;
        if (i >= rows) continue;
        const int t = t0 + i;
        const float sk = sink[(size_t)t * H + h];
        const float M = fmaxf(m[ri], sk);
        const float f = __expf(m[ri] - M);
        const float inv = 1.0f / (l[ri] * f + __expf(sk - M));
#pragma unroll
        for (int j = 0; j < HD / 8; ++j)
#pragma unroll
            for (int x = 0; x < 2; ++x) {
                const int d = j * 8 + t4 * 2 + x;
                const float gate = sigmoidf(lin[(size_t)t * ld + gate_off + h * HD + d]);
                out[(size_t)t * QD + h * HD + d] = f2bf(o[j][ri * 2 + x] * f * inv * gate);
            }
    }
}

// Merges the splits of (t, head) with the sink (none when null) and applies the output gate (none when lin
// is null): out BF16 [T, 3072] = attention * sigmoid(gate), gate at lin[t * ld + gate_off + h * 128 + d].
__global__ void __launch_bounds__(HD) attn_merge_kernel(const float* __restrict__ part, int splits, const float* __restrict__ sink,
                                                        const float* __restrict__ lin, int ld, int gate_off, bf16* out) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float* base = part + ((size_t)t * H + h) * splits * (2 + HD);
    const float sk = sink ? sink[(size_t)t * H + h] : -INFINITY;
    float M = sk;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, base[s * (2 + HD)]);
    float L = sink ? __expf(sk - M) : 0.0f, o = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float* b = base + s * (2 + HD);
        if (b[1] == 0.0f) continue;
        const float w = __expf(b[0] - M);
        L += b[1] * w;
        o += b[2 + d] * w;
    }
    const float gate = lin ? sigmoidf(lin[(size_t)t * ld + gate_off + h * HD + d]) : 1.0f;
    out[(size_t)t * QD + h * HD + d] = f2bf(o / L * gate);
}

// ---------------------------------------------------------------------------------------------------------
// Router: score = sigmoid(logit); the 8 experts of largest score + bias; weights = score / sum * scale.
__global__ void __launch_bounds__(E) route_kernel(const float* __restrict__ logits, const float* __restrict__ bias, float scale, int* sel_e,
                                                  float* sel_w) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float key[E], sc[E];
    __shared__ float red[32];
    const int t = blockIdx.x, e = threadIdx.x;
    const float s = sigmoidf(logits[(size_t)t * E + e]);
    sc[e] = s;
    key[e] = s + bias[e];
    __syncthreads();
    int rank = 0;
    const float me = key[e];
    for (int f = 0; f < E; ++f) rank += key[f] > me || (key[f] == me && f < e);
    const bool chosen = rank < TOPK;
    const float total = block_sum(chosen ? s : 0.0f, red);
    if (chosen) {
        sel_e[t * TOPK + rank] = e;
        sel_w[t * TOPK + rank] = s / (total + 1e-20f) * scale;
    }
}

// Routing of up to 16 tokens and the pairs' order by expert (ties by pair), one block of 1024 threads
// (8 tokens at a time, 128 threads a token).
__global__ void __launch_bounds__(1024) route_order_kernel(const float* __restrict__ logits, const float* __restrict__ bias, float scale, int T,
                                                           int* sel_e, float* sel_w, int* order) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float key[8][E], picked[8][TOPK];
    __shared__ int se[16 * TOPK];
    const int g = threadIdx.x / E, e = threadIdx.x % E;
    for (int t0 = 0; t0 < T; t0 += 8) {
        const int t = t0 + g;
        float sc = 0.0f;
        if (t < T) {
            sc = sigmoidf(logits[(size_t)t * E + e]);
            key[g][e] = sc + bias[e];
        }
        __syncthreads();
        if (t < T) {
            const float me = key[g][e];
            int rank = 0;
            for (int f = 0; f < E; ++f) rank += key[g][f] > me || (key[g][f] == me && f < e);
            if (rank < TOPK) {
                se[t * TOPK + rank] = e;
                picked[g][rank] = sc;
            }
        }
        __syncthreads();
        if (t < T && e < TOPK) {
            float total = 0.0f;
            for (int k = 0; k < TOPK; ++k) total += picked[g][k];
            sel_e[t * TOPK + e] = se[t * TOPK + e];
            sel_w[t * TOPK + e] = picked[g][e] / (total + 1e-20f) * scale;
        }
        __syncthreads();
    }
    const int pairs = T * TOPK;
    if (threadIdx.x < pairs) {
        const int p = threadIdx.x, ep = se[p];
        int rank = 0;
        for (int q = 0; q < pairs; ++q) rank += se[q] < ep || (se[q] == ep && q < p);
        order[rank] = p;
    }
}

// Pairs (t * 8 + k) ordered by expert: order[...] (one block).
__global__ void __launch_bounds__(1024) moe_order_kernel(const int* __restrict__ sel_e, int pairs, int* order) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ int count[E], cursor[E];
    for (int i = threadIdx.x; i < E; i += blockDim.x) count[i] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) atomicAdd(&count[sel_e[i]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0;
        for (int i = 0; i < E; ++i) {
            cursor[i] = o;
            o += count[i];
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) order[atomicAdd(&cursor[sel_e[i]], 1)] = i;
}

// Expert weights: NVFP4 (codes [E, R, K / 2] low nibble first, E4M3 scales [E, R, K / 16], s2 [E]) or MXFP8
// (E4M3 [E, R, K], UE8M0 scales [E, R, K / 32]).
struct ExpertW {
    const uint8_t* w;
    const uint8_t* sc;
    const float* s2;
};

// Rows n0 .. n0 + R - 1 of expert e times the BF16 vector x (length K): one warp, F32 sums in all lanes.
// The rows are one contiguous run of 16-byte chunks (NVFP4: 32 values and 2 scale bytes; MXFP8: 16 values
// and half a scale byte); lane l takes chunks l, l + 32, ..., so every load is 16 bytes and every lane is
// busy (R * chunks per row is a multiple of 32 for every Inkling shape).
template <bool FP4, int R>
SQ_DEVICE void rows_dot(const ExpertW& wt, int e, int rows_per_expert, int n0, int K, const bf16* x, float (&out)[R]) {
    constexpr int CV = FP4 ? 32 : 16;  // values per chunk
    const int lane = threadIdx.x & 31, cpr = K / CV, total = R * cpr;
    const size_t row0 = (size_t)e * rows_per_expert + n0;
#pragma unroll
    for (int r = 0; r < R; ++r) out[r] = 0.0f;
    // Three chunks' weights in flight per lane (R * chunks per row is a multiple of 96 for every shape).
    constexpr int U = 3;
    for (int c0 = lane; c0 < total; c0 += 32 * U) {
        uint4 wv[U];
        uint16_t sv[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int c = c0 + 32 * u, r = c / cpr, cc = c % cpr;
            const bool ok = c < total;
            wv[u] = ok ? ld_stream(wt.w + (row0 + r) * (FP4 ? K / 2 : K) + (size_t)cc * 16) : make_uint4(0, 0, 0, 0);
            if constexpr (FP4)
                sv[u] = ok ? *reinterpret_cast<const uint16_t*>(wt.sc + (row0 + r) * (K / 16) + cc * 2) : 0;
            else
                sv[u] = ok ? wt.sc[(row0 + r) * (K / 32) + cc / 2] : 0;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int c = c0 + 32 * u;
            if (c >= total) break;
            const int r = c / cpr, cc = c % cpr;
            const size_t row = row0 + r;
            const uint4 w = wv[u];
            float xs[CV];
            const uint4* src = reinterpret_cast<const uint4*>(x + cc * CV);
#pragma unroll
            for (int h = 0; h < CV / 8; ++h) {
                const uint4 v = src[h];
                const bf16* pv = reinterpret_cast<const bf16*>(&v);
#pragma unroll
                for (int i = 0; i < 8; ++i) xs[h * 8 + i] = bf2f(pv[i]);
            }
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&w);
            float acc;
            if constexpr (FP4) {
                const uint16_t sc2 = sv[u];
                float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2(b[i], __NV_E2M1);
                    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
                    if (i < 8) {
                        a0 = fmaf(f.x, xs[2 * i], a0);
                        a0 = fmaf(f.y, xs[2 * i + 1], a0);
                    } else {
                        a1 = fmaf(f.x, xs[2 * i], a1);
                        a1 = fmaf(f.y, xs[2 * i + 1], a1);
                    }
                }
                acc = a0 * e4m3_value(sc2 & 0xff) + a1 * e4m3_value(sc2 >> 8);
            } else {
                acc = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(*reinterpret_cast<const __nv_fp8x2_storage_t*>(b + 2 * i), __NV_E4M3);
                    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
                    acc = fmaf(f.x, xs[2 * i], acc);
                    acc = fmaf(f.y, xs[2 * i + 1], acc);
                }
                acc *= ue8m0((uint8_t)sv[u]);
            }
#pragma unroll
            for (int q = 0; q < R; ++q)
                if (q == r) out[q] += acc;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) out[r] = warp_sum(out[r]);
}

SQ_DEVICE float softcap(float x, float cap) { return tanhf(x / cap) * cap; }

// Gate/up of every (pair, 8 interleaved rows) item, pairs in expert order: act[pair, i] for 4 indices i =
// rb(softcap(g, 4) * sigmoid(g) * softcap(u, 25)), g = rb(row 2i . x), u = rb(row 2i + 1 . x).
constexpr int EX_R = 8;

template <bool FP4>
__global__ void __launch_bounds__(256) experts_gate_up_kernel(const bf16* __restrict__ x, const int* __restrict__ sel_e, const int* __restrict__ order,
                                                              ExpertW wt, int pairs, float gcap, float ucap, bf16* act) {
    pdl_launch_dependents();
    pdl_wait();
    const int lane = threadIdx.x & 31;
    const int warps = gridDim.x * 8, items = pairs * (2 * FF / EX_R);
    for (int item = blockIdx.x * 8 + (threadIdx.x >> 5); item < items; item += warps) {
        const int pair = order[item % pairs], n0 = item / pairs * EX_R;
        const int t = pair / TOPK, e = sel_e[pair];
        float v[EX_R];
        rows_dot<FP4, EX_R>(wt, e, 2 * FF, n0, D, x + (size_t)t * D, v);
        const float s2 = FP4 ? wt.s2[e] : 1.0f;
        if (lane < EX_R / 2) {
            float gv = 0.0f, uv = 0.0f;
#pragma unroll
            for (int i = 0; i < EX_R / 2; ++i)
                if (i == lane) {
                    gv = v[2 * i];
                    uv = v[2 * i + 1];
                }
            const float g = gv * s2, u = uv * s2;
            act[(size_t)pair * FF + n0 / 2 + lane] = f2bf(softcap(g, gcap) * sigmoidf(g) * softcap(u, ucap));
        }
    }
}

// Down of every (pair, 8 rows) item: y[pair, n] = rb(row n . act[pair]) (F32 [pairs, D]).
template <bool FP4>
__global__ void __launch_bounds__(256) experts_down_kernel(const bf16* __restrict__ act, const int* __restrict__ sel_e, const int* __restrict__ order,
                                                           ExpertW wt, int pairs, float* y) {
    pdl_launch_dependents();
    pdl_wait();
    const int lane = threadIdx.x & 31;
    const int warps = gridDim.x * 8, items = pairs * (D / EX_R);
    for (int item = blockIdx.x * 8 + (threadIdx.x >> 5); item < items; item += warps) {
        const int pair = order[item % pairs], n0 = item / pairs * EX_R;
        const int e = sel_e[pair];
        float v[EX_R];
        rows_dot<FP4, EX_R>(wt, e, D, n0, FF, act + (size_t)pair * FF, v);
        const float s2 = FP4 ? wt.s2[e] : 1.0f;
        if (lane < EX_R) {
            float vv = 0.0f;
#pragma unroll
            for (int i = 0; i < EX_R; ++i)
                if (i == lane) vv = v[i];
            y[(size_t)pair * D + n0 + lane] = vv * s2;
        }
    }
}

// out[t, n] = rb(sum_k w[t, k] * y[t * 8 + k, n]) (F32).
__global__ void moe_combine_kernel(const float* __restrict__ y, const float* __restrict__ sel_w, int T, float* out) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * D) return;
    const size_t t = i / D, n = i % D;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) acc += sel_w[t * TOPK + k] * y[(t * TOPK + k) * D + n];
    out[i] = acc;
}

// ---- Grouped expert GEMMs (prefill) ----------------------------------------------------------------
// Pairs grouped by expert into rows, tiles of `bm` rows per expert (moe.cu's NVFP4 GEMM layout):
// offsets [E + 1], tile_expert / tile_row per tile, num_tiles, pair_row[pair] = the pair's row. One block.
__global__ void __launch_bounds__(1024) moe_plan_kernel(const int* __restrict__ sel_e, int pairs, int bm, int* offsets, int* tile_expert,
                                                        int* tile_row, int* num_tiles, int* pair_row) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ int count[E], cursor[E], tiles[E + 1], rows[E + 1];
    for (int i = threadIdx.x; i < E; i += blockDim.x) count[i] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) atomicAdd(&count[sel_e[i]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int r = 0, tl = 0;
        for (int e = 0; e < E; ++e) {
            rows[e] = r;
            tiles[e] = tl;
            r += count[e];
            tl += (count[e] + bm - 1) / bm;
        }
        rows[E] = r;
        tiles[E] = tl;
        *num_tiles = tl;
    }
    __syncthreads();
    for (int e = threadIdx.x; e <= E; e += blockDim.x) offsets[e] = rows[e];
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        cursor[e] = rows[e];
        for (int i = 0; i * bm < count[e]; ++i) {
            tile_expert[tiles[e] + i] = e;
            tile_row[tiles[e] + i] = rows[e] + i * bm;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) pair_row[i] = atomicAdd(&cursor[sel_e[i]], 1);
}

// xs[pair_row[pair]] = x[pair / 8] (BF16 rows of D).
__global__ void moe_gather_kernel(const bf16* __restrict__ x, const int* __restrict__ pair_row, bf16* xs) {
    pdl_launch_dependents();
    pdl_wait();
    const int pair = blockIdx.x;
    const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)(pair / TOPK) * D);
    uint4* dst = reinterpret_cast<uint4*>(xs + (size_t)pair_row[pair] * D);
    for (int i = threadIdx.x; i < D / 8; i += blockDim.x) dst[i] = src[i];
}

// act[row, i] = softcap(g, gcap) * sigmoid(g) * softcap(u, ucap), g = y[row, 2i], u = y[row, 2i + 1].
__global__ void moe_act_kernel(const float* __restrict__ y, int rows, float gcap, float ucap, bf16* act) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * FF) return;
    const float2 gu = reinterpret_cast<const float2*>(y)[i];
    act[i] = f2bf(softcap(gu.x, gcap) * sigmoidf(gu.x) * softcap(gu.y, ucap));
}

// out[t, n] = sum_k w[t, k] * y[pair_row[t * 8 + k], n].
__global__ void moe_combine_rows_kernel(const float* __restrict__ y, const int* __restrict__ pair_row, const float* __restrict__ sel_w, int T,
                                        float* out) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * D) return;
    const size_t t = i / D, n = i % D;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) acc += sel_w[t * TOPK + k] * y[(size_t)pair_row[t * TOPK + k] * D + n];
    out[i] = acc;
}

// Grouped MXFP8 GEMM (prefill experts of the MXFP8 layers), the layout of moe.cu's NVFP4 GEMM: Y[row, N] =
// Xs[row, K] . W_e[N, K]^T for the rows of expert e's tiles; W_e E4M3 [N, K] at w + e N K, UE8M0 scales
// [N, K / 32] at sc + e N K / 32. E4M3 times a power of two is exact in BF16, so the B tiles are exact.
constexpr int MXG_BM = 64, MXG_BN = 128, MXG_BK = 32, MXG_LDS = MXG_BK + 8;

__global__ void __launch_bounds__(256) mx8_grouped_gemm_kernel(const bf16* __restrict__ X, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
                                                               const int* __restrict__ offsets, const int* __restrict__ tile_expert,
                                                               const int* __restrict__ tile_row, const int* __restrict__ num_tiles, float* Y, int N,
                                                               int K) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ __align__(16) bf16 sA[2][MXG_BM * MXG_LDS];
    __shared__ __align__(16) bf16 sB[2][MXG_BN * MXG_LDS];
    if ((int)blockIdx.y >= *num_tiles) return;
    const int e = tile_expert[blockIdx.y], r0 = tile_row[blockIdx.y];
    const int r_end = min(r0 + MXG_BM, offsets[e + 1]);
    const uint8_t* W = w + (size_t)e * N * K;
    const uint8_t* Sc = sc + (size_t)e * N * (K / 32);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp >> 2, warp_n = warp & 3;
    const int n0 = blockIdx.x * MXG_BN;
    const int k_tiles = K / MXG_BK;
    auto load_a = [&](int stage, int kt) {
        const int r = tid >> 2, c = (tid & 3) * 8;
        const bool ok = r0 + r < r_end;
        cp_async16(&sA[stage][r * MXG_LDS + c], X + (size_t)(ok ? r0 + r : r0) * K + kt * MXG_BK + c, ok);
    };
    uint4 b0 = make_uint4(0, 0, 0, 0), b1 = b0;
    uint8_t bsc = 0;
    const bool b_row = tid < MXG_BN && n0 + tid < N;
    auto fetch_b = [&](int kt) {
        if (b_row) {
            const uint4* src = reinterpret_cast<const uint4*>(W + (size_t)(n0 + tid) * K + kt * MXG_BK);
            b0 = src[0];
            b1 = src[1];
            bsc = Sc[(size_t)(n0 + tid) * (K / 32) + kt];
        }
    };
    auto store_b = [&](int stage) {
        if (tid >= MXG_BN) return;
        bf16* dst = &sB[stage][tid * MXG_LDS];
        uint32_t packed[16];
        const float scale = ue8m0(bsc);
        const uint8_t* bb[2] = {reinterpret_cast<const uint8_t*>(&b0), reinterpret_cast<const uint8_t*>(&b1)};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (b_row) {
                const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(*reinterpret_cast<const __nv_fp8x2_storage_t*>(bb[i / 8] + 2 * (i % 8)), __NV_E4M3);
                const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
                const __nv_bfloat162 v = __floats2bfloat162_rn(f.x * scale, f.y * scale);
                packed[i] = *reinterpret_cast<const uint32_t*>(&v);
            } else {
                packed[i] = 0;
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
            *reinterpret_cast<uint4*>(dst + 8 * i) = make_uint4(packed[4 * i], packed[4 * i + 1], packed[4 * i + 2], packed[4 * i + 3]);
    };
    float acc[2][4][4] = {};
    load_a(0, 0);
    cp_async_commit();
    fetch_b(0);
    store_b(0);
    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_async_wait<0>();
        __syncthreads();
        const bool more = kt + 1 < k_tiles;
        if (more) {
            load_a((kt + 1) % 2, kt + 1);
            fetch_b(kt + 1);
        }
        cp_async_commit();
        const bf16* a_tile = sA[kt % 2];
        const bf16* b_tile = sB[kt % 2];
#pragma unroll
        for (int kk = 0; kk < MXG_BK; kk += 16) {
            uint32_t a[2][4], b[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) ldmatrix_x4(a[i], &a_tile[(warp_m * 32 + i * 16 + (lane & 15)) * MXG_LDS + kk + (lane >> 4) * 8]);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                ldmatrix_x4(b[j], &b_tile[(warp_n * 32 + j * 16 + (lane & 7) + ((lane >> 4) << 3)) * MXG_LDS + kk + ((lane >> 3) & 1) * 8]);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_bf16(acc[i][j], a[i], &b[j >> 1][(j & 1) * 2]);
        }
        if (more) store_b((kt + 1) % 2);
    }
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = r0 + warp_m * 32 + i * 16 + g + h * 8;
                const int n = n0 + warp_n * 32 + j * 8 + t4 * 2;
                if (r >= r_end) continue;
                if (n < N) Y[(size_t)r * N + n] = acc[i][j][h * 2];
                if (n + 1 < N) Y[(size_t)r * N + n + 1] = acc[i][j][h * 2 + 1];
            }
}

// x = cap * tanh(x / cap) over rows of n values (row stride ld).
__global__ void softcap_kernel(float* x, int rows, int n, int ld, float cap) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * n) return;
    float* p = x + (i / n) * ld + i % n;
    *p = softcap(*p, cap);
}

// ---------------------------------------------------------------------------------------------------------
// DSpark drafter.

// F32 rows [T, D] to BF16 rows at stride ldd.
__global__ void to_bf16_rows_kernel(const float* __restrict__ src, int T, bf16* dst, int ldd) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * D) return;
    dst[(i / D) * ldd + i % D] = f2bf(src[i]);
}

// out row r = bf16(w * rmsnorm(sum_k alpha[k] * feats[row, k])) with row = rows_from ? *rows_from : r
// (feats BF16 [*, 4, D]).
constexpr int TAPS = 4;
__global__ void __launch_bounds__(RT) mix_kernel(const bf16* __restrict__ feats, const int* __restrict__ rows_from, const float* __restrict__ alpha,
                                                 const bf16* __restrict__ w, bf16* out, int ldo) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float red[32];
    const int r = blockIdx.x;
    const int row = rows_from ? *rows_from : r;
    const bf16* f = feats + (size_t)row * TAPS * D;
    float v[RPT], ss = 0.0f;
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
        const int c = threadIdx.x + RT * i;
        float m = 0.0f;
#pragma unroll
        for (int k = 0; k < TAPS; ++k) m += alpha[k] * bf2f(f[k * D + c]);
        v[i] = m;
        ss += m * m;
    }
    const float inv = rsqrtf(block_sum(ss, red) / D + EPS);
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
        const int c = threadIdx.x + RT * i;
        out[(size_t)r * ldo + c] = f2bf(v[i] * inv * bf2f(w[c]));
    }
}

// Neox RoPE of one 128-wide head held as 4 values per lane (dims 4 lane .. 4 lane + 3): dims d < 64 pair
// with d + 64 (lane + 16).
SQ_DEVICE void rope_head(float (&v)[4], int pos, float theta) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane * 4 + i, j = d & 63;
        const float inv_freq = 1.0f / powf(theta, (float)(2 * j) / HD);
        float sn, cs;
        sincosf((float)pos * inv_freq, &sn, &cs);
        const float other = __shfl_xor_sync(0xffffffff, v[i], 16);
        v[i] = d < 64 ? v[i] * cs - other * sn : v[i] * cs + other * sn;
    }
}

// Per-head RMSNorm (weight w, rounded to BF16) of 4 values per lane.
SQ_DEVICE void head_norm(float (&v)[4], const bf16* w) {
    const int lane = threadIdx.x & 31;
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) ss += v[i] * v[i];
    const float inv = rsqrtf(warp_sum(ss) / HD + EPS);
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = rb(v[i] * inv * bf2f(w[lane * 4 + i]));
}

// Context keys and values of a drafter layer from kv F32 [T, 2048] (keys | values): keys normalized per head
// (knorm) and rotated by position start + t; both stored to the caches. Block t, 8 warps (heads).
__global__ void __launch_bounds__(256) ctx_kv_kernel(const float* __restrict__ kv, const bf16* __restrict__ knorm, float theta,
                                                     const int* __restrict__ start_ptr, bf16* kcache, bf16* vcache) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x, h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = *start_ptr + t;
    const float* row = kv + (size_t)t * 2 * KVD + h * HD + lane * 4;
    float k[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) k[i] = row[i];
    head_norm(k, knorm);
    rope_head(k, pos, theta);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kcache[(size_t)pos * KVD + h * HD + lane * 4 + i] = f2bf(k[i]);
        vcache[(size_t)pos * KVD + h * HD + lane * 4 + i] = f2bf(row[KVD + i]);
    }
}

// The drafter block's queries, keys and values from qkv F32 [T, 5120]: q and k normalized per head and
// rotated (position start + t); q to BF16 [T, 3072], k and v to the caches. Block (t, head of 40), 1 warp.
__global__ void __launch_bounds__(32) draft_qkv_kernel(const float* __restrict__ qkv, const bf16* __restrict__ qnorm, const bf16* __restrict__ knorm,
                                                       float theta, const int* __restrict__ start_ptr, bf16* q, bf16* kcache, bf16* vcache) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    const int pos = *start_ptr + t;
    const float* row = qkv + (size_t)t * (QD + 2 * KVD) + h * HD + lane * 4;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = row[i];
    bf16* dst;
    if (h < H) {
        head_norm(v, qnorm);
        rope_head(v, pos, theta);
        dst = q + (size_t)t * QD + h * HD;
    } else if (h < H + KVH) {
        head_norm(v, knorm);
        rope_head(v, pos, theta);
        dst = kcache + (size_t)pos * KVD + (h - H) * HD;
    } else {
        dst = vcache + (size_t)pos * KVD + (h - H - KVH) * HD;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = f2bf(v[i]);
}

// out BF16 [T, n] = silu(gu[:, :n]) * gu[:, n:] (gu F32 [T, 2n]).
__global__ void silu_mul_kernel(const float* __restrict__ gu, int n, int T, bf16* out) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * n) return;
    const size_t t = i / n, c = i % n;
    const float g = gu[t * 2 * n + c];
    out[i] = f2bf(g / (1.0f + expf(-g)) * gu[t * 2 * n + n + c]);
}

// The drafter's block: row 0 = w * rmsnorm(table[*anchor]), rows 1 .. rows - 1 = mask (F32 [rows, D]).
__global__ void __launch_bounds__(RT) block_embed_kernel(const int* __restrict__ anchor, const bf16* __restrict__ table, const bf16* __restrict__ w,
                                                         const bf16* __restrict__ mask, float* x) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float red[32];
    const int r = blockIdx.x;
    if (r > 0) {
#pragma unroll
        for (int i = 0; i < RPT; ++i) x[(size_t)r * D + threadIdx.x + RT * i] = bf2f(mask[threadIdx.x + RT * i]);
        return;
    }
    const bf16* row = table + (size_t)*anchor * D;
    float v[RPT], ss = 0.0f;
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
        v[i] = bf2f(row[threadIdx.x + RT * i]);
        ss += v[i] * v[i];
    }
    const float inv = rsqrtf(block_sum(ss, red) / D + EPS);
#pragma unroll
    for (int i = 0; i < RPT; ++i) x[threadIdx.x + RT * i] = rb(bf2f(w[threadIdx.x + RT * i]) * (v[i] * inv));
}

// One Markov step: raw = ctx + bias + W [state, w1[prev]] (W [768, 512]); state' = gate * state + (1 -
// gate) * tanh(cand), latent = tanh(out), each op rounded to BF16 as the reference. Block b, warp w: channel
// c = 8 b + w (rows c, 256 + c, 512 + c).
constexpr int MR = 256;
__global__ void __launch_bounds__(256) markov_step_kernel(const bf16* __restrict__ w, const float* __restrict__ ctx, const bf16* __restrict__ bias,
                                                          const bf16* state_in, const bf16* __restrict__ w1, const int* prev, bf16* state_out,
                                                          bf16* latent) {
    pdl_launch_dependents();
    pdl_wait();
    const int lane = threadIdx.x & 31, c = blockIdx.x * 8 + (threadIdx.x >> 5);
    // prev and the state are rewritten at the same addresses between steps: read once per block from L2
    // (an SM's non-coherent L1 may still hold an earlier step's line).
    __shared__ uint4 st_s[MR / 8];
    __shared__ int prev_s;
    if (threadIdx.x < MR / 8) st_s[threadIdx.x] = __ldcg(reinterpret_cast<const uint4*>(state_in) + threadIdx.x);
    if (threadIdx.x == 0) prev_s = __ldcg(prev);
    __syncthreads();
    const bf16* emb = w1 + (size_t)prev_s * MR;
    const bf16* st = reinterpret_cast<const bf16*>(st_s);
    float in[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        in[i] = bf2f(st[lane * 8 + i]);
        in[8 + i] = bf2f(emb[lane * 8 + i]);
    }
    float raw[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const bf16* row = w + (size_t)(g * MR + c) * 2 * MR;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc += bf2f(row[lane * 8 + i]) * in[i] + bf2f(row[MR + lane * 8 + i]) * in[8 + i];
        raw[g] = rb(warp_sum(acc) + ctx[g * MR + c] + bf2f(bias[g * MR + c]));
    }
    if (lane == 0) {
        const float gate = rb(1.0f / (1.0f + expf(-raw[0])));
        const float cand = rb(tanhf(raw[1]));
        const float keep = rb(gate * bf2f(st[c]));
        const float update = rb(rb(1.0f - gate) * cand);
        state_out[c] = f2bf(keep + update);
        latent[c] = f2bf(tanhf(raw[2]));
    }
}

// Logits of a Markov step: cap * tanh(bf16(base + fold_b + fold_w . latent) / cap) over V rows, reduced to
// one (max, index) per block (block 256 threads, warp per row, 64 rows per block).
constexpr int MV_ROWS = 64;
__global__ void __launch_bounds__(256) markov_vocab_kernel(const bf16* __restrict__ fold_w, const float* __restrict__ fold_b, const float* __restrict__ base,
                                                           const bf16* latent, int V, float cap, float* best_v, int* best_i) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ uint4 lat_s[MR / 8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    // The latent is rewritten at the same address every step: one L2 read per block (not per thread,
    // which would serialize on its few L2 lines), past the possibly stale L1.
    if (threadIdx.x < MR / 8) lat_s[threadIdx.x] = __ldcg(reinterpret_cast<const uint4*>(latent) + threadIdx.x);
    __syncthreads();
    float lat[8];
    {
        const uint4 lv = lat_s[lane];
        const bf16* lp = reinterpret_cast<const bf16*>(&lv);
#pragma unroll
        for (int i = 0; i < 8; ++i) lat[i] = bf2f(lp[i]);
    }
    float mv = -INFINITY;
    int mi = 0;
    // Each warp takes 8 consecutive rows at once (8 loads in flight per lane).
    const int r0 = blockIdx.x * MV_ROWS + warp * 8;
    uint4 wv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) wv[j] = r0 + j < V ? __ldg(reinterpret_cast<const uint4*>(fold_w + (size_t)(r0 + j) * MR + lane * 8)) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const bf16* wp = reinterpret_cast<const bf16*>(&wv[j]);
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc += bf2f(wp[i]) * lat[i];
        acc = warp_sum(acc);
        const int r = r0 + j;
        if (r < V) {
            const float x = softcap(rb(base[r] + fold_b[r] + acc), cap);
            if (x > mv) {
                mv = x;
                mi = r;
            }
        }
    }
    if (lane == 0) {
        bv[warp] = mv;
        bi[warp] = mi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = bv[0];
        int idx = bi[0];
        for (int w = 1; w < 8; ++w)
            if (bv[w] > v || (bv[w] == v && bi[w] < idx)) {
                v = bv[w];
                idx = bi[w];
            }
        best_v[blockIdx.x] = v;
        best_i[blockIdx.x] = idx;
    }
}

// The step's pick: the best (max, lowest index) of the blocks' candidates into *out and *prev.
__global__ void __launch_bounds__(256) markov_pick_kernel(const float* best_v, const int* best_i, int n, int* out, int* prev) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float sv[256];
    __shared__ int si[256];
    float v = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += 256) {
        const float bv = __ldcg(best_v + i);
        const int bi = __ldcg(best_i + i);
        if (bv > v || (bv == v && bi < idx)) {
            v = bv;
            idx = bi;
        }
    }
    sv[threadIdx.x] = v;
    si[threadIdx.x] = idx;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float ov = sv[threadIdx.x + s];
            const int oi = si[threadIdx.x + s];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *out = si[0];
        *prev = si[0];
    }
}

int sm_count() {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    return sms;
}

unsigned blocks_for(size_t n, int threads) { return (unsigned)((n + threads - 1) / threads); }

}  // namespace

extern "C" {

cudaError_t quench6k_ink_embed(const int* tokens, int T, const void* table, const void* w, float* out, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(embed_kernel, T, RT, 0, st, tokens, static_cast<const bf16*>(table), static_cast<const bf16*>(w), out));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_ink_rms(int sets, const float* const* x, const void* const* w, void* const* out, const int* ldo, int T, cudaStream_t st) {
    if (T == 0 || sets == 0) return cudaSuccess;
    if (sets > MAX_SETS) return cudaErrorInvalidValue;
    RmsSets r{};
    for (int i = 0; i < sets; ++i) {
        r.x[i] = x[i];
        r.w[i] = static_cast<const bf16*>(w[i]);
        r.out[i] = static_cast<bf16*>(out[i]);
        r.ldo[i] = ldo[i];
    }
    SQ_CHECK_LAUNCH(sq_launch(rms_kernel, dim3(T, sets), RT, 0, st, r));
    SQ_RETURN_LAUNCH();
}

// out (null: none) F32 [nq][T, D]; norm_o[q] (null: none) BF16 rows at stride ldo = norm_w[q] * rmsnorm(out q).
cudaError_t quench6k_ink_attn_res(const float* const* src, int ns, const float* queries, int nq, int T, float* out, const void* const* norm_w,
                                  void* const* norm_o, int ldo, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    if (ns < 1 || ns > MAX_SRC || nq < 1 || nq > MAX_Q) return cudaErrorInvalidValue;
    Srcs s{};
    for (int i = 0; i < ns; ++i) s.p[i] = src[i];
    ResNorms n{};
    n.ldo = ldo;
    for (int i = 0; i < nq && norm_o; ++i) {
        n.w[i] = static_cast<const bf16*>(norm_w[i]);
        n.o[i] = static_cast<bf16*>(norm_o[i]);
    }
    SQ_CHECK_LAUNCH(sq_launch(attn_res_kernel, T, ART, 0, st, s, ns, queries, nq, out, (size_t)T * D, n));
    SQ_RETURN_LAUNCH();
}

// Message gates of a group of `n` (hs, m, down, fg, out per gate); cat BF16 [n][T][2D], z BF16 [n][T][64].
cudaError_t quench6k_ink_msg_gates(int n, const float* const* hs, const float* const* m, const void* const* down, const void* const* fg,
                                   float* const* out, int T, void* cat, void* z, cudaStream_t st) {
    if (T == 0 || n == 0) return cudaSuccess;
    if (n > MAX_Q) return cudaErrorInvalidValue;
    MsgGates g{};
    RmsSets r{};
    for (int i = 0; i < n; ++i) {
        g.hs[i] = hs[i];
        g.m[i] = m[i];
        g.down[i] = static_cast<const bf16*>(down[i]);
        g.fg[i] = static_cast<const bf16*>(fg[i]);
        g.out[i] = out[i];
        bf16* c = static_cast<bf16*>(cat) + (size_t)i * T * 2 * D;
        r.x[2 * i] = hs[i];
        r.out[2 * i] = c;
        r.x[2 * i + 1] = m[i];
        r.out[2 * i + 1] = c + D;
        r.ldo[2 * i] = r.ldo[2 * i + 1] = 2 * D;
    }
    SQ_CHECK_LAUNCH(sq_launch(rms_kernel, dim3(T, 2 * n), RT, 0, st, r));
    SQ_CHECK_LAUNCH(sq_launch(msg_down_kernel, dim3(n, RANK / 8, T), 256, 0, st, g, static_cast<const bf16*>(cat), T, static_cast<bf16*>(z)));
    SQ_CHECK_LAUNCH(sq_launch(msg_apply_kernel, dim3(n * T, (D + 255) / 256), 256, 0, st, g, static_cast<const bf16*>(z), T));
    SQ_RETURN_LAUNCH();
}

// The query side (lin_q [T, ldq]: q | gate | relative states | sinks) and the key and value convolutions
// (lin_k / lin_v [T, ldk]: projections, then the full layers' per-token taps): conv static taps [KVD, 4]
// (sliding layers) or null with static_w [4, KVD] (full layers).
cudaError_t quench6k_ink_qkv_prep(const float* lin_q, int ldq, const void* qscale, const void* rscale, float qclip, float rclip, int full,
                                  void* q, float* rel, float* sink, const float* lin_k, const float* lin_v, int ldk, void* kring, void* vring,
                                  const void* kconv, const void* vconv, const void* kstatic, const void* vstatic, void* kcache, void* vcache,
                                  int ring_rows, int cache_rows, const int* start, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    const QPrep qp{lin_q, ldq, static_cast<const bf16*>(qscale), static_cast<const bf16*>(rscale), qclip, rclip, full, static_cast<bf16*>(q),
                   rel, sink};
    const bool dyn = kconv == nullptr;
    const KvConv k{lin_k, ldk, static_cast<bf16*>(kring), static_cast<const bf16*>(kconv), static_cast<const bf16*>(kstatic),
                   dyn ? lin_k + KVD : nullptr, static_cast<bf16*>(kcache)};
    const KvConv v{lin_v, ldk, static_cast<bf16*>(vring), static_cast<const bf16*>(vconv), static_cast<const bf16*>(vstatic),
                   dyn ? lin_v + KVD : nullptr, static_cast<bf16*>(vcache)};
    SQ_CHECK_LAUNCH(sq_launch(qkv_prep_kernel, dim3(T, 3), 256, 0, st, qp, k, v, ring_rows, cache_rows, qclip, start));
    SQ_RETURN_LAUNCH();
}

// x from src F32 [T, D], or (cy non-null) the combine sum_k cw[t, k] cy[t * 8 + k]; see hidden_conv_kernel.
cudaError_t quench6k_ink_hidden_conv(const float* src, const float* cy, const float* cw, void* ring, int ring_rows, const void* conv,
                                     const float* base, const int* start, int T, float* out, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(hidden_conv_kernel, blocks_for((size_t)T * D, 256), 256, 0, st, src, cy, cw, static_cast<bf16*>(ring), ring_rows,
                              static_cast<const bf16*>(conv), base, start, T, out));
    SQ_RETURN_LAUNCH();
}

size_t quench6k_ink_attn_part_floats(int T, int splits) { return (size_t)T * H * splits * (2 + HD); }

cudaError_t quench6k_ink_attention(const void* q, const float* rel, const float* sink, const void* kcache, const void* vcache, int cache_rows,
                                   const void* proj, int window, int block, const int* start, int T, int splits, int min_keys,
                                   const float* lin, int ld, int gate_off, float* part, void* out, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    const AttnArgs a{static_cast<const bf16*>(q), rel, static_cast<const bf16*>(kcache), static_cast<const bf16*>(vcache), cache_rows,
                     static_cast<const bf16*>(proj), window, block, min_keys, start, part};
    if (rel && sink && lin && block == 0 && T > 16) {
        static bool configured = false;
        if (!configured) {
            const cudaError_t e = cudaFuncSetAttribute(attn_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM);
            if (e != cudaSuccess) return e;
            configured = true;
        }
        SQ_CHECK_LAUNCH(sq_launch(attn_tc_kernel, dim3((T + TC_Q - 1) / TC_Q, KVH), TC_W * 32, TC_SMEM, st, a, T, sink, lin, ld, gate_off,
                                  static_cast<bf16*>(out)));
        SQ_RETURN_LAUNCH();
    }
    SQ_CHECK_LAUNCH(sq_launch(attn_kernel, dim3(T, KVH, splits), AT_THREADS, 0, st, a));
    SQ_CHECK_LAUNCH(sq_launch(attn_merge_kernel, dim3(T, H), HD, 0, st, part, splits, sink, lin, ld, gate_off, static_cast<bf16*>(out)));
    SQ_RETURN_LAUNCH();
}

// order (non-null, T <= 16): also the pairs' order by expert, in the same launch.
cudaError_t quench6k_ink_route(const float* logits, const float* bias, float scale, int T, int* sel_e, float* sel_w, int* order, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    if (order) {
        if (T > 16) return cudaErrorInvalidValue;
        SQ_CHECK_LAUNCH(sq_launch(route_order_kernel, 1, 1024, 0, st, logits, bias, scale, T, sel_e, sel_w, order));
        SQ_RETURN_LAUNCH();
    }
    SQ_CHECK_LAUNCH(sq_launch(route_kernel, T, E, 0, st, logits, bias, scale, sel_e, sel_w));
    SQ_RETURN_LAUNCH();
}

// The routed experts of T tokens: order (pairs in expert order; computed here unless `ordered`), act BF16
// [T * 8, 1152], y F32 [T * 8, D], out F32 [T, D] (null: no combine). fp4: NVFP4 weights with s2, else MXFP8.
cudaError_t quench6k_ink_experts(const void* x, const int* sel_e, const float* sel_w, int fp4, const uint8_t* gu_w, const uint8_t* gu_sc,
                                 const float* gu_s2, const uint8_t* d_w, const uint8_t* d_sc, const float* d_s2, float gcap, float ucap,
                                 int T, int* order, int ordered, void* act, float* y, float* out, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    const int pairs = T * TOPK;
    if (!ordered) SQ_CHECK_LAUNCH(sq_launch(moe_order_kernel, 1, 1024, 0, st, sel_e, pairs, order));
    const ExpertW gu{gu_w, gu_sc, gu_s2}, dn{d_w, d_sc, d_s2};
    const int sms = sm_count();
    const unsigned gb = (unsigned)std::min<size_t>((size_t)pairs * (2 * FF / EX_R) / 8 + 1, (size_t)sms * 8);
    const unsigned db = (unsigned)std::min<size_t>((size_t)pairs * (D / EX_R) / 8 + 1, (size_t)sms * 8);
    if (fp4) {
        SQ_CHECK_LAUNCH(sq_launch(experts_gate_up_kernel<true>, gb, 256, 0, st, static_cast<const bf16*>(x), sel_e, (const int*)order, gu, pairs,
                                  gcap, ucap, static_cast<bf16*>(act)));
        SQ_CHECK_LAUNCH(sq_launch(experts_down_kernel<true>, db, 256, 0, st, static_cast<const bf16*>(act), sel_e, (const int*)order, dn, pairs, y));
    } else {
        SQ_CHECK_LAUNCH(sq_launch(experts_gate_up_kernel<false>, gb, 256, 0, st, static_cast<const bf16*>(x), sel_e, (const int*)order, gu,
                                  pairs, gcap, ucap, static_cast<bf16*>(act)));
        SQ_CHECK_LAUNCH(sq_launch(experts_down_kernel<false>, db, 256, 0, st, static_cast<const bf16*>(act), sel_e, (const int*)order, dn, pairs, y));
    }
    if (out) SQ_CHECK_LAUNCH(sq_launch(moe_combine_kernel, blocks_for((size_t)T * D, 256), 256, 0, st, (const float*)y, sel_w, T, out));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_ink_softcap(float* x, int rows, int n, int ld, float cap, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(softcap_kernel, blocks_for((size_t)rows * n, 256), 256, 0, st, x, rows, n, ld, cap));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_ink_to_bf16_rows(const float* src, int T, void* dst, int ldd, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(to_bf16_rows_kernel, blocks_for((size_t)T * D, 256), 256, 0, st, src, T, static_cast<bf16*>(dst), ldd));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_ink_mix(const void* feats, const int* rows_from, const float* alpha, const void* w, void* out, int ldo, int rows,
                             cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(mix_kernel, rows, RT, 0, st, static_cast<const bf16*>(feats), rows_from, alpha, static_cast<const bf16*>(w),
                              static_cast<bf16*>(out), ldo));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_ink_ctx_kv(const float* kv, const void* knorm, float theta, const int* start, int T, void* kcache, void* vcache,
                                cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(ctx_kv_kernel, T, 256, 0, st, kv, static_cast<const bf16*>(knorm), theta, start, static_cast<bf16*>(kcache),
                              static_cast<bf16*>(vcache)));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_ink_draft_qkv(const float* qkv, const void* qnorm, const void* knorm, float theta, const int* start, int T, void* q,
                                   void* kcache, void* vcache, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(draft_qkv_kernel, dim3(T, H + 2 * KVH), 32, 0, st, qkv, static_cast<const bf16*>(qnorm),
                              static_cast<const bf16*>(knorm), theta, start, static_cast<bf16*>(q), static_cast<bf16*>(kcache),
                              static_cast<bf16*>(vcache)));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_ink_silu_mul(const float* gu, int n, int T, void* out, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(silu_mul_kernel, blocks_for((size_t)T * n, 256), 256, 0, st, gu, n, T, static_cast<bf16*>(out)));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_ink_block_embed(const int* anchor, const void* table, const void* w, const void* mask, int rows, float* x, cudaStream_t st) {
    SQ_CHECK_LAUNCH(sq_launch(block_embed_kernel, rows, RT, 0, st, anchor, static_cast<const bf16*>(table), static_cast<const bf16*>(w),
                              static_cast<const bf16*>(mask), x));
    SQ_RETURN_LAUNCH();
}

size_t quench6k_ink_markov_blocks(int V) { return (size_t)(V + MV_ROWS - 1) / MV_ROWS; }

// One Markov step (see markov_step_kernel / markov_vocab_kernel): writes the pick to *out and *prev.
cudaError_t quench6k_ink_markov(const void* w, const float* ctx, const void* bias, const void* state_in, const void* w1, int* prev, void* state_out,
                                void* latent, const void* fold_w, const float* fold_b, const float* base, int V, float cap, float* best_v,
                                int* best_i, int* out, cudaStream_t st) {
    SQ_CHECK_LAUNCH(sq_launch(markov_step_kernel, MR / 8, 256, 0, st, static_cast<const bf16*>(w), ctx, static_cast<const bf16*>(bias),
                              static_cast<const bf16*>(state_in), static_cast<const bf16*>(w1), (const int*)prev, static_cast<bf16*>(state_out),
                              static_cast<bf16*>(latent)));
    const int nb = (V + MV_ROWS - 1) / MV_ROWS;
    SQ_CHECK_LAUNCH(sq_launch(markov_vocab_kernel, nb, 256, 0, st, static_cast<const bf16*>(fold_w), fold_b, base,
                              static_cast<const bf16*>(latent), V, cap, best_v, best_i));
    SQ_CHECK_LAUNCH(sq_launch(markov_pick_kernel, 1, 256, 0, st, (const float*)best_v, (const int*)best_i, nb, out, prev));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_ink_moe_plan(const int* sel_e, int T, int bm, int* offsets, int* tile_expert, int* tile_row, int* num_tiles, int* pair_row,
                                  cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(moe_plan_kernel, 1, 1024, 0, st, sel_e, T * TOPK, bm, offsets, tile_expert, tile_row, num_tiles, pair_row));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_ink_moe_gather(const void* x, const int* pair_row, int T, void* xs, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(moe_gather_kernel, T * TOPK, 128, 0, st, static_cast<const bf16*>(x), pair_row, static_cast<bf16*>(xs)));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_ink_moe_act(const float* y, int rows, float gcap, float ucap, void* act, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(moe_act_kernel, blocks_for((size_t)rows * FF, 256), 256, 0, st, y, rows, gcap, ucap, static_cast<bf16*>(act)));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_ink_moe_combine_rows(const float* y, const int* pair_row, const float* sel_w, int T, float* out, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(moe_combine_rows_kernel, blocks_for((size_t)T * D, 256), 256, 0, st, y, pair_row, sel_w, T, out));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_ink_mx8_grouped_gemm(const void* xs, const uint8_t* w, const uint8_t* sc, const int* offsets, const int* tile_expert,
                                          const int* tile_row, const int* num_tiles, int max_tiles, float* y, int N, int K, cudaStream_t st) {
    if (K % MXG_BK != 0) return cudaErrorInvalidValue;
    if (max_tiles == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(mx8_grouped_gemm_kernel, dim3((N + MXG_BN - 1) / MXG_BN, max_tiles), 256, 0, st, static_cast<const bf16*>(xs), w,
                              sc, offsets, tile_expert, tile_row, num_tiles, y, N, K));
    SQ_RETURN_LAUNCH();
}

}  // extern "C"
