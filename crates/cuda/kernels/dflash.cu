// DFlash 2 block drafter (incoai/Qwen3.8-Flash-Next-DFlash2): five dense layers that draft a block of 8
// positions ([x, mask x 7]) in one pass, attending to K/V of the target's features at earlier positions
// (a ring of the last 2048), with two-tap dynamic convolutions around each sublayer and a candidate
// selector that walks one path through the top 16 tokens of each position.
//
// The drafter header (int32): [LO] first position held by the ring, [CTX] position of the first context
// row of this step, [BLK] position of the block's first token, [TOK ..] the block's tokens.

#include "common.cuh"

namespace {

constexpr int D = 2560, H = 4 * D, TAPS = 5, BLOCK = 8, QH = 32, KVH = 8, HD = 128, RING = 2048, WINDOW = 2048;
constexpr int GROUP = 16, RANK = 256, CAND = 16, DRAFTS = BLOCK - 1;
constexpr int H_LO = 0, H_CTX = 1, H_BLK = 2, H_TOK = 3;
constexpr float EPS = 1e-6f;

// out[r, tap, :] = mean over the 4 residual streams of h[r] (the completed output of a target layer).
__global__ void tap_kernel(const float* __restrict__ h, __nv_bfloat16* __restrict__ out, int tap) {
    pdl_launch_dependents();
    pdl_wait();
    const int r = blockIdx.x;
    const float* row = h + (size_t)r * H;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        out[((size_t)r * TAPS + tap) * D + d] = f2bf(0.25f * (row[d] + row[D + d] + row[2 * D + d] + row[3 * D + d]));
}

// Qwen3 RMSNorm (weight w, not 1 + w): out[r] = w * bf16(x[r] / rms(x[r])).
__global__ void rmsnorm_kernel(const float* __restrict__ x, int ldx, const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out, int ldo) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const float* row = x + (size_t)blockIdx.x * ldx;
    float ss = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) ss += row[d] * row[d];
    const float r = rsqrtf(block_sum(ss, scratch) / D + EPS);
    for (int d = threadIdx.x; d < D; d += blockDim.x) out[(size_t)blockIdx.x * ldo + d] = f2bf(bf2f(w[d]) * bf2f(f2bf(row[d] * r)));
}

// The block's token embeddings: h[i, :] = table[tokens[i]] (FP32 rows of D).
__global__ void embed_kernel(const int* __restrict__ tokens, const __nv_bfloat16* __restrict__ table, float* __restrict__ h) {
    pdl_launch_dependents();
    pdl_wait();
    const __nv_bfloat16* row = table + (size_t)tokens[blockIdx.x] * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) h[(size_t)blockIdx.x * D + d] = bf2f(row[d]);
}

// Positions of a drafting step from the round control: c context rows end at the next token's position,
// where the block starts with that token followed by mask tokens.
__global__ void header_kernel(const int* __restrict__ p, const int* __restrict__ x, int c, int* hdr, int mask) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = threadIdx.x;
    if (i == 0) {
        hdr[H_CTX] = *p - c;
        hdr[H_BLK] = *p;
    }
    if (i < BLOCK) hdr[H_TOK + i] = i == 0 ? *x : mask;
}

// Two-tap dynamic convolution over the block rows: y[t, c] = sum_o (base[o, c] + dyn[t, o, c / 16]) x[t - o, c]
// (x[-1] = 0), dyn[t, o, g] at dyn + t * ld_dyn + o * D / 16 + g. ADD: y is FP32 and accumulates.
template <bool IN_BF16, bool ADD>
__global__ void conv_kernel(const void* __restrict__ x, const float* __restrict__ dyn, int ld_dyn, const __nv_bfloat16* __restrict__ base, void* y) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x;
    auto xv = [&](int r, int c) {
        return IN_BF16 ? bf2f(static_cast<const __nv_bfloat16*>(x)[(size_t)r * D + c]) : static_cast<const float*>(x)[(size_t)r * D + c];
    };
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int o = 0; o < 2; ++o) {
            if (t - o < 0) continue;
            const float k = bf2f(base[o * D + c]) + dyn[(size_t)t * ld_dyn + o * (D / GROUP) + c / GROUP];
            v += k * xv(t - o, c);
        }
        if (ADD) static_cast<float*>(y)[(size_t)t * D + c] += v;
        else static_cast<__nv_bfloat16*>(y)[(size_t)t * D + c] = f2bf(v);
    }
}

SQ_DEVICE int head_pos(const int* pos, int add) { return (pos != nullptr ? *pos : 0) + add; }

// Per-head RMSNorm and rotary embedding (default RoPE, theta 10000, rotate-half) of projected rows
// [q (nq heads) | k | v] at column offsets; row r is at position *pos + add + r. Queries go to q (FP32
// [rows, 32, 128]); keys and values to kc / vc (BF16 [slot, 8, 128]), slot = position % RING with
// `ring`, else r.
__global__ void qkv_kernel(const float* __restrict__ x, int ld, int q_off, int k_off, int v_off, const __nv_bfloat16* __restrict__ q_norm,
                           const __nv_bfloat16* __restrict__ k_norm, const int* pos_ptr, int pos_add, int nq, float* __restrict__ q,
                           __nv_bfloat16* __restrict__ kc, __nv_bfloat16* __restrict__ vc, int ring) {
    pdl_launch_dependents();
    pdl_wait();
    const int r = blockIdx.x, head = blockIdx.y, d = threadIdx.x;  // 128 threads
    const int pos = head_pos(pos_ptr, pos_add) + r;
    const bool is_q = head < nq;
    const int kv = head - nq;
    const float* src = x + (size_t)r * ld + (is_q ? q_off + head * HD : k_off + kv * HD);
    __shared__ float vals[HD];
    __shared__ float scratch[32];
    const float v = src[d];
    const float rr = rsqrtf(block_sum(v * v, scratch) / HD + EPS);
    vals[d] = bf2f((is_q ? q_norm : k_norm)[d]) * bf2f(f2bf(v * rr));
    __syncthreads();
    const int i = d % (HD / 2);
    const float inv_freq = 1.0f / powf(10000.0f, (float)(2 * i) / (float)HD);
    float sn, cs;
    sincosf((float)pos * inv_freq, &sn, &cs);
    const float other = d < HD / 2 ? -vals[d + HD / 2] : vals[d - HD / 2];
    const float out = vals[d] * cs + other * sn;
    if (is_q) {
        q[((size_t)r * QH + head) * HD + d] = out;
        return;
    }
    const size_t slot = ring ? (size_t)(pos % RING) : (size_t)r;
    kc[(slot * KVH + kv) * HD + d] = f2bf(out);
    vc[(slot * KVH + kv) * HD + d] = f2bf(x[(size_t)r * ld + v_off + kv * HD + d]);
}

// Block attention, split over key ranges. Keys: ring positions [max(LO, BLK - WINDOW + 1), BLK) then the
// block's 8 keys; block row i (position BLK + i) sees ring keys after BLK + i - WINDOW and every block key
// (the drafter is not causal). One CTA per (KV head, split) covers that head's 4 query heads x 8 rows
// (row rho = j * 8 + i: query head g * 4 + j at block row i), register-tiled: each thread scores 2 rows x 2
// keys of a tile and accumulates 4 rows x 8 dims of the output.
constexpr int SPLITS = 64, TILE = 16, ROWS = 4 * BLOCK, LDS = HD + 4, ATT_THREADS = 128;
__global__ void __launch_bounds__(ATT_THREADS) attention_kernel(const float* __restrict__ q, const __nv_bfloat16* __restrict__ rk, const __nv_bfloat16* __restrict__ rv,
                                                        const __nv_bfloat16* __restrict__ bk, const __nv_bfloat16* __restrict__ bv, const int* __restrict__ hdr,
                                                        float* __restrict__ part) {
    pdl_launch_dependents();
    pdl_wait();
    const int g = blockIdx.x, split = blockIdx.y, tid = threadIdx.x;
    const int blk = hdr[H_BLK];
    const int first = max(hdr[H_LO], blk - WINDOW + 1);
    const int ring_keys = max(0, blk - first);
    const int keys = ring_keys + BLOCK;
    const int chunk = (keys + SPLITS - 1) / SPLITS;
    const int k0 = split * chunk, k1 = min(keys, k0 + chunk);
    __shared__ __align__(16) float qs[ROWS][LDS];
    __shared__ __align__(16) float ks[TILE][LDS];
    __shared__ __align__(16) float vs[TILE][LDS];
    __shared__ float sc[ROWS][TILE + 1];
    const float scale = rsqrtf((float)HD);
    for (int e = tid; e < ROWS * HD / 4; e += ATT_THREADS) {
        const int rho = e / (HD / 4), d = (e % (HD / 4)) * 4, j = rho / BLOCK, i = rho % BLOCK;
        const float4 v = *reinterpret_cast<const float4*>(q + ((size_t)i * QH + g * 4 + j) * HD + d);
        *reinterpret_cast<float4*>(&qs[rho][d]) = make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
    }
    const int sr = (tid / 8) * 2, sk = (tid % 8) * 2;    // scoring: rows sr, sr + 1; keys sk, sk + 1
    const int pr = (tid / 16) * 4, pd = (tid % 16) * 8;  // output: rows pr .. pr + 3, dims pd .. pd + 7
    float m[4], l[4], acc[4][8];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[r][e] = 0.0f;
    }
    for (int t0 = k0; t0 < k1; t0 += TILE) {
        __syncthreads();
        // K and V tiles: 16 keys x 16 chunks of 8 dims, two of each per thread.
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int e = tid + h * ATT_THREADS, kk = e / 16, d0 = (e % 16) * 8, key = t0 + kk;
            uint4 kq = make_uint4(0, 0, 0, 0), vq = kq;
            if (key < k1) {
                const __nv_bfloat16 *kp, *vp;
                if (key < ring_keys) {
                    const size_t slot = (size_t)((first + key) % RING);
                    kp = rk + (slot * KVH + g) * HD;
                    vp = rv + (slot * KVH + g) * HD;
                } else {
                    kp = bk + ((size_t)(key - ring_keys) * KVH + g) * HD;
                    vp = bv + ((size_t)(key - ring_keys) * KVH + g) * HD;
                }
                kq = *reinterpret_cast<const uint4*>(kp + d0);
                vq = *reinterpret_cast<const uint4*>(vp + d0);
            }
            const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&kq);
            const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&vq);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 kf = __bfloat1622float2(kb[j]), vf = __bfloat1622float2(vb[j]);
                ks[kk][d0 + 2 * j] = kf.x;
                ks[kk][d0 + 2 * j + 1] = kf.y;
                vs[kk][d0 + 2 * j] = vf.x;
                vs[kk][d0 + 2 * j + 1] = vf.y;
            }
        }
        __syncthreads();
        float s[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
#pragma unroll 8
        for (int d = 0; d < HD; d += 4) {
            const float4 a0 = *reinterpret_cast<const float4*>(&qs[sr][d]), a1 = *reinterpret_cast<const float4*>(&qs[sr + 1][d]);
            const float4 b0 = *reinterpret_cast<const float4*>(&ks[sk][d]), b1 = *reinterpret_cast<const float4*>(&ks[sk + 1][d]);
            s[0][0] += a0.x * b0.x + a0.y * b0.y + a0.z * b0.z + a0.w * b0.w;
            s[0][1] += a0.x * b1.x + a0.y * b1.y + a0.z * b1.z + a0.w * b1.w;
            s[1][0] += a1.x * b0.x + a1.y * b0.y + a1.z * b0.z + a1.w * b0.w;
            s[1][1] += a1.x * b1.x + a1.y * b1.y + a1.z * b1.z + a1.w * b1.w;
        }
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int key = t0 + sk + c, ri = (sr + a) % BLOCK;
                const bool visible = key < k1 && (key >= ring_keys || first + key > blk + ri - WINDOW);
                sc[sr + a][sk + c] = visible ? s[a][c] : -INFINITY;
            }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            float tmax = -INFINITY;
#pragma unroll
            for (int kk = 0; kk < TILE; ++kk) tmax = fmaxf(tmax, sc[pr + r][kk]);
            const float mn = fmaxf(m[r], tmax);
            if (mn == -INFINITY) continue;
            const float f = __expf(m[r] - mn);
            l[r] *= f;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[r][e] *= f;
            m[r] = mn;
        }
#pragma unroll 4
        for (int kk = 0; kk < TILE; ++kk) {
            const float4 v0 = *reinterpret_cast<const float4*>(&vs[kk][pd]), v1 = *reinterpret_cast<const float4*>(&vs[kk][pd + 4]);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float p = m[r] == -INFINITY ? 0.0f : __expf(sc[pr + r][kk] - m[r]);
                l[r] += p;
                acc[r][0] += p * v0.x;
                acc[r][1] += p * v0.y;
                acc[r][2] += p * v0.z;
                acc[r][3] += p * v0.w;
                acc[r][4] += p * v1.x;
                acc[r][5] += p * v1.y;
                acc[r][6] += p * v1.z;
                acc[r][7] += p * v1.w;
            }
        }
    }
    // part[(g, split, row)]: m, l, o[128]
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        float* out = part + ((size_t)(g * SPLITS + split) * ROWS + pr + r) * (HD + 2);
        if (pd == 0) {
            out[0] = m[r];
            out[1] = l[r];
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) out[2 + pd + e] = acc[r][e];
    }
}

// Combines the splits: att[i, h * 128 + d] (BF16 [8, 4096]).
__global__ void attention_combine_kernel(const float* __restrict__ part, __nv_bfloat16* __restrict__ att) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int g = h / 4, rho = (h % 4) * BLOCK + i;
    auto at = [&](int s) { return part + ((size_t)(g * SPLITS + s) * ROWS + rho) * (HD + 2); };
    float mx = -INFINITY;
    for (int s = 0; s < SPLITS; ++s) mx = fmaxf(mx, at(s)[0]);
    float l = 0.0f, o = 0.0f;
    for (int s = 0; s < SPLITS; ++s) {
        const float* p = at(s);
        if (p[0] == -INFINITY) continue;
        const float w = __expf(p[0] - mx);
        l += w * p[1];
        o += w * p[2 + d];
    }
    att[(size_t)i * QH * HD + h * HD + d] = f2bf(o / l);
}

// The candidate selector over block rows 1 .. 7: candidates ids / unary logits [7, 16], the context gate
// hp [7, 256]. Position i scores candidate b after predecessor a (the previous choice; x for i = 0) as
// unary(b) + <A(a) * hp_i, B(b)>, and takes the best one, or samples one at draft_sampled percent of the
// temperature (recording the distribution in qs[i] for the verification).
__global__ void __launch_bounds__(256) select_kernel(const int* __restrict__ ids, const float* __restrict__ vals, const float* __restrict__ hp,
                                                     const __nv_bfloat16* __restrict__ pred_cb, const __nv_bfloat16* __restrict__ succ_cb, const int* __restrict__ x,
                                                     int* drafts, DraftQ* qs, SampleParams* sp) {
    pdl_launch_dependents();
    pdl_wait();
    const int tid = threadIdx.x, k = tid / 16, part = tid % 16;
    __shared__ float score[CAND];
    __shared__ int pred;
    if (tid == 0) pred = *x;
    const bool sampled = sp != nullptr && sp->draft_sampled > 0 && sp->temperature > 0.0f;
    __syncthreads();
    for (int i = 0; i < DRAFTS; ++i) {
        const int cand = ids[i * CAND + k];
        const __nv_bfloat16* a = pred_cb + (size_t)pred * RANK;
        const __nv_bfloat16* b = succ_cb + (size_t)cand * RANK;
        float s = 0.0f;
#pragma unroll
        for (int e = 0; e < RANK / 16; ++e) {
            const int r = part * (RANK / 16) + e;
            s += bf2f(a[r]) * hp[i * RANK + r] * bf2f(b[r]);
        }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (part == 0) score[k] = vals[i * CAND + k] + s;
        __syncthreads();
        if (tid == 0) {
            int best = 0;
            for (int c = 1; c < CAND; ++c)
                if (score[c] > score[best]) best = c;
            int pick = best;
            if (sampled) {
                const double t = fmax((double)sp->temperature * sp->draft_sampled / 100.0, 1e-4);
                double p[CAND], total = 0.0;
                for (int c = 0; c < CAND; ++c) total += p[c] = exp(((double)score[c] - score[best]) / t);
                double u = sample_uniform(sp) * total;
                pick = CAND - 1;
                for (int c = 0; c < CAND; ++c) {
                    u -= p[c];
                    if (u <= 0.0) {
                        pick = c;
                        break;
                    }
                }
                DraftQ& q = qs[i];
                q.count = CAND;
                for (int c = 0; c < CAND; ++c) {
                    q.ids[c] = ids[i * CAND + c];
                    q.probs[c] = (float)(p[c] / total);
                }
            }
            pred = ids[i * CAND + pick];
            drafts[i] = pred;
        }
        __syncthreads();
    }
}

}  // namespace

extern "C" int quench6k_dflash_attention_floats() { return KVH * SPLITS * ROWS * (HD + 2); }

extern "C" cudaError_t quench6k_dflash_tap(const float* h, int rows, void* out, int tap, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(tap_kernel, rows, 256, 0, st, h, static_cast<__nv_bfloat16*>(out), tap));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_dflash_rmsnorm(const float* x, int ldx, const void* w, void* out, int ldo, int rows, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(rmsnorm_kernel, rows, 256, 0, st, x, ldx, static_cast<const __nv_bfloat16*>(w), static_cast<__nv_bfloat16*>(out), ldo));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_dflash_embed(const int* hdr, const void* table, float* h, cudaStream_t st) {
    SQ_CHECK_LAUNCH(sq_launch(embed_kernel, BLOCK, 256, 0, st, hdr + H_TOK, static_cast<const __nv_bfloat16*>(table), h));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_dflash_header(const int* p, const int* x, int c, int* hdr, int mask, cudaStream_t st) {
    SQ_CHECK_LAUNCH(sq_launch(header_kernel, 1, 32, 0, st, p, x, c, hdr, mask));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_dflash_conv(const void* x, int x_bf16, const float* dyn, int ld_dyn, const void* base, void* y, int add, cudaStream_t st) {
    const auto* b = static_cast<const __nv_bfloat16*>(base);
    if (x_bf16 && !add) SQ_CHECK_LAUNCH(sq_launch(conv_kernel<true, false>, BLOCK, 256, 0, st, x, dyn, ld_dyn, b, y));
    else if (!x_bf16 && add) SQ_CHECK_LAUNCH(sq_launch(conv_kernel<false, true>, BLOCK, 256, 0, st, x, dyn, ld_dyn, b, y));
    else return cudaErrorInvalidValue;
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_dflash_qkv(const float* x, int ld, int q_off, int k_off, int v_off, const void* q_norm, const void* k_norm, const int* pos,
                                           int pos_add, int rows, int nq, float* q, void* kc, void* vc, int ring, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(qkv_kernel, dim3(rows, nq + KVH), HD, 0, st, x, ld, q_off, k_off, v_off, static_cast<const __nv_bfloat16*>(q_norm),
                              static_cast<const __nv_bfloat16*>(k_norm), pos, pos_add, nq, q, static_cast<__nv_bfloat16*>(kc), static_cast<__nv_bfloat16*>(vc), ring));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_dflash_attention(const float* q, const void* rk, const void* rv, const void* bk, const void* bv, const int* hdr, float* part,
                                                 void* att, cudaStream_t st) {
    using B = const __nv_bfloat16*;
    if (rk != nullptr) SQ_CHECK_LAUNCH(sq_launch(attention_kernel, dim3(KVH, SPLITS), ATT_THREADS, 0, st, q, static_cast<B>(rk), static_cast<B>(rv), static_cast<B>(bk), static_cast<B>(bv), hdr, part));
    if (att != nullptr) SQ_CHECK_LAUNCH(sq_launch(attention_combine_kernel, dim3(BLOCK, QH), HD, 0, st, (const float*)part, static_cast<__nv_bfloat16*>(att)));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_dflash_select(const int* ids, const float* vals, const float* hp, const void* pred_cb, const void* succ_cb, const int* x,
                                              int* drafts, void* qs, void* params, cudaStream_t st) {
    SQ_CHECK_LAUNCH(sq_launch(select_kernel, 1, 256, 0, st, ids, vals, hp, static_cast<const __nv_bfloat16*>(pred_cb), static_cast<const __nv_bfloat16*>(succ_cb), x,
                              drafts, (DraftQ*)qs, (SampleParams*)params));
    SQ_RETURN_LAUNCH();
}
