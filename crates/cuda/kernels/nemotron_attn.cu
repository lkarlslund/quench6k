// Nemotron-H attention: 32 query heads over 2 KV heads, head dim 128, no positional encoding, causal, over
// a per-sequence position-major cache in BF16 or FP8 E4M3 with a scale per (position, KV head) = amax / 448.
//
// Canonical per row (token at position p, KV head g, its 16 query heads as one m16 MMA tile): keys in tiles
// of 64 absolute positions, online softmax within absolute groups of 1024 positions (16 tiles), groups
// folded in order. Scores are BF16 MMAs (FP8 codes convert exactly) times a per-key factor (scale/sqrt(128)
// in base 2, times the key's FP8 scale); P (times the value's FP8 scale) is rounded to BF16 for P.V.
// Decode computes each group in its own CTA (partials folded by attn_combine_kernel); prefill folds them
// in the same order inside one CTA. Both use the same tile function and fold, so a row's bits match.

#include "mma.cuh"
#include "nemotron.cuh"

using namespace nem;

namespace {

constexpr float SCORE_C = 0.08838834764831845f * 1.4426950408889634f;  // 1/sqrt(128) * log2(e)
constexpr int KLD = HD + 8, VLD = ATTN_TILE + 8;

struct KvArgs {
    long long off, ld;
    int fp8;
};

struct TileSmem {
    __align__(16) bf16 k[ATTN_TILE * KLD];
    __align__(16) bf16 vt[HD * VLD];
    float kf[ATTN_TILE], vf[ATTN_TILE];
};

SQ_DEVICE const unsigned char* kv_row(const Seqs& q, int s, const KvArgs& kv, int j) {
    return reinterpret_cast<const unsigned char*>(q.kv[s]) + (size_t)j * kv.ld + kv.off;
}

// Tile of keys j0 .. j0 + 63 of KV head g of sequence s into shared memory: K [key][d], V transposed [d][key]
// (BF16), per-key score factor and value scale; keys past `last` are zeros. Ends with a barrier.
SQ_DEVICE void load_tile(TileSmem& sm, const Seqs& q, int s, const KvArgs& kv, int g, int j0, int last) {
    const int tid = threadIdx.x, nt = blockDim.x;
    if (kv.fp8) {
        for (int i = tid; i < ATTN_TILE * (HD / 16); i += nt) {
            const int r = i / (HD / 16), c = (i % (HD / 16)) * 16, j = j0 + r;
            uint4 kq = make_uint4(0, 0, 0, 0), vq = kq;
            if (j <= last) {
                const unsigned char* row = kv_row(q, s, kv, j);
                kq = __ldcg(reinterpret_cast<const uint4*>(row + g * HD + c));
                vq = __ldcg(reinterpret_cast<const uint4*>(row + KVH * HD + g * HD + c));
            }
            const __nv_fp8x2_storage_t* kb = reinterpret_cast<const __nv_fp8x2_storage_t*>(&kq);
            const __nv_fp8x2_storage_t* vb = reinterpret_cast<const __nv_fp8x2_storage_t*>(&vq);
            __align__(16) bf16 ko[16];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const __half2_raw kh = __nv_cvt_fp8x2_to_halfraw2(kb[e], __NV_E4M3);
                const __half2_raw vh = __nv_cvt_fp8x2_to_halfraw2(vb[e], __NV_E4M3);
                const float2 kf2 = __half22float2(*reinterpret_cast<const __half2*>(&kh));
                const float2 vf2 = __half22float2(*reinterpret_cast<const __half2*>(&vh));
                ko[2 * e] = f2bf(kf2.x), ko[2 * e + 1] = f2bf(kf2.y);
                sm.vt[(c + 2 * e) * VLD + r] = f2bf(vf2.x);
                sm.vt[(c + 2 * e + 1) * VLD + r] = f2bf(vf2.y);
            }
            *reinterpret_cast<uint4*>(&sm.k[r * KLD + c]) = reinterpret_cast<const uint4*>(ko)[0];
            *reinterpret_cast<uint4*>(&sm.k[r * KLD + c + 8]) = reinterpret_cast<const uint4*>(ko)[1];
        }
        for (int r = tid; r < ATTN_TILE; r += nt) {
            const int j = j0 + r;
            float ks = 0.0f, vs = 0.0f;
            if (j <= last) {
                const float* sc = reinterpret_cast<const float*>(kv_row(q, s, kv, j) + 2 * KVH * HD);
                ks = __ldcg(sc + g);
                vs = __ldcg(sc + KVH + g);
            }
            sm.kf[r] = __fmul_rn(ks, SCORE_C);
            sm.vf[r] = vs;
        }
    } else {
        for (int i = tid; i < ATTN_TILE * (HD / 8); i += nt) {
            const int r = i / (HD / 8), c = (i % (HD / 8)) * 8, j = j0 + r;
            uint4 kq = make_uint4(0, 0, 0, 0), vq = kq;
            if (j <= last) {
                const bf16* row = reinterpret_cast<const bf16*>(kv_row(q, s, kv, j));
                kq = __ldcg(reinterpret_cast<const uint4*>(row + g * HD + c));
                vq = __ldcg(reinterpret_cast<const uint4*>(row + KVH * HD + g * HD + c));
            }
            *reinterpret_cast<uint4*>(&sm.k[r * KLD + c]) = kq;
            const bf16* vb = reinterpret_cast<const bf16*>(&vq);
#pragma unroll
            for (int e = 0; e < 8; ++e) sm.vt[(c + e) * VLD + r] = vb[e];
        }
        for (int r = tid; r < ATTN_TILE; r += nt) {
            const bool ok = j0 + r <= last;
            sm.kf[r] = ok ? SCORE_C : 0.0f;
            sm.vf[r] = ok ? 1.0f : 0.0f;
        }
    }
    __syncthreads();
}

// Q fragments (A operand, 16 heads x 128) of row r, KV head g.
SQ_DEVICE void load_q(const bf16* qrow, int g, uint32_t (&qa)[HD / 16][4]) {
    const int lane = threadIdx.x & 31, g8 = lane >> 2, t4 = lane & 3;
    const uint32_t* base = reinterpret_cast<const uint32_t*>(qrow + (size_t)g * QPK * HD);
#pragma unroll
    for (int kk = 0; kk < HD / 16; ++kk) {
        const int c = kk * 16 + t4 * 2;
        qa[kk][0] = __ldcg(base + (g8 * HD + c) / 2);
        qa[kk][1] = __ldcg(base + ((g8 + 8) * HD + c) / 2);
        qa[kk][2] = __ldcg(base + (g8 * HD + c + 8) / 2);
        qa[kk][3] = __ldcg(base + ((g8 + 8) * HD + c + 8) / 2);
    }
}

// Online softmax state of a warp's 16 rows: this thread's rows g8 and g8 + 8; acc over NT 8-wide dim tiles
// starting at dim d0.
template <int NT>
struct Partial {
    float m[2], l[2];
    float acc[NT][4];
    SQ_DEVICE void reset() {
        m[0] = m[1] = -INFINITY;
        l[0] = l[1] = 0.0f;
#pragma unroll
        for (int i = 0; i < NT; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    }
};

SQ_DEVICE uint32_t pack_bf16(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// One key tile (keys j0 .. j0 + 63, those past `pos` masked) into the partial: the same arithmetic and
// order for every caller.
template <int NT>
SQ_DEVICE void attn_tile(const TileSmem& sm, const uint32_t (&qa)[HD / 16][4], int j0, int pos, int d0, Partial<NT>& st) {
    const int lane = threadIdx.x & 31, g8 = lane >> 2, t4 = lane & 3;
    float sc[ATTN_TILE / 8][4];
#pragma unroll
    for (int nt = 0; nt < ATTN_TILE / 8; ++nt) sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.0f;
#pragma unroll
    for (int nt = 0; nt < ATTN_TILE / 8; ++nt)
#pragma unroll
        for (int kk = 0; kk < HD / 16; ++kk) {
            const bf16* kp = &sm.k[(nt * 8 + g8) * KLD + kk * 16 + t4 * 2];
            const uint32_t b[2] = {*reinterpret_cast<const uint32_t*>(kp), *reinterpret_cast<const uint32_t*>(kp + 8)};
            mma_bf16(sc[nt], qa[kk], b);
        }
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nt = 0; nt < ATTN_TILE / 8; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int kj = nt * 8 + t4 * 2 + (c & 1);
            const float v = j0 + kj <= pos ? __fmul_rn(sc[nt][c], sm.kf[kj]) : -INFINITY;
            sc[nt][c] = v;
            mx[c >> 1] = fmaxf(mx[c >> 1], v);
        }
    float alpha[2], mn[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffff, mx[h], 1));
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffff, mx[h], 2));
        mn[h] = fmaxf(st.m[h], mx[h]);
        alpha[h] = exp2f(st.m[h] - mn[h]);
        st.m[h] = mn[h];
    }
    float sum[2] = {0.0f, 0.0f};
    uint32_t pa[ATTN_TILE / 16][4];
#pragma unroll
    for (int nt = 0; nt < ATTN_TILE / 8; ++nt) {
        float pv[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float p = exp2f(sc[nt][c] - mn[c >> 1]);
            sum[c >> 1] += p;
            pv[c] = __fmul_rn(p, sm.vf[nt * 8 + t4 * 2 + (c & 1)]);
        }
        pa[nt >> 1][(nt & 1) * 2] = pack_bf16(pv[0], pv[1]);
        pa[nt >> 1][(nt & 1) * 2 + 1] = pack_bf16(pv[2], pv[3]);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        sum[h] += __shfl_xor_sync(0xffffffff, sum[h], 1);
        sum[h] += __shfl_xor_sync(0xffffffff, sum[h], 2);
        st.l[h] = __fadd_rn(__fmul_rn(st.l[h], alpha[h]), sum[h]);
    }
#pragma unroll
    for (int dn = 0; dn < NT; ++dn) {
        st.acc[dn][0] = __fmul_rn(st.acc[dn][0], alpha[0]);
        st.acc[dn][1] = __fmul_rn(st.acc[dn][1], alpha[0]);
        st.acc[dn][2] = __fmul_rn(st.acc[dn][2], alpha[1]);
        st.acc[dn][3] = __fmul_rn(st.acc[dn][3], alpha[1]);
#pragma unroll
        for (int kk = 0; kk < ATTN_TILE / 16; ++kk) {
            const bf16* vp = &sm.vt[(d0 + dn * 8 + g8) * VLD + kk * 16 + t4 * 2];
            const uint32_t b[2] = {*reinterpret_cast<const uint32_t*>(vp), *reinterpret_cast<const uint32_t*>(vp + 8)};
            mma_bf16(st.acc[dn], pa[kk], b);
        }
    }
}

// Fold of a group partial (mg, lg, ag) into the running total (mt, lt, at), the same in both kernels.
SQ_DEVICE void fold_ml(float& mt, float& lt, float mg, float lg, float& a, float& b) {
    const float m = fmaxf(mt, mg);
    a = exp2f(mt - m);
    b = exp2f(mg - m);
    lt = __fadd_rn(__fmul_rn(lt, a), __fmul_rn(lg, b));
    mt = m;
}
SQ_DEVICE float fold_acc(float at, float ag, float a, float b) { return __fadd_rn(__fmul_rn(at, a), __fmul_rn(ag, b)); }

// Grid (rows, 2): q = bf16(qkv[q_col ..]) per row; k, v (BF16-rounded) into the cache at the row's position.
__global__ void __launch_bounds__(HD) attn_prep_kernel(const float* qkv, int ld, int q_col, int k_col, int v_col, const Seqs* sq, KvArgs kv,
                                                       bf16* qout) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ Seqs q;
    __shared__ float scratch[32];
    load_seqs(q, sq);
    const int r = blockIdx.x, part = blockIdx.y, d = threadIdx.x;
    const float* row = qkv + (size_t)r * ld;
    for (int h = part; h < QH; h += 2) qout[(size_t)r * QH * HD + h * HD + d] = f2bf(__ldcg(row + q_col + h * HD + d));
    const int s = seq_of_row(q, r);
    if (s < 0) return;
    const int g = part;  // block y = KV head for the cache part
    const int pos = q.start[s] + r - q.row0[s];
    unsigned char* dst = reinterpret_cast<unsigned char*>(q.kv[s]) + (size_t)pos * kv.ld + kv.off;
    const float k = rb(__ldcg(row + k_col + g * HD + d)), v = rb(__ldcg(row + v_col + g * HD + d));
    if (kv.fp8) {
        const float ka = block_max(fabsf(k), scratch), va = block_max(fabsf(v), scratch);
        const float ks = ka > 0.0f ? ka / 448.0f : 1.0f, vs = va > 0.0f ? va / 448.0f : 1.0f;
        dst[g * HD + d] = __nv_fp8_e4m3(k / ks).__x;
        dst[KVH * HD + g * HD + d] = __nv_fp8_e4m3(v / vs).__x;
        if (d == 0) {
            float* sc = reinterpret_cast<float*>(dst + 2 * KVH * HD);
            sc[g] = ks;
            sc[KVH + g] = vs;
        }
    } else {
        bf16* b = reinterpret_cast<bf16*>(dst);
        b[g * HD + d] = f2bf(k);
        b[KVH * HD + g * HD + d] = f2bf(v);
    }
}

constexpr int PART_FLOATS = QPK * HD + 2 * QPK;  // acc [16][128], m [16], l [16]

// Decode: grid (rows, 2, max_groups), 4 warps. Every warp computes the scores of the row's 16 heads; warp w
// accumulates dims 32 w .. 32 w + 31. Writes the group's partial.
__global__ void __launch_bounds__(128) attn_decode_kernel(const bf16* qin, const Seqs* sq, KvArgs kv, int max_groups, float* part) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ Seqs q;
    __shared__ TileSmem sm;
    load_seqs(q, sq);
    const int r = blockIdx.x, g = blockIdx.y, gi = blockIdx.z;
    const int s = seq_of_row(q, r);
    if (s < 0) return;
    const int pos = q.start[s] + r - q.row0[s];
    if (gi * ATTN_GROUP > pos) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g8 = lane >> 2, t4 = lane & 3;
    uint32_t qa[HD / 16][4];
    load_q(qin + (size_t)r * QH * HD, g, qa);
    Partial<4> st;
    st.reset();
    const int t_first = gi * GROUP_TILES, t_last = min(gi * GROUP_TILES + GROUP_TILES - 1, pos / ATTN_TILE);
    for (int t = t_first; t <= t_last; ++t) {
        __syncthreads();
        load_tile(sm, q, s, kv, g, t * ATTN_TILE, pos);
        attn_tile(sm, qa, t * ATTN_TILE, pos, warp * 32, st);
    }
    float* out = part + (((size_t)r * KVH + g) * max_groups + gi) * PART_FLOATS;
#pragma unroll
    for (int dn = 0; dn < 4; ++dn) {
        const int d = warp * 32 + dn * 8 + t4 * 2;
        out[g8 * HD + d] = st.acc[dn][0];
        out[g8 * HD + d + 1] = st.acc[dn][1];
        out[(g8 + 8) * HD + d] = st.acc[dn][2];
        out[(g8 + 8) * HD + d + 1] = st.acc[dn][3];
    }
    if (warp == 0 && t4 == 0) {
        out[QPK * HD + g8] = st.m[0];
        out[QPK * HD + g8 + 8] = st.m[1];
        out[QPK * HD + QPK + g8] = st.l[0];
        out[QPK * HD + QPK + g8 + 8] = st.l[1];
    }
}

// Grid (rows, 2), thread = dim: folds the row's groups in order and normalizes.
__global__ void __launch_bounds__(HD) attn_combine_kernel(const float* part, const Seqs* sq, int max_groups, bf16* out) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ Seqs q;
    load_seqs(q, sq);
    const int r = blockIdx.x, g = blockIdx.y, d = threadIdx.x;
    const int s = seq_of_row(q, r);
    if (s < 0) return;
    const int pos = q.start[s] + r - q.row0[s];
    const int ng = pos / ATTN_GROUP + 1;
    const float* base = part + ((size_t)r * KVH + g) * max_groups * PART_FLOATS;
    for (int h = 0; h < QPK; ++h) {
        float mt = __ldcg(base + QPK * HD + h), lt = __ldcg(base + QPK * HD + QPK + h), at = __ldcg(base + h * HD + d);
        for (int gi = 1; gi < ng; ++gi) {
            const float* p = base + (size_t)gi * PART_FLOATS;
            float a, b;
            fold_ml(mt, lt, __ldcg(p + QPK * HD + h), __ldcg(p + QPK * HD + QPK + h), a, b);
            at = fold_acc(at, __ldcg(p + h * HD + d), a, b);
        }
        out[(size_t)r * QH * HD + (g * QPK + h) * HD + d] = f2bf(__fdiv_rn(at, lt));
    }
}

// Prefill: grid (ceil(rows / 4), 2), 4 warps, warp w = row 4 blockIdx.x + w over all 128 dims. Rows of one
// sequence share the key tiles; groups are folded as they end (the running total in shared memory, each
// thread its own fragment slots).
constexpr int PF_WARPS = 4;
struct PrefillSmem {
    TileSmem tile;
    float total[PF_WARPS][HD / 8 * 4][32];
};

__global__ void __launch_bounds__(PF_WARPS * 32) attn_prefill_kernel(const bf16* qin, int rows, const Seqs* sq, KvArgs kv, bf16* out) {
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    PrefillSmem& sm = *reinterpret_cast<PrefillSmem*>(smem_raw);
    __shared__ Seqs q;
    __shared__ int wseq[PF_WARPS], wpos[PF_WARPS];
    load_seqs(q, sq);
    const int g = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g8 = lane >> 2, t4 = lane & 3;
    const int r = blockIdx.x * PF_WARPS + warp;
    if (lane == 0) {
        const int s = r < rows ? seq_of_row(q, r) : -1;
        wseq[warp] = s;
        wpos[warp] = s >= 0 ? q.start[s] + r - q.row0[s] : -1;
    }
    __syncthreads();
    const int my_s = wseq[warp], pos = wpos[warp];
    uint32_t qa[HD / 16][4];
    if (my_s >= 0) load_q(qin + (size_t)r * QH * HD, g, qa);
    Partial<HD / 8> st;
    float mt[2], lt[2];
    float (*tot)[32] = sm.total[warp];
    for (int u = 0; u < PF_WARPS; ++u) {
        const int su = wseq[u];
        bool first = su >= 0;
        for (int v = 0; v < u; ++v) first = first && wseq[v] != su;
        if (!first) continue;
        int last = -1;
        for (int v = 0; v < PF_WARPS; ++v)
            if (wseq[v] == su) last = max(last, wpos[v]);
        const bool active = my_s == su;
        const int my_last_tile = active ? pos / ATTN_TILE : -1;
        for (int t = 0; t <= last / ATTN_TILE; ++t) {
            __syncthreads();
            load_tile(sm.tile, q, su, kv, g, t * ATTN_TILE, last);
            if (t > my_last_tile) continue;
            if (t % GROUP_TILES == 0) {
                if (t > 0) {
                    // Fold the group that just ended into the total.
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float a, b;
                        const float mg = st.m[h], lg = st.l[h];
                        if (t == GROUP_TILES) {
                            mt[h] = mg, lt[h] = lg, a = 0.0f, b = 1.0f;
                        } else {
                            fold_ml(mt[h], lt[h], mg, lg, a, b);
                        }
#pragma unroll
                        for (int dn = 0; dn < HD / 8; ++dn)
#pragma unroll
                            for (int c = 2 * h; c < 2 * h + 2; ++c) {
                                float& tv = tot[dn * 4 + c][lane];
                                tv = t == GROUP_TILES ? st.acc[dn][c] : fold_acc(tv, st.acc[dn][c], a, b);
                            }
                    }
                }
                st.reset();
            }
            attn_tile(sm.tile, qa, t * ATTN_TILE, pos, 0, st);
        }
        if (active) {
            const int ng = pos / ATTN_GROUP + 1;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float a = 0.0f, b = 1.0f;
                if (ng > 1) fold_ml(mt[h], lt[h], st.m[h], st.l[h], a, b);
                else lt[h] = st.l[h];
#pragma unroll
                for (int dn = 0; dn < HD / 8; ++dn)
#pragma unroll
                    for (int c = 2 * h; c < 2 * h + 2; ++c) {
                        const float at = ng > 1 ? fold_acc(tot[dn * 4 + c][lane], st.acc[dn][c], a, b) : st.acc[dn][c];
                        const int head = g8 + 8 * h, d = dn * 8 + t4 * 2 + (c & 1);
                        out[(size_t)r * QH * HD + (g * QPK + head) * HD + d] = f2bf(__fdiv_rn(at, lt[h]));
                    }
            }
        }
    }
}

}  // namespace

extern "C" cudaError_t quench6k_nem_attn_prep(const float* qkv, int ld, int q_col, int k_col, int v_col, int rows, const void* seqs, long long kv_off,
                                              long long kv_ld, int fp8, void* q, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    const KvArgs kv{kv_off, kv_ld, fp8};
    SQ_CHECK_LAUNCH(sq_launch(attn_prep_kernel, dim3(rows, 2), HD, 0, st, qkv, ld, q_col, k_col, v_col, static_cast<const Seqs*>(seqs), kv,
                              static_cast<bf16*>(q)));
    SQ_RETURN_LAUNCH();
}

extern "C" size_t quench6k_nem_attn_part_floats(int rows, int max_groups) { return (size_t)rows * KVH * max_groups * PART_FLOATS; }

extern "C" cudaError_t quench6k_nem_attention_small(const void* q, int rows, const void* seqs, long long kv_off, long long kv_ld, int fp8,
                                                    int max_groups, float* part, void* out, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    if (max_groups <= 0) return cudaErrorInvalidValue;
    const KvArgs kv{kv_off, kv_ld, fp8};
    SQ_CHECK_LAUNCH(sq_launch(attn_decode_kernel, dim3(rows, KVH, max_groups), 128, 0, st, static_cast<const bf16*>(q), static_cast<const Seqs*>(seqs),
                              kv, max_groups, part));
    SQ_CHECK_LAUNCH(sq_launch(attn_combine_kernel, dim3(rows, KVH), HD, 0, st, (const float*)part, static_cast<const Seqs*>(seqs), max_groups,
                              static_cast<bf16*>(out)));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_nem_attention_prefill(const void* q, int rows, const void* seqs, long long kv_off, long long kv_ld, int fp8, void* out,
                                                      cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    static bool configured = false;
    if (!configured) {
        const cudaError_t e = cudaFuncSetAttribute(attn_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(PrefillSmem));
        if (e != cudaSuccess) return e;
        configured = true;
    }
    const KvArgs kv{kv_off, kv_ld, fp8};
    SQ_CHECK_LAUNCH(sq_launch(attn_prefill_kernel, dim3((rows + PF_WARPS - 1) / PF_WARPS, KVH), PF_WARPS * 32, sizeof(PrefillSmem), st,
                              static_cast<const bf16*>(q), rows, static_cast<const Seqs*>(seqs), kv, static_cast<bf16*>(out)));
    SQ_RETURN_LAUNCH();
}
