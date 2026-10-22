// Recursive Flint (FlintForCausalLM): GQA attention with per-head QK RMSNorm, learned sinks, sliding-window
// or full attention (RoPE only on sliding layers), and a sigmoid-routed MoE (256 experts, top 6) with an
// ungated shared expert. Weights are NVFP4: codes U8 [N, K/2] (low nibble first), E4M3 block scales U8
// [N, K/16] and one FP32 scale per matrix (per expert).
//
// The MoE path treats every FFN as a list of 512-wide "slots" per token: a MoE layer's 6 routed experts
// (router weights) plus the shared expert split into two 512-wide halves (weight 1), and a dense layer's
// 6144-wide MLP as 12 slices (weight 1). A slot's gate/up rows and down columns are one expert-table entry.

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "fp4.cuh"
#include "mma.cuh"
#include "variants.cuh"

namespace {

// Launch with PDL unless variant flint.pdl=0 (plain stream ordering).
template <typename... Exp, typename... Act>
cudaError_t fl_launch(int bit, void (*kernel)(Exp...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Act&&... args) {
    if (sq_variant("flint.pdl", -1) & bit) return sq_launch(kernel, grid, block, smem, st, std::forward<Act>(args)...);
    kernel<<<grid, block, smem, st>>>(static_cast<Exp>(args)...);
    return cudaGetLastError();
}

constexpr int D = 2048, HQ = 32, HKV = 4, HD = 128, GROUP = HQ / HKV, EW = 512;
constexpr int E = 256, TOPK = 6, SLOTS_MAX = 12;

SQ_DEVICE void load_bf16x8(const __nv_bfloat16* p, float* out) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

SQ_DEVICE float e4m3f(uint8_t v) {
    __nv_fp8_e4m3 f;
    f.__x = v;
    return float(f);
}

// ---- Load-time repacking --------------------------------------------------------------------------
// llama.cpp block_nvfp4 (64 values: 4 UE4M3 scales, 32 code bytes; sub-block s byte j holds value j in
// the low nibble and value j + 8 in the high nibble; UE4M3 is half an E4M3 and the codes are doubled
// E2M1, so value = E2M1 * E4M3) -> codes/scales as above. `pb` 64-value blocks form one row of a
// destination piece: source row n's block kb goes to piece kb / pb (pieces of [rows, pb * 64] each).
__global__ void repack_kernel(const uint8_t* raw, size_t blocks, size_t first, int kblocks, int rows, int pb,
                              uint8_t* codes, uint8_t* scales) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= blocks) return;
    const size_t b = first + i;
    const size_t n = b / kblocks, kb = b % kblocks;
    const size_t dst = (kb / pb) * ((size_t)rows * pb) + n * pb + kb % pb;
    const uint8_t* src = raw + i * 36;
    uint8_t out[32];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const uint8_t sc = src[s];
        scales[dst * 4 + s] = sc == 0x7f ? 0 : sc;
        const uint8_t* q = src + 4 + s * 8;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int j0 = 2 * m, j1 = 2 * m + 1;
            const uint8_t e0 = j0 < 8 ? q[j0] & 15 : q[j0 - 8] >> 4;
            const uint8_t e1 = j1 < 8 ? q[j1] & 15 : q[j1 - 8] >> 4;
            out[s * 8 + m] = e0 | (e1 << 4);
        }
    }
    uint4* o = reinterpret_cast<uint4*>(codes + dst * 32);
    o[0] = *reinterpret_cast<uint4*>(out);
    o[1] = *reinterpret_cast<uint4*>(out + 16);
}

// ---- Elementwise ------------------------------------------------------------------------------------
__global__ void embed_kernel(const int* tokens, const __nv_bfloat16* table, float* h) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x;
    const __nv_bfloat16* row = table + (size_t)tokens[t] * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) h[(size_t)t * D + i] = bf2f(row[i]);
}

// out[t] = bf16(x[t] * rsqrt(mean(x[t]^2) + eps) * w), D = 2048, 256 threads.
__global__ void __launch_bounds__(256) rmsnorm_kernel(const float* x, const float* w, float eps,
                                                      __nv_bfloat16* out) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float red[32];
    const int t = blockIdx.x;
    const float* xr = x + (size_t)t * D;
    float v[8], ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = xr[threadIdx.x + 256 * i];
        ss = fmaf(v[i], v[i], ss);
    }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / D + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = threadIdx.x + 256 * i;
        out[(size_t)t * D + c] = f2bf(v[i] * r * w[c]);
    }
}

// Per-head RMSNorm of q and k, RoPE (NeoX halves, theta 10000) when `rope`, K/V into the caches.
// Grid (T, HQ + HKV), 128 threads (one per dimension). q: BF16 [T, HQ, HD]. Cache row of position p:
// slot p & mask ([slot, HKV, HD] BF16).
__global__ void __launch_bounds__(128) qk_prep_kernel(const float* qi, int ldq, const float* ki, int ldk,
                                                      const float* vi, int ldv, const float* q_norm,
                                                      const float* k_norm, int rope, const int* start_p, int mask,
                                                      __nv_bfloat16* q, __nv_bfloat16* kc, __nv_bfloat16* vc) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float red[32], xs[HD];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int pos = *start_p + t;
    const bool is_q = h < HQ;
    float x = is_q ? qi[(size_t)t * ldq + h * HD + d] : ki[(size_t)t * ldk + (h - HQ) * HD + d];
    const float ss = block_sum(x * x, red);
    x = x * rsqrtf(ss / HD + 1e-6f) * (is_q ? q_norm[d] : k_norm[d]);
    if (rope) {
        xs[d] = x;
        __syncthreads();
        const int i = d & (HD / 2 - 1);
        const float inv = exp2f(-(float)(2 * i) / HD * 13.287712379549449f);  // log2(10000)
        float sn, cs;
        sincosf((float)pos * inv, &sn, &cs);
        x = d < HD / 2 ? xs[d] * cs - xs[d + HD / 2] * sn : xs[d] * cs + xs[d - HD / 2] * sn;
    }
    if (is_q) {
        q[((size_t)t * HQ + h) * HD + d] = f2bf(x);
    } else {
        const int kh = h - HQ;
        const size_t slot = (size_t)(pos & mask);
        kc[(slot * HKV + kh) * HD + d] = f2bf(x);
        vc[(slot * HKV + kh) * HD + d] = f2bf(vi[(size_t)t * ldv + kh * HD + d]);
    }
}

// ---- Attention ----------------------------------------------------------------------------------------
// FlashAttention-2 style over one KV head: a CTA takes 64 query rows (8 tokens x the 8 query heads of the
// group, row = token * 8 + head) and one split of the key range, 4 warps of 16 rows, key tiles of 64.
// Row r (token t) sees keys j <= pos(t) and, with a window W, j > pos(t) - W. Writes the split's
// unnormalized output and (max, sum) in base 2; attn_merge adds the sinks and normalizes.
constexpr int AT_ROWS = 64, AT_KEYS = 64, AT_LD = HD + 8;

struct __align__(16) AttnSmem {
    __nv_bfloat16 q[AT_ROWS * AT_LD];
    __nv_bfloat16 k[2][AT_KEYS * AT_LD];
    __nv_bfloat16 v[2][AT_KEYS * AT_LD];
};

__global__ void __launch_bounds__(128) attn_kernel(const __nv_bfloat16* q, const __nv_bfloat16* kc,
                                                   const __nv_bfloat16* vc, int T, const int* start_p, int window,
                                                   int mask, int splits, float* part_o, float* part_ml) {
    pdl_launch_dependents();
    pdl_wait();
    const int start = *start_p;
    extern __shared__ __align__(16) uint8_t at_smem[];
    AttnSmem& sm = *reinterpret_cast<AttnSmem*>(at_smem);
    const int kh = blockIdx.y, split = blockIdx.z;
    const int tok0 = blockIdx.x * (AT_ROWS / GROUP);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int ntok = min(AT_ROWS / GROUP, T - tok0);
    // Key range of the whole tile, split in key-tile units.
    const int p_lo = start + tok0, p_hi = start + tok0 + ntok - 1;
    const int k_begin_all = window > 0 ? max(0, p_lo - window + 1) : 0;
    const int k_end_all = p_hi + 1;
    const int tiles_all = (k_end_all - k_begin_all + AT_KEYS - 1) / AT_KEYS;
    const int per = (tiles_all + splits - 1) / splits;
    const int t_begin = split * per, t_end = min(tiles_all, t_begin + per);

    // Q rows -> shared.
    for (int i = tid; i < AT_ROWS * (HD / 8); i += 128) {
        const int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
        const int tt = r / GROUP, hh = r % GROUP;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (tt < ntok) val = *reinterpret_cast<const uint4*>(q + ((size_t)(tok0 + tt) * HQ + kh * GROUP + hh) * HD + c);
        *reinterpret_cast<uint4*>(&sm.q[r * AT_LD + c]) = val;
    }
    auto load_kv = [&](int buf, int tile) {
        const int j0 = k_begin_all + tile * AT_KEYS;
        for (int i = tid; i < AT_KEYS * (HD / 8); i += 128) {
            const int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
            const int j = j0 + r;
            const bool ok = j < k_end_all;
            const size_t off = ((size_t)((ok ? j : 0) & mask) * HKV + kh) * HD + c;
            cp_async16(&sm.k[buf][r * AT_LD + c], kc + off, ok);
            cp_async16(&sm.v[buf][r * AT_LD + c], vc + off, ok);
        }
    };
    const int g8 = lane >> 2, t4 = lane & 3;
    // This thread's two rows (g8 and g8 + 8 of the warp's 16) and their limits.
    int lim_hi[2], lim_lo[2];
#pragma unroll
    for (int hrow = 0; hrow < 2; ++hrow) {
        const int r = warp * 16 + g8 + hrow * 8, tt = r / GROUP;
        const int pos = start + tok0 + min(tt, ntok - 1);
        lim_hi[hrow] = tt < ntok ? pos : -1;
        lim_lo[hrow] = window > 0 ? pos - window + 1 : 0;
    }
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    float o[HD / 8][4] = {};
    const float scale = 0.08838834764831845f * 1.4426950408889634f;  // 1/sqrt(128) * log2(e)

    if (t_begin < t_end) load_kv(0, t_begin);
    cp_async_commit();
    __syncthreads();
    uint32_t qa[HD / 16][4];
#pragma unroll
    for (int ks = 0; ks < HD / 16; ++ks)
        ldmatrix_x4(qa[ks], &sm.q[(warp * 16 + (lane & 15)) * AT_LD + ks * 16 + (lane >> 4) * 8]);

    for (int tile = t_begin; tile < t_end; ++tile) {
        const int buf = (tile - t_begin) & 1;
        if (tile + 1 < t_end) load_kv(buf ^ 1, tile + 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const int j0 = k_begin_all + tile * AT_KEYS;
        float s[AT_KEYS / 8][4] = {};
#pragma unroll
        for (int ks = 0; ks < HD / 16; ++ks) {
#pragma unroll
            for (int nj = 0; nj < AT_KEYS / 16; ++nj) {
                uint32_t r4[4];
                const int kr = nj * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldmatrix_x4(r4, &sm.k[buf][kr * AT_LD + ks * 16 + ((lane >> 3) & 1) * 8]);
                const uint32_t b0[2] = {r4[0], r4[1]}, b1[2] = {r4[2], r4[3]};
                mma_bf16(s[2 * nj], qa[ks], b0);
                mma_bf16(s[2 * nj + 1], qa[ks], b1);
            }
        }
        float mx[2] = {m[0], m[1]};
#pragma unroll
        for (int n = 0; n < AT_KEYS / 8; ++n)
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                const int hrow = x >> 1, j = j0 + n * 8 + 2 * t4 + (x & 1);
                const bool ok = j <= lim_hi[hrow] && j >= lim_lo[hrow];
                s[n][x] = ok ? s[n][x] * scale : -INFINITY;
                mx[hrow] = fmaxf(mx[hrow], s[n][x]);
            }
        float corr[2];
#pragma unroll
        for (int hrow = 0; hrow < 2; ++hrow) {
            mx[hrow] = fmaxf(mx[hrow], __shfl_xor_sync(0xffffffff, mx[hrow], 1));
            mx[hrow] = fmaxf(mx[hrow], __shfl_xor_sync(0xffffffff, mx[hrow], 2));
            corr[hrow] = mx[hrow] == -INFINITY ? 1.0f : exp2f(m[hrow] - mx[hrow]);
            m[hrow] = mx[hrow];
            l[hrow] *= corr[hrow];
        }
#pragma unroll
        for (int n = 0; n < HD / 8; ++n) {
            o[n][0] *= corr[0], o[n][1] *= corr[0], o[n][2] *= corr[1], o[n][3] *= corr[1];
        }
        uint32_t pa[AT_KEYS / 16][4];
#pragma unroll
        for (int n = 0; n < AT_KEYS / 8; ++n) {
            float p[4];
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                const int hrow = x >> 1;
                p[x] = m[hrow] == -INFINITY ? 0.0f : exp2f(s[n][x] - m[hrow]);
            }
            // Round to BF16 before summing so that l matches the P used in P.V.
            const __nv_bfloat162 lo = __floats2bfloat162_rn(p[0], p[1]), hi = __floats2bfloat162_rn(p[2], p[3]);
            const float2 flo = __bfloat1622float2(lo), fhi = __bfloat1622float2(hi);
            l[0] += flo.x + flo.y;
            l[1] += fhi.x + fhi.y;
            const int ks = n / 2, half = n % 2;
            pa[ks][half * 2 + 0] = *reinterpret_cast<const uint32_t*>(&lo);
            pa[ks][half * 2 + 1] = *reinterpret_cast<const uint32_t*>(&hi);
        }
#pragma unroll
        for (int ks = 0; ks < AT_KEYS / 16; ++ks) {
#pragma unroll
            for (int nd = 0; nd < HD / 16; ++nd) {
                uint32_t r4[4];
                const int vr = ks * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
                ldmatrix_x4_trans(r4, &sm.v[buf][vr * AT_LD + nd * 16 + (lane >> 4) * 8]);
                const uint32_t b0[2] = {r4[0], r4[1]}, b1[2] = {r4[2], r4[3]};
                mma_bf16(o[2 * nd], pa[ks], b0);
                mma_bf16(o[2 * nd + 1], pa[ks], b1);
            }
        }
        __syncthreads();
    }
    cp_async_wait<0>();
#pragma unroll
    for (int hrow = 0; hrow < 2; ++hrow) {
        l[hrow] += __shfl_xor_sync(0xffffffff, l[hrow], 1);
        l[hrow] += __shfl_xor_sync(0xffffffff, l[hrow], 2);
        const int r = warp * 16 + g8 + hrow * 8, tt = r / GROUP, hh = r % GROUP;
        if (tt >= ntok) continue;
        const size_t row = (size_t)(tok0 + tt) * HQ + kh * GROUP + hh;
        float* po = part_o + (row * splits + split) * HD;
#pragma unroll
        for (int n = 0; n < HD / 8; ++n)
            *reinterpret_cast<float2*>(po + n * 8 + 2 * t4) = make_float2(o[n][2 * hrow], o[n][2 * hrow + 1]);
        if (t4 == 0) {
            part_ml[(row * splits + split) * 2] = m[hrow];
            part_ml[(row * splits + split) * 2 + 1] = l[hrow];
        }
    }
}

// out[t, h * 128 + d] = bf16(sum_s 2^(m_s - M) o_s / (sum_s 2^(m_s - M) l_s + 2^(sink_h log2 e - M))).
// Grid T * HQ, 128 threads.
__global__ void __launch_bounds__(128) attn_merge_kernel(const float* part_o, const float* part_ml,
                                                         int splits, const float* sinks, __nv_bfloat16* out) {
    pdl_launch_dependents();
    pdl_wait();
    const int row = blockIdx.x, h = row % HQ, d = threadIdx.x;
    const float sink = sinks[h] * 1.4426950408889634f;
    float M = sink;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, part_ml[((size_t)row * splits + s) * 2]);
    float den = exp2f(sink - M), num = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float ms = part_ml[((size_t)row * splits + s) * 2];
        if (ms == -INFINITY) continue;
        const float w = exp2f(ms - M);
        den = fmaf(w, part_ml[((size_t)row * splits + s) * 2 + 1], den);
        num = fmaf(w, part_o[((size_t)row * splits + s) * HD + d], num);
    }
    out[(size_t)row * HD + d] = f2bf(num / den);
}

// ---- NVFP4 GEMV (few rows) -----------------------------------------------------------------------------
// y[m, c] (+)= s2 * x[m, :] . W[n, :] for up to three matrices whose output columns are concatenated
// (c = column offset of the matrix + n). x BF16 [M, K] in shared memory; a warp per output row, each lane
// 8 consecutive weights per 256-wide step (4 code bytes; the block scale is shared by lane pairs).
struct Mats {
    const uint8_t* codes[3];
    const uint8_t* scales[3];
    const float* s2[3];
    int end[3];  // cumulative output columns
};

SQ_DEVICE void fp4x8(uint32_t c, float sc, float* w) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2_raw r = __nv_cvt_fp4x2_to_halfraw2((uint8_t)(c >> (8 * i)), __NV_E2M1);
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&r));
        w[2 * i] = f.x * sc;
        w[2 * i + 1] = f.y * sc;
    }
}

constexpr int GV_WARPS = 8, GV_KMAX = 4096;

// x: BF16 [M, K], or with NORM the FP32 residual rows [M, K] normalized here (RMSNorm, weight norm_w,
// eps 1e-6) into the BF16 copy every CTA keeps in shared memory. A warp computes R rows; all of a row's
// weight codes and scales are loaded before the PDL wait.
template <int M, bool NORM, int R>
__global__ void __launch_bounds__(GV_WARPS * 32) fp4_gemv_kernel(const void* xin, const float* norm_w, int K, Mats mats, float* y,
                                                                 int ldy, int accumulate) {
    pdl_launch_dependents();
    extern __shared__ __align__(16) __nv_bfloat16 gv_x[];
    __shared__ float red[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n_total = mats.end[2];
    const int row0 = (blockIdx.x * GV_WARPS + warp) * R;
    const int steps = K / 256;
    uint32_t c[R][GV_KMAX / 256];
    uint8_t sb[R][GV_KMAX / 256];
    int mat[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, n_total - 1);
        mat[r] = row < mats.end[0] ? 0 : (row < mats.end[1] ? 1 : 2);
        const int n = row - (mat[r] == 0 ? 0 : mats.end[mat[r] - 1]);
        const uint8_t* wc = mats.codes[mat[r]] + (size_t)n * (K / 2);
        const uint8_t* ws = mats.scales[mat[r]] + (size_t)n * (K / 16);
#pragma unroll
        for (int u = 0; u < GV_KMAX / 256; ++u) {
            if (u < steps) {
                const int k = u * 256 + lane * 8;
                c[r][u] = ld_stream_u32(wc + k / 2);
                sb[r][u] = ws[k / 16];
            }
        }
    }
    pdl_wait();
    if constexpr (NORM) {
        const float* h = static_cast<const float*>(xin);
        for (int mm = 0; mm < M; ++mm) {
            float ss = 0.0f;
            for (int i = threadIdx.x; i < K; i += blockDim.x) ss = fmaf(h[(size_t)mm * K + i], h[(size_t)mm * K + i], ss);
            ss = block_sum(ss, red);
            const float rn = rsqrtf(ss / K + 1e-6f);
            for (int i = threadIdx.x; i < K; i += blockDim.x) gv_x[(size_t)mm * K + i] = f2bf(h[(size_t)mm * K + i] * rn * norm_w[i]);
        }
    } else {
        const uint4* x = static_cast<const uint4*>(xin);
        for (int i = threadIdx.x; i < M * K / 8; i += blockDim.x) reinterpret_cast<uint4*>(gv_x)[i] = x[i];
    }
    __syncthreads();
    if (row0 >= n_total) return;
    float acc[R][M] = {};
#pragma unroll
    for (int u = 0; u < GV_KMAX / 256; ++u) {
        if (u >= steps) break;
        const int k = u * 256 + lane * 8;
        float xv[M][8];
#pragma unroll
        for (int mm = 0; mm < M; ++mm) load_bf16x8(gv_x + (size_t)mm * K + k, xv[mm]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float w[8];
            fp4x8(c[r][u], e4m3f(sb[r][u]), w);
#pragma unroll
            for (int mm = 0; mm < M; ++mm)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[r][mm] = fmaf(w[i], xv[mm][i], acc[r][mm]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        const float s2 = *mats.s2[mat[r]];
#pragma unroll
        for (int mm = 0; mm < M; ++mm) {
            const float v = warp_sum(acc[r][mm]) * s2;
            if (lane == 0 && row < n_total) {
                float* dst = y + (size_t)mm * ldy + row;
                *dst = accumulate ? *dst + v : v;
            }
        }
    }
}

// One token: each lane holds the 8-value slices k = u * 256 + 8 lane of x in registers (BF16; with NORM
// computed from the FP32 residual row by the warp itself), so there is no shared memory or block sync.
template <bool NORM, int R>
__global__ void __launch_bounds__(GV_WARPS * 32) fp4_gemv1_kernel(const void* xin, const float* norm_w, int K, Mats mats, float* y,
                                                                  int accumulate) {
    pdl_launch_dependents();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n_total = mats.end[2];
    const int row0 = (blockIdx.x * GV_WARPS + warp) * R;
    const int steps = K / 256;
    uint32_t c[R][GV_KMAX / 256];
    uint8_t sb[R][GV_KMAX / 256];
    int mat[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, n_total - 1);
        mat[r] = row < mats.end[0] ? 0 : (row < mats.end[1] ? 1 : 2);
        const int n = row - (mat[r] == 0 ? 0 : mats.end[mat[r] - 1]);
        const uint8_t* wc = mats.codes[mat[r]] + (size_t)n * (K / 2);
        const uint8_t* ws = mats.scales[mat[r]] + (size_t)n * (K / 16);
#pragma unroll
        for (int u = 0; u < GV_KMAX / 256; ++u)
            if (u < steps) {
                const int k = u * 256 + lane * 8;
                c[r][u] = ld_stream_u32(wc + k / 2);
                sb[r][u] = ws[k / 16];
            }
    }
    pdl_wait();
    uint4 xr[GV_KMAX / 256];
    if constexpr (NORM) {
        const float* h = static_cast<const float*>(xin);
        float ss = 0.0f;
#pragma unroll
        for (int u = 0; u < GV_KMAX / 256; ++u)
            if (u < steps) {
                const float4 a = *reinterpret_cast<const float4*>(h + u * 256 + lane * 8);
                const float4 b = *reinterpret_cast<const float4*>(h + u * 256 + lane * 8 + 4);
                ss += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
            }
        const float rn = rsqrtf(warp_sum(ss) / K + 1e-6f);
#pragma unroll
        for (int u = 0; u < GV_KMAX / 256; ++u)
            if (u < steps) {
                const int k = u * 256 + lane * 8;
                const float4 a = *reinterpret_cast<const float4*>(h + k), b = *reinterpret_cast<const float4*>(h + k + 4);
                const float4 wa = *reinterpret_cast<const float4*>(norm_w + k), wb = *reinterpret_cast<const float4*>(norm_w + k + 4);
                __nv_bfloat162 p[4] = {__floats2bfloat162_rn(a.x * rn * wa.x, a.y * rn * wa.y), __floats2bfloat162_rn(a.z * rn * wa.z, a.w * rn * wa.w),
                                       __floats2bfloat162_rn(b.x * rn * wb.x, b.y * rn * wb.y), __floats2bfloat162_rn(b.z * rn * wb.z, b.w * rn * wb.w)};
                xr[u] = *reinterpret_cast<uint4*>(p);
            }
    } else {
        const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(xin);
#pragma unroll
        for (int u = 0; u < GV_KMAX / 256; ++u)
            if (u < steps) xr[u] = *reinterpret_cast<const uint4*>(x + u * 256 + lane * 8);
    }
    if (row0 >= n_total) return;
    float acc[R] = {};
#pragma unroll
    for (int u = 0; u < GV_KMAX / 256; ++u) {
        if (u >= steps) break;
        float xv[8];
        load_bf16x8(reinterpret_cast<const __nv_bfloat16*>(&xr[u]), xv);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float w[8];
            fp4x8(c[r][u], e4m3f(sb[r][u]), w);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[r] = fmaf(w[i], xv[i], acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        const float v = warp_sum(acc[r]) * *mats.s2[mat[r]];
        if (lane == 0 && row < n_total) y[row] = accumulate ? y[row] + v : v;
    }
}

// One token, x BF16 [K] staged per CTA in shared memory with its 16-byte chunks XOR-swizzled
// (chunk j at j ^ ((j >> 3) & 7)), so that lanes reading 32 consecutive values each hit distinct banks.
// Each lane takes 32 consecutive weights per 1024-wide step (16 code bytes, two scale bytes); a warp
// computes R rows. STEPS = K / 1024.
SQ_DEVICE int swz16(int j) { return j ^ ((j >> 3) & 7); }

template <int STEPS, int R>
__global__ void __launch_bounds__(GV_WARPS * 32) fp4_gemv2_kernel(const __nv_bfloat16* x, Mats mats, float* y, int accumulate) {
    pdl_launch_dependents();
    constexpr int K = STEPS * 1024;
    __shared__ __align__(16) uint4 xs[K / 8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n_total = mats.end[2];
    const int row0 = (blockIdx.x * GV_WARPS + warp) * R;
    uint4 c[R][STEPS];
    uint16_t sb[R][STEPS];
    int mat[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, n_total - 1);
        mat[r] = row < mats.end[0] ? 0 : (row < mats.end[1] ? 1 : 2);
        const int n = row - (mat[r] == 0 ? 0 : mats.end[mat[r] - 1]);
        const uint8_t* wc = mats.codes[mat[r]] + (size_t)n * (K / 2);
        const uint8_t* ws = mats.scales[mat[r]] + (size_t)n * (K / 16);
#pragma unroll
        for (int u = 0; u < STEPS; ++u) {
            const int k = u * 1024 + lane * 32;
            c[r][u] = ld_stream(wc + k / 2);
            sb[r][u] = *reinterpret_cast<const uint16_t*>(ws + k / 16);
        }
    }
    pdl_wait();
    for (int j = threadIdx.x; j < K / 8; j += blockDim.x) xs[swz16(j)] = reinterpret_cast<const uint4*>(x)[j];
    __syncthreads();
    if (row0 >= n_total) return;
    float acc[R] = {};
#pragma unroll
    for (int u = 0; u < STEPS; ++u) {
        float xv[32];
#pragma unroll
        for (int q = 0; q < 4; ++q) load_bf16x8(reinterpret_cast<const __nv_bfloat16*>(&xs[swz16(u * 128 + lane * 4 + q)]), xv + 8 * q);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float s0 = e4m3f(sb[r][u] & 0xff), s1 = e4m3f(sb[r][u] >> 8);
            float w[32];
            fp4x8(c[r][u].x, s0, w);
            fp4x8(c[r][u].y, s0, w + 8);
            fp4x8(c[r][u].z, s1, w + 16);
            fp4x8(c[r][u].w, s1, w + 24);
#pragma unroll
            for (int i = 0; i < 32; ++i) acc[r] = fmaf(w[i], xv[i], acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        const float v = warp_sum(acc[r]) * *mats.s2[mat[r]];
        if (lane == 0 && row < n_total) y[row] = accumulate ? y[row] + v : v;
    }
}

// ---- Routing --------------------------------------------------------------------------------------------
// One warp per token: sigmoid scores, top 6 (ties: lower expert), weights renormalized; then the shared
// expert's two halves (entries 256, 257) with weight 1. Slots per token: 8.
__global__ void route_kernel(const float* logits, int T, int* sel_e, float* sel_w) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (t >= T) return;
    float v[E / 32];
#pragma unroll
    for (int j = 0; j < E / 32; ++j) {
        const float z = logits[(size_t)t * E + j * 32 + lane];
        v[j] = 1.0f / (1.0f + expf(-z));
        if (!(v[j] >= 0.0f)) v[j] = 0.0f;
    }
    float chosen[TOPK], sum = 0.0f;
    int chosen_i[TOPK];
    for (int r = 0; r < TOPK; ++r) {
        float best = -1.0f;
        int best_i = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < E / 32; ++j)
            if (v[j] > best) best = v[j], best_i = j * 32 + lane;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffff, best, o);
            const int oi = __shfl_xor_sync(0xffffffff, best_i, o);
            if (ob > best || (ob == best && oi < best_i)) best = ob, best_i = oi;
        }
        chosen[r] = best;
        chosen_i[r] = best_i;
        sum += best;
        if ((best_i & 31) == lane) v[best_i / 32] = -2.0f;
    }
    if (lane == 0) {
        for (int r = 0; r < TOPK; ++r) {
            sel_e[t * 8 + r] = chosen_i[r];
            sel_w[t * 8 + r] = chosen[r] / (sum + 1e-20f);
        }
        sel_e[t * 8 + 6] = E, sel_w[t * 8 + 6] = 1.0f;
        sel_e[t * 8 + 7] = E + 1, sel_w[t * 8 + 7] = 1.0f;
    }
}

// Dense layers: slot s of every token is slice s with weight 1.
__global__ void dense_slots_kernel(int T, int S, int* sel_e, float* sel_w) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * S) return;
    sel_e[i] = i % S;
    sel_w[i] = 1.0f;
}

// ---- MoE, few tokens ---------------------------------------------------------------------------------
// act[t * S + s, n] = silu(g) * u for gate/up rows n of slot s's expert, input xn[t] (BF16 [T, 2048]).
// Grid (T * S, EW / GV_WARPS), a warp per row n; the token's input row is in registers (64 per lane).
__global__ void __launch_bounds__(GV_WARPS * 32) moe_gate_up_kernel(const __nv_bfloat16* xn, int S,
                                                                    const int* sel_e, const uint64_t* gw,
                                                                    const uint64_t* gs, const float* g2,
                                                                    const uint64_t* uw, const uint64_t* us,
                                                                    const float* u2, float* act) {
    pdl_launch_dependents();
    pdl_wait();
    const int pair = blockIdx.x, t = pair / S, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = blockIdx.y * GV_WARPS + warp;
    const int e = sel_e[pair];
    const uint8_t* gc = reinterpret_cast<const uint8_t*>(gw[e]) + (size_t)n * (D / 2);
    const uint8_t* gsc = reinterpret_cast<const uint8_t*>(gs[e]) + (size_t)n * (D / 16);
    const uint8_t* uc = reinterpret_cast<const uint8_t*>(uw[e]) + (size_t)n * (D / 2);
    const uint8_t* usc = reinterpret_cast<const uint8_t*>(us[e]) + (size_t)n * (D / 16);
    uint32_t cg[D / 256], cu[D / 256];
    uint8_t sg[D / 256], su[D / 256];
#pragma unroll
    for (int u = 0; u < D / 256; ++u) {
        const int k = u * 256 + lane * 8;
        cg[u] = ld_stream_u32(gc + k / 2);
        cu[u] = ld_stream_u32(uc + k / 2);
        sg[u] = gsc[k / 16];
        su[u] = usc[k / 16];
    }
    float ag = 0.0f, au = 0.0f;
    const __nv_bfloat16* xr = xn + (size_t)t * D;
#pragma unroll
    for (int u = 0; u < D / 256; ++u) {
        const int k = u * 256 + lane * 8;
        float xv[8], w[8], v[8];
        load_bf16x8(xr + k, xv);
        fp4x8(cg[u], e4m3f(sg[u]), w);
        fp4x8(cu[u], e4m3f(su[u]), v);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            ag = fmaf(w[i], xv[i], ag);
            au = fmaf(v[i], xv[i], au);
        }
    }
    ag = warp_sum(ag) * g2[e];
    au = warp_sum(au) * u2[e];
    if (lane == 0) act[(size_t)pair * EW + n] = silu(ag) * au;
}

// h[t, n] += sum_s w[t, s] * d2[e_s] * act[t * S + s, :] . Wd_{e_s}[n, :] (K = 512), slots in order.
// Grid (T, D / GV_WARPS).
__global__ void __launch_bounds__(GV_WARPS * 32) moe_down_kernel(const float* act, int S, const int* sel_e,
                                                                 const float* sel_w, const uint64_t* dw,
                                                                 const uint64_t* ds, const float* d2,
                                                                 float* h) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = blockIdx.y * GV_WARPS + warp;
    float total = 0.0f;
    for (int s = 0; s < S; ++s) {
        const int e = sel_e[t * S + s];
        const uint8_t* c = reinterpret_cast<const uint8_t*>(dw[e]) + (size_t)n * (EW / 2);
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(ds[e]) + (size_t)n * (EW / 16);
        const float* a = act + (size_t)(t * S + s) * EW;
        float acc = 0.0f;
#pragma unroll
        for (int u = 0; u < EW / 256; ++u) {
            const int k = u * 256 + lane * 8;
            float w[8];
            fp4x8(ld_stream_u32(c + k / 2), e4m3f(sc[k / 16]), w);
            const float4 a0 = *reinterpret_cast<const float4*>(a + k), a1 = *reinterpret_cast<const float4*>(a + k + 4);
            acc = fmaf(w[0], a0.x, acc), acc = fmaf(w[1], a0.y, acc), acc = fmaf(w[2], a0.z, acc), acc = fmaf(w[3], a0.w, acc);
            acc = fmaf(w[4], a1.x, acc), acc = fmaf(w[5], a1.y, acc), acc = fmaf(w[6], a1.z, acc), acc = fmaf(w[7], a1.w, acc);
        }
        acc = warp_sum(acc);
        total = fmaf(sel_w[t * S + s], acc * d2[e], total);
    }
    if (lane == 0) h[(size_t)t * D + n] += total;
}

// ---- Router, few tokens ----------------------------------------------------------------------------
// logits[t, e] = xn[t] . Wr[e] (BF16 [256, 2048]) with xn = RMSNorm(h[t]) computed per CTA (CTA 0 also
// writes it for the experts); the last CTA to finish (ticket) routes every token as route_kernel and
// resets the ticket. Grid E / GV_WARPS, a warp per expert row.
constexpr int RT_TMAX = 8;

SQ_DEVICE void route_token(const float* logits, int t, int lane, int* sel_e, float* sel_w) {
    float v[E / 32];
#pragma unroll
    for (int j = 0; j < E / 32; ++j) {
        const float z = __ldcg(logits + (size_t)t * E + j * 32 + lane);
        v[j] = 1.0f / (1.0f + expf(-z));
        if (!(v[j] >= 0.0f)) v[j] = 0.0f;
    }
    float chosen[TOPK], sum = 0.0f;
    int chosen_i[TOPK];
    for (int r = 0; r < TOPK; ++r) {
        float best = -1.0f;
        int best_i = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < E / 32; ++j)
            if (v[j] > best) best = v[j], best_i = j * 32 + lane;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffff, best, o);
            const int oi = __shfl_xor_sync(0xffffffff, best_i, o);
            if (ob > best || (ob == best && oi < best_i)) best = ob, best_i = oi;
        }
        chosen[r] = best;
        chosen_i[r] = best_i;
        sum += best;
        if ((best_i & 31) == lane) v[best_i / 32] = -2.0f;
    }
    if (lane == 0) {
        for (int r = 0; r < TOPK; ++r) {
            sel_e[t * 8 + r] = chosen_i[r];
            sel_w[t * 8 + r] = chosen[r] / (sum + 1e-20f);
        }
        sel_e[t * 8 + 6] = E, sel_w[t * 8 + 6] = 1.0f;
        sel_e[t * 8 + 7] = E + 1, sel_w[t * 8 + 7] = 1.0f;
    }
}

__global__ void __launch_bounds__(128) router_kernel(const float* h, const float* norm_w, const __nv_bfloat16* W, int T, float* logits,
                                                    __nv_bfloat16* xn, int* ticket, int* sel_e, float* sel_w) {
    pdl_launch_dependents();
    __shared__ bool last;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 4 + warp;
    uint4 w[D / 256];
#pragma unroll
    for (int u = 0; u < D / 256; ++u) w[u] = ld_stream(W + (size_t)row * D + u * 256 + lane * 8);
    pdl_wait();
    // logit = rsqrt(mean(h^2) + eps) * (Wr . (h * norm_w)); the warp reads the whole row, so it also
    // has the sum of squares. Warp 0 of CTA 0 writes the normalized row for the experts.
    for (int t = 0; t < T; ++t) {
        const float* hr = h + (size_t)t * D;
        float acc = 0.0f, ss = 0.0f;
        float hv[D / 256][8];
#pragma unroll
        for (int u = 0; u < D / 256; ++u) {
            const int k = u * 256 + lane * 8;
            const float4 a = *reinterpret_cast<const float4*>(hr + k), b = *reinterpret_cast<const float4*>(hr + k + 4);
            const float4 na = *reinterpret_cast<const float4*>(norm_w + k), nb = *reinterpret_cast<const float4*>(norm_w + k + 4);
            const float x[8] = {a.x * na.x, a.y * na.y, a.z * na.z, a.w * na.w, b.x * nb.x, b.y * nb.y, b.z * nb.z, b.w * nb.w};
            const float hh[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            float wv[8];
            load_bf16x8(reinterpret_cast<const __nv_bfloat16*>(&w[u]), wv);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc = fmaf(wv[i], x[i], acc);
                ss = fmaf(hh[i], hh[i], ss);
                hv[u][i] = x[i];
            }
        }
        const float rn = rsqrtf(warp_sum(ss) / D + 1e-6f);
        acc = warp_sum(acc) * rn;
        if (lane == 0) logits[(size_t)t * E + row] = acc;
        if (blockIdx.x == 0 && warp == 0) {
#pragma unroll
            for (int u = 0; u < D / 256; ++u) {
                __nv_bfloat162 p[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(hv[u][2 * i] * rn, hv[u][2 * i + 1] * rn);
                *reinterpret_cast<uint4*>(xn + (size_t)t * D + u * 256 + lane * 8) = *reinterpret_cast<uint4*>(p);
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(ticket, 1) == (int)gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int t = warp; t < T; t += 4) route_token(logits, t, lane, sel_e, sel_w);
    if (threadIdx.x == 0) *ticket = 0;
}

// ---- MoE, few tokens (v2) -------------------------------------------------------------------------------
// act[p, n] = silu(xn[t] . G[n]) * (xn[t] . U[n]) for pair p = t * S + s. Grid (T * S, EW / 16): a warp
// computes two gate and two up rows, each lane 16 consecutive weights per 512-wide step.
__global__ void __launch_bounds__(GV_WARPS * 32) moe_gate_up2_kernel(const __nv_bfloat16* xn, int S, const int* sel_e,
                                                                     const uint64_t* gw, const uint64_t* gs, const float* g2,
                                                                     const uint64_t* uw, const uint64_t* us, const float* u2, float* act) {
    pdl_launch_dependents();
    __shared__ __align__(16) __nv_bfloat16 xs[D];
    pdl_wait();
    const int pair = blockIdx.x, t = pair / S, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < D / 8; i += blockDim.x) reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(xn + (size_t)t * D)[i];
    const int e = sel_e[pair];
    const int n0 = blockIdx.y * 16 + warp * 2;
    const uint8_t* mc[2] = {reinterpret_cast<const uint8_t*>(gw[e]), reinterpret_cast<const uint8_t*>(uw[e])};
    const uint8_t* ms[2] = {reinterpret_cast<const uint8_t*>(gs[e]), reinterpret_cast<const uint8_t*>(us[e])};
    uint2 c[2][2][D / 512];
    uint8_t sc[2][2][D / 512];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int u = 0; u < D / 512; ++u) {
                const int k = u * 512 + lane * 16;
                c[m][r][u] = ld_stream_u2(mc[m] + (size_t)(n0 + r) * (D / 2) + k / 2);
                sc[m][r][u] = ms[m][(size_t)(n0 + r) * (D / 16) + k / 16];
            }
    __syncthreads();
    float v[4] = {};  // slot m * 2 + r
#pragma unroll
    for (int u = 0; u < D / 512; ++u) {
        const int k = u * 512 + lane * 16;
        float xv[16];
        load_bf16x8(xs + k, xv);
        load_bf16x8(xs + k + 8, xv + 8);
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                float w[16];
                const float scl = e4m3f(sc[m][r][u]);
                fp4x8(c[m][r][u].x, scl, w);
                fp4x8(c[m][r][u].y, scl, w + 8);
#pragma unroll
                for (int i = 0; i < 16; ++i) v[m * 2 + r] = fmaf(w[i], xv[i], v[m * 2 + r]);
            }
    }
    const float sum = warp_sum_slots<4>(v);  // lane holds slot lane >> 3
    const int slot = lane >> 3;
    const float gsum = __shfl_sync(0xffffffff, sum, (slot & 1) * 8) * g2[e];       // gate row (slot & 1)
    const float usum = __shfl_sync(0xffffffff, sum, 16 + (slot & 1) * 8) * u2[e];  // up row (slot & 1)
    if (lane == 0 || lane == 8) act[(size_t)pair * EW + n0 + (lane >> 3)] = silu(gsum) * usum;
}

// h[t, n] += sum_s w[t, s] d2[e_s] act[t * S + s] . Wd_{e_s}[n]. Grid (T, D / DN_ROWS): warp w takes
// slots w, w + 8, .. for DN_ROWS rows (each lane 16 of the 512 inputs), slot sums combined in slot order.
constexpr int DN_ROWS = 8;
__global__ void __launch_bounds__(GV_WARPS * 32) moe_down2_kernel(const float* act, int S, const int* sel_e, const float* sel_w,
                                                                  const uint64_t* dw, const uint64_t* ds, const float* d2, float* h) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float part[SLOTS_MAX][DN_ROWS];
    const int t = blockIdx.x, n0 = blockIdx.y * DN_ROWS, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int s = warp; s < S; s += GV_WARPS) {
        const int e = sel_e[t * S + s];
        const uint8_t* c = reinterpret_cast<const uint8_t*>(dw[e]);
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(ds[e]);
        uint2 cw[DN_ROWS];
        uint8_t sw[DN_ROWS];
#pragma unroll
        for (int r = 0; r < DN_ROWS; ++r) {
            cw[r] = ld_stream_u2(c + (size_t)(n0 + r) * (EW / 2) + lane * 8);
            sw[r] = sc[(size_t)(n0 + r) * (EW / 16) + lane];
        }
        float a[16];
        const float4* ap = reinterpret_cast<const float4*>(act + (size_t)(t * S + s) * EW + lane * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 q = ap[i];
            a[4 * i] = q.x, a[4 * i + 1] = q.y, a[4 * i + 2] = q.z, a[4 * i + 3] = q.w;
        }
        float v[DN_ROWS];
#pragma unroll
        for (int r = 0; r < DN_ROWS; ++r) {
            float w[16];
            const float scl = e4m3f(sw[r]);
            fp4x8(cw[r].x, scl, w);
            fp4x8(cw[r].y, scl, w + 8);
            float acc = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) acc = fmaf(w[i], a[i], acc);
            v[r] = acc;
        }
        const float sum = warp_sum_slots<DN_ROWS>(v);  // lane holds row lane / (32 / DN_ROWS)
        if (lane % (32 / DN_ROWS) == 0) part[s][lane / (32 / DN_ROWS)] = sum * d2[e] * sel_w[t * S + s];
    }
    __syncthreads();
    if (threadIdx.x < DN_ROWS) {
        float total = 0.0f;
        for (int s = 0; s < S; ++s) total += part[s][threadIdx.x];
        h[(size_t)t * D + n0 + threadIdx.x] += total;
    }
}

// ---- MoE, many tokens (grouped GEMM tables as moe.cu) -------------------------------------------------
__global__ void count_kernel(const int* sel_e, int pairs, int* counts) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) atomicAdd(&counts[sel_e[i]], 1);
}

// One block of 512 threads over `ne` <= 512 table entries.
__global__ void plan_kernel(const int* counts, int ne, int* offsets, int* cursor, int* tile_expert, int* tile_row,
                            int* num_tiles, int bm) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ int s_rows[512], s_tiles[512];
    const int e = threadIdx.x;
    const int c = e < ne ? counts[e] : 0;
    s_rows[e] = c;
    s_tiles[e] = (c + bm - 1) / bm;
    __syncthreads();
    for (int off = 1; off < 512; off <<= 1) {
        const int a = e >= off ? s_rows[e - off] : 0, b = e >= off ? s_tiles[e - off] : 0;
        __syncthreads();
        s_rows[e] += a;
        s_tiles[e] += b;
        __syncthreads();
    }
    const int row_begin = s_rows[e] - c, tile_begin = s_tiles[e] - (c + bm - 1) / bm;
    if (e < ne) {
        offsets[e] = row_begin;
        cursor[e] = row_begin;
    }
    if (e == 511) {
        offsets[ne] = s_rows[e];
        *num_tiles = s_tiles[e];
    }
    for (int i = 0; i * bm < c; ++i) {
        tile_expert[tile_begin + i] = e;
        tile_row[tile_begin + i] = row_begin + i * bm;
    }
}

__global__ void scatter_kernel(const int* sel_e, int pairs, int* cursor, int* pair_row) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) pair_row[i] = atomicAdd(&cursor[sel_e[i]], 1);
}

// xs[pair_row[p]] = xn[p / S]
__global__ void gather_kernel(const __nv_bfloat16* xn, const int* pair_row, int S, __nv_bfloat16* xs) {
    pdl_launch_dependents();
    pdl_wait();
    const int p = blockIdx.x;
    const uint4* src = reinterpret_cast<const uint4*>(xn + (size_t)(p / S) * D);
    uint4* dst = reinterpret_cast<uint4*>(xs + (size_t)pair_row[p] * D);
    for (int i = threadIdx.x; i < D / 8; i += blockDim.x) dst[i] = src[i];
}

// act[r, n] = bf16(silu(g[r, n]) * u[r, n]), rows x 512.
__global__ void swiglu_kernel(const float* g, const float* u, size_t count, __nv_bfloat16* act) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) act[i] = f2bf(silu(g[i]) * u[i]);
}

// h[t, d] += sum_s w[t, s] * yd[pair_row[t * S + s], d]
__global__ void combine_kernel(const float* yd, const int* pair_row, const float* sel_w, int S,
                               float* h) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < S; ++s) acc = fmaf(sel_w[t * S + s], yd[(size_t)pair_row[t * S + s] * D + d], acc);
        h[(size_t)t * D + d] += acc;
    }
}

// Tables of a single "expert" covering T rows (dense projections through the grouped GEMM).
__global__ void dense_plan_kernel(int T, int bm, int* offsets, int* tile_expert, int* tile_row, int* num_tiles) {
    pdl_launch_dependents();
    pdl_wait();
    const int tiles = (T + bm - 1) / bm;
    for (int i = threadIdx.x; i < tiles; i += blockDim.x) {
        tile_expert[i] = 0;
        tile_row[i] = i * bm;
    }
    if (threadIdx.x == 0) {
        offsets[0] = 0;
        offsets[1] = T;
        *num_tiles = tiles;
    }
}

}  // namespace

extern "C" {

cudaError_t quench6k_flint_repack(const uint8_t* raw, size_t blocks, size_t first, int kblocks, int rows, int pb, uint8_t* codes,
                                  uint8_t* scales, cudaStream_t st) {
    if (blocks == 0) return cudaSuccess;
    repack_kernel<<<(unsigned)((blocks + 255) / 256), 256, 0, st>>>(raw, blocks, first, kblocks, rows, pb, codes, scales);
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_flint_embed(const int* tokens, int T, const void* table, float* h, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(fl_launch(1, embed_kernel, T, 256, 0, st, tokens, static_cast<const __nv_bfloat16*>(table), h));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_flint_rmsnorm(const float* x, const float* w, float eps, void* out, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(fl_launch(2, rmsnorm_kernel, T, 256, 0, st, x, w, eps, static_cast<__nv_bfloat16*>(out)));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_flint_qk_prep(const float* qi, int ldq, const float* ki, int ldk, const float* vi, int ldv, const float* q_norm,
                                   const float* k_norm, int rope, const int* start, int mask, int T, void* q, void* kc, void* vc,
                                   cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(fl_launch(4, qk_prep_kernel, dim3(T, HQ + HKV), 128, 0, st, qi, ldq, ki, ldk, vi, ldv, q_norm, k_norm, rope, start, mask,
                              static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(kc), static_cast<__nv_bfloat16*>(vc)));
    SQ_RETURN_LAUNCH();
}

// Splits for a T-token step whose keys end at start + T (window 0: full).
int quench6k_flint_attn_splits(int T, int start, int window) {
    const int keys = window > 0 ? min(start + T, window + T) : start + T;
    const int tiles = (keys + AT_KEYS - 1) / AT_KEYS;
    const int ctas = ((T * GROUP + AT_ROWS - 1) / AT_ROWS) * HKV;
    int splits = 1;
    while (splits * 2 <= tiles && ctas * splits * 2 <= 4 * 188 && splits < 64) splits *= 2;
    return splits;
}

cudaError_t quench6k_flint_attention(const void* q, const void* kc, const void* vc, int T, const int* start, int window, int mask, int splits,
                                     const float* sinks, float* part_o, float* part_ml, void* out, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    constexpr size_t smem = sizeof(AttnSmem);
    static bool configured = false;
    if (!configured) {
        const cudaError_t err = cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (err != cudaSuccess) return err;
        configured = true;
    }
    const dim3 grid((T * GROUP + AT_ROWS - 1) / AT_ROWS, HKV, splits);
    SQ_CHECK_LAUNCH(fl_launch(8, attn_kernel, grid, 128, smem, st, static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(kc),
                              static_cast<const __nv_bfloat16*>(vc), T, start, window, mask, splits, part_o, part_ml));
    SQ_CHECK_LAUNCH(fl_launch(8, attn_merge_kernel, T * HQ, 128, 0, st, (const float*)part_o, (const float*)part_ml, splits, sinks,
                              static_cast<__nv_bfloat16*>(out)));
    SQ_RETURN_LAUNCH();
}

int quench6k_flint_gemv_max_m() { return 8; }

}  // extern "C"

namespace {

template <int M, bool NORM, int R>
cudaError_t gemv_launch(const void* x, const float* norm_w, int K, const Mats& mats, int total, float* y, int ldy, int accumulate,
                        cudaStream_t st) {
    const size_t smem = (size_t)M * K * 2;
    static bool configured = false;
    if (!configured) {
        const cudaError_t err = cudaFuncSetAttribute(fp4_gemv_kernel<M, NORM, R>, cudaFuncAttributeMaxDynamicSharedMemorySize, M * GV_KMAX * 2);
        if (err != cudaSuccess) return err;
        configured = true;
    }
    const unsigned grid = (total + GV_WARPS * R - 1) / (GV_WARPS * R);
    SQ_CHECK_LAUNCH(fl_launch(16, fp4_gemv_kernel<M, NORM, R>, grid, GV_WARPS * 32, smem, st, x, norm_w, K, mats, y, ldy, accumulate));
    SQ_RETURN_LAUNCH();
}

template <bool NORM, int R>
cudaError_t gemv1_launch(const void* x, const float* norm_w, int K, const Mats& mats, int total, float* y, int accumulate, cudaStream_t st) {
    const unsigned grid = (total + GV_WARPS * R - 1) / (GV_WARPS * R);
    SQ_CHECK_LAUNCH(fl_launch(16, fp4_gemv1_kernel<NORM, R>, grid, GV_WARPS * 32, 0, st, x, norm_w, K, mats, y, accumulate));
    SQ_RETURN_LAUNCH();
}

template <int STEPS, int R>
cudaError_t gemv2_launch(const void* x, const Mats& mats, int total, float* y, int accumulate, cudaStream_t st) {
    const unsigned grid = (total + GV_WARPS * R - 1) / (GV_WARPS * R);
    SQ_CHECK_LAUNCH(fl_launch(16, fp4_gemv2_kernel<STEPS, R>, grid, GV_WARPS * 32, 0, st, static_cast<const __nv_bfloat16*>(x), mats, y, accumulate));
    SQ_RETURN_LAUNCH();
}

template <int M>
cudaError_t gemv_m(const void* x, const float* norm_w, int K, const Mats& mats, int total, float* y, int ldy, int accumulate, cudaStream_t st) {
    const int g = sq_variant("flint.gemv", 2);
    if (M == 1 && !norm_w && g == 2 && (K == 2048 || K == 4096)) {
        const int R = sq_variant("flint.gemv_r", 0);
        if (K == 2048) return (R == 2 || (R == 0 && total < 4 * 188 * GV_WARPS * 4)) ? gemv2_launch<2, 2>(x, mats, total, y, accumulate, st)
                                                                                     : gemv2_launch<2, 4>(x, mats, total, y, accumulate, st);
        return (R == 1 || (R == 0 && total < 4 * 188 * GV_WARPS * 2)) ? gemv2_launch<4, 1>(x, mats, total, y, accumulate, st)
                                                                      : gemv2_launch<4, 2>(x, mats, total, y, accumulate, st);
    }
    if (M == 1 && g == 1) {
        const bool r2 = total >= 2 * 188 * GV_WARPS * 2;
        if (norm_w) return r2 ? gemv1_launch<true, 2>(x, norm_w, K, mats, total, y, accumulate, st)
                              : gemv1_launch<true, 1>(x, norm_w, K, mats, total, y, accumulate, st);
        return r2 ? gemv1_launch<false, 2>(x, norm_w, K, mats, total, y, accumulate, st)
                  : gemv1_launch<false, 1>(x, norm_w, K, mats, total, y, accumulate, st);
    }
    // Two rows per warp unless that leaves fewer than ~2 CTAs per SM.
    const bool r2 = total >= 2 * 188 * GV_WARPS * 2;
    if (norm_w) return r2 ? gemv_launch<M, true, 2>(x, norm_w, K, mats, total, y, ldy, accumulate, st)
                          : gemv_launch<M, true, 1>(x, norm_w, K, mats, total, y, ldy, accumulate, st);
    return r2 ? gemv_launch<M, false, 2>(x, norm_w, K, mats, total, y, ldy, accumulate, st)
              : gemv_launch<M, false, 1>(x, norm_w, K, mats, total, y, ldy, accumulate, st);
}

}  // namespace

extern "C" {

// Up to three NVFP4 matrices (n_i rows each, null codes for unused) with concatenated output columns.
// norm_w non-null: x is FP32 [M, K] and is RMS-normalized with weight norm_w first.
cudaError_t quench6k_flint_gemv(const void* x, const float* norm_w, int M, int K, const uint8_t* const* codes, const uint8_t* const* scales,
                                const float* const* s2, const int* n, float* y, int ldy, int accumulate, cudaStream_t st) {
    if (M == 0) return cudaSuccess;
    if (K % 256 != 0 || M > 8 || K > GV_KMAX) return cudaErrorInvalidValue;
    Mats mats{};
    int total = 0;
    for (int i = 0; i < 3; ++i) {
        mats.codes[i] = codes[i] ? codes[i] : codes[0];
        mats.scales[i] = scales[i] ? scales[i] : scales[0];
        mats.s2[i] = s2[i] ? s2[i] : s2[0];
        total += codes[i] ? n[i] : 0;
        mats.end[i] = total;
    }
    switch (M) {
        case 1: return gemv_m<1>(x, norm_w, K, mats, total, y, ldy, accumulate, st);
        case 2: return gemv_m<2>(x, norm_w, K, mats, total, y, ldy, accumulate, st);
        case 3: return gemv_m<3>(x, norm_w, K, mats, total, y, ldy, accumulate, st);
        case 4: return gemv_m<4>(x, norm_w, K, mats, total, y, ldy, accumulate, st);
        case 5: return gemv_m<5>(x, norm_w, K, mats, total, y, ldy, accumulate, st);
        case 6: return gemv_m<6>(x, norm_w, K, mats, total, y, ldy, accumulate, st);
        case 7: return gemv_m<7>(x, norm_w, K, mats, total, y, ldy, accumulate, st);
        default: return gemv_m<8>(x, norm_w, K, mats, total, y, ldy, accumulate, st);
    }
}

cudaError_t quench6k_flint_router(const float* h, const float* norm_w, const void* w, int T, float* logits, void* xn, int* ticket, int* sel_e,
                                  float* sel_w, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    if (T > RT_TMAX) return cudaErrorInvalidValue;
    SQ_CHECK_LAUNCH(fl_launch(32, router_kernel, E / 4, 128, 0, st, h, norm_w, static_cast<const __nv_bfloat16*>(w), T, logits,
                              static_cast<__nv_bfloat16*>(xn), ticket, sel_e, sel_w));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_flint_route(const float* logits, int T, int* sel_e, float* sel_w, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(fl_launch(32, route_kernel, (T + 3) / 4, 128, 0, st, logits, T, sel_e, sel_w));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_flint_dense_slots(int T, int S, int* sel_e, float* sel_w, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(fl_launch(64, dense_slots_kernel, (T * S + 255) / 256, 256, 0, st, T, S, sel_e, sel_w));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_flint_moe_small(const void* xn, int T, int S, const int* sel_e, const float* sel_w, const uint64_t* gw,
                                     const uint64_t* gs, const float* g2, const uint64_t* uw, const uint64_t* us, const float* u2,
                                     const uint64_t* dw, const uint64_t* ds, const float* d2, float* act, float* h, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    if (sq_variant("flint.moe_v1", 0)) {
        SQ_CHECK_LAUNCH(fl_launch(128, moe_gate_up_kernel, dim3(T * S, EW / GV_WARPS), GV_WARPS * 32, 0, st, static_cast<const __nv_bfloat16*>(xn),
                                  S, sel_e, gw, gs, g2, uw, us, u2, act));
        SQ_CHECK_LAUNCH(fl_launch(256, moe_down_kernel, dim3(T, D / GV_WARPS), GV_WARPS * 32, 0, st, (const float*)act, S, sel_e, sel_w, dw, ds,
                                  d2, h));
    } else {
        SQ_CHECK_LAUNCH(fl_launch(128, moe_gate_up2_kernel, dim3(T * S, EW / 16), GV_WARPS * 32, 0, st, static_cast<const __nv_bfloat16*>(xn), S,
                                  sel_e, gw, gs, g2, uw, us, u2, act));
        SQ_CHECK_LAUNCH(fl_launch(256, moe_down2_kernel, dim3(T, D / DN_ROWS), GV_WARPS * 32, 0, st, (const float*)act, S, sel_e, sel_w, dw, ds, d2, h));
    }
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_flint_moe_plan(const int* sel_e, int pairs, int ne, int bm, int* counts, int* cursor, int* offsets,
                                    int* tile_expert, int* tile_row, int* num_tiles, int* pair_row, cudaStream_t st) {
    if (ne > 512) return cudaErrorInvalidValue;
    SQ_CHECK_LAUNCH(cudaMemsetAsync(counts, 0, sizeof(int) * ne, st));
    if (pairs > 0) SQ_CHECK_LAUNCH(fl_launch(512, count_kernel, (pairs + 255) / 256, 256, 0, st, sel_e, pairs, counts));
    SQ_CHECK_LAUNCH(fl_launch(512, plan_kernel, 1, 512, 0, st, (const int*)counts, ne, offsets, cursor, tile_expert, tile_row, num_tiles, bm));
    if (pairs > 0) SQ_CHECK_LAUNCH(fl_launch(512, scatter_kernel, (pairs + 255) / 256, 256, 0, st, sel_e, pairs, cursor, pair_row));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_flint_gather(const void* xn, const int* pair_row, int pairs, int S, void* xs, cudaStream_t st) {
    if (pairs == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(fl_launch(512, gather_kernel, pairs, 128, 0, st, static_cast<const __nv_bfloat16*>(xn), pair_row, S,
                              static_cast<__nv_bfloat16*>(xs)));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_flint_swiglu(const float* g, const float* u, size_t count, void* act, cudaStream_t st) {
    if (count == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(fl_launch(512, swiglu_kernel, (unsigned)((count + 255) / 256), 256, 0, st, g, u, count, static_cast<__nv_bfloat16*>(act)));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_flint_combine(const float* yd, const int* pair_row, const float* sel_w, int T, int S, float* h, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(fl_launch(512, combine_kernel, T, 256, 0, st, yd, pair_row, sel_w, S, h));
    SQ_RETURN_LAUNCH();
}

cudaError_t quench6k_flint_dense_plan(int T, int bm, int* offsets, int* tile_expert, int* tile_row, int* num_tiles, cudaStream_t st) {
    SQ_CHECK_LAUNCH(fl_launch(512, dense_plan_kernel, 1, 256, 0, st, T, bm, offsets, tile_expert, tile_row, num_tiles));
    SQ_RETURN_LAUNCH();
}

}  // extern "C"
