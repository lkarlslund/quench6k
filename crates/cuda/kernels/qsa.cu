#include <cstdlib>
// QSA attention (SPEC §6): projections' epilogues, caches, the group indexer and selected attention.
//
// Per-sequence, per-layer cache (indexed by absolute position p):
//   K, V        BF16 [ctx, 2, 256], or with FP8 KV rows of 272 bytes per (position, KV head): 256 E4M3
//               values, then their FP32 scale (amax / 448) and 12 bytes of padding
//   raw index   BF16 [ctx, 128]
//   pooled      BF16 [ctx / 4, 128]   (rotated, normalized group keys; group g = positions 4g..4g+3)
// and per sequence: pos3 I32 [ctx, 3] MRoPE positions (t, h, w).
// A step processes T new tokens at positions start .. start + T - 1.

#include <cuda_fp8.h>

#include "mma.cuh"

namespace {

constexpr int QH = 24, KVH = 2, HD = 256, ROT = 64, IH = 4, ID = 128, GROUP = 4, TOPG = 512;
constexpr float EPS = 1e-6f;
constexpr int KV8_ROW = HD + 16;  // bytes per FP8 (position, KV head) row

// Address of the (position, KV head) row; BF16 rows hold HD values, FP8 rows KV8_ROW bytes.
template <bool FP8>
SQ_DEVICE const unsigned char* kv_row(const void* cache, int key, int kvh) {
    return static_cast<const unsigned char*>(cache) + ((size_t)key * KVH + kvh) * (FP8 ? KV8_ROW : HD * 2);
}

// Dequantizes 16 E4M3 values (row + col) to BF16 in dst: value * scale, rounded once to BF16.
SQ_DEVICE void kv8_load16(const unsigned char* row, int col, __nv_bfloat16* dst) {
    const uint4 raw = *reinterpret_cast<const uint4*>(row + col);
    const float scale = *reinterpret_cast<const float*>(row + HD);
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&raw);
    __align__(16) __nv_bfloat16 out[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        __nv_fp8_e4m3 v;
        v.__x = b[i];
        out[i] = f2bf(float(v) * scale);
    }
    reinterpret_cast<uint4*>(dst)[0] = reinterpret_cast<const uint4*>(out)[0];
    reinterpret_cast<uint4*>(dst)[1] = reinterpret_cast<const uint4*>(out)[1];
}

SQ_DEVICE int rope_axis(int pair) {
    if (pair % 3 == 1 && pair < 33) return 1;
    if (pair % 3 == 2 && pair < 30) return 2;
    return 0;
}

// Rotates x[0..64) of a vector held in shared memory by its MRoPE position; threads 0..31 each do one pair.
SQ_DEVICE void rope_inplace(float* x, const int* pos3) {
    const int i = threadIdx.x;
    if (i < ROT / 2) {
        const float inv_freq = 1.0f / powf(10000000.0f, (float)(2 * i) / ROT);
        const float angle = (float)pos3[rope_axis(i)] * inv_freq;
        float s, c;
        sincosf(angle, &s, &c);
        const float x1 = x[i], x2 = x[i + ROT / 2];
        x[i] = x1 * c - x2 * s;
        x[i + ROT / 2] = x2 * c + x1 * s;
    }
}

// Blocks (t, j): j < 24 normalizes and rotates query head j into q; j = 24, 25 normalize/rotate key head
// j - 24 into the K cache and store the matching V. 256 threads.
__global__ void qkv_prep_kernel(const float* __restrict__ qg, const float* __restrict__ kv, int ld,
                                const __nv_bfloat16* __restrict__ q_norm, const __nv_bfloat16* __restrict__ k_norm,
                                float* q, void* kcache, void* vcache, const int* __restrict__ pos3,
                                const int* __restrict__ start_ptr, int fp8) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float x[HD];
    const int start = *start_ptr;
    __shared__ float scratch[32];
    const int t = blockIdx.x / (QH + KVH), j = blockIdx.x % (QH + KVH), i = threadIdx.x;
    const int p = start + t;
    const bool is_q = j < QH;
    const float v = is_q ? qg[(size_t)t * ld + j * 2 * HD + i] : kv[(size_t)t * ld + (j - QH) * HD + i];
    const float r = rsqrtf(block_sum(v * v, scratch) / HD + EPS);
    x[i] = v * r * (1.0f + bf2f((is_q ? q_norm : k_norm)[i]));
    __syncthreads();
    rope_inplace(x, pos3 + 3 * p);
    __syncthreads();
    if (is_q) {
        q[(size_t)t * QH * HD + j * HD + i] = x[i];
    } else {
        const int h = j - QH;
        const float vv = kv[(size_t)t * ld + KVH * HD + h * HD + i];
        if (fp8) {
            const float ks = fmaxf(block_max(fabsf(x[i]), scratch), 1e-30f) / 448.f;
            const float vs = fmaxf(block_max(fabsf(vv), scratch), 1e-30f) / 448.f;
            unsigned char* krow = static_cast<unsigned char*>(kcache) + ((size_t)p * KVH + h) * KV8_ROW;
            unsigned char* vrow = static_cast<unsigned char*>(vcache) + ((size_t)p * KVH + h) * KV8_ROW;
            krow[i] = __nv_fp8_e4m3(x[i] / ks).__x;
            vrow[i] = __nv_fp8_e4m3(vv / vs).__x;
            if (i == 0) {
                *reinterpret_cast<float*>(krow + HD) = ks;
                *reinterpret_cast<float*>(vrow + HD) = vs;
            }
        } else {
            static_cast<__nv_bfloat16*>(kcache)[((size_t)p * KVH + h) * HD + i] = f2bf(x[i]);
            static_cast<__nv_bfloat16*>(vcache)[((size_t)p * KVH + h) * HD + i] = f2bf(vv);
        }
    }
}

// Blocks (t, j): j < 4 normalizes/rotates indexer query head j into iq; j = 4 stores the raw key. 128 threads.
__global__ void index_prep_kernel(const float* __restrict__ idx, int ld, const __nv_bfloat16* __restrict__ q_ln, float* iq,
                                  __nv_bfloat16* raw, const int* __restrict__ pos3, const int* __restrict__ start_ptr) {
    pdl_launch_dependents();
    pdl_wait();
    const int start = *start_ptr;
    __shared__ float x[ID];
    __shared__ float scratch[32];
    const int t = blockIdx.x / (IH + 1), j = blockIdx.x % (IH + 1), i = threadIdx.x;
    const int p = start + t;
    const float v = idx[(size_t)t * ld + j * ID + i];
    if (j == IH) {
        raw[(size_t)p * ID + i] = f2bf(v);
        return;
    }
    const float r = rsqrtf(block_sum(v * v, scratch) / ID + EPS);
    x[i] = v * r * (1.0f + bf2f(q_ln[i]));
    __syncthreads();
    rope_inplace(x, pos3 + 3 * p);
    __syncthreads();
    iq[(size_t)t * IH * ID + j * ID + i] = x[i];
}

// Pools every group completed by this step's tokens (g in [start / 4, (start + T) / 4)): mean of 4 raw
// keys (rounded to BF16), RMSNorm, RoPE at 4g. Launched with T / 4 + 1 blocks; extra blocks exit.
__global__ void pool_kernel(const __nv_bfloat16* __restrict__ raw, const __nv_bfloat16* __restrict__ k_ln,
                            __nv_bfloat16* pooled, const int* __restrict__ pos3, const int* __restrict__ start_ptr,
                            int T) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float x[ID];
    __shared__ float scratch[32];
    const int start = *start_ptr;
    const int g = start / GROUP + blockIdx.x, i = threadIdx.x;
    if (g >= (start + T) / GROUP) return;
    float m = 0;
#pragma unroll
    for (int j = 0; j < GROUP; ++j) m += bf2f(raw[(size_t)(g * GROUP + j) * ID + i]);
    m = bf2f(f2bf(m / GROUP));
    const float r = rsqrtf(block_sum(m * m, scratch) / ID + EPS);
    x[i] = bf2f(f2bf(m * r * (1.0f + bf2f(k_ln[i]))));
    __syncthreads();
    rope_inplace(x, pos3 + 3 * (g * GROUP));
    __syncthreads();
    pooled[(size_t)g * ID + i] = f2bf(x[i]);
}

// scores[t, g] = sum_h relu(iq[t, h] . pooled[g]) / sqrt(128) for complete groups g < (p + 1) / 4,
// -inf for the rest of the row (row stride `ld`). Grid (T, ceil(ld / 256)), 256 threads.
__global__ void index_score_kernel(const float* __restrict__ iq, const __nv_bfloat16* __restrict__ pooled, float* scores,
                                   int ld, const int* __restrict__ start_ptr) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float q[IH * ID];
    const int start = *start_ptr;
    const int t = blockIdx.x, g = blockIdx.y * blockDim.x + threadIdx.x;
    const int complete = (start + t + 1) / GROUP;
    if (complete <= TOPG || (int)(blockIdx.y * blockDim.x) >= complete) return;  // dense, or past the row
    for (int i = threadIdx.x; i < IH * ID; i += blockDim.x) q[i] = iq[(size_t)t * IH * ID + i];
    __syncthreads();
    if (g >= ld || g >= complete) return;
    float out = -INFINITY;
    if (g < complete) {
        float acc[IH] = {};
        const __nv_bfloat16* k = pooled + (size_t)g * ID;
        for (int d = 0; d < ID; d += 8) {
            const uint4 packed = *reinterpret_cast<const uint4*>(k + d);
            const __nv_bfloat16* kv = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float kf = bf2f(kv[e]);
#pragma unroll
                for (int h = 0; h < IH; ++h) acc[h] += q[h * ID + d + e] * kf;
            }
        }
        out = 0;
#pragma unroll
        for (int h = 0; h < IH; ++h) out += fmaxf(acc[h], 0.0f);
        out *= rsqrtf((float)ID);
    }
    scores[(size_t)t * ld + g] = out;
}

// Prefill scores on the tensor cores: tile of 64 query rows x 128 groups, per indexer head
// C_h = q_h . pooled^T with q split into hi + lo BF16 parts (pooled is BF16, so the products keep about
// 16 mantissa bits), FP32 accumulation; score = sum_h relu(C_h) / sqrt(128), written for the groups a
// row can select (g < complete, rows past the dense threshold). 8 warps of 32 x 32.
constexpr int ST_BM = 64, ST_BN = 128, ST_LD = ID + 8;
constexpr size_t SCORE_TC_SMEM = (size_t)(ST_BN + 2 * ST_BM) * ST_LD * 2;
__global__ void __launch_bounds__(256) index_score_tc_kernel(const float* __restrict__ iq, const __nv_bfloat16* __restrict__ pooled,
                                                             float* scores, int ld, const int* __restrict__ start_ptr, int T) {
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ __align__(16) unsigned char sc_smem[];
    __nv_bfloat16* Bs = reinterpret_cast<__nv_bfloat16*>(sc_smem);  // [ST_BN][ST_LD]
    __nv_bfloat16* Ah = Bs + ST_BN * ST_LD;                          // [ST_BM][ST_LD]
    __nv_bfloat16* Al = Ah + ST_BM * ST_LD;
    const int start = *start_ptr, t0 = blockIdx.y * ST_BM, g0 = blockIdx.x * ST_BN;
    const int cmax = (start + min(t0 + ST_BM, T)) / GROUP;
    if (cmax <= TOPG || g0 >= cmax) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, warp_m = warp >> 2, warp_n = warp & 3;
    for (int c = tid; c < ST_BN * (ID / 8); c += 256) {
        const int r = c / (ID / 8), col = (c % (ID / 8)) * 8, g = g0 + r;
        const bool ok = g < cmax && g < ld;
        cp_async16(Bs + r * ST_LD + col, pooled + (size_t)(ok ? g : 0) * ID + col, ok);
    }
    cp_async_commit();
    float o[2][4][4] = {};
    for (int h = 0; h < IH; ++h) {
        __syncthreads();  // the previous head's fragments are read
        for (int c = tid; c < ST_BM * (ID / 4); c += 256) {
            const int r = c / (ID / 4), col = (c % (ID / 4)) * 4, t = t0 + r;
            const float4 v = t < T ? *reinterpret_cast<const float4*>(iq + (size_t)t * IH * ID + h * ID + col) : make_float4(0.f, 0.f, 0.f, 0.f);
            const float f[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const __nv_bfloat16 hi = f2bf(f[e]);
                Ah[r * ST_LD + col + e] = hi;
                Al[r * ST_LD + col + e] = f2bf(f[e] - bf2f(hi));
            }
        }
        cp_async_wait<0>();
        __syncthreads();
        float c4[2][4][4] = {};
#pragma unroll
        for (int kk = 0; kk < ID; kk += 16) {
            uint32_t ah[2][4], al[2][4], b[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = warp_m * 32 + i * 16 + (lane & 15), c = kk + (lane >> 4) * 8;
                ldmatrix_x4(ah[i], Ah + r * ST_LD + c);
                ldmatrix_x4(al[i], Al + r * ST_LD + c);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j)
                ldmatrix_x4(b[j], Bs + (warp_n * 32 + j * 16 + (lane & 7) + ((lane >> 4) << 3)) * ST_LD + kk + ((lane >> 3) & 1) * 8);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma_bf16(c4[i][j], ah[i], &b[j >> 1][(j & 1) * 2]);
                    mma_bf16(c4[i][j], al[i], &b[j >> 1][(j & 1) * 2]);
                }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) o[i][j][e] += fmaxf(c4[i][j][e], 0.0f);
    }
    const int gq = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int t = t0 + warp_m * 32 + i * 16 + gq + h * 8;
            if (t >= T) continue;
            const int complete = (start + t + 1) / GROUP;
            if (complete <= TOPG) continue;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const int g = g0 + warp_n * 32 + j * 8 + t4 * 2 + x;
                    if (g < complete && g < ld) scores[(size_t)t * ld + g] = o[i][j][h * 2 + x] * rsqrtf((float)ID);
                }
        }
}

// Small batches (T <= 16): one thread per group g scores every row t from a single read of pooled[g],
// with the same per-row arithmetic as index_score_kernel. Grid ceil(ld / 256), 256 threads.
constexpr int SCORE_MAXT = 16;
__global__ void __launch_bounds__(256) index_score_small_kernel(const float* __restrict__ iq,
                                                                const __nv_bfloat16* __restrict__ pooled, float* scores,
                                                                int ld, const int* __restrict__ start_ptr, int T) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float q[SCORE_MAXT][IH * ID];
    const int start = *start_ptr;
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int last = (start + T) / GROUP;  // complete groups of the last row
    if (last <= TOPG || (int)(blockIdx.x * blockDim.x) >= last) return;
    for (int i = threadIdx.x; i < T * IH * ID; i += blockDim.x) q[i / (IH * ID)][i % (IH * ID)] = iq[i];
    __syncthreads();
    if (g >= ld) return;
    uint4 packed[ID / 8];
    if (g < last)
#pragma unroll
        for (int c = 0; c < ID / 8; ++c) packed[c] = *reinterpret_cast<const uint4*>(pooled + (size_t)g * ID + c * 8);
    for (int t = 0; t < T; ++t) {
        const int complete = (start + t + 1) / GROUP;
        if (complete <= TOPG) continue;  // dense row
        float out = -INFINITY;
        if (g < complete) {
            float acc[IH] = {};
#pragma unroll
            for (int c = 0; c < ID / 8; ++c) {
                const __nv_bfloat16* kv = reinterpret_cast<const __nv_bfloat16*>(&packed[c]);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float kf = bf2f(kv[e]);
#pragma unroll
                    for (int h = 0; h < IH; ++h) acc[h] += q[t][h * ID + c * 8 + e] * kf;
                }
            }
            out = 0;
#pragma unroll
            for (int h = 0; h < IH; ++h) out += fmaxf(acc[h], 0.0f);
            out *= rsqrtf((float)ID);
        }
        scores[(size_t)t * ld + g] = out;
    }
}

constexpr int ATT_THREADS = 256, KT = 32, QPG = QH / KVH;  // 12 query heads per KV head

constexpr int CHUNK = 32, SPLITS = (TOPG * GROUP + GROUP - 1 + CHUNK - 1) / CHUNK;

SQ_DEVICE int key_at(int i, bool dense, const int* sel_row, int tail_begin) {
    return dense ? i : (i < TOPG * GROUP ? sel_row[i / GROUP] * GROUP + i % GROUP : tail_begin + i - TOPG * GROUP);
}

// Partial attention of one (query t, KV head, key chunk): writes the chunk's running max and sum per
// head and the unnormalized output [12, 256] to part[(t, kvh, split)].
template <bool FP8>
__global__ void __launch_bounds__(ATT_THREADS) attention_split_kernel(const float* __restrict__ q,
                                                                       const void* __restrict__ kcache,
                                                                       const void* __restrict__ vcache,
                                                                       const int* __restrict__ sel,
                                                                       const int* __restrict__ start_ptr, float* part) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float sq[QPG][HD];
    __shared__ __nv_bfloat16 sk[KT][HD], sv[KT][HD];
    __shared__ float sp[QPG][KT];
    __shared__ float smax[QPG], ssum[QPG], sscale[QPG];
    __shared__ int keys[KT];

    const int t = blockIdx.x, kvh = blockIdx.y, split = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int p = *start_ptr + t;
    const int complete = (p + 1) / GROUP;
    const bool dense = complete <= TOPG;
    const int tail_begin = complete * GROUP;
    const int nkeys = dense ? p + 1 : TOPG * GROUP + (p + 1 - tail_begin);
    const int k_begin = split * CHUNK, k_end = min(nkeys, k_begin + CHUNK);
    const int* sel_row = sel + (size_t)t * TOPG;
    float* out = part + (((size_t)t * KVH + kvh) * SPLITS + split) * (2 * QPG + QPG * HD);

    if (k_begin >= k_end) {
        if (tid < QPG) {
            out[tid] = -INFINITY;
            out[QPG + tid] = 0;
        }
        return;
    }
    for (int i = tid; i < QPG * HD; i += ATT_THREADS)
        sq[i / HD][i % HD] = q[(size_t)t * QH * HD + (kvh * QPG + i / HD) * HD + i % HD];
    if (tid < QPG) {
        smax[tid] = -INFINITY;
        ssum[tid] = 0;
    }
    float acc[QPG] = {};
    __syncthreads();
    for (int k0 = k_begin; k0 < k_end; k0 += KT) {
        const int nk = min(KT, k_end - k0);
        if (tid < nk) keys[tid] = key_at(k0 + tid, dense, sel_row, tail_begin);
        __syncthreads();
        if constexpr (FP8) {
            for (int i = tid; i < nk * HD / 16; i += ATT_THREADS) {
                const int r = i / (HD / 16), c = (i % (HD / 16)) * 16;
                kv8_load16(kv_row<true>(kcache, keys[r], kvh), c, &sk[r][c]);
                kv8_load16(kv_row<true>(vcache, keys[r], kvh), c, &sv[r][c]);
            }
        } else {
            for (int i = tid; i < nk * HD / 8; i += ATT_THREADS) {
                const int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
                *reinterpret_cast<uint4*>(&sk[r][c]) = *reinterpret_cast<const uint4*>(kv_row<false>(kcache, keys[r], kvh) + c * 2);
                *reinterpret_cast<uint4*>(&sv[r][c]) = *reinterpret_cast<const uint4*>(kv_row<false>(vcache, keys[r], kvh) + c * 2);
            }
        }
        __syncthreads();
        for (int r = warp; r < nk; r += ATT_THREADS / 32) {
            float partv[QPG] = {};
            for (int d = lane; d < HD; d += 32) {
                const float kf = bf2f(sk[r][d]);
#pragma unroll
                for (int h = 0; h < QPG; ++h) partv[h] += sq[h][d] * kf;
            }
#pragma unroll
            for (int h = 0; h < QPG; ++h) {
                const float sc = warp_sum(partv[h]);
                if (lane == 0) sp[h][r] = sc * (1.0f / 16.0f);
            }
        }
        __syncthreads();
        for (int h = warp; h < QPG; h += ATT_THREADS / 32) {
            float m = lane < nk ? sp[h][lane] : -INFINITY;
            m = warp_max(m);
            const float new_max = fmaxf(smax[h], m);
            const float e = lane < nk ? __expf(sp[h][lane] - new_max) : 0.0f;
            if (lane < nk) sp[h][lane] = e;
            const float sum = warp_sum(e);
            if (lane == 0) {
                const float scale = __expf(smax[h] - new_max);
                sscale[h] = scale;
                ssum[h] = ssum[h] * scale + sum;
                smax[h] = new_max;
            }
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < QPG; ++h) acc[h] *= sscale[h];
        for (int r = 0; r < nk; ++r) {
            const float vf = bf2f(sv[r][tid]);
#pragma unroll
            for (int h = 0; h < QPG; ++h) acc[h] += sp[h][r] * vf;
        }
        __syncthreads();
    }
    if (tid < QPG) {
        out[tid] = smax[tid];
        out[QPG + tid] = ssum[tid];
    }
#pragma unroll
    for (int h = 0; h < QPG; ++h) out[2 * QPG + h * HD + tid] = acc[h];
}

// Merges the splits of (t, head) in split order and applies the output gate. grid (T, 24), 256 threads:
// warp 0 turns the used splits' (max, sum) pairs into weights in shared memory, then every thread sums
// its output dimension over the splits.
__global__ void attention_merge_kernel(const float* __restrict__ part, const float* __restrict__ qg, int ld,
                                       const int* __restrict__ start_ptr, __nv_bfloat16* out) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float weight[SPLITS];
    __shared__ float inv_l;
    const int t = blockIdx.x, head = blockIdx.y, kvh = head / QPG, h = head % QPG, d = threadIdx.x;
    const int p = *start_ptr + t, complete = (p + 1) / GROUP;
    const int nkeys = complete <= TOPG ? p + 1 : TOPG * GROUP + (p + 1 - complete * GROUP);
    const int used = (nkeys + CHUNK - 1) / CHUNK;
    constexpr int STRIDE = 2 * QPG + QPG * HD;
    const float* base = part + ((size_t)t * KVH + kvh) * SPLITS * STRIDE;
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        float m = -INFINITY;
        for (int s = lane; s < used; s += 32) m = fmaxf(m, base[s * STRIDE + h]);
        m = warp_max(m);
        float l = 0;
        for (int s = lane; s < used; s += 32) {
            const float w = __expf(base[s * STRIDE + h] - m);
            weight[s] = w;
            l += w * base[s * STRIDE + QPG + h];
        }
        l = warp_sum(l);
        if (lane == 0) inv_l = 1.0f / l;
    }
    __syncthreads();
    float o = 0;
#pragma unroll 8
    for (int s = 0; s < used; ++s) o += weight[s] * base[s * STRIDE + 2 * QPG + h * HD + d];
    const float gate = qg[(size_t)t * ld + head * 2 * HD + HD + d];
    out[(size_t)t * QH * HD + head * HD + d] = f2bf(o * inv_l * sigmoidf(gate));
}

}  // namespace

extern "C" cudaError_t squash_qsa_qkv_prep(const float* qg, const float* kv, int ld, const void* q_norm, const void* k_norm,
                                           float* q, void* kcache, void* vcache, const int* pos3, const int* start, int T,
                                           int fp8, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(qkv_prep_kernel, T * (QH + KVH), HD, 0, st, qg, kv, ld, static_cast<const __nv_bfloat16*>(q_norm),
                                                   static_cast<const __nv_bfloat16*>(k_norm), q, kcache, vcache, pos3, start, fp8));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_qsa_index_prep(const float* idx, int ld, const void* q_ln, float* iq, void* raw, const int* pos3,
                                             const int* start, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(index_prep_kernel, T * (IH + 1), ID, 0, st, idx, ld, static_cast<const __nv_bfloat16*>(q_ln), iq,
                                                   static_cast<__nv_bfloat16*>(raw), pos3, start));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_qsa_pool(const void* raw, const void* k_ln, void* pooled, const int* pos3, const int* start,
                                       int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(pool_kernel, T / GROUP + 1, ID, 0, st, static_cast<const __nv_bfloat16*>(raw),
                                              static_cast<const __nv_bfloat16*>(k_ln),
                                              static_cast<__nv_bfloat16*>(pooled), pos3, start, T));
    SQ_RETURN_LAUNCH();
}

// Scores rows that need group selection; `ld` must cover the largest possible group count.
extern "C" cudaError_t squash_qsa_index_scores(const float* iq, const void* pooled, float* scores, int ld,
                                               const int* start, int T, cudaStream_t st) {
    if (T == 0 || ld == 0) return cudaSuccess;
    if (T <= SCORE_MAXT) {
        SQ_CHECK_LAUNCH(sq_launch(index_score_small_kernel, (ld + 255) / 256, 256, 0, st, iq,
                                  static_cast<const __nv_bfloat16*>(pooled), scores, ld, start, T));
        SQ_RETURN_LAUNCH();
    }
    if (std::getenv("SQUASH_SCORE_FP32") == nullptr) {
        static bool configured = false;
        if (!configured) {
            const cudaError_t e = cudaFuncSetAttribute(index_score_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SCORE_TC_SMEM);
            if (e != cudaSuccess) return e;
            configured = true;
        }
        SQ_CHECK_LAUNCH(sq_launch(index_score_tc_kernel, dim3((ld + ST_BN - 1) / ST_BN, (T + ST_BM - 1) / ST_BM), 256, SCORE_TC_SMEM, st, iq,
                                  static_cast<const __nv_bfloat16*>(pooled), scores, ld, start, T));
        SQ_RETURN_LAUNCH();
    }
    SQ_CHECK_LAUNCH(sq_launch(index_score_kernel, dim3(T, (ld + 255) / 256), 256, 0, st, iq, static_cast<const __nv_bfloat16*>(pooled), scores,
                                                                  ld, start));
    SQ_RETURN_LAUNCH();
}

namespace {

// Tensor-core selected attention for prefill: one block (4 warps) per (query t, KV head). The 12
// query heads form the MMA rows (padded to 16); keys are streamed in tiles of 64 (the same key list
// as attention_kernel). FP32 values that meet BF16 tensor cores (q, the probabilities) are split
// into hi + lo BF16 parts, so products keep about 16 mantissa bits; accumulation is FP32.
// Split q and the probabilities into hi + lo BF16 parts (about 16 mantissa bits); false: single BF16.
constexpr bool ATT_SPLIT = false;
constexpr int MT = 32, MW = 4, MPAD = HD + 8, PPAD = MT + 8, NSPLIT = ATT_SPLIT ? 2 : 1;
// 32-key tiles and no lo buffers keep this at about 46 KB: two blocks per SM overlap their gathers.
constexpr size_t MMA_ATT_SMEM = (size_t)(NSPLIT * 16 * MPAD + 2 * MT * MPAD + NSPLIT * 16 * PPAD) * 2 + 16 * (MT + 4) * 4;
// FP8 caches: each thread prefetches its share of a tile's raw rows into registers (8 chunks of 16 values).
constexpr int KV8_CHUNKS = MT * HD / 16 / (MW * 32);

SQ_DEVICE void split_bf16(float x, __nv_bfloat16& hi, __nv_bfloat16& lo) {
    hi = f2bf(x);
    lo = f2bf(x - bf2f(hi));
}

template <bool FP8>
__global__ void __launch_bounds__(MW * 32) attention_mma_kernel(const float* __restrict__ q, const float* __restrict__ qg, int ld,
                                                                const void* __restrict__ kcache,
                                                                const void* __restrict__ vcache,
                                                                const int* __restrict__ sel,
                                                                const int* __restrict__ start_ptr, __nv_bfloat16* out) {
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* Qh = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [16][MPAD]
    __nv_bfloat16* Ql = Qh + (NSPLIT - 1) * 16 * MPAD;
    __nv_bfloat16* Ks = Ql + 16 * MPAD;  // [MT][MPAD]
    __nv_bfloat16* Vs = Ks + MT * MPAD;  // [MT][MPAD]
    __nv_bfloat16* Ph = Vs + MT * MPAD;  // [16][PPAD]
    __nv_bfloat16* Pl = Ph + (NSPLIT - 1) * 16 * PPAD;
    float* Ss = reinterpret_cast<float*>(Pl + 16 * PPAD);  // [16][MT + 4]
    __shared__ int keys[2][MT];
    __shared__ float rmax[16], rsum[16], rscale[16];

    const int t = blockIdx.x, kvh = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int p = *start_ptr + t;
    const int complete = (p + 1) / GROUP;
    const bool dense = complete <= TOPG;
    const int tail_begin = complete * GROUP;
    const int nkeys = dense ? p + 1 : TOPG * GROUP + (p + 1 - tail_begin);
    const int tiles = (nkeys + MT - 1) / MT;

    for (int i = tid; i < 16 * HD; i += MW * 32) {
        const int r = i / HD, d = i % HD;
        __nv_bfloat16 hi = f2bf(0.0f), lo = f2bf(0.0f);
        if (r < QPG) split_bf16(q[(size_t)t * QH * HD + (kvh * QPG + r) * HD + d], hi, lo);
        Qh[r * MPAD + d] = hi;
        if constexpr (ATT_SPLIT) Ql[r * MPAD + d] = lo;
    }
    if (tid < 16) {
        rmax[tid] = -INFINITY;
        rsum[tid] = 0.0f;
    }
    auto key_of = [&](int i) {
        return dense ? i : (i < TOPG * GROUP ? sel[(size_t)t * TOPG + i / GROUP] * GROUP + i % GROUP : tail_begin + i - TOPG * GROUP);
    };
    // Tile `tile`'s key list into keys[tile & 1] (tid < MT), then its K or V rows (cp.async, zero past nkeys).
    auto load_keys = [&](int tile) {
        if (tid < MT) {
            const int i = tile * MT + tid;
            keys[tile & 1][tid] = i < nkeys ? key_of(i) : -1;
        }
    };
    // FP8 rows go to registers (thread: rows tid / 16 + 8 i, columns (tid % 16) * 16 ..); dequant() then
    // fills dst. Chunk c = tid + i * 128 covers row c / 16.
    uint4 kraw[KV8_CHUNKS], vraw[KV8_CHUNKS];
    float kscl[KV8_CHUNKS], vscl[KV8_CHUNKS];
    auto load_rows = [&](int tile, __nv_bfloat16* dst, const void* cache) {
        if constexpr (FP8) {
            uint4* raw = dst == Ks ? kraw : vraw;
            float* scl = dst == Ks ? kscl : vscl;
#pragma unroll
            for (int i = 0; i < KV8_CHUNKS; ++i) {
                const int r = tid / 16 + i * (MW * 32 / 16), col = (tid % 16) * 16;
                const int key = keys[tile & 1][r];
                if (key >= 0) {
                    const unsigned char* row = kv_row<true>(cache, key, kvh);
                    raw[i] = *reinterpret_cast<const uint4*>(row + col);
                    scl[i] = *reinterpret_cast<const float*>(row + HD);
                } else {
                    raw[i] = make_uint4(0, 0, 0, 0);
                    scl[i] = 0.0f;
                }
            }
        } else {
            for (int c = tid; c < MT * HD / 8; c += MW * 32) {
                const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
                const int key = keys[tile & 1][r];
                cp_async16(dst + r * MPAD + col, kv_row<false>(cache, max(key, 0), kvh) + col * 2, key >= 0);
            }
        }
    };
    auto dequant = [&](__nv_bfloat16* dst) {
        if constexpr (FP8) {
            const uint4* raw = dst == Ks ? kraw : vraw;
            const float* scl = dst == Ks ? kscl : vscl;
#pragma unroll
            for (int i = 0; i < KV8_CHUNKS; ++i) {
                const int r = tid / 16 + i * (MW * 32 / 16), col = (tid % 16) * 16;
                const unsigned char* b = reinterpret_cast<const unsigned char*>(&raw[i]);
                __align__(16) __nv_bfloat16 v[16];
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    __nv_fp8_e4m3 e;
                    e.__x = b[j];
                    v[j] = f2bf(float(e) * scl[i]);
                }
                reinterpret_cast<uint4*>(dst + r * MPAD + col)[0] = reinterpret_cast<const uint4*>(v)[0];
                reinterpret_cast<uint4*>(dst + r * MPAD + col)[1] = reinterpret_cast<const uint4*>(v)[1];
            }
            __syncthreads();
        }
    };
    load_keys(0);
    __syncthreads();
    load_rows(0, Ks, kcache);
    cp_async_commit();
    load_rows(0, Vs, vcache);
    cp_async_commit();

    constexpr int ONT = HD / MW / 8;  // output n-tiles per warp
    float o[ONT][4] = {};
    const int g = lane >> 2, t4 = lane & 3;
    for (int tile = 0; tile < tiles; ++tile) {
        cp_async_wait<1>();  // K of this tile
        __syncthreads();
        dequant(Ks);
        // S = Q K^T / 16 for this warp's 8 keys (one n-tile), hi and lo parts of Q.
        float sacc[4] = {};
#pragma unroll
        for (int kk = 0; kk < HD; kk += 16) {
            uint32_t ah[4], al[4], b[2];
            ldmatrix_x4(ah, Qh + (lane & 15) * MPAD + kk + (lane >> 4) * 8);
            if constexpr (ATT_SPLIT) ldmatrix_x4(al, Ql + (lane & 15) * MPAD + kk + (lane >> 4) * 8);
            ldmatrix_x2(b, Ks + (warp * 8 + (lane & 7)) * MPAD + kk + ((lane >> 3) & 1) * 8);
            mma_bf16(sacc, ah, b);
            if constexpr (ATT_SPLIT) mma_bf16(sacc, al, b);
        }
        const int nk = min(MT, nkeys - tile * MT);
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int x = 0; x < 2; ++x) {
                const int col = warp * 8 + t4 * 2 + x;
                Ss[(g + h * 8) * (MT + 4) + col] = col < nk ? sacc[h * 2 + x] * (1.0f / 16.0f) : -INFINITY;
            }
        __syncthreads();
        // The next tile's keys and K rows load while this tile's softmax and P V run.
        if (tile + 1 < tiles) {
            load_keys(tile + 1);
            __syncthreads();
            load_rows(tile + 1, Ks, kcache);
        }
        cp_async_commit();
        // Online softmax: thread (row tid / 16, 4 keys).
        {
            constexpr int TPR = MW * 32 / 16, KPT = MT / TPR;  // threads per row, keys per thread
            const int r = tid / TPR, c0 = (tid % TPR) * KPT;
            float v[KPT], m = -INFINITY;
#pragma unroll
            for (int x = 0; x < KPT; ++x) {
                v[x] = Ss[r * (MT + 4) + c0 + x];
                m = fmaxf(m, v[x]);
            }
#pragma unroll
            for (int off = 1; off < TPR; off <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
            const float m_old = rmax[r], m_new = fmaxf(m_old, m);
            float sum = 0.0f;
#pragma unroll
            for (int x = 0; x < KPT; ++x) {
                const float e = m_new == -INFINITY ? 0.0f : __expf(v[x] - m_new);
                sum += e;
                __nv_bfloat16 hi, lo;
                split_bf16(e, hi, lo);
                Ph[r * PPAD + c0 + x] = hi;
                if constexpr (ATT_SPLIT) Pl[r * PPAD + c0 + x] = lo;
            }
#pragma unroll
            for (int off = 1; off < TPR; off <<= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
            __syncwarp();
            if ((tid % TPR) == 0) {
                const float scale = m_old == -INFINITY ? 0.0f : __expf(m_old - m_new);
                rscale[r] = scale;
                rsum[r] = rsum[r] * scale + sum;
                rmax[r] = m_new;
            }
        }
        cp_async_wait<1>();  // V of this tile
        __syncthreads();
        dequant(Vs);
        // O = O * scale + P V for this warp's HD / MW output columns.
        {
            const float s0 = rscale[g], s1 = rscale[g + 8];
#pragma unroll
            for (int nt = 0; nt < ONT; ++nt) {
                o[nt][0] *= s0;
                o[nt][1] *= s0;
                o[nt][2] *= s1;
                o[nt][3] *= s1;
            }
#pragma unroll
            for (int kk = 0; kk < MT; kk += 16) {
                uint32_t ah[4], al[4];
                ldmatrix_x4(ah, Ph + (lane & 15) * PPAD + kk + (lane >> 4) * 8);
                if constexpr (ATT_SPLIT) ldmatrix_x4(al, Pl + (lane & 15) * PPAD + kk + (lane >> 4) * 8);
#pragma unroll
                for (int nt = 0; nt < ONT; nt += 2) {
                    uint32_t b[4];
                    const int m = lane >> 3, rr = lane & 7;
                    ldmatrix_x4_trans(b, Vs + (kk + (m & 1) * 8 + rr) * MPAD + warp * (HD / MW) + nt * 8 + (m >> 1) * 8);
                    mma_bf16(o[nt], ah, &b[0]);
                    if constexpr (ATT_SPLIT) mma_bf16(o[nt], al, &b[0]);
                    mma_bf16(o[nt + 1], ah, &b[2]);
                    if constexpr (ATT_SPLIT) mma_bf16(o[nt + 1], al, &b[2]);
                }
            }
        }
        __syncthreads();
        if (tile + 1 < tiles) load_rows(tile + 1, Vs, vcache);
        cp_async_commit();
    }
    cp_async_wait<0>();
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = g + h * 8;
        if (r >= QPG) continue;
        const int head = kvh * QPG + r;
        const float inv = 1.0f / rsum[r];
#pragma unroll
        for (int nt = 0; nt < ONT; ++nt)
#pragma unroll
            for (int x = 0; x < 2; ++x) {
                const int d = warp * (HD / MW) + nt * 8 + t4 * 2 + x;
                const float gate = qg[(size_t)t * ld + head * 2 * HD + HD + d];
                out[(size_t)t * QH * HD + head * HD + d] = f2bf(o[nt][h * 2 + x] * inv * sigmoidf(gate));
            }
    }
}

}  // namespace

extern "C" cudaError_t squash_qsa_attention(const float* q, const float* qg, int ld, const void* kcache, const void* vcache,
                                            const int* sel, const int* start, int T, int fp8, void* out, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    static bool configured = false;
    if (!configured) {
        for (auto kern : {attention_mma_kernel<false>, attention_mma_kernel<true>}) {
            const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)MMA_ATT_SMEM);
            if (e != cudaSuccess) return e;
        }
        configured = true;
    }
    SQ_CHECK_LAUNCH(sq_launch(fp8 ? attention_mma_kernel<true> : attention_mma_kernel<false>, dim3(T, KVH), MW * 32, MMA_ATT_SMEM, st,
                              q, qg, ld, kcache, vcache, sel, start, static_cast<__nv_bfloat16*>(out)));
    SQ_RETURN_LAUNCH();
}

namespace {

// Writes this step's T positions (3 axes each) into the sequence's position table at start ...
__global__ void store_positions_kernel(const int* __restrict__ src, int* pos3, const int* __restrict__ start_ptr, int T) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 3 * T) pos3[3 * *start_ptr + i] = src[i];
}

}  // namespace

extern "C" cudaError_t squash_store_positions(const int* src, int* pos3, const int* start, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(store_positions_kernel, (3 * T + 255) / 256, 256, 0, st, src, pos3, start, T));
    SQ_RETURN_LAUNCH();
}

extern "C" size_t squash_qsa_split_floats(int T) { return (size_t)T * KVH * SPLITS * (2 * QPG + QPG * HD); }

// Attention for small batches: key-split partials into `part` (squash_qsa_split_floats(T) floats), then merge.
extern "C" cudaError_t squash_qsa_attention_split(const float* q, const float* qg, int ld, const void* kcache, const void* vcache,
                                                  const int* sel, const int* start, int T, int fp8, float* part, void* out,
                                                  cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(fp8 ? attention_split_kernel<true> : attention_split_kernel<false>, dim3(T, KVH, SPLITS), ATT_THREADS, 0, st,
                              q, kcache, vcache, sel, start, part));
    SQ_CHECK_LAUNCH(sq_launch(attention_merge_kernel, dim3(T, QH), HD, 0, st, part, qg, ld, start, static_cast<__nv_bfloat16*>(out)));
    SQ_RETURN_LAUNCH();
}
