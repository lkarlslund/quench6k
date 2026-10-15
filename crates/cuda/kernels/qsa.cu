// QSA attention (SPEC §6): projections' epilogues, caches, the group indexer and selected attention.
//
// Per-sequence, per-layer cache (indexed by absolute position p):
//   K, V        BF16 [ctx, 2, 256]
//   raw index   BF16 [ctx, 128]
//   pooled      BF16 [ctx / 4, 128]   (rotated, normalized group keys; group g = positions 4g..4g+3)
// and per sequence: pos3 I32 [ctx, 3] MRoPE positions (t, h, w).
// A step processes T new tokens at positions start .. start + T - 1.

#include "common.cuh"

namespace {

constexpr int QH = 24, KVH = 2, HD = 256, ROT = 64, IH = 4, ID = 128, GROUP = 4, TOPG = 512;
constexpr float EPS = 1e-6f;

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
                                float* q, __nv_bfloat16* kcache, __nv_bfloat16* vcache, const int* __restrict__ pos3,
                                const int* __restrict__ start_ptr) {
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
        kcache[((size_t)p * KVH + h) * HD + i] = f2bf(x[i]);
        vcache[((size_t)p * KVH + h) * HD + i] = f2bf(kv[(size_t)t * ld + KVH * HD + h * HD + i]);
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

constexpr int ATT_THREADS = 256, KT = 32, QPG = QH / KVH;  // 12 query heads per KV head

// Selected attention for one (query t, KV head). Keys: every position 0..p when the query has at most
// 512 complete groups, else the 4 positions of each selected group plus the incomplete tail group.
// out[t, head, :] = softmax(q k / 16) v * sigmoid(gate), written as BF16 for o_proj.
__global__ void __launch_bounds__(ATT_THREADS) attention_kernel(const float* __restrict__ q, const float* __restrict__ qg, int ld,
                                                                 const __nv_bfloat16* __restrict__ kcache,
                                                                 const __nv_bfloat16* __restrict__ vcache,
                                                                 const int* __restrict__ sel,
                                                                 const int* __restrict__ start_ptr, __nv_bfloat16* out) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float sq[QPG][HD];
    __shared__ __nv_bfloat16 sk[KT][HD], sv[KT][HD];
    __shared__ float sp[QPG][KT];
    __shared__ float smax[QPG], ssum[QPG], sscale[QPG];
    __shared__ int keys[KT];

    const int t = blockIdx.x, kvh = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int p = *start_ptr + t;
    const int complete = (p + 1) / GROUP;
    const bool dense = complete <= TOPG;
    const int tail_begin = complete * GROUP;
    const int nkeys = dense ? p + 1 : TOPG * GROUP + (p + 1 - tail_begin);

    for (int i = tid; i < QPG * HD; i += ATT_THREADS)
        sq[i / HD][i % HD] = q[(size_t)t * QH * HD + (kvh * QPG + i / HD) * HD + i % HD];
    if (tid < QPG) {
        smax[tid] = -INFINITY;
        ssum[tid] = 0;
    }
    float acc[QPG] = {};
    __syncthreads();

    for (int k0 = 0; k0 < nkeys; k0 += KT) {
        const int nk = min(KT, nkeys - k0);
        if (tid < nk) {
            const int i = k0 + tid;
            keys[tid] = dense ? i : (i < TOPG * GROUP ? sel[(size_t)t * TOPG + i / GROUP] * GROUP + i % GROUP
                                                      : tail_begin + i - TOPG * GROUP);
        }
        __syncthreads();
        for (int i = tid; i < nk * HD / 8; i += ATT_THREADS) {
            const int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
            const size_t src = ((size_t)keys[r] * KVH + kvh) * HD + c;
            *reinterpret_cast<uint4*>(&sk[r][c]) = *reinterpret_cast<const uint4*>(kcache + src);
            *reinterpret_cast<uint4*>(&sv[r][c]) = *reinterpret_cast<const uint4*>(vcache + src);
        }
        __syncthreads();
        // scores: warp w handles keys w, w+8, w+16, w+24 for all 12 heads
        for (int r = warp; r < nk; r += ATT_THREADS / 32) {
            float part[QPG] = {};
            for (int d = lane; d < HD; d += 32) {
                const float kf = bf2f(sk[r][d]);
#pragma unroll
                for (int h = 0; h < QPG; ++h) part[h] += sq[h][d] * kf;
            }
#pragma unroll
            for (int h = 0; h < QPG; ++h) {
                const float s = warp_sum(part[h]);
                if (lane == 0) sp[h][r] = s * (1.0f / 16.0f);
            }
        }
        __syncthreads();
        // online softmax update: warp h < 12 handles head h
        if (warp < QPG / 1 && warp < 8) {
            for (int h = warp; h < QPG; h += 8) {
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
        }
        __syncthreads();
        // PV: thread owns dim tid for all 12 heads
#pragma unroll
        for (int h = 0; h < QPG; ++h) acc[h] *= sscale[h];
        for (int r = 0; r < nk; ++r) {
            const float vf = bf2f(sv[r][tid]);
#pragma unroll
            for (int h = 0; h < QPG; ++h) acc[h] += sp[h][r] * vf;
        }
        __syncthreads();
    }
#pragma unroll
    for (int h = 0; h < QPG; ++h) {
        const int head = kvh * QPG + h;
        const float gate = qg[(size_t)t * ld + head * 2 * HD + HD + tid];
        out[(size_t)t * QH * HD + head * HD + tid] = f2bf(acc[h] / ssum[h] * sigmoidf(gate));
    }
}


// ---- split-key attention for small batches -------------------------------------------------
// A query attends to at most 512 * 4 + 3 = 2051 keys, so SPLITS chunks of CHUNK keys always cover it.
constexpr int CHUNK = 32, SPLITS = (TOPG * GROUP + GROUP - 1 + CHUNK - 1) / CHUNK;

SQ_DEVICE int key_at(int i, bool dense, const int* sel_row, int tail_begin) {
    return dense ? i : (i < TOPG * GROUP ? sel_row[i / GROUP] * GROUP + i % GROUP : tail_begin + i - TOPG * GROUP);
}

// Partial attention of one (query t, KV head, key chunk): writes the chunk's running max and sum per
// head and the unnormalized output [12, 256] to part[(t, kvh, split)].
__global__ void __launch_bounds__(ATT_THREADS) attention_split_kernel(const float* __restrict__ q,
                                                                       const __nv_bfloat16* __restrict__ kcache,
                                                                       const __nv_bfloat16* __restrict__ vcache,
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
        for (int i = tid; i < nk * HD / 8; i += ATT_THREADS) {
            const int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
            const size_t src = ((size_t)keys[r] * KVH + kvh) * HD + c;
            *reinterpret_cast<uint4*>(&sk[r][c]) = *reinterpret_cast<const uint4*>(kcache + src);
            *reinterpret_cast<uint4*>(&sv[r][c]) = *reinterpret_cast<const uint4*>(vcache + src);
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
                                           cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(qkv_prep_kernel, T * (QH + KVH), HD, 0, st, qg, kv, ld, static_cast<const __nv_bfloat16*>(q_norm),
                                                   static_cast<const __nv_bfloat16*>(k_norm), q,
                                                   static_cast<__nv_bfloat16*>(kcache), static_cast<__nv_bfloat16*>(vcache),
                                                   pos3, start));
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
    SQ_CHECK_LAUNCH(sq_launch(index_score_kernel, dim3(T, (ld + 255) / 256), 256, 0, st, iq, static_cast<const __nv_bfloat16*>(pooled), scores,
                                                                  ld, start));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_qsa_attention(const float* q, const float* qg, int ld, const void* kcache, const void* vcache,
                                            const int* sel, const int* start, int T, void* out, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(attention_kernel, dim3(T, KVH), ATT_THREADS, 0, st, q, qg, ld, static_cast<const __nv_bfloat16*>(kcache),
                                                           static_cast<const __nv_bfloat16*>(vcache), sel, start,
                                                           static_cast<__nv_bfloat16*>(out)));
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
                                                  const int* sel, const int* start, int T, float* part, void* out,
                                                  cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(attention_split_kernel, dim3(T, KVH, SPLITS), ATT_THREADS, 0, st, q, static_cast<const __nv_bfloat16*>(kcache),
                                                                        static_cast<const __nv_bfloat16*>(vcache), sel,
                                                                        start, part));
    SQ_CHECK_LAUNCH(sq_launch(attention_merge_kernel, dim3(T, QH), HD, 0, st, part, qg, ld, start, static_cast<__nv_bfloat16*>(out)));
    SQ_RETURN_LAUNCH();
}
