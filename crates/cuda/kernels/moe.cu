// Sparse MoE (SPEC §7): routing, expert grouping, NVFP4 grouped GEMM (W4A16) and the combine.
//
// A step routes T tokens to 10 experts each: P = 10 T (token, slot) pairs, pair = t * 10 + slot.
// Pairs are grouped by expert into "rows" (row order within an expert is arbitrary); every expert
// GEMM writes one output row per pair, and the combine reads them back in fixed slot order, so the
// result does not depend on that arbitrary order.

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "mma.cuh"
#include "wfmt.cuh"

namespace {

constexpr int E = 512, TOPK = 10, D = 2560;

// One warp per token: softmax over 512 router logits, top 10 (ties: lower expert), renormalized.
__global__ void route_kernel(const float* __restrict__ logits, int ld, int T, int* top_i, float* top_w) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (t >= T) return;
    float v[E / 32];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < E / 32; ++j) {
        v[j] = logits[(size_t)t * ld + j * 32 + lane];
        mx = fmaxf(mx, v[j]);
    }
    mx = warp_max(mx);
    float sum = 0;
#pragma unroll
    for (int j = 0; j < E / 32; ++j) {
        v[j] = __expf(v[j] - mx);
        if (!(v[j] >= 0.0f)) v[j] = 0.0f;  // NaN logits: never pick an out-of-range expert
        sum += v[j];
    }
    sum = warp_sum(sum);
    float chosen_sum = 0, chosen[TOPK];
    int chosen_i[TOPK];
    for (int r = 0; r < TOPK; ++r) {
        float best = -1;
        int best_i = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < E / 32; ++j)
            if (v[j] > best) {
                best = v[j];
                best_i = j * 32 + lane;
            }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffff, best, o);
            const int oi = __shfl_xor_sync(0xffffffff, best_i, o);
            if (ob > best || (ob == best && oi < best_i)) {
                best = ob;
                best_i = oi;
            }
        }
        chosen[r] = best / sum;
        chosen_i[r] = best_i;
        chosen_sum += chosen[r];
        if ((best_i & 31) == lane) v[best_i / 32] = -2;  // remove from further rounds
    }
    if (lane == 0)
        for (int r = 0; r < TOPK; ++r) {
            top_i[t * TOPK + r] = chosen_i[r];
            top_w[t * TOPK + r] = chosen[r] / chosen_sum;
        }
}

__global__ void count_kernel(const int* __restrict__ top_i, int pairs, int* counts) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) atomicAdd(&counts[top_i[i]], 1);
}

// One block of 512 threads: expert row offsets, and the list of BM-row tiles (expert, first row).
__global__ void plan_kernel(const int* __restrict__ counts, int* offsets, int* cursor, int* tile_expert, int* tile_row,
                            int* num_tiles, int bm) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ int s_rows[E], s_tiles[E];
    const int e = threadIdx.x;
    const int c = counts[e];
    s_rows[e] = c;
    s_tiles[e] = (c + bm - 1) / bm;
    __syncthreads();
    // Simple inclusive scans (512 elements, log steps).
    for (int off = 1; off < E; off <<= 1) {
        const int a = e >= off ? s_rows[e - off] : 0, b = e >= off ? s_tiles[e - off] : 0;
        __syncthreads();
        s_rows[e] += a;
        s_tiles[e] += b;
        __syncthreads();
    }
    const int row_begin = s_rows[e] - c, tile_begin = s_tiles[e] - (c + bm - 1) / bm;
    offsets[e] = row_begin;
    cursor[e] = row_begin;
    if (e == E - 1) {
        offsets[E] = s_rows[e];
        *num_tiles = s_tiles[e];
    }
    for (int i = 0; i * bm < c; ++i) {
        tile_expert[tile_begin + i] = e;
        tile_row[tile_begin + i] = row_begin + i * bm;
    }
}

__global__ void scatter_kernel(const int* __restrict__ top_i, int pairs, int* cursor, int* pair_row) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) pair_row[i] = atomicAdd(&cursor[top_i[i]], 1);
}

// xs[pair_row[pair]] = m[pair / 10]
__global__ void gather_kernel(const __nv_bfloat16* __restrict__ m, const int* __restrict__ pair_row, int pairs,
                              __nv_bfloat16* xs) {
    pdl_launch_dependents();
    pdl_wait();
    const int pair = blockIdx.x;
    const uint4* src = reinterpret_cast<const uint4*>(m + (size_t)(pair / TOPK) * D);
    uint4* dst = reinterpret_cast<uint4*>(xs + (size_t)pair_row[pair] * D);
    for (int i = threadIdx.x; i < D / 8; i += blockDim.x) dst[i] = src[i];
}

// ---- NVFP4 (W4A16) grouped GEMM ------------------------------------------------------------
// Y[row, N] = ws2[e] * Xs[row, K] . dequant(W_e)[N, K]^T for every row of expert e's tiles.
// W_e: codes U8 [N, K/2] (low nibble first), scales F8E4M3 [N, K/16].

constexpr int BM = 64, BN = 128, BK = 32, STAGES = 2, THREADS = 256, LDS = BK + 8;

SQ_DEVICE float fp8_to_float(uint8_t v) {
    __nv_fp8_e4m3 f;
    f.__x = v;
    return float(f);
}

__global__ void __launch_bounds__(THREADS) nvfp4_gemm_kernel(const __nv_bfloat16* __restrict__ X,
                                                             const uint64_t* __restrict__ w_ptr,
                                                             const uint64_t* __restrict__ s_ptr,
                                                             const float* __restrict__ ws2, const int* __restrict__ offsets,
                                                             const int* __restrict__ tile_expert,
                                                             const int* __restrict__ tile_row,
                                                             const int* __restrict__ num_tiles, float* Y, int N, int K) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ __align__(16) __nv_bfloat16 sA[STAGES][BM * LDS];
    __shared__ __align__(16) __nv_bfloat16 sB[STAGES][BN * LDS];
    if ((int)blockIdx.y >= *num_tiles) return;
    const int e = tile_expert[blockIdx.y], r0 = tile_row[blockIdx.y];
    const int r_end = min(r0 + BM, offsets[e + 1]);
    const uint8_t* W = reinterpret_cast<const uint8_t*>(w_ptr[e]);
    const uint8_t* Sc = reinterpret_cast<const uint8_t*>(s_ptr[e]);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp >> 2, warp_n = warp & 3;
    const int n0 = blockIdx.x * BN;
    const int k_tiles = K / BK;

    // A tiles via cp.async; B (FP4 codes + FP8 scales) is fetched into registers one k-tile ahead and
    // decoded into shared memory after the current tile's MMAs.
    auto load_a = [&](int stage, int kt) {
        const int k = kt * BK;
        const int r = tid >> 2, c = (tid & 3) * 8;
        const bool ok = r0 + r < r_end;
        cp_async16(&sA[stage][r * LDS + c], X + (size_t)(ok ? r0 + r : r0) * K + k + c, ok);
    };
    uint4 bcodes = make_uint4(0, 0, 0, 0);
    uint16_t bsc = 0;
    const bool b_row = tid < BN && n0 + tid < N;
    auto fetch_b = [&](int kt) {
        if (b_row) {
            const int n = n0 + tid, k = kt * BK;
            bcodes = *reinterpret_cast<const uint4*>(W + (size_t)n * (K / 2) + k / 2);
            bsc = *reinterpret_cast<const uint16_t*>(Sc + (size_t)n * (K / 16) + k / 16);
        }
    };
    auto store_b = [&](int stage) {
        if (tid >= BN) return;
        __nv_bfloat16* dst = &sB[stage][tid * LDS];
        uint32_t packed[16];
        if (b_row) {
            // FP4 x FP8 products are exact in FP16 and then in BF16 (hardware conversions).
            const __half h0 = __float2half(fp8_to_float(bsc & 0xff)), h1 = __float2half(fp8_to_float(bsc >> 8));
            const __half2 s0 = __halves2half2(h0, h0), s1 = __halves2half2(h1, h1);
            const uint8_t* bb = reinterpret_cast<const uint8_t*>(&bcodes);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const __half2_raw r = __nv_cvt_fp4x2_to_halfraw2(bb[i], __NV_E2M1);
                const float2 f = __half22float2(__hmul2(*reinterpret_cast<const __half2*>(&r), i < 8 ? s0 : s1));
                const __nv_bfloat162 v = __floats2bfloat162_rn(f.x, f.y);
                packed[i] = *reinterpret_cast<const uint32_t*>(&v);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) packed[i] = 0;
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
            load_a((kt + 1) % STAGES, kt + 1);
            fetch_b(kt + 1);
        }
        cp_async_commit();
        const __nv_bfloat16* a_tile = sA[kt % STAGES];
        const __nv_bfloat16* b_tile = sB[kt % STAGES];
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[2][4], b[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                ldmatrix_x4(a[i], &a_tile[(warp_m * 32 + i * 16 + (lane & 15)) * LDS + kk + (lane >> 4) * 8]);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                ldmatrix_x4(b[j], &b_tile[(warp_n * 32 + j * 16 + (lane & 7) + ((lane >> 4) << 3)) * LDS + kk +
                                          ((lane >> 3) & 1) * 8]);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_bf16(acc[i][j], a[i], &b[j >> 1][(j & 1) * 2]);
        }
        // The other stage was last read by the previous tile's MMAs, which every thread finished
        // before this tile's barrier.
        if (more) store_b((kt + 1) % STAGES);
    }
    const float alpha = ws2[e];
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
                if (n < N) Y[(size_t)r * N + n] = alpha * acc[i][j][h * 2];
                if (n + 1 < N) Y[(size_t)r * N + n + 1] = alpha * acc[i][j][h * 2 + 1];
            }
}

// act[r, i] = bf16(silu(gate[r * ld + i]) * up[r * ld + i]) for i < width
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, int ld, int width,
                              __nv_bfloat16* act, size_t count) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= count) return;
    const size_t r = i / width, c = i % width;
    act[i] = f2bf(silu(gate[r * ld + c]) * up[r * ld + c]);
}

// y[t, d] = sum_k w[t, k] * Yd[pair_row[t*10+k], d] + sigmoid(sg[t]) * sh[t, d]
template <typename YT>
__global__ void combine_kernel(const YT* __restrict__ yd, const int* __restrict__ pair_row,
                               const float* __restrict__ top_w, const float* __restrict__ sg, int ld_sg,
                               const float* __restrict__ sh, float* y) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.x;
    const float g = sigmoidf(sg[(size_t)t * ld_sg]);
    for (int d = blockIdx.y * blockDim.x + threadIdx.x; d < D; d += gridDim.y * blockDim.x) {
        float acc = 0;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) acc += top_w[t * TOPK + k] * (float)yd[(size_t)pair_row[t * TOPK + k] * D + d];
        y[(size_t)t * D + d] = acc + g * sh[(size_t)t * D + d];
    }
}


// Compacts the experts with at least one routed row into active[0 .. *num_active). One block of 512.
__global__ void active_kernel(const int* __restrict__ counts, int* active, int* num_active) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ int s[E];
    const int e = threadIdx.x;
    const int flag = counts[e] > 0;
    s[e] = flag;
    __syncthreads();
    for (int off = 1; off < E; off <<= 1) {
        const int a = e >= off ? s[e - off] : 0;
        __syncthreads();
        s[e] += a;
        __syncthreads();
    }
    if (flag) active[s[e] - 1] = e;
    if (e == E - 1) *num_active = s[e];
}

constexpr int SMALL_T = 16, SMALL_PAIRS = SMALL_T * TOPK;

// One block of 512 threads for T <= 16 tokens: routing (softmax, top 10, renormalize) and grouping of
// the pairs by expert. Outputs top_i/top_w [T, 10], offsets [513], pair_row [10 T], row_token [10 T],
// active [<= 10 T] and num_active.
__global__ void __launch_bounds__(512) route_small_kernel(const float* __restrict__ logits, int ld, int T, int* top_i,
                                                          float* top_w, int* offsets, int* pair_row, int* row_token,
                                                          int* active, int* num_active) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ int s_top[SMALL_PAIRS];
    __shared__ int s_count[E], s_scan[E];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    s_count[tid] = 0;
    __syncthreads();
    // warp w routes token w (16 warps)
    if (warp < T) {
        const int t = warp;
        float v[E / 32];
        float mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < E / 32; ++j) {
            v[j] = logits[(size_t)t * ld + j * 32 + lane];
            mx = fmaxf(mx, v[j]);
        }
        mx = warp_max(mx);
        float sum = 0;
#pragma unroll
        for (int j = 0; j < E / 32; ++j) {
            v[j] = __expf(v[j] - mx);
            if (!(v[j] >= 0.0f)) v[j] = 0.0f;  // NaN logits: never pick an out-of-range expert
            sum += v[j];
        }
        sum = warp_sum(sum);
        float chosen[TOPK], chosen_sum = 0;
        int chosen_i[TOPK];
        for (int r = 0; r < TOPK; ++r) {
            float best = -1;
            int best_i = 0x7fffffff;
#pragma unroll
            for (int j = 0; j < E / 32; ++j)
                if (v[j] > best) {
                    best = v[j];
                    best_i = j * 32 + lane;
                }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ob = __shfl_xor_sync(0xffffffff, best, o);
                const int oi = __shfl_xor_sync(0xffffffff, best_i, o);
                if (ob > best || (ob == best && oi < best_i)) {
                    best = ob;
                    best_i = oi;
                }
            }
            chosen[r] = best / sum;
            chosen_i[r] = best_i;
            chosen_sum += chosen[r];
            if ((best_i & 31) == lane) v[best_i / 32] = -2;
        }
        if (lane < TOPK) {
            float c = 0;
            int ci = 0;
#pragma unroll
            for (int r = 0; r < TOPK; ++r)
                if (r == lane) {
                    c = chosen[r];
                    ci = chosen_i[r];
                }
            top_i[t * TOPK + lane] = ci;
            top_w[t * TOPK + lane] = c / chosen_sum;
            s_top[t * TOPK + lane] = ci;
            atomicAdd(&s_count[ci], 1);
        }
    }
    __syncthreads();
    // exclusive scan of counts -> offsets; compact active experts
    const int c = s_count[tid];
    s_scan[tid] = c;
    __syncthreads();
    for (int off = 1; off < E; off <<= 1) {
        const int a = tid >= off ? s_scan[tid - off] : 0;
        __syncthreads();
        s_scan[tid] += a;
        __syncthreads();
    }
    offsets[tid] = s_scan[tid] - c;
    if (tid == E - 1) offsets[E] = s_scan[tid];
    __syncthreads();
    // active list: reuse s_count as a 0/1 flag scan
    const int flag = c > 0;
    s_count[tid] = flag;
    __syncthreads();
    for (int off = 1; off < E; off <<= 1) {
        const int a = tid >= off ? s_count[tid - off] : 0;
        __syncthreads();
        s_count[tid] += a;
        __syncthreads();
    }
    if (flag) active[s_count[tid] - 1] = tid;
    if (tid == E - 1) *num_active = s_count[tid];
    // rows: pairs ordered by (expert, token, slot) deterministically
    const int pairs = T * TOPK;
    if (tid < pairs) {
        const int e = s_top[tid];
        int rank = 0;
        for (int j = 0; j < tid; ++j) rank += s_top[j] == e;
        const int row = (s_scan[e] - (s_scan[e] - (e > 0 ? s_scan[e - 1] : 0))) + rank;
        pair_row[tid] = row;
        row_token[row] = tid / TOPK;
    }
}

// Decodes 32 NVFP4 weights (16 code bytes, two FP8 block scales) to floats; exact in FP16.
SQ_DEVICE void decode32(const uint4 codes, uint16_t sc, float* w) {
    __half2 s0, s1;
    {
        const __half h0 = __float2half(fp8_to_float(sc & 0xff)), h1 = __float2half(fp8_to_float(sc >> 8));
        s0 = __halves2half2(h0, h0);
        s1 = __halves2half2(h1, h1);
    }
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&codes);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const __half2_raw r = __nv_cvt_fp4x2_to_halfraw2(b[i], __NV_E2M1);
        const __half2 h = __hmul2(*reinterpret_cast<const __half2*>(&r), i < 8 ? s0 : s1);
        const float2 f = __half22float2(h);
        w[2 * i] = f.x;
        w[2 * i + 1] = f.y;
    }
}

constexpr int EW = 8;  // warps per expert-GEMV CTA

// x[m] . w for one 32-weight chunk of each routed row m, accumulated into acc[m].
SQ_DEVICE void chunk_dot(const float* w, const __nv_bfloat16* xrow, float& acc) {
    const uint4* xp = reinterpret_cast<const uint4*>(xrow);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint4 xv = __ldg(xp + c);
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 xf = __bfloat1622float2(x2[i]);
            acc = fmaf(w[c * 8 + 2 * i], xf.x, acc);
            acc = fmaf(w[c * 8 + 2 * i + 1], xf.y, acc);
        }
    }
}

// act[row, n] = bf16(silu(gate_e[n] . x) * (up_e[n] . x)) for every row of active expert e,
// x = m[row_token[row]]. grid (640 / 16, active slots). Warp w owns features 2w, 2w + 1 of the block:
// lanes 0-15 stream the gate rows and lanes 16-31 the up rows, five 32-weight chunks per row per lane,
// all loads issued before any arithmetic.
template <int MAXR, int FEAT>
__global__ void __launch_bounds__(EW * 32) gate_up_kernel(const __nv_bfloat16* __restrict__ m,
                                                          const int* __restrict__ row_token,
                                                          const uint64_t* __restrict__ gw, const uint64_t* __restrict__ gs,
                                                          const float* __restrict__ g2, const uint64_t* __restrict__ uw,
                                                          const uint64_t* __restrict__ us, const float* __restrict__ u2,
                                                          const int* __restrict__ offsets, const int* __restrict__ active,
                                                          const int* __restrict__ num_active, __nv_bfloat16* act) {
    pdl_wait();
    constexpr int N = 640, K = D, CHUNKS = K / 32 / 16;
    if ((int)blockIdx.y >= *num_active) return;
    const int e = active[blockIdx.y];
    const int r0 = offsets[e], rows = min(offsets[e + 1] - r0, MAXR);
    const int lane = threadIdx.x & 31, half = lane >> 4, hl = lane & 15;
    const int n0 = (blockIdx.x * EW + (threadIdx.x >> 5)) * FEAT;
    const uint8_t* W = reinterpret_cast<const uint8_t*>(half ? uw[e] : gw[e]) + (size_t)n0 * (K / 2);
    const uint8_t* Sc = reinterpret_cast<const uint8_t*>(half ? us[e] : gs[e]) + (size_t)n0 * (K / 16);
    uint4 codes[FEAT][CHUNKS];
    uint16_t scales[FEAT][CHUNKS];
#pragma unroll
    for (int f = 0; f < FEAT; ++f)
#pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            const int k = (hl + 16 * c) * 32;
            codes[f][c] = __ldg(reinterpret_cast<const uint4*>(W + (size_t)f * (K / 2) + k / 2));
            scales[f][c] = __ldg(reinterpret_cast<const uint16_t*>(Sc + (size_t)f * (K / 16) + k / 16));
        }
    const __nv_bfloat16* xrow[MAXR];
#pragma unroll
    for (int mm = 0; mm < MAXR; ++mm) xrow[mm] = m + (size_t)row_token[r0 + min(mm, rows - 1)] * K;
    float acc[FEAT][MAXR] = {};
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int k = (hl + 16 * c) * 32;
#pragma unroll
        for (int f = 0; f < FEAT; ++f) {
            float w[32];
            decode32(codes[f][c], scales[f][c], w);
#pragma unroll
            for (int mm = 0; mm < MAXR; ++mm)
                if (mm < rows) chunk_dot(w, xrow[mm] + k, acc[f][mm]);
        }
    }
    const float sg = g2[e], su = u2[e];
    // Nothing of the successor can start before this kernel's results: signal at the end.
    pdl_launch_dependents();
#pragma unroll
    for (int f = 0; f < FEAT; ++f)
#pragma unroll
        for (int mm = 0; mm < MAXR; ++mm) {
            if (mm < rows) {
                float v = acc[f][mm];
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
                const float other = __shfl_xor_sync(0xffffffff, v, 16);
                if (lane == 0) act[(size_t)(r0 + mm) * N + n0 + f] = f2bf(silu(v * sg) * (other * su));
            }
        }
}

// Multi-row variant of gate_up_kernel: half-warp h of warp w owns feature 2w + h and streams both its
// gate and up rows, so each activation chunk it loads serves two weight rows. Same per-row arithmetic
// and reduction order as gate_up_kernel (chunk k = (hl + 16 c) * 32; lanes 0..15 sum, gate then up).
template <int MAXR>
__global__ void __launch_bounds__(EW * 32) gate_up_pair_kernel(const __nv_bfloat16* __restrict__ m,
                                                               const int* __restrict__ row_token,
                                                               const uint64_t* __restrict__ gw,
                                                               const uint64_t* __restrict__ gs, const float* __restrict__ g2,
                                                               const uint64_t* __restrict__ uw,
                                                               const uint64_t* __restrict__ us, const float* __restrict__ u2,
                                                               const int* __restrict__ offsets,
                                                               const int* __restrict__ active,
                                                               const int* __restrict__ num_active, __nv_bfloat16* act) {
    constexpr int N = 640, K = D, CHUNKS = K / 32 / 16;
    pdl_wait();
    if ((int)blockIdx.y >= *num_active) return;
    const int e = active[blockIdx.y];
    const int r0 = offsets[e], rows = min(offsets[e + 1] - r0, MAXR);
    const int lane = threadIdx.x & 31, half = lane >> 4, hl = lane & 15;
    const int n = (blockIdx.x * EW + (threadIdx.x >> 5)) * 2 + half;
    const uint8_t* G = reinterpret_cast<const uint8_t*>(gw[e]) + (size_t)n * (K / 2);
    const uint8_t* GS = reinterpret_cast<const uint8_t*>(gs[e]) + (size_t)n * (K / 16);
    const uint8_t* U = reinterpret_cast<const uint8_t*>(uw[e]) + (size_t)n * (K / 2);
    const uint8_t* US = reinterpret_cast<const uint8_t*>(us[e]) + (size_t)n * (K / 16);
    uint4 gc[CHUNKS], uc[CHUNKS];
    uint16_t gsc[CHUNKS], usc[CHUNKS];
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int k = (hl + 16 * c) * 32;
        gc[c] = __ldg(reinterpret_cast<const uint4*>(G + k / 2));
        uc[c] = __ldg(reinterpret_cast<const uint4*>(U + k / 2));
        gsc[c] = __ldg(reinterpret_cast<const uint16_t*>(GS + k / 16));
        usc[c] = __ldg(reinterpret_cast<const uint16_t*>(US + k / 16));
    }
    const __nv_bfloat16* xrow[MAXR];
#pragma unroll
    for (int mm = 0; mm < MAXR; ++mm) xrow[mm] = m + (size_t)row_token[r0 + min(mm, rows - 1)] * K;
    float ag[MAXR] = {}, au[MAXR] = {};
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int k = (hl + 16 * c) * 32;
        float wg[32], wu[32];
        decode32(gc[c], gsc[c], wg);
        decode32(uc[c], usc[c], wu);
#pragma unroll
        for (int mm = 0; mm < MAXR; ++mm) {
            if (mm >= rows) break;
            const uint4* xp = reinterpret_cast<const uint4*>(xrow[mm] + k);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 xv = __ldg(xp + q);
                const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 xf = __bfloat1622float2(x2[i]);
                    ag[mm] = fmaf(wg[q * 8 + 2 * i], xf.x, ag[mm]);
                    ag[mm] = fmaf(wg[q * 8 + 2 * i + 1], xf.y, ag[mm]);
                    au[mm] = fmaf(wu[q * 8 + 2 * i], xf.x, au[mm]);
                    au[mm] = fmaf(wu[q * 8 + 2 * i + 1], xf.y, au[mm]);
                }
            }
        }
    }
    pdl_launch_dependents();
    const float sg = g2[e], su = u2[e];
#pragma unroll
    for (int mm = 0; mm < MAXR; ++mm) {
        if (mm >= rows) break;
        float g = ag[mm], u = au[mm];
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) {
            g += __shfl_xor_sync(0xffffffff, g, o);
            u += __shfl_xor_sync(0xffffffff, u, o);
        }
        if (hl == 0) act[(size_t)(r0 + mm) * N + n] = f2bf(silu(g * sg) * (u * su));
    }
}

// yd[row, n] = down_e[n] . act[row] for every row of active expert e. grid (2560 / 64, active slots):
// each warp owns 8 features with 4 lanes per feature, five 32-weight chunks per lane loaded up front.
template <int MAXR>
__global__ void __launch_bounds__(EW * 32) down_kernel(const __nv_bfloat16* __restrict__ act,
                                                       const uint64_t* __restrict__ dw, const uint64_t* __restrict__ ds,
                                                       const float* __restrict__ d2, const int* __restrict__ offsets,
                                                       const int* __restrict__ active, const int* __restrict__ num_active,
                                                       float* yd) {
    pdl_wait();
    constexpr int K = 640, CHUNKS = K / 32 / 4;  // 5 chunks per lane
    if ((int)blockIdx.y >= *num_active) return;
    const int e = active[blockIdx.y];
    const int r0 = offsets[e], rows = min(offsets[e + 1] - r0, MAXR);
    const int lane = threadIdx.x & 31, q = lane & 3;
    const int n = (blockIdx.x * EW + (threadIdx.x >> 5)) * 8 + (lane >> 2);
    const uint8_t* W = reinterpret_cast<const uint8_t*>(dw[e]) + (size_t)n * (K / 2);
    const uint8_t* Sc = reinterpret_cast<const uint8_t*>(ds[e]) + (size_t)n * (K / 16);
    uint4 codes[CHUNKS];
    uint16_t scales[CHUNKS];
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int k = (q + 4 * c) * 32;
        codes[c] = __ldg(reinterpret_cast<const uint4*>(W + k / 2));
        scales[c] = __ldg(reinterpret_cast<const uint16_t*>(Sc + k / 16));
    }
    float acc[MAXR] = {};
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int k = (q + 4 * c) * 32;
        float w[32];
        decode32(codes[c], scales[c], w);
#pragma unroll
        for (int mm = 0; mm < MAXR; ++mm)
            if (mm < rows) chunk_dot(w, act + (size_t)(r0 + mm) * K + k, acc[mm]);
    }
    // Nothing of the successor can start before this kernel's results: signal at the end.
    pdl_launch_dependents();
    const float s = d2[e];
#pragma unroll
    for (int mm = 0; mm < MAXR; ++mm) {
        if (mm < rows) {
            float v = acc[mm];
            v += __shfl_xor_sync(0xffffffff, v, 2);
            v += __shfl_xor_sync(0xffffffff, v, 1);
            if (q == 0) yd[(size_t)(r0 + mm) * D + n] = v * s;
        }
    }
}


// Small batches: the shared expert's down projection with its SwiGLU input computed on the fly from
// the fused MoE projection rows (gate at column 512, up at 1152, shared-expert gate logit at 1792),
// and the routed combine in the epilogue:
//   y[t, n] = sum_k w[t, k] * yd[pair_row[t, k], n] + sigmoid(sg[t]) * (down_s[n] . swiglu(t)).
// One warp per output feature n; K = 640 = 40 chunks of 16, two per lane, loaded before the PDL wait.
template <WFmt F, int MAXT>
__global__ void __launch_bounds__(256) shared_combine_kernel(const float* __restrict__ proj, int ld, int T,
                                                              const uint8_t* __restrict__ W,
                                                              const float* __restrict__ scale,
                                                              const float* __restrict__ yd,
                                                              const int* __restrict__ pair_row,
                                                              const float* __restrict__ top_w, float* y) {
    pdl_launch_dependents();
    constexpr int K = 640, GATE = 512, UP = 1152, SG = 1792;
    const int lane = threadIdx.x & 31, n = blockIdx.x * 8 + (threadIdx.x >> 5);
    const uint8_t* row = W + (size_t)n * K * elem_bytes<F>();
    Raw16<F> pre[2];
    float gs[2] = {1.0f, 1.0f};
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        const int k = (lane + 32 * c) * 16;
        if (k < K) {
            pre[c] = load_raw16<F>(row, k);
            if constexpr (F == WFmt::INT8G) gs[c] = __ldg(scale + (size_t)n * (K / GROUP) + k / GROUP);
        }
    }
    pdl_wait();
    // The SwiGLU input bf16(silu(gate) * up) of every token, computed once per block; column k is
    // stored at k + k / 16 so the lanes' 16-column chunks fall in different banks.
    constexpr int KP = K + K / 16;
    __shared__ float xs[MAXT][KP];
    for (int i = threadIdx.x; i < T * K; i += blockDim.x) {
        const float* pr = proj + (size_t)(i / K) * ld;
        const int kk = i % K;
        xs[i / K][kk + kk / 16] = bf2f(f2bf(silu(pr[GATE + kk]) * pr[UP + kk]));
    }
    __syncthreads();
    float acc[MAXT] = {};
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        const int k = (lane + 32 * c) * 16;
        if (k >= K) break;
        float w[16];
        decode_raw16<F>(pre[c], w);
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= T) break;
#pragma unroll
            for (int i = 0; i < 16; ++i) acc[t] = fmaf(w[i] * gs[c], xs[t][k + k / 16 + i], acc[t]);
        }
    }
    // Routed sum: lanes 10 j + k load slot k of token t0 + j (three tokens per pass); the operands are
    // shuffled to every lane and added in slot order, the same arithmetic and order as combine_kernel,
    // so every batch size gives identical results.
    constexpr int TPP = 32 / TOPK;
#pragma unroll
    for (int t0 = 0; t0 < MAXT; t0 += TPP) {
        if (t0 >= T) break;
        const int tl = t0 + lane / TOPK, pidx = tl * TOPK + lane % TOPK;
        float tw = 0, yv = 0;
        if (lane < TPP * TOPK && tl < T) {
            tw = top_w[pidx];
            yv = yd[(size_t)pair_row[pidx] * D + n];
        }
#pragma unroll
        for (int j = 0; j < TPP; ++j) {
            const int t = t0 + j;
            if (t >= MAXT) break;
            float routed = 0;
#pragma unroll
            for (int k = 0; k < TOPK; ++k) {
                const float a = __shfl_sync(0xffffffff, tw, j * TOPK + k), b = __shfl_sync(0xffffffff, yv, j * TOPK + k);
                routed += a * b;
            }
            const float v = warp_sum(acc[t]);
            if (lane == 0 && t < T) y[(size_t)t * D + n] = routed + sigmoidf(proj[(size_t)t * ld + SG]) * v;
        }
    }
}


// ---- single-token decode: routing fused into the expert gate/up kernel ------------------------

// Softmax over the 512 router logits and top-10 (ties: lower expert), renormalized; computed by one
// warp, results in shared memory. Identical in every block that calls it.
SQ_DEVICE void route_one(const float* logits, int lane, int* s_top, float* s_w) {
    float v[E / 32];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < E / 32; ++j) {
        v[j] = logits[j * 32 + lane];
        mx = fmaxf(mx, v[j]);
    }
    mx = warp_max(mx);
    float sum = 0;
#pragma unroll
    for (int j = 0; j < E / 32; ++j) {
        v[j] = __expf(v[j] - mx);
        if (!(v[j] >= 0.0f)) v[j] = 0.0f;  // NaN logits: never pick an out-of-range expert
        sum += v[j];
    }
    sum = warp_sum(sum);
    float chosen_sum = 0;
    for (int r = 0; r < TOPK; ++r) {
        float best = -1;
        int best_i = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < E / 32; ++j)
            if (v[j] > best) {
                best = v[j];
                best_i = j * 32 + lane;
            }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffff, best, o);
            const int oi = __shfl_xor_sync(0xffffffff, best_i, o);
            if (ob > best || (ob == best && oi < best_i)) {
                best = ob;
                best_i = oi;
            }
        }
        if (lane == 0) {
            s_top[r] = best_i;
            s_w[r] = best / sum;
        }
        chosen_sum += best / sum;
        if ((best_i & 31) == lane) v[best_i / 32] = -2;
    }
    if (lane == 0)
        for (int r = 0; r < TOPK; ++r) s_w[r] /= chosen_sum;
}

// grid (640 / 16, 10 routing slots). Each block routes the token itself, then computes
// act[slot, n] = silu(gate_e . m) * (up_e . m) for expert e = top[slot]; block (0, 0) publishes
// top_i/top_w and pair_row[k] = k for the down and combine kernels.
__global__ void __launch_bounds__(EW * 32) gate_up_t1_kernel(const __nv_bfloat16* __restrict__ m,
                                                             const float* __restrict__ logits,
                                                             const uint64_t* __restrict__ gw, const uint64_t* __restrict__ gs,
                                                             const float* __restrict__ g2, const uint64_t* __restrict__ uw,
                                                             const uint64_t* __restrict__ us, const float* __restrict__ u2,
                                                             int* top_i, float* top_w, int* pair_row,
                                                             __nv_bfloat16* act) {
    constexpr int N = 640, K = D, CHUNKS = K / 32 / 16, FEAT = 2;
    __shared__ int s_top[TOPK];
    __shared__ float s_w[TOPK];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, half = lane >> 4, hl = lane & 15;
    pdl_wait();
    if (warp == 0) route_one(logits, lane, s_top, s_w);
    __syncthreads();
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < TOPK) {
        top_i[threadIdx.x] = s_top[threadIdx.x];
        top_w[threadIdx.x] = s_w[threadIdx.x];
        pair_row[threadIdx.x] = threadIdx.x;
    }
    const int slot = blockIdx.y, e = s_top[slot];
    const int n0 = (blockIdx.x * EW + warp) * FEAT;
    const uint8_t* W = reinterpret_cast<const uint8_t*>(half ? uw[e] : gw[e]) + (size_t)n0 * (K / 2);
    const uint8_t* Sc = reinterpret_cast<const uint8_t*>(half ? us[e] : gs[e]) + (size_t)n0 * (K / 16);
    uint4 codes[FEAT][CHUNKS];
    uint16_t scales[FEAT][CHUNKS];
#pragma unroll
    for (int f = 0; f < FEAT; ++f)
#pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            const int k = (hl + 16 * c) * 32;
            codes[f][c] = __ldg(reinterpret_cast<const uint4*>(W + (size_t)f * (K / 2) + k / 2));
            scales[f][c] = __ldg(reinterpret_cast<const uint16_t*>(Sc + (size_t)f * (K / 16) + k / 16));
        }
    float acc[FEAT] = {};
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int k = (hl + 16 * c) * 32;
#pragma unroll
        for (int f = 0; f < FEAT; ++f) {
            float w[32];
            decode32(codes[f][c], scales[f][c], w);
            chunk_dot(w, m + k, acc[f]);
        }
    }
    pdl_launch_dependents();
    const float sg = g2[e], su = u2[e];
#pragma unroll
    for (int f = 0; f < FEAT; ++f) {
        float v = acc[f];
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        const float other = __shfl_xor_sync(0xffffffff, v, 16);
        if (lane == 0) act[(size_t)slot * N + n0 + f] = f2bf(silu(v * sg) * (other * su));
    }
}

// grid (2560 / 64, 10 slots): yd[slot, n] = down_e[n] . act[slot] with e = top_i[slot].
__global__ void __launch_bounds__(EW * 32) down_t1_kernel(const __nv_bfloat16* __restrict__ act,
                                                          const uint64_t* __restrict__ dw, const uint64_t* __restrict__ ds,
                                                          const float* __restrict__ d2, const int* __restrict__ top_i,
                                                          float* yd) {
    constexpr int K = 640, CHUNKS = K / 32 / 4;
    pdl_wait();
    const int slot = blockIdx.y, e = top_i[slot];
    const int lane = threadIdx.x & 31, q = lane & 3;
    const int n = (blockIdx.x * EW + (threadIdx.x >> 5)) * 8 + (lane >> 2);
    const uint8_t* W = reinterpret_cast<const uint8_t*>(dw[e]) + (size_t)n * (K / 2);
    const uint8_t* Sc = reinterpret_cast<const uint8_t*>(ds[e]) + (size_t)n * (K / 16);
    uint4 codes[CHUNKS];
    uint16_t scales[CHUNKS];
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int k = (q + 4 * c) * 32;
        codes[c] = __ldg(reinterpret_cast<const uint4*>(W + k / 2));
        scales[c] = __ldg(reinterpret_cast<const uint16_t*>(Sc + k / 16));
    }
    float acc = 0;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int k = (q + 4 * c) * 32;
        float w[32];
        decode32(codes[c], scales[c], w);
        chunk_dot(w, act + (size_t)slot * K + k, acc);
    }
    pdl_launch_dependents();
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);
    if (q == 0) yd[(size_t)slot * D + n] = acc * d2[e];
}


// Dense NVFP4 GEMV for one token over K = 2560: y[n] = scale2 * W[n] . x, W in the checkpoint NVFP4
// layout (codes [N, 1280], block scales [N, 160]). Used for the drafter's quantized LM head. A warp
// owns two rows; lanes 0-15 and 16-31 each stream one row, five 32-weight chunks per lane.
__global__ void __launch_bounds__(EW * 32) nvfp4_dense_kernel(const __nv_bfloat16* __restrict__ x,
                                                              const uint8_t* __restrict__ codes,
                                                              const uint8_t* __restrict__ scales, const float* scale2,
                                                              float* y, int N) {
    constexpr int K = D, CHUNKS = K / 32 / 16;
    pdl_launch_dependents();
    const int lane = threadIdx.x & 31, half = lane >> 4, hl = lane & 15;
    const int n = (blockIdx.x * EW + (threadIdx.x >> 5)) * 2 + half;
    const int row = min(n, N - 1);
    const uint8_t* W = codes + (size_t)row * (K / 2);
    const uint8_t* Sc = scales + (size_t)row * (K / 16);
    uint4 c[CHUNKS];
    uint16_t sc[CHUNKS];
#pragma unroll
    for (int i = 0; i < CHUNKS; ++i) {
        const int k = (hl + 16 * i) * 32;
        c[i] = __ldg(reinterpret_cast<const uint4*>(W + k / 2));
        sc[i] = __ldg(reinterpret_cast<const uint16_t*>(Sc + k / 16));
    }
    pdl_wait();
    float acc = 0;
#pragma unroll
    for (int i = 0; i < CHUNKS; ++i) {
        const int k = (hl + 16 * i) * 32;
        float w[32];
        decode32(c[i], sc[i], w);
        chunk_dot(w, x + k, acc);
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (hl == 0 && n < N) y[n] = acc * *scale2;
}

}  // namespace

extern "C" int squash_moe_tile_rows() { return BM; }

extern "C" cudaError_t squash_moe_route(const float* logits, int ld, int T, int* top_i, float* top_w, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(route_kernel, (T + 7) / 8, 256, 0, st, logits, ld, T, top_i, top_w));
    SQ_RETURN_LAUNCH();
}

// Groups the T*10 routed pairs by expert. Scratch: counts/cursor [512], offsets [513],
// tile_expert/tile_row [max tiles], num_tiles [1], pair_row [T*10].
extern "C" cudaError_t squash_moe_plan(const int* top_i, int T, int* counts, int* cursor, int* offsets, int* tile_expert,
                                       int* tile_row, int* num_tiles, int* pair_row, int tile_rows, cudaStream_t st) {
    const int pairs = T * TOPK;
    cudaError_t err = cudaMemsetAsync(counts, 0, E * sizeof(int), st);
    if (err != cudaSuccess) return err;
    if (pairs > 0) SQ_CHECK_LAUNCH(sq_launch(count_kernel, (pairs + 255) / 256, 256, 0, st, top_i, pairs, counts));
    SQ_CHECK_LAUNCH(sq_launch(plan_kernel, 1, E, 0, st, counts, offsets, cursor, tile_expert, tile_row, num_tiles, tile_rows));
    if (pairs > 0) SQ_CHECK_LAUNCH(sq_launch(scatter_kernel, (pairs + 255) / 256, 256, 0, st, top_i, pairs, cursor, pair_row));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_moe_gather(const void* m, const int* pair_row, int T, void* xs, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(gather_kernel, T * TOPK, 128, 0, st, static_cast<const __nv_bfloat16*>(m), pair_row, T * TOPK,
                                            static_cast<__nv_bfloat16*>(xs)));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_moe_nvfp4_gemm(const void* xs, const uint64_t* w_ptr, const uint64_t* s_ptr,
                                             const float* ws2, const int* offsets, const int* tile_expert,
                                             const int* tile_row, const int* num_tiles, int max_tiles, float* y, int N,
                                             int K, cudaStream_t st) {
    if (K % BK != 0 || max_tiles == 0) return max_tiles == 0 ? cudaSuccess : cudaErrorInvalidValue;
    SQ_CHECK_LAUNCH(sq_launch(nvfp4_gemm_kernel, dim3((N + BN - 1) / BN, max_tiles), THREADS, 0, st, 
        static_cast<const __nv_bfloat16*>(xs), w_ptr, s_ptr, ws2, offsets, tile_expert, tile_row, num_tiles, y, N, K));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_swiglu(const float* gate, const float* up, int ld, int width, void* act, size_t count,
                                     cudaStream_t st) {
    if (count == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(swiglu_kernel, (unsigned)((count + 255) / 256), 256, 0, st, gate, up, ld, width, static_cast<__nv_bfloat16*>(act),
                                                                    count));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_moe_combine(const float* yd, const int* pair_row, const float* top_w, const float* sg,
                                          int ld_sg, const float* sh, float* y, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(combine_kernel<float>, dim3(T, D / 256), 256, 0, st, yd, pair_row, top_w, sg, ld_sg, sh, y));
    SQ_RETURN_LAUNCH();
}

// As squash_moe_combine with BF16 expert outputs.
extern "C" cudaError_t squash_moe_combine_bf16(const void* yd, const int* pair_row, const float* top_w, const float* sg,
                                               int ld_sg, const float* sh, float* y, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(combine_kernel<__nv_bfloat16>, dim3(T, D / 256), 256, 0, st, static_cast<const __nv_bfloat16*>(yd), pair_row,
                              top_w, sg, ld_sg, sh, y));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_moe_active(const int* counts, int* active, int* num_active, cudaStream_t st) {
    SQ_CHECK_LAUNCH(sq_launch(active_kernel, 1, E, 0, st, counts, active, num_active));
    SQ_RETURN_LAUNCH();
}

extern "C" int squash_moe_small_tokens() { return SMALL_T; }

extern "C" cudaError_t squash_moe_route_small(const float* logits, int ld, int T, int* top_i, float* top_w, int* offsets,
                                              int* pair_row, int* row_token, int* active, int* num_active,
                                              cudaStream_t st) {
    if (T == 0 || T > SMALL_T) return T == 0 ? cudaSuccess : cudaErrorInvalidValue;
    SQ_CHECK_LAUNCH(sq_launch(route_small_kernel, 1, 512, 0, st, logits, ld, T, top_i, top_w, offsets, pair_row, row_token, active, num_active));
    SQ_RETURN_LAUNCH();
}

// gate/up tables: [codes, scales] u64 x 512 each and weight_scale_2 f32 x 512. T bounds the rows per expert.
extern "C" cudaError_t squash_moe_gate_up_small(const void* m, const int* row_token, const uint64_t* gw,
                                                const uint64_t* gs, const float* g2, const uint64_t* uw,
                                                const uint64_t* us, const float* u2, const int* offsets,
                                                const int* active, const int* num_active, int max_active, int T,
                                                void* act, cudaStream_t st) {
    if (max_active == 0) return cudaSuccess;
    // Two features per warp, each half-warp streaming one feature's gate and up rows.
    const dim3 grid(640 / (EW * 2), max_active);
    const auto* x = static_cast<const __nv_bfloat16*>(m);
    auto* a = static_cast<__nv_bfloat16*>(act);
#define SQ_GU(R) SQ_CHECK_LAUNCH(sq_launch(gate_up_pair_kernel<R>, grid, EW * 32, 0, st, x, row_token, gw, gs, g2, uw, us, u2, offsets, active, num_active, a))
    if (T <= 1) SQ_GU(1);
    else if (T <= 2) SQ_GU(2);
    else if (T <= 4) SQ_GU(4);
    else if (T <= 8) SQ_GU(8);
    else SQ_GU(16);
#undef SQ_GU
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_moe_down_small(const void* act, const uint64_t* dw, const uint64_t* ds, const float* d2,
                                             const int* offsets, const int* active, const int* num_active,
                                             int max_active, int T, float* yd, cudaStream_t st) {
    if (max_active == 0) return cudaSuccess;
    const dim3 grid(D / (EW * 8), max_active);
    const auto* x = static_cast<const __nv_bfloat16*>(act);
#define SQ_DN(R) SQ_CHECK_LAUNCH(sq_launch(down_kernel<R>, grid, EW * 32, 0, st, x, dw, ds, d2, offsets, active, num_active, yd))
    if (T <= 1) SQ_DN(1);
    else if (T <= 2) SQ_DN(2);
    else if (T <= 4) SQ_DN(4);
    else if (T <= 8) SQ_DN(8);
    else SQ_DN(16);
#undef SQ_DN
    SQ_RETURN_LAUNCH();
}

// fmt 0 BF16 or 3 INT8 (scale per 64) shared-expert down weights [2560, 640].
extern "C" cudaError_t squash_moe_shared_combine(int fmt, const float* proj, int ld, int T, const void* w,
                                                 const float* scale, const float* yd, const int* pair_row,
                                                 const float* top_w, float* y, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    if (T > SMALL_T) return cudaErrorInvalidValue;
    const auto* w8 = static_cast<const uint8_t*>(w);
#define SQ_SC(FMT, R) SQ_CHECK_LAUNCH(sq_launch(shared_combine_kernel<FMT, R>, D / 8, 256, 0, st, proj, ld, T, w8, scale, yd, pair_row, top_w, y))
#define SQ_SC_T(FMT) \
    if (T <= 1) SQ_SC(FMT, 1); \
    else if (T <= 2) SQ_SC(FMT, 2); \
    else if (T <= 4) SQ_SC(FMT, 4); \
    else if (T <= 8) SQ_SC(FMT, 8); \
    else SQ_SC(FMT, 16);
    if (fmt == 0) {
        SQ_SC_T(WFmt::BF16)
    } else if (fmt == 3) {
        SQ_SC_T(WFmt::INT8G)
    } else {
        return cudaErrorInvalidValue;
    }
#undef SQ_SC_T
#undef SQ_SC
    SQ_RETURN_LAUNCH();
}

// Single-token MoE experts: routing (from the router logits row) fused into gate/up, then down.
// Writes top_i, top_w, pair_row (= slot) and yd [10, 2560] for squash_moe_shared_combine.
extern "C" cudaError_t squash_moe_experts_t1(const void* m, const float* logits, const uint64_t* gw, const uint64_t* gs,
                                            const float* g2, const uint64_t* uw, const uint64_t* us, const float* u2,
                                            const uint64_t* dw, const uint64_t* ds, const float* d2, int* top_i,
                                            float* top_w, int* pair_row, void* act, float* yd, cudaStream_t st) {
    const auto* mb = static_cast<const __nv_bfloat16*>(m);
    auto* ab = static_cast<__nv_bfloat16*>(act);
    SQ_CHECK_LAUNCH(sq_launch(gate_up_t1_kernel, dim3(640 / (EW * 2), TOPK), EW * 32, 0, st, mb, logits, gw, gs, g2, uw, us,
                              u2, top_i, top_w, pair_row, ab));
    SQ_CHECK_LAUNCH(sq_launch(down_t1_kernel, dim3(D / (EW * 8), TOPK), EW * 32, 0, st, static_cast<const __nv_bfloat16*>(ab),
                              dw, ds, d2, static_cast<const int*>(top_i), yd));
    SQ_RETURN_LAUNCH();
}

namespace {

// ---- draft head shortlist: a 1-bit sketch of the head ranks the vocabulary, the exact NVFP4 rows
// rescore each slice's best candidates ------------------------------------------------------------

constexpr int SK_GROUPS = D / 64, SK_SLICE = 8192, SK_PICK = 8, SK_ROWS = 1;  // rows per sketch thread

// masks [group][row]: bit i = (w[row, 64 group + i] >= 0); scales [group][row] = mean |w| of the group.
__global__ void build_sketch_kernel(const __nv_bfloat16* __restrict__ w, unsigned long long* masks, __half* scales, int N) {
    const size_t item = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (item >= (size_t)N * SK_GROUPS) return;
    const int row = (int)(item % N), g = (int)(item / N);
    const __nv_bfloat16* p = w + (size_t)row * D + g * 64;
    unsigned long long m = 0;
    float sum = 0.0f;
    for (int i = 0; i < 64; ++i) {
        const float v = bf2f(p[i]);
        sum += fabsf(v);
        if (v >= 0.0f) m |= 1ull << i;
    }
    masks[item] = m;
    scales[item] = __float2half(sum / 64.0f);
}

// y[row] = sum_g scale_g (2 sum_{bits} x - sum_g x): one row per thread. The block first tabulates,
// for each group and 4-bit position, the sum of x over each of the 16 bit patterns (40 KB), so a
// group costs 16 lookups; every lane reads the same 16-entry table at a time (no bank conflicts).
__global__ void __launch_bounds__(256) sketch_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                                          const unsigned long long* __restrict__ masks,
                                                          const __half* __restrict__ scales, float* y, int N) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float lut[SK_GROUPS * 16][16];  // [group * 16 + nibble position][pattern]
    __shared__ float gsum[SK_GROUPS];
    for (int e = threadIdx.x; e < SK_GROUPS * 16 * 16; e += blockDim.x) {
        const int pos = e / 16, pat = e % 16;
        const __nv_bfloat16* xp = x + pos * 4;
        float v = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            if (pat >> i & 1) v += bf2f(xp[i]);
        lut[pos][pat] = v;
    }
    __syncthreads();
    if (threadIdx.x < SK_GROUPS) {
        float t = 0.0f;
        for (int p = 0; p < 16; ++p) t += lut[threadIdx.x * 16 + p][15];
        gsum[threadIdx.x] = t;
    }
    __syncthreads();
    for (int rr = 0; rr < SK_ROWS; ++rr) {
        const int row = (blockIdx.x * SK_ROWS + rr) * blockDim.x + threadIdx.x;
        if (row >= N) return;
        float acc = 0.0f;
        constexpr int GB = 8;
        for (int g0 = 0; g0 < SK_GROUPS; g0 += GB) {
            unsigned long long m[GB];
            float sc[GB];
#pragma unroll
            for (int j = 0; j < GB; ++j) {
                m[j] = __ldg(masks + (size_t)(g0 + j) * N + row);
                sc[j] = __half2float(__ldg(scales + (size_t)(g0 + j) * N + row));
            }
#pragma unroll
            for (int j = 0; j < GB; ++j) {
                const int g = g0 + j;
                float sel = 0.0f;
#pragma unroll
                for (int p = 0; p < 16; ++p) sel += lut[g * 16 + p][(m[j] >> (4 * p)) & 15];
                acc = fmaf(sc[j], 2.0f * sel - gsum[g], acc);
            }
        }
        y[row] = acc;
    }
}

// One block per slice of SK_SLICE rows: the SK_PICK best sketch logits, each rescored exactly with its
// NVFP4 row (one warp per candidate); writes the slice's best (exact logit, row).
__global__ void __launch_bounds__(256) draft_pick_kernel(const float* __restrict__ ys, int N,
                                                         const __nv_bfloat16* __restrict__ x,
                                                         const uint8_t* __restrict__ codes,
                                                         const uint8_t* __restrict__ scales, const float* scale2,
                                                         float* best_v, int* best_i) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float wv[8];
    __shared__ int wi[8], picks[SK_PICK];
    __shared__ float exact[SK_PICK];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int base = blockIdx.x * SK_SLICE;
    constexpr int PER = SK_SLICE / 256;
    float v[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int r = base + tid + 256 * j;
        v[j] = r < N ? __ldcg(ys + r) : -INFINITY;
    }
    for (int p = 0; p < SK_PICK; ++p) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < PER; ++j)
            if (v[j] > bv) {
                bv = v[j];
                bi = base + tid + 256 * j;
            }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, bv, o);
            const int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            wv[warp] = bv;
            wi[warp] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            float b = wv[0];
            int ib = wi[0];
            for (int w = 1; w < 8; ++w)
                if (wv[w] > b || (wv[w] == b && wi[w] < ib)) {
                    b = wv[w];
                    ib = wi[w];
                }
            picks[p] = b == -INFINITY ? -1 : ib;
        }
        __syncthreads();
        const int pick = picks[p];
#pragma unroll
        for (int j = 0; j < PER; ++j)
            if (base + tid + 256 * j == pick) v[j] = -INFINITY;
    }
    // Exact NVFP4 logit of candidate `warp`: 80 chunks of 32 weights over the lanes.
    {
        const int r = picks[warp];
        float acc = 0.0f;
        if (r >= 0) {
            const uint8_t* W = codes + (size_t)r * (D / 2);
            const uint8_t* Sc = scales + (size_t)r * (D / 16);
            for (int c = lane; c < D / 32; c += 32) {
                const uint4 cd = __ldg(reinterpret_cast<const uint4*>(W + c * 16));
                const uint16_t sc = __ldg(reinterpret_cast<const uint16_t*>(Sc + c * 2));
                float w[32];
                decode32(cd, sc, w);
                chunk_dot(w, x + c * 32, acc);
            }
            acc = warp_sum(acc) * *scale2;
        }
        if (lane == 0) exact[warp] = r >= 0 ? acc : -INFINITY;
    }
    __syncthreads();
    if (tid == 0) {
        float b = -INFINITY;
        int ib = 0x7fffffff;
        for (int p = 0; p < SK_PICK; ++p)
            if (picks[p] >= 0 && (exact[p] > b || (exact[p] == b && picks[p] < ib))) {
                b = exact[p];
                ib = picks[p];
            }
        best_v[blockIdx.x] = b;
        best_i[blockIdx.x] = ib;
    }
}

__global__ void draft_final_kernel(const float* __restrict__ best_v, const int* __restrict__ best_i, int slices, int* out) {
    pdl_launch_dependents();
    pdl_wait();
    const int lane = threadIdx.x;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int s = lane; s < slices; s += 32)
        if (best_v[s] > bv || (best_v[s] == bv && best_i[s] < bi)) {
            bv = best_v[s];
            bi = best_i[s];
        }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, bv, o);
        const int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) *out = bi;
}

}  // namespace

extern "C" size_t squash_draft_sketch_bytes(int N) { return (size_t)N * SK_GROUPS * (8 + 2); }
extern "C" int squash_draft_slices(int N) { return (N + SK_SLICE - 1) / SK_SLICE; }

// sketch: squash_draft_sketch_bytes(N) bytes (masks, then scales), from the BF16 head w [N, 2560].
extern "C" cudaError_t squash_build_draft_sketch(const void* w, void* sketch, int N, cudaStream_t st) {
    auto* masks = static_cast<unsigned long long*>(sketch);
    auto* sc = reinterpret_cast<__half*>(masks + (size_t)N * SK_GROUPS);
    const size_t items = (size_t)N * SK_GROUPS;
    build_sketch_kernel<<<(unsigned)((items + 255) / 256), 256, 0, st>>>(static_cast<const __nv_bfloat16*>(w), masks, sc, N);
    return cudaGetLastError();
}

// Draft token for one hidden row x: sketch ranking (logits into ys [N]), per-slice NVFP4 rescoring of
// the best candidates (pick: 2 * slices words), argmax into *out.
extern "C" cudaError_t squash_draft_head_argmax(const void* x, const void* sketch, const void* codes, const void* scales,
                                                const float* scale2, float* ys, void* pick, int N, int* out,
                                                cudaStream_t st) {
    const auto* xb = static_cast<const __nv_bfloat16*>(x);
    const auto* masks = static_cast<const unsigned long long*>(sketch);
    const auto* sc = reinterpret_cast<const __half*>(masks + (size_t)N * SK_GROUPS);
    const int slices = (N + SK_SLICE - 1) / SK_SLICE;
    float* bv = static_cast<float*>(pick);
    int* bi = reinterpret_cast<int*>(bv + slices);
    SQ_CHECK_LAUNCH(sq_launch(sketch_gemv_kernel, (N + 256 * SK_ROWS - 1) / (256 * SK_ROWS), 256, 0, st, xb, masks, sc, ys, N));
    // Launched without PDL: reading the sketch logits right behind their producer under PDL gave stale
    // values (the buffer is rewritten for every draft).
    draft_pick_kernel<<<slices, 256, 0, st>>>(static_cast<const float*>(ys), N, xb, static_cast<const uint8_t*>(codes),
                                               static_cast<const uint8_t*>(scales), scale2, bv, bi);
    SQ_CHECK_LAUNCH(sq_launch(draft_final_kernel, 1, 32, 0, st, static_cast<const float*>(bv), static_cast<const int*>(bi), slices, out));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_nvfp4_dense_gemv(const void* x, const void* codes, const void* scales, const float* scale2,
                                               float* y, int N, cudaStream_t st) {
    SQ_CHECK_LAUNCH(sq_launch(nvfp4_dense_kernel, (N + 2 * EW - 1) / (2 * EW), EW * 32, 0, st,
                              static_cast<const __nv_bfloat16*>(x), static_cast<const uint8_t*>(codes),
                              static_cast<const uint8_t*>(scales), scale2, y, N));
    SQ_RETURN_LAUNCH();
}
