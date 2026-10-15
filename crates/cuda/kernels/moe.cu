// Sparse MoE (SPEC §7): routing, expert grouping, NVFP4 grouped GEMM (W4A16) and the combine.
//
// A step routes T tokens to 10 experts each: P = 10 T (token, slot) pairs, pair = t * 10 + slot.
// Pairs are grouped by expert into "rows" (row order within an expert is arbitrary); every expert
// GEMM writes one output row per pair, and the combine reads them back in fixed slot order, so the
// result does not depend on that arbitrary order.

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "mma.cuh"

namespace {

constexpr int E = 512, TOPK = 10, D = 2560;

// One warp per token: softmax over 512 router logits, top 10 (ties: lower expert), renormalized.
__global__ void route_kernel(const float* __restrict__ logits, int T, int* top_i, float* top_w) {
    const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (t >= T) return;
    float v[E / 32];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < E / 32; ++j) {
        v[j] = logits[(size_t)t * E + j * 32 + lane];
        mx = fmaxf(mx, v[j]);
    }
    mx = warp_max(mx);
    float sum = 0;
#pragma unroll
    for (int j = 0; j < E / 32; ++j) {
        v[j] = __expf(v[j] - mx);
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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) atomicAdd(&counts[top_i[i]], 1);
}

// One block of 512 threads: expert row offsets, and the list of BM-row tiles (expert, first row).
__global__ void plan_kernel(const int* __restrict__ counts, int* offsets, int* cursor, int* tile_expert, int* tile_row,
                            int* num_tiles, int bm) {
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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) pair_row[i] = atomicAdd(&cursor[top_i[i]], 1);
}

// xs[pair_row[pair]] = m[pair / 10]
__global__ void gather_kernel(const __nv_bfloat16* __restrict__ m, const int* __restrict__ pair_row, int pairs,
                              __nv_bfloat16* xs) {
    const int pair = blockIdx.x;
    const uint4* src = reinterpret_cast<const uint4*>(m + (size_t)(pair / TOPK) * D);
    uint4* dst = reinterpret_cast<uint4*>(xs + (size_t)pair_row[pair] * D);
    for (int i = threadIdx.x; i < D / 8; i += blockDim.x) dst[i] = src[i];
}

// ---- NVFP4 (W4A16) grouped GEMM ------------------------------------------------------------
// Y[row, N] = ws2[e] * Xs[row, K] . dequant(W_e)[N, K]^T for every row of expert e's tiles.
// W_e: codes U8 [N, K/2] (low nibble first), scales F8E4M3 [N, K/16].

constexpr int BM = 64, BN = 128, BK = 32, STAGES = 2, THREADS = 256, LDS = BK + 8;

__constant__ float kE2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

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

    auto load = [&](int stage, int kt) {
        const int k = kt * BK;
        {
            const int r = tid >> 2, c = (tid & 3) * 8;
            const bool ok = r0 + r < r_end;
            cp_async16(&sA[stage][r * LDS + c], X + (size_t)(ok ? r0 + r : r0) * K + k + c, ok);
        }
        if (tid < BN) {
            const int n = n0 + tid;
            __nv_bfloat16* dst = &sB[stage][tid * LDS];
            if (n < N) {
                const uint4 codes = *reinterpret_cast<const uint4*>(W + (size_t)n * (K / 2) + k / 2);
                const float s0 = fp8_to_float(Sc[(size_t)n * (K / 16) + k / 16]);
                const float s1 = fp8_to_float(Sc[(size_t)n * (K / 16) + k / 16 + 1]);
                const uint8_t* b = reinterpret_cast<const uint8_t*>(&codes);
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const float s = i < 8 ? s0 : s1;
                    dst[2 * i] = f2bf(kE2M1[b[i] & 15] * s);
                    dst[2 * i + 1] = f2bf(kE2M1[b[i] >> 4] * s);
                }
            } else {
#pragma unroll
                for (int i = 0; i < BK; ++i) dst[i] = f2bf(0.0f);
            }
        }
    };

    float acc[2][4][4] = {};
    load(0, 0);
    cp_async_commit();
    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_async_wait<0>();
        __syncthreads();
        if (kt + 1 < k_tiles) load((kt + 1) % STAGES, kt + 1);
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
        __syncthreads();
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

// act[r, i] = bf16(silu(gate[r, i]) * up[r, i])
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, __nv_bfloat16* act,
                              size_t count) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) act[i] = f2bf(silu(gate[i]) * up[i]);
}

// y[t, d] = sum_k w[t, k] * Yd[pair_row[t*10+k], d] + sigmoid(sg[t]) * sh[t, d]
__global__ void combine_kernel(const float* __restrict__ yd, const int* __restrict__ pair_row,
                               const float* __restrict__ top_w, const float* __restrict__ sg,
                               const float* __restrict__ sh, float* y) {
    const int t = blockIdx.x;
    const float g = sigmoidf(sg[t]);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) acc += top_w[t * TOPK + k] * yd[(size_t)pair_row[t * TOPK + k] * D + d];
        y[(size_t)t * D + d] = acc + g * sh[(size_t)t * D + d];
    }
}


// Compacts the experts with at least one routed row into active[0 .. *num_active). One block of 512.
__global__ void active_kernel(const int* __restrict__ counts, int* active, int* num_active) {
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

constexpr int GV_WARPS = 8, GV_ROWS_PER_WARP = 2, GV_ROWS = GV_WARPS * GV_ROWS_PER_WARP, GV_MAX_M = 16;

// Decode-sized expert GEMV, W4A16. grid (row blocks of 16 output features, active-expert slots);
// each CTA applies one expert's rows n to all (<= 16) gathered inputs routed to that expert.
__global__ void __launch_bounds__(GV_WARPS * 32) nvfp4_gemv_kernel(const __nv_bfloat16* __restrict__ X,
                                                                   const uint64_t* __restrict__ w_ptr,
                                                                   const uint64_t* __restrict__ s_ptr,
                                                                   const float* __restrict__ ws2,
                                                                   const int* __restrict__ offsets,
                                                                   const int* __restrict__ active,
                                                                   const int* __restrict__ num_active, float* Y, int N,
                                                                   int K) {
    if ((int)blockIdx.y >= *num_active) return;
    const int e = active[blockIdx.y];
    const int r0 = offsets[e], rows = min(offsets[e + 1] - r0, GV_MAX_M);
    const uint8_t* W = reinterpret_cast<const uint8_t*>(w_ptr[e]);
    const uint8_t* Sc = reinterpret_cast<const uint8_t*>(s_ptr[e]);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float alpha = ws2[e];
#pragma unroll
    for (int rr = 0; rr < GV_ROWS_PER_WARP; ++rr) {
        const int n = blockIdx.x * GV_ROWS + warp * GV_ROWS_PER_WARP + rr;
        if (n >= N) break;
        float acc[GV_MAX_M] = {};
        for (int k = lane * 32; k < K; k += 32 * 32) {
            const uint4 codes = __ldg(reinterpret_cast<const uint4*>(W + (size_t)n * (K / 2) + k / 2));
            const uint16_t sc = __ldg(reinterpret_cast<const uint16_t*>(Sc + (size_t)n * (K / 16) + k / 16));
            const float s0 = fp8_to_float(sc & 0xff), s1 = fp8_to_float(sc >> 8);
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&codes);
            float w[32];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float s = i < 8 ? s0 : s1;
                w[2 * i] = kE2M1[b[i] & 15] * s;
                w[2 * i + 1] = kE2M1[b[i] >> 4] * s;
            }
#pragma unroll
            for (int m = 0; m < GV_MAX_M; ++m) {
                if (m < rows) {
                    const uint4* xp = reinterpret_cast<const uint4*>(X + (size_t)(r0 + m) * K + k);
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const uint4 xv = __ldg(xp + c);
                        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float2 xf = __bfloat1622float2(x2[i]);
                            acc[m] = fmaf(w[c * 8 + 2 * i], xf.x, acc[m]);
                            acc[m] = fmaf(w[c * 8 + 2 * i + 1], xf.y, acc[m]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < GV_MAX_M; ++m) {
            if (m < rows) {
                const float v = warp_sum(acc[m]);
                if (lane == 0) Y[(size_t)(r0 + m) * N + n] = alpha * v;
            }
        }
    }
}


// ---- decode-sized path (T <= 16) -----------------------------------------------------------

constexpr int SMALL_T = 16, SMALL_PAIRS = SMALL_T * TOPK;

// One block of 512 threads for T <= 16 tokens: routing (softmax, top 10, renormalize) and grouping of
// the pairs by expert. Outputs top_i/top_w [T, 10], offsets [513], pair_row [10 T], row_token [10 T],
// active [<= 10 T] and num_active.
__global__ void __launch_bounds__(512) route_small_kernel(const float* __restrict__ logits, int T, int* top_i,
                                                          float* top_w, int* offsets, int* pair_row, int* row_token,
                                                          int* active, int* num_active) {
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
            v[j] = logits[(size_t)t * E + j * 32 + lane];
            mx = fmaxf(mx, v[j]);
        }
        mx = warp_max(mx);
        float sum = 0;
#pragma unroll
        for (int j = 0; j < E / 32; ++j) {
            v[j] = __expf(v[j] - mx);
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

// acc[m] += w[0..32) . x_m[k .. k+32) for the rows m < rows, x_m = X[row_index(m)].
template <typename RowOf>
SQ_DEVICE void dot32(const float* w, const __nv_bfloat16* X, int ldx, int k, int rows, RowOf row_of, float* acc) {
#pragma unroll
    for (int m = 0; m < SMALL_T; ++m) {
        if (m < rows) {
            const uint4* xp = reinterpret_cast<const uint4*>(X + (size_t)row_of(m) * ldx + k);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 xv = __ldg(xp + c);
                const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 xf = __bfloat1622float2(x2[i]);
                    acc[m] = fmaf(w[c * 8 + 2 * i], xf.x, acc[m]);
                    acc[m] = fmaf(w[c * 8 + 2 * i + 1], xf.y, acc[m]);
                }
            }
        }
    }
}

constexpr int EW = 8;  // warps per expert-GEMV CTA

// act[row, n] = bf16(silu(gate_e[n] . x) * (up_e[n] . x)) for every row of active expert e,
// x = m[row_token[row]]. grid (640 / 8, active slots): warp w handles output feature n.
__global__ void __launch_bounds__(EW * 32) gate_up_kernel(const __nv_bfloat16* __restrict__ m,
                                                          const int* __restrict__ row_token,
                                                          const uint64_t* __restrict__ gw, const uint64_t* __restrict__ gs,
                                                          const float* __restrict__ g2, const uint64_t* __restrict__ uw,
                                                          const uint64_t* __restrict__ us, const float* __restrict__ u2,
                                                          const int* __restrict__ offsets, const int* __restrict__ active,
                                                          const int* __restrict__ num_active, __nv_bfloat16* act) {
    constexpr int N = 640, K = D;
    if ((int)blockIdx.y >= *num_active) return;
    const int e = active[blockIdx.y];
    const int r0 = offsets[e], rows = min(offsets[e + 1] - r0, SMALL_T);
    const int lane = threadIdx.x & 31, n = blockIdx.x * EW + (threadIdx.x >> 5);
    const uint8_t* GW = reinterpret_cast<const uint8_t*>(gw[e]) + (size_t)n * (K / 2);
    const uint8_t* GS = reinterpret_cast<const uint8_t*>(gs[e]) + (size_t)n * (K / 16);
    const uint8_t* UW = reinterpret_cast<const uint8_t*>(uw[e]) + (size_t)n * (K / 2);
    const uint8_t* US = reinterpret_cast<const uint8_t*>(us[e]) + (size_t)n * (K / 16);
    auto row_of = [&](int mm) { return row_token[r0 + mm]; };
    float ag[SMALL_T] = {}, au[SMALL_T] = {};
    for (int k = lane * 32; k < K; k += 32 * 32) {
        float w[32];
        decode32(__ldg(reinterpret_cast<const uint4*>(GW + k / 2)), __ldg(reinterpret_cast<const uint16_t*>(GS + k / 16)), w);
        dot32(w, m, K, k, rows, row_of, ag);
        decode32(__ldg(reinterpret_cast<const uint4*>(UW + k / 2)), __ldg(reinterpret_cast<const uint16_t*>(US + k / 16)), w);
        dot32(w, m, K, k, rows, row_of, au);
    }
    const float sg = g2[e], su = u2[e];
#pragma unroll
    for (int mm = 0; mm < SMALL_T; ++mm) {
        if (mm < rows) {
            const float g = warp_sum(ag[mm]) * sg, u = warp_sum(au[mm]) * su;
            if (lane == 0) act[(size_t)(r0 + mm) * N + n] = f2bf(silu(g) * u);
        }
    }
    (void)N;
}

// yd[row, n] = down_e[n] . act[row] for every row of active expert e. grid (2560 / 8, active slots).
__global__ void __launch_bounds__(EW * 32) down_kernel(const __nv_bfloat16* __restrict__ act,
                                                       const uint64_t* __restrict__ dw, const uint64_t* __restrict__ ds,
                                                       const float* __restrict__ d2, const int* __restrict__ offsets,
                                                       const int* __restrict__ active, const int* __restrict__ num_active,
                                                       float* yd) {
    constexpr int K = 640;
    if ((int)blockIdx.y >= *num_active) return;
    const int e = active[blockIdx.y];
    const int r0 = offsets[e], rows = min(offsets[e + 1] - r0, SMALL_T);
    const int lane = threadIdx.x & 31, n = blockIdx.x * EW + (threadIdx.x >> 5);
    const uint8_t* W = reinterpret_cast<const uint8_t*>(dw[e]) + (size_t)n * (K / 2);
    const uint8_t* Sc = reinterpret_cast<const uint8_t*>(ds[e]) + (size_t)n * (K / 16);
    auto row_of = [&](int mm) { return r0 + mm; };
    float acc[SMALL_T] = {};
    for (int k = lane * 32; k < K; k += 32 * 32) {
        float w[32];
        decode32(__ldg(reinterpret_cast<const uint4*>(W + k / 2)), __ldg(reinterpret_cast<const uint16_t*>(Sc + k / 16)), w);
        dot32(w, act, K, k, rows, row_of, acc);
    }
    const float s = d2[e];
#pragma unroll
    for (int mm = 0; mm < SMALL_T; ++mm) {
        if (mm < rows) {
            const float v = warp_sum(acc[mm]) * s;
            if (lane == 0) yd[(size_t)(r0 + mm) * D + n] = v;
        }
    }
}

}  // namespace

extern "C" int squash_moe_tile_rows() { return BM; }

extern "C" cudaError_t squash_moe_route(const float* logits, int T, int* top_i, float* top_w, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    route_kernel<<<(T + 7) / 8, 256, 0, st>>>(logits, T, top_i, top_w);
    SQ_RETURN_LAUNCH();
}

// Groups the T*10 routed pairs by expert. Scratch: counts/cursor [512], offsets [513],
// tile_expert/tile_row [max tiles], num_tiles [1], pair_row [T*10].
extern "C" cudaError_t squash_moe_plan(const int* top_i, int T, int* counts, int* cursor, int* offsets, int* tile_expert,
                                       int* tile_row, int* num_tiles, int* pair_row, cudaStream_t st) {
    const int pairs = T * TOPK;
    cudaError_t err = cudaMemsetAsync(counts, 0, E * sizeof(int), st);
    if (err != cudaSuccess) return err;
    if (pairs > 0) count_kernel<<<(pairs + 255) / 256, 256, 0, st>>>(top_i, pairs, counts);
    plan_kernel<<<1, E, 0, st>>>(counts, offsets, cursor, tile_expert, tile_row, num_tiles, BM);
    if (pairs > 0) scatter_kernel<<<(pairs + 255) / 256, 256, 0, st>>>(top_i, pairs, cursor, pair_row);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_moe_gather(const void* m, const int* pair_row, int T, void* xs, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    gather_kernel<<<T * TOPK, 128, 0, st>>>(static_cast<const __nv_bfloat16*>(m), pair_row, T * TOPK,
                                            static_cast<__nv_bfloat16*>(xs));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_moe_nvfp4_gemm(const void* xs, const uint64_t* w_ptr, const uint64_t* s_ptr,
                                             const float* ws2, const int* offsets, const int* tile_expert,
                                             const int* tile_row, const int* num_tiles, int max_tiles, float* y, int N,
                                             int K, cudaStream_t st) {
    if (K % BK != 0 || max_tiles == 0) return max_tiles == 0 ? cudaSuccess : cudaErrorInvalidValue;
    nvfp4_gemm_kernel<<<dim3((N + BN - 1) / BN, max_tiles), THREADS, 0, st>>>(
        static_cast<const __nv_bfloat16*>(xs), w_ptr, s_ptr, ws2, offsets, tile_expert, tile_row, num_tiles, y, N, K);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_swiglu(const float* gate, const float* up, void* act, size_t count, cudaStream_t st) {
    if (count == 0) return cudaSuccess;
    swiglu_kernel<<<(unsigned)((count + 255) / 256), 256, 0, st>>>(gate, up, static_cast<__nv_bfloat16*>(act), count);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_moe_combine(const float* yd, const int* pair_row, const float* top_w, const float* sg,
                                          const float* sh, float* y, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    combine_kernel<<<T, 256, 0, st>>>(yd, pair_row, top_w, sg, sh, y);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_moe_active(const int* counts, int* active, int* num_active, cudaStream_t st) {
    active_kernel<<<1, E, 0, st>>>(counts, active, num_active);
    SQ_RETURN_LAUNCH();
}

// max_active bounds the grid (<= min(512, 10 * tokens)); every expert must have <= 16 rows.
extern "C" cudaError_t squash_moe_nvfp4_gemv(const void* xs, const uint64_t* w_ptr, const uint64_t* s_ptr,
                                             const float* ws2, const int* offsets, const int* active,
                                             const int* num_active, int max_active, float* y, int N, int K,
                                             cudaStream_t st) {
    if (K % 32 != 0 || max_active == 0) return max_active == 0 ? cudaSuccess : cudaErrorInvalidValue;
    nvfp4_gemv_kernel<<<dim3((N + GV_ROWS - 1) / GV_ROWS, max_active), GV_WARPS * 32, 0, st>>>(
        static_cast<const __nv_bfloat16*>(xs), w_ptr, s_ptr, ws2, offsets, active, num_active, y, N, K);
    SQ_RETURN_LAUNCH();
}

extern "C" int squash_moe_small_tokens() { return SMALL_T; }

extern "C" cudaError_t squash_moe_route_small(const float* logits, int T, int* top_i, float* top_w, int* offsets,
                                              int* pair_row, int* row_token, int* active, int* num_active,
                                              cudaStream_t st) {
    if (T == 0 || T > SMALL_T) return T == 0 ? cudaSuccess : cudaErrorInvalidValue;
    route_small_kernel<<<1, 512, 0, st>>>(logits, T, top_i, top_w, offsets, pair_row, row_token, active, num_active);
    SQ_RETURN_LAUNCH();
}

// gate/up tables: [codes, scales] u64 x 512 each and weight_scale_2 f32 x 512.
extern "C" cudaError_t squash_moe_gate_up_small(const void* m, const int* row_token, const uint64_t* gw,
                                                const uint64_t* gs, const float* g2, const uint64_t* uw,
                                                const uint64_t* us, const float* u2, const int* offsets,
                                                const int* active, const int* num_active, int max_active, void* act,
                                                cudaStream_t st) {
    if (max_active == 0) return cudaSuccess;
    gate_up_kernel<<<dim3(640 / EW, max_active), EW * 32, 0, st>>>(static_cast<const __nv_bfloat16*>(m), row_token, gw,
                                                                    gs, g2, uw, us, u2, offsets, active, num_active,
                                                                    static_cast<__nv_bfloat16*>(act));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_moe_down_small(const void* act, const uint64_t* dw, const uint64_t* ds, const float* d2,
                                             const int* offsets, const int* active, const int* num_active,
                                             int max_active, float* yd, cudaStream_t st) {
    if (max_active == 0) return cudaSuccess;
    down_kernel<<<dim3(D / EW, max_active), EW * 32, 0, st>>>(static_cast<const __nv_bfloat16*>(act), dw, ds, d2, offsets,
                                                              active, num_active, yd);
    SQ_RETURN_LAUNCH();
}
