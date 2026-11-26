// Nemotron-H latent MoE: sigmoid router with correction bias (top 22 of 512), NVFP4 routed experts without
// a gate (down(relu(up(x))²)) in the 1024-wide latent space, the relu² of BF16 Linear outputs and the
// weighted combine (see crates/cuda/src/nemotron.rs).
//
// Small steps: GEMV-style; pairs grouped by expert, so an expert's weights are read once per step.
// Prefill: pairs grouped by expert into tiles for the shared W4A16 grouped GEMM
// (moe.cu quench6k_moe_nvfp4_gemm). Each pair's result depends only on its own row and expert.

#include <algorithm>
#include <cuda_fp4.h>

#include "mma.cuh"
#include "nemotron.cuh"
#include "variants.cuh"

using namespace nem;

namespace {

SQ_DEVICE float e4m3f(uint8_t v) {
    __nv_fp8_e4m3 f;
    f.__x = v;
    return float(f);
}

// ---- Router ------------------------------------------------------------------------------------------

// One block of 512 threads per row: 22 rounds of block-wide argmax of score + bias (ties: lower expert).
__global__ void __launch_bounds__(EXPERTS) moe_route_kernel(const float* logits, int ld, const float* bias, float scaling, int* sel_e,
                                                            float* sel_w) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float cv[EXPERTS / 32];
    __shared__ int ci[EXPERTS / 32];
    __shared__ int picked;
    __shared__ int wcount[EXPERTS / 32];
    __shared__ float ws[TOPK];
    __shared__ int es[TOPK];
    const int t = blockIdx.x, e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const float score = 1.0f / (1.0f + expf(-__ldcg(logits + (size_t)t * ld + e)));
    float choice = score + bias[e];
    if (!(choice == choice)) choice = -INFINITY;
    bool taken = false;
    for (int k = 0; k < TOPK; ++k) {
        float v = taken ? -INFINITY : choice;
        int idx = taken ? EXPERTS : e;
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, v, off);
            const int oi = __shfl_xor_sync(0xffffffff, idx, off);
            if (ov > v || (ov == v && oi < idx)) v = ov, idx = oi;
        }
        if (lane == 0) cv[warp] = v, ci[warp] = idx;
        __syncthreads();
        if (warp == 0) {
            v = lane < EXPERTS / 32 ? cv[lane] : -INFINITY;
            idx = lane < EXPERTS / 32 ? ci[lane] : EXPERTS;
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffff, v, off);
                const int oi = __shfl_xor_sync(0xffffffff, idx, off);
                if (ov > v || (ov == v && oi < idx)) v = ov, idx = oi;
            }
            if (lane == 0) picked = idx;
        }
        __syncthreads();
        if (e == picked) taken = true;
    }
    // Ascending expert order: rank among the taken experts.
    const unsigned m = __ballot_sync(0xffffffff, taken);
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
        float sum = 0.0f;
        for (int k = 0; k < TOPK; ++k) sum += ws[k];
        sel_e[(size_t)t * TOPK + e] = es[e];
        sel_w[(size_t)t * TOPK + e] = ws[e] / (sum + 1e-20f) * scaling;
    }
}

// ---- Small-step experts ------------------------------------------------------------------------------

// Order of a small step's pairs by expert (ties by pair): order[rank] = pair. One block, a thread per pair.
__global__ void pair_order_kernel(const int* sel_e, int pairs, int* order) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ int es[SMALL_ROWS * TOPK];
    for (int q = threadIdx.x; q < pairs; q += blockDim.x) es[q] = __ldcg(sel_e + q);
    __syncthreads();
    const int p = threadIdx.x;
    if (p >= pairs) return;
    const int e = es[p];
    int rank = 0;
    bool first = true;
    for (int q = 0; q < pairs; ++q) {
        const int f = es[q];
        rank += f < e || (f == e && q < p);
        first = first && !(f == e && q < p);
    }
    // Bit 30: first pair of its expert (the grouped kernels' group starts).
    order[rank] = p | (first ? 1 << 30 : 0);
}

// 16 NVFP4 weights (8 code bytes, low nibble first) times 16 activations, times the block scale.
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

constexpr int EX_WARPS = 8, EX_ROWS = 4, EX_CTAS_PER_SM = 2;
constexpr int UP_CPL = LATENT / 16 / 32;            // 2 chunks of 16 inputs per lane
constexpr int DOWN_CHUNKS = MOE_INTER / 16;         // 168
constexpr int DOWN_CPL = (DOWN_CHUNKS + 31) / 32;   // 6 (the last for lanes < 8)

// Pairs of one expert form a group; a CTA item is (group, tile of 8 warps x R rows). Each warp loads its R
// weight rows once, the group's activations are staged in shared memory, and every pair of the group is
// multiplied with the rows in registers. Per pair and row: lane l takes input chunks l + 32 j in order
// (fp4_dot16 each, summed from 0), then the warp_sum tree; so a pair's result does not depend on which
// other pairs share its expert or the step.
constexpr int GMAX = SMALL_ROWS * TOPK;

struct Groups {
    int pair[GMAX];
    int start[GMAX + 1];
    int count;
};

// The step's groups from the order (bit 30 marks a group's first pair). Ends with a barrier.
SQ_DEVICE void load_groups(Groups& g, const int* order, int pairs) {
    __shared__ int wsum[(GMAX + 255) / 256 * 8];  // one per warp of every pass of 256 pairs
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int base = 0; base < pairs; base += blockDim.x) {
        const int i = base + tid;
        const int v = i < pairs ? __ldcg(order + i) : 0;
        if (i < pairs) g.pair[i] = v & 0xffff;
        const bool f = i < pairs && (v >> 30 & 1);
        const unsigned m = __ballot_sync(0xffffffff, f);
        if (lane == 0) wsum[base / 32 + warp] = __popc(m);
        __syncthreads();
        if (f) {
            int pos = __popc(m & ((1u << lane) - 1));
            for (int w = 0; w < base / 32 + warp; ++w) pos += wsum[w];
            g.start[pos] = i;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int n = 0;
        for (int w = 0; w < (pairs + 31) / 32; ++w) n += wsum[w];
        g.count = n;
        g.start[n] = pairs;
    }
    __syncthreads();
}

SQ_DEVICE void bf16x16_s(const bf16* p, float* out) {
    const uint4 a = reinterpret_cast<const uint4*>(p)[0], b = reinterpret_cast<const uint4*>(p)[1];
    const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 fa = __bfloat1622float2(pa[i]), fb = __bfloat1622float2(pb[i]);
        out[2 * i] = fa.x, out[2 * i + 1] = fa.y, out[8 + 2 * i] = fb.x, out[8 + 2 * i + 1] = fb.y;
    }
}

template <int R>
__global__ void __launch_bounds__(EX_WARPS * 32) experts_up_grouped(const bf16* x, const int* sel_e, const int* order, NvExperts up, int pairs,
                                                                    bf16* act) {
    constexpr int TILE = EX_WARPS * R, TILES = MOE_INTER / TILE;
    // Wait first, then let the down kernel start: it reads the routing and the grouping before its own wait,
    // which is safe only once every CTA of this kernel has passed its wait (the grouping kernel, and through
    // it the routing, complete).
    pdl_wait();
    pdl_launch_dependents();
    __shared__ Groups gr;
    __shared__ __align__(16) bf16 xs[SMALL_ROWS][LATENT];
    load_groups(gr, order, pairs);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int items = gr.count * TILES;
    for (int item = blockIdx.x; item < items; item += gridDim.x) {
        const int k = item / TILES, n0 = (item % TILES) * TILE + warp * R;
        const int p0 = gr.start[k], np = gr.start[k + 1] - p0;
        const int e = __ldcg(sel_e + gr.pair[p0]);
        const size_t row0 = (size_t)e * MOE_INTER + n0;
        uint2 c[R][UP_CPL];
        uint8_t sc[R][UP_CPL];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int j = 0; j < UP_CPL; ++j) {
                const int chunk = lane + 32 * j;
                c[r][j] = ld_stream_u2(up.codes + (row0 + r) * (LATENT / 2) + chunk * 8);
                sc[r][j] = up.scales[(row0 + r) * (LATENT / 16) + chunk];
            }
        __syncthreads();
        for (int i = threadIdx.x; i < np * (LATENT / 8); i += blockDim.x) {
            const int q = i / (LATENT / 8), o = i % (LATENT / 8);
            reinterpret_cast<uint4*>(xs[q])[o] = __ldcg(reinterpret_cast<const uint4*>(x + (size_t)(gr.pair[p0 + q] / TOPK) * LATENT) + o);
        }
        __syncthreads();
        const float s2 = up.s2[e];
        for (int q = 0; q < np; ++q) {
            // Keep the codes packed (decode per pair, not hoisted out of the loop into 16 floats per chunk).
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int j = 0; j < UP_CPL; ++j) asm volatile("" : "+r"(c[r][j].x), "+r"(c[r][j].y));
            float acc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll
            for (int j = 0; j < UP_CPL; ++j) {
                float xv[16];
                bf16x16_s(&xs[q][(lane + 32 * j) * 16], xv);
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r] += fp4_dot16(c[r][j], sc[r][j], xv);
            }
            const int pair = gr.pair[p0 + q];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float v = warp_sum(acc[r]);
                if (lane == r) {
                    const float a = fmaxf(rb(v * s2), 0.0f);
                    act[(size_t)pair * MOE_INTER + n0 + r] = f2bf(a * a);
                }
            }
        }
    }
}

constexpr int DOWN_PB = 4;  // pairs' activations staged at a time

template <int R>
__global__ void __launch_bounds__(EX_WARPS * 32) experts_down_grouped(const bf16* act, const int* sel_e, const int* order, NvExperts down, int pairs,
                                                                      float* y) {
    constexpr int TILE = EX_WARPS * R, TILES = LATENT / TILE;
    pdl_launch_dependents();
    __shared__ Groups gr;
    __shared__ __align__(16) bf16 as[DOWN_PB][MOE_INTER];
    // The order and the routing are complete once this kernel starts (experts_up_grouped triggers it only after
    // its own wait), so the grouping and the first weights load while the up kernel finishes.
    load_groups(gr, order, pairs);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int items = gr.count * TILES;
    bool waited = false;
    for (int item = blockIdx.x; item < items; item += gridDim.x) {
        const int k = item / TILES, n0 = (item % TILES) * TILE + warp * R;
        const int p0 = gr.start[k], np = gr.start[k + 1] - p0;
        const int e = __ldcg(sel_e + gr.pair[p0]);
        const size_t row0 = (size_t)e * LATENT + n0;
        uint2 c[R][DOWN_CPL];
        uint8_t sc[R][DOWN_CPL];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int j = 0; j < DOWN_CPL; ++j) {
                const int chunk = lane + 32 * j;
                if (chunk < DOWN_CHUNKS) {
                    c[r][j] = ld_stream_u2(down.codes + (row0 + r) * (MOE_INTER / 2) + chunk * 8);
                    sc[r][j] = down.scales[(row0 + r) * (MOE_INTER / 16) + chunk];
                } else {
                    c[r][j] = make_uint2(0, 0);
                    sc[r][j] = 0;
                }
            }
        if (!waited) {
            pdl_wait();
            waited = true;
        }
        const float s2 = down.s2[e];
        for (int b0 = 0; b0 < np; b0 += DOWN_PB) {
            const int nb = min(DOWN_PB, np - b0);
            __syncthreads();
            for (int i = threadIdx.x; i < nb * (MOE_INTER / 8); i += blockDim.x) {
                const int q = i / (MOE_INTER / 8), o = i % (MOE_INTER / 8);
                reinterpret_cast<uint4*>(as[q])[o] = __ldcg(reinterpret_cast<const uint4*>(act + (size_t)gr.pair[p0 + b0 + q] * MOE_INTER) + o);
            }
            __syncthreads();
            for (int q = 0; q < nb; ++q) {
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int j = 0; j < DOWN_CPL; ++j) asm volatile("" : "+r"(c[r][j].x), "+r"(c[r][j].y));
                float acc[R];
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll
                for (int j = 0; j < DOWN_CPL; ++j) {
                    const int chunk = lane + 32 * j;
                    if (chunk >= DOWN_CHUNKS) continue;
                    float xv[16];
                    bf16x16_s(&as[q][chunk * 16], xv);
#pragma unroll
                    for (int r = 0; r < R; ++r) acc[r] += fp4_dot16(c[r][j], sc[r][j], xv);
                }
                const int pair = gr.pair[p0 + b0 + q];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float v = warp_sum(acc[r]);
                    if (lane == r) y[(size_t)pair * LATENT + n0 + r] = v * s2;
                }
            }
        }
    }
    if (!waited) pdl_wait();
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

// ---- Prefill grouping ----------------------------------------------------------------------------------

__global__ void count_kernel(const int* sel_e, int pairs, int* counts) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) atomicAdd(&counts[__ldcg(sel_e + i)], 1);
}

// One block of 512 threads: expert row offsets and the list of tile_rows-row tiles (expert, first row).
__global__ void __launch_bounds__(EXPERTS) plan_kernel(const int* counts, int* offsets, int* cursor, int* tile_expert, int* tile_row,
                                                       int* num_tiles, int bm) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ int s_rows[EXPERTS], s_tiles[EXPERTS];
    const int e = threadIdx.x;
    const int c = __ldcg(counts + e);
    s_rows[e] = c;
    s_tiles[e] = (c + bm - 1) / bm;
    __syncthreads();
    for (int off = 1; off < EXPERTS; off <<= 1) {
        const int a = e >= off ? s_rows[e - off] : 0, b = e >= off ? s_tiles[e - off] : 0;
        __syncthreads();
        s_rows[e] += a;
        s_tiles[e] += b;
        __syncthreads();
    }
    const int row_begin = s_rows[e] - c, tile_begin = s_tiles[e] - (c + bm - 1) / bm;
    offsets[e] = row_begin;
    cursor[e] = row_begin;
    if (e == EXPERTS - 1) {
        offsets[EXPERTS] = s_rows[e];
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
    if (i < pairs) pair_row[i] = atomicAdd(&cursor[__ldcg(sel_e + i)], 1);
}

// xs[pair_row[pair]] = x[pair / 22] (rows of LATENT BF16).
__global__ void gather_kernel(const bf16* x, const int* pair_row, bf16* xs) {
    pdl_launch_dependents();
    pdl_wait();
    const int pair = blockIdx.x;
    const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)(pair / TOPK) * LATENT);
    uint4* dst = reinterpret_cast<uint4*>(xs + (size_t)__ldcg(pair_row + pair) * LATENT);
    for (int i = threadIdx.x; i < LATENT / 8; i += blockDim.x) dst[i] = __ldcg(src + i);
}

__global__ void relu2_kernel(const float* x, int ldx, int width, int rows, bf16* out, int ldo) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * width) return;
    const int r = (int)(i / width), c = (int)(i % width);
    const float v = fmaxf(rb(__ldcg(x + (size_t)r * ldx + c)), 0.0f);
    out[(size_t)r * ldo + c] = f2bf(v * v);
}

// Grid (rows, LATENT / 256): out[r, n] = bf16(FP32 sum in ascending expert order of bf16(y) * w).
__global__ void combine_kernel(const float* y, const int* pair_row, const float* sel_w, bf16* out) {
    pdl_launch_dependents();
    pdl_wait();
    const int r = blockIdx.x, n = blockIdx.y * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int k = 0; k < TOPK; ++k) {
        const int pair = r * TOPK + k;
        const int row = pair_row ? __ldcg(pair_row + pair) : pair;
        acc = __fadd_rn(acc, __fmul_rn(rb(__ldcg(y + (size_t)row * LATENT + n)), __ldcg(sel_w + pair)));
    }
    out[(size_t)r * LATENT + n] = f2bf(acc);
}

// ---- Prefill W4A16 grouped GEMM ------------------------------------------------------------------------
// Y[row, N] = s2[e] * X[row, K] . dequant(W_e)[N, K]^T for every row of expert e's tiles (W contiguous over
// experts: codes [N][K/2], scales E4M3 [N][K/16]). The same values and MMA sequence per output as moe.cu's
// nvfp4_gemm_kernel (weights decoded exactly to BF16, m16n8k16 MMAs in ascending k, FP32 accumulators, times
// s2), so the bits equal it; with EPI 1 the epilogue is relu2's (bf16(relu(bf16(y))²)), with EPI 2 it stores
// bf16(y). A and the weight codes stream through STAGES cp.async stages; a tile's scales load once.
constexpr int WG_BN = 128, WG_BK = 32, WG_LDS = WG_BK + 8, WG_THREADS = 256;

SQ_DEVICE void cp_async8(void* smem, const void* gmem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(gmem));
}

template <int BM, int STAGES>
constexpr size_t wg_smem(int K) {
    return (size_t)STAGES * BM * WG_LDS * 2 + (size_t)STAGES * WG_BN * 16 + (size_t)2 * WG_BN * WG_LDS * 2 + (size_t)WG_BN * (K / 16);
}

template <int BM, int STAGES, int EPI>
__global__ void __launch_bounds__(WG_THREADS) w4_gemm_kernel(const bf16* X, NvExperts W, const int* offsets, const int* tile_expert,
                                                             const int* tile_row, const int* num_tiles, void* Y, int N, int K) {
    constexpr int MI = BM / 32;  // m16 blocks per warp (2 x 4 warps, warp tile BM / 2 x 32)
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ __align__(16) unsigned char wg_raw[];
    bf16* sA = reinterpret_cast<bf16*>(wg_raw);
    uint8_t* sR = wg_raw + (size_t)STAGES * BM * WG_LDS * 2;
    bf16* sB = reinterpret_cast<bf16*>(sR + STAGES * WG_BN * 16);
    uint8_t* sS = reinterpret_cast<uint8_t*>(sB + 2 * WG_BN * WG_LDS);
    if ((int)blockIdx.y >= __ldcg(num_tiles)) return;
    const int e = __ldcg(tile_expert + blockIdx.y), r0 = __ldcg(tile_row + blockIdx.y);
    const int r_end = min(r0 + BM, __ldcg(offsets + e + 1));
    const uint8_t* Wc = W.codes + (size_t)e * N * (K / 2);
    const uint8_t* Ws = W.scales + (size_t)e * N * (K / 16);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp >> 2, warp_n = warp & 3;
    const int n0 = blockIdx.x * WG_BN;
    const int k_tiles = K / WG_BK, SR = K / 16;

    // The tile's scales (rows past N zero), part of the first group.
    for (int i = tid; i < WG_BN * (SR / 8); i += WG_THREADS) {
        const int r = i / (SR / 8), c = i % (SR / 8);
        if (n0 + r < N) cp_async8(sS + r * SR + c * 8, Ws + (size_t)(n0 + r) * SR + c * 8);
        else *reinterpret_cast<uint2*>(sS + r * SR + c * 8) = make_uint2(0, 0);
    }
    auto issue = [&](int kt) {
        if (kt < k_tiles) {
            const int st = kt % STAGES, k = kt * WG_BK;
            for (int i = tid; i < BM * 4; i += WG_THREADS) {
                const int r = i >> 2, c = (i & 3) * 8;
                const bool ok = r0 + r < r_end;
                cp_async16(&sA[(st * BM + r) * WG_LDS + c], X + (size_t)(ok ? r0 + r : r0) * K + k + c, ok);
            }
            if (tid < WG_BN) {
                const bool ok = n0 + tid < N;
                cp_async16(sR + (st * WG_BN + tid) * 16, Wc + (size_t)(ok ? n0 + tid : n0) * (K / 2) + k / 2, ok);
            }
        }
        cp_async_commit();
    };
    // Raw codes of k-tile kt -> BF16 weights in sB[buf]: thread = (row, half of the 32 values).
    auto decode = [&](int kt, int buf) {
        const int r = tid >> 1, hf = tid & 1;
        const uint8_t* raw = sR + ((kt % STAGES) * WG_BN + r) * 16 + hf * 8;
        const uint2 cw = *reinterpret_cast<const uint2*>(raw);
        const __half hs = __float2half(e4m3f(sS[r * SR + kt * 2 + hf]));
        const __half2 sc = __halves2half2(hs, hs);
        const uint8_t* bb = reinterpret_cast<const uint8_t*>(&cw);
        uint32_t packed[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2(bb[i], __NV_E2M1);
            const float2 f = __half22float2(__hmul2(*reinterpret_cast<const __half2*>(&hr), sc));
            const __nv_bfloat162 v = __floats2bfloat162_rn(f.x, f.y);
            packed[i] = *reinterpret_cast<const uint32_t*>(&v);
        }
        uint4* dst = reinterpret_cast<uint4*>(&sB[(buf * WG_BN + r) * WG_LDS + hf * 16]);
        dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
    };

    float acc[MI][4][4] = {};
#pragma unroll
    for (int s0 = 0; s0 < STAGES - 1; ++s0) issue(s0);
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    decode(0, 0);
    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_async_wait<STAGES - 3>();
        __syncthreads();
        issue(kt + STAGES - 1);
        const bf16* a_tile = sA + (size_t)(kt % STAGES) * BM * WG_LDS;
        const bf16* b_tile = sB + (size_t)(kt & 1) * WG_BN * WG_LDS;
#pragma unroll
        for (int kk = 0; kk < WG_BK; kk += 16) {
            uint32_t a[MI][4], b[2][4];
#pragma unroll
            for (int i = 0; i < MI; ++i)
                ldmatrix_x4(a[i], &a_tile[(warp_m * (BM / 2) + i * 16 + (lane & 15)) * WG_LDS + kk + (lane >> 4) * 8]);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                ldmatrix_x4(b[j], &b_tile[(warp_n * 32 + j * 16 + (lane & 7) + ((lane >> 4) << 3)) * WG_LDS + kk + ((lane >> 3) & 1) * 8]);
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_bf16(acc[i][j], a[i], &b[j >> 1][(j & 1) * 2]);
        }
        if (kt + 1 < k_tiles) decode(kt + 1, (kt + 1) & 1);
    }
    const float alpha = W.s2[e];
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = r0 + warp_m * (BM / 2) + i * 16 + g + h * 8;
                const int n = n0 + warp_n * 32 + j * 8 + t4 * 2;
                if (r >= r_end || n >= N) continue;  // N is even: n + 1 < N
                const float v0 = alpha * acc[i][j][h * 2], v1 = alpha * acc[i][j][h * 2 + 1];
                if constexpr (EPI == 0) {
                    *reinterpret_cast<float2*>(static_cast<float*>(Y) + (size_t)r * N + n) = make_float2(v0, v1);
                } else if constexpr (EPI == 1) {
                    const float a0 = fmaxf(rb(v0), 0.0f), a1 = fmaxf(rb(v1), 0.0f);
                    *reinterpret_cast<__nv_bfloat162*>(static_cast<bf16*>(Y) + (size_t)r * N + n) = __floats2bfloat162_rn(a0 * a0, a1 * a1);
                } else {
                    *reinterpret_cast<__nv_bfloat162*>(static_cast<bf16*>(Y) + (size_t)r * N + n) = __floats2bfloat162_rn(v0, v1);
                }
            }
}

// Version 3 of the grouped GEMM (same outputs, bit for bit; nem.w4v=3, the default): 48 KB of shared
// memory, so two CTAs share an SM and one's weight decode overlaps the other's MMAs. A streams through
// V3_ST cp.async stages, swizzled (16-byte chunk c of row r at c ^ ((r / 2) & 3), no padding); the codes and
// scale of k-tile kt + 2 are fetched into registers while kt's MMAs run and decoded to BF16 (swizzled,
// double-buffered) after them; the CTA's weight rows are bulk-prefetched into L2 at its start. Each warp
// row takes every other m16 block of the tile and skips blocks past the tile's rows, so the MMA work
// follows an expert's rows in steps of 16 instead of 128. Shared-memory addresses are precomputed per
// thread and the pointers advance per k-tile (the 128-register budget of two CTAs per SM).
constexpr int V3_BM = 128, V3_BK = 32, V3_ST = 4;
constexpr size_t V3_SMEM = (size_t)(V3_ST * V3_BM + 2 * WG_BN) * V3_BK * 2;
// Bulk L2 prefetch of [p, p + bytes) in 8 KiB pieces spread over the block's threads.
SQ_DEVICE void l2_prefetch(const void* p, size_t bytes, int tid, int nthreads) {
    constexpr size_t PIECE = 8192;
    const char* c = static_cast<const char*>(p);
    for (size_t off = (size_t)tid * PIECE; off < bytes; off += (size_t)nthreads * PIECE) {
        const unsigned len = (unsigned)min(PIECE, bytes - off);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(c + off), "r"(len) : "memory");
    }
}

SQ_DEVICE void cp16s(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
SQ_DEVICE void ldsm4(uint32_t* r, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
SQ_DEVICE void sts128(uint32_t a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
// Byte offset of 16-byte chunk c of row r in a swizzled [rows][32] BF16 tile.
SQ_DEVICE uint32_t v3_off(int r, int c) { return (uint32_t)(r * 64 + ((c ^ ((r >> 1) & 3)) << 4)); }

template <int EPI>
__global__ void __launch_bounds__(WG_THREADS, 2) w4v3_kernel(const bf16* X, NvExperts W, const int* offsets, const int* tile_expert,
                                                             const int* tile_row, const int* num_tiles, void* Y, int N, int K) {
    constexpr int BM = V3_BM, MI = BM / 32, ST = V3_ST, TILE_B = BM * V3_BK * 2;  // 8 KiB per A stage or B buffer
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ __align__(128) unsigned char v3_raw[];
    const uint32_t sA = static_cast<uint32_t>(__cvta_generic_to_shared(v3_raw)), sB = sA + ST * TILE_B;
    if ((int)blockIdx.y >= __ldcg(num_tiles)) return;
    const int e = __ldcg(tile_expert + blockIdx.y), r0 = __ldcg(tile_row + blockIdx.y);
    const int nrows = min(BM, __ldcg(offsets + e + 1) - r0), nr16 = (nrows + 15) & ~15;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp >> 2, warp_n = warp & 3;
    const int n0 = blockIdx.x * WG_BN;
    const int k_tiles = K / V3_BK, SR = K / 16;
    {
        const int rows = min(WG_BN, N - n0);
        l2_prefetch(W.codes + ((size_t)e * N + n0) * (K / 2), (size_t)rows * (K / 2), tid, WG_THREADS);
        l2_prefetch(W.scales + ((size_t)e * N + n0) * SR, (size_t)rows * SR, tid, WG_THREADS);
    }
    // A copies: rows ra and ra + 64, chunk ca.
    const int ra = tid >> 2, ca = tid & 3;
    const uint32_t a_dst = sA + v3_off(ra, ca);
    const bool go0 = ra < nr16, ok0 = ra < nrows, go1 = ra + 64 < nr16, ok1 = ra + 64 < nrows;
    const bf16* src0 = X + (size_t)(r0 + (ok0 ? ra : 0)) * K + ca * 8;
    const bf16* src1 = X + (size_t)(r0 + (ok1 ? ra + 64 : 0)) * K + ca * 8;
    auto issue_a = [&](int kt) {
        if (kt < k_tiles) {
            const uint32_t d = a_dst + (kt % ST) * TILE_B;
            const int k = kt * V3_BK;
            if (go0) cp16s(d, src0 + k, ok0);
            if (go1) cp16s(d + 64 * 64, src1 + k, ok1);
        }
        cp_async_commit();
    };
    // Weights: thread = (row br, half hf): 8 code bytes and one scale per k-tile.
    const int br = tid >> 1, hf = tid & 1;
    const bool bok = n0 + br < N;
    const uint8_t* wc = W.codes + ((size_t)e * N + (bok ? n0 + br : 0)) * (K / 2) + hf * 8;
    const uint8_t* ws = W.scales + ((size_t)e * N + (bok ? n0 + br : 0)) * SR + hf;
    const uint32_t d_off0 = v3_off(br, hf * 2), d_off1 = v3_off(br, hf * 2 + 1);
    auto fetch = [&](int kt, uint2& c, uint32_t& sc) {
        c = make_uint2(0, 0);
        sc = 0;
        if (kt < k_tiles && bok) {
            c = __ldg(reinterpret_cast<const uint2*>(wc + kt * 16));
            sc = __ldg(ws + kt * 2);
        }
    };
    auto decode = [&](const uint2& c, uint32_t sc, int buf) {
        const __half hs = __float2half(e4m3f((uint8_t)sc));
        const __half2 h2 = __halves2half2(hs, hs);
        const uint8_t* bb = reinterpret_cast<const uint8_t*>(&c);
        uint32_t pk[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2(bb[i], __NV_E2M1);
            const float2 f = __half22float2(__hmul2(*reinterpret_cast<const __half2*>(&hr), h2));
            const __nv_bfloat162 v = __floats2bfloat162_rn(f.x, f.y);
            pk[i] = *reinterpret_cast<const uint32_t*>(&v);
        }
        const uint32_t t = sB + buf * TILE_B;
        sts128(t + d_off0, pk[0], pk[1], pk[2], pk[3]);
        sts128(t + d_off1, pk[4], pk[5], pk[6], pk[7]);
    };
    // ldmatrix addresses (k16 step 0; step 1 is chunk ^ 2 = offset ^ 32): A block i at + i * 2048 (32 rows),
    // B fragment j at + j * 1024 (16 rows).
    const int ar = warp_m * 16 + (lane & 15);
    const uint32_t a_ld0 = v3_off(ar, lane >> 4);
    const int brow = warp_n * 32 + (lane & 7) + ((lane >> 4) << 3);
    const uint32_t b_ld0 = v3_off(brow, (lane >> 3) & 1);
    bool live[MI];
#pragma unroll
    for (int i = 0; i < MI; ++i) live[i] = warp_m * 16 + i * 32 < nrows;

    float acc[MI][4][4] = {};
    uint2 c0, c1;
    uint32_t s0, s1;
    fetch(0, c0, s0);
    fetch(1, c1, s1);
#pragma unroll
    for (int st = 0; st < ST - 1; ++st) issue_a(st);
    decode(c0, s0, 0);
    auto step = [&](int kt, uint2& cf, uint32_t& sf, const uint2& cd, uint32_t sd) {
        cp_async_wait<ST - 2>();
        __syncthreads();
        issue_a(kt + ST - 1);
        fetch(kt + 2, cf, sf);
        const uint32_t at = sA + (kt % ST) * TILE_B, bt = sB + (kt & 1) * TILE_B;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t b[2][4];
            const uint32_t ao = at + (a_ld0 ^ (kk * 32)), bo = bt + (b_ld0 ^ (kk * 32));
            ldsm4(b[0], bo);
            ldsm4(b[1], bo + 1024);
#pragma unroll
            for (int i = 0; i < MI; ++i)
                if (live[i]) {
                    uint32_t a[4];
                    ldsm4(a, ao + i * 2048);
#pragma unroll
                    for (int j = 0; j < 4; ++j) mma_bf16(acc[i][j], a, &b[j >> 1][(j & 1) * 2]);
                }
        }
        if (kt + 1 < k_tiles) decode(cd, sd, (kt + 1) & 1);
    };
    for (int kt = 0; kt < k_tiles; kt += 2) {
        step(kt, c0, s0, c1, s1);
        step(kt + 1, c1, s1, c0, s0);
    }
    cp_async_wait<0>();

    const float alpha = W.s2[e];
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int rl = warp_m * 16 + i * 32 + g + h * 8;
                const int n = n0 + warp_n * 32 + j * 8 + t4 * 2;
                if (rl >= nrows || n >= N) continue;  // N is even: n + 1 < N
                const int r = r0 + rl;
                const float v0 = alpha * acc[i][j][h * 2], v1 = alpha * acc[i][j][h * 2 + 1];
                if constexpr (EPI == 1) {
                    const float a0 = fmaxf(rb(v0), 0.0f), a1 = fmaxf(rb(v1), 0.0f);
                    *reinterpret_cast<__nv_bfloat162*>(static_cast<bf16*>(Y) + (size_t)r * N + n) = __floats2bfloat162_rn(a0 * a0, a1 * a1);
                } else {
                    *reinterpret_cast<__nv_bfloat162*>(static_cast<bf16*>(Y) + (size_t)r * N + n) = __floats2bfloat162_rn(v0, v1);
                }
            }
}

template <int EPI>
cudaError_t launch_w4v3(const void* x, NvExperts w, const int* offsets, const int* te, const int* tr, const int* nt, int max_tiles, void* y, int N, int K,
                        cudaStream_t st) {
    if (K % (2 * V3_BK) != 0) return cudaErrorInvalidValue;
    static bool configured = false;
    if (!configured) {
        SQ_CHECK_LAUNCH(cudaFuncSetAttribute(w4v3_kernel<EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)V3_SMEM));
        SQ_CHECK_LAUNCH(cudaFuncSetAttribute(w4v3_kernel<EPI>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        configured = true;
    }
    SQ_CHECK_LAUNCH(sq_launch(w4v3_kernel<EPI>, dim3((N + WG_BN - 1) / WG_BN, max_tiles), WG_THREADS, V3_SMEM, st, static_cast<const bf16*>(x), w,
                              offsets, te, tr, nt, y, N, K));
    return cudaSuccess;
}

// Thin tiles for small prefill steps (few rows per expert): 32-row tiles, 128 output columns per CTA, 128-wide
// k-tiles. Each warp owns 16 columns over all the tile's rows; its lanes read their weight rows' codes and
// scales straight into registers (one k-tile ahead) and decode them into the MMA's B fragments (lane quad
// q of a column takes code bytes q and q + 4 of each 8: the fragment's k, k + 1 and k + 8, k + 9), so the
// weights need no shared memory and no barrier; only A is staged (cp.async, 3 stages, 16-byte chunk c of
// row r at c ^ (r & 7)). Same decoded values and m16n8k16 sequence per output as the other versions.
constexpr int TH_BM = 32, TH_BK = 128, TH_ST = 3;
constexpr size_t TH_SMEM = (size_t)TH_ST * TH_BM * TH_BK * 2;

template <int EPI>
__global__ void __launch_bounds__(WG_THREADS, 2) w4thin_kernel(const bf16* X, NvExperts W, const int* offsets, const int* tile_expert,
                                                               const int* tile_row, const int* num_tiles, void* Y, int N, int K) {
    constexpr int TILE_A = TH_BM * TH_BK * 2;  // 8 KiB per stage
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ __align__(128) unsigned char th_raw[];
    const uint32_t sA = static_cast<uint32_t>(__cvta_generic_to_shared(th_raw));
    if ((int)blockIdx.y >= __ldcg(num_tiles)) return;
    const int e = __ldcg(tile_expert + blockIdx.y), r0 = __ldcg(tile_row + blockIdx.y);
    const int nrows = min(TH_BM, __ldcg(offsets + e + 1) - r0), nr16 = (nrows + 15) & ~15;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n0 = blockIdx.x * WG_BN;
    const int k_tiles = K / TH_BK, SR = K / 16;
    {
        const int rows = min(WG_BN, N - n0);
        l2_prefetch(W.codes + ((size_t)e * N + n0) * (K / 2), (size_t)rows * (K / 2), tid, WG_THREADS);
        l2_prefetch(W.scales + ((size_t)e * N + n0) * SR, (size_t)rows * SR, tid, WG_THREADS);
    }
    // A copies: 32 rows x 16 chunks per stage, two per thread (rows ra and ra + 16, chunk ca).
    const int ra = tid >> 4, ca = tid & 15;
    auto a_off = [](int r, int c) { return (uint32_t)(r * 256 + ((c ^ (r & 7)) << 4)); };
    const uint32_t a_dst0 = sA + a_off(ra, ca), a_dst1 = sA + a_off(ra + 16, ca);
    const bool go0 = ra < nr16, ok0 = ra < nrows, go1 = ra + 16 < nr16, ok1 = ra + 16 < nrows;
    const bf16* src0 = X + (size_t)(r0 + (ok0 ? ra : 0)) * K + ca * 8;
    const bf16* src1 = X + (size_t)(r0 + (ok1 ? ra + 16 : 0)) * K + ca * 8;
    auto issue_a = [&](int kt) {
        if (kt < k_tiles) {
            const uint32_t st = (kt % TH_ST) * TILE_A;
            const int k = kt * TH_BK;
            if (go0) cp16s(a_dst0 + st, src0 + k, ok0);
            if (go1) cp16s(a_dst1 + st, src1 + k, ok1);
        }
        cp_async_commit();
    };
    // This lane's weight rows: column n0 + warp * 16 + j * 8 + lane / 4 for n8 tile j; quad lane q.
    const int q = lane & 3;
    const uint8_t* wc[2];
    const uint8_t* wsc[2];
    bool bok[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int n = n0 + warp * 16 + j * 8 + (lane >> 2);
        bok[j] = n < N;
        wc[j] = W.codes + ((size_t)e * N + (bok[j] ? n : 0)) * (K / 2);
        wsc[j] = W.scales + ((size_t)e * N + (bok[j] ? n : 0)) * SR;
    }
    struct Slab {
        uint4 c[2][4];  // 64 code bytes (128 values) per n8 tile
        uint2 s[2];     // 8 scales per n8 tile
    };
    auto fetch = [&](int kt, Slab& b) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (kt < k_tiles && bok[j]) {
                const uint4* cp = reinterpret_cast<const uint4*>(wc[j] + (size_t)kt * (TH_BK / 2));
#pragma unroll
                for (int u = 0; u < 4; ++u) b.c[j][u] = __ldg(cp + u);
                b.s[j] = __ldg(reinterpret_cast<const uint2*>(wsc[j] + kt * (TH_BK / 16)));
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) b.c[j][u] = make_uint4(0, 0, 0, 0);
                b.s[j] = make_uint2(0, 0);
            }
        }
    };
    // Code byte (low nibble: the even k) -> BF16 pair times the block scale, as the other versions decode.
    auto dec = [](uint32_t byte, __half2 sc) {
        const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)byte, __NV_E2M1);
        const float2 f = __half22float2(__hmul2(*reinterpret_cast<const __half2*>(&hr), sc));
        const __nv_bfloat162 v = __floats2bfloat162_rn(f.x, f.y);
        return *reinterpret_cast<const uint32_t*>(&v);
    };
    const bool live1 = nrows > 16;
    const int ar = lane & 15;
    float acc[2][2][4] = {};
    auto compute = [&](int kt, const Slab& b) {
        const uint32_t at = sA + (kt % TH_ST) * TILE_A;
#pragma unroll
        for (int kb = 0; kb < TH_BK / 16; ++kb) {
            uint32_t bf[2][2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                // Words 2 kb, 2 kb + 1 of the 16: uint4 kb / 2, components (kb % 2) * 2, + 1.
                const uint4 cw = b.c[j][kb >> 1];
                const uint32_t wlo = kb & 1 ? cw.z : cw.x, whi = kb & 1 ? cw.w : cw.y;
                const uint32_t sw = kb < 4 ? b.s[j].x : b.s[j].y;
                const __half_raw hs = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)__byte_perm(sw, 0, kb & 3), __NV_E4M3);
                const __half h = hs;
                const __half2 sc = __halves2half2(h, h);
                bf[j][0] = dec(__byte_perm(wlo, 0, q), sc);
                bf[j][1] = dec(__byte_perm(whi, 0, q), sc);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                if (i == 1 && !live1) continue;
                uint32_t a[4];
                const int r = i * 16 + ar, c = kb * 2 + (lane >> 4);
                ldsm4(a, at + a_off(r, c));
#pragma unroll
                for (int j = 0; j < 2; ++j) mma_bf16(acc[i][j], a, bf[j]);
            }
        }
    };

    Slab b0, b1;
    fetch(0, b0);
#pragma unroll
    for (int st = 0; st < TH_ST - 1; ++st) issue_a(st);
    auto step = [&](int kt, const Slab& cur, Slab& nxt) {
        cp_async_wait<TH_ST - 2>();
        __syncthreads();
        issue_a(kt + TH_ST - 1);
        fetch(kt + 1, nxt);
        compute(kt, cur);
    };
    int kt = 0;
    for (; kt + 1 < k_tiles; kt += 2) {
        step(kt, b0, b1);
        step(kt + 1, b1, b0);
    }
    if (kt < k_tiles) step(kt, b0, b1);
    cp_async_wait<0>();

    const float alpha = W.s2[e];
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int rl = i * 16 + g + h * 8;
                const int n = n0 + warp * 16 + j * 8 + t4 * 2;
                if (rl >= nrows || n >= N) continue;
                const int r = r0 + rl;
                const float v0 = alpha * acc[i][j][h * 2], v1 = alpha * acc[i][j][h * 2 + 1];
                if constexpr (EPI == 1) {
                    const float a0 = fmaxf(rb(v0), 0.0f), a1 = fmaxf(rb(v1), 0.0f);
                    *reinterpret_cast<__nv_bfloat162*>(static_cast<bf16*>(Y) + (size_t)r * N + n) = __floats2bfloat162_rn(a0 * a0, a1 * a1);
                } else {
                    *reinterpret_cast<__nv_bfloat162*>(static_cast<bf16*>(Y) + (size_t)r * N + n) = __floats2bfloat162_rn(v0, v1);
                }
            }
}

template <int EPI>
cudaError_t launch_w4thin(const void* x, NvExperts w, const int* offsets, const int* te, const int* tr, const int* nt, int max_tiles, void* y, int N,
                          int K, cudaStream_t st) {
    if (K % TH_BK != 0) return cudaErrorInvalidValue;
    static bool configured = false;
    if (!configured) {
        SQ_CHECK_LAUNCH(cudaFuncSetAttribute(w4thin_kernel<EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TH_SMEM));
        configured = true;
    }
    SQ_CHECK_LAUNCH(sq_launch(w4thin_kernel<EPI>, dim3((N + WG_BN - 1) / WG_BN, max_tiles), WG_THREADS, TH_SMEM, st, static_cast<const bf16*>(x), w,
                              offsets, te, tr, nt, y, N, K));
    return cudaSuccess;
}

// moe_combine from BF16 expert outputs (bf16(y) stored by the down GEMM: the same values combine_kernel rounds).
__global__ void combine_bf16_kernel(const bf16* y, const int* pair_row, const float* sel_w, bf16* out) {
    pdl_launch_dependents();
    pdl_wait();
    const int r = blockIdx.x, n = blockIdx.y * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int k = 0; k < TOPK; ++k) {
        const int pair = r * TOPK + k;
        const int row = __ldcg(pair_row + pair);
        acc = __fadd_rn(acc, __fmul_rn(bf2f(__ldcg(y + (size_t)row * LATENT + n)), __ldcg(sel_w + pair)));
    }
    out[(size_t)r * LATENT + n] = f2bf(acc);
}

template <int BM, int STAGES, int EPI>
cudaError_t launch_w4(const void* x, NvExperts w, const int* offsets, const int* te, const int* tr, const int* nt, int max_tiles, void* y, int N, int K,
                      cudaStream_t st) {
    const size_t smem = wg_smem<BM, STAGES>(K);
    static bool configured = false;
    if (!configured) {
        SQ_CHECK_LAUNCH(cudaFuncSetAttribute(w4_gemm_kernel<BM, STAGES, EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)wg_smem<BM, STAGES>(MOE_INTER)));
        configured = true;
    }
    SQ_CHECK_LAUNCH(sq_launch(w4_gemm_kernel<BM, STAGES, EPI>, dim3((N + WG_BN - 1) / WG_BN, max_tiles), WG_THREADS, smem, st, static_cast<const bf16*>(x), w,
                              offsets, te, tr, nt, y, N, K));
    return cudaSuccess;
}

int w4_variant() { return sq_variant("nem.w4", 1284); }

}  // namespace

extern "C" cudaError_t quench6k_nem_moe_route(const float* logits, int ld, const float* bias, float scaling, int rows, int* sel_e, float* sel_w,
                                              cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(moe_route_kernel, rows, EXPERTS, 0, st, logits, ld, bias, scaling, sel_e, sel_w));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_nem_moe_experts_small(const void* x, const int* sel_e, NvExperts up, NvExperts down, int rows, int* order,
                                                      void* act, float* y, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    if (rows > SMALL_ROWS) return cudaErrorInvalidValue;
    const int pairs = rows * TOPK;
    SQ_CHECK_LAUNCH(sq_launch(pair_order_kernel, 1, (pairs + 31) / 32 * 32, 0, st, sel_e, pairs, order));
    const int blocks = sm_count() * EX_CTAS_PER_SM;
    SQ_CHECK_LAUNCH(sq_launch(experts_up_grouped<EX_ROWS>, blocks, EX_WARPS * 32, 0, st, static_cast<const bf16*>(x), sel_e, (const int*)order, up,
                              pairs, static_cast<bf16*>(act)));
    SQ_CHECK_LAUNCH(sq_launch(experts_down_grouped<EX_ROWS>, blocks, EX_WARPS * 32, 0, st, (const bf16*)act, sel_e, (const int*)order, down, pairs,
                              y));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_nem_moe_plan(const int* sel_e, int rows, int tile_rows, int* counts, int* offsets, int* cursor, int* tile_expert,
                                             int* tile_row, int* num_tiles, int* pair_row, cudaStream_t st) {
    if (tile_rows <= 0) return cudaErrorInvalidValue;
    const int pairs = rows * TOPK;
    SQ_CHECK_LAUNCH(cudaMemsetAsync(counts, 0, EXPERTS * sizeof(int), st));
    if (pairs > 0) SQ_CHECK_LAUNCH(sq_launch(count_kernel, (pairs + 255) / 256, 256, 0, st, sel_e, pairs, counts));
    SQ_CHECK_LAUNCH(sq_launch(plan_kernel, 1, EXPERTS, 0, st, (const int*)counts, offsets, cursor, tile_expert, tile_row, num_tiles, tile_rows));
    if (pairs > 0) SQ_CHECK_LAUNCH(sq_launch(scatter_kernel, (pairs + 255) / 256, 256, 0, st, sel_e, pairs, cursor, pair_row));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_nem_moe_gather(const void* x, const int* pair_row, int rows, void* xs, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(gather_kernel, rows * TOPK, 128, 0, st, static_cast<const bf16*>(x), pair_row, static_cast<bf16*>(xs)));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_nem_relu2(const float* x, int ldx, int width, int rows, void* out, int ldo, cudaStream_t st) {
    const size_t count = (size_t)rows * width;
    if (count == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(relu2_kernel, (unsigned)((count + 255) / 256), 256, 0, st, x, ldx, width, rows, static_cast<bf16*>(out), ldo));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_nem_moe_combine(const float* y, const int* pair_row, const float* sel_w, int rows, void* out, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(combine_kernel, dim3(rows, LATENT / 256), 256, 0, st, y, pair_row, sel_w, static_cast<bf16*>(out)));
    SQ_RETURN_LAUNCH();
}

// Prefill tile rows of quench6k_nem_moe_experts_prefill (moe_plan's tile_rows).
extern "C" int quench6k_nem_moe_prefill_tile_rows() { return w4_variant() / 10; }

// Routed experts of a prefill step after moe_plan(tile_rows = quench6k_nem_moe_prefill_tile_rows()) and
// moe_gather: act = relu2(up) BF16 [pairs, 2688] (fused epilogue), y = bf16(down) BF16 [pairs, 1024].
extern "C" cudaError_t quench6k_nem_moe_experts_prefill(const void* xs, NvExperts up, NvExperts down, const int* offsets, const int* tile_expert,
                                                        const int* tile_row, const int* num_tiles, int max_tiles, void* act, void* y,
                                                        cudaStream_t st) {
    if (max_tiles == 0) return cudaSuccess;
    // nem.w4 = BM * 10 + stages (e.g. 1284: 128-row tiles, 4 stages).
    // nem.w4v: 3 (default) the version-3 kernel (128-row tiles), 1 the first one (nem.w4 = BM * 10 + stages).
    if (sq_variant("nem.w4v", 3) == 3) {
        if (w4_variant() / 10 != V3_BM) return cudaErrorInvalidValue;
        SQ_CHECK_LAUNCH((launch_w4v3<1>(xs, up, offsets, tile_expert, tile_row, num_tiles, max_tiles, act, MOE_INTER, LATENT, st)));
        SQ_CHECK_LAUNCH((launch_w4v3<2>(act, down, offsets, tile_expert, tile_row, num_tiles, max_tiles, y, LATENT, MOE_INTER, st)));
        SQ_RETURN_LAUNCH();
    }
    const int v = w4_variant(), bm = v / 10, stages = v % 10;
#define SQ_W4(B, S)                                                                                                                      \
    if (bm == B && stages == S) {                                                                                                        \
        SQ_CHECK_LAUNCH((launch_w4<B, S, 1>(xs, up, offsets, tile_expert, tile_row, num_tiles, max_tiles, act, MOE_INTER, LATENT, st)));  \
        SQ_CHECK_LAUNCH((launch_w4<B, S, 2>(act, down, offsets, tile_expert, tile_row, num_tiles, max_tiles, y, LATENT, MOE_INTER, st))); \
        SQ_RETURN_LAUNCH();                                                                                                              \
    }
    SQ_W4(128, 4)
    SQ_W4(128, 3)
    SQ_W4(64, 3)
    SQ_W4(64, 4)
#undef SQ_W4
    return cudaErrorInvalidValue;
}

extern "C" cudaError_t quench6k_nem_moe_combine_bf16(const void* y, const int* pair_row, const float* sel_w, int rows, void* out, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(combine_bf16_kernel, dim3(rows, LATENT / 256), 256, 0, st, static_cast<const bf16*>(y), pair_row, sel_w,
                              static_cast<bf16*>(out)));
    SQ_RETURN_LAUNCH();
}

// Tile rows of the prefill expert kernels for a step of `rows` rows (nem.thin: the most rows that take
// the thin 32-row tiles, default 1024; 0: never): 32 or quench6k_nem_moe_prefill_tile_rows().
extern "C" int quench6k_nem_moe_prefill_tile_rows_for(int rows) {
    // (The engine sizes its tile lists for thin steps of at most 1024 rows.)
    return rows <= std::min(sq_variant("nem.thin", 1024), 1024) ? TH_BM : quench6k_nem_moe_prefill_tile_rows();
}

// quench6k_nem_moe_experts_prefill after moe_plan with `tile_rows` from quench6k_nem_moe_prefill_tile_rows_for.
extern "C" cudaError_t quench6k_nem_moe_experts_prefill_tiles(const void* xs, NvExperts up, NvExperts down, const int* offsets, const int* tile_expert,
                                                              const int* tile_row, const int* num_tiles, int max_tiles, int tile_rows, void* act,
                                                              void* y, cudaStream_t st) {
    if (tile_rows != TH_BM)
        return quench6k_nem_moe_experts_prefill(xs, up, down, offsets, tile_expert, tile_row, num_tiles, max_tiles, act, y, st);
    if (max_tiles == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH((launch_w4thin<1>(xs, up, offsets, tile_expert, tile_row, num_tiles, max_tiles, act, MOE_INTER, LATENT, st)));
    SQ_CHECK_LAUNCH((launch_w4thin<2>(act, down, offsets, tile_expert, tile_row, num_tiles, max_tiles, y, LATENT, MOE_INTER, st)));
    SQ_RETURN_LAUNCH();
}
