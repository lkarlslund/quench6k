// Nemotron-H latent MoE: sigmoid router with correction bias (top 22 of 512), NVFP4 routed experts without
// a gate (down(relu(up(x))²)) in the 1024-wide latent space, the relu² of BF16 Linear outputs and the
// weighted combine (see crates/cuda/src/nemotron.rs).
//
// Small steps: GEMV-style; pairs grouped by expert, so an expert's weights are read once per step.
// Prefill: pairs grouped by expert into tiles for the shared W4A16 grouped GEMM
// (moe.cu quench6k_moe_nvfp4_gemm). Each pair's result depends only on its own row and expert.

#include <algorithm>
#include <cuda_fp4.h>

#include "nemotron.cuh"

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
    pdl_launch_dependents();
    pdl_wait();
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
    // The order and the routing come before the up kernel: weights load while it finishes.
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
