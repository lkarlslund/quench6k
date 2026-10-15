// One cooperative kernel per HyperConnection for small batches (T <= 16), SPEC §4:
//
//   phase 1  x = h (+ y * inj_prev, the pending residual update of the previous block) per token;
//            per-split sums of squares and partial dot products of x with the norm-folded down and
//            injection rows, over stream-aligned 512-column splits
//   phase 2  per-stream RMS factors r; z = bf16(silu(down / 4)); inj = 2 sigmoid(inject / 4)
//   phase 3  u = up . z for the 4 rows of every feature d; m[d] = mean_s sigmoid(u_s) * x_s r_s (1 + g_s),
//            and the updated residual x is written back to h
//
// separated by grid-wide barriers. Values produced inside the kernel are read with ordinary loads.

#include <cooperative_groups.h>

#include "wfmt.cuh"

namespace cg = cooperative_groups;

namespace {

constexpr int D = 2560, S = 4, H = S * D, RANK = 320, SPLIT = 512, SPLITS = H / SPLIT, WARPS = 8;
constexpr float EPS = 1e-6f;

struct Args {
    float* h;
    int T;
    const float* y;         // pending update [T, D], or null
    const float* inj_prev;  // [T, 4]
    const uint8_t* wd;      // down [320, H]
    const float* sd;
    const uint8_t* wi;      // injection [4, H], or null (final mixers)
    const float* si;
    const uint8_t* wu;      // up [H, 320]
    const float* su;
    const __nv_bfloat16* g;  // HC norm weight [H]
    float* partial;          // [SPLITS, T, N]
    float* xsq;              // [SPLITS, T]
    __nv_bfloat16* z;        // [T, 320]
    float* r;                // [T, 4]
    float* inj;              // [T, 4], or null
    __nv_bfloat16* m;        // [T, D]
};

// Multiplies 16 weights loaded at column k of a row by the format's in-row scale (group scales only).
template <WFmt F>
SQ_DEVICE void group_scale(const float* scales, int row, int K, int k, float* w) {
    if constexpr (F == WFmt::INT8G) {
        const float gs = __ldg(scales + (size_t)row * (K / GROUP) + k / GROUP);
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] *= gs;
    }
}

template <WFmt F>
SQ_DEVICE float row_scale(const float* scales, int row) {
    return F == WFmt::FP8 || F == WFmt::INT8 ? __ldg(scales + row) : 1.0f;
}

template <WFmt F, int MAXT>
__global__ void __launch_bounds__(WARPS * 32) hc_small_kernel(Args a) {
    pdl_launch_dependents();
    cg::grid_group grid = cg::this_grid();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, T = a.T;
    const int N = a.wi ? RANK + S : RANK;
    constexpr int ROWS = WARPS * 2;
    const int row_blocks = (N + ROWS - 1) / ROWS;
    // The first phase-1 item's weight chunks are static: loaded raw before the PDL wait.
    Raw16<F> first[2];
    float first_gs[2] = {1.0f, 1.0f};
    if ((int)blockIdx.x < row_blocks * SPLITS) {
        const int rb = blockIdx.x / SPLITS, sp = blockIdx.x % SPLITS, k = sp * SPLIT + lane * 16;
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int row = min(rb * ROWS + warp * 2 + rr, N - 1);
            const bool down = row < RANK;
            const int wr = down ? row : row - RANK;
            first[rr] = load_raw16<F>((down ? a.wd : a.wi) + (size_t)wr * H * elem_bytes<F>(), k);
            if constexpr (F == WFmt::INT8G) first_gs[rr] = __ldg((down ? a.sd : a.si) + (size_t)wr * (H / GROUP) + k / GROUP);
        }
    }
    pdl_wait();

    // ---- phase 1: partial down/injection dot products and sums of squares
    for (int item = blockIdx.x; item < row_blocks * SPLITS; item += gridDim.x) {
        const int rb = item / SPLITS, sp = item % SPLITS, s = sp / (SPLITS / S);
        const int k = sp * SPLIT + lane * 16, d = k - s * D;
        float x[MAXT][16];
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= T) break;
            load_x8(a.h + (size_t)t * H + k, x[t]);
            load_x8(a.h + (size_t)t * H + k + 8, x[t] + 8);
            if (a.y) {
                float yv[16];
                load_x8(a.y + (size_t)t * D + d, yv);
                load_x8(a.y + (size_t)t * D + d + 8, yv + 8);
                const float iv = a.inj_prev[t * S + s];
#pragma unroll
                for (int i = 0; i < 16; ++i) x[t][i] = fmaf(yv[i], iv, x[t][i]);
            }
        }
        if (rb == 0 && warp == 0) {
#pragma unroll
            for (int t = 0; t < MAXT; ++t) {
                if (t >= T) break;
                float ss = 0;
#pragma unroll
                for (int i = 0; i < 16; ++i) ss = fmaf(x[t][i], x[t][i], ss);
                ss = warp_sum(ss);
                if (lane == 0) a.xsq[sp * T + t] = ss;
            }
        }
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int row = rb * ROWS + warp * 2 + rr;
            if (row >= N) break;
            const bool down = row < RANK;
            const uint8_t* w8 = down ? a.wd : a.wi;
            const float* sc = down ? a.sd : a.si;
            const int wr = down ? row : row - RANK;
            float w[16];
            if (item == (int)blockIdx.x) {
                decode_raw16<F>(first[rr], w);
                if constexpr (F == WFmt::INT8G)
#pragma unroll
                    for (int i = 0; i < 16; ++i) w[i] *= first_gs[rr];
            } else {
                load16<F>(w8 + (size_t)wr * H * elem_bytes<F>(), k, w);
                group_scale<F>(sc, wr, H, k, w);
            }
            const float rs = row_scale<F>(sc, wr);
#pragma unroll
            for (int t = 0; t < MAXT; ++t) {
                if (t >= T) break;
                float acc = 0;
#pragma unroll
                for (int i = 0; i < 16; ++i) acc = fmaf(w[i], x[t][i], acc);
                acc = warp_sum(acc);
                if (lane == 0) a.partial[((size_t)sp * T + t) * N + row] = acc * rs;
            }
        }
    }
    grid.sync();

    // ---- phase 2: RMS factors, low-rank activation and injection weights
    for (int item = blockIdx.x * blockDim.x + threadIdx.x; item < T * N; item += gridDim.x * blockDim.x) {
        const int t = item / N, j = item % N;
        float rs[S];
#pragma unroll
        for (int s = 0; s < S; ++s) {
            float ss = 0;
            for (int q = 0; q < SPLITS / S; ++q) ss += a.xsq[(s * (SPLITS / S) + q) * T + t];
            rs[s] = rsqrtf(ss / D + EPS);
        }
        float x = 0;
        for (int sp = 0; sp < SPLITS; ++sp) x += rs[sp / (SPLITS / S)] * a.partial[((size_t)sp * T + t) * N + j];
        x /= S;
        if (j < RANK) a.z[t * RANK + j] = f2bf(silu(x));
        else a.inj[t * S + j - RANK] = 2.0f * sigmoidf(x);
        if (j == 0)
#pragma unroll
            for (int s = 0; s < S; ++s) a.r[t * S + s] = rs[s];
    }
    grid.sync();

    // ---- phase 3: up projection of the four rows of feature d, mix, residual write-back
    for (int d = blockIdx.x * WARPS + warp; d < D; d += gridDim.x * WARPS) {
        const bool active = lane < RANK / 16;  // 20 lanes cover K = 320
        const int k = lane * 16;
        float u[S][MAXT];
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const int row = s * D + d;
            float w[16] = {};
            if (active) {
                load16<F>(a.wu + (size_t)row * RANK * elem_bytes<F>(), k, w);
                group_scale<F>(a.su, row, RANK, k, w);
            }
            const float rs = row_scale<F>(a.su, row);
#pragma unroll
            for (int t = 0; t < MAXT; ++t) {
                if (t >= T) break;
                float acc = 0;
                if (active) {
                    const uint4* zp = reinterpret_cast<const uint4*>(a.z + t * RANK + k);
                    const uint4 z0 = zp[0], z1 = zp[1];
                    const __nv_bfloat16* zb0 = reinterpret_cast<const __nv_bfloat16*>(&z0);
                    const __nv_bfloat16* zb1 = reinterpret_cast<const __nv_bfloat16*>(&z1);
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc = fmaf(w[i], bf2f(zb0[i]), acc);
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc = fmaf(w[8 + i], bf2f(zb1[i]), acc);
                }
                u[s][t] = warp_sum(acc) * rs;
            }
        }
        // Lane (s = lane % 4, token tq + lane / 4) finishes one stream of one token; 4-lane sums give m.
        const int s = lane & 3;
        for (int tq = 0; tq < T; tq += 8) {
            const int t = tq + (lane >> 2);
            float us = 0;
#pragma unroll
            for (int ss = 0; ss < S; ++ss)
#pragma unroll
                for (int tt = 0; tt < MAXT; ++tt)
                    if (ss == s && tt == t) us = u[ss][tt];
            float term = 0;
            if (t < T) {
                const size_t idx = (size_t)t * H + s * D + d;
                float x = a.h[idx];
                if (a.y) {
                    x = fmaf(a.y[(size_t)t * D + d], a.inj_prev[t * S + s], x);
                    a.h[idx] = x;
                }
                term = sigmoidf(us) * x * a.r[t * S + s] * (1.0f + bf2f(a.g[s * D + d]));
            }
            term += __shfl_xor_sync(0xffffffff, term, 1);
            term += __shfl_xor_sync(0xffffffff, term, 2);
            if (s == 0 && t < T) a.m[(size_t)t * D + d] = f2bf(term / S);
        }
    }
}

template <WFmt F, int MAXT>
cudaError_t launch(const Args& args, cudaStream_t st) {
    static int max_blocks = 0;
    if (max_blocks == 0) {
        int per_sm = 0, sms = 0, dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, hc_small_kernel<F, MAXT>, WARPS * 32, 0);
        max_blocks = std::min(per_sm, 3) * sms;
    }
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[1].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(max_blocks);
    cfg.blockDim = dim3(WARPS * 32);
    cfg.stream = st;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    return cudaLaunchKernelEx(&cfg, hc_small_kernel<F, MAXT>, args);
}

template <WFmt F>
cudaError_t dispatch_t(const Args& a, cudaStream_t st) {
    if (a.T <= 1) return launch<F, 1>(a, st);
    if (a.T <= 2) return launch<F, 2>(a, st);
    if (a.T <= 4) return launch<F, 4>(a, st);
    if (a.T <= 8) return launch<F, 8>(a, st);
    return launch<F, 16>(a, st);
}

}  // namespace

extern "C" size_t squash_hc_small_scratch_floats(int T) { return (size_t)SPLITS * T * (RANK + S) + SPLITS * T; }

// fmt: weight format of down, injection and up (all equal). scratch: squash_hc_small_scratch_floats(T)
// floats; z, r are small-batch buffers [T, 320] BF16 and [T, 4] FP32.
extern "C" cudaError_t squash_hc_small(int fmt, float* h, int T, const float* y, const float* inj_prev,
                                       const void* wd, const float* sd, const void* wi, const float* si,
                                       const void* wu, const float* su, const void* g, float* scratch, void* z,
                                       float* r, float* inj, void* m, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    if (T > 16 || (wi != nullptr) != (inj != nullptr)) return cudaErrorInvalidValue;
    Args a{h, T, y, inj_prev, static_cast<const uint8_t*>(wd), sd, static_cast<const uint8_t*>(wi), si,
           static_cast<const uint8_t*>(wu), su, static_cast<const __nv_bfloat16*>(g), scratch,
           scratch + (size_t)SPLITS * T * (RANK + S), static_cast<__nv_bfloat16*>(z), r, inj,
           static_cast<__nv_bfloat16*>(m)};
    switch (fmt) {
        case 0: return dispatch_t<WFmt::BF16>(a, st);
        case 3: return dispatch_t<WFmt::INT8G>(a, st);
        default: return cudaErrorInvalidValue;
    }
}
