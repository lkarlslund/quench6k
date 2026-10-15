// Small-batch HyperConnection (T <= 16) as three PDL-chained kernels, SPEC §4. Each loads its static
// weights before the dependency wait; activations are read only after it.
//
//   hc_down    x = h (+ y * inj_prev, the previous block's pending update), per-split sums of squares
//              and partial dot products with the norm-folded down/injection rows (stream-aligned
//              512-column splits)
//   hc_finish  per-stream RMS factors r; z = bf16(silu(down / 4)); inj = 2 sigmoid(inject / 4)
//   hc_up_mix  u = up . z for the 4 rows of each feature d; m[d] = mean_s sigmoid(u_s) x_s r_s (1 + g_s);
//              the updated residual x is written back to h

#include "wfmt.cuh"

namespace {

constexpr int D = 2560, S = 4, H = S * D, RANK = 320, SPLIT = 512, SPLITS = H / SPLIT, WARPS = 8;
constexpr float EPS = 1e-6f;

template <WFmt F>
SQ_DEVICE float row_scale(const float* scales, int row) {
    return F == WFmt::FP8 || F == WFmt::INT8 ? __ldg(scales + row) : 1.0f;
}

// grid (ceil(N / 16), SPLITS); warp w owns rows 16 blockIdx.x + 2w, +1 (down rows, then injection rows).
template <WFmt F, int MAXT>
__global__ void __launch_bounds__(WARPS * 32) hc_down_kernel(const float* h, int T, const float* y, const float* inj_prev,
                                                             const uint8_t* __restrict__ wd, const float* __restrict__ sd,
                                                             const uint8_t* __restrict__ wi, const float* __restrict__ si,
                                                             int N, float* partial, float* xsq) {
    pdl_launch_dependents();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, sp = blockIdx.y, s = sp / (SPLITS / S);
    const int k = sp * SPLIT + lane * 16, d = k - s * D;
    Raw16<F> raw[2];
    float gs[2] = {1.0f, 1.0f}, rs[2] = {0.0f, 0.0f};
    int rows[2];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        rows[rr] = blockIdx.x * 16 + warp * 2 + rr;
        const int row = min(rows[rr], N - 1);
        const bool down = row < RANK;
        const int wr = down ? row : row - RANK;
        raw[rr] = load_raw16<F>((down ? wd : wi) + (size_t)wr * H * elem_bytes<F>(), k);
        if constexpr (F == WFmt::INT8G) gs[rr] = __ldg((down ? sd : si) + (size_t)wr * (H / GROUP) + k / GROUP);
        rs[rr] = row_scale<F>(down ? sd : si, wr);
    }
    float w[2][16];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        decode_raw16<F>(raw[rr], w[rr]);
#pragma unroll
        for (int i = 0; i < 16; ++i) w[rr][i] *= gs[rr];
    }
    pdl_wait();
    // Tokens in groups of up to four; the group's 2 x TG dot products are reduced together (slot
    // 2 t + rr), each bitwise as a warp_sum of its own.
    constexpr int TG = MAXT < 4 ? MAXT : 4, NS = 2 * TG;
#pragma unroll
    for (int t0 = 0; t0 < MAXT; t0 += TG) {
        if (t0 >= T) break;
        float acc[NS] = {};
#pragma unroll
        for (int j = 0; j < TG; ++j) {
            const int t = t0 + j;
            if (t >= T) break;
            float x[16];
            load_x8(h + (size_t)t * H + k, x);
            load_x8(h + (size_t)t * H + k + 8, x + 8);
            if (y) {
                float yv[16];
                load_x8(y + (size_t)t * D + d, yv);
                load_x8(y + (size_t)t * D + d + 8, yv + 8);
                const float iv = inj_prev[t * S + s];
#pragma unroll
                for (int i = 0; i < 16; ++i) x[i] = fmaf(yv[i], iv, x[i]);
            }
            if (blockIdx.x == 0 && warp == 0) {
                float ss = 0;
#pragma unroll
                for (int i = 0; i < 16; ++i) ss = fmaf(x[i], x[i], ss);
                ss = warp_sum(ss);
                if (lane == 0) xsq[sp * T + t] = ss;
            }
#pragma unroll
            for (int rr = 0; rr < 2; ++rr)
#pragma unroll
                for (int i = 0; i < 16; ++i) acc[2 * j + rr] = fmaf(w[rr][i], x[i], acc[2 * j + rr]);
        }
        const float v = warp_sum_slots(acc);
        const int slot = lane / (32 / NS), t = t0 + slot / 2, rr = slot % 2;
        const int row = rr ? rows[1] : rows[0];
        if (lane % (32 / NS) == 0 && t < T && row < N) partial[((size_t)sp * T + t) * N + row] = v * (rr ? rs[1] : rs[0]);
    }
}

// One thread per (t, j < N).
__global__ void hc_finish_kernel(const float* partial, const float* xsq, int T, int N, __nv_bfloat16* z, float* inj,
                                 float* r) {
    pdl_launch_dependents();
    pdl_wait();
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= T * N) return;
    const int t = item / N, j = item % N;
    float rs[S];
#pragma unroll
    for (int s = 0; s < S; ++s) {
        float ss = 0;
#pragma unroll
        for (int q = 0; q < SPLITS / S; ++q) ss += xsq[(s * (SPLITS / S) + q) * T + t];
        rs[s] = rsqrtf(ss / D + EPS);
    }
    float x = 0;
#pragma unroll
    for (int sp = 0; sp < SPLITS; ++sp) x += rs[sp / (SPLITS / S)] * partial[((size_t)sp * T + t) * N + j];
    x /= S;
    if (j < RANK) z[t * RANK + j] = f2bf(silu(x));
    else inj[t * S + j - RANK] = 2.0f * sigmoidf(x);
    if (j == 0)
#pragma unroll
        for (int s = 0; s < S; ++s) r[t * S + s] = rs[s];
}

// grid (D / 8): warp w owns feature d = 8 blockIdx.x + w, i.e. up rows s * D + d; lanes 0..19 cover K = 320.
template <WFmt F, int MAXT>
__global__ void __launch_bounds__(WARPS * 32) hc_up_mix_kernel(float* h, int T, const float* y, const float* inj_prev,
                                                               const uint8_t* __restrict__ wu, const float* __restrict__ su,
                                                               const __nv_bfloat16* __restrict__ g,
                                                               const __nv_bfloat16* z, const float* r,
                                                               __nv_bfloat16* m) {
    pdl_launch_dependents();
    const int lane = threadIdx.x & 31, d = blockIdx.x * WARPS + (threadIdx.x >> 5);
    const bool active = lane < RANK / 16;
    const int k = lane * 16;
    Raw16<F> raw[S];
    float gs[S], rs[S];
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const int row = s * D + d;
        gs[s] = 1.0f;
        if (active) {
            raw[s] = load_raw16<F>(wu + (size_t)row * RANK * elem_bytes<F>(), k);
            if constexpr (F == WFmt::INT8G) gs[s] = __ldg(su + (size_t)row * (RANK / GROUP) + k / GROUP);
        }
        rs[s] = row_scale<F>(su, row);
    }
    // Tokens in groups of up to four: slot 4 t + s of the group (u for stream s of token t) is reduced
    // over the warp together with the others, landing in lanes lane / (32 / NS) == slot.
    constexpr int TG = MAXT < 4 ? MAXT : 4, NS = S * TG, LPS = 32 / NS;
    const int slot = lane / LPS, ls = slot % S, lt = slot / S;
    const float gw = 1.0f + bf2f(g[ls * D + d]);
    // Weights decoded (group scale applied) once, before the wait.
    float w[S][16];
#pragma unroll
    for (int s = 0; s < S; ++s) {
        decode_raw16<F>(raw[s], w[s]);
#pragma unroll
        for (int i = 0; i < 16; ++i) w[s][i] *= gs[s];
    }
    pdl_wait();

#pragma unroll
    for (int t0 = 0; t0 < MAXT; t0 += TG) {
        if (t0 >= T) break;
        float acc[NS] = {};
#pragma unroll
        for (int j = 0; j < TG; ++j) {
            if (t0 + j >= T) break;
            if (active) {
                float zf[16];
                load_x8(z + (t0 + j) * RANK + k, zf);
                load_x8(z + (t0 + j) * RANK + k + 8, zf + 8);
#pragma unroll
                for (int s = 0; s < S; ++s)
#pragma unroll
                    for (int i = 0; i < 16; ++i) acc[S * j + s] = fmaf(w[s][i], zf[i], acc[S * j + s]);
            }
        }
        float rsl = rs[0];
#pragma unroll
        for (int s = 1; s < S; ++s)
            if (ls == s) rsl = rs[s];
        const float us = warp_sum_slots(acc) * rsl;
        // Lane group (stream ls, token t) finishes one stream; sums over the 4 streams give m.
        const int t = t0 + lt;
        float term = 0;
        if (t < T) {
            const size_t idx = (size_t)t * H + ls * D + d;
            float x = h[idx];
            if (y) {
                x = fmaf(y[(size_t)t * D + d], inj_prev[t * S + ls], x);
                if (lane % LPS == 0) h[idx] = x;
            }
            term = sigmoidf(us) * x * r[t * S + ls] * gw;
        }
        term += __shfl_xor_sync(0xffffffff, term, LPS);
        term += __shfl_xor_sync(0xffffffff, term, 2 * LPS);
        if (ls == 0 && lane % LPS == 0 && t < T) m[(size_t)t * D + d] = f2bf(term / S);
    }
}

template <WFmt F, int MAXT>
cudaError_t launch(float* h, int T, const float* y, const float* inj_prev, const uint8_t* wd, const float* sd,
                   const uint8_t* wi, const float* si, const uint8_t* wu, const float* su, const __nv_bfloat16* g,
                   float* scratch, __nv_bfloat16* z, float* r, float* inj, __nv_bfloat16* m, cudaStream_t st) {
    const int N = wi ? RANK + S : RANK;
    float* partial = scratch;
    float* xsq = scratch + (size_t)SPLITS * T * (RANK + S);
    SQ_CHECK_LAUNCH(sq_launch(hc_down_kernel<F, MAXT>, dim3((N + 15) / 16, SPLITS), WARPS * 32, 0, st, h, T, y, inj_prev,
                              wd, sd, wi, si, N, partial, xsq));
    SQ_CHECK_LAUNCH(sq_launch(hc_finish_kernel, (T * N + 255) / 256, 256, 0, st, partial, xsq, T, N, z, inj, r));
    SQ_CHECK_LAUNCH(sq_launch(hc_up_mix_kernel<F, MAXT>, D / WARPS, WARPS * 32, 0, st, h, T, y, inj_prev, wu, su, g, z, r, m));
    return cudaGetLastError();
}

template <WFmt F>
cudaError_t dispatch_t(float* h, int T, const float* y, const float* inj_prev, const uint8_t* wd, const float* sd,
                       const uint8_t* wi, const float* si, const uint8_t* wu, const float* su, const __nv_bfloat16* g,
                       float* scratch, __nv_bfloat16* z, float* r, float* inj, __nv_bfloat16* m, cudaStream_t st) {
#define SQ_ARGS h, T, y, inj_prev, wd, sd, wi, si, wu, su, g, scratch, z, r, inj, m, st
    if (T <= 1) return launch<F, 1>(SQ_ARGS);
    if (T <= 2) return launch<F, 2>(SQ_ARGS);
    if (T <= 4) return launch<F, 4>(SQ_ARGS);
    if (T <= 8) return launch<F, 8>(SQ_ARGS);
    return launch<F, 16>(SQ_ARGS);
#undef SQ_ARGS
}

}  // namespace

extern "C" size_t squash_hc_small_scratch_floats(int T) { return (size_t)SPLITS * T * (RANK + S) + SPLITS * T; }

// fmt: weight format of down, injection and up (0 BF16 or 3 INT8 with a scale per 64 weights).
// scratch: squash_hc_small_scratch_floats(T) floats; z [T, 320] BF16 and r [T, 4] are small-batch buffers.
extern "C" cudaError_t squash_hc_small(int fmt, float* h, int T, const float* y, const float* inj_prev, const void* wd,
                                       const float* sd, const void* wi, const float* si, const void* wu, const float* su,
                                       const void* g, float* scratch, void* z, float* r, float* inj, void* m,
                                       cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    if (T > 16 || (wi != nullptr) != (inj != nullptr)) return cudaErrorInvalidValue;
    const auto* d8 = static_cast<const uint8_t*>(wd);
    const auto* i8 = static_cast<const uint8_t*>(wi);
    const auto* u8 = static_cast<const uint8_t*>(wu);
    const auto* gb = static_cast<const __nv_bfloat16*>(g);
    auto* zb = static_cast<__nv_bfloat16*>(z);
    auto* mb = static_cast<__nv_bfloat16*>(m);
    if (fmt == 0) return dispatch_t<WFmt::BF16>(h, T, y, inj_prev, d8, sd, i8, si, u8, su, gb, scratch, zb, r, inj, mb, st);
    if (fmt == 3) return dispatch_t<WFmt::INT8G>(h, T, y, inj_prev, d8, sd, i8, si, u8, su, gb, scratch, zb, r, inj, mb, st);
    return cudaErrorInvalidValue;
}
