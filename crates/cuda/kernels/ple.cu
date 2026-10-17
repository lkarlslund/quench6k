// PLE injection (SPEC §8.2). The hashed table rows are gathered on the host; these kernels take the
// key/value projections of the gathered embedding and update the residual.
// Per-sequence state: conv history [9, 10240] FP32 (oldest first).

#include "common.cuh"

namespace {

constexpr int D = 2560, S = 4, H = S * D, HIST = 9;
constexpr float EPS = 1e-6f;

// Block (t, s): gv[t, s, :] = sigmoid(signed_sqrt(kn . qn / sqrt(D))) * v[t, :], xn = rms(gv) * (1 + wc).
__global__ void gate_kernel(const float* __restrict__ kp, const float* __restrict__ h, const float* __restrict__ v,
                            const __nv_bfloat16* __restrict__ wk, const __nv_bfloat16* __restrict__ wq,
                            const __nv_bfloat16* __restrict__ wc, float* gv, float* xn) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const int t = blockIdx.x / S, s = blockIdx.x % S;
    const float* k = kp + (size_t)t * H + s * D;
    const float* q = h + (size_t)t * H + s * D;
    float kk = 0, qq = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        kk += k[d] * k[d];
        qq += q[d] * q[d];
    }
    const float rk = rsqrtf(block_sum(kk, scratch) / D + EPS);
    const float rq = rsqrtf(block_sum(qq, scratch) / D + EPS);
    float dot = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const int i = s * D + d;
        dot += (k[d] * rk * (1.0f + bf2f(wk[i]))) * (q[d] * rq * (1.0f + bf2f(wq[i])));
    }
    float a = block_sum(dot, scratch) * rsqrtf((float)D);
    a = copysignf(sqrtf(fmaxf(fabsf(a), 1e-6f)), a);
    const float gate = sigmoidf(a);
    const float* vt = v + (size_t)t * D;
    float vv = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) vv += vt[d] * vt[d];
    const float rv = rsqrtf(block_sum(vv, scratch) / D * gate * gate + EPS);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const size_t i = (size_t)t * H + s * D + d;
        const float g = gate * vt[d];
        gv[i] = g;
        xn[i] = g * rv * (1.0f + bf2f(wc[s * D + d]));
    }
}

constexpr int REC_TOKENS = 16;

// h[t, c] += gv[t, c] + silu(sum_j w[c, j] * xn[t - 9 + 3j, c]), with history from `state`. Speculative
// decoding (rec/ctl set) keeps the history committed one step behind like the GDN kernel: the previous
// step's accepted xn values (ctl[0] tokens, records of parity ctl[1]) are replayed first; with `defer`
// this step's values go to the other record buffer instead of the committed history.
__global__ void conv_kernel(const float* __restrict__ gv, const float* __restrict__ xn, const __nv_bfloat16* __restrict__ w,
                            float* state, float* h, int T, float* rec, const int* ctl, int defer) {
    pdl_launch_dependents();
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= H) return;
    float hist[HIST];
#pragma unroll
    for (int j = 0; j < HIST; ++j) hist[j] = state[j * H + c];
    auto shift = [&](float cur) {
#pragma unroll
        for (int j = 0; j < HIST - 1; ++j) hist[j] = hist[j + 1];
        hist[HIST - 1] = cur;
    };
    float* next = nullptr;
    if (rec) {
        const int n_commit = ctl[0], parity = ctl[1];
        const float* prev = rec + (size_t)parity * REC_TOKENS * H;
        for (int t = 0; t < n_commit; ++t) shift(prev[(size_t)t * H + c]);
        if (n_commit > 0 && defer)
#pragma unroll
            for (int j = 0; j < HIST; ++j) state[j * H + c] = hist[j];
        next = rec + (size_t)(parity ^ 1) * REC_TOKENS * H;
    }
    const float w0 = bf2f(w[c * 4]), w1 = bf2f(w[c * 4 + 1]), w2 = bf2f(w[c * 4 + 2]), w3 = bf2f(w[c * 4 + 3]);
    for (int t = 0; t < T; ++t) {
        const size_t i = (size_t)t * H + c;
        const float cur = xn[i];
        h[i] += gv[i] + silu(w0 * hist[0] + w1 * hist[3] + w2 * hist[6] + w3 * cur);
        if (defer && next) next[(size_t)t * H + c] = cur;
        shift(cur);
    }
    if (!defer)
#pragma unroll
        for (int j = 0; j < HIST; ++j) state[j * H + c] = hist[j];
}

// Prefill form of conv_kernel (committed history, no records): the filter reads only inputs, so tiles
// of PT tokens run in parallel; grid (H / 256, ceil(T / PT)). The state is advanced afterwards.
constexpr int PT = 16;
__global__ void conv_tiled_kernel(const float* __restrict__ gv, const float* __restrict__ xn, const __nv_bfloat16* __restrict__ w,
                                  const float* __restrict__ state, float* h, int T) {
    pdl_launch_dependents();
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t0 = blockIdx.y * PT;
    if (c >= H) return;
    float hist[HIST];
#pragma unroll
    for (int j = 0; j < HIST; ++j) {
        const int t = t0 - HIST + j;
        hist[j] = t >= 0 ? xn[(size_t)t * H + c] : state[(HIST + t) * H + c];
    }
    const float w0 = bf2f(w[c * 4]), w1 = bf2f(w[c * 4 + 1]), w2 = bf2f(w[c * 4 + 2]), w3 = bf2f(w[c * 4 + 3]);
    const int t1 = min(T, t0 + PT);
    for (int t = t0; t < t1; ++t) {
        const size_t i = (size_t)t * H + c;
        const float cur = xn[i];
        h[i] += gv[i] + silu(w0 * hist[0] + w1 * hist[3] + w2 * hist[6] + w3 * cur);
#pragma unroll
        for (int j = 0; j < HIST - 1; ++j) hist[j] = hist[j + 1];
        hist[HIST - 1] = cur;
    }
}

// state <- the last HIST inputs (T >= HIST).
__global__ void conv_advance_kernel(const float* __restrict__ xn, float* state, int T) {
    pdl_launch_dependents();
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= H) return;
#pragma unroll
    for (int j = 0; j < HIST; ++j) state[j * H + c] = xn[(size_t)(T - HIST + j) * H + c];
}

}  // namespace

extern "C" cudaError_t quench6k_ple_gate(const float* kp, const float* h, const float* v, const void* wk, const void* wq,
                                       const void* wc, float* gv, float* xn, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(gate_kernel, T * S, 256, 0, st, kp, h, v, static_cast<const __nv_bfloat16*>(wk),
                                       static_cast<const __nv_bfloat16*>(wq), static_cast<const __nv_bfloat16*>(wc), gv,
                                       xn));
    SQ_RETURN_LAUNCH();
}

extern "C" size_t quench6k_ple_record_floats() { return (size_t)2 * REC_TOKENS * H; }

// rec/ctl/defer as for quench6k_gdn_fused_small (records: quench6k_ple_record_floats() floats).
extern "C" cudaError_t quench6k_ple_conv(const float* gv, const float* xn, const void* w, float* state, float* h, int T,
                                       float* rec, const int* ctl, int defer, cudaStream_t st) {
    if (T > REC_TOKENS && rec && defer) return cudaErrorInvalidValue;
    if (T >= 64 && !defer) {
        // Prefill: commit any pending speculative records into the state first (a zero-token step),
        // then the token tiles in parallel and the state update.
        if (rec)
            SQ_CHECK_LAUNCH(sq_launch(conv_kernel, H / 256, 256, 0, st, gv, xn, static_cast<const __nv_bfloat16*>(w), state, h, 0, rec, ctl,
                                      0));
        SQ_CHECK_LAUNCH(sq_launch(conv_tiled_kernel, dim3(H / 256, (T + PT - 1) / PT), 256, 0, st, gv, xn,
                                  static_cast<const __nv_bfloat16*>(w), static_cast<const float*>(state), h, T));
        SQ_CHECK_LAUNCH(sq_launch(conv_advance_kernel, H / 256, 256, 0, st, xn, state, T));
        SQ_RETURN_LAUNCH();
    }
    SQ_CHECK_LAUNCH(sq_launch(conv_kernel, H / 256, 256, 0, st, gv, xn, static_cast<const __nv_bfloat16*>(w), state, h, T,
                              rec, ctl, defer));
    SQ_RETURN_LAUNCH();
}
