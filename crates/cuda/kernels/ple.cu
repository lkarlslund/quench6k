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

// h[t, c] += gv[t, c] + silu(sum_j w[c, j] * xn[t - 9 + 3j, c]), with history from `state` (then advanced).
__global__ void conv_kernel(const float* __restrict__ gv, const float* __restrict__ xn, const __nv_bfloat16* __restrict__ w,
                            float* state, float* h, int T) {
    pdl_launch_dependents();
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= H) return;
    float hist[HIST];
#pragma unroll
    for (int j = 0; j < HIST; ++j) hist[j] = state[j * H + c];
    const float w0 = bf2f(w[c * 4]), w1 = bf2f(w[c * 4 + 1]), w2 = bf2f(w[c * 4 + 2]), w3 = bf2f(w[c * 4 + 3]);
    for (int t = 0; t < T; ++t) {
        const size_t i = (size_t)t * H + c;
        const float cur = xn[i];
        h[i] += gv[i] + silu(w0 * hist[0] + w1 * hist[3] + w2 * hist[6] + w3 * cur);
#pragma unroll
        for (int j = 0; j < HIST - 1; ++j) hist[j] = hist[j + 1];
        hist[HIST - 1] = cur;
    }
#pragma unroll
    for (int j = 0; j < HIST; ++j) state[j * H + c] = hist[j];
}

}  // namespace

extern "C" cudaError_t squash_ple_gate(const float* kp, const float* h, const float* v, const void* wk, const void* wq,
                                       const void* wc, float* gv, float* xn, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(gate_kernel, T * S, 256, 0, st, kp, h, v, static_cast<const __nv_bfloat16*>(wk),
                                       static_cast<const __nv_bfloat16*>(wq), static_cast<const __nv_bfloat16*>(wc), gv,
                                       xn));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_ple_conv(const float* gv, const float* xn, const void* w, float* state, float* h, int T,
                                       cudaStream_t st) {
    SQ_CHECK_LAUNCH(sq_launch(conv_kernel, H / 256, 256, 0, st, gv, xn, static_cast<const __nv_bfloat16*>(w), state, h, T));
    SQ_RETURN_LAUNCH();
}
