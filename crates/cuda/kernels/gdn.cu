// Gated DeltaNet (SPEC §5): causal conv, gating, the delta-rule recurrence and the gated norm.
//
// Per-sequence state: conv [3, 10240] FP32 (oldest first) and recurrent [48, 128(k), 128(v)] FP32.

#include "common.cuh"

namespace {

constexpr int C = 10240, KH = 16, VH = 48, DK = 128, DV = 128, TAPS = 4;
constexpr float EPS = 1e-6f;

// out[t, c] = silu(sum_j w[c, j] * x[t + j - 3, c]); history from `state`, which is then advanced.
__global__ void conv_kernel(const float* __restrict__ x, int ldx, float* out, float* state,
                            const __nv_bfloat16* __restrict__ w, int T) {
    pdl_launch_dependents();
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float w0 = bf2f(w[c * TAPS]), w1 = bf2f(w[c * TAPS + 1]), w2 = bf2f(w[c * TAPS + 2]), w3 = bf2f(w[c * TAPS + 3]);
    float p0 = state[c], p1 = state[C + c], p2 = state[2 * C + c];  // x[t-3], x[t-2], x[t-1]
    for (int t = 0; t < T; ++t) {
        const float cur = x[(size_t)t * ldx + c];
        out[(size_t)t * C + c] = silu(w0 * p0 + w1 * p1 + w2 * p2 + w3 * cur);
        p0 = p1;
        p1 = p2;
        p2 = cur;
    }
    state[c] = p0;
    state[C + c] = p1;
    state[2 * C + c] = p2;
}

// In place on the conv output: l2-normalize q and k per key head (q also / sqrt(128)); and
// beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias). One block of 128 threads per (t, key head).
__global__ void prep_kernel(float* qkv, const float* __restrict__ a, const float* __restrict__ b, int ldab,
                            const __nv_bfloat16* __restrict__ a_log, const __nv_bfloat16* __restrict__ dt_bias,
                            float* beta, float* g) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const int t = blockIdx.x / KH, h = blockIdx.x % KH, i = threadIdx.x;
    float* q = qkv + (size_t)t * C + h * DK;
    float* k = qkv + (size_t)t * C + KH * DK + h * DK;
    const float qv = q[i], kv = k[i];
    const float qn = rsqrtf(block_sum(qv * qv, scratch) + EPS);
    const float kn = rsqrtf(block_sum(kv * kv, scratch) + EPS);
    q[i] = qv * qn * rsqrtf((float)DK);
    k[i] = kv * kn;
    if (i < 3) {
        const int vh = h * 3 + i;
        const float x = a[(size_t)t * ldab + vh] + bf2f(dt_bias[vh]);
        const float softplus = x > 20.0f ? x : log1pf(__expf(x));
        g[t * VH + vh] = -__expf(bf2f(a_log[vh])) * softplus;
        beta[t * VH + vh] = sigmoidf(b[(size_t)t * ldab + vh]);
    }
}

// Token-sequential delta rule. Each thread owns one value column of one head's state in registers;
// grid (48 heads, 4 column groups) x 32 threads.
__global__ void __launch_bounds__(32) recurrent_kernel(const float* __restrict__ qkv, const float* __restrict__ beta,
                                                       const float* __restrict__ g, float* state, float* out, int T) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float sq[DK], sk[DK];
    const int vh = blockIdx.x, col = blockIdx.y * 32 + threadIdx.x, kh = vh / 3;
    float s[DK];
    float* st = state + (size_t)vh * DK * DV;
#pragma unroll
    for (int j = 0; j < DK; ++j) s[j] = st[j * DV + col];
    for (int t = 0; t < T; ++t) {
        const float* row = qkv + (size_t)t * C;
        __syncwarp();
#pragma unroll
        for (int j = threadIdx.x; j < DK; j += 32) {
            sq[j] = row[kh * DK + j];
            sk[j] = row[KH * DK + kh * DK + j];
        }
        __syncwarp();
        const float decay = __expf(g[t * VH + vh]);
        float kvm = 0;
#pragma unroll
        for (int j = 0; j < DK; ++j) {
            s[j] *= decay;
            kvm += s[j] * sk[j];
        }
        const float v = row[2 * KH * DK + vh * DV + col];
        const float delta = beta[t * VH + vh] * (v - kvm);
        float o = 0;
#pragma unroll
        for (int j = 0; j < DK; ++j) {
            s[j] += sk[j] * delta;
            o += s[j] * sq[j];
        }
        out[(size_t)t * VH * DV + vh * DV + col] = o;
    }
#pragma unroll
    for (int j = 0; j < DK; ++j) st[j * DV + col] = s[j];
}

// y[t, h, :] = bf16(rms(o[t, h, :]) * w * sigmoid(z[t, h, :])); one block of 128 per (t, head).
__global__ void gated_norm_kernel(const float* __restrict__ o, const float* __restrict__ z, int ldz,
                                  const __nv_bfloat16* __restrict__ w, __nv_bfloat16* y) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const size_t base = (size_t)blockIdx.x * DV;
    const int t = blockIdx.x / VH, h = blockIdx.x % VH, i = threadIdx.x;
    const float x = o[base + i];
    const float r = rsqrtf(block_sum(x * x, scratch) / DV + EPS);
    y[base + i] = f2bf(x * r * bf2f(w[i]) * sigmoidf(z[(size_t)t * ldz + h * DV + i]));
}

}  // namespace

extern "C" cudaError_t squash_gdn_conv(const float* x, int ldx, float* out, float* state, const void* w, int T,
                                       cudaStream_t st) {
    SQ_CHECK_LAUNCH(sq_launch(conv_kernel, C / 256, 256, 0, st, x, ldx, out, state, static_cast<const __nv_bfloat16*>(w), T));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_gdn_prep(float* qkv, const float* a, const float* b, int ldab, const void* a_log,
                                       const void* dt_bias, float* beta, float* g, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(prep_kernel, T * KH, DK, 0, st, qkv, a, b, ldab, static_cast<const __nv_bfloat16*>(a_log),
                                       static_cast<const __nv_bfloat16*>(dt_bias), beta, g));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_gdn_recurrent(const float* qkv, const float* beta, const float* g, float* state, float* out,
                                            int T, cudaStream_t st) {
    SQ_CHECK_LAUNCH(sq_launch(recurrent_kernel, dim3(VH, DV / 32), 32, 0, st, qkv, beta, g, state, out, T));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_gdn_gated_norm(const float* o, const float* z, int ldz, const void* w, void* y, int T,
                                             cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(gated_norm_kernel, T * VH, DV, 0, st, o, z, ldz, static_cast<const __nv_bfloat16*>(w), static_cast<__nv_bfloat16*>(y)));
    SQ_RETURN_LAUNCH();
}

namespace {

constexpr int FT = 512;  // threads of the fused kernel: 4 per value column

// Fused small-batch GDN (SPEC §5) for one value head per block, tokens processed in order:
// causal conv + SiLU of the head's q, k (shared key head) and v channels, l2-norm, gating, the
// delta-rule update and the gated RMSNorm, writing y[t, vh, :] (BF16). Thread (col, part) keeps
// rows part*32 .. part*32+31 of state column col in registers. Conv history for q/k channels is
// advanced only by the first of the three value heads sharing them.
// Speculative records: per token, the raw (pre-conv) q/k/v channels and the a, b gate inputs.
constexpr int REC = C + 2 * VH, REC_TOKENS = 16;

__global__ void __launch_bounds__(FT) fused_small_kernel(const float* __restrict__ proj, int ld, int T,
                                                         const __nv_bfloat16* __restrict__ conv_w,
                                                         float* conv_state, float* state,
                                                         const __nv_bfloat16* __restrict__ a_log,
                                                         const __nv_bfloat16* __restrict__ dt_bias,
                                                         const __nv_bfloat16* __restrict__ norm_w, int z_col, int a_col,
                                                         int b_col, __nv_bfloat16* y, float* rec, const int* ctl,
                                                         int defer) {
    pdl_launch_dependents();
    __shared__ float sq[DK], sk[DK], sv[DV], so[DV];
    __shared__ float scratch[32];
    __shared__ float s_beta, s_decay;
    const int vh = blockIdx.x, kh = vh / 3, tid = threadIdx.x;
    const int col = tid >> 2, part = tid & 3;
    // Channels handled by this block: q[kh], k[kh], v[vh] (384); threads 0..383 own one each.
    const int ch = tid < DK ? kh * DK + tid : tid < 2 * DK ? KH * DK + kh * DK + (tid - DK) : 2 * KH * DK + vh * DV + (tid - 2 * DK);
    const bool owns_channel = tid < 3 * DK;
    const bool writes_conv = owns_channel && (tid >= 2 * DK || vh % 3 == 0);
    float hist[3] = {}, cw[4] = {};
    if (owns_channel) {
#pragma unroll
        for (int j = 0; j < 3; ++j) hist[j] = conv_state[j * C + ch];
#pragma unroll
        for (int j = 0; j < 4; ++j) cw[j] = bf2f(conv_w[ch * TAPS + j]);
    }
    float s[32];
    float* st = state + (size_t)vh * DK * DV;
#pragma unroll
    for (int j = 0; j < 32; ++j) s[j] = st[(part * 32 + j) * DV + col];
    const float A = -__expf(bf2f(a_log[vh])), bias = bf2f(dt_bias[vh]);
    // State, conv history and weights come from earlier steps or are static: loaded before the wait.
    pdl_wait();

    // Advances conv history and state by one token whose raw inputs are in `row` (a, b at a_off, b_off);
    // with `out`, also writes the gated-norm output for row `t`.
    auto advance = [&](const float* row, int a_off, int b_off, bool out, int t) {
        if (owns_channel) {
            const float x = row[ch];
            const float c = silu(cw[0] * hist[0] + cw[1] * hist[1] + cw[2] * hist[2] + cw[3] * x);
            hist[0] = hist[1];
            hist[1] = hist[2];
            hist[2] = x;
            if (tid < DK) sq[tid] = c;
            else if (tid < 2 * DK) sk[tid - DK] = c;
            else sv[tid - 2 * DK] = c;
        }
        if (tid == 0) {
            const float xa = row[a_off + vh] + bias;
            s_decay = __expf(A * (xa > 20.0f ? xa : log1pf(__expf(xa))));
            s_beta = sigmoidf(row[b_off + vh]);
        }
        __syncthreads();
        // l2-normalize q (also / sqrt(128)) and k
        const float qv = tid < DK ? sq[tid] : 0.0f, kv = tid < DK ? sk[tid] : 0.0f;
        const float qn = rsqrtf(block_sum(qv * qv, scratch) + EPS);
        const float kn = rsqrtf(block_sum(kv * kv, scratch) + EPS);
        __syncthreads();
        if (tid < DK) {
            sq[tid] = qv * qn * rsqrtf((float)DK);
            sk[tid] = kv * kn;
        }
        __syncthreads();
        const float decay = s_decay, beta = s_beta;
        float kvm = 0;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            s[j] *= decay;
            kvm += s[j] * sk[part * 32 + j];
        }
        kvm += __shfl_xor_sync(0xffffffff, kvm, 1);
        kvm += __shfl_xor_sync(0xffffffff, kvm, 2);
        const float delta = beta * (sv[col] - kvm);
        float o = 0;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            s[j] += sk[part * 32 + j] * delta;
            if (out) o += s[j] * sq[part * 32 + j];
        }
        if (out) {
            o += __shfl_xor_sync(0xffffffff, o, 1);
            o += __shfl_xor_sync(0xffffffff, o, 2);
            if (part == 0) so[col] = o;
            __syncthreads();
            // gated RMSNorm over the head's 128 outputs
            const float ov = tid < DV ? so[tid] : 0.0f;
            const float r = rsqrtf(block_sum(ov * ov, scratch) / DV + EPS);
            if (tid < DV)
                y[(size_t)t * VH * DV + vh * DV + tid] = f2bf(ov * r * bf2f(norm_w[tid]) * sigmoidf(row[z_col + vh * DV + tid]));
        }
        __syncthreads();
    };
    auto commit = [&]() {
        if (writes_conv)
#pragma unroll
            for (int j = 0; j < 3; ++j) conv_state[j * C + ch] = hist[j];
#pragma unroll
        for (int j = 0; j < 32; ++j) st[(part * 32 + j) * DV + col] = s[j];
    };

    // Speculative mode: first replay the accepted tokens of the previous step from its records
    // (parity ctl[1]) onto the committed state, and commit that.
    if (rec) {
        const int n_commit = ctl[0], parity = ctl[1];
        const float* prev = rec + (size_t)parity * REC_TOKENS * REC;
        for (int t = 0; t < n_commit; ++t) advance(prev + (size_t)t * REC, C, C + VH, false, 0);
        if (n_commit > 0 && defer) commit();
    }
    float* next = rec ? rec + (size_t)(rec ? (ctl[1] ^ 1) : 0) * REC_TOKENS * REC : nullptr;
    for (int t = 0; t < T; ++t) {
        const float* row = proj + (size_t)t * ld;
        if (defer && next) {
            if (writes_conv) next[(size_t)t * REC + ch] = row[ch];
            if (tid == 0) {
                next[(size_t)t * REC + C + vh] = row[a_col + vh];
                next[(size_t)t * REC + C + VH + vh] = row[b_col + vh];
            }
        }
        advance(row, a_col, b_col, true, t);
    }
    // Plain steps commit everything; deferred steps leave this step's tokens to the next replay.
    if (!defer) commit();
}

}  // namespace

// proj: the fused GDN projection rows [T, ld] (qkv at column 0, z/a/b at the given columns).
extern "C" size_t squash_gdn_record_floats() { return (size_t)2 * REC_TOKENS * REC; }

// rec/ctl (optional, speculative decoding): double-buffered per-token records
// (squash_gdn_record_floats() floats) and the control word {tokens to replay, record parity}.
// defer = 1: replay and commit the previous step's accepted tokens, process this step without
// committing it, and record its tokens; defer = 0: replay, then commit everything.
extern "C" cudaError_t squash_gdn_fused_small(const float* proj, int ld, int T, const void* conv_w, float* conv_state,
                                             float* state, const void* a_log, const void* dt_bias, const void* norm_w,
                                             int z_col, int a_col, int b_col, void* y, float* rec, const int* ctl,
                                             int defer, cudaStream_t st) {
    if (T > REC_TOKENS && rec && defer) return cudaErrorInvalidValue;
    if (T == 0 && !rec) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(fused_small_kernel, VH, FT, 0, st, proj, ld, T, static_cast<const __nv_bfloat16*>(conv_w), conv_state, state,
                                          static_cast<const __nv_bfloat16*>(a_log),
                                          static_cast<const __nv_bfloat16*>(dt_bias),
                                          static_cast<const __nv_bfloat16*>(norm_w), z_col, a_col, b_col,
                                          static_cast<__nv_bfloat16*>(y), rec, ctl, defer));
    SQ_RETURN_LAUNCH();
}
