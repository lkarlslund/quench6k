// Gated DeltaNet (SPEC §5): causal conv, gating, the delta-rule recurrence and the gated norm.
//
// Per-sequence state: conv [3, 10240] FP32 (oldest first) followed by squash_gdn_conv_tickets() zeroed
// u32 for the small-batch kernel, and recurrent [48, 128(k), 128(v)] FP32.

#include "common.cuh"

namespace {

constexpr int C = 10240, KH = 16, VH = 48, DK = 128, DV = 128, TAPS = 4;
constexpr float EPS = 1e-6f;

// out[t, c] = silu(sum_j w[c, j] * x[t + j - 3, c]); history from `state`. A filter over the inputs
// only, so every (t, c) is independent: grid (C / 256, ceil(T / CONV_TT)), each thread CONV_TT tokens
// of one channel. advance_conv_kernel then moves the state to the last three inputs.
constexpr int CONV_TT = 16;
__global__ void conv_kernel(const float* __restrict__ x, int ldx, float* out, const float* __restrict__ state,
                            const __nv_bfloat16* __restrict__ w, int T) {
    pdl_launch_dependents();
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t0 = blockIdx.y * CONV_TT;
    if (c >= C) return;
    const float w0 = bf2f(w[c * TAPS]), w1 = bf2f(w[c * TAPS + 1]), w2 = bf2f(w[c * TAPS + 2]), w3 = bf2f(w[c * TAPS + 3]);
    auto input = [&](int t) { return t >= 0 ? x[(size_t)t * ldx + c] : state[(3 + t) * C + c]; };
    float p0 = input(t0 - 3), p1 = input(t0 - 2), p2 = input(t0 - 1);
    const int t1 = min(T, t0 + CONV_TT);
    for (int t = t0; t < t1; ++t) {
        const float cur = x[(size_t)t * ldx + c];
        out[(size_t)t * C + c] = silu(w0 * p0 + w1 * p1 + w2 * p2 + w3 * cur);
        p0 = p1;
        p1 = p2;
        p2 = cur;
    }
}

// state <- the last three inputs (from x, or the old state when T < 3). Runs after conv_kernel.
__global__ void advance_conv_kernel(const float* __restrict__ x, int ldx, float* state, int T) {
    pdl_launch_dependents();
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int t = T - 3 + j;
        h[j] = t >= 0 ? x[(size_t)t * ldx + c] : state[(3 + t) * C + c];
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) state[j * C + c] = h[j];
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

// conv_kernel then prep_kernel in one pass (prefill): one block of 128 threads per (token, key head)
// computes the causal conv of that head's q and k channels and of its three value heads' channels
// (thread i: channel i of each), l2-normalizes q and k, and computes beta and g of the three value
// heads; the same arithmetic as the two kernels. The conv state is advanced afterwards.
__global__ void __launch_bounds__(DK) conv_prep_kernel(const float* __restrict__ x, int ldx, float* out, const float* __restrict__ state,
                                                       const __nv_bfloat16* __restrict__ w, const float* __restrict__ a,
                                                       const float* __restrict__ b, int ldab, const __nv_bfloat16* __restrict__ a_log,
                                                       const __nv_bfloat16* __restrict__ dt_bias, float* beta, float* g) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const int t = blockIdx.x / KH, h = blockIdx.x % KH, i = threadIdx.x;
    auto conv = [&](int c) {
        const float w0 = bf2f(w[c * TAPS]), w1 = bf2f(w[c * TAPS + 1]), w2 = bf2f(w[c * TAPS + 2]), w3 = bf2f(w[c * TAPS + 3]);
        auto input = [&](int tt) { return tt >= 0 ? x[(size_t)tt * ldx + c] : state[(3 + tt) * C + c]; };
        const float p0 = input(t - 3), p1 = input(t - 2), p2 = input(t - 1), cur = x[(size_t)t * ldx + c];
        return silu(w0 * p0 + w1 * p1 + w2 * p2 + w3 * cur);
    };
    const int cq = h * DK + i, ck = KH * DK + h * DK + i;
    const float qv = conv(cq), kv = conv(ck);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int cv = 2 * KH * DK + (h * 3 + j) * DV + i;
        out[(size_t)t * C + cv] = conv(cv);
    }
    const float qn = rsqrtf(block_sum(qv * qv, scratch) + EPS);
    const float kn = rsqrtf(block_sum(kv * kv, scratch) + EPS);
    out[(size_t)t * C + cq] = qv * qn * rsqrtf((float)DK);
    out[(size_t)t * C + ck] = kv * kn;
    if (i < 3) {
        const int vh = h * 3 + i;
        const float xa = a[(size_t)t * ldab + vh] + bf2f(dt_bias[vh]);
        const float softplus = xa > 20.0f ? xa : log1pf(__expf(xa));
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

// Chunked delta rule (WY form), mathematically equal to recurrent_kernel. Per chunk of L tokens with
// chunk-initial state S0 [k][v] and cumulative log decays G_i (sum of g over the chunk up to i):
//   A[i][j] = beta_i exp(G_i - G_j) (k_i . k_j) for j < i;  T = (I + A)^-1 (unit lower triangular)
//   W = T (beta exp(G) * K),  U = T (beta * V),  Vn = U - W S0          (the per-token updates u_i)
//   O[i] = exp(G_i) S0^T q_i + sum_{j <= i} exp(G_i - G_j) (q_i . k_j) Vn[j]
//   S = exp(G_L) S0 + sum_j exp(G_L - G_j) k_j Vn[j]^T
// chunk_prep_kernel computes the state-independent W, U and P[i][j] = exp(G_i - G_j) (q_i . k_j)
// for every (chunk, head) in parallel; chunk_state_kernel walks the chunks of each (head, column
// group) with the state. FP32 throughout.
constexpr int CL = 32, CW = 32, CT = 256, KP = DK + 1;  // padded rows: conflict-free column reads
constexpr int NV = CL * CW / CT, NS = DK * CW / CT;  // per-thread outputs: Vn / O, state
constexpr int CHUNK_W = CL * DK, CHUNK_U = CL * DV, CHUNK_P = CL * CL, CHUNK_FLOATS = CHUNK_W + CHUNK_U + CHUNK_P;
constexpr size_t PREP_SMEM = (2 * CL * KP + CL * DV + 3 * CL * (CL + 1)) * sizeof(float);
constexpr size_t STATE_SMEM = (DK * CW + 3 * CL * KP + 3 * CL * CW + CL * CL) * sizeof(float);

// Inclusive prefix sum of g over the chunk (warp 0 lanes = rows) and the derived factors.
SQ_DEVICE void chunk_decays(const float* __restrict__ g, const float* __restrict__ beta, int t0, int len, int vh,
                            float* Gs, float* Bs) {
    const int r = threadIdx.x & 31;
    float gv = r < len ? g[(t0 + r) * VH + vh] : 0.0f;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float n = __shfl_up_sync(0xffffffff, gv, o);
        if (r >= o) gv += n;
    }
    Gs[r] = gv;
    if (Bs) Bs[r] = r < len ? beta[(t0 + r) * VH + vh] : 0.0f;
}

// grid (chunks, VH), CT threads.
__global__ void __launch_bounds__(CT) chunk_prep_kernel(const float* __restrict__ qkv, const float* __restrict__ beta,
                                                        const float* __restrict__ g, float* scratch, int T) {
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ float sm[];
    float* Ks = sm;                  // [CL][KP]
    float* Qs = Ks + CL * KP;        // [CL][KP]
    float* Vs = Qs + CL * KP;        // [CL][DV]
    float* Tm = Vs + CL * DV;        // [CL][CL + 1]: A, then T
    float* TB = Tm + CL * (CL + 1);  // T[i][j] beta_j exp(G_j)
    float* TV = TB + CL * (CL + 1);  // T[i][j] beta_j
    __shared__ float Gs[CL], Bs[CL];
    const int ch = blockIdx.x, vh = blockIdx.y, kh = vh / 3, t0 = ch * CL, len = min(CL, T - t0);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float* outw = scratch + ((size_t)ch * VH + vh) * CHUNK_FLOATS;
    float* outu = outw + CHUNK_W;
    float* outp = outu + CHUNK_U;
    for (int i = tid; i < CL * DK; i += CT) {
        const int r = i / DK, d = i % DK;
        const bool ok = r < len;
        const float* row = qkv + (size_t)(t0 + r) * C;
        Qs[r * KP + d] = ok ? row[kh * DK + d] : 0.0f;
        Ks[r * KP + d] = ok ? row[KH * DK + kh * DK + d] : 0.0f;
        Vs[i] = ok ? row[2 * KH * DK + vh * DV + d] : 0.0f;
    }
    if (warp == 0) chunk_decays(g, beta, t0, len, vh, Gs, Bs);
    __syncthreads();
    // A (strictly lower) and P (lower incl. diagonal): thread (i, 4 consecutive j).
    {
        const int i = tid / 8, j0 = (tid % 8) * 4;
        float kk[4] = {}, qk[4] = {};
#pragma unroll 4
        for (int d = 0; d < DK; ++d) {
            const float ki = Ks[i * KP + d], qi = Qs[i * KP + d];
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const float kj = Ks[(j0 + jj) * KP + d];
                kk[jj] = fmaf(ki, kj, kk[jj]);
                qk[jj] = fmaf(qi, kj, qk[jj]);
            }
        }
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const int j = j0 + jj;
            const float dec = j <= i ? __expf(Gs[i] - Gs[j]) : 0.0f;
            Tm[i * (CL + 1) + j] = j < i ? Bs[i] * dec * kk[jj] : 0.0f;
            outp[i * CL + j] = j <= i ? dec * qk[jj] : 0.0f;
        }
    }
    __syncthreads();
    // T = (I + A)^-1 by forward substitution: lane j owns column j.
    if (warp == 0) {
        const int j = lane;
        for (int i = 1; i < CL; ++i) {
            float v = 0.0f;
            if (i > j) {
                v = -Tm[i * (CL + 1) + j];
                for (int m = j + 1; m < i; ++m) v = fmaf(-Tm[i * (CL + 1) + m], Tm[m * (CL + 1) + j], v);
            }
            __syncwarp();
            if (i > j) Tm[i * (CL + 1) + j] = v;
            __syncwarp();
        }
        Tm[j * (CL + 1) + j] = 1.0f;
    }
    __syncthreads();
    for (int e = tid; e < CL * CL; e += CT) {
        const int i = e / CL, j = e % CL;
        const float t = Tm[i * (CL + 1) + j];
        TB[i * (CL + 1) + j] = t * Bs[j] * __expf(Gs[j]);
        TV[i * (CL + 1) + j] = t * Bs[j];
    }
    __syncthreads();
    // W = TB K and U = TV V: thread (row i, 16 consecutive columns).
    {
        const int i = tid / 8, d0 = (tid % 8) * 16;
        float w[16] = {}, u[16] = {};
        for (int j = 0; j <= i; ++j) {
            const float tb = TB[i * (CL + 1) + j], tv = TV[i * (CL + 1) + j];
#pragma unroll
            for (int x = 0; x < 16; ++x) {
                w[x] = fmaf(tb, Ks[j * KP + d0 + x], w[x]);
                u[x] = fmaf(tv, Vs[j * DV + d0 + x], u[x]);
            }
        }
#pragma unroll
        for (int x = 0; x < 16; x += 4) {
            *reinterpret_cast<float4*>(outw + i * DK + d0 + x) = make_float4(w[x], w[x + 1], w[x + 2], w[x + 3]);
            *reinterpret_cast<float4*>(outu + i * DV + d0 + x) = make_float4(u[x], u[x + 1], u[x + 2], u[x + 3]);
        }
    }
}

// grid (VH, DV / CW), CT threads: the chunks in order, state columns [c0, c0 + CW) in shared memory.
__global__ void __launch_bounds__(CT) chunk_state_kernel(const float* __restrict__ qkv, const float* __restrict__ g,
                                                         const float* __restrict__ scratch, float* state, float* out,
                                                         int T) {
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ float sm[];
    float* Ss = sm;                  // [DK][CW]
    float* Ws = Ss + DK * CW;        // [CL][KP]
    float* Qd = Ws + CL * KP;        // [CL][KP] exp(G_i) q_i
    float* Kd = Qd + CL * KP;        // [CL][KP] exp(G_L - G_j) k_j
    float* Us = Kd + CL * KP;        // [CL][CW]
    float* Vn = Us + CL * CW;        // [CL][CW]
    float* Pm = Vn + CL * CW;        // [CL][CL]
    __shared__ float Gs[CL];
    const int vh = blockIdx.x, kh = vh / 3, c0 = blockIdx.y * CW, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float* st = state + (size_t)vh * DK * DV;
    for (int i = tid; i < DK * CW; i += CT) Ss[i] = st[(i / CW) * DV + c0 + i % CW];
    const int chunks = (T + CL - 1) / CL;
    // The next chunk's inputs are loaded into registers while the current one is computed.
    constexpr int NW = CL * DK / CT, NU = CL * CW / CT, NP = CL * CL / CT;
    float pw[NW], pq[NW], pk[NW], pu[NU], pp[NP], pg = 0.0f;
    auto prefetch = [&](int ch) {
        if (ch >= chunks) return;
        const int t0 = ch * CL, len = min(CL, T - t0);
        const float* sw = scratch + ((size_t)ch * VH + vh) * CHUNK_FLOATS;
#pragma unroll
        for (int n = 0; n < NW; ++n) {
            const int e = tid + n * CT, r = e / DK, d = e % DK;
            const bool ok = r < len;
            const float* row = qkv + (size_t)(t0 + r) * C;
            pw[n] = sw[e];
            pq[n] = ok ? row[kh * DK + d] : 0.0f;
            pk[n] = ok ? row[KH * DK + kh * DK + d] : 0.0f;
        }
#pragma unroll
        for (int n = 0; n < NU; ++n) {
            const int e = tid + n * CT;
            pu[n] = sw[CHUNK_W + (e / CW) * DV + c0 + e % CW];
        }
#pragma unroll
        for (int n = 0; n < NP; ++n) pp[n] = sw[CHUNK_W + CHUNK_U + tid + n * CT];
        if (warp == 0) pg = lane < len ? g[(t0 + lane) * VH + vh] : 0.0f;
    };
    prefetch(0);
    for (int ch = 0; ch < chunks; ++ch) {
        const int t0 = ch * CL, len = min(CL, T - t0);
        if (warp == 0) {
            float gv = pg;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const float n = __shfl_up_sync(0xffffffff, gv, o);
                if (lane >= o) gv += n;
            }
            Gs[lane] = gv;
        }
        __syncthreads();  // also: the previous chunk is done with every buffer
        const float GL = Gs[CL - 1];
#pragma unroll
        for (int n = 0; n < NW; ++n) {
            const int e = tid + n * CT, r = e / DK, d = e % DK;
            Ws[r * KP + d] = pw[n];
            Qd[r * KP + d] = __expf(Gs[r]) * pq[n];
            Kd[r * KP + d] = __expf(GL - Gs[r]) * pk[n];
        }
#pragma unroll
        for (int n = 0; n < NU; ++n) Us[tid + n * CT] = pu[n];
#pragma unroll
        for (int n = 0; n < NP; ++n) Pm[tid + n * CT] = pp[n];
        __syncthreads();
        prefetch(ch + 1);
        const int i = tid / (CW / NV), cc = (tid % (CW / NV)) * NV;
        // Vn = U - W S0 and Q S0: thread (row i, NV consecutive columns).
        float ws[NV] = {}, qs[NV] = {};
#pragma unroll 4
        for (int d = 0; d < DK; ++d) {
            const float wv = Ws[i * KP + d], qv = Qd[i * KP + d];
#pragma unroll
            for (int x = 0; x < NV; ++x) {
                const float sv = Ss[d * CW + cc + x];
                ws[x] = fmaf(wv, sv, ws[x]);
                qs[x] = fmaf(qv, sv, qs[x]);
            }
        }
#pragma unroll
        for (int x = 0; x < NV; ++x) Vn[i * CW + cc + x] = Us[i * CW + cc + x] - ws[x];
        __syncthreads();
        // O = exp(G) Q S0 + P Vn
        {
            float o[NV];
#pragma unroll
            for (int x = 0; x < NV; ++x) o[x] = qs[x];
            for (int j = 0; j <= i; ++j) {
                const float p = Pm[i * CL + j];
#pragma unroll
                for (int x = 0; x < NV; ++x) o[x] = fmaf(p, Vn[j * CW + cc + x], o[x]);
            }
            if (i < len)
#pragma unroll
                for (int x = 0; x < NV; ++x) out[(size_t)(t0 + i) * VH * DV + vh * DV + c0 + cc + x] = o[x];
        }
        // S = exp(G_L) S0 + Kd^T Vn: thread (row d, NS consecutive columns).
        {
            const int d = tid / (CW / NS), cs = (tid % (CW / NS)) * NS;
            const float eg = __expf(GL);
            float sv[NS];
#pragma unroll
            for (int x = 0; x < NS; ++x) sv[x] = eg * Ss[d * CW + cs + x];
            for (int j = 0; j < len; ++j) {
                const float kd = Kd[j * KP + d];
#pragma unroll
                for (int x = 0; x < NS; ++x) sv[x] = fmaf(kd, Vn[j * CW + cs + x], sv[x]);
            }
            __syncthreads();  // every thread has read S0
#pragma unroll
            for (int x = 0; x < NS; ++x) Ss[d * CW + cs + x] = sv[x];
        }
    }
    __syncthreads();
    for (int i = tid; i < DK * CW; i += CT) st[(i / CW) * DV + c0 + i % CW] = Ss[i];
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
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(conv_kernel, dim3(C / 256, (T + CONV_TT - 1) / CONV_TT), 256, 0, st, x, ldx, out,
                              static_cast<const float*>(state), static_cast<const __nv_bfloat16*>(w), T));
    SQ_CHECK_LAUNCH(sq_launch(advance_conv_kernel, C / 256, 256, 0, st, x, ldx, state, T));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_gdn_conv_prep(const float* x, int ldx, float* out, float* state, const void* w, const float* a, const float* b,
                                            int ldab, const void* a_log, const void* dt_bias, float* beta, float* g, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(conv_prep_kernel, T * KH, DK, 0, st, x, ldx, out, static_cast<const float*>(state),
                              static_cast<const __nv_bfloat16*>(w), a, b, ldab, static_cast<const __nv_bfloat16*>(a_log),
                              static_cast<const __nv_bfloat16*>(dt_bias), beta, g));
    SQ_CHECK_LAUNCH(sq_launch(advance_conv_kernel, C / 256, 256, 0, st, x, ldx, state, T));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_gdn_prep(float* qkv, const float* a, const float* b, int ldab, const void* a_log,
                                       const void* dt_bias, float* beta, float* g, int T, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(prep_kernel, T * KH, DK, 0, st, qkv, a, b, ldab, static_cast<const __nv_bfloat16*>(a_log),
                                       static_cast<const __nv_bfloat16*>(dt_bias), beta, g));
    SQ_RETURN_LAUNCH();
}

extern "C" size_t squash_gdn_chunk_scratch_floats(int T) { return (size_t)((T + CL - 1) / CL) * VH * CHUNK_FLOATS; }

// With `scratch` (squash_gdn_chunk_scratch_floats(T) floats) and T >= 64: the chunked form.
extern "C" cudaError_t squash_gdn_recurrent(const float* qkv, const float* beta, const float* g, float* state, float* out,
                                            int T, float* scratch, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    if (scratch && T >= 64) {
        static bool configured = false;
        if (!configured) {
            cudaError_t e = cudaFuncSetAttribute(chunk_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PREP_SMEM);
            if (e == cudaSuccess)
                e = cudaFuncSetAttribute(chunk_state_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)STATE_SMEM);
            if (e != cudaSuccess) return e;
            configured = true;
        }
        const int chunks = (T + CL - 1) / CL;
        SQ_CHECK_LAUNCH(sq_launch(chunk_prep_kernel, dim3(chunks, VH), CT, PREP_SMEM, st, qkv, beta, g, scratch, T));
        SQ_CHECK_LAUNCH(sq_launch(chunk_state_kernel, dim3(VH, DV / CW), CT, STATE_SMEM, st, qkv, g,
                                  static_cast<const float*>(scratch), state, out, T));
        SQ_RETURN_LAUNCH();
    }
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

// Shared memory of the fused kernel: raw inputs of all tokens it processes (replayed + new).
constexpr int MAX_STEPS = 2 * REC_TOKENS;

__global__ void __launch_bounds__(FT) fused_small_kernel(const float* __restrict__ proj, int ld, int T,
                                                         const __nv_bfloat16* __restrict__ conv_w,
                                                         float* conv_state, float* state,
                                                         const __nv_bfloat16* __restrict__ a_log,
                                                         const __nv_bfloat16* __restrict__ dt_bias,
                                                         const __nv_bfloat16* __restrict__ norm_w, int z_col, int a_col,
                                                         int b_col, __nv_bfloat16* y, float* rec, const int* ctl,
                                                         int defer) {
    pdl_launch_dependents();
    // Dynamic shared memory: xs [steps][384] raw channel inputs (v replaced in place by its conv
    // output), qk [steps][256] conv outputs of q and k stored interleaved (row part * 32 + j at
    // 4 j + part, so the four parts of a column read four different banks), so [T][128] head outputs.
    extern __shared__ float smem[];
    float* xs = smem;
    float* qk = xs + MAX_STEPS * 3 * DK;
    float* so = qk + MAX_STEPS * 2 * DK;
    __shared__ float red[MAX_STEPS][8], red2[REC_TOKENS][4];
    __shared__ float s_beta[MAX_STEPS], s_decay[MAX_STEPS];
    const int vh = blockIdx.x, kh = vh / 3, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int col = tid >> 2, part = tid & 3;
    // Channels handled by this block: q[kh], k[kh], v[vh] (384); threads 0..383 own one each.
    const int ch = tid < DK ? kh * DK + tid : tid < 2 * DK ? KH * DK + kh * DK + (tid - DK) : 2 * KH * DK + vh * DV + (tid - 2 * DK);
    const bool owns_channel = tid < 3 * DK;
    float hist[3] = {}, cw[4] = {};
    if (owns_channel) {
#pragma unroll
        for (int j = 0; j < 3; ++j) hist[j] = conv_state[j * C + ch];
#pragma unroll
        for (int j = 0; j < 4; ++j) cw[j] = bf2f(conv_w[ch * TAPS + j]);
    }
    // The q / k conv history is shared by the three value heads of key head kh: it is written back
    // only by the last of the three blocks to have read it (a ticket per key head, after the history
    // conv_state[0, 3 C), which the last block resets).
    __shared__ bool s_qk_writer;
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        unsigned* ticket = reinterpret_cast<unsigned*>(conv_state + 3 * C) + kh;
        s_qk_writer = atomicAdd(ticket, 1u) == 2;
        if (s_qk_writer) atomicExch(ticket, 0u);
    }
    __syncthreads();
    const bool writes_conv = owns_channel && (tid >= 2 * DK || s_qk_writer);
    float s[32];
    float* st = state + (size_t)vh * DK * DV;
#pragma unroll
    for (int j = 0; j < 32; ++j) s[j] = st[(part * 32 + j) * DV + col];
    const float A = -__expf(bf2f(a_log[vh])), bias = bf2f(dt_bias[vh]);
    // State, conv history and weights come from earlier steps or are static: loaded before the wait.
    pdl_wait();

    // All raw inputs up front: replayed records (previous step's accepted tokens), then this step's rows.
    const int replay = rec ? ctl[0] : 0, parity = rec ? ctl[1] : 0;
    const float* prev = rec ? rec + (size_t)parity * REC_TOKENS * REC : nullptr;
    const int steps = replay + T;
    // Eight steps' loads in flight at a time.
    if (owns_channel)
        for (int i0 = 0; i0 < steps; i0 += 8) {
            float v[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int i = i0 + j;
                if (i < steps) v[j] = (i < replay ? prev + (size_t)i * REC : proj + (size_t)(i - replay) * ld)[ch];
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (i0 + j < steps) xs[(i0 + j) * 3 * DK + tid] = v[j];
        }
    if (tid < steps) {
        const int i = tid;
        const float* row = i < replay ? prev + (size_t)i * REC : proj + (size_t)(i - replay) * ld;
        const float xa = row[(i < replay ? C : a_col) + vh] + bias;
        s_decay[i] = __expf(A * (xa > 20.0f ? xa : log1pf(__expf(xa))));
        s_beta[i] = sigmoidf(row[(i < replay ? C + VH : b_col) + vh]);
    }
    // Record this step's tokens for the next replay (other parity) when deferring.
    float* next = rec ? rec + (size_t)(parity ^ 1) * REC_TOKENS * REC : nullptr;
    if (defer && next)
        for (int t = 0; t < T; ++t) {
            const float* row = proj + (size_t)t * ld;
            if (writes_conv) next[(size_t)t * REC + ch] = row[ch];
            if (tid == 0) {
                next[(size_t)t * REC + C + vh] = row[a_col + vh];
                next[(size_t)t * REC + C + VH + vh] = row[b_col + vh];
            }
        }

    // Causal conv + SiLU of every step (each thread reads and writes only its own channel), and the
    // per-warp q / k sums of squares (warps 0-3 hold q, 4-7 hold k).
    if (owns_channel) {
        const int qki = tid < 2 * DK ? (tid & ~(DK - 1)) + (((tid & 31) << 2) | ((tid & (DK - 1)) >> 5)) : 0;
        for (int i = 0; i < steps; ++i) {
            const float x = xs[i * 3 * DK + tid];
            const float c = silu(cw[0] * hist[0] + cw[1] * hist[1] + cw[2] * hist[2] + cw[3] * x);
            hist[0] = hist[1];
            hist[1] = hist[2];
            hist[2] = x;
            if (tid < 2 * DK) {
                qk[i * 2 * DK + qki] = c;
                const float sum = warp_sum(c * c);
                if (lane == 0) red[i][warp] = sum;
            } else {
                xs[i * 3 * DK + tid] = c;
            }
            if (i == replay - 1 && defer && writes_conv)
#pragma unroll
                for (int j = 0; j < 3; ++j) conv_state[j * C + ch] = hist[j];
        }
        if (!defer && writes_conv)
#pragma unroll
            for (int j = 0; j < 3; ++j) conv_state[j * C + ch] = hist[j];
    }
    __syncthreads();

    // The recurrence: registers only, no block barriers.
    for (int i = 0; i < steps; ++i) {
        const float* sq = qk + i * 2 * DK;
        const float* sk = sq + DK;
        const bool out = i >= replay;
        const float qn = rsqrtf(red[i][0] + red[i][1] + red[i][2] + red[i][3] + EPS) * rsqrtf((float)DK);
        const float kn = rsqrtf(red[i][4] + red[i][5] + red[i][6] + red[i][7] + EPS);
        const float decay = s_decay[i], beta = s_beta[i];
        float kvm = 0;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            s[j] *= decay;
            kvm += s[j] * sk[4 * j + part];
        }
        kvm *= kn;
        kvm += __shfl_xor_sync(0xffffffff, kvm, 1);
        kvm += __shfl_xor_sync(0xffffffff, kvm, 2);
        const float delta = beta * (xs[i * 3 * DK + 2 * DK + col] - kvm);
        float o = 0;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            s[j] += sk[4 * j + part] * kn * delta;
            if (out) o += s[j] * sq[4 * j + part];
        }
        // The replayed prefix is the new committed state.
        if (i == replay - 1 && defer)
#pragma unroll
            for (int j = 0; j < 32; ++j) st[(part * 32 + j) * DV + col] = s[j];
        if (out) {
            o *= qn;
            o += __shfl_xor_sync(0xffffffff, o, 1);
            o += __shfl_xor_sync(0xffffffff, o, 2);
            if (part == 0) so[(i - replay) * DV + col] = o;
        }
    }
    // Plain steps commit everything; deferred steps leave this step's tokens to the next replay.
    if (!defer)
#pragma unroll
        for (int j = 0; j < 32; ++j) st[(part * 32 + j) * DV + col] = s[j];
    __syncthreads();

    // Gated RMSNorm over each token's 128 outputs: warp w sums columns 32 (w & 3) .. + 31 of token
    // w / 4 (+ 4 per pass), the same lanes and order for every batch size.
    const int q4 = warp & 3, oc = q4 * 32 + lane;
    for (int t = warp >> 2; t < T; t += FT / 32 / 4) {
        const float ov = so[t * DV + oc];
        const float sum = warp_sum(ov * ov);
        if (lane == 0) red2[t][q4] = sum;
    }
    __syncthreads();
    for (int t = warp >> 2; t < T; t += FT / 32 / 4) {
        const float r = rsqrtf((red2[t][0] + red2[t][1] + red2[t][2] + red2[t][3]) / DV + EPS);
        y[(size_t)t * VH * DV + vh * DV + oc] =
            f2bf(so[t * DV + oc] * r * bf2f(norm_w[oc]) * sigmoidf(proj[(size_t)t * ld + z_col + vh * DV + oc]));
    }
}

}  // namespace

extern "C" size_t squash_gdn_conv_tickets() { return KH; }

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
    // Dynamic shared memory holds the raw inputs of every processed token (replay + new).
    const size_t smem = ((size_t)MAX_STEPS * 5 * DK + (size_t)REC_TOKENS * DV) * sizeof(float);
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(fused_small_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    if (T > REC_TOKENS) return cudaErrorInvalidValue;
    SQ_CHECK_LAUNCH(sq_launch(fused_small_kernel, VH, FT, smem, st, proj, ld, T, static_cast<const __nv_bfloat16*>(conv_w), conv_state, state,
                                          static_cast<const __nv_bfloat16*>(a_log),
                                          static_cast<const __nv_bfloat16*>(dt_bias),
                                          static_cast<const __nv_bfloat16*>(norm_w), z_col, a_col, b_col,
                                          static_cast<__nv_bfloat16*>(y), rec, ctl, defer));
    SQ_RETURN_LAUNCH();
}
