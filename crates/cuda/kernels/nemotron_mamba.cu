// Nemotron-H Mamba-2 mixer between in_proj and out_proj (see crates/cuda/src/nemotron.rs).
//
// mamba_conv: causal depthwise conv + SiLU of xBC and dt = softplus(dt + dt_bias) per row (reads state);
// mamba_conv_commit: advances the conv state or records a deferred step; mamba_scan: the selective-state
// recurrence, exact and sequential per sequence (same per-token arithmetic for every row count), with the
// D skip; mamba_gated_norm: y * silu(z), RMS-normalized per group of 1024 channels.
//
// Speculative steps (defer) keep the state at the start of the step and record the step's rows; the next
// launch of the layer replays the accepted ones (count and record parity in the sequence's header) first.

#include "mma.cuh"
#include "nemotron.cuh"
#include "variants.cuh"

using namespace nem;

namespace {

// Raw (pre-conv) xBC value of channel c of in_proj row r, as the model's BF16 Linear output.
SQ_DEVICE float raw_xbc(const float* proj, int ld, int r, int c) { return rb(__ldcg(proj + (size_t)r * ld + XBC_COL + c)); }

// Committed conv history value k (0 oldest .. 2) of channel c: the last three of (conv state, replayed rows).
SQ_DEVICE float window(const Seqs& q, int s, int layer, int rec_rows, int replay, int parity, int k, int c) {
    const int j = replay + k;
    if (j >= 3) return bf2f(__ldcg(rec_raw(rec_row(q, s, layer, rec_rows, parity, j - 3)) + c));
    return __ldcg(conv_state(q, s, layer, rec_rows) + (size_t)j * CONV_DIM + c);
}

constexpr int CONV_THREADS = 256, CONV_ROWS = 16;

// Grid (ceil((CONV_DIM + HEADS) / 256), ceil(rows / CONV_ROWS)): thread = (channel, CONV_ROWS rows). Channels
// past CONV_DIM compute dt of head c - CONV_DIM.
__global__ void __launch_bounds__(CONV_THREADS) mamba_conv_kernel(const float* proj, int ld, int rows, const Seqs* sq, int layer,
                                                                  int rec_rows, MambaW w, bf16* xbc, float* dt) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ Seqs q;
    __shared__ int ctl[MAX_SEQ][2];
    load_seqs(q, sq, ctl);
    const int c = blockIdx.x * CONV_THREADS + threadIdx.x;
    const int r0 = blockIdx.y * CONV_ROWS, r1 = min(rows, r0 + CONV_ROWS);
    if (c >= CONV_DIM + HEADS) return;
    if (c >= CONV_DIM) {
        const int h = c - CONV_DIM;
        const float bias = w.dt_bias[h];
        for (int r = r0; r < r1; ++r) {
            if (seq_of_row(q, r) < 0) break;
            const float v = rb(__ldcg(proj + (size_t)r * ld + DT_COL + h)) + bias;
            const float sp = v <= 20.0f ? log1pf(expf(v)) : v;
            dt[(size_t)r * HEADS + h] = fminf(fmaxf(sp, w.dt_min), w.dt_max);
        }
        return;
    }
    const float w0 = w.conv_w[c * TAPS], w1 = w.conv_w[c * TAPS + 1], w2 = w.conv_w[c * TAPS + 2], w3 = w.conv_w[c * TAPS + 3];
    const float b = w.conv_b[c];
    int s = -1;
    float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f;
    for (int r = r0; r < r1; ++r) {
        const int sr = seq_of_row(q, r);
        if (sr < 0) break;
        const int i = r - q.row0[sr];
        if (sr != s) {
            s = sr;
            const int replay = ctl[s][0], parity = ctl[s][1];
            float hv[3];
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const int idx = i - 3 + j;
                hv[j] = idx >= 0 ? raw_xbc(proj, ld, r - 3 + j, c) : window(q, s, layer, rec_rows, replay, parity, 3 + idx, c);
            }
            h0 = hv[0], h1 = hv[1], h2 = hv[2];
        }
        const float x = raw_xbc(proj, ld, r, c);
        float acc = b;
        acc = fmaf(w0, h0, acc);
        acc = fmaf(w1, h1, acc);
        acc = fmaf(w2, h2, acc);
        acc = fmaf(w3, x, acc);
        xbc[(size_t)r * CONV_DIM + c] = f2bf(acc / (1.0f + expf(-acc)));
        h0 = h1, h1 = h2, h2 = x;
    }
}

// Grid (CONV_DIM / 256, nseq): thread = (channel, sequence).
__global__ void __launch_bounds__(CONV_THREADS) mamba_conv_commit_kernel(const float* proj, int ld, const Seqs* sq, int layer, int rec_rows,
                                                                         const bf16* xbc, const float* dt, int defer, int flush) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ Seqs q;
    __shared__ int ctl[MAX_SEQ][2];
    load_seqs(q, sq, ctl);
    const int s = blockIdx.y, c = blockIdx.x * CONV_THREADS + threadIdx.x;
    if (s >= q.nseq || c >= CONV_DIM) return;
    const int replay = ctl[s][0], parity = ctl[s][1];
    const int nrows = flush ? 0 : q.row0[s + 1] - q.row0[s], rbase = q.row0[s];
    float eff[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) eff[k] = window(q, s, layer, rec_rows, replay, parity, k, c);
    float* cs = conv_state(q, s, layer, rec_rows);
    if (defer) {
        if (replay > 0)
#pragma unroll
            for (int k = 0; k < 3; ++k) cs[(size_t)k * CONV_DIM + c] = eff[k];
        const int nrec = min(nrows, rec_rows);
        for (int i = 0; i < nrec; ++i) {
            unsigned char* row = rec_row(q, s, layer, rec_rows, parity ^ 1, i);
            const int r = rbase + i;
            rec_raw(row)[c] = f2bf(raw_xbc(proj, ld, r, c));
            rec_conv(row)[c] = __ldcg(xbc + (size_t)r * CONV_DIM + c);
            if (c < HEADS) rec_dt(row)[c] = __ldcg(dt + (size_t)r * HEADS + c);
        }
    } else if (replay > 0 || nrows > 0) {
        float v[3];
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int j = nrows + k;
            v[k] = j < 3 ? eff[j] : raw_xbc(proj, ld, rbase + j - 3, c);
        }
#pragma unroll
        for (int k = 0; k < 3; ++k) cs[(size_t)k * CONV_DIM + c] = v[k];
    }
}

// Selective scan: block = (head h, PB channels) x sequence, warps of SLOTS channels x 128 state entries (lane:
// n = 4 lane .. 4 lane + 3). Per token and channel: dA = exp(dt A), S = fmaf(S, dA, (B dt) x); y = the
// warp_sum tree over lanes of the lane partial S0 C0 + S1 C1 + S2 C2 + S3 C3 (fma order), + x D. The same
// operations for every row count and every SLOTS / PB (warp_sum_slots is bitwise warp_sum), so prefill
// equals decode. Replayed records first, then the step's rows; tokens' raw BF16 x / B / C and dt are staged
// with cp.async, the next chunk of TT2 tokens loading while the current one is computed.
constexpr int TT2 = 32;

SQ_DEVICE void cp_async4(void* smem, const void* gmem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem));
}

template <int PB>
struct ScanStage2 {
    __align__(16) bf16 x[2][TT2][PB];
    __align__(16) bf16 B[2][TT2][STATE];
    __align__(16) bf16 C[2][TT2][STATE];
    float dt[2][TT2];
};

template <int SLOTS, int PB>
__global__ void __launch_bounds__(PB / SLOTS * 32) mamba_scan2_kernel(const bf16* xbc, const float* dt, const Seqs* sq, int layer, int rec_rows,
                                                                      MambaW w, int defer, int flush, bf16* y) {
    constexpr int THREADS = PB / SLOTS * 32, PCH2 = HEAD_DIM / PB;
    constexpr int XP = PB / 8, PIECES = XP + 2 * (STATE / 8);  // 16-byte pieces per token
    pdl_launch_dependents();
    pdl_wait();
    __shared__ Seqs q;
    __shared__ ScanStage2<PB> stg;
    __shared__ int ctl[MAX_SEQ][2];
    load_seqs(q, sq, ctl);
    const int s = blockIdx.y;
    if (s >= q.nseq) return;
    const int h = blockIdx.x / PCH2, pc = blockIdx.x % PCH2, g = h / HPG;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int pw = warp * SLOTS;  // first channel of this warp within the block
    float* st = ssm_state(q, s, layer, rec_rows) + ((size_t)h * HEAD_DIM + pc * PB + pw) * STATE + lane * 4;
    float S[SLOTS][4];
#pragma unroll
    for (int j = 0; j < SLOTS; ++j) {
        const float4 v = __ldcg(reinterpret_cast<const float4*>(st + (size_t)j * STATE));
        S[j][0] = v.x, S[j][1] = v.y, S[j][2] = v.z, S[j][3] = v.w;
    }
    auto store = [&] {
#pragma unroll
        for (int j = 0; j < SLOTS; ++j) *reinterpret_cast<float4*>(st + (size_t)j * STATE) = make_float4(S[j][0], S[j][1], S[j][2], S[j][3]);
    };
    const float A = -expf(w.a_log[h]), Dh = w.d[h];
    const int replay = ctl[s][0], parity = ctl[s][1];
    const int nrows = flush ? 0 : q.row0[s + 1] - q.row0[s], rbase = q.row0[s];
    const int total = replay + nrows;
    const int chunks = (total + TT2 - 1) / TT2;
    auto issue = [&](int c) {
        const int b = c & 1, t0 = c * TT2, nt = min(TT2, total - t0);
        for (int e = tid; e < nt * (PIECES + 1); e += THREADS) {
            const int tt = e / (PIECES + 1), k = e % (PIECES + 1), i = t0 + tt;
            const bf16* conv;
            const float* dtp;
            if (i < replay) {
                unsigned char* row = rec_row(q, s, layer, rec_rows, parity, i);
                conv = rec_conv(row);
                dtp = rec_dt(row);
            } else {
                const int r = rbase + i - replay;
                conv = xbc + (size_t)r * CONV_DIM;
                dtp = dt + (size_t)r * HEADS;
            }
            if (k < XP) cp_async16(&stg.x[b][tt][k * 8], conv + h * HEAD_DIM + pc * PB + k * 8, true);
            else if (k < XP + STATE / 8) cp_async16(&stg.B[b][tt][(k - XP) * 8], conv + INNER + g * STATE + (k - XP) * 8, true);
            else if (k < PIECES) cp_async16(&stg.C[b][tt][(k - XP - STATE / 8) * 8], conv + INNER + GROUPS * STATE + g * STATE + (k - XP - STATE / 8) * 8, true);
            else cp_async4(&stg.dt[b][tt], dtp + h);
        }
        cp_async_commit();
    };
    if (chunks > 0) issue(0);
    for (int c = 0; c < chunks; ++c) {
        const int b = c & 1, t0 = c * TT2, nt = min(TT2, total - t0);
        if (c + 1 < chunks) {
            issue(c + 1);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        for (int tt = 0; tt < nt; ++tt) {
            const int i = t0 + tt;
            const float d = stg.dt[b][tt];
            const float dA = expf(d * A);
            const uint2 braw = *reinterpret_cast<const uint2*>(&stg.B[b][tt][lane * 4]);
            const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&braw.x));
            const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&braw.y));
            const float db[4] = {b01.x * d, b01.y * d, b23.x * d, b23.y * d};
            float xs[SLOTS];
#pragma unroll
            for (int j = 0; j < SLOTS; ++j) xs[j] = bf2f(stg.x[b][tt][pw + j]);
#pragma unroll
            for (int j = 0; j < SLOTS; ++j)
#pragma unroll
                for (int k = 0; k < 4; ++k) S[j][k] = fmaf(S[j][k], dA, __fmul_rn(db[k], xs[j]));
            if (i == replay - 1 && (defer || flush)) store();
            if (i >= replay) {
                const uint2 craw = *reinterpret_cast<const uint2*>(&stg.C[b][tt][lane * 4]);
                const float2 c01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&craw.x));
                const float2 c23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&craw.y));
                float part[SLOTS];
#pragma unroll
                for (int j = 0; j < SLOTS; ++j) {
                    float a = __fmul_rn(S[j][0], c01.x);
                    a = fmaf(S[j][1], c01.y, a);
                    a = fmaf(S[j][2], c23.x, a);
                    part[j] = fmaf(S[j][3], c23.y, a);
                }
                float v;
                if constexpr (SLOTS == 1) v = warp_sum(part[0]);
                else v = warp_sum_slots<SLOTS>(part);
                constexpr int SH = SLOTS == 8 ? 2 : SLOTS == 4 ? 3 : SLOTS == 2 ? 4 : 5;
                if ((lane & ((1 << SH) - 1)) == 0) {
                    const int j = lane >> SH;
                    const float out = __fadd_rn(v, __fmul_rn(bf2f(stg.x[b][tt][pw + j]), Dh));
                    y[(size_t)(rbase + i - replay) * INNER + h * HEAD_DIM + pc * PB + pw + j] = f2bf(out);
                }
            }
        }
        __syncthreads();
    }
    if (!defer && !flush && total > 0) store();
}

constexpr int NORM_THREADS = 256;

// Grid (rows, GROUPS): one block per (row, group of 1024 channels).
__global__ void __launch_bounds__(NORM_THREADS) mamba_gated_norm_kernel(const bf16* y, const float* proj, int ld, const float* norm_w, float eps,
                                                                        bf16* out) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float scratch[32];
    const int r = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    float v[NORM_GROUP / NORM_THREADS];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < NORM_GROUP / NORM_THREADS; ++k) {
        const int c = g * NORM_GROUP + k * NORM_THREADS + tid;
        const float z = rb(__ldcg(proj + (size_t)r * ld + c));
        const float gate = z * (1.0f / (1.0f + expf(-z)));
        v[k] = bf2f(__ldcg(y + (size_t)r * INNER + c)) * gate;
        ss = fmaf(v[k], v[k], ss);
    }
    ss = block_sum(ss, scratch);
    const float rstd = 1.0f / sqrtf(ss / (float)NORM_GROUP + eps);
#pragma unroll
    for (int k = 0; k < NORM_GROUP / NORM_THREADS; ++k) {
        const int c = g * NORM_GROUP + k * NORM_THREADS + tid;
        out[(size_t)r * INNER + c] = f2bf(v[k] * rstd * norm_w[c]);
    }
}

__global__ void mamba_set_ctl_kernel(const Seqs* sq, const int* accepted, int deferred) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ Seqs q;
    load_seqs(q, sq);
    const int s = threadIdx.x;
    if (s >= q.nseq) return;
    int* hdr = header(q, s);
    if (deferred) {
        const int rows = q.row0[s + 1] - q.row0[s];
        const int a = accepted ? __ldcg(accepted + s) : 0;
        hdr[0] = a > 0 ? min(a, rows) : rows;
        hdr[1] = __ldcg(hdr + 1) ^ 1;
    } else {
        hdr[0] = 0;
    }
}

}  // namespace

extern "C" size_t quench6k_nem_seqs_bytes() { return sizeof(Seqs); }

extern "C" cudaError_t quench6k_nem_mamba_conv(const float* proj, int ld, int rows, const void* seqs, int nseq, int layer, int rec_rows, MambaW w,
                                               void* xbc, float* dt, cudaStream_t st) {
    (void)nseq;
    if (rows == 0) return cudaSuccess;
    if (ld < DT_COL + HEADS) return cudaErrorInvalidValue;
    const dim3 grid((CONV_DIM + HEADS + CONV_THREADS - 1) / CONV_THREADS, (rows + CONV_ROWS - 1) / CONV_ROWS);
    SQ_CHECK_LAUNCH(sq_launch(mamba_conv_kernel, grid, CONV_THREADS, 0, st, proj, ld, rows, static_cast<const Seqs*>(seqs), layer, rec_rows, w,
                              static_cast<bf16*>(xbc), dt));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_nem_mamba_conv_commit(const float* proj, int ld, int rows, const void* seqs, int nseq, int layer, int rec_rows,
                                                      const void* xbc, const float* dt, int defer, int flush, cudaStream_t st) {
    (void)rows;
    if (nseq <= 0) return cudaSuccess;
    if (nseq > MAX_SEQ || (defer && rec_rows <= 0)) return cudaErrorInvalidValue;
    const dim3 grid(CONV_DIM / CONV_THREADS, nseq);
    SQ_CHECK_LAUNCH(sq_launch(mamba_conv_commit_kernel, grid, CONV_THREADS, 0, st, proj, ld, static_cast<const Seqs*>(seqs), layer, rec_rows,
                              static_cast<const bf16*>(xbc), dt, defer, flush));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_nem_mamba_scan(const void* xbc, const float* dt, int rows, const void* seqs, int nseq, int layer, int rec_rows,
                                               MambaW w, int defer, int flush, void* y, cudaStream_t st) {
    (void)rows;
    if (nseq <= 0) return cudaSuccess;
    if (nseq > MAX_SEQ) return cudaErrorInvalidValue;
    const bf16* x = static_cast<const bf16*>(xbc);
    const Seqs* q = static_cast<const Seqs*>(seqs);
    bf16* yo = static_cast<bf16*>(y);
    const int v = sq_variant("nem.scan", 432);
#define SQ_SCAN2(SL, PBV)                                                                                                          \
    if (v == SL * 100 + PBV)                                                                                                       \
        SQ_CHECK_LAUNCH(sq_launch(mamba_scan2_kernel<SL, PBV>, dim3(HEADS * (HEAD_DIM / PBV), nseq), PBV / SL * 32, 0, st, x, dt, q, layer, \
                                  rec_rows, w, defer, flush, yo));
    SQ_SCAN2(4, 32)
    SQ_SCAN2(8, 64)
    SQ_SCAN2(2, 16)
#undef SQ_SCAN2
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_nem_mamba_gated_norm(const void* y, const float* proj, int ld, int rows, const float* norm_w, float eps, void* out,
                                                     cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(mamba_gated_norm_kernel, dim3(rows, GROUPS), NORM_THREADS, 0, st, static_cast<const bf16*>(y), proj, ld, norm_w, eps,
                              static_cast<bf16*>(out)));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_nem_mamba_set_ctl(const void* seqs, int nseq, const int* accepted, int deferred, cudaStream_t st) {
    (void)nseq;
    SQ_CHECK_LAUNCH(sq_launch(mamba_set_ctl_kernel, 1, 32, 0, st, static_cast<const Seqs*>(seqs), accepted, deferred));
    SQ_RETURN_LAUNCH();
}
