// Chunked Gated DeltaNet prefill on the tensor cores (SPEC §5): the WY/UT-transform of the CUDA-core
// chunk kernels in gdn.cu with every matrix product on TF32 MMAs (m16n8k8, FP32 accumulation). With
// TF32X3 each product is split into high and low TF32 parts (hi*hi + hi*lo + lo*hi), which keeps
// FP32-level accuracy at three times the MMA count.
//
// Per (chunk, head), rows i, j in [0, L), G = inclusive prefix sum of g within the chunk:
//   A[i][j] = beta_i exp(G_i - G_j) k_i.k_j (j < i),  T = (I + A)^-1,
//   W = T diag(beta exp(G)) K,  U = T diag(beta) V,  P[i][j] = exp(G_i - G_j) q_i.k_j (j <= i);
// then per head, the chunks in order with the state S [128 k][128 v]:
//   Vn = U - W S,  O = diag(exp(G)) (Q S) + P Vn,  S = exp(G_L) S + K^T (diag(exp(G_L - G)) Vn).
// The state walk keeps every per-chunk buffer in shared memory and prefetches the next chunk's copy of
// each with cp.async as soon as the current chunk is done with it.

#include "mma.cuh"

namespace {

constexpr int C = 10240, KH = 16, VH = 48, DK = 128, DV = 128;
constexpr int L = 32;                // chunk length
constexpr int LDK = DK + 4;          // padded 128-wide rows
constexpr int CW = 32, LDS = CW + 4;  // state columns per CTA; padded 32-wide rows
constexpr int THREADS = 256;
constexpr int CH_W = L * DK, CH_U = L * DV, CH_P = L * L, CH_FLOATS = CH_W + CH_U + CH_P;
constexpr bool TF32X3 = true;

SQ_DEVICE uint32_t tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

SQ_DEVICE void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// c[16 x 8] += A[16 x K] B[K x 8] over k in [0, K): A(r, k) = A[r * ar + k * ak], B(k, n) = B[k * bk + n * bn].
// The high products and the low corrections accumulate separately (two independent MMA chains) and
// are added at the end.
// `ks` (optional): B row k is scaled by ks[k].
template <int K>
SQ_DEVICE void mma_tile(float* c, const float* A, int ar, int ak, const float* B, int bk, int bn, const float* ks = nullptr) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    float lo[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
    for (int k = 0; k < K; k += 8) {
        const float af[4] = {A[g * ar + (k + t) * ak], A[(g + 8) * ar + (k + t) * ak], A[g * ar + (k + t + 4) * ak],
                             A[(g + 8) * ar + (k + t + 4) * ak]};
        float bf[2] = {B[(k + t) * bk + g * bn], B[(k + t + 4) * bk + g * bn]};
        if (ks) bf[0] *= ks[k + t], bf[1] *= ks[k + t + 4];
        uint32_t ah[4], bh[2];
#pragma unroll
        for (int x = 0; x < 4; ++x) ah[x] = tf32(af[x]);
#pragma unroll
        for (int x = 0; x < 2; ++x) bh[x] = tf32(bf[x]);
        if constexpr (TF32X3) {
            uint32_t al[4], bl[2];
#pragma unroll
            for (int x = 0; x < 4; ++x) al[x] = tf32(af[x] - __uint_as_float(ah[x]));
#pragma unroll
            for (int x = 0; x < 2; ++x) bl[x] = tf32(bf[x] - __uint_as_float(bh[x]));
            mma_tf32(lo, al, bh);
            mma_tf32(lo, ah, bl);
        }
        mma_tf32(c, ah, bh);
    }
    if constexpr (TF32X3)
#pragma unroll
        for (int x = 0; x < 4; ++x) c[x] += lo[x];
}

// Accumulator element e of an m16n8 tile sits at row g (+8 for e >= 2), column 2t (+1 for odd e).
SQ_DEVICE int frag_row(int e) { return ((threadIdx.x & 31) >> 2) + (e >> 1) * 8; }
SQ_DEVICE int frag_col(int e) { return ((threadIdx.x & 3) << 1) + (e & 1); }

SQ_DEVICE void cp_async4(void* smem, const void* gmem, bool valid) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(valid ? 4 : 0));
}

// Rows t0 .. t0 + L - 1 (zero past `len`) of `cols` floats at column `col` of the qkv rows into dst [L][ld].
SQ_DEVICE void async_rows(float* dst, int ld, const float* __restrict__ qkv, int t0, int len, int col, int cols) {
    const int per = cols / 4;
    for (int e = threadIdx.x; e < L * per; e += THREADS) {
        const int r = e / per, c4 = (e % per) * 4;
        const bool ok = r < len;
        cp_async16(dst + r * ld + c4, qkv + (size_t)(ok ? t0 + r : 0) * C + col + c4, ok);
    }
}

// Lanes 0..31 of warp 0: inclusive prefix sum of the chunk's g (zero past `len`).
SQ_DEVICE float warp_scan(float gv) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float n = __shfl_up_sync(0xffffffff, gv, o);
        if (lane >= o) gv += n;
    }
    return gv;
}

// grid (chunks, VH), THREADS threads. Shared: Kb, Qb [L][LDK] (Qb reused for V), Am [L][L+1] (A, then
// TB), Tm, Tv [L][L+1].
constexpr int LT = L + 1;
constexpr size_t PREP_SMEM = (2 * L * LDK + 3 * L * LT) * sizeof(float);
__global__ void __launch_bounds__(THREADS) prep_tc_kernel(const float* __restrict__ qkv, const float* __restrict__ beta,
                                                          const float* __restrict__ g, float* scratch, int T) {
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ float sm[];
    float* Kb = sm;
    float* Qb = Kb + L * LDK;
    float* Am = Qb + L * LDK;
    float* Tm = Am + L * LT;
    float* Tv = Tm + L * LT;
    __shared__ float Gs[L], Bs[L];
    const int ch = blockIdx.x, vh = blockIdx.y, kh = vh / 3, t0 = ch * L, len = min(L, T - t0);
    const int tid = threadIdx.x, warp = tid >> 5;
    float* outw = scratch + ((size_t)ch * VH + vh) * CH_FLOATS;
    float* outu = outw + CH_W;
    float* outp = outu + CH_U;
    async_rows(Kb, LDK, qkv, t0, len, KH * DK + kh * DK, DK);
    async_rows(Qb, LDK, qkv, t0, len, kh * DK, DK);
    cp_async_commit();
    if (warp == 0) {
        const int r = tid;
        Gs[r] = warp_scan(r < len ? g[(t0 + r) * VH + vh] : 0.0f);
        Bs[r] = r < len ? beta[(t0 + r) * VH + vh] : 0.0f;
    }
    cp_async_wait<0>();
    __syncthreads();
    // A = K K^T and P = Q K^T: warp (row block w / 4, columns (w % 4) * 8).
    {
        const int rb = warp >> 2, cb = (warp & 3) * 8;
        float ka[4] = {}, qa[4] = {};
        mma_tile<DK>(ka, Kb + rb * 16 * LDK, LDK, 1, Kb + cb * LDK, 1, LDK);
        mma_tile<DK>(qa, Qb + rb * 16 * LDK, LDK, 1, Kb + cb * LDK, 1, LDK);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int i = rb * 16 + frag_row(e), j = cb + frag_col(e);
            const float dec = j <= i ? __expf(Gs[i] - Gs[j]) : 0.0f;
            Am[i * LT + j] = j < i ? Bs[i] * dec * ka[e] : 0.0f;
            outp[i * L + j] = j <= i ? dec * qa[e] : 0.0f;
        }
    }
    __syncthreads();
    // V into Qb (asynchronously) while T = (I + A)^-1 is solved: thread j owns column j of T.
    async_rows(Qb, LDK, qkv, t0, len, 2 * KH * DK + vh * DV, DV);
    cp_async_commit();
    if (tid < L) {
        const int j = tid;
        for (int i = 0; i < L; ++i) Tm[i * LT + j] = i == j ? 1.0f : 0.0f;
        for (int i = j + 1; i < L; ++i) {
            float v = -Am[i * LT + j];
            for (int m = j + 1; m < i; ++m) v = fmaf(-Am[i * LT + m], Tm[m * LT + j], v);
            Tm[i * LT + j] = v;
        }
    }
    __syncthreads();
    // TB = T diag(beta exp(G)) into Am, TV = T diag(beta) into Tv.
    for (int e = tid; e < L * L; e += THREADS) {
        const int i = e / L, j = e % L;
        const float t = Tm[i * LT + j];
        Am[i * LT + j] = t * Bs[j] * __expf(Gs[j]);
        Tv[i * LT + j] = t * Bs[j];
    }
    cp_async_wait<0>();
    __syncthreads();
    // W = TB K and U = TV V: warp (row block w / 4, 32 columns (w % 4) * 32).
    {
        const int rb = warp >> 2, cb = (warp & 3) * 32;
        float w[4][4] = {}, u[4][4] = {};
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            mma_tile<L>(w[n], Am + rb * 16 * LT, LT, 1, Kb + cb + n * 8, LDK, 1);
            mma_tile<L>(u[n], Tv + rb * 16 * LT, LT, 1, Qb + cb + n * 8, LDK, 1);
        }
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = rb * 16 + frag_row(e), d = cb + n * 8 + frag_col(e);
                outw[i * DK + d] = w[n][e];
                outu[i * DV + d] = u[n][e];
            }
    }
}

// grid (VH, 3), ST_THREADS threads: the chunks in order, state columns [c0, c0 + nc) with slabs of 48, 40
// and 40 columns (144 CTAs: one wave).
constexpr int ST_THREADS = 384, NCMAX = 48, LDN = NCMAX + 4, LDP = L + 4;
constexpr size_t STATE_SMEM = (DK * LDN + 3 * L * LDK + L * LDP + 2 * L * LDN) * sizeof(float);
__global__ void __launch_bounds__(ST_THREADS) state_tc_kernel(const float* __restrict__ qkv, const float* __restrict__ g,
                                                              const float* __restrict__ scratch, float* state, float* out, int T) {
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ float sm[];
    float* Ss = sm;               // [DK][LDN]
    float* Wb = Ss + DK * LDN;    // [L][LDK]
    float* Qb = Wb + L * LDK;     // [L][LDK]
    float* Kb = Qb + L * LDK;     // [L][LDK]
    float* Pb = Kb + L * LDK;     // [L][LDP]
    float* Ub = Pb + L * LDP;     // [L][LDN]
    float* Vn = Ub + L * LDN;     // [L][LDN]
    __shared__ float gbuf[2][L];
    __shared__ float Gs[L], Kd[L];
    const int vh = blockIdx.x, kh = vh / 3, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int c0 = blockIdx.y == 0 ? 0 : 8 + 40 * blockIdx.y, nc = blockIdx.y == 0 ? 48 : 40, nt = nc / 8;
    float* st = state + (size_t)vh * DK * DV;
    for (int e = tid; e < DK * nc; e += ST_THREADS) Ss[(e / nc) * LDN + e % nc] = st[(e / nc) * DV + c0 + e % nc];
    const int chunks = (T + L - 1) / L;

    auto chunk_w = [&](int ch) { return scratch + ((size_t)ch * VH + vh) * CH_FLOATS; };
    auto rows = [&](float* dst, int ld, int t0, int len, int col) {
        for (int e = tid; e < L * DK / 4; e += ST_THREADS) {
            const int r = e / (DK / 4), c4 = (e % (DK / 4)) * 4;
            const bool ok = r < len;
            cp_async16(dst + r * ld + c4, qkv + (size_t)(ok ? t0 + r : 0) * C + col + c4, ok);
        }
    };
    auto load_wq = [&](int ch) {  // W, Q and g of chunk ch
        if (ch < chunks) {
            const float* sw = chunk_w(ch);
            for (int e = tid; e < L * DK / 4; e += ST_THREADS) {
                const int r = e / (DK / 4), c4 = (e % (DK / 4)) * 4;
                cp_async16(Wb + r * LDK + c4, sw + r * DK + c4, true);
            }
            const int t0 = ch * L, len = min(L, T - t0);
            rows(Qb, LDK, t0, len, kh * DK);
            if (tid < L) cp_async4(&gbuf[ch & 1][tid], g + (size_t)(tid < len ? t0 + tid : 0) * VH + vh, tid < len);
        }
        cp_async_commit();
    };
    auto load_u = [&](int ch) {
        if (ch < chunks) {
            const float* su = chunk_w(ch) + CH_W;
            for (int e = tid; e < L * nc / 4; e += ST_THREADS) {
                const int r = e / (nc / 4), c4 = (e % (nc / 4)) * 4;
                cp_async16(Ub + r * LDN + c4, su + r * DV + c0 + c4, true);
            }
        }
        cp_async_commit();
    };
    auto load_p = [&](int ch) {
        if (ch < chunks) {
            const float* sp = chunk_w(ch) + CH_W + CH_U;
            for (int e = tid; e < L * L / 4; e += ST_THREADS) {
                const int r = e / (L / 4), c4 = (e % (L / 4)) * 4;
                cp_async16(Pb + r * LDP + c4, sp + r * L + c4, true);
            }
        }
        cp_async_commit();
    };
    auto load_k = [&](int ch) {
        if (ch < chunks) {
            const int t0 = ch * L, len = min(L, T - t0);
            rows(Kb, LDK, t0, len, KH * DK + kh * DK);
        }
        cp_async_commit();
    };
    // Groups are committed in the order W/Q, U, P, K; each phase waits for the oldest of four.
    load_wq(0);
    load_u(0);
    load_p(0);
    load_k(0);
    // 32 x nc outputs (Q S, W S, O, Vn) as m16n8 tiles: warp w < 2 nt takes row block w % 2, n tile w / 2.
    const bool has = warp < 2 * nt;
    const int rb = warp & 1, cb = (warp >> 1) * 8;
    for (int ch = 0; ch < chunks; ++ch) {
        const int t0 = ch * L, len = min(L, T - t0);
        cp_async_wait<3>();  // W, Q, g of this chunk
        __syncthreads();
        if (warp == 0) Gs[lane] = warp_scan(gbuf[ch & 1][lane]);
        // Phase 1: W S and Q S.
        float ws[4] = {}, qs[4] = {};
        if (has) {
            mma_tile<DK>(ws, Wb + rb * 16 * LDK, LDK, 1, Ss + cb, LDN, 1);
            mma_tile<DK>(qs, Qb + rb * 16 * LDK, LDK, 1, Ss + cb, LDN, 1);
        }
        __syncthreads();  // W, Q free; Gs visible
        load_wq(ch + 1);
        const float GL = Gs[L - 1];
        if (tid < L) Kd[tid] = __expf(GL - Gs[tid]);
        cp_async_wait<3>();  // U
        __syncthreads();
        // Phase 2: Vn = U - W S.
        if (has)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = rb * 16 + frag_row(e), c = cb + frag_col(e);
                Vn[i * LDN + c] = Ub[i * LDN + c] - ws[e];
            }
        __syncthreads();  // U free; Vn, Kd visible
        load_u(ch + 1);
        cp_async_wait<3>();  // P
        __syncthreads();
        // Phase 3: O = exp(G) (Q S) + P Vn.
        if (has) {
            float o[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) o[e] = __expf(Gs[rb * 16 + frag_row(e)]) * qs[e];
            mma_tile<L>(o, Pb + rb * 16 * LDP, LDP, 1, Vn + cb, LDN, 1);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = rb * 16 + frag_row(e), c = cb + frag_col(e);
                if (i < len) out[(size_t)(t0 + i) * VH * DV + vh * DV + c0 + c] = o[e];
            }
        }
        __syncthreads();  // P free
        load_p(ch + 1);
        cp_async_wait<3>();  // K
        __syncthreads();
        // Phase 4: S = exp(G_L) S + K^T diag(exp(G_L - G)) Vn: 128 x nc outputs, 8 row blocks x nt tiles.
        {
            const float egl = __expf(GL);
            for (int id = warp; id < 8 * nt; id += ST_THREADS / 32) {
                const int rk = id & 7, n = id >> 3;
                float sa[4] = {};
                mma_tile<L>(sa, Kb + rk * 16, 1, LDK, Vn + n * 8, LDN, 1, Kd);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int d = rk * 16 + frag_row(e), c = n * 8 + frag_col(e);
                    Ss[d * LDN + c] = fmaf(egl, Ss[d * LDN + c], sa[e]);
                }
            }
        }
        __syncthreads();  // K free; S updated
        load_k(ch + 1);
    }
    cp_async_wait<0>();
    __syncthreads();
    for (int e = tid; e < DK * nc; e += ST_THREADS) st[(e / nc) * DV + c0 + e % nc] = Ss[(e / nc) * LDN + e % nc];
}

}  // namespace

extern "C" size_t squash_gdn_tc_scratch_floats(int T) { return (size_t)((T + L - 1) / L) * VH * CH_FLOATS; }

extern "C" cudaError_t squash_gdn_chunked_tc(const float* qkv, const float* beta, const float* g, float* state, float* out, int T,
                                             float* scratch, cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    static bool configured = false;
    if (!configured) {
        cudaError_t e = cudaFuncSetAttribute(prep_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PREP_SMEM);
        if (e == cudaSuccess) e = cudaFuncSetAttribute(state_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)STATE_SMEM);
        if (e != cudaSuccess) return e;
        configured = true;
    }
    const int chunks = (T + L - 1) / L;
    SQ_CHECK_LAUNCH(sq_launch(prep_tc_kernel, dim3(chunks, VH), THREADS, PREP_SMEM, st, qkv, beta, g, scratch, T));
    SQ_CHECK_LAUNCH(sq_launch(state_tc_kernel, dim3(VH, 3), ST_THREADS, STATE_SMEM, st, qkv, g, static_cast<const float*>(scratch),
                              state, out, T));
    SQ_RETURN_LAUNCH();
}
