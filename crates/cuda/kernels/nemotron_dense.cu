// Nemotron-H prefill: 8-bit dense weights expanded to BF16 once per large step, so the projection runs the
// BF16 GEMM. The values are those the GEMM's 8-bit paths stage (gemm.cu: bf16(fp8 or int8 code x its
// block or group scale), round to nearest), so the outputs keep their bits.

#include <cuda_fp8.h>

#include "mma.cuh"
#include "nemotron.cuh"

using namespace nem;

namespace {

// One thread per 8 weights of a row (k % 8 == 0). fmt 5: E4M3 with a scale per 128 x 128 block
// ([ceil(n / 128), k / 128]); fmt 3: INT8 with a scale per 64 weights of a row ([n, k / 64]).
template <int FMT>
__global__ void expand_kernel(const uint8_t* w, const float* scale, int n, int k, bf16* out) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * k / 8) return;
    const int row = (int)(i / (k / 8)), c = (int)(i % (k / 8)) * 8;
    const uint2 raw = __ldg(reinterpret_cast<const uint2*>(w + (size_t)row * k + c));
    const float s = FMT == 5 ? __ldg(scale + (size_t)(row >> 7) * (k >> 7) + (c >> 7)) : __ldg(scale + (size_t)row * (k / 64) + c / 64);
    uint32_t packed[4];
    if constexpr (FMT == 5) {
        const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(p[j], __NV_E4M3);
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
            const __nv_bfloat162 v = __floats2bfloat162_rn(f.x * s, f.y * s);
            packed[j] = *reinterpret_cast<const uint32_t*>(&v);
        }
    } else {
        const int8_t* p = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __nv_bfloat162 v = __floats2bfloat162_rn((float)p[2 * j] * s, (float)p[2 * j + 1] * s);
            packed[j] = *reinterpret_cast<const uint32_t*>(&v);
        }
    }
    *reinterpret_cast<uint4*>(out + (size_t)row * k + c) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
}

}  // namespace

// out BF16 [n, k] = the weights of an FP8-block (fmt 5) or INT8-group (fmt 3) projection as the GEMM stages them.
extern "C" cudaError_t quench6k_nem_expand_bf16(int fmt, const void* w, const float* scale, int n, int k, void* out, cudaStream_t st) {
    if (k % 128 != 0 || (fmt != 3 && fmt != 5)) return cudaErrorInvalidValue;
    const size_t threads = (size_t)n * k / 8;
    if (threads == 0) return cudaSuccess;
    const unsigned blocks = (unsigned)((threads + 255) / 256);
    if (fmt == 5) SQ_CHECK_LAUNCH(sq_launch(expand_kernel<5>, blocks, 256, 0, st, static_cast<const uint8_t*>(w), scale, n, k, static_cast<bf16*>(out)));
    else SQ_CHECK_LAUNCH(sq_launch(expand_kernel<3>, blocks, 256, 0, st, static_cast<const uint8_t*>(w), scale, n, k, static_cast<bf16*>(out)));
    SQ_RETURN_LAUNCH();
}

// ---- Prefill projection GEMM with larger CTA tiles --------------------------------------------------------
// Y[M, N] (= or +=) alpha X[M, K] . W[N, K]^T, FP32 accumulation, weight formats BF16 (0), INT8 with an FP32
// scale per 64 weights of a row (3) or FP8 E4M3 with a scale per 128 x 128 block (5). Every output is the
// gemm.cu kernel's: the same BF16 operands (8-bit weights converted with gemm.cu's arithmetic, bf16(code x
// scale) rounded to nearest), the same m16n8k16 MMAs in ascending k from a zero accumulator, the same
// epilogue; only the CTA tile (L2 traffic per MMA, waves) and where the 8-bit weights are converted differ.
namespace {

constexpr int DG_BK = 32;
// BF16 tiles: 16-byte chunk c / 8 of row r at c ^ ((r >> 1) & 3) (gemm.cu's layout).
SQ_DEVICE int dg_sw(int r, int c) { return r * DG_BK + ((((c >> 3) ^ ((r >> 1) & 3))) << 3); }
// 8-bit tiles (32 bytes per row): 16-byte chunk c of row r at c ^ ((r >> 2) & 1), so a warp's fragment reads
// (rows g = 0..7, bytes 2 t .. 2 t + 1 and 8 + 2 t ..) hit 32 distinct banks.
SQ_DEVICE int dg_sw8(int r, int c) { return r * 32 + ((c ^ ((r >> 2) & 1)) << 4); }

SQ_DEVICE void dg_cp4(void* smem, const void* gmem, bool valid) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(valid ? 4 : 0));
}

// Bytes per stage: A BF16 [BM][32]; B BF16 [BN][32], or 8-bit [BN][32] plus a scale per row.
constexpr size_t dg_stage(int wf, int bm, int bn) { return (size_t)bm * DG_BK * 2 + (wf == 0 ? (size_t)bn * DG_BK * 2 : (size_t)bn * (DG_BK + 4)); }
constexpr size_t dg_smem(int wf, int bm, int bn, int stages) { return (size_t)stages * dg_stage(wf, bm, bn); }

// bf16x2 of the two 8-bit weights in `v` (lower k in the low half) times s, as gemm.cu stages them.
template <int WF>
SQ_DEVICE uint32_t dg_cvt(unsigned short v, float s) {
    __nv_bfloat162 r;
    if constexpr (WF == 5) {
        const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(v, __NV_E4M3);
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
        r = __floats2bfloat162_rn(f.x * s, f.y * s);
    } else {
        r = __floats2bfloat162_rn((float)(int8_t)(v & 0xff) * s, (float)(int8_t)(v >> 8) * s);
    }
    return *reinterpret_cast<const uint32_t*>(&r);
}

// BM x BN CTA tile, WM x WN warps of (BM / WM) x (BN / WN), STAGES cp.async stages. 8-bit weights are staged
// raw (with each row's scale for the k-tile) and converted into the B fragments by each warp.
template <int WF, int BM, int BN, int WM, int WN, int STAGES, int MINB = 1>
__global__ void __launch_bounds__(WM * WN * 32, MINB)
    dense_kernel(const bf16* __restrict__ X, int ldx, const void* __restrict__ Wv, int ldw, const float* __restrict__ gscale, float* Y, int ldy,
                 int M, int N, int K, float alpha, bool accumulate) {
    constexpr int THREADS = WM * WN * 32, TM = BM / WM, TN = BN / WN, MI = TM / 16, NJ = TN / 8;
    constexpr bool EIGHT = WF == 3 || WF == 5;
    static_assert(TM % 16 == 0 && TN % 16 == 0, "warp tile");
    constexpr size_t STAGE = dg_stage(WF, BM, BN);
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ __align__(16) unsigned char dg_smem_raw[];
    auto sa = [&](int st) { return reinterpret_cast<bf16*>(dg_smem_raw + st * STAGE); };
    auto sb = [&](int st) { return dg_smem_raw + st * STAGE + (size_t)BM * DG_BK * 2; };
    auto ss = [&](int st) { return reinterpret_cast<float*>(dg_smem_raw + st * STAGE + (size_t)BM * DG_BK * 2 + (size_t)BN * DG_BK); };
    const bf16* W = static_cast<const bf16*>(Wv);
    const uint8_t* W8 = static_cast<const uint8_t*>(Wv);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp / WN, warp_n = warp % WN;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int k_tiles = K / DG_BK;

    auto load = [&](int stage, int kt) {
        const int k = kt * DG_BK;
#pragma unroll
        for (int i = 0; i < (BM * 4 + THREADS - 1) / THREADS; ++i) {
            const int idx = tid + i * THREADS;
            if ((BM * 4) % THREADS == 0 || idx < BM * 4) {
                const int r = idx >> 2, c = (idx & 3) * 8, gm = m0 + r;
                const bool ok = gm < M;
                cp_async16(&sa(stage)[dg_sw(r, c)], X + (size_t)(ok ? gm : 0) * ldx + k + c, ok);
            }
        }
        if constexpr (!EIGHT) {
#pragma unroll
            for (int i = 0; i < (BN * 4 + THREADS - 1) / THREADS; ++i) {
                const int idx = tid + i * THREADS;
                if ((BN * 4) % THREADS == 0 || idx < BN * 4) {
                    const int r = idx >> 2, c = (idx & 3) * 8, gn = n0 + r;
                    const bool ok = gn < N;
                    cp_async16(reinterpret_cast<bf16*>(sb(stage)) + dg_sw(r, c), W + (size_t)(ok ? gn : 0) * ldw + k + c, ok);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < (BN * 2 + THREADS - 1) / THREADS; ++i) {
                const int idx = tid + i * THREADS;
                if ((BN * 2) % THREADS == 0 || idx < BN * 2) {
                    const int r = idx >> 1, c = idx & 1, gn = n0 + r;
                    const bool ok = gn < N;
                    cp_async16(sb(stage) + dg_sw8(r, c), W8 + (size_t)(ok ? gn : 0) * ldw + k + c * 16, ok);
                }
            }
#pragma unroll
            for (int i = 0; i < (BN + THREADS - 1) / THREADS; ++i) {
                const int r = tid + i * THREADS;
                if (BN % THREADS == 0 || r < BN) {
                    const int gn = n0 + r;
                    const bool ok = gn < N;
                    const size_t si = WF == 5 ? (size_t)((ok ? gn : 0) >> 7) * (K >> 7) + (k >> 7) : (size_t)(ok ? gn : 0) * (K / 64) + k / 64;
                    dg_cp4(&ss(stage)[r], gscale + si, ok);
                }
            }
        }
    };

    float acc[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < k_tiles) load(s, s);
        cp_async_commit();
    }

    const int g = lane >> 2, t4 = lane & 3;
    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int next = kt + STAGES - 1;
        if (next < k_tiles) load(next % STAGES, next);
        cp_async_commit();

        const int cur = kt % STAGES;
        const bf16* a_tile = sa(cur);
        float bsc[NJ];
        if constexpr (EIGHT) {
#pragma unroll
            for (int j = 0; j < NJ; ++j) bsc[j] = ss(cur)[warp_n * TN + j * 8 + g];
        }
#pragma unroll
        for (int kk = 0; kk < DG_BK; kk += 16) {
            uint32_t b[NJ][2];
            if constexpr (EIGHT) {
                const unsigned char* bt = sb(cur);
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const int r = warp_n * TN + j * 8 + g;
                    const unsigned char* row = bt + dg_sw8(r, kk >> 4) + 2 * t4;
                    b[j][0] = dg_cvt<WF>(*reinterpret_cast<const unsigned short*>(row), bsc[j]);
                    b[j][1] = dg_cvt<WF>(*reinterpret_cast<const unsigned short*>(row + 8), bsc[j]);
                }
            } else {
                const bf16* b_tile = reinterpret_cast<const bf16*>(sb(cur));
#pragma unroll
                for (int j = 0; j < NJ / 2; ++j) {
                    const int r = warp_n * TN + j * 16 + (lane & 7) + ((lane >> 4) << 3);
                    const int c = kk + ((lane >> 3) & 1) * 8;
                    uint32_t q[4];
                    ldmatrix_x4(q, &b_tile[dg_sw(r, c)]);
                    b[2 * j][0] = q[0];
                    b[2 * j][1] = q[1];
                    b[2 * j + 1][0] = q[2];
                    b[2 * j + 1][1] = q[3];
                }
            }
#pragma unroll
            for (int i = 0; i < MI; ++i) {
                uint32_t a[4];
                const int r = warp_m * TM + i * 16 + (lane & 15);
                const int c = kk + (lane >> 4) * 8;
                ldmatrix_x4(a, &a_tile[dg_sw(r, c)]);
#pragma unroll
                for (int j = 0; j < NJ; ++j) mma_bf16(acc[i][j], a, b[j]);
            }
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = m0 + warp_m * TM + i * 16 + g + h * 8;
                const int n = n0 + warp_n * TN + j * 8 + t4 * 2;
                if (m >= M) continue;
                const float s0 = 1.0f, s1 = 1.0f;
                float* p = &Y[(size_t)m * ldy + n];
                if (n < N) {
                    const float v = alpha * s0 * acc[i][j][h * 2];
                    p[0] = accumulate ? p[0] + v : v;
                }
                if (n + 1 < N) {
                    const float v = alpha * s1 * acc[i][j][h * 2 + 1];
                    p[1] = accumulate ? p[1] + v : v;
                }
            }
}

template <int WF, int BM, int BN, int WM, int WN, int ST, int MINB = 1>
cudaError_t dg_go(const bf16* x, int ldx, const void* w, int ldw, const float* sc, float* y, int ldy, int M, int N, int K, float alpha, bool acc,
                  cudaStream_t st) {
    constexpr size_t smem = dg_smem(WF, BM, BN, ST);
    static_assert(smem <= 99 * 1024, "shared memory");
    static bool configured = false;
    if (!configured) {
        const cudaError_t e = cudaFuncSetAttribute(dense_kernel<WF, BM, BN, WM, WN, ST, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) return e;
        configured = true;
    }
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    return sq_launch(dense_kernel<WF, BM, BN, WM, WN, ST, MINB>, grid, WM * WN * 32, smem, st, x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc);
}

template <int WF>
cudaError_t dg_variant(int v, const bf16* x, int ldx, const void* w, int ldw, const float* sc, float* y, int ldy, int M, int N, int K,
                       float alpha, bool acc, cudaStream_t st) {
    switch (v) {
        case 1: return dg_go<WF, 256, 128, 4, 2, 3>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 2: return dg_go<WF, 256, 128, 4, 2, 4>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 3: return dg_go<WF, 256, 192, 2, 4, 3>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 4:
            if constexpr (WF == 0) return dg_go<WF, 192, 256, 2, 4, 3>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
            else return dg_go<WF, 256, 192, 2, 4, 4>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 5: return dg_go<WF, 128, 256, 2, 4, 3>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 6: return dg_go<WF, 128, 256, 2, 4, 4>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 7: return dg_go<WF, 128, 192, 2, 4, WF == 0 ? 2 : 3, 2>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 8: return dg_go<WF, 64, 128, 2, 4, 3, 2>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 9: return dg_go<WF, 128, 128, 2, 4, 3, 2>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 10: return dg_go<WF, 64, 64, 2, 2, 3, 4>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 11: return dg_go<WF, 128, 128, 2, 4, 4, 2>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 12: return dg_go<WF, 128, 128, 2, 4, 6>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 13: return dg_go<WF, 64, 128, 2, 4, 6>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 14: return dg_go<WF, 64, 192, 2, 4, 6>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 15: return dg_go<WF, 64, 64, 2, 2, 6, 2>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 16: return dg_go<WF, 128, 192, 2, 4, 4>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        case 17: return dg_go<WF, 64, 256, 2, 4, WF == 0 ? 4 : 5>(x, ldx, w, ldw, sc, y, ldy, M, N, K, alpha, acc, st);
        default: return cudaErrorInvalidValue;
    }
}

// The fastest tile per projection shape, weight format and row count (0: gemm.cu's launch), from
// interleaved cold-weight timings at 600 W of every variant against gemm.cu's launch (dev/KBENCH.md,
// 2026-11-26 dense block): rows up to 256, 512, 1024, 1536, 2048, 4096, 6144 and beyond. A variant is
// listed where it measured at least 1% faster. BF16 below 2048 rows: only the router (the other
// projections are 8-bit then); 8-bit from 2048 rows: only the in_proj (the others are expanded to BF16
// then, faster than any 8-bit tile; their rows here serve nem.expand=0).
constexpr int DG_GRID[8] = {256, 512, 1024, 1536, 2048, 4096, 6144, 8192};
struct DgRow {
    int n, k, fmt;
    signed char v[8];
};
constexpr DgRow DG_TABLE[] = {
    {18560, 4096, 5, { 1,  1,  0,  0,  0,  0,  0,  3}},  // in_proj
    {18560, 4096, 3, { 2,  1,  9,  2,  4,  4,  4,  4}},  // in_proj
    {4096, 8192, 5, {15, 14, 16,  3,  3,  3,  3,  3}},  // out_proj
    {4096, 8192, 3, {15, 14, 16,  4,  4,  4,  4,  4}},  // out_proj
    {4096, 8192, 0, { 0,  0,  0,  0,  4,  4,  4,  4}},  // out_proj
    {5376, 4096, 5, {13, 11,  2,  3,  0,  3,  3,  3}},  // sh_up
    {5376, 4096, 3, {15, 12,  2,  4,  2,  2,  4,  4}},  // sh_up
    {5376, 4096, 0, { 0,  0,  0,  0,  2,  2,  4,  4}},  // sh_up
    {4096, 5376, 5, {15, 14, 16,  3,  3,  4,  4,  3}},  // sh_down
    {4096, 5376, 3, {15, 14, 16,  4,  4,  4,  4,  4}},  // sh_down
    {4096, 5376, 0, { 0,  0,  0,  0,  3,  4,  4,  4}},  // sh_down
    {4608, 4096, 5, {15, 11,  0,  3,  1,  0,  0,  0}},  // qkv
    {4608, 4096, 3, {15, 12,  2,  4,  2,  9,  5,  4}},  // qkv
    {4608, 4096, 0, { 0,  0,  0,  0,  2,  9,  2,  2}},  // qkv
    {4096, 4096, 5, {15, 14, 16,  3,  3,  3,  3,  3}},  // o
    {4096, 4096, 3, {15, 14, 16,  4,  4,  4,  4,  4}},  // o
    {4096, 4096, 0, { 0,  0,  0,  0,  3,  4,  4,  3}},  // o
    {1024, 4096, 5, {10, 15, 15, 14,  0,  1,  3,  9}},  // fc1
    {1024, 4096, 3, {10, 15, 15, 14,  0,  1,  4,  9}},  // fc1
    {1024, 4096, 0, { 0,  0,  0,  0, 11,  9,  4,  4}},  // fc1
    {4096, 1024, 5, {15, 14, 16,  0,  3,  3,  3,  3}},  // fc2
    {4096, 1024, 3, {15, 14, 16,  4,  4,  4,  4,  4}},  // fc2
    {4096, 1024, 0, { 0,  0,  0,  0,  4,  4,  4,  4}},  // fc2
    {512, 4096, 0, {10, 10, 15, 13, 13, 11, 16,  9}},  // router
};

int dg_pick(int fmt, int M, int N, int K) {
    int g = 0;
    while (g < 7 && M > DG_GRID[g]) ++g;
    for (const DgRow& r : DG_TABLE)
        if (r.n == N && r.k == K && r.fmt == fmt) return r.v[g];
    return 0;
}

}  // namespace

extern "C" cudaError_t quench6k_gemm_bf16_f32_ext(int fmt, const void* X, int ldx, const void* W, int ldw, const float* scale, float* Y,
                                                int ldy, int M, int N, int K, float alpha, int accumulate, int flags, cudaStream_t stream);

// Prefill projection Y[M, N] (= or +=) alpha X . W^T with the tile variant `v` (see dg_variant), -1: the
// table's for this shape and row count, 0: gemm.cu's launch (quench6k_gemm_bf16_f32_ext, flags 2). Every
// variant gives that launch's bits (fmt 0, 3 and 5).
extern "C" cudaError_t quench6k_nem_gemm(int fmt, const void* X, int ldx, const void* W, int ldw, const float* scale, float* Y, int ldy, int M,
                                         int N, int K, float alpha, int accumulate, int v, cudaStream_t st) {
    if (K % DG_BK != 0 || ldx % 8 != 0 || ldw % 16 != 0 || (fmt != 0 && K % 128 != 0)) return cudaErrorInvalidValue;
    if (M == 0 || N == 0) return cudaSuccess;
    if (v < 0) v = dg_pick(fmt, M, N, K);
    if (v == 0) return quench6k_gemm_bf16_f32_ext(fmt, X, ldx, W, ldw, scale, Y, ldy, M, N, K, alpha, accumulate, 2, st);
    const bf16* x = static_cast<const bf16*>(X);
    if (fmt == 0) SQ_CHECK_LAUNCH(dg_variant<0>(v, x, ldx, W, ldw, nullptr, Y, ldy, M, N, K, alpha, accumulate != 0, st));
    else if (fmt == 3) SQ_CHECK_LAUNCH(dg_variant<3>(v, x, ldx, W, ldw, scale, Y, ldy, M, N, K, alpha, accumulate != 0, st));
    else if (fmt == 5) SQ_CHECK_LAUNCH(dg_variant<5>(v, x, ldx, W, ldw, scale, Y, ldy, M, N, K, alpha, accumulate != 0, st));
    else return cudaErrorInvalidValue;
    SQ_RETURN_LAUNCH();
}

namespace {

__global__ void count_diff_kernel(const uint32_t* a, const uint32_t* b, size_t n, unsigned long long* out) {
    unsigned long long c = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) c += a[i] != b[i];
    c = __reduce_add_sync(0xffffffffu, (unsigned)c);
    if ((threadIdx.x & 31) == 0 && c) atomicAdd(out, c);
}

}  // namespace

// Tests: *out = the number of 32-bit words at which a [n] and b [n] differ.
extern "C" cudaError_t quench6k_nem_count_diff(const void* a, const void* b, size_t n, unsigned long long* out, cudaStream_t st) {
    SQ_CHECK_LAUNCH(cudaMemsetAsync(out, 0, sizeof(*out), st));
    SQ_CHECK_LAUNCH(sq_launch(count_diff_kernel, 512, 256, 0, st, static_cast<const uint32_t*>(a), static_cast<const uint32_t*>(b), n, out));
    SQ_RETURN_LAUNCH();
}
