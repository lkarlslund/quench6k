// BF16 tensor-core GEMM: Y[M, N] (= or +=) alpha * X[M, K] . W[N, K]^T, FP32 accumulation.
//
// Both operands are K-contiguous (activations row-major, weights in PyTorch Linear layout), which
// is exactly mma.sync's row.col form. 128x128x32 CTA tiles, 8 warps of 64x32, three cp.async stages
// (dynamic shared memory).
// K must be a multiple of 32 and rows 16-byte aligned; M and N are arbitrary.

#include <cuda_fp8.h>

#include "mma.cuh"

namespace {

constexpr int BM = 128, BN = 128, BK = 32, THREADS = 256;
constexpr int LDS = BK + 8;  // padded smem row (elements): conflict-free ldmatrix
// Three stages by default; two (41 KB, two CTAs per SM) when a launch has between one and two CTAs per
// SM, so that it runs as one wave instead of a full wave plus a small remainder.
constexpr size_t gemm_smem_bytes(int stages, int bm = BM, int bn = BN) { return (size_t)stages * (bm + bn) * LDS * 2; }
constexpr size_t GEMM_SMEM = gemm_smem_bytes(3);

template <typename OutT>
SQ_DEVICE void store_out(OutT* p, float v, bool accumulate);
template <>
SQ_DEVICE void store_out<float>(float* p, float v, bool accumulate) { *p = accumulate ? *p + v : v; }
template <>
SQ_DEVICE void store_out<__nv_bfloat16>(__nv_bfloat16* p, float v, bool accumulate) {
    *p = f2bf(accumulate ? bf2f(*p) + v : v);
}

// WF (weight format): 0 BF16; 1 FP8 E4M3 or 2 INT8 [N, K] (ldw in elements = bytes) with an FP32 scale
// per row, converted exactly to BF16 while staging the tile, the row scale applied in the epilogue;
// 3 INT8 with an FP32 scale per 64 weights of a row ([N, K / 64]), applied while staging.
// HyperConnection up-projection with the mix fused into the epilogue (MIX): the CTA's 128 columns are
// 32 features d0 .. d0 + 31 of each of the 4 streams (warp_n = stream), so
//   m[t, d] = mean_s sigmoid(u[t, s, d]) * n[t, s, d] * (1 + g[s, d])
// is reduced in shared memory (streams in order, as hc_mix) without writing u.
constexpr int HC_D = 2560, HC_S = 4;

struct MixArgs {
    const __nv_bfloat16* n;
    const __nv_bfloat16* g;
    __nv_bfloat16* m;
};

// BMT x BNT: CTA tile, 128 x 128 (warps 2 x 4 of 64 x 32), 256 x 128 (warps 4 x 2 of 64 x 64: each
// staged weight tile serves twice the rows, fewer fragment loads per MMA) or 128 x 256.
template <typename OutT, int WF, bool MIX = false, int STAGES = 3, int BNT = 128, int BMT = 128>
__global__ void __launch_bounds__(THREADS, STAGES == 2 ? 2 : 1) gemm_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
                                                       const void* __restrict__ Wv, int ldw,
                                                       const float* __restrict__ row_scale, OutT* Y, int ldy,
                                                       int M, int N, int K, float alpha, bool accumulate,
                                                       MixArgs mix = {}) {
    pdl_launch_dependents();
    pdl_wait();
    const __nv_bfloat16* W = static_cast<const __nv_bfloat16*>(Wv);
    const uint8_t* W8 = static_cast<const uint8_t*>(Wv);
    extern __shared__ __align__(16) unsigned char gemm_smem[];
    auto sA = reinterpret_cast<__nv_bfloat16 (*)[BMT * LDS]>(gemm_smem);
    static_assert((BNT == 128 && BMT == 128) || (!MIX && (WF == 0 || WF == 3) && BNT * BMT == 128 * 256), "unsupported tile");
    constexpr int WM = BMT / 64, WN = 8 / WM, WTN = BNT / WN;  // warp grid; warp tile 64 x WTN
    auto sB = reinterpret_cast<__nv_bfloat16 (*)[BNT * LDS]>(gemm_smem + (size_t)STAGES * BMT * LDS * 2);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp / WN, warp_n = warp % WN;
    const int m0 = blockIdx.y * BMT, n0 = blockIdx.x * BNT;
    const int k_tiles = K / BK;
    // Global weight row of the tile's local column r.
    auto wrow = [&](int r) { return MIX ? (r >> 5) * HC_D + blockIdx.x * 32 + (r & 31) : n0 + r; };

    auto load_a = [&](int stage, int kt) {  // A: 128 rows x 4 chunks of 8 elements
        const int k = kt * BK;
#pragma unroll
        for (int i = 0; i < BMT * 4 / THREADS; ++i) {
            const int idx = tid + i * THREADS;
            const int r = idx >> 2, c = (idx & 3) * 8;
            const int gm = m0 + r;
            const bool ok = gm < M;
            cp_async16(&sA[stage][r * LDS + c], X + (size_t)(ok ? gm : 0) * ldx + k + c, ok);
        }
    };
    auto load = [&](int stage, int kt) {
        const int k = kt * BK;
        load_a(stage, kt);
        if constexpr (WF == 0) {
#pragma unroll
            for (int i = 0; i < BNT * 4 / THREADS; ++i) {  // B: BNT rows x 4 chunks
                const int idx = tid + i * THREADS;
                const int r = idx >> 2, c = (idx & 3) * 8;
                const int gn = wrow(r);
                const bool ok = gn < N;
                cp_async16(&sB[stage][r * LDS + c], W + (size_t)(ok ? gn : 0) * ldw + k + c, ok);
            }
        } else {  // B: 128 rows x 2 chunks of 16 8-bit values, converted to BF16
            const int r = tid >> 1, c = (tid & 1) * 16;
            const int gn = wrow(r);
            __nv_bfloat16* dst = &sB[stage][r * LDS + c];
            if (gn < N) {
                const uint4 v = __ldg(reinterpret_cast<const uint4*>(W8 + (size_t)gn * ldw + k + c));
                if constexpr (WF == 1) {
                    const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&v);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3);
                        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
                        dst[2 * i] = f2bf(f.x);
                        dst[2 * i + 1] = f2bf(f.y);
                    }
                } else {
                    const int8_t* p = reinterpret_cast<const int8_t*>(&v);
                    const float gs = WF == 3 ? row_scale[(size_t)gn * (K / 64) + (k + c) / 64] : 1.0f;
#pragma unroll
                    for (int i = 0; i < 16; ++i) dst[i] = f2bf((float)p[i] * gs);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) dst[i] = f2bf(0.0f);
            }
        }
    };

    constexpr int MI = 4;         // 16-row MMA tiles per warp
    constexpr int NJ = WTN / 8;   // 8-column MMA tiles per warp
    float acc[MI][NJ][4] = {};

    // 8-bit weights: the raw bytes (and group scale) of the tile STAGES - 1 ahead are fetched into
    // registers before this tile's MMAs and converted into shared memory after them. Each thread
    // takes BPT consecutive bytes of one row.
    constexpr int BPT = BNT * BK / THREADS, NU = BPT / 16;
    uint4 braw[NU];
#pragma unroll
    for (int u = 0; u < NU; ++u) braw[u] = make_uint4(0, 0, 0, 0);
    float bgs = 1.0f;
    const int br = tid / (BK / BPT), bc = (tid % (BK / BPT)) * BPT, bgn = wrow(br);
    auto fetch_b = [&](int kt) {
        if (bgn < N) {
            const int k = kt * BK;
#pragma unroll
            for (int u = 0; u < NU; ++u) braw[u] = __ldg(reinterpret_cast<const uint4*>(W8 + (size_t)bgn * ldw + k + bc + 16 * u));
            if constexpr (WF == 3) bgs = row_scale[(size_t)bgn * (K / 64) + (k + bc) / 64];
        }
    };
    auto store_b = [&](int stage) {
#pragma unroll
        for (int u = 0; u < NU; ++u) {
            __nv_bfloat16* dst = &sB[stage][br * LDS + bc + 16 * u];
            if (bgn < N) {
                const int8_t* p = reinterpret_cast<const int8_t*>(&braw[u]);
                uint32_t packed[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const __nv_bfloat162 v = __floats2bfloat162_rn((float)p[2 * i] * bgs, (float)p[2 * i + 1] * bgs);
                    packed[i] = *reinterpret_cast<const uint32_t*>(&v);
                }
                *reinterpret_cast<uint4*>(dst) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
                *reinterpret_cast<uint4*>(dst + 8) = make_uint4(packed[4], packed[5], packed[6], packed[7]);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0, 0, 0, 0);
            }
        }
    };
    constexpr bool PIPE_B = WF == 2 || WF == 3;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < k_tiles) {
            if constexpr (PIPE_B) {
                load_a(s, s);
                fetch_b(s);
                store_b(s);
            } else {
                load(s, s);
            }
        }
        cp_async_commit();
    }

    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int next = kt + STAGES - 1;
        if (next < k_tiles) {
            if constexpr (PIPE_B) {
                load_a(next % STAGES, next);
                fetch_b(next);
            } else {
                load(next % STAGES, next);
            }
        }
        cp_async_commit();

        const __nv_bfloat16* a_tile = sA[kt % STAGES];
        const __nv_bfloat16* b_tile = sB[kt % STAGES];
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[MI][4], b[NJ / 2][4];
#pragma unroll
            for (int i = 0; i < MI; ++i) {
                const int r = warp_m * 64 + i * 16 + (lane & 15);
                const int c = kk + (lane >> 4) * 8;
                ldmatrix_x4(a[i], &a_tile[r * LDS + c]);
            }
#pragma unroll
            for (int j = 0; j < NJ / 2; ++j) {
                const int r = warp_n * WTN + j * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int c = kk + ((lane >> 3) & 1) * 8;
                ldmatrix_x4(b[j], &b_tile[r * LDS + c]);
            }
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NJ; ++j) mma_bf16(acc[i][j], a[i], &b[j >> 1][(j & 1) * 2]);
        }
        // Stage `next` was last read by tile kt - 1's MMAs, finished before this tile's barrier.
        if constexpr (PIPE_B)
            if (next < k_tiles) store_b(next % STAGES);
    }
    cp_async_wait<0>();

    const int g = lane >> 2, t4 = lane & 3;
    if constexpr (MIX) {
        // sigmoid(u) (u as the plain epilogue would store it) for 64 rows at a time into shared memory
        // [stream][row][feature] (rows padded to 33), then per (row, feature) the stream-ordered mix sum
        // with coalesced reads of n.
        __syncthreads();
        constexpr int RS = 33;
        float* sg = reinterpret_cast<float*>(gemm_smem);  // [4][64][33]
        const int d0 = blockIdx.x * 32;
        for (int half = 0; half < 2; ++half) {
            if (warp_m == half) {
#pragma unroll
                for (int i = 0; i < MI; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
#pragma unroll
                        for (int h = 0; h < 2; ++h)
#pragma unroll
                            for (int e = 0; e < 2; ++e) {
                                const int rl = i * 16 + g + h * 8, dl = j * 8 + t4 * 2 + e;
                                sg[(warp_n * 64 + rl) * RS + dl] = sigmoidf(alpha * 1.0f * acc[i][j][h * 2 + e]);
                            }
            }
            __syncthreads();
            for (int i = tid; i < 64 * 32; i += THREADS) {
                const int rl = i >> 5, dl = i & 31, m = m0 + half * 64 + rl, d = d0 + dl;
                if (m >= M) continue;
                float a = 0.0f;
#pragma unroll
                for (int st = 0; st < HC_S; ++st) {
                    const float nv = bf2f(mix.n[(size_t)m * (HC_S * HC_D) + st * HC_D + d]);
                    a = __fadd_rn(a, __fmul_rn(__fmul_rn(sg[(st * 64 + rl) * RS + dl], nv), mix.g ? 1.0f + bf2f(mix.g[st * HC_D + d]) : 1.0f));
                }
                mix.m[(size_t)m * HC_D + d] = f2bf(a / HC_S);
            }
            __syncthreads();
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = m0 + warp_m * 64 + i * 16 + g + h * 8;
                const int n = n0 + warp_n * WTN + j * 8 + t4 * 2;
                if (m >= M) continue;
                const float s0 = (WF == 1 || WF == 2) && n < N ? row_scale[n] : 1.0f;
                const float s1 = (WF == 1 || WF == 2) && n + 1 < N ? row_scale[n + 1] : 1.0f;
                if (n < N) store_out(&Y[(size_t)m * ldy + n], alpha * s0 * acc[i][j][h * 2], accumulate);
                if (n + 1 < N) store_out(&Y[(size_t)m * ldy + n + 1], alpha * s1 * acc[i][j][h * 2 + 1], accumulate);
            }
}

template <typename OutT>
cudaError_t launch(int fmt, const void* X, int ldx, const void* W, int ldw, const float* scale, void* Y, int ldy, int M,
                   int N, int K, float alpha, int accumulate, cudaStream_t stream) {
    if (K % BK != 0 || ldx % 8 != 0 || ldw % 16 != 0) return cudaErrorInvalidValue;
    if (M == 0 || N == 0) return cudaSuccess;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(X);
    OutT* y = static_cast<OutT*>(Y);
    static bool configured = false;
    static int sms = 0;
    if (!configured) {
        for (auto kern : {gemm_kernel<OutT, 0>, gemm_kernel<OutT, 3>}) {
            const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GEMM_SMEM);
            if (e != cudaSuccess) return e;
        }
        for (auto kern : {gemm_kernel<OutT, 0, false, 2>, gemm_kernel<OutT, 3, false, 2>}) {
            const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gemm_smem_bytes(2));
            if (e != cudaSuccess) return e;
        }
        for (auto kern : {gemm_kernel<OutT, 0, false, 3, 128, 256>, gemm_kernel<OutT, 3, false, 3, 128, 256>}) {
            const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gemm_smem_bytes(3, 256, 128));
            if (e != cudaSuccess) return e;
        }
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        configured = true;
    }
    const int ctas = grid.x * grid.y;
    const bool two = ctas > sms && ctas <= 2 * sms;
    // 256-row tiles when there are enough of them for several waves.
    const dim3 grid_w((N + BN - 1) / BN, (M + 255) / 256);
    const bool wide = !two && (int)(grid_w.x * grid_w.y) >= 4 * sms;
    const float* sc = fmt == 3 ? scale : nullptr;
    auto go = [&](auto kern, size_t smem, dim3 g) {
        return sq_launch(kern, g, THREADS, smem, stream, x, ldx, W, ldw, sc, y, ldy, M, N, K, alpha, accumulate != 0, MixArgs{});
    };
    const size_t wide_smem = gemm_smem_bytes(3, 256, 128);
    if (fmt == 3)
        SQ_CHECK_LAUNCH(wide ? go(gemm_kernel<OutT, 3, false, 3, 128, 256>, wide_smem, grid_w)
                        : two ? go(gemm_kernel<OutT, 3, false, 2>, gemm_smem_bytes(2), grid) : go(gemm_kernel<OutT, 3>, GEMM_SMEM, grid));
    else if (fmt == 0)
        SQ_CHECK_LAUNCH(wide ? go(gemm_kernel<OutT, 0, false, 3, 128, 256>, wide_smem, grid_w)
                        : two ? go(gemm_kernel<OutT, 0, false, 2>, gemm_smem_bytes(2), grid) : go(gemm_kernel<OutT, 0>, GEMM_SMEM, grid));
    else
        return cudaErrorInvalidValue;
    SQ_RETURN_LAUNCH();
}

}  // namespace

// HyperConnection up-projection (W [10240, K]) fused with the mix: m [M, 2560] BF16 from
// z [M, K] BF16, n [M, 10240] BF16 and the optional norm weight g [10240] (see MIX above).
extern "C" cudaError_t squash_gemm_hc_up_mix(int fmt, const void* X, int ldx, const void* W, int ldw, const float* scale,
                                             const void* n, const void* g, void* m, int M, int K, cudaStream_t stream) {
    if (K % BK != 0 || ldx % 8 != 0 || ldw % 16 != 0) return cudaErrorInvalidValue;
    if (M == 0) return cudaSuccess;
    static bool configured = false;
    if (!configured) {
        for (auto kern : {gemm_kernel<float, 0, true>, gemm_kernel<float, 3, true>}) {
            const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GEMM_SMEM);
            if (e != cudaSuccess) return e;
        }
        configured = true;
    }
    const MixArgs mix{static_cast<const __nv_bfloat16*>(n), static_cast<const __nv_bfloat16*>(g), static_cast<__nv_bfloat16*>(m)};
    const dim3 grid(HC_D / 32, (M + BM - 1) / BM);
    const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(X);
    if (fmt == 3)
        SQ_CHECK_LAUNCH(sq_launch(gemm_kernel<float, 3, true>, grid, THREADS, GEMM_SMEM, stream, x, ldx, W, ldw, scale, (float*)nullptr, 0, M,
                                  HC_S * HC_D, K, 1.0f, false, mix));
    else if (fmt == 0)
        SQ_CHECK_LAUNCH(sq_launch(gemm_kernel<float, 0, true>, grid, THREADS, GEMM_SMEM, stream, x, ldx, W, ldw, (const float*)nullptr,
                                  (float*)nullptr, 0, M, HC_S * HC_D, K, 1.0f, false, mix));
    else
        return cudaErrorInvalidValue;
    SQ_RETURN_LAUNCH();
}

// fmt 0: BF16 weights; 3: INT8 weights with an FP32 scale per 64 weights of a row.
extern "C" cudaError_t squash_gemm_bf16_f32(int fmt, const void* X, int ldx, const void* W, int ldw, const float* scale,
                                            float* Y, int ldy, int M, int N, int K, float alpha, int accumulate,
                                            cudaStream_t stream) {
    return launch<float>(fmt, X, ldx, W, ldw, scale, Y, ldy, M, N, K, alpha, accumulate, stream);
}

extern "C" cudaError_t squash_gemm_bf16_bf16(int fmt, const void* X, int ldx, const void* W, int ldw, const float* scale,
                                             void* Y, int ldy, int M, int N, int K, float alpha, int accumulate,
                                             cudaStream_t stream) {
    return launch<__nv_bfloat16>(fmt, X, ldx, W, ldw, scale, Y, ldy, M, N, K, alpha, accumulate, stream);
}
