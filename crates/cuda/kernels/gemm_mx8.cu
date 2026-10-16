// MXFP8 tensor-core GEMM: Y[M, N] = A[M, K] . W[N, K]^T with both operands FP8 E4M3 and a power-of-two
// (UE8M0) scale per 32 consecutive values of a row, applied by the MMA
// (mma.sync kind::mxf8f6f4.block_scale.scale_vec::1X m16n8k32, FP32 accumulation). Activations are
// quantized per row by squash_mxfp8_rows with the same rule as the weights:
//   scale = 2^ceil(log2(amax / 448)), value = e4m3(x / scale) (round to nearest, saturating).
//
// Tiles of 64 rows x 128 columns, k-tiles of 128 values (128 bytes per row, chunk c of row r stored at
// c ^ (r & 7)), three cp.async stages, 8 warps of 32 x 32.

#include <cuda_fp8.h>

#include <type_traits>

#include "mma.cuh"

namespace {

// One thread per 32-value block: x BF16 [rows, cols] (row stride ld) -> codes [rows, cols], scale bytes
// [rows, cols / 32].
__global__ void mxfp8_rows_kernel(const __nv_bfloat16* __restrict__ x, int rows, int cols, int ld, uint8_t* codes,
                                  uint8_t* scales) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const int per = cols / 32;
    if (i >= (size_t)rows * per) return;
    const size_t r = i / per, b = i % per;
    const __nv_bfloat16* p = x + r * ld + b * 32;
    float v[32], amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; j += 8) {
        const uint4 u = *reinterpret_cast<const uint4*>(p + j);
        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[j + e] = bf2f(h[e]);
            amax = fmaxf(amax, fabsf(v[j + e]));
        }
    }
    int ex = 0;
    float scale = 1.0f;
    if (amax > 0.0f) {
        ex = (int)ceilf(log2f(amax / 448.0f));
        scale = exp2f((float)ex);
    }
    scales[r * per + b] = (uint8_t)(ex + 127);
    uint32_t q[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        uint32_t w = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) w |= (uint32_t)__nv_fp8_e4m3(v[4 * j + e] / scale).__x << (8 * e);
        q[j] = w;
    }
    uint4* dst = reinterpret_cast<uint4*>(codes + r * cols + b * 32);
    dst[0] = make_uint4(q[0], q[1], q[2], q[3]);
    dst[1] = make_uint4(q[4], q[5], q[6], q[7]);
}

constexpr int BM = 128, BN = 128, BK = 128, STAGES = 3, THREADS = 256;
constexpr int MI = BM / 32;  // m16 tiles per warp (warps 2 x 4, warp tile BM / 2 x 32)

SQ_DEVICE int swz(int r, int c) { return r * BK + ((c ^ (r & 7)) << 4); }

struct __align__(16) Stage {
    uint8_t a[BM * BK];
    uint8_t b[BN * BK];
    uint32_t as[BM];  // the 4 scale bytes of the k-tile
    uint32_t bs[BN];
};

template <int S>
SQ_DEVICE void mma_mx8(float* c, const uint32_t* a, const uint32_t* b, uint32_t sa, uint32_t sb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, 0}, %11, {%12, 0};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb), "n"(S));
}

SQ_DEVICE void cp_async4(void* smem, const void* gmem, bool valid) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(valid ? 4 : 0));
}

__global__ void __launch_bounds__(THREADS) mx8_gemm_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ As,
                                                           const uint8_t* __restrict__ W, const uint8_t* __restrict__ Ws,
                                                           float* Y, int ldy, int M, int N, int K) {
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ __align__(16) uint8_t mx8_smem[];
    Stage* st = reinterpret_cast<Stage*>(mx8_smem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, warp_m = warp >> 2, warp_n = warp & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, k_tiles = K / BK, KS = K / 32;
    auto load = [&](int s, int kt) {
        Stage& g = st[s];
#pragma unroll
        for (int i = 0; i < BM * 8 / THREADS; ++i) {  // A: BM rows x 8 chunks
            const int idx = tid + i * THREADS, r = idx >> 3, c = idx & 7;
            const bool ok = m0 + r < M;
            cp_async16(&g.a[swz(r, c)], A + (size_t)(ok ? m0 + r : 0) * K + kt * BK + c * 16, ok);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {  // B: 128 rows x 8 chunks
            const int idx = tid + i * THREADS, r = idx >> 3, c = idx & 7;
            const bool ok = n0 + r < N;
            cp_async16(&g.b[swz(r, c)], W + (size_t)(ok ? n0 + r : 0) * K + kt * BK + c * 16, ok);
        }
        if (tid < BM) {
            const bool ok = m0 + tid < M;
            cp_async4(&g.as[tid], As + (size_t)(ok ? m0 + tid : 0) * KS + kt * 4, ok);
        } else if (tid < BM + BN) {
            const int r = tid - BM;
            const bool ok = n0 + r < N;
            cp_async4(&g.bs[r], Ws + (size_t)(ok ? n0 + r : 0) * KS + kt * 4, ok);
        }
    };
    float acc[MI][4][4] = {};
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < k_tiles) load(s, s);
        cp_async_commit();
    }
    const int ta = lane / 4 + (lane % 2) * 8, tb = lane / 4;
    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int next = kt + STAGES - 1;
        if (next < k_tiles) load(next % STAGES, next);
        cp_async_commit();
        const Stage& g = st[kt % STAGES];
        uint32_t sa[MI], sb[4];
#pragma unroll
        for (int i = 0; i < MI; ++i) sa[i] = g.as[warp_m * (BM / 2) + i * 16 + ta];
#pragma unroll
        for (int j = 0; j < 4; ++j) sb[j] = g.bs[warp_n * 32 + j * 8 + tb];
        auto step = [&](auto sc) {
            constexpr int s = decltype(sc)::value;
            uint32_t a[MI][4], b[4][2];
#pragma unroll
            for (int i = 0; i < MI; ++i) ldmatrix_x4(a[i], &g.a[swz(warp_m * (BM / 2) + i * 16 + (lane & 15), s * 2 + (lane >> 4))]);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                uint32_t r4[4];
                ldmatrix_x4(r4, &g.b[swz(warp_n * 32 + j * 16 + (lane & 7) + ((lane >> 4) << 3), s * 2 + ((lane >> 3) & 1))]);
                b[2 * j][0] = r4[0], b[2 * j][1] = r4[1], b[2 * j + 1][0] = r4[2], b[2 * j + 1][1] = r4[3];
            }
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_mx8<s>(acc[i][j], a[i], b[j], sa[i], sb[j]);
        };
        step(std::integral_constant<int, 0>{});
        step(std::integral_constant<int, 1>{});
        step(std::integral_constant<int, 2>{});
        step(std::integral_constant<int, 3>{});
    }
    cp_async_wait<0>();
    const int g8 = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int m = m0 + warp_m * (BM / 2) + i * 16 + g8 + h * 8;
            if (m >= M) continue;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int n = n0 + warp_n * 32 + j * 8 + 2 * t4;
                if (n < N) Y[(size_t)m * ldy + n] = acc[i][j][2 * h];
                if (n + 1 < N) Y[(size_t)m * ldy + n + 1] = acc[i][j][2 * h + 1];
            }
        }
}

}  // namespace

extern "C" cudaError_t squash_mxfp8_rows(const void* x, int rows, int cols, int ld, uint8_t* codes, uint8_t* scales, cudaStream_t st) {
    if (cols % 32 != 0 || ld % 8 != 0) return cudaErrorInvalidValue;
    const size_t n = (size_t)rows * (cols / 32);
    if (n == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(mxfp8_rows_kernel, (unsigned)((n + 255) / 256), 256, 0, st, static_cast<const __nv_bfloat16*>(x), rows, cols,
                              ld, codes, scales));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_gemm_mx8(const uint8_t* a, const uint8_t* as, const uint8_t* w, const uint8_t* ws, float* y, int ldy,
                                       int M, int N, int K, cudaStream_t st) {
    if (K % BK != 0) return cudaErrorInvalidValue;
    if (M == 0 || N == 0) return cudaSuccess;
    constexpr size_t smem = STAGES * sizeof(Stage);
    static bool configured = false;
    if (!configured) {
        const cudaError_t e = cudaFuncSetAttribute(mx8_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) return e;
        configured = true;
    }
    SQ_CHECK_LAUNCH(sq_launch(mx8_gemm_kernel, dim3((N + BN - 1) / BN, (M + BM - 1) / BM), THREADS, smem, st, a, as, w, ws, y, ldy, M, N, K));
    SQ_RETURN_LAUNCH();
}
