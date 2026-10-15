// BF16 tensor-core GEMM: Y[M, N] (= or +=) alpha * X[M, K] . W[N, K]^T, FP32 accumulation.
//
// Both operands are K-contiguous (activations row-major, weights in PyTorch Linear layout), which
// is exactly mma.sync's row.col form. 64x128x32 CTA tiles, 8 warps of 32x32, three cp.async stages.
// K must be a multiple of 32 and rows 16-byte aligned; M and N are arbitrary.

#include "mma.cuh"

namespace {

constexpr int BM = 64, BN = 128, BK = 32, STAGES = 3, THREADS = 256;
constexpr int LDS = BK + 8;  // padded smem row (elements): conflict-free ldmatrix

template <typename OutT>
SQ_DEVICE void store_out(OutT* p, float v, bool accumulate);
template <>
SQ_DEVICE void store_out<float>(float* p, float v, bool accumulate) { *p = accumulate ? *p + v : v; }
template <>
SQ_DEVICE void store_out<__nv_bfloat16>(__nv_bfloat16* p, float v, bool accumulate) {
    *p = f2bf(accumulate ? bf2f(*p) + v : v);
}

template <typename OutT>
__global__ void __launch_bounds__(THREADS) gemm_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
                                                       const __nv_bfloat16* __restrict__ W, int ldw, OutT* Y, int ldy,
                                                       int M, int N, int K, float alpha, bool accumulate) {
    __shared__ __align__(16) __nv_bfloat16 sA[STAGES][BM * LDS];
    __shared__ __align__(16) __nv_bfloat16 sB[STAGES][BN * LDS];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp >> 2, warp_n = warp & 3;  // 2 x 4 warps
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int k_tiles = K / BK;

    auto load = [&](int stage, int kt) {
        const int k = kt * BK;
        {  // A: 64 rows x 4 chunks of 8 elements
            const int r = tid >> 2, c = (tid & 3) * 8;
            const int gm = m0 + r;
            const bool ok = gm < M;
            cp_async16(&sA[stage][r * LDS + c], X + (size_t)(ok ? gm : 0) * ldx + k + c, ok);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {  // B: 128 rows x 4 chunks
            const int idx = tid + i * THREADS;
            const int r = idx >> 2, c = (idx & 3) * 8;
            const int gn = n0 + r;
            const bool ok = gn < N;
            cp_async16(&sB[stage][r * LDS + c], W + (size_t)(ok ? gn : 0) * ldw + k + c, ok);
        }
    };

    float acc[2][4][4] = {};

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < k_tiles) load(s, s);
        cp_async_commit();
    }

    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int next = kt + STAGES - 1;
        if (next < k_tiles) load(next % STAGES, next);
        cp_async_commit();

        const __nv_bfloat16* a_tile = sA[kt % STAGES];
        const __nv_bfloat16* b_tile = sB[kt % STAGES];
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[2][4], b[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = warp_m * 32 + i * 16 + (lane & 15);
                const int c = kk + (lane >> 4) * 8;
                ldmatrix_x4(a[i], &a_tile[r * LDS + c]);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int r = warp_n * 32 + j * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int c = kk + ((lane >> 3) & 1) * 8;
                ldmatrix_x4(b[j], &b_tile[r * LDS + c]);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_bf16(acc[i][j], a[i], &b[j >> 1][(j & 1) * 2]);
        }
    }
    cp_async_wait<0>();

    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = m0 + warp_m * 32 + i * 16 + g + h * 8;
                const int n = n0 + warp_n * 32 + j * 8 + t4 * 2;
                if (m >= M) continue;
                if (n < N) store_out(&Y[(size_t)m * ldy + n], alpha * acc[i][j][h * 2], accumulate);
                if (n + 1 < N) store_out(&Y[(size_t)m * ldy + n + 1], alpha * acc[i][j][h * 2 + 1], accumulate);
            }
}

template <typename OutT>
cudaError_t launch(const void* X, int ldx, const void* W, int ldw, void* Y, int ldy, int M, int N, int K, float alpha,
                   int accumulate, cudaStream_t stream) {
    if (K % BK != 0 || ldx % 8 != 0 || ldw % 8 != 0) return cudaErrorInvalidValue;
    if (M == 0 || N == 0) return cudaSuccess;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_kernel<OutT><<<grid, THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(X), ldx,
                                                    static_cast<const __nv_bfloat16*>(W), ldw, static_cast<OutT*>(Y),
                                                    ldy, M, N, K, alpha, accumulate != 0);
    SQ_RETURN_LAUNCH();
}

}  // namespace

extern "C" cudaError_t squash_gemm_bf16_f32(const void* X, int ldx, const void* W, int ldw, float* Y, int ldy, int M,
                                            int N, int K, float alpha, int accumulate, cudaStream_t stream) {
    return launch<float>(X, ldx, W, ldw, Y, ldy, M, N, K, alpha, accumulate, stream);
}

extern "C" cudaError_t squash_gemm_bf16_bf16(const void* X, int ldx, const void* W, int ldw, void* Y, int ldy, int M,
                                             int N, int K, float alpha, int accumulate, cudaStream_t stream) {
    return launch<__nv_bfloat16>(X, ldx, W, ldw, Y, ldy, M, N, K, alpha, accumulate, stream);
}
