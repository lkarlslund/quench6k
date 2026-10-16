// W4A4 NVFP4 expert GEMM for prefill: activations quantized to NVFP4 per row (E2M1 values, E4M3 block
// scales per 16 chosen by least squared error, one FP32 scale per row = amax / (6 * 448)), multiplied
// with the checkpoint's NVFP4 expert weights on the FP4 tensor cores
// (mma.sync kind::mxf4nvf4.block_scale.scale_vec::4X m16n8k64, block scales applied by the MMA).
//
// Quantized rows: codes U8 [rows, K/2] (low nibble first), block scales U8 [rows, K/16], row scales
// F32 [rows]. Weights per expert: codes U8 [N, K/2], scales F8E4M3 [N, K/16], global scale ws2[e].

#include "fp4.cuh"
#include "mma.cuh"

namespace {

constexpr int TOPK = 10;

// x BF16 [rows, K] -> codes, scales, row scales. A CTA takes QROWS rows: each warp finds the absolute
// maximum of QROWS / 8 rows, then the CTA's threads quantize the rows' 16-value blocks.
constexpr int QROWS = 16;

__global__ void __launch_bounds__(256) quant_rows_kernel(const __nv_bfloat16* __restrict__ x, int rows, int K,
                                                          uint8_t* codes, uint8_t* scales, float* rscale) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float rs_s[QROWS];
    const int row0 = blockIdx.x * QROWS, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = warp; r < QROWS; r += 8) {
        const int row = row0 + r;
        float amax = 0.0f;
        if (row < rows) {
            const __nv_bfloat16* xr = x + (size_t)row * K;
            for (int i = lane * 8; i < K; i += 256) {
                const uint4 u = *reinterpret_cast<const uint4*>(xr + i);
                const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(h[j]);
                    amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
                }
            }
        }
        amax = warp_max(amax);
        if (lane == 0) {
            rs_s[r] = amax / (6.0f * 448.0f);
            if (row < rows) rscale[row] = rs_s[r];
        }
    }
    __syncthreads();
    const int nb = K / 16;
    for (int i = threadIdx.x; i < QROWS * nb; i += 256) {
        const int r = i / nb, b = i % nb, row = row0 + r;
        if (row >= rows) break;
        const float rs = rs_s[r];
        float v[16];
        const __nv_bfloat16* xr = x + (size_t)row * K + 16 * b;
        const uint4 u0 = *reinterpret_cast<const uint4*>(xr);
        const uint4 u1 = *reinterpret_cast<const uint4*>(xr + 8);
        const __nv_bfloat16* h0 = reinterpret_cast<const __nv_bfloat16*>(&u0);
        const __nv_bfloat16* h1 = reinterpret_cast<const __nv_bfloat16*>(&u1);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = rs > 0.0f ? bf2f(h0[j]) / rs : 0.0f, v[8 + j] = rs > 0.0f ? bf2f(h1[j]) / rs : 0.0f;
        const int code = rs > 0.0f ? fp4_block_scale(v) : 0;
        const float bs = ue4m3_value(code);
        uint32_t q[2] = {0, 0};
#pragma unroll
        for (int j = 0; j < 16; ++j) q[j / 8] |= (uint32_t)(bs > 0.0f ? e2m1_code(v[j] / bs) : 0) << (4 * (j % 8));
        *reinterpret_cast<uint2*>(codes + (size_t)row * (K / 2) + 8 * b) = make_uint2(q[0], q[1]);
        scales[(size_t)row * (K / 16) + b] = (uint8_t)code;
    }
}

// Quantized row of token pair / 10 -> grouped row pair_row[pair] (K = 2560).
__global__ void gather_fp4_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                                  const float* __restrict__ rscale, const int* __restrict__ pair_row, int K,
                                  uint8_t* gcodes, uint8_t* gscales, float* grscale) {
    pdl_launch_dependents();
    pdl_wait();
    const int pair = blockIdx.x, tok = pair / TOPK, dst = pair_row[pair];
    const uint4* src = reinterpret_cast<const uint4*>(codes + (size_t)tok * (K / 2));
    uint4* out = reinterpret_cast<uint4*>(gcodes + (size_t)dst * (K / 2));
    for (int i = threadIdx.x; i < K / 32; i += blockDim.x) out[i] = src[i];
    const uint4* ssrc = reinterpret_cast<const uint4*>(scales + (size_t)tok * (K / 16));
    uint4* sout = reinterpret_cast<uint4*>(gscales + (size_t)dst * (K / 16));
    for (int i = threadIdx.x; i < K / 256; i += blockDim.x) sout[i] = ssrc[i];
    if (threadIdx.x == 0) grscale[dst] = rscale[tok];
}

SQ_DEVICE void cp_async8(void* smem, const void* gmem, bool valid) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int size = valid ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s), "l"(gmem), "r"(size));
}

SQ_DEVICE void mma_fp4(float* c, const uint32_t* a, const uint32_t* b, uint32_t sa, uint32_t sb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}

// Tile: BM rows of one expert x BN outputs; k-tiles of BK values (BK / 2 bytes per row). Rows in
// shared memory are 64 bytes (four 16-byte chunks), chunk c of row r stored at c ^ ((r >> 1) & 3) so
// that ldmatrix reads of 8 rows hit distinct banks.
constexpr int BM = 64, BN = 128, BK = 128, RB = BK / 2, STAGES = 4, THREADS = 256;

SQ_DEVICE int swz(int r, int c) { return r * RB + ((c ^ ((r >> 1) & 3)) << 4); }

struct __align__(16) Stage {
    uint8_t a[BM * RB];
    uint8_t b[BN * RB];
    uint8_t as[BM * 8];
    uint8_t bs[BN * 8];
};

__global__ void __launch_bounds__(THREADS) fp4_gemm_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ As,
                                                           const float* __restrict__ Ar, const uint64_t* __restrict__ w_ptr,
                                                           const uint64_t* __restrict__ s_ptr, const float* __restrict__ ws2,
                                                           const int* __restrict__ offsets, const int* __restrict__ tile_expert,
                                                           const int* __restrict__ tile_row, const int* __restrict__ num_tiles,
                                                           float* Y, int N, int K) {
    pdl_launch_dependents();
    pdl_wait();
    extern __shared__ __align__(16) uint8_t fp4_smem[];
    Stage* st = reinterpret_cast<Stage*>(fp4_smem);
    if ((int)blockIdx.y >= *num_tiles) return;
    const int e = tile_expert[blockIdx.y], r0 = tile_row[blockIdx.y];
    const int r_end = min(r0 + BM, offsets[e + 1]);
    const uint8_t* W = reinterpret_cast<const uint8_t*>(w_ptr[e]);
    const uint8_t* Ws = reinterpret_cast<const uint8_t*>(s_ptr[e]);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp >> 2, warp_n = warp & 3;
    const int n0 = blockIdx.x * BN;
    const int k_tiles = K / BK, KB = K / 2, KS = K / 16;

    auto load = [&](int s, int kt) {
        Stage& g = st[s];
        {  // A: 64 rows x 4 chunks, one per thread
            const int r = tid >> 2, c = tid & 3;
            const bool ok = r0 + r < r_end;
            cp_async16(&g.a[swz(r, c)], A + (size_t)(ok ? r0 + r : r0) * KB + kt * RB + c * 16, ok);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {  // B: 128 rows x 4 chunks
            const int idx = tid + i * THREADS, r = idx >> 2, c = idx & 3;
            const bool ok = n0 + r < N;
            cp_async16(&g.b[swz(r, c)], W + (size_t)(ok ? n0 + r : n0) * KB + kt * RB + c * 16, ok);
        }
        if (tid < BM) {
            const bool ok = r0 + tid < r_end;
            cp_async8(&g.as[tid * 8], As + (size_t)(ok ? r0 + tid : r0) * KS + kt * 8, ok);
        } else if (tid < BM + BN) {
            const int r = tid - BM;
            const bool ok = n0 + r < N;
            cp_async8(&g.bs[r * 8], Ws + (size_t)(ok ? n0 + r : n0) * KS + kt * 8, ok);
        }
    };

    float acc[2][4][4] = {};
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
#pragma unroll
        for (int s = 0; s < 2; ++s) {  // two k64 steps per k-tile
            uint32_t a[2][4], b[4][2], sa[2], sb[4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = warp_m * 32 + i * 16 + (lane & 15);
                ldmatrix_x4(a[i], &g.a[swz(r, s * 2 + (lane >> 4))]);
                sa[i] = *reinterpret_cast<const uint32_t*>(&g.as[(warp_m * 32 + i * 16 + ta) * 8 + s * 4]);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                uint32_t r4[4];
                const int r = warp_n * 32 + j * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldmatrix_x4(r4, &g.b[swz(r, s * 2 + ((lane >> 3) & 1))]);
                b[2 * j][0] = r4[0], b[2 * j][1] = r4[1], b[2 * j + 1][0] = r4[2], b[2 * j + 1][1] = r4[3];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) sb[j] = *reinterpret_cast<const uint32_t*>(&g.bs[(warp_n * 32 + j * 8 + tb) * 8 + s * 4]);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_fp4(acc[i][j], a[i], b[j], sa[i], sb[j]);
        }
    }
    cp_async_wait<0>();
    const float s2 = ws2[e];
    const int g8 = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = r0 + warp_m * 32 + i * 16 + g8 + h * 8;
            if (r >= r_end) continue;
            const float sc = Ar[r] * s2;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int n = n0 + warp_n * 32 + j * 8 + 2 * t4;
                if (n + 1 < N) {
                    *reinterpret_cast<float2*>(Y + (size_t)r * N + n) = make_float2(acc[i][j][2 * h] * sc, acc[i][j][2 * h + 1] * sc);
                } else if (n < N) {
                    Y[(size_t)r * N + n] = acc[i][j][2 * h] * sc;
                }
            }
        }
}

}  // namespace

extern "C" int squash_fp4_tile_rows() { return BM; }

extern "C" cudaError_t squash_fp4_quant_rows(const void* x, int rows, int K, uint8_t* codes, uint8_t* scales, float* rscale,
                                             cudaStream_t st) {
    if (K % 256 != 0 && K % 128 != 0) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(quant_rows_kernel, (rows + QROWS - 1) / QROWS, 256, 0, st, static_cast<const __nv_bfloat16*>(x), rows, K, codes,
                              scales, rscale));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_fp4_gather(const uint8_t* codes, const uint8_t* scales, const float* rscale, const int* pair_row,
                                         int T, int K, uint8_t* gcodes, uint8_t* gscales, float* grscale, cudaStream_t st) {
    if (K % 256 != 0) return cudaErrorInvalidValue;
    if (T == 0) return cudaSuccess;
    SQ_CHECK_LAUNCH(sq_launch(gather_fp4_kernel, T * TOPK, 64, 0, st, codes, scales, rscale, pair_row, K, gcodes, gscales, grscale));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_moe_fp4_gemm(const uint8_t* a, const uint8_t* as, const float* ar, const uint64_t* w_ptr,
                                           const uint64_t* s_ptr, const float* ws2, const int* offsets, const int* tile_expert,
                                           const int* tile_row, const int* num_tiles, int max_tiles, float* y, int N, int K,
                                           cudaStream_t st) {
    if (K % BK != 0) return cudaErrorInvalidValue;
    if (max_tiles == 0) return cudaSuccess;
    constexpr size_t smem = STAGES * sizeof(Stage);
    static bool configured = false;
    if (!configured) {
        cudaError_t err = cudaFuncSetAttribute(fp4_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (err != cudaSuccess) return err;
        configured = true;
    }
    SQ_CHECK_LAUNCH(sq_launch(fp4_gemm_kernel, dim3((N + BN - 1) / BN, max_tiles), THREADS, smem, st, a, as, ar, w_ptr, s_ptr, ws2,
                              offsets, tile_expert, tile_row, num_tiles, y, N, K));
    SQ_RETURN_LAUNCH();
}
