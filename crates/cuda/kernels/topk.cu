// Row-wise top-k by exact radix selection, k <= 512. One 1024-thread block per row.
//
// The k-th largest value is found with four 8-bit histogram passes over order-preserving integer
// keys; all larger values and enough copies of the k-th are collected, then the k picks are
// bitonic-sorted either by value (descending, ties by lower index) or by index (ascending).

#include "common.cuh"

namespace {

constexpr int THREADS = 1024, MAXK = 512;

SQ_DEVICE unsigned order_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// With `start_ptr` set (QSA group selection), row r covers n = (start + r + 1) / group values and is
// skipped when n <= k: such queries attend densely.
__global__ void __launch_bounds__(THREADS) topk_kernel(const float* __restrict__ x, int n, size_t ld, int k,
                                                       int sort_by_index, int* out_idx, float* out_val,
                                                       const int* __restrict__ start_ptr, int group) {
    if (start_ptr) {
        n = (*start_ptr + (int)blockIdx.x + 1) / group;
        if (n <= k) return;
    }
    __shared__ unsigned hist[256];
    __shared__ unsigned s_prefix, s_mask, s_remaining, s_gt, s_eq;
    __shared__ int pick_idx[MAXK];
    __shared__ float pick_val[MAXK];

    const float* row = x + blockIdx.x * ld;
    const int tid = threadIdx.x;
    if (tid == 0) {
        s_prefix = 0;
        s_mask = 0;
        s_remaining = k;
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < n; i += THREADS) {
            const unsigned u = order_key(row[i]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned above = 0, digit = 0;
            for (int d = 255; d >= 0; --d) {
                if (above + hist[d] >= s_remaining) {
                    digit = d;
                    break;
                }
                above += hist[d];
            }
            s_remaining -= above;
            s_prefix |= digit << shift;
            s_mask |= 255u << shift;
            s_gt = 0;
            s_eq = 0;
        }
        __syncthreads();
    }
    const unsigned threshold = s_prefix, take_equal = s_remaining;
    const unsigned greater = k - take_equal;
    for (int i = tid; i < n; i += THREADS) {
        const float v = row[i];
        const unsigned u = order_key(v);
        if (u > threshold) {
            const unsigned slot = atomicAdd(&s_gt, 1u);
            pick_idx[slot] = i;
            pick_val[slot] = v;
        } else if (u == threshold) {
            const unsigned slot = atomicAdd(&s_eq, 1u);
            if (slot < take_equal) {
                pick_idx[greater + slot] = i;
                pick_val[greater + slot] = v;
            }
        }
    }
    __syncthreads();

    int size = 1;
    while (size < k) size <<= 1;
    for (int i = k + tid; i < size; i += THREADS) {
        pick_idx[i] = 0x7fffffff;
        pick_val[i] = -INFINITY;
    }
    __syncthreads();
    // Bitonic sort so that "first" means: larger value (ties: lower index), or lower index.
    for (int len = 2; len <= size; len <<= 1) {
        for (int stride = len >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < size; i += THREADS) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool up = (i & len) == 0;
                    const int ai = pick_idx[i], aj = pick_idx[j];
                    const float vi = pick_val[i], vj = pick_val[j];
                    const bool i_first = sort_by_index ? ai < aj : (vi > vj || (vi == vj && ai < aj));
                    if (i_first != up) {
                        pick_idx[i] = aj;
                        pick_idx[j] = ai;
                        pick_val[i] = vj;
                        pick_val[j] = vi;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < k; i += THREADS) {
        out_idx[blockIdx.x * (size_t)k + i] = pick_idx[i];
        if (out_val) out_val[blockIdx.x * (size_t)k + i] = pick_val[i];
    }
}

}  // namespace

extern "C" cudaError_t squash_topk(const float* x, int rows, int n, size_t ld, int k, int sort_by_index, int* out_idx,
                                   float* out_val, cudaStream_t st) {
    if (k > MAXK || k > n || k <= 0) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    topk_kernel<<<rows, THREADS, 0, st>>>(x, n, ld, k, sort_by_index, out_idx, out_val, nullptr, 0);
    SQ_RETURN_LAUNCH();
}

namespace {

// lse[r] = logsumexp(x[r, :n]); target_logit[r] = x[r, targets[r]] (if targets). One block per row.
__global__ void lse_kernel(const float* __restrict__ x, int n, size_t ld, const int* __restrict__ targets, float* lse,
                           float* target_logit) {
    __shared__ float scratch[32];
    const float* row = x + blockIdx.x * ld;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, row[i]);
    mx = block_max(mx, scratch);
    float sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += expf(row[i] - mx);
    sum = block_sum(sum, scratch);
    if (threadIdx.x == 0) {
        lse[blockIdx.x] = mx + logf(sum);
        if (targets) target_logit[blockIdx.x] = row[targets[blockIdx.x]];
    }
}

}  // namespace

extern "C" cudaError_t squash_row_lse(const float* x, int rows, int n, size_t ld, const int* targets, float* lse,
                                      float* target_logit, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    lse_kernel<<<rows, 1024, 0, st>>>(x, n, ld, targets, lse, target_logit);
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_qsa_select(const float* scores, int T, size_t ld, const int* start, int* sel,
                                         cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    topk_kernel<<<T, THREADS, 0, st>>>(scores, 0, ld, MAXK, 1, sel, nullptr, start, 4);
    SQ_RETURN_LAUNCH();
}
