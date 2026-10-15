// Row-wise top-k by exact radix selection, k <= 512. One 1024-thread block per row.
//
// The k-th largest value is found with four 8-bit histogram passes over order-preserving integer
// keys; all larger values and enough copies of the k-th are collected, then the k picks are
// bitonic-sorted either by value (descending, ties by lower index) or by index (ascending).

#include <algorithm>
#include <cooperative_groups.h>

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
    pdl_launch_dependents();
    pdl_wait();
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
    SQ_CHECK_LAUNCH(sq_launch(topk_kernel, rows, THREADS, 0, st, x, n, ld, k, sort_by_index, out_idx, out_val, nullptr, 0));
    SQ_RETURN_LAUNCH();
}

namespace {

// lse[r] = logsumexp(x[r, :n]); target_logit[r] = x[r, targets[r]] (if targets). One block per row.
__global__ void lse_kernel(const float* __restrict__ x, int n, size_t ld, const int* __restrict__ targets, float* lse,
                           float* target_logit) {
    pdl_launch_dependents();
    pdl_wait();
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
    SQ_CHECK_LAUNCH(sq_launch(lse_kernel, rows, 1024, 0, st, x, n, ld, targets, lse, target_logit));
    SQ_RETURN_LAUNCH();
}


namespace {

// Per-row argmax in two passes: blocks of 1024 threads reduce 8192-wide slices to (value, index)
// candidates, then one warp per row picks the best (ties: lower index).
constexpr int AM_SLICE = 8192;

__global__ void argmax_partial_kernel(const float* __restrict__ x, int n, size_t ld, float* pv, int* pi) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* row = x + blockIdx.y * ld;
    const int begin = blockIdx.x * AM_SLICE, end = min(n, begin + AM_SLICE);
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x)
        if (row[i] > best) {
            best = row[i];
            bi = i;
        }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, best, o);
        const int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > best || (ov == best && oi < bi)) {
            best = ov;
            bi = oi;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) {
        sv[warp] = best;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
        best = lane < (int)(blockDim.x / 32) ? sv[lane] : -INFINITY;
        bi = lane < (int)(blockDim.x / 32) ? si[lane] : 0x7fffffff;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, best, o);
            const int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > best || (ov == best && oi < bi)) {
                best = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            pv[blockIdx.y * gridDim.x + blockIdx.x] = best;
            pi[blockIdx.y * gridDim.x + blockIdx.x] = bi;
        }
    }
}

__global__ void argmax_final_kernel(const float* __restrict__ pv, const int* __restrict__ pi, int slices, int* out) {
    pdl_launch_dependents();
    pdl_wait();
    const int r = blockIdx.x, lane = threadIdx.x;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = lane; i < slices; i += 32) {
        const float v = pv[r * slices + i];
        const int idx = pi[r * slices + i];
        if (v > best || (v == best && idx < bi)) {
            best = v;
            bi = idx;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, best, o);
        const int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > best || (ov == best && oi < bi)) {
            best = ov;
            bi = oi;
        }
    }
    if (lane == 0) out[r] = bi;
}

}  // namespace

extern "C" size_t squash_argmax_scratch(int rows, int n) { return (size_t)rows * ((n + AM_SLICE - 1) / AM_SLICE) * 8; }

// scratch: squash_argmax_scratch(rows, n) bytes.
extern "C" cudaError_t squash_argmax(const float* x, int rows, int n, size_t ld, void* scratch, int* out, cudaStream_t st) {
    if (rows == 0) return cudaSuccess;
    const int slices = (n + AM_SLICE - 1) / AM_SLICE;
    float* pv = static_cast<float*>(scratch);
    int* pi = reinterpret_cast<int*>(pv + (size_t)rows * slices);
    SQ_CHECK_LAUNCH(sq_launch(argmax_partial_kernel, dim3(slices, rows), 1024, 0, st, x, n, ld, pv, pi));
    SQ_CHECK_LAUNCH(sq_launch(argmax_final_kernel, rows, 32, 0, st, pv, pi, slices, out));
    SQ_RETURN_LAUNCH();
}

namespace {

// QSA group selection: per query row, the 512 highest-scoring complete groups, written in ascending
// group order. Exact radix selection in three passes (11, 11, 10 bits) over order-preserving keys;
// ties at the threshold go to the lowest group indices. Each thread owns a contiguous range of the
// row, so the compaction is a prefix sum over thread counts: deterministic and already sorted.
constexpr int SEL_THREADS = 1024, SEL_K = 512, SEL_BINS = 2048;

// Inclusive prefix sum over the block's per-thread values (1024 threads). `scratch` holds 32 ints.
SQ_DEVICE int block_inclusive_scan(int v, int* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up_sync(0xffffffff, v, o);
        if (lane >= o) v += n;
    }
    __syncthreads();
    if (lane == 31) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int w = scratch[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int n = __shfl_up_sync(0xffffffff, w, o);
            if (lane >= o) w += n;
        }
        scratch[lane] = w;
    }
    __syncthreads();
    return v + (warp > 0 ? scratch[warp - 1] : 0);
}

__global__ void __launch_bounds__(SEL_THREADS) qsa_select_kernel(const float* __restrict__ scores, size_t ld,
                                                                 const int* __restrict__ start_ptr, int group,
                                                                 int* out) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ int hist[SEL_BINS];
    __shared__ int scratch[32];
    __shared__ unsigned s_prefix, s_mask;
    __shared__ int s_remaining;
    const int n = (*start_ptr + (int)blockIdx.x + 1) / group;
    if (n <= SEL_K) return;  // dense row
    const float* row = scores + blockIdx.x * ld;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) {
        s_prefix = 0;
        s_mask = 0;
        s_remaining = SEL_K;
    }
    const int shifts[3] = {21, 10, 0}, widths[3] = {11, 11, 10};
    for (int pass = 0; pass < 3; ++pass) {
        const int shift = shifts[pass], bins = 1 << widths[pass];
        for (int b = tid; b < SEL_BINS; b += SEL_THREADS) hist[b] = 0;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        // float4 loads, four per step: 16 scores per thread in flight.
        for (int i0 = tid * 4; i0 < n; i0 += SEL_THREADS * 16) {
            float4 v[4];
#pragma unroll
            for (int u4 = 0; u4 < 4; ++u4) {
                const int i = i0 + u4 * SEL_THREADS * 4;
                v[u4] = i < n ? *reinterpret_cast<const float4*>(row + i) : make_float4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u4 = 0; u4 < 4; ++u4) {
                const int i = i0 + u4 * SEL_THREADS * 4;
                const float e[4] = {v[u4].x, v[u4].y, v[u4].z, v[u4].w};
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    if (i + c >= n) break;
                    const unsigned u = order_key(e[c]);
                    if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & (bins - 1)], 1);
                }
            }
        }
        __syncthreads();
        // Find the bin holding the remaining-th largest key: warp w sums bins [64 w, 64 w + 64).
        const int per_warp = SEL_BINS / 32;
        int wsum = 0;
        for (int b = lane; b < per_warp; b += 32) wsum += hist[warp * per_warp + b];
        wsum = __reduce_add_sync(0xffffffff, wsum);
        if (lane == 0) scratch[warp] = wsum;
        __syncthreads();
        if (tid == 0) {
            int above = 0, w = 31;
            for (; w > 0; --w) {
                if (above + scratch[w] >= s_remaining) break;
                above += scratch[w];
            }
            int b = w * per_warp + per_warp - 1;
            for (; b > w * per_warp; --b) {
                if (above + hist[b] >= s_remaining) break;
                above += hist[b];
            }
            s_remaining -= above;
            s_prefix |= (unsigned)b << shift;
            s_mask |= (unsigned)(bins - 1) << shift;
        }
        __syncthreads();
    }
    const unsigned threshold = s_prefix;
    const int take_equal = s_remaining;
    // Compaction over contiguous per-thread ranges (multiples of 4, read as float4).
    const int per = ((n + SEL_THREADS - 1) / SEL_THREADS + 3) / 4 * 4, b0 = min(n, tid * per), b1 = min(n, b0 + per);
    int gt = 0, eq = 0;
    for (int i = b0; i < b1; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(row + i);
        const float e[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (i + c >= b1) break;
            const unsigned u = order_key(e[c]);
            gt += u > threshold;
            eq += u == threshold;
        }
    }
    const int eq_before = block_inclusive_scan(eq, scratch) - eq;
    const int eq_take = max(0, min(eq, take_equal - eq_before));
    const int mine = gt + eq_take;
    int pos = block_inclusive_scan(mine, scratch) - mine;
    int eq_seen = 0;
    int* dst = out + (size_t)blockIdx.x * SEL_K;
    for (int i = b0; i < b1; ++i) {
        const unsigned u = order_key(row[i]);
        if (u > threshold || (u == threshold && eq_seen++ < eq_take)) dst[pos++] = i;
    }
    (void)lane;
}

// Cluster variant: CL blocks per row, block r owning the contiguous slice [r per, (r + 1) per) of the
// row as order keys in shared memory (one global read). Each pass histograms the slices locally,
// sums the histograms over the cluster through distributed shared memory (block r sums bins
// [r 256, r 256 + 256)), and every block makes the same threshold decision. Compaction offsets come
// from the lower-ranked blocks' counts, so the output equals qsa_select_kernel's exactly.
namespace cg = cooperative_groups;
constexpr int CL = 8, CL_THREADS = 512, CL_SLICE_BINS = SEL_BINS / CL;
// Rows up to SOLO_N groups are selected by the cluster's first block alone (no cluster barriers).
constexpr int SOLO_N = 8192;

__global__ void __launch_bounds__(CL_THREADS) qsa_select_cluster_kernel(const float* __restrict__ scores, size_t ld,
                                                                        const int* __restrict__ start_ptr, int group,
                                                                        int* out) {
    pdl_launch_dependents();
    pdl_wait();
    cg::cluster_group cluster = cg::this_cluster();
    extern __shared__ unsigned keys[];
    __shared__ int hist[SEL_BINS];
    __shared__ int gsum[CL_SLICE_BINS];
    __shared__ int scratch[32];
    __shared__ int s_counts[2];
    __shared__ unsigned s_prefix, s_mask;
    __shared__ int s_remaining;
    __shared__ bool s_done;
    const int rank = (int)cluster.block_rank(), r = blockIdx.x / CL;
    const int n = (*start_ptr + r + 1) / group;
    if (n <= SEL_K) return;  // dense row (the whole cluster returns)
    const bool solo = n <= SOLO_N;
    if (solo && rank != 0) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, warps = CL_THREADS / 32;
    const float* row = scores + (size_t)r * ld;
    const int per = solo ? n : ((n + CL - 1) / CL + 3) / 4 * 4, b0 = min(n, rank * per), m = min(n, b0 + per) - b0;
    for (int i = tid * 4; i < m; i += CL_THREADS * 4) {
        const float4 v = *reinterpret_cast<const float4*>(row + b0 + i);
        keys[i] = order_key(v.x);
        if (i + 1 < m) keys[i + 1] = order_key(v.y);
        if (i + 2 < m) keys[i + 2] = order_key(v.z);
        if (i + 3 < m) keys[i + 3] = order_key(v.w);
    }
    if (tid == 0) {
        s_prefix = 0;
        s_mask = 0;
        s_remaining = SEL_K;
        s_done = false;
    }
    const int shifts[3] = {21, 10, 0}, widths[3] = {11, 11, 10};
    for (int pass = 0; pass < 3; ++pass) {
        const int shift = shifts[pass], bins = 1 << widths[pass];
        for (int b = tid; b < SEL_BINS; b += CL_THREADS) hist[b] = 0;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < m; i += CL_THREADS) {
            const unsigned u = keys[i];
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & (bins - 1)], 1);
        }
        if (!solo) {
            // Block q sums bins [q 256, q 256 + 256) over the cluster, then every block gathers them.
            cluster.sync();
            for (int b = tid; b < CL_SLICE_BINS; b += CL_THREADS) {
                int sum = 0;
#pragma unroll
                for (int q = 0; q < CL; ++q) sum += cluster.map_shared_rank(hist, q)[rank * CL_SLICE_BINS + b];
                gsum[b] = sum;
            }
            cluster.sync();
            for (int b = tid; b < SEL_BINS; b += CL_THREADS) hist[b] = cluster.map_shared_rank(gsum, b / CL_SLICE_BINS)[b % CL_SLICE_BINS];
        }
        __syncthreads();
        // Find the bin holding the remaining-th largest key (warp 0: warp sums, then the chosen warp's
        // bins, four per lane, as suffix sums and ballots).
        constexpr int per_warp = SEL_BINS / (CL_THREADS / 32), BPL = per_warp / 32;
        int wsum = 0;
#pragma unroll
        for (int j = 0; j < BPL; ++j) wsum += hist[warp * per_warp + lane * BPL + j];
        wsum = __reduce_add_sync(0xffffffff, wsum);
        if (lane == 0) scratch[warp] = wsum;
        __syncthreads();
        if (warp == 0) {
            const int rem = s_remaining;
            const int ws = lane < warps ? scratch[lane] : 0;
            int sw = ws;  // sum over warps >= lane
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_down_sync(0xffffffff, sw, o);
                if (lane + o < 32) sw += v;
            }
            const int w = 31 - __clz(__ballot_sync(0xffffffff, lane < warps && sw >= rem) | 1u);
            const int above_w = __shfl_sync(0xffffffff, sw - ws, w);
            const int base = w * per_warp + lane * BPL;
            int hb[BPL], lt = 0;
#pragma unroll
            for (int j = 0; j < BPL; ++j) lt += hb[j] = hist[base + j];
            int sl = lt;  // sum over this warp's lanes >= lane
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_down_sync(0xffffffff, sl, o);
                if (lane + o < 32) sl += v;
            }
            const int L = 31 - __clz(__ballot_sync(0xffffffff, above_w + sl >= rem) | 1u);
            if (lane == L) {
                int above = above_w + sl - lt, j = BPL - 1;
                for (; j > 0; --j) {
                    if (above + hb[j] >= rem) break;
                    above += hb[j];
                }
                s_remaining = rem - above;
                s_done = hb[j] == rem - above;  // the whole bin is taken: no ties to break
                s_prefix |= (unsigned)(base + j) << shift;
                s_mask |= (unsigned)(bins - 1) << shift;
            }
        }
        __syncthreads();
        if (s_done) break;
    }
    // Keys compare on the resolved bits only (all of them unless a pass ended the search early).
    const unsigned threshold = s_prefix, rmask = s_mask;
    const int take_equal = s_remaining;
    // Per-thread contiguous ranges of the slice.
    const int tper = (m + CL_THREADS - 1) / CL_THREADS, t0 = min(m, tid * tper), t1 = min(m, t0 + tper);
    int gt = 0, eq = 0;
    for (int i = t0; i < t1; ++i) {
        const unsigned u = keys[i] & rmask;
        gt += u > threshold;
        eq += u == threshold;
    }
    const int eq_in = block_inclusive_scan(eq, scratch);
    const int gt_in = block_inclusive_scan(gt, scratch);
    if (tid == CL_THREADS - 1) {
        s_counts[0] = gt_in;
        s_counts[1] = eq_in;
    }
    int gt_below = 0, eq_below = 0;
    if (!solo) {
        cluster.sync();
        for (int q = 0; q < rank; ++q) {
            const int* c = cluster.map_shared_rank(s_counts, q);
            gt_below += c[0];
            eq_below += c[1];
        }
        cluster.sync();  // no block leaves while others read its counts
    }
    // Lower ranks' takes of threshold ties: min(their ties, what was left for them).
    const int eq_taken_below = min(eq_below, take_equal);
    const int eq_before = eq_below + eq_in - eq;
    const int eq_take = max(0, min(eq, take_equal - eq_before));
    const int mine = gt + eq_take;
    int pos = gt_below + eq_taken_below + block_inclusive_scan(mine, scratch) - mine;
    int eq_seen = 0;
    int* dst = out + (size_t)r * SEL_K;
    for (int i = t0; i < t1; ++i) {
        const unsigned u = keys[i] & rmask;
        if (u > threshold || (u == threshold && eq_seen++ < eq_take)) dst[pos++] = b0 + i;
    }
}

}  // namespace

// `ld` must be a multiple of 4 (rows are read as float4).
extern "C" cudaError_t squash_qsa_select(const float* scores, int T, size_t ld, const int* start, int* sel,
                                            cudaStream_t st) {
    if (T == 0) return cudaSuccess;
    if (ld % 4 != 0) return cudaErrorInvalidValue;
    // Cluster variant while a row slice fits in shared memory.
    const size_t smem = std::max(((ld + CL - 1) / CL + 3) / 4 * 4, std::min(ld, (size_t)SOLO_N)) * sizeof(unsigned);
    if (smem <= 96 * 1024) {
        static size_t configured = 0;
        if (smem > configured) {
            const cudaError_t e = cudaFuncSetAttribute(qsa_select_cluster_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            if (e != cudaSuccess) return e;
            configured = smem;
        }
        cudaLaunchAttribute attr[2];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        attr[1].id = cudaLaunchAttributeClusterDimension;
        attr[1].val.clusterDim.x = CL;
        attr[1].val.clusterDim.y = 1;
        attr[1].val.clusterDim.z = 1;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(T * CL);
        cfg.blockDim = dim3(CL_THREADS);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = st;
        cfg.attrs = attr;
        cfg.numAttrs = 2;
        SQ_CHECK_LAUNCH(cudaLaunchKernelEx(&cfg, qsa_select_cluster_kernel, scores, ld, start, 4, sel));
        SQ_RETURN_LAUNCH();
    }
    SQ_CHECK_LAUNCH(sq_launch(qsa_select_kernel, T, SEL_THREADS, 0, st, scores, ld, start, 4, sel));
    SQ_RETURN_LAUNCH();
}
