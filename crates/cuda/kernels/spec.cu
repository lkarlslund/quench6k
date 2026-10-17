// Device-side bookkeeping of greedy MTP speculative decoding, so a draft-and-verify round runs as CUDA
// graphs. The round control buffer (int32) holds:
//   [P] position of the next token x   [X] x   [C] tokens to catch up   [A] last accepted draft count
//   [CATCH ..] tokens the MTP cache has seen only as drafts   [D ..] drafts   [G ..] target greedy tokens

#include "common.cuh"

namespace {

constexpr int NMAX = 8, P = 0, X = 1, CNT = 2, ACC = 3, CATCH = 4, DRAFT = CATCH + NMAX + 1, GREEDY = DRAFT + NMAX;

// Writes the step block (start, tokens, MRoPE positions) of MTP step j: step 0 covers the catch-up
// tokens and x (c + 1 tokens from position p - c); step j > 0 covers draft j - 1 at position p + j.
__global__ void mtp_header_kernel(const int* rc, int* hdr, int tokens_off, int pos_off, int j, int c) {
    pdl_launch_dependents();
    pdl_wait();
    const int i = threadIdx.x;
    const int p = rc[P];
    const int start = j == 0 ? p - c : p + j;
    const int T = j == 0 ? c + 1 : 1;
    if (i == 0) hdr[0] = start;
    if (i < T) {
        const int tok = j == 0 ? (i < c ? rc[CATCH + i] : rc[X]) : rc[DRAFT + j - 1];
        hdr[tokens_off / 4 + i] = tok;
#pragma unroll
        for (int a = 0; a < 3; ++a) hdr[pos_off / 4 + 3 * i + a] = start + i;
    }
}

// After verification of [x, d1 .. dn]: a = number of leading drafts the target agrees with; commits
// [x, d1 .. da] (they become the next catch-up), advances p and x, and sets the recurrent-state control
// word to replay a + 1 recorded tokens from the record buffer just written.
__global__ void accept_kernel(int* rc, int* ctl, int n) {
    pdl_launch_dependents();
    pdl_wait();
    if (threadIdx.x != 0) return;
    int a = 0;
    while (a < n && rc[GREEDY + a] == rc[DRAFT + a]) ++a;
    rc[CATCH] = rc[X];
    for (int i = 1; i <= a; ++i) rc[CATCH + i] = rc[DRAFT + i - 1];
    rc[CNT] = a + 1;
    rc[ACC] = a;
    rc[P] += a + 1;
    rc[X] = rc[GREEDY + a];
    ctl[0] = a + 1;
    ctl[1] ^= 1;
}

SQ_DEVICE double uniform(SampleParams* sp) { return sample_uniform(sp); }

// Sampled verification of [x, d1 .. dn] (rows 0 .. n of sorted top-K ids and logits). Each row's
// distribution p is the plain sampler's: softmax at the temperature over the top k, cut to the smallest
// prefix reaching top_p, renormalized. Row n draws from p. Row r < n checks draft d_{r+1}:
//  - greedy drafts: kept with probability p(d), otherwise a draw from p without d;
//  - sampled drafts (d ~ q, the drafter's distribution in qs[r]): kept with probability
//    min(1, p(d) / q(d)), otherwise a draw from max(0, p - q), renormalized.
// Either way the output has the plain sampler's distribution. The results go where the greedy tokens
// go, so the accept kernel keeps exactly the accepted drafts and the corrected token.
__global__ void sample_rows_kernel(const int* __restrict__ ids, const float* __restrict__ vals, int K, int n, int* rc,
                                   SampleParams* sp, const DraftQ* __restrict__ qs) {
    pdl_launch_dependents();
    pdl_wait();
    // The exponentials (slow in FP64) of every row on all lanes of the warp; lane 0 does the rest.
    const int k = min(sp->k, K);
    const double t = fmax((double)sp->temperature, 1e-4);
    __shared__ double ex[17][64];
    for (int r = 0; r <= n; ++r)
        for (int i = threadIdx.x; i < k; i += 32) ex[r][i] = exp(((double)vals[(size_t)r * K + i] - (double)vals[(size_t)r * K]) / t);
    __syncwarp();
    if (threadIdx.x != 0) return;
    double probs[64];
    for (int r = 0; r <= n; ++r) {
        const int* id = ids + (size_t)r * K;
        const float* v = vals + (size_t)r * K;
        (void)v;
        double total = 0.0;
        for (int i = 0; i < k; ++i) {
            probs[i] = ex[r][i];
            total += probs[i];
        }
        int keep = k;
        double acc = 0.0;
        for (int i = 0; i < k; ++i) {
            probs[i] /= total;
        }
        if (sp->top_p < 1.0f) {
            for (int i = 0; i < k; ++i) {
                acc += probs[i];
                if (acc >= (double)sp->top_p) {
                    keep = i + 1;
                    break;
                }
            }
        }
        // min_p: also a prefix of the sorted list (ratios to the first survive renormalizing).
        for (int i = 1; i < keep; ++i)
            if (probs[i] < (double)sp->min_p * probs[0]) {
                keep = i;
                break;
            }
        double mass = 0.0;
        for (int i = 0; i < keep; ++i) mass += probs[i];
        if (r < n && sp->draft_sampled && qs != nullptr) {
            const int d = rc[DRAFT + r];
            const DraftQ& q = qs[r];
            auto q_of = [&](int tok) {
                for (int j = 0; j < q.count; ++j)
                    if (q.ids[j] == tok) return (double)q.probs[j];
                return 0.0;
            };
            double pd = 0.0;
            for (int i = 0; i < keep; ++i)
                if (id[i] == d) pd = probs[i] / mass;
            const double qd = q_of(d);
            if (qd > 0.0 && uniform(sp) * qd < pd) {
                rc[GREEDY + r] = d;
                continue;
            }
            double resid = 0.0;
            for (int i = 0; i < keep; ++i) resid += fmax(probs[i] / mass - q_of(id[i]), 0.0);
            int tok = -1;
            if (resid > 0.0) {
                double rem = resid * uniform(sp);
                for (int i = 0; i < keep; ++i) {
                    const double w = fmax(probs[i] / mass - q_of(id[i]), 0.0);
                    if (w <= 0.0) continue;
                    tok = id[i];
                    rem -= w;
                    if (rem <= 0.0) break;
                }
            } else {  // p <= q everywhere (rounding): draw from p
                double rem = mass * uniform(sp);
                for (int i = 0; i < keep; ++i) {
                    tok = id[i];
                    rem -= probs[i];
                    if (rem <= 0.0) break;
                }
            }
            rc[GREEDY + r] = tok < 0 ? id[0] : tok;
            return;
        }
        int skip = -1;
        double pd = 0.0;
        if (r < n) {
            const int d = rc[DRAFT + r];
            for (int i = 0; i < keep; ++i)
                if (id[i] == d) skip = i, pd = probs[i] / mass;
            if (skip >= 0 && uniform(sp) < pd) {
                rc[GREEDY + r] = d;
                continue;
            }
        }
        // Draw from the kept candidates without the rejected draft.
        double rem = (mass - (skip >= 0 ? probs[skip] : 0.0)) * uniform(sp);
        int tok = -1;
        for (int i = 0; i < keep; ++i) {
            if (i == skip) continue;
            tok = id[i];
            rem -= probs[i];
            if (rem <= 0.0) break;
        }
        if (tok < 0) tok = id[0];  // only the draft was kept and it was rejected: cannot happen (pd = 1)
        rc[GREEDY + r] = tok;
        return;  // later rows are past the first rejection
    }
}


// Repetition and presence penalties (vLLM semantics). A request's history is a set of token ids:
// flags[id] bit 0 = in the prompt or the output, bit 1 = in the output; `list` holds each id of the
// set once (st->count of them). A logit x of a token in the set becomes x / rep (x > 0) or x * rep,
// and then, for an output token, x - presence.
struct PenState {
    float rep, presence;
    int count, pad;
};

SQ_DEVICE float penalize(float x, bool out, const PenState& st) {
    x = x > 0.0f ? x / st.rep : x * st.rep;
    return out ? x - st.presence : x;
}

// Adds tokens to the history with flag bits `bits`: ids[0 .. n) (n_ptr: n = *n_ptr instead).
__global__ void penalty_mark_kernel(PenState* st, unsigned* flags, int* list, const int* ids, int n, const int* n_ptr, unsigned bits) {
    pdl_launch_dependents();
    pdl_wait();
    if (n_ptr != nullptr) n = *n_ptr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int tok = ids[i];
        if ((atomicOr(&flags[tok], bits) & 1u) == 0) list[atomicAdd(&st->count, 1)] = tok;
    }
}

// Penalizes `rows` logits rows (stride ld). With `rc` (a speculative verification of [x, d1 .. dn]),
// row r also counts x, d1 .. dr as output tokens: they precede it in the sequence it scores.
__global__ void penalty_apply_kernel(float* logits, int ld, const PenState* st, const unsigned* flags, const int* list, const int* rc) {
    pdl_launch_dependents();
    pdl_wait();
    const int r = blockIdx.x;
    float* row = logits + (size_t)r * ld;
    const int extra = rc != nullptr ? r + 1 : 0;
    __shared__ int ex[NMAX + 1];
    if (threadIdx.x < extra) ex[threadIdx.x] = threadIdx.x == 0 ? rc[X] : rc[DRAFT + threadIdx.x - 1];
    __syncthreads();
    const PenState s = *st;
    for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < s.count; i += gridDim.y * blockDim.x) {
        const int tok = list[i];
        bool out = (flags[tok] & 2u) != 0;
        for (int j = 0; j < extra; ++j) out |= ex[j] == tok;
        row[tok] = penalize(row[tok], out, s);
    }
    // Speculative tokens outside the history, each once.
    if (blockIdx.y == 0 && threadIdx.x < extra) {
        const int j = threadIdx.x, tok = ex[j];
        bool first = (flags[tok] & 1u) == 0;
        for (int k = 0; k < j; ++k) first &= ex[k] != tok;
        if (first) row[tok] = penalize(row[tok], true, s);
    }
}

}  // namespace

extern "C" cudaError_t quench6k_spec_sample_rows(const int* ids, const float* vals, int k_list, int n, int* rc, void* params,
                                               const void* qs, cudaStream_t st) {
    if (n > NMAX || k_list > 64) return cudaErrorInvalidValue;
    SQ_CHECK_LAUNCH(sq_launch(sample_rows_kernel, 1, 32, 0, st, ids, vals, k_list, n, rc, (SampleParams*)params, (const DraftQ*)qs));
    SQ_RETURN_LAUNCH();
}

extern "C" int quench6k_spec_layout(int field) {
    if (field == 4) return (int)sizeof(DraftQ);
    if (field == 5) return CATCH;
    if (field == 6) return CNT;
    switch (field) {
        case 0: return NMAX;
        case 1: return DRAFT;
        case 2: return GREEDY;
        default: return GREEDY + NMAX + 1;  // total ints
    }
}

extern "C" cudaError_t quench6k_spec_mtp_header(const int* rc, int* hdr, int tokens_off, int pos_off, int j, int c,
                                              cudaStream_t st) {
    if (c > NMAX + 1) return cudaErrorInvalidValue;
    SQ_CHECK_LAUNCH(sq_launch(mtp_header_kernel, 1, 32, 0, st, rc, hdr, tokens_off, pos_off, j, c));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_spec_accept(int* rc, int* ctl, int n, cudaStream_t st) {
    if (n > NMAX) return cudaErrorInvalidValue;
    SQ_CHECK_LAUNCH(sq_launch(accept_kernel, 1, 32, 0, st, rc, ctl, n));
    SQ_RETURN_LAUNCH();
}

extern "C" int quench6k_penalty_state_bytes() { return (int)sizeof(PenState); }

extern "C" cudaError_t quench6k_penalty_mark(void* st, unsigned* flags, int* list, const int* ids, int n, const int* n_ptr, unsigned bits,
                                           cudaStream_t s) {
    const int blocks = n_ptr != nullptr ? 1 : max(1, min(256, (n + 255) / 256));
    SQ_CHECK_LAUNCH(sq_launch(penalty_mark_kernel, blocks, 256, 0, s, (PenState*)st, flags, list, ids, n, n_ptr, bits));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t quench6k_penalty_apply(float* logits, int rows, int ld, const void* st, const unsigned* flags, const int* list,
                                            const int* rc, cudaStream_t s) {
    if (rows > NMAX + 1 || (rc == nullptr && rows != 1)) return cudaErrorInvalidValue;
    SQ_CHECK_LAUNCH(sq_launch(penalty_apply_kernel, dim3(rows, 32), 256, 0, s, logits, ld, (const PenState*)st, flags, list, rc));
    SQ_RETURN_LAUNCH();
}
