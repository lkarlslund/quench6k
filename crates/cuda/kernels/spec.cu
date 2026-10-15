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

}  // namespace

extern "C" int squash_spec_layout(int field) {
    switch (field) {
        case 0: return NMAX;
        case 1: return DRAFT;
        case 2: return GREEDY;
        default: return GREEDY + NMAX + 1;  // total ints
    }
}

extern "C" cudaError_t squash_spec_mtp_header(const int* rc, int* hdr, int tokens_off, int pos_off, int j, int c,
                                              cudaStream_t st) {
    if (c > NMAX + 1) return cudaErrorInvalidValue;
    SQ_CHECK_LAUNCH(sq_launch(mtp_header_kernel, 1, 32, 0, st, rc, hdr, tokens_off, pos_off, j, c));
    SQ_RETURN_LAUNCH();
}

extern "C" cudaError_t squash_spec_accept(int* rc, int* ctl, int n, cudaStream_t st) {
    if (n > NMAX) return cudaErrorInvalidValue;
    SQ_CHECK_LAUNCH(sq_launch(accept_kernel, 1, 32, 0, st, rc, ctl, n));
    SQ_RETURN_LAUNCH();
}
