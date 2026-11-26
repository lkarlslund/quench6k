// Nemotron-H: shared descriptors and constants (crates/cuda/src/nemotron.rs mirrors them).
#pragma once

#include <cuda_fp8.h>

#include "common.cuh"

namespace nem {

using bf16 = __nv_bfloat16;

constexpr int MAX_SEQ = 16;

// A step's sequences: sequence s has rows row0[s] .. row0[s + 1] from position start[s], its Mamba block at
// mamba[s] and its attention cache at kv[s].
struct Seqs {
    unsigned long long mamba[MAX_SEQ];
    unsigned long long kv[MAX_SEQ];
    int start[MAX_SEQ];
    int row0[MAX_SEQ + 1];
    int nseq;
};

struct MambaW {
    const float *conv_w, *conv_b, *dt_bias, *a_log, *d, *norm_w;
    float dt_min, dt_max;
};

struct NvExperts {
    const uint8_t* codes;
    const uint8_t* scales;
    const float* s2;
};

constexpr int HIDDEN = 4096;
constexpr int HEADS = 128, HEAD_DIM = 64, STATE = 128, GROUPS = 8, HPG = HEADS / GROUPS;
constexpr int INNER = HEADS * HEAD_DIM, CONV_DIM = INNER + 2 * GROUPS * STATE, TAPS = 4;
constexpr int XBC_COL = INNER, DT_COL = INNER + CONV_DIM;
constexpr int NORM_GROUP = INNER / GROUPS;

constexpr int HEADER_BYTES = 256;
constexpr size_t SSM_BYTES = (size_t)HEADS * HEAD_DIM * STATE * 4;
constexpr size_t CONV_BYTES = (size_t)3 * CONV_DIM * 4;
// Record row: dt FP32 [HEADS], raw xBC BF16 [CONV_DIM], convolved xBC BF16 [CONV_DIM].
constexpr size_t REC_ROW_BYTES = (size_t)HEADS * 4 + (size_t)2 * CONV_DIM * 2;

constexpr int EXPERTS = 512, TOPK = 22, LATENT = 1024, MOE_INTER = 2688;
constexpr int SMALL_ROWS = 16;

constexpr int QH = 32, KVH = 2, HD = 128, QPK = QH / KVH;
constexpr int KV_ROW_BYTES = 2 * KVH * HD;
constexpr int ATTN_TILE = 64, ATTN_GROUP = 1024, GROUP_TILES = ATTN_GROUP / ATTN_TILE;

SQ_DEVICE float rb(float x) { return bf2f(f2bf(x)); }

SQ_DEVICE size_t layer_bytes(int rec_rows) { return SSM_BYTES + CONV_BYTES + 2 * (size_t)rec_rows * REC_ROW_BYTES; }

SQ_DEVICE unsigned char* layer_base(const Seqs& q, int s, int layer, int rec_rows) {
    return reinterpret_cast<unsigned char*>(q.mamba[s]) + HEADER_BYTES + (size_t)layer * layer_bytes(rec_rows);
}
SQ_DEVICE float* ssm_state(const Seqs& q, int s, int layer, int rec_rows) {
    return reinterpret_cast<float*>(layer_base(q, s, layer, rec_rows));
}
SQ_DEVICE float* conv_state(const Seqs& q, int s, int layer, int rec_rows) {
    return reinterpret_cast<float*>(layer_base(q, s, layer, rec_rows) + SSM_BYTES);
}
// Record row i of parity `parity`.
SQ_DEVICE unsigned char* rec_row(const Seqs& q, int s, int layer, int rec_rows, int parity, int i) {
    return layer_base(q, s, layer, rec_rows) + SSM_BYTES + CONV_BYTES + ((size_t)parity * rec_rows + i) * REC_ROW_BYTES;
}
SQ_DEVICE float* rec_dt(unsigned char* row) { return reinterpret_cast<float*>(row); }
SQ_DEVICE bf16* rec_raw(unsigned char* row) { return reinterpret_cast<bf16*>(row + HEADS * 4); }
SQ_DEVICE bf16* rec_conv(unsigned char* row) { return reinterpret_cast<bf16*>(row + HEADS * 4 + CONV_DIM * 2); }
// Header: replay count, record parity.
SQ_DEVICE int* header(const Seqs& q, int s) { return reinterpret_cast<int*>(q.mamba[s]); }

// Sequence of row r (-1 when r is past the descriptor's rows).
SQ_DEVICE int seq_of_row(const Seqs& q, int r) {
    const int ns = q.nseq;
    for (int s = 0; s < ns; ++s)
        if (r < q.row0[s + 1]) return r >= q.row0[s] ? s : -1;
    return -1;
}

}  // namespace nem

namespace nem {
// The descriptor into shared memory, read through L2 (it may have been rewritten since an earlier kernel
// ran on this SM). Call after pdl_wait; ends with a block barrier.
SQ_DEVICE void load_seqs(Seqs& dst, const Seqs* src) {
    static_assert(sizeof(Seqs) % 4 == 0, "Seqs size");
    const int* s = reinterpret_cast<const int*>(src);
    int* d = reinterpret_cast<int*>(&dst);
    for (int i = threadIdx.x; i < (int)(sizeof(Seqs) / 4); i += blockDim.x) d[i] = __ldcg(s + i);
    __syncthreads();
}

// As load_seqs, plus each sequence's Mamba header (replay count, record parity) into ctl.
SQ_DEVICE void load_seqs(Seqs& dst, const Seqs* src, int (&ctl)[MAX_SEQ][2]) {
    load_seqs(dst, src);
    const int s = threadIdx.x;
    if (s < dst.nseq) {
        const int* hdr = header(dst, s);
        ctl[s][0] = __ldcg(hdr);
        ctl[s][1] = __ldcg(hdr + 1);
    }
    __syncthreads();
}
}  // namespace nem
