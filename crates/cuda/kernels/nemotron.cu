// Nemotron-H kernels (API stubs; implementations follow).
#include "common.cuh"

namespace {
constexpr int MAX_SEQ = 16;
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
}  // namespace

extern "C" {
size_t quench6k_nem_seqs_bytes() { return sizeof(Seqs); }
cudaError_t quench6k_nem_mamba_conv(const float*, int, int, const void*, int, int, int, MambaW, void*, float*, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t quench6k_nem_mamba_conv_commit(const float*, int, int, const void*, int, int, int, const void*, const float*, int, int, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t quench6k_nem_mamba_scan(const void*, const float*, int, const void*, int, int, int, MambaW, int, int, void*, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t quench6k_nem_mamba_gated_norm(const void*, const float*, int, int, const float*, float, void*, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t quench6k_nem_mamba_set_ctl(const void*, int, const int*, int, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t quench6k_nem_moe_route(const float*, int, const float*, float, int, int*, float*, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t quench6k_nem_moe_experts_small(const void*, const int*, NvExperts, NvExperts, int, int*, void*, float*, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t quench6k_nem_moe_plan(const int*, int, int, int*, int*, int*, int*, int*, int*, int*, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t quench6k_nem_moe_gather(const void*, const int*, int, void*, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t quench6k_nem_relu2(const float*, int, int, int, void*, int, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t quench6k_nem_moe_combine(const float*, const int*, const float*, int, void*, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t quench6k_nem_attn_prep(const float*, int, int, int, int, int, const void*, long long, long long, float, float, void*, cudaStream_t) { return cudaErrorNotSupported; }
size_t quench6k_nem_attn_part_floats(int, int) { return 0; }
cudaError_t quench6k_nem_attention_small(const void*, int, const void*, long long, long long, float, float, int, float*, void*, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t quench6k_nem_attention_prefill(const void*, int, const void*, long long, long long, float, float, void*, cudaStream_t) { return cudaErrorNotSupported; }
}
