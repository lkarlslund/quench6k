// Utility kernels: byte checksums for verifying uploads.

#include "common.cuh"

namespace {

// Wrapping sum of little-endian u32 words; a trailing partial word is zero-padded.
__global__ void checksum_kernel(const uint8_t* data, size_t bytes, unsigned long long* out) {
    pdl_launch_dependents();
    pdl_wait();
    const size_t words = bytes / 4;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(data);
    unsigned long long sum = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) {
        sum += w[i];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t tail = 0;
        for (size_t i = words * 4; i < bytes; ++i) tail |= uint32_t(data[i]) << (8 * (i - words * 4));
        sum += tail;
    }
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, sum);
}

}  // namespace

extern "C" cudaError_t squash_checksum(const void* data, size_t bytes, unsigned long long* out, cudaStream_t stream) {
    cudaError_t err = cudaMemsetAsync(out, 0, sizeof(*out), stream);
    if (err != cudaSuccess) return err;
    SQ_CHECK_LAUNCH(sq_launch(checksum_kernel, 264, 256, 0, stream, static_cast<const uint8_t*>(data), bytes, out));
    return cudaGetLastError();
}
