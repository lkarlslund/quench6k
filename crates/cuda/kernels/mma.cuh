// Tensor-core and async-copy primitives (sm_80+ mma.sync / cp.async / ldmatrix).
#pragma once

#include "common.cuh"

SQ_DEVICE void cp_async16(void* smem, const void* gmem, bool valid) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int size = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(size));
}
SQ_DEVICE void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
SQ_DEVICE void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

SQ_DEVICE void ldmatrix_x4(uint32_t* r, const void* smem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

SQ_DEVICE void ldmatrix_x2(uint32_t* r, const void* smem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}

// Transposing variant: for B fragments from a [k][n] (n-contiguous) tile.
SQ_DEVICE void ldmatrix_x4_trans(uint32_t* r, const void* smem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

SQ_DEVICE void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
