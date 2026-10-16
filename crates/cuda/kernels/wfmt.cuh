// Projection weight formats and their loaders, shared by the GEMV-style kernels.
#pragma once

#include <cuda_fp8.h>

#include "common.cuh"

// MX8: FP8 E4M3 with a power-of-two scale per 32 consecutive weights of a row (scale bytes [N, K / 32],
// UE8M0: 2^(byte - 127)).
enum class WFmt { BF16 = 0, FP8 = 1, INT8 = 2, INT8G = 3, MX8 = 4 };

SQ_DEVICE float ue8m0_value(uint8_t b) { return __uint_as_float((uint32_t)b << 23); }

// INT8G: INT8 with one FP32 scale per GROUP consecutive weights of a row (scales [N, K / GROUP]).
constexpr int GROUP = 64;

// Loads 8 consecutive weights (16 bytes of BF16 or 8 bytes of FP8) as floats.
template <WFmt F>
SQ_DEVICE void load8(const uint8_t* row, int k, float* w);

template <>
SQ_DEVICE void load8<WFmt::BF16>(const uint8_t* row, int k, float* w) {
    const uint4 v = __ldg(reinterpret_cast<const uint4*>(row + (size_t)k * 2));
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(p[i]);
        w[2 * i] = f.x;
        w[2 * i + 1] = f.y;
    }
}

template <>
SQ_DEVICE void load8<WFmt::FP8>(const uint8_t* row, int k, float* w) {
    const uint2 v = __ldg(reinterpret_cast<const uint2*>(row + k));
    const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3);
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
        w[2 * i] = f.x;
        w[2 * i + 1] = f.y;
    }
}

template <>
SQ_DEVICE void load8<WFmt::INT8>(const uint8_t* row, int k, float* w) {
    const uint2 v = __ldg(reinterpret_cast<const uint2*>(row + k));
    const int8_t* p = reinterpret_cast<const int8_t*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = (float)p[i];
}

template <>
SQ_DEVICE void load8<WFmt::INT8G>(const uint8_t* row, int k, float* w) {
    load8<WFmt::INT8>(row, k, w);
}

template <WFmt F>
constexpr int elem_bytes() {
    return F == WFmt::BF16 ? 2 : 1;
}

// Loads 8 consecutive activations as floats from BF16 or FP32 storage.
SQ_DEVICE void load_x8(const __nv_bfloat16* x, float* out) {
    const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x));
    const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(xp[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}
SQ_DEVICE void load_x8(const float* x, float* out) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(x)), b = __ldg(reinterpret_cast<const float4*>(x + 4));
    out[0] = a.x, out[1] = a.y, out[2] = a.z, out[3] = a.w, out[4] = b.x, out[5] = b.y, out[6] = b.z, out[7] = b.w;
}

// Loads 16 consecutive weights starting at element k as floats (scale not applied).
template <WFmt F>
SQ_DEVICE void load16(const uint8_t* row, int k, float* w) {
    if constexpr (F == WFmt::BF16) {
        load8<F>(row, k, w);
        load8<F>(row, k + 8, w + 8);
    } else {
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(row + k));
        if constexpr (F == WFmt::FP8) {
            const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&v);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3);
                const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
                w[2 * i] = f.x;
                w[2 * i + 1] = f.y;
            }
        } else {
            const int8_t* p = reinterpret_cast<const int8_t*>(&v);
#pragma unroll
            for (int i = 0; i < 16; ++i) w[i] = (float)p[i];
        }
    }
}

// Raw storage of 16 consecutive weights: one 16-byte vector for 8-bit formats, two for BF16.
template <WFmt F>
struct Raw16 {
    uint4 v[F == WFmt::BF16 ? 2 : 1];
};

template <WFmt F>
SQ_DEVICE Raw16<F> load_raw16(const uint8_t* row, int k) {
    Raw16<F> r;
    if constexpr (F == WFmt::BF16) {
        r.v[0] = __ldg(reinterpret_cast<const uint4*>(row + (size_t)k * 2));
        r.v[1] = __ldg(reinterpret_cast<const uint4*>(row + (size_t)k * 2 + 16));
    } else {
        r.v[0] = __ldg(reinterpret_cast<const uint4*>(row + k));
    }
    return r;
}

template <WFmt F>
SQ_DEVICE void decode_raw16(const Raw16<F>& r, float* w) {
    if constexpr (F == WFmt::BF16) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&r.v[h]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __bfloat1622float2(p[i]);
                w[8 * h + 2 * i] = f.x;
                w[8 * h + 2 * i + 1] = f.y;
            }
        }
    } else if constexpr (F == WFmt::FP8 || F == WFmt::MX8) {
        const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&r.v[0]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3);
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
            w[2 * i] = f.x;
            w[2 * i + 1] = f.y;
        }
    } else {
        const int8_t* p = reinterpret_cast<const int8_t*>(&r.v[0]);
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = (float)p[i];
    }
}
