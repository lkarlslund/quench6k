// NVFP4 (E2M1 values, E4M3 block scales per 16) helpers shared by the quantization kernels.
#pragma once

#include <cuda_fp8.h>

#include "common.cuh"

// Round-to-nearest-even onto the E2M1 grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}, saturating; returns the 4-bit code.
// Branchless: ties go to the even code (0.25 -> 0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0, 2.5 -> 2,
// 3.5 -> 4, 5 -> 4).
SQ_DEVICE uint8_t e2m1_code(float x) {
    const float a = fabsf(x);
    const uint8_t c = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
    return c | (x < 0 && c != 0 ? 8 : 0);
}

SQ_DEVICE float e2m1_value(uint8_t c) {
    const int m = c & 7;
    const float a = m < 4 ? 0.5f * m : (m == 7 ? 6.0f : (float)(m - 2));
    return c & 8 ? -a : a;
}

SQ_DEVICE float ue4m3_value(int code) {
    __nv_fp8_e4m3 v;
    v.__x = (__nv_fp8_storage_t)code;
    return float(v);
}

// E2M1 code magnitude (0..7) of q >= 0 as e2m1_code rounds it, by arithmetic: below 2 the grid step is
// 0.5, from 2 to 4 it is 1, then 4 or 6 (rintf rounds ties to even, as the thresholds do).
SQ_DEVICE float e2m1_mag(float q) { return q < 2.0f ? rintf(2.0f * q) : (q < 4.0f ? rintf(q) + 2.0f : (q > 5.0f ? 7.0f : 6.0f)); }

// Magnitude of the E2M1 value of code magnitude m (as e2m1_value).
SQ_DEVICE float e2m1_mag_value(float m) { return m < 4.0f ? 0.5f * m : (m == 7.0f ? 6.0f : m - 2.0f); }

// Squared error of one 16-value block quantized with block scale `bs`, and its 16 codes (sign in bit 3,
// as e2m1_code) packed low nibble first. The same values as e2m1_code / e2m1_value give: the error of a
// negative value is the negated difference, whose square is the same.
SQ_DEVICE float fp4_block_error_codes(const float* v, float bs, uint2& packed) {
    packed = make_uint2(0, 0);
    if (bs <= 0.0f) return 3.4e38f;
    float e = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float a = fabsf(v[i]);
        const float m = e2m1_mag(a / bs);
        const float d = a - e2m1_mag_value(m) * bs;
        e = fmaf(d, d, e);
        const uint32_t c = (uint32_t)m | (v[i] < 0.0f && m != 0.0f ? 8u : 0u);
        if (i < 8)
            packed.x |= c << (4 * i);
        else
            packed.y |= c << (4 * (i - 8));
    }
    return e;
}

SQ_DEVICE float fp4_block_error(const float* v, float bs) {
    uint2 unused;
    return fp4_block_error_codes(v, bs, unused);
}

// The E4M3 block scale code for 16 values (already divided by the row scale): the nearest code to
// amax / 6 or one of its +-1, +-2 neighbours, whichever quantizes the block with the least squared
// error (as llama.cpp's NVFP4 activation quantizer). `packed`: the block's codes at that scale (zero
// when the scale is zero).
SQ_DEVICE int fp4_block_scale(const float* v, uint2& packed) {
    float am = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) am = fmaxf(am, fabsf(v[i]));
    const int first = (int)__nv_fp8_e4m3(am / 6.0f).__x;
    int best = first;
    float best_e = fp4_block_error_codes(v, ue4m3_value(first), packed);
    constexpr int offs[4] = {-1, 1, -2, 2};
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int c = first + offs[t];
        if (c < 0 || c > 0x7e) continue;
        uint2 q;
        const float e = fp4_block_error_codes(v, ue4m3_value(c), q);
        if (e < best_e) best_e = e, best = c, packed = q;
    }
    return best;
}

SQ_DEVICE int fp4_block_scale(const float* v) {
    uint2 unused;
    return fp4_block_scale(v, unused);
}
