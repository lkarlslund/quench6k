// Runtime kernel variants (dev/KBENCH.md): tuning parameters that launchers read when they launch
// (for CUDA graphs: at capture). Every key defaults to the engine's normal behaviour. Values come from
// quench6k_set_variant (kbench sweeps) or from QUENCH6K_VARIANTS="key=value,key=value" at startup.
#pragma once

// The value of `key`, or `fallback` when it is not set.
int sq_variant(const char* key, int fallback);

// The value of "key/n" (a setting for one shape, e.g. gemv.rows/16480 for a GEMV of 16480 output rows),
// else of `key`, else `fallback`.
int sq_variant_shape(const char* key, int n, int fallback);

// Row instance for a small batch of T rows: T itself when bit T of variant `key` is set, else the next
// power of two (a launcher without an instance for T rounds up to the next instance it has).
inline int sq_rows(int T, const char* key, int exact_default) {
    if (T >= 1 && T <= 8 && (sq_variant(key, exact_default) >> T & 1)) return T;
    int r = 1;
    while (r < T) r <<= 1;
    return r;
}

