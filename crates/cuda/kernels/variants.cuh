// Runtime kernel variants (docs/KBENCH.md): tuning parameters that launchers read when they launch
// (for CUDA graphs: at capture). Every key defaults to the engine's normal behaviour. Values come from
// quench6k_set_variant (kbench sweeps) or from QUENCH6K_VARIANTS="key=value,key=value" at startup.
#pragma once

// The value of `key`, or `fallback` when it is not set.
int sq_variant(const char* key, int fallback);
