# Oracle fixtures

Reference token distributions for the correctness gate in [GOAL.md](../GOAL.md). Each fixture
directory holds:

- `fixture.json`: prompt token IDs; `spec.json`: the rendered prompt text and generation settings;
- `sequence.json`: the canonical sequence (llama.cpp W4A4 greedy);
- `llama-w4a4.{json,bin}`: reference A, llama.cpp native NVFP4 W4A4, dynamic activation scales;
- `llama-q8.{json,bin}`: reference B, the same weights with 8-bit activations.

Each `.bin` holds little-endian arrays for `positions` P and `top_k` K, in this order: F32 log-sum-exp
`[P]`, F32 logit of the sequence token `[P]`, I32 top-K IDs `[P,K]`, F32 top-K logits `[P,K]`.

The fixtures were generated with llama.cpp from the
`swift-1.5-qwen3.8-flash-next-nvfp4.gguf` conversion of
`ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4`.
