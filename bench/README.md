# Benchmark corpus

`context.ids`: 261,760 space-separated token IDs (Qwen3.8 Flash-Next tokenizer, no special tokens, no
chat template). It is the context corpus of NInfer 6000's Flash-Next benchmark
(`bench/fixtures/qwen3_8_flash_next_context.ids`, SHA-256 `e99ca11f…3f85` of the IDs), copied as data
so both engines are measured on identical prompts: `squash bench --ctx N` uses its first N tokens.
The text is meaningful but tiled, so long prompts repeat it; MTP acceptance on it is high.
