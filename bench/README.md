# Benchmark data

- `context.ids`: the prompt corpus, 261,760 space-separated token IDs (Qwen3.8 Flash-Next tokenizer,
  no special tokens, no chat template). It is the corpus of NInfer 6000's Flash-Next benchmark
  (SHA-256 of the IDs `e99ca11f…3f85`), so both engines are measured on identical prompts;
  `quench6k bench --ctx N` uses its first N tokens. The text is tiled, so long prompts repeat it and
  MTP acceptance on it is high.
- `results.jsonl`: quench6k measurements, written by `quench6k bench --json`.
- `ninfer-6000.jsonl`: NInfer 6000 measurements on the same machine (8-bit variant, BF16 KV, 600 W).
- `results.svg`: rendered from both by `crates/benchplot` (see [DEVELOPMENT.md](../DEVELOPMENT.md)).

One JSON object per line:

| Field | |
|---|---|
| `engine` | `quench6k` or `ninfer` |
| `date` | UTC date of the run |
| `mode` | `plain` or `mtp` |
| `drafts`, `adaptive` | MTP draft length, and whether each round picks 1..drafts |
| `kv` | `bf16` or `fp8` |
| `ctx` | Prompt length in tokens |
| `generate` | Tokens generated for the decode measurement |
| `prefill_tok_s`, `decode_tok_s` | Throughput |
| `tokens_per_round` | MTP tokens per draft-and-verify round |

Later records replace earlier ones with the same engine, mode, drafts, adaptive flag, KV format and
context.
