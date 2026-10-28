# Benchmark data

- `context.ids`: the prompt corpus, 261,760 space-separated token IDs (Qwen3.8 Flash-Next tokenizer,
  no special tokens, no chat template). It is the corpus of NInfer 6000's Flash-Next benchmark
  (SHA-256 of the IDs `e99ca11f…3f85`), so both engines are measured on identical prompts;
  `quench6k bench --ctx N` uses its first N tokens. The text is tiled, so long prompts repeat it and
  MTP acceptance on it is high.
- `results.jsonl`: quench6k measurements, written by `quench6k bench --json`.
- `radixark.jsonl`: quench6k on the RadixArk/Qwen3.8-Flash-Next-NVFP4 checkpoint (not in the chart).
- `ninfer-6000.jsonl`: NInfer 6000 measurements on the same machine (8-bit variant, BF16 KV, 600 W).
- `flint.jsonl`: quench6k on Recursive Flint (vendor NVFP4 GGUF), written by `quench6k flint bench
  --json`, 600 W. The corpus is the text of `context.ids` tokenized with Flint's tokenizer (280,513
  tokens); `copied` counts greedy tokens equal to the corpus continuation (the text is tiled, so it is
  128 of 128 once the prompt has repeated).
- `flint-llamacpp.jsonl`: llama.cpp (c7fc50941, `llama-bench -fa 1 -ub 2048`, F16 KV) on the same
  file and machine: `pp<ctx>` for prefill and `tg128 @ d<ctx>` for decode.
- `kbench.jsonl`: the experiment ledger, one record per kernel configuration measured by `quench6k
  kbench --log` (fields in [dev/KBENCH.md](../dev/KBENCH.md) "Experiment ledger"). Numbers and
  settings only: prompts from real conversations (`quench6k context-ids`) never go in this directory.
- `results.svg`: rendered from both by `crates/benchplot` (see [DEVELOPMENT.md](../DEVELOPMENT.md)).

One JSON object per line:

| Field | |
|---|---|
| `engine` | `quench6k` or `ninfer` |
| `date` | UTC date of the run |
| `mode` | `plain` or `mtp` |
| `drafts`, `adaptive` | MTP draft length, and whether each round picks 1..drafts |
| `sampled` | Sampled with the server's defaults instead of greedy (absent: greedy) |
| `kv` | `bf16` or `fp8` |
| `ctx` | Prompt length in tokens |
| `generate` | Tokens generated for the decode measurement |
| `prefill_tok_s`, `decode_tok_s` | Throughput |
| `tokens_per_round` | MTP tokens per draft-and-verify round |
| `gpu` | GPU during the measurement (NVML, every 200 ms): `temp_max_c`, `sm_mhz_min`, `sm_mhz_mean`, `power_w_mean`, `power_limit_w` (enforced limit; from 2026-10-27), `clock_events` (`power_cap` is normal under load; `sw_thermal`, `hw_thermal`, `hw_slowdown`, `hw_power_brake` set `slowed`) |

Later records replace earlier ones with the same engine, mode, drafts, adaptive and sampled flags, KV
format and context.
