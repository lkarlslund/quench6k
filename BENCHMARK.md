# Benchmark

![Decode and prefill throughput by context length](bench/results.svg)

One request on an RTX PRO 6000 Blackwell at 600 W, compared with NInfer 6000 measured on the same
machine. Tokens per second.

<!-- bench:begin -->
BF16 KV cache (the NInfer 6000 baseline's configuration):

| Context | Decode | NInfer decode | MTP, 3 drafts | NInfer MTP, 3 drafts | MTP, adaptive ≤ 6 | Prefill | NInfer prefill |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 194.1 | 172.0 | 316.3 | 274.8 | 313.1 | 6,595 | 5,905 |
| 1K | 191.5 | – | 253.2 | – | 250.9 | 8,905 | – |
| 8K | 189.5 | 170.6 | 415.3 | 401.3 | 518.8 | 15,058 | 13,908 |
| 64K | 184.5 | 153.9 | 403.3 | 380.3 | 497.5 | 14,352 | 12,171 |
| 128K | 180.7 | 149.8 | 398.7 | 368.0 | 488.8 | 13,909 | 11,032 |
| 256K | 177.3 | 149.0 | 389.9 | 360.6 | 482.1 | 13,182 | 9,154 |

FP8 KV cache (`--kv fp8`, the serving configuration):

| Context | Decode | MTP, adaptive ≤ 6 | Prefill |
|---:|---:|---:|---:|
| 512 | 194.3 | 256.3 | 6,616 |
| 1K | 191.7 | 314.3 | 8,885 |
| 8K | 189.3 | 518.4 | 14,298 |
| 64K | 184.2 | 496.4 | 13,490 |
| 128K | 180.3 | 482.8 | 13,064 |
| 256K | 177.2 | 480.8 | 12,365 |
<!-- bench:end -->

Real prompts through the server (`--kv fp8`, sampled at temperature 0.6 and 1.0 with top_p 0.95 and
top_k 20, 600-token answers to a coding, an explanation and a story prompt): 223–338 tokens/s,
282 overall. Code is fastest, prose slowest.

## Method

- `quench6k bench`: the prompt is the first N tokens of `bench/context.ids` (the 256K row uses
  260,000 tokens, the corpus length); decode generates 256 tokens after it, greedy. Plain decode
  reports the median of three runs of 85 tokens.
- Every run first checks 64 greedy tokens of the code fixture against the reference. Each context is
  measured after an unreported run at the same context, so the embedding rows of the generated
  tokens are in the page cache, as in a running server.
- MTP rows run with the MTP head's cache filled during prefill, as the server does; their prefill
  includes that work.
- From 8K on, the corpus repeats its text, so MTP drafts are almost always accepted (about 7 tokens
  per round). At 512 and 1K it does not repeat yet, and rounds commit 2–3 tokens, as on real prompts.
- Data: `bench/results.jsonl` and `bench/ninfer-6000.jsonl`. The tables and the chart are generated
  from them with `crates/benchplot` ([DEVELOPMENT.md](DEVELOPMENT.md#benchmarks)).
