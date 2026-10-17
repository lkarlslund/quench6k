# Benchmark

![Decode and prefill throughput by context length](bench/results.svg)

One request on an RTX PRO 6000 Blackwell at 600 W, compared with NInfer 6000 measured on the same
machine. Tokens per second.

**Greedy and sampled.** Greedy decoding takes the most likely token; normal requests sample at
temperature 1.0 (top_p 0.95, top_k 20). Sampled text is less predictable, so the MTP head's drafts are
accepted less often and each round commits fewer tokens.

<!-- bench:begin -->
BF16 KV cache (the NInfer 6000 baseline's configuration):

| Context | Decode | NInfer decode | MTP greedy, 3 drafts | NInfer MTP, 3 drafts | MTP greedy, adaptive ≤ 6 | MTP sampled, adaptive ≤ 6 | Prefill | NInfer prefill |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 194.1 | 172.0 | 316.3 | 274.8 | 313.1 | 264.8 | 6,595 | 5,905 |
| 1K | 191.5 | – | 253.2 | – | 250.9 | 234.5 | 8,905 | – |
| 8K | 189.5 | 170.6 | 415.3 | 401.3 | 518.8 | 513.6 | 15,058 | 13,908 |
| 64K | 184.5 | 153.9 | 403.3 | 380.3 | 497.5 | 491.8 | 14,352 | 12,171 |
| 128K | 180.7 | 149.8 | 398.7 | 368.0 | 488.8 | 481.9 | 13,909 | 11,032 |
| 256K | 177.3 | 149.0 | 389.9 | 360.6 | 482.1 | 474.4 | 13,182 | 9,154 |

FP8 KV cache (`--kv fp8`, the serving configuration):

| Context | Decode | MTP greedy, adaptive ≤ 6 | MTP sampled, adaptive ≤ 6 | Prefill |
|---:|---:|---:|---:|---:|
| 512 | 194.3 | 256.3 | 255.1 | 6,616 |
| 1K | 191.7 | 314.3 | 256.1 | 8,885 |
| 8K | 189.3 | 518.4 | 514.5 | 14,298 |
| 64K | 184.2 | 496.4 | 492.7 | 13,490 |
| 128K | 180.3 | 482.8 | 480.7 | 13,064 |
| 256K | 177.2 | 480.8 | 477.8 | 12,365 |
<!-- bench:end -->

Real prompts through the server (`--kv fp8`, sampled at temperature 0.6 and 1.0 with top_p 0.95 and
top_k 20, 600-token answers to a coding, an explanation and a story prompt): 223–338 tokens/s,
282 overall. Code is fastest, prose slowest.

Two concurrent requests (`batch-check`, greedy, the code and multiturn fixtures, 192 tokens each):
418 tokens/s together with speculative pair rounds (3.4 tokens per request and round), 276 with
batched plain steps.

## Method

- `quench6k bench`: the prompt is the first N tokens of `bench/context.ids` (the 256K row uses
  260,000 tokens, the corpus length); decode generates 256 tokens after it, greedy. Plain decode
  reports the median of three runs of 85 tokens.
- Every run first checks 64 greedy tokens of the code fixture against the reference. Each context is
  measured after an unreported run at the same context, so the embedding rows of the generated
  tokens are in the page cache, as in a running server.
- Sampled rows use the server's default parameters and report the mean over four seeds of 256
  tokens each. They are checked through the same sampling kernels with top_k 1 against the
  reference, and print the start of their output.
- Each measurement records the GPU's temperature, SM clock, power and clock-event reasons (NVML) and
  warns when the GPU slowed down. Prefill depends on the SM clock, which falls as the GPU heats
  (2.8 GHz cool, about 2.25 GHz at 90 °C); decode is bound by memory bandwidth and barely changes.
- MTP rows run with the MTP head's cache filled during prefill, as the server does; their prefill
  includes that work.
- From 8K on, the corpus repeats its text, so MTP drafts are almost always accepted (about 7 tokens
  per round), greedy or sampled. At 512 and 1K it does not repeat yet, and rounds commit 2–3 tokens,
  as on real prompts.
- Data: `bench/results.jsonl` and `bench/ninfer-6000.jsonl`. The tables and the chart are generated
  from them with `crates/benchplot` ([DEVELOPMENT.md](DEVELOPMENT.md#benchmarks)).
