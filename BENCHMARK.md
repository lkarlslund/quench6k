# Benchmark

![Decode and prefill throughput by context length](bench/results.svg)

One request on an RTX PRO 6000 Blackwell at 600 W, compared with NInfer 6000 measured on the same
machine. Tokens per second. Dotted lines with open markers are the FP8 KV cache (`--kv fp8`, the
serving configuration), solid lines BF16.

**Greedy and sampled.** Greedy decoding takes the most likely token; normal requests sample at
temperature 1.0 (top_p 0.95, top_k 20). Sampled text is less predictable, so the MTP head's drafts are
accepted less often and each round commits fewer tokens.

<!-- bench:begin -->
BF16 KV cache (the NInfer 6000 baseline's configuration):

| Context | Decode | NInfer decode | MTP greedy, 3 drafts | NInfer MTP, 3 drafts | MTP greedy, adaptive ≤ 6 | MTP sampled, adaptive ≤ 6 | Prefill | NInfer prefill |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 194.1 | 172.0 | 346.1 | 274.8 | 340.9 | 287.5 | 6,595 | 5,905 |
| 1K | 191.5 | – | 277.8 | – | 275.5 | 246.5 | 8,905 | – |
| 8K | 189.5 | 170.6 | 454.0 | 401.3 | 563.7 | 556.4 | 15,058 | 13,908 |
| 64K | 184.5 | 153.9 | 439.6 | 380.3 | 540.2 | 533.1 | 14,352 | 12,171 |
| 128K | 180.7 | 149.8 | 433.7 | 368.0 | 529.6 | 523.1 | 13,909 | 11,032 |
| 256K | 177.3 | 149.0 | 422.6 | 360.6 | 521.1 | 515.5 | 13,182 | 9,154 |

FP8 KV cache (`--kv fp8`, the serving configuration):

| Context | Decode | MTP greedy, adaptive ≤ 6 | MTP sampled, adaptive ≤ 6 | Prefill |
|---:|---:|---:|---:|---:|
| 512 | 197.9 | 301.9 | 322.0 | 6,700 |
| 1K | 196.8 | 329.2 | 284.9 | 8,973 |
| 8K | 194.3 | 581.8 | 573.2 | 15,099 |
| 64K | 189.4 | 557.9 | 549.8 | 14,283 |
| 128K | 182.5 | 542.0 | 536.0 | 13,777 |
| 256K | 179.9 | 537.4 | 529.7 | 12,373 |
<!-- bench:end -->

Real prompts through the server (`--kv fp8`, sampled at temperature 0.6 and 1.0 with top_p 0.95 and
top_k 20, 600-token answers to a coding, an explanation and a story prompt): 223–338 tokens/s,
282 overall. Code is fastest, prose slowest.

Two concurrent requests (`batch-check`, greedy, the code and multiturn fixtures, 192 tokens each):
418 tokens/s together with speculative pair rounds (3.4 tokens per request and round), 276 with
batched plain steps.

## Inkling 2 Mini

One request, 600 W, greedy, 128 tokens generated after the first N tokens of the Qwen corpus (`bench/context.ids`) tokenized
with Inkling's tokenizer (`quench6k inkling bench --generate 128`, BF16 caches). Plain decoding, and
decoding with the checkpoint's DSpark drafter (tokens per step in parentheses). Prefill is the same work
in both runs.

| Context | Prefill | Decode, plain | Decode, DSpark drafts |
|---:|---:|---:|---:|
| 512 | 6,228 | 250.8 | 265.3 (1.76) |
| 8K | 13,391 | 234.6 | 427.1 (3.85) |
| 64K | 9,926 | 175.6 | 341.9 (5.29) |
| 128K | 7,637 | 140.6 | 177.2 (3.10) |
| 260,000 | 5,210 | 101.0 | 108.2 (2.40) |

Past 512 the corpus repeats, so the 128 tokens continue it exactly in every run. One sequence of
1,042,346 tokens prefills at 1.7K tokens/s and decodes at 34 tokens/s with 38.9 GiB of caches (450 W).
Data: `bench/inkling.jsonl` (with the GPU's temperature, clocks and power during each run).

## Nemotron 3 Super / Munin 2.0

One request, 600 W, greedy, 128 tokens generated after the first N tokens of the Qwen corpus
(`bench/context.ids`) tokenized with Nemotron's tokenizer (`quench6k nemotron bench --generate 128
--drafts 0 3`, BF16 caches), on Munin 2.0 (NVIDIA's checkpoint has the same layers, formats and
tokenizer). Plain decoding, and decoding with three drafts of the checkpoint's MTP layer per step
(tokens per step in parentheses). Prefill is the same work in both runs, in steps of up to 2048 tokens;
the table shows the plain run's.

| Context | Prefill | Decode, plain | Decode, 3 MTP drafts |
|---:|---:|---:|---:|
| 512 | 4,959 | 130.1 | 178.7 (2.54) |
| 8K | 7,153 | 127.8 | 246.5 (3.53) |
| 32K | 6,765 | 124.3 | 218.9 (3.26) |

Long contexts, one request (each context in a run of its own):

| Context | Prefill | Decode, plain | Decode, 3 MTP drafts |
|---:|---:|---:|---:|
| 128K | 5,281 | 119.1 | 235.9 (3.63) |
| 260,000 | 3,977 | 106.9 | 211.6 (3.53) |

Several sequences decoded together (`--seqs N`: N windows of the corpus, each its own prompt of the same
length, in batched steps; each sequence's tokens are the same as alone). Decode tokens per second summed
over the sequences; with drafts, tokens per sequence step in parentheses:

| Context | Drafts | 1 | 2 | 4 | 8 |
|---:|---|---:|---:|---:|---:|
| 8K | none | 127.8 | 204.9 | 299.4 | 420.6 |
| 8K | 3 | 246.5 (3.53) | 365.9 (3.48) | 497.0 (3.48) | 521.1 (3.39) |

The server prefills several whole 2048-token chunks in one step (up to 8192 tokens) while no other
request decodes; measured with such steps (`QUENCH6K_NEM_MAXT=8192`), prefill is 7,467 tokens/s at 8K
and 7,656 at 32K (a 512-token prompt is a single chunk either way).

The corpus repeats its text after about 1K tokens, so from 8K on the greedy tokens continue it (128 of
128 at 8K, 100 at 32K) and drafts are accepted far more often than on real text; the served requests
below (median decode rate per request) are the realistic figure. During the runs the GPU stayed at its
power cap with no thermal slowdown: SM clock 2.57-2.81 GHz on average, 47-87 °C (the hottest at
260,000). Data: `bench/nemotron.jsonl` (with the GPU's temperature, clocks and power during each run).

## Several requests at once

Served requests started together, each its own ~1K-token prompt asking for a long answer (320 or 480
tokens, all reaching max_tokens), greedy, drafts on, 600 W. Decode tokens per second of the steps with
that many sequences, summed over the requests:

| Model | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Inkling 2 Mini | 392-404 | - | 617-618 | 749-772 |
| Hy4-flash | 126 | 170 | 252 | 366 |
| Recursive Flint | 543-545 | 964-974 | 1,259-1,260 | 1,588-1,589 |

A request's tokens do not depend on which others share its steps: the same prompts give the same text
alone and eight at once. Details and the before/after of each change: [dev/KBENCH.md](dev/KBENCH.md).

Munin 2.0 served (`--slots 8 --context 65536`, MTP drafts on, BF16 caches), multi-turn conversations
from `tools/loadgen.py consistency --shared-system --between none` (each conversation alone, then all at
once on the same server): 4 conversations of 8 turns growing to 24.8K tokens, greedy and sampled
(temperature 0.8, seeded), and 8 conversations of 5 turns growing to 16.7K, greedy. Every conversation's
text was identical alone and together (4/4, 4/4, 8/8; no errors). Prompt tokens served from cache: 71% /
72% / 58% alone, 65% / 66% / 60% together (a turn recomputes back to the last 2048-token boundary).
Decode tokens per second per request (median): 162 / 154 / 166 alone, 115 / 74 / 149 together, with
other requests' prefills sharing the GPU.

Recursive Flint against llama.cpp, one request by context: [README](README.md#recursive-flint-against-llamacpp).

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
