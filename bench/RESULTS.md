# Benchmark, 2026-10-16 (afternoon)

RTX PRO 6000 Blackwell at 600 W, same machine as the NInfer 6000 baseline (GOAL.md §2). One request,
greedy, BF16 KV, `bench/context.ids` corpus, decode = 256 generated tokens after a prompt of the
given length (`squash bench --generate 256`; 256K is measured at 260,000 tokens, the corpus length).
Every run first checks 64 greedy tokens of the code fixture against the reference (all passed). The 512
row is the second, warm measurement. Correctness gate (GOAL.md §1): PASS on all fixtures for this
build (FP4 prefill experts, MXFP8 GDN input projections, tensor-core GDN, multi-query attention).

| Context | Decode | NInfer | MTP3 (LM-head draft) | NInfer MTP3 + LM-head draft | Adaptive MTP (≤6) | Prefill | NInfer prefill |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | **193.3** | 172.0 | **308.7** | 274.8 | 307.1 | **6,599** | 5,905 |
| 8K | **188.6** | 170.6 | **406.9** | 401.3 | 515.3 | **14,792** | 13,908 |
| 64K | **183.7** | 153.9 | **395.4** | 380.3 | 493.7 | **14,300** | 12,171 |
| 128K | **180.0** | 149.8 | **390.6** | 368.0 | 484.9 | **13,860** | 11,032 |
| 256K | **176.7** | 149.0 | **382.7** | 360.6 | 478.6 | **13,000** | 9,154 |

- Prefill at 8K and above: re-measured after the BF16-expanded INT8 GEMMs (below), +6-7%.
- Decode, MTP and prefill beat every NInfer cell (run-to-run variation on this machine is about 3%; the first, cold 8K run after other work measured 11,036 because the PLE pages
  were evicted, which the serving path warms in the background). The 8K and 512 rows are warm runs.
- With MTP enabled (the serving path) prefill also feeds the drafter's cache:
  14,125 / 13,188 at 8K / 128K (before the GEMM change: 13,245 / 12,230).
- The corpus repeats few distinct tokens, so its draft acceptance is near 100% (4.0 tokens per MTP3
  round); on real prompts acceptance is lower.
- Sampled requests (temperature > 0) use MTP too (exact rejection sampling against the deterministic
  drafts): 262-273 tok/s against 166-179 plain on three prompts at temperature 0.6 and 1.0, with the
  token distribution equal to plain sampling (two-sample test, 600 samples per position).

## Prefill, later the same day

- INT8-group dense weights are expanded to BF16 once per GEMM for batches of 2048+ rows (68 MiB
  workspace buffer) and multiplied with the BF16 kernel, which stages tiles with cp.async instead of
  converting them in registers: 8192x2560x6144 0.86 -> 0.64 ms. Bit-identical (gate outputs equal to
  the previous run byte for byte; kernel test). `SQUASH_DEQUANT_GEMM=0` disables.

## Prefill, what changed today (8K chunk: 1,108 ms -> about 585 ms on the GPU)

- FP4 tensor cores for the routed experts in prefill (W4A4: activations quantized per row with the
  least-error E4M3 block scale, as llama.cpp; gate PASS), gate/up/SwiGLU fused, BF16 expert outputs.
- MXFP8 (E4M3, power-of-two scale per 32) for the GDN input projections, prefill and decode (gate PASS;
  MXFP8 on all dense projections, or on the HyperConnections, fails the gate).
- Tensor-core chunked GDN (3xTF32, 32-token chunks, cp.async prefetch, one wave of CTAs).
- QSA: tensor-core indexer scores, one-block-per-row top-512 selection for prefill, attention over the
  union of 4 neighbouring queries' selections.
- HyperConnection: up-projection and mix fused, residual update fused into the next norm.
- Dense GEMM: XOR-swizzled shared memory (no padding), two CTAs per SM with three stages, 256x128
  tiles for large launches, split-K for grids smaller than the GPU (short prompts).
- Fusions (bit-identical): GDN conv + q/k normalization, MoE combine into the next HC norm, parallel
  PLE conv over token tiles.
- Host: PLE rows faulted in by 64 threads ahead of the GPU (MADV_RANDOM: no 128 KiB readahead per
  320-byte row), gathered on 16 threads, overlapping layer 0 on the GPU.

Restart (`squash serve`, GPU empty): about 31 s to the first answered request, most of it loading
72.5 GiB of weights from NVMe (the checkpoint does not fit the free page cache).
