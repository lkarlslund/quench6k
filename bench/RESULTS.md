# Final benchmark (M12), 2026-10-16

RTX PRO 6000 Blackwell at 600 W, same machine as the NInfer 6000 baseline (GOAL.md §2). One request,
greedy, BF16 KV, `bench/context.ids` corpus, decode = 256 generated tokens after a prompt of the
given length (`squash bench --generate 256`; 256K is measured at 260,000 tokens, the corpus length).
Every run first checks 64 greedy tokens of the code fixture against the reference (all passed).

| Context | Decode | NInfer | MTP3 (LM-head draft) | NInfer MTP3 + LM-head draft | Adaptive MTP (≤6) | Prefill | NInfer prefill |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | **177.6** | 172.0 | **317.8** | 274.8 | 325.7 | 2,702 (1,643 cold) | 5,905 |
| 8K | **185.0** | 170.6 | **403.6** | 401.3 | 509.8 | 5,873 | 13,908 |
| 64K | **178.5** | 153.9 | **389.0** | 380.3 | 484.2 | 6,314 | 12,171 |
| 128K | **176.4** | 149.8 | **385.4** | 368.0 | 478.8 | 5,989 | 11,032 |
| 256K | **172.9** | 149.0 | **376.1** | 360.6 | 468.5 | 5,484 | 9,154 |

- Decode and MTP beat every NInfer cell. The corpus repeats few distinct tokens, so its draft
  acceptance is near 100% (4.0 tokens per MTP3 round); on the oracle fixtures adaptive drafting
  gives 454 (code), 282 (multiturn), 335 (math), 340 (tool) tok/s.
- Prefill does not beat NInfer: the correctness gate rejects low-precision activations (NVFP4 W4A4
  and INT8 activations both fail it in fake-quant experiments), so prefill GEMMs run on BF16
  activations.
- FP8 KV (`--kv fp8`, 47% less K/V memory) passes the gate and measures the same decode speed
  (185.2 / 173.3 plain, 404.1 / 376.6 MTP3 at 8K / 260K) with prefill within 1%.

Batch of two (`squash batch-check`, code + multiturn prompts, 256 tokens each): 269.9 tok/s eager,
245.6 tok/s graph-captured, both sequences token-identical to single-sequence decoding; the gate
passes at batch 2 with distributions byte-identical to batch 1. Through `squash serve --slots 2`:
236 tok/s aggregate with both responses identical to their solo runs.

Restart (`squash serve`, GPU empty): 31.3 s to the first answered request, of which 27.7 s loads
72.67 GiB of weights (INT8 conversion included). The raw loader reads 77.35 GiB at 3.3 GiB/s with
8, 16 or 32 threads, O_DIRECT or buffered alike: the checkpoint does not fit the ~40 GiB page cache
this machine has free, so restart is bound by NVMe reads.
