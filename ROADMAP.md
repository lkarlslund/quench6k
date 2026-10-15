# Roadmap

Milestones toward [GOAL.md](GOAL.md), in order. A milestone is done when its exit criterion holds
and the work is committed. The progress log at the bottom lets a new work session pick up where the
previous one stopped.

| # | Milestone | Exit criterion |
|---|---|---|
| M0 | Model spec | `SPEC.md` covers every op, dimension, dtype, state tensor and checkpoint tensor name, checked against HF `qwen4_exp` and llama.cpp |
| M1 | Workspace and loader | Cargo workspace, `build.rs` compiles CUDA for `sm_120a`, safetensors mmap loader uploads every text weight with parallel pinned I/O; load time measured |
| M2 | Reference harness | Per-op FP32 reference on real layer weights; a teacher-forced scoring tool that applies the GOAL.md gate to `oracle/fixtures` |
| M3 | Correct text model | Eager prefill and decode (embedding, HyperConnection, GDN, QSA with indexer, NVFP4 MoE, PLE, mixer, LM head) passes the gate on all fixtures |
| M4 | Fast decode | CUDA Graphs, fused kernels, FP8 non-expert weights; beats 172 tok/s at 512 and every plain-decode cell |
| M5 | MTP | Draft/verify with state rollback, adaptive draft length, LM-head draft; beats every MTP cell |
| M6 | Fast prefill and long context | Chunked prefill to 256K; beats every prefill cell; quantized KV profiles pass the gate |
| M7 | Concurrency | N requests in one decode batch, tuned for 2; the gate passes at batch 2 |
| M8 | Context cache | Multiple resident contexts, GPU/host/disk tiers, persistence across restart, reuse of any prefix |
| M9 | Steering and capture | Runtime packs, per-request strength, zero strength bit-identical; activation capture |
| M10 | Vision | Image and video input through the 27-layer tower and merger, checked against the HF reference |
| M11 | Serving | Tokenizer, chat template, OpenAI and Anthropic HTTP APIs, streaming, tool calls |
| M12 | Final benchmark | Full GOAL.md table measured against NInfer 6000 on the same machine; batch-2 and restart results |

## Progress log

- 2026-10-15: repo created; GOAL.md agreed; oracle fixtures imported.
- 2026-10-15: M0 done: SPEC.md written from HF qwen4_exp and llama.cpp qwen4exp. Open details, deferred to their milestones: MTP K/V positions (M5), vision preprocessing and multimodal positions (M10). Next: M1.
- 2026-10-15: M1 done. Cargo workspace (`squash-cuda` with nvcc `build.rs` for sm_120a and a hand-written cudart FFI; `squash-engine`; `squash` CLI). The loader indexes all 81 safetensors files in 0.4 s and uploads the 296,013 text + MTP tensors (77.35 GiB) as 50 byte-exact segments with no repacking, verified by device checksums. Measured: 23.3 s, 3.3 GiB/s, identical for buffered and O_DIRECT reads and for 8-32 threads. The limit is the GPU's PCIe link, which trained at Gen 3 x4 (pinned H2D 3.61 GB/s; GPU supports Gen 5, host slot reports Gen 4); the drive alone reads ~8.9 GB/s. Restart time is therefore PCIe-bound on this machine. Next: M2.
- 2026-10-15: M2 done. `tools/ref/ref.py` (dev-only PyTorch oracle written from SPEC.md, FP32, W4A16 experts) passes the gate on all six fixtures, including the 14K-token long-context fixture where QSA group selection is active; it sits about 2x closer to llama.cpp reference B than A and B are to each other. `squash gate` implements the gate in Rust. Next: M3.
- 2026-10-15: M3 done. Eager CUDA text model (BF16 tensor-core GEMM, NVFP4 W4A16 grouped expert GEMM, GDN recurrence, QSA with the group indexer and radix top-k, PLE with host-gathered rows). `squash score` passes the gate on all six fixtures both as chunked prefill and with the completion decoded one token per step (bit-identical: every kernel is batch-invariant). `squash generate` passes: code and capital-short reproduce the reference exactly; the other four diverge first at near-ties (margins -2.18 to 0.02). Unoptimized speed: decode ~20 tok/s, long-context prefill ~600-900 tok/s. The loader now starts each segment at a tensor so that every tensor >= 16 bytes is 16-byte aligned (layer 1's file has small I64 tensors that shift the rest by 8 bytes). Next: M4.
- 2026-10-15: M4 in progress. Added decode GEMVs (BF16/FP8/INT8/INT8-group-64 weights, split-K with fixed-order reduction), FP8 and INT8 conversion at load, CUDA Graph capture of the decode step (all per-step inputs come from a pinned step block), a fused small-batch MoE path (single-CTA routing, gate/up+SwiGLU and down GEMVs with hardware FP4 decode) and split-key attention, plus `bench --profile`. Weight precision: FP8 per-row and INT8 per-row fail the gate (math-thinking/multiturn/tool-call completions up to 3x the floor; an ablation showed no single projection class dominates); INT8 with one scale per 64 weights passes all fixtures (worst completion at 0.67 of the bound, BF16 0.49) and is the default profile. Decode at 512 context: 16 -> 114 tok/s (target 172).
- 2026-10-15: M4 progress. Fused projection groups (GDN in_proj, QSA in_proj, MoE router+shared, HC down+inject; members stored contiguously by the loader, BF16 profile writes the same column layout per member), HC norm folded into the down/inject weights at load so small batches project the raw FP32 residual with stream-aligned K splits, fused small-batch GDN kernel (conv, gating, delta rule, gated norm), restructured NVFP4 expert GEMVs (all loads in flight, templated on rows per expert), multi-CTA argmax. Gate passes for prefill and decode. Decode: 149 tok/s at 512, 140 at 8K (graph 6.7-7.0 ms/token, host prepare 0.04-0.2 ms). Note: another process (imageorganizer) shares the GPU and occasionally halves measured throughput; benchmark numbers need repeat runs.
- 2026-10-15: M4 progress. Cooperative single-kernel HyperConnection for small batches (residual update of the previous block deferred into it), programmatic dependent launch on every kernel with static weights / recurrent state loaded before the dependency wait, one-row-per-warp GEMVs instead of split-K for small N, attention merge over used splits, bench now sanity-checks 64 greedy tokens of the code fixture through the benchmarked graph path and reports the median of three decode runs. Removed the FP8 and per-row INT8 profiles (they fail the gate). Decode 165 tok/s at 512 (graph 6.03 ms/token); gate passes for prefill and decode.
