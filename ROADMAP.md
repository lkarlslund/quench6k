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

- 2026-10-15: repo created; GOAL.md agreed; oracle fixtures imported. Next: M0.
