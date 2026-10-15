# Goal

Build a Rust/CUDA inference engine for Qwen3.8 Flash-Next 125B-A6B on one RTX PRO 6000 Blackwell
that is **faster than NInfer 6000, at least as correct, and has better features**.

Correctness is a gate, speed is the objective, and features define the product. A change that makes
the engine faster but fails the correctness gate is not progress.

## 1. Correctness gate: oracle-correct within the quantization noise floor

The engine runs a quantized model, so bit-exact agreement with any single reference is not the bar.
The bar is: **squash may not deviate from a valid reference more than two valid quantized references
deviate from each other.**

References, under `oracle/fixtures/` (6 fixtures, 22 to 14K-token prompts):

- **A**: llama.cpp, native NVFP4 W4A4 with dynamic activation scales;
- **B**: llama.cpp, the same weights with 8-bit activations.

The spread between A and B is the noise floor that quantization alone introduces. For each fixture
and region (prompt, completion), using teacher-forced scoring of the fixture sequence:

| Check | Pass condition |
|---|---|
| mean \|Δlogp\| of the sequence token vs the nearer reference | ≤ max(2 × floor mean, 0.01) |
| p99 \|Δlogp\| vs the nearer reference | ≤ max(2 × floor p99, 0.05) |
| greedy generation | identical to the reference sequence, or it diverges only at a near-tie (reference top-2 margin ≤ tie threshold) |

Every speed configuration that ships must pass the gate: each KV-cache precision, MTP draft length,
CUDA Graphs on and off, and batch sizes 1 and 2. Individual kernels are also tested against an
FP32/FP64 reference of their mathematical contract; the end-to-end gate does not replace that.

Mathematical source of truth: HF transformers `qwen4_exp`. Second implementation:
`~/github-others/llama.cpp`.

## 2. Speed objective: beat NInfer 6000

Baseline: NInfer 6000 8-bit variant (FP8 non-expert weights, FP8 PLE, NVFP4 experts), one request,
BF16 KV, greedy, RTX PRO 6000 at 600 W. Throughput in tokens per second; decode generates 256
tokens after a prompt of the given length.

| Context | Decode | MTP3 | MTP3 + LM-head draft | Prefill (no MTP) |
|---:|---:|---:|---:|---:|
| 512 | 172.0 | 256.2 | 274.8 | 5,905 |
| 8K | 170.6 | 381.5 | 401.3 | 13,908 |
| 64K | 153.9 | 353.5 | 380.3 | 12,171 |
| 128K | 149.8 | 348.1 | 368.0 | 11,032 |
| 256K | 149.0 | 346.8 | 360.6 | 9,154 |

Target: beat every cell, then measure two concurrent requests as the primary serving workload.
Time to restart the service with the GPU empty and a warm page cache is also measured, and must stay
dominated by raw weight I/O.

## 3. Features: better than NInfer

Required:

- Text and vision (image and video) input.
- MTP speculative decoding with an adaptive draft length.
- Concurrent requests: designed for N, tuned for 2.
- Direction-removal activation steering: load a pack at runtime and apply it per request with a
  strength in [0, 1]. Zero strength must be bit-identical to an unsteered run. Activation capture
  for building packs.
- A flexible context cache: prefix reuse across requests and conversations, with more retention and
  placement options than NInfer (to be specified).
- Loads the ModelOpt NVFP4 safetensors checkpoint directly, with no repacked format, and restarts
  fast.
- OpenAI- and Anthropic-compatible HTTP serving, including streaming and tool calls.

Further features are to be decided and will be added here when agreed.
