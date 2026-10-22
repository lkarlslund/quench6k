# quench6k

An inference engine for one model on one GPU: Qwen3.8 Flash-Next 125B-A6B in NVFP4 on an NVIDIA RTX
PRO 6000 Blackwell (96 GB). Rust host code, hand-written CUDA kernels for `sm_120a`, weights loaded
directly from the safetensors checkpoint ([supported checkpoints](DOCS.md#checkpoints)).

[![Decode and prefill throughput by context length](bench/results.svg)](BENCHMARK.md)

## Features

- OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) HTTP APIs: streaming, reasoning,
  tool calls (validated and regenerated when malformed), images and video.
- MTP speculative decoding with an adaptive draft length, for greedy and sampled requests.
- Context up to 262,144 tokens; BF16 or FP8 KV cache.
- Two concurrent requests.
- Context cache: snapshots on the GPU, a pinned host tier and a disk tier that survives restarts.
- Direction-removal steering packs with a per-request strength; activation capture.

## Install

Daily builds of `main` are on the [releases page](https://github.com/lkarlslund/quench6k/releases),
tagged `rNNNN` (the number of commits on `main`): Linux x86_64, Linux arm64 and Windows x86_64. They
need an NVIDIA driver that supports CUDA 13.4. Only Linux x86_64 is tested on hardware.

From source (Rust stable, CUDA toolkit 13.x with `sm_120a` support):

```
cargo build --release
```

## Run

```
quench6k serve --model /path/to/Qwen3.8-Flash-Next-NVFP4 --listen 0.0.0.0:8003 \
    --context 262144 --kv fp8 --cache-dir ~/.cache/quench6k
```

Loading takes about 30 s (72.7 GiB uploaded to the GPU). Keep the checkpoint on fast local storage:
prefill reads per-layer embedding rows from it on demand.

```
curl http://localhost:8003/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

## Recursive Flint

quench6k also runs [Recursive Flint](https://huggingface.co/RecursiveSI/flint) (40B-A2.5B MoE) from
the vendor's [NVFP4 GGUF](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf). Point `--model` at a
directory holding `flint-nvfp4.gguf` together with `tokenizer.json`, `chat_template.jinja`,
`config.json` and `generation_config.json` from the BF16 repository:

```
quench6k serve --model /path/to/flint-nvfp4-gguf --listen 0.0.0.0:8013 --context 262144 --kv fp8
```

The server serves one request at a time, reuses the prompt prefix across requests, and handles the
model's own `<think>` reasoning and its tool-call format. Throughput on the same machine at 600 W, with
greedy decoding of 128 tokens and FP8 KV from 64K up (BF16 KV below that):

| Context | Prefill (tok/s) | | Decode (tok/s) | |
|---:|---:|---:|---:|---:|
| | quench6k | llama.cpp | quench6k | llama.cpp |
| 512 | 7,862 | 5,691 | 499 | 252 |
| 8,192 | 20,377 | 11,515 | 452 | 228 |
| 65,536 | 12,886 | 7,991 | 337 | 188 |
| 131,072 | 8,982 | 5,591 | 264 | 151 |
| 260,000 | 5,488 | 3,462 | 189 | 110 |

llama.cpp at c7fc50941 (`llama-bench -fa 1 -ub 2048`, F16 KV). Prefill is the BF16 KV run (FP8 is
4–16% slower there). Records: `bench/flint.jsonl`, `bench/flint-llamacpp.jsonl`.

## Documentation

- [DOCS.md](DOCS.md): commands, server options, API, sampling, caching, steering.
- [BENCHMARK.md](BENCHMARK.md): throughput and how it is measured.
- [DEVELOPMENT.md](DEVELOPMENT.md): building, testing, the correctness gate, project history.

No license has been chosen yet; all rights reserved.
