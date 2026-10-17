# quench6k

An inference engine for one model on one GPU: Qwen3.8 Flash-Next 125B-A6B
([ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4))
on an NVIDIA RTX PRO 6000 Blackwell (96 GB). Rust host code, hand-written CUDA kernels for `sm_120a`,
weights loaded directly from the NVFP4 safetensors checkpoint.

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
quench6k serve --model /path/to/Swift-1.5-Qwen3.8-Flash-Next-NVFP4 --listen 0.0.0.0:8003 \
    --context 262144 --kv fp8 --cache-dir ~/.cache/quench6k
```

Loading takes about 30 s (72.7 GiB uploaded to the GPU). Keep the checkpoint on fast local storage:
prefill reads per-layer embedding rows from it on demand.

```
curl http://localhost:8003/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

## Documentation

- [DOCS.md](DOCS.md): commands, server options, API, sampling, caching, steering.
- [BENCHMARK.md](BENCHMARK.md): throughput and how it is measured.
- [DEVELOPMENT.md](DEVELOPMENT.md): building, testing, the correctness gate, project history.

No license has been chosen yet; all rights reserved.
