# squash

A from-scratch inference engine for Qwen3.8 Flash-Next 125B-A6B on a single NVIDIA RTX PRO 6000
Blackwell GPU: Rust host, hand-written CUDA kernels for `sm_120a`. Weights load directly from the
ModelOpt NVFP4 safetensors checkpoint; no Python at run time.

No license has been chosen yet; all rights reserved until one is.

## Status

Goal and milestones: [GOAL.md](GOAL.md), [ROADMAP.md](ROADMAP.md) (with the dated progress log).
Model specification: [SPEC.md](SPEC.md). Latest benchmark: [bench/RESULTS.md](bench/RESULTS.md).

- Passes the correctness gate (GOAL.md §1) on all oracle fixtures, at batch 1 and 2 and with FP8 KV.
- Faster than NInfer 6000 in every decode and MTP cell (e.g. 185 vs 171 tok/s plain and 404 vs 401
  MTP3 at 8K; 173 vs 149 and 376 vs 361 at 256K); prefill is about half of NInfer's.

## Features

- Text, image and video input (27-block ViT + merger on the GPU; ffmpeg decodes video).
- MTP speculative decoding with an LM-head sketch draft and adaptive draft length; greedy output
  identical to plain decoding.
- Two concurrent requests in one decode batch (`--slots 2`).
- Direction-removal steering packs (`ninfer.steering/1`), per-request strength; strength 0 is
  bit-identical to no pack. Activation capture to safetensors for building packs.
- Context cache: recurrent snapshots at turn boundaries on the GPU, a pinned-host tier and a disk
  tier that persists across restarts.
- BF16 or FP8 KV cache (`--kv fp8`).
- OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) HTTP APIs with streaming,
  reasoning content, tool calls (validated, regenerated when malformed), images and video.

## Build

Needs Rust and the CUDA toolkit (nvcc with `sm_120a`).

```
cargo build --release
```

## Use

```
M=/path/to/Swift-1.5-Qwen3.8-Flash-Next-NVFP4

# Serve (OpenAI + Anthropic APIs)
./target/release/squash serve --model $M --listen 127.0.0.1:8080 --context 65536 \
    --cache-dir ~/.cache/squash [--kv fp8] [--steering pack.nsp --strength 0.5] [--capture-dir DIR]

# One-off chat, optionally with images or video
./target/release/squash chat --model $M --prompt "Describe this." --image photo.jpg

# Benchmarks and checks
./target/release/squash bench --model $M --ctx 8192 65536 --generate 256 [--mtp 3 | --mtp 6 --adaptive]
./target/release/squash score --model $M --fixture oracle/fixtures/* --out out/squash --suffix=-x --decode
./target/release/squash gate --candidates out/squash --suffix=-x
./target/release/squash spec-check --model $M --fixture oracle/fixtures/code --n 192 --graphs
./target/release/squash vision-check --model $M --reference out/vision/shapes
```

Server request extensions: `"steering": {"steering_strength": x}` and
`"capture": {"name": ...}` (with `--capture-dir`). Images and videos are sent as data URLs
(`image_url` / `video_url` content parts, or Anthropic base64 image blocks).

## References

`tools/ref/ref.py` is an independent PyTorch implementation of SPEC.md used for golden data;
`tools/ref/vision_ref.py` runs the HF transformers vision tower; `tools/vllm_ref.py` produces vLLM
reference distributions. The gate's references are llama.cpp NVFP4 W4A4 and 8-bit-activation runs
(`oracle/`).
