# quench6k documentation

## Commands

| Command | Purpose |
|---|---|
| `quench6k serve` | HTTP server (OpenAI and Anthropic APIs) |
| `quench6k chat` | One chat turn from the command line, streamed to stdout (`--image`, `--video`, `--think`, `--temperature`) |
| `quench6k bench` | Prefill and decode throughput at given context lengths ([BENCHMARK.md](BENCHMARK.md)) |

Development commands (`score`, `gate`, `generate`, `spec-check`, `batch-check`, `vision-check`,
`kbench`, `mtp-probe`, `load`) are described in [DEVELOPMENT.md](DEVELOPMENT.md). Every command
prints its options with `--help`.

## Checkpoints

ModelOpt NVFP4 exports that quantize only the routed experts of the 48 text layers and keep every
other tensor in BF16; the PLE n-gram table may be BF16 or FP8 E4M3 with a scale (Qwen's FP8 table).

| Checkpoint | |
|---|---|
| [ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | The correctness gate's reference model (a finetune) |
| [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | Qwen's model; FP8 PLE table |

Not supported: quantized attention or shared experts, compressed-tensors exports, GGUF.

## Server options

| Option | Default | |
|---|---|---|
| `--model DIR` | | Checkpoint directory |
| `--listen ADDR` | `127.0.0.1:8080` | |
| `--name NAME` | `quench6k` | Model name in `/v1/models` |
| `--context N` | 65536 | Longest prompt + completion, per slot |
| `--slots N` | 2 | Concurrent requests (1 or 2) |
| `--kv bf16\|fp8` | `bf16` | Attention KV cache format (global flag, before `serve`); FP8 halves its memory |
| `--drafts N` | 6 | Longest speculative draft; each round picks 1..N |
| `--default-max-tokens N` | 65536 | Completion limit when a request sets none |
| `--tool-retries N` | 2 | Regenerations of a malformed tool call |
| `--cache-host-gib N` | 16 | Pinned host memory for stored contexts |
| `--cache-dir DIR` | | Disk tier for stored contexts, kept across restarts |
| `--chat-template FILE` | | Overrides the checkpoint's `chat_template.jinja` |
| `--steering FILE` | | Steering pack to load |
| `--strength X` | pack default | Strength of the steered model |
| `--steered-name NAME` | `<name>-steered` | Model name that applies the pack |
| `--capture-dir DIR` | | Enables activation capture requests |
| `--no-vision` | | Skips the vision tower (image and video requests are rejected) |
| `--image-max-pixels N` | 4194304 | Largest image area after resizing (each 32×32 block is one token) |
| `--video-max-frames N` | 128 | Most frames sampled from a video (2 per second) |

`--context 262144 --slots 2 --kv fp8` fits in 96 GB together with the weights and the vision tower.

## HTTP API

- `GET /v1/models`
- `POST /v1/chat/completions`: OpenAI chat completions, streaming with `"stream": true`. Reasoning is
  returned in `reasoning_content`, tool calls in `tool_calls`; `usage.prompt_tokens_details.cached_tokens`
  counts prompt tokens served from the context cache.
- `POST /v1/messages`: Anthropic messages (system, content blocks, `thinking`, `tool_use` /
  `tool_result`, `stop_sequences`), streaming with the Anthropic event types.

### Request fields

| Field | Default | |
|---|---|---|
| `temperature` | 1.0 | 0: greedy |
| `top_p` | 0.95 | |
| `top_k` | 20 | Capped at 64; 0: 64 |
| `min_p` | 0.0 | |
| `presence_penalty` | 0.0 | Subtracted from the logits of tokens already generated (−2 to 2) |
| `repetition_penalty` | 1.0 | Divides positive and multiplies negative logits of tokens in the prompt or output |
| `seed` | random | |
| `max_tokens` / `max_completion_tokens` | `--default-max-tokens` | Capped by the context |
| `stop` | | String or list of strings |
| `tools` | | OpenAI function tools |
| `chat_template_kwargs.enable_thinking` | template default | Anthropic: `"thinking": {"type": "enabled"}` |
| `reasoning_effort` | | Passed to the chat template |

Extensions:

| Field | |
|---|---|
| `"mtp": false` | Plain decoding instead of speculative |
| `"steering": {"steering_strength": x}` | Steering strength for this request (needs `--steering`) |
| `"capture": {"name": "..."}` | Writes an activation capture (needs `--capture-dir`) |
| `"return_progress": true` | Streams prefill progress chunks (`prompt_progress`, as llama.cpp's server) |

Images and videos are sent as data URLs in `image_url` / `video_url` content parts, or as Anthropic
base64 image blocks. Video needs `ffmpeg` and `ffprobe` on the `PATH`.

## Decoding

A request that runs alone decodes speculatively: the MTP head drafts up to `--drafts` tokens, the
model verifies them in one step, and the draft length adapts to the observed acceptance. Greedy output
is identical to plain greedy decoding; sampled output has exactly the distribution of plain sampling.
Penalties are applied in the verification step, so they keep this property.

When a second request arrives, both continue in one batched decode step without speculation; the
first request's speculative state is committed first.

## Tool calls

Every `<tool_call>` in a response is checked: it parses, is closed, names a declared function, has the
required parameters, and its argument types match the schema. A failing call is regenerated from just
before `<tool_call>` with a new seed and a temperature of at least 0.7, up to `--tool-retries` times;
the text before it is kept. While a tool call is being generated, streaming clients receive
`tool_call_progress` chunks with the byte count instead of partial text.

## Context cache

- The GPU holds the current conversation of each slot plus recurrent-state snapshots at turn
  boundaries and at intervals inside long prompts, so a follow-up turn or an edited prompt reuses
  the longest cached prefix.
- When a request would overwrite a different conversation, that conversation's state is moved to
  pinned host memory (`--cache-host-gib`) and, with `--cache-dir`, to disk.
- Stored contexts are keyed by their tokens (images by content) and are valid only for the model
  configuration (checkpoint, KV format) that wrote them.

## Steering and capture

A steering pack (`ninfer.steering/1`) removes directions from the attention-input activations:
y = x − a · Σₖ (x · dₖ) dₖ. With `--steering`, the server offers two models: `--name` (unsteered) and
`--steered-name` (strength `--strength`); a request can set its own strength. Strength 0 is
bit-identical to no pack.

A capture request writes `<name>.safetensors` to `--capture-dir`: per-layer activations and the top-20
log-probabilities at the last prompt token and at the last generated token, with the request as JSON
metadata. Capture requests use plain decoding.

## Environment variables

| Variable | |
|---|---|
| `QUENCH6K_TRACE=FILE` | Appends each request's prompt token IDs and generated tokens to FILE (JSON lines) |
| `QUENCH6K_TIMING=1` | Logs chat-template rendering and tokenization times |
| `QUENCH6K_SAMPLED_DRAFTS=P` | Sampled requests draw drafts at P% of the temperature (default 100; 0: greedy drafts) |
| `QUENCH6K_PLE_WARM_THREADS=N` | Threads that read embedding rows ahead of prefill (default 64) |
