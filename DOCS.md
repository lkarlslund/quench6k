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

ModelOpt NVFP4 exports of Qwen3.8 Flash-Next with NVFP4 routed experts in the 48 text layers. The other
projections may be BF16 (quantized to 8 bits at load) or MXFP8 (E4M3 with a UE8M0 scale per 32, used as
stored); NVFP4 block scales may be plain or in ModelOpt's lossless CSF form (decoded at load); the PLE
n-gram table may be BF16, FP8 E4M3 with a scale (Qwen's FP8 table) or NVFP4; MTP experts may be BF16
(quantized to NVFP4 at load) or NVFP4.

| Checkpoint | |
|---|---|
| [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | NVIDIA's quantization-aware distilled checkpoint for one GPU: MXFP8 attention, GDN and shared experts, NVFP4 PLE table (96 GB) |
| [ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4) | The correctness gate's reference model (a finetune) |
| [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | Qwen's model; FP8 PLE table |

The gate's reference distributions (`oracle/fixtures`) were produced with Swift's weights; they do not
judge other checkpoints, whose numerics differ by design.

Not supported: NVFP4 shared experts or FP8 block-scaled MTP experts (e.g.
[nvidia/Qwen3.8-Flash-Next-NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4-QAD)),
compressed-tensors exports, GGUF.

### Hy4-flash

Tencent's Hy4-flash (`hy_v4_flash`: 42 layers of gated MLA attention, sliding-window or indexer-selected,
512 latent experts per layer, hashed n-gram "Engram" memory) runs from an NVFP4 export with routed experts
stacked per layer and attention projections BF16 or FP8 with a scale per 128 x 128 block:

| Checkpoint | |
|---|---|
| [Avifenesh/Hy4-flash-NVFP4](https://huggingface.co/Avifenesh/Hy4-flash-NVFP4) | NVFP4 experts with a scale per expert, FP8 attention, FP8 Engram tables with a scale per row (161 GB) |

`quench6k serve --model DIR` detects the model and serves one sequence at a time (prefix reuse,
prompt-lookup drafts, tool calls, reasoning efforts `none`/`high`/`max`); steering, adapters, vision,
edits and stored contexts are not available for it. About 85.5 GiB of weights load onto the GPU; the
shared experts, the latent projections around the routed experts and the LM head are quantized to FP8 at
load (`QUENCH6K_HY4_FP8`). The four Engram tables (63 GB) stay on the host, memory-mapped; 64 random
rows per token are gathered from them, so keep them in the page cache (`QUENCH6K_HY4_ENGRAM_PRELOAD=1`
reads them in at load) and on uncompressed storage, as for the PLE table below. The drafter (DFly2) and
the vision tower are not supported yet.

Its correctness gate compares teacher-forced distributions against `tools/ref/hy4_ref.py`, an
independent PyTorch implementation run on the same weights in BF16 (reference A) and FP32 (B):

```
python3 -I tools/ref/hy4_fixtures.py --model $M --qwen $SWIFT --out out/hy4-fixtures
python3 -I tools/ref/hy4_ref.py --model $M --fixture out/hy4-fixtures/* --out out/hy4-ref --suffix=-bf16
python3 -I tools/ref/hy4_ref.py --model $M --dtype fp32 --fixture out/hy4-fixtures/* --out out/hy4-ref --suffix=-fp32
# copy each <fixture>-bf16/-fp32 pair into its fixture directory as ref-bf16 / ref-fp32, then
quench6k hy4 score --model $M --fixture out/hy4-fixtures/* --out out/hy4-eng
quench6k gate --fixtures out/hy4-fixtures --candidates out/hy4-eng --ref-a ref-bf16 --ref-b ref-fp32
```

`quench6k hy4 generate` and `quench6k hy4 bench` run greedy generation and throughput measurements.

Keep the PLE n-gram table's files (Swift: 92 GiB of BF16 in `embedding-model-*.safetensors`) on
storage without filesystem compression. The table is memory-mapped and gathered on the host, 16 random
rows per token, and most of it is never in the page cache; on a compressed file (btrfs
`compress=zstd`, for example) every row read decompresses a whole extent (measured: 600 µs per row,
prefill 4.2K instead of 10.7K tokens/s). Check with `filefrag -v FILE | grep -c encoded` (must be 0);
to rewrite a file uncompressed on btrfs, create the copy with `chattr +m` set before writing to it
(`cp --reflink=never`), then replace the original.

## Server options

| Option | Default | |
|---|---|---|
| `--model DIR` | | Checkpoint directory |
| `--listen ADDR` | `127.0.0.1:8080` | |
| `--name NAME` | `quench6k` | Model name in `/v1/models` |
| `--context N` | 65536 | Longest prompt + completion, per slot |
| `--slots N` | 2 | Concurrent requests (1 to 8), each keeping its conversation on the GPU |
| `--plan-tokens N` | 16384 | Completion tokens a request's memory plan counts on (see below) |
| `--prefill-share F` | 0.5 | Share of GPU time a new prompt's prefill gets while other requests decode (1: they pause) |
| `--kv bf16\|fp8` | `bf16` | Attention KV cache format (global flag, before `serve`); FP8 halves its memory |
| `--drafts N` | 6 | Longest speculative draft; each round picks 1..N |
| `--default-max-tokens N` | 65536 | Completion limit when a request sets none |
| `--tool-retries N` | 2 | Regenerations of a malformed tool call |
| `--no-thinking` | | Chat requests do not think unless they set `enable_thinking` or `reasoning_effort` |
| `--cache-host-gib N` | 16 | Host memory for stored contexts waiting to be written to disk (without `--cache-dir`: for all stored contexts) |
| `--cache-dir DIR` | | Disk tier for stored contexts, kept across restarts |
| `--cache-disk-gib N` | 200 | Most disk space for the disk tier (0: no limit) |
| `--cache-max-age-days N` | | Deletes stored contexts unused for N days |
| `--chat-template FILE` | | Overrides the checkpoint's `chat_template.jinja` |
| `--steering FILE` | | Steering pack to load |
| `--strength X` | pack default | Strength of the steered model |
| `--steered-name NAME` | `<name>-steered` | Model name that applies the pack |
| `--adapter DIR` | | Rank-1 adapter on the output projections (PEFT directory: `adapter_config.json`, `adapter_model.safetensors`), served as a second model |
| `--adapted-name NAME` | `<name>-adapted` | Model name that applies the adapter |
| `--capture-dir DIR` | | Enables activation capture requests |
| `--no-vision` | | Skips the vision tower (image and video requests are rejected) |
| `--image-max-pixels N` | 4194304 | Largest image area after resizing (each 32×32 block is one token) |
| `--video-max-frames N` | 128 | Most frames sampled from a video (2 per second) |
| `--unload-idle SECS` | | Unloads the model after this many seconds without requests (see [Unloading](#unloading)) |

Attention-cache memory is used per token a slot holds, not reserved per slot: about 18 KB per token
with `--kv fp8` (36 KB with BF16). With `--context 262144 --slots 8 --kv fp8`, the weights and the
vision tower, about 12 GB remain for conversations (some 600K tokens); idle slots give up theirs
when that runs out (see [Context cache](#context-cache)).

Requests start by plan: a request starts only when its prompt plus its expected completion
(`--plan-tokens`, at most its `max_tokens`) fits in device memory next to the plans of the running
requests, counting what idle slots hold as free. Otherwise it waits for a running request to finish,
so a newer request never cuts a running one short. A request alone always starts. A completion longer
than its plan grows into free memory; only if none is left does it fail.

## HTTP API

- `GET /`: embedded performance dashboard with stacked per-slot generation and uncached prefill
  throughput, request concurrency, queue depth, prefix reuse and slot-level KV memory. Alpine.js and
  ECharts are vendored into the binary; the browser needs no external assets. The vertically aligned
  prefill, generation and activity plots share their cursor and pan/zoom window. Scroll or use the
  range slider to inspect history; Live returns to the moving view.
- `GET /stats`: dashboard JSON with one-second live samples and durable minute history, collected
  without a browser open. `?range=SECONDS` selects up to the retention limit (default 3600).
  `?since=TIMESTAMP` filters samples newer than that Unix timestamp in milliseconds; incremental
  requests are intended for live ranges. Long ranges are downsampled to about 1800 points: throughput
  averages, peak concurrency/queue/KV memory, summed request and token counts. `sample_interval_ms`
  describes the long-range resolution; live responses can include archived minute data before their
  in-memory samples. Pan, zoom and hover remain linked between charts; tooltips omit zero-rate slots.
  The usage table totals the visible samples, including a zoomed range, at sample boundaries.
  `totals` persist across restarts; `session_totals` cover the current process. `requests` counts
  JSON generation HTTP requests, including rejected requests; `generations` counts completed attempts
  (tool-call retries count again). Input/prompt and cached token counts cover completed attempts;
  output and uncached prefill tokens include in-flight work. No prompt text or output content is stored.
  Statistics use embedded tsink with a 32 MiB engine memory budget and durable WAL writes. Configure
  `--stats-dir` (default `$XDG_STATE_HOME/quench6k/stats`, or `~/.local/state/quench6k/stats`) and
  `--stats-retention-days` (default 31). Use a distinct directory for each server. Minute aggregates
  are immutable; the current minute and cumulative totals are checkpointed atomically every sample
  and on graceful shutdown. Abrupt termination can lose activity since the last checkpoint (normally
  about one second). Lifetime totals remain after graph retention expires. Persistence failures are
  logged and shown in the dashboard as `server.storage_error`. Disk work runs outside inference and
  history locks; concurrent historical scans are limited to two.
- `GET /health`: `{"status": "ok", "model": "loaded"}` once the server is up; `model` is `loaded`,
  `loading` or `unloaded` (see [Unloading](#unloading)).
- `POST /v1/unload`: unloads the model and answers `{"status": "unloaded"}` once it is gone.
- `GET /v1/models`, `GET /v1/models/{id}`: the served model names, each with its context limit in
  `max_model_len` (as vLLM) and `context_length` (as OpenRouter): `--context`.
- `POST /v1/chat/completions`: OpenAI chat completions, streaming with `"stream": true`. Reasoning is
  returned in `reasoning_content`, tool calls in `tool_calls`; `usage.prompt_tokens_details.cached_tokens`
  counts prompt tokens served from the context cache.
- `POST /v1/messages`: Anthropic messages (system, content blocks, `thinking`, `tool_use` /
  `tool_result`, `stop_sequences`), streaming with the Anthropic event types.
- `POST /v1/messages/count_tokens`: `{"input_tokens": n}`, the prompt's tokens after the chat template
  and image and video expansion.

- `POST /v1/completions`: OpenAI (legacy) completions of a raw `prompt` (a string, special tokens
  written as text, or a list of token ids), streaming with `"stream": true`. No chat template and no
  reasoning or tool-call parsing; `logprobs`, `echo` and `n` > 1 are rejected.
- `POST /tokenize`: `{"tokens": [...], "count": n, "max_model_len": c}` for `messages` (through the
  chat template, as a chat request would be prompted) or a raw `prompt` / `content` string.
- `POST /detokenize`: `{"tokens": [...]}` to text, returned as `prompt` (vLLM) and `content` (llama.cpp).
- `GET /metrics`: Prometheus counters per slot (generations, prompt tokens, prompt tokens served from
  the cache, generated tokens, prefill and decode seconds, attention-cache bytes in use), failed
  generations, the waiting and running requests, the model state (`quench6k_model_loaded`: 0
  unloaded, 1 loading, 2 loaded) and reloads after an unload.

### Unloading

`POST /v1/unload`, or `--unload-idle SECS` without requests, frees the model's device and host memory
while the server keeps listening. It waits until no request is running or queued (requests queued
before it finish first), writes the slots' contexts to the context store as a shutdown does, and drops
the model; the CUDA context itself (well under 1 GB) stays. The next generation request loads the
model again and is processed once loading finishes; requests arriving meanwhile queue behind it, and
a request whose prompt was stored continues from the store. If loading fails, the waiting requests
get the error and the next request tries again. Tokenizing, `/health`, `/metrics` and the dashboard
answer without loading the model.

A prompt of `--context` tokens or more is rejected with HTTP 400 and code `context_length_exceeded`.

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
| `chat_template_kwargs.enable_thinking` | template default (off with `--no-thinking`) | Anthropic: `"thinking": {"type": "enabled"}` |
| `reasoning_effort` | template default (`xhigh`) | `low`, `medium`, `high` (as `xhigh`) or `xhigh`; `none` and `minimal` turn thinking off |

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

Concurrent requests decode speculatively together: each drafter runs, then one batched verification
step covers every request's drafts, and each request accepts and samples on its own. They draft the
same length per round, chosen per number of running requests (more requests verify more rows per
draft token, so they draft shorter; at most 32 rows are verified, in 16-row slices of the decoding
kernels, so greedy output stays identical: up to 6 drafts for 2 to 4 requests, 5 for 5, 4 for 6, 3
for 7 and 8). Requests with plain decoding (`"mtp": false`, capture) take batched plain steps
alongside.

A new request's prompt is prefilled in slices of 4096 tokens that take turns with the running
requests' steps, by time (`--prefill-share`): the running requests keep decoding at about half speed
instead of pausing, and the prefill takes up to twice as long. A request alone prefills in full
chunks.

`--dflash DIR` (any command that decodes speculatively) drafts with the
[DFlash 2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) block drafter instead of the MTP
head: five dense layers draft 7 tokens in one pass, conditioned on the outputs of target layers 5, 15,
25, 35 and 45 (captured during verification) through a 2048-position K/V window, and a candidate
selector picks one path through each position's top 16 tokens. Verification is unchanged, so the
output is too; `--drafts` may then be up to 7. The drafter was trained on Qwen's model: on Swift it
accepts 8–10% more tokens per round than MTP on chat prompts, but a round costs more (drafting about
1.4 ms), so it is about 8% slower, and the service keeps MTP.

## Tool calls

Every `<tool_call>` in a response is checked: it parses, is closed, names a declared function, has the
required parameters, and its argument types match the schema. A failing call is regenerated from just
before `<tool_call>` with a new seed and a temperature of at least 0.7, up to `--tool-retries` times;
the text before it is kept. While a tool call is being generated, streaming clients receive
`tool_call_progress` chunks with the byte count instead of partial text.

Qwen3.8 models sometimes write a URL of the fetch proxy Alibaba used in their training environment
(`https://routify-file-proxy-sg.oss-ap-southeast-1.aliyuncs.com/proxy_temp_file/production/...` with an
invented signature) instead of the URL they mean; a client that runs such a call contacts Alibaba Cloud.
A tool call containing `routify-file-proxy`, `routify.alibaba-inc` or `proxy_temp_file/production` is
treated as invalid and regenerated at least four times (more if `--tool-retries` is higher); if every
attempt produces one, the calls are dropped and only the text before them is returned.
`/metrics` counts them (`quench6k_proxy_tool_calls_total`, `quench6k_proxy_tool_calls_dropped_total`).
This only covers calls this server generates: block the domain at the client's egress as well.

## Context cache

- The GPU holds the current conversation of each slot plus recurrent-state snapshots at turn
  boundaries and at intervals inside long prompts, so a follow-up turn or an edited prompt reuses
  the longest cached prefix.
- A request starts on the free slot that holds the most of its prompt, or else on the one holding the
  shortest conversation, so conversations running side by side each keep their own slot.
- When device memory runs out, it is taken from the slots without a running request, least recently
  used first: memory past what a slot holds, then periodic snapshots, then a slot's whole conversation
  (stored first, so the next request of that conversation loads it from the store).
- When a request would overwrite a different conversation, that conversation's state is stored: copied
  to host memory and, with `--cache-dir`, written to disk by a bounded background writer, after which
  the process frees its copy. Loading a stored context maps its file, so recently used contexts stay in
  the kernel's page cache, which the system manages and reclaims under memory pressure; the process
  pins no host memory for them. Contexts waiting to be written count toward `--cache-host-gib`; if the
  writer cannot keep up, storing waits for it rather than growing an unbounded queue. Shutdown drains
  pending writes. Without `--cache-dir`, stored contexts stay in host memory up to `--cache-host-gib`.
- Keep `--cache-dir` on a filesystem without transparent compression (as for the PLE table, above):
  every load of a context not in the page cache would decompress it.
- The disk tier keeps within `--cache-disk-gib` and `--cache-max-age-days` by deleting the least
  recently used contexts first (a file's modification time is its last use; reusing a context renews
  it). Files written for another model configuration count toward the limit and go first.
- Stored contexts are keyed by their tokens (images by content) and are valid only for the model
  configuration (checkpoint, KV format) that wrote them.

Cache operations taking at least 100 ms log allocation, transfer, disk-read/write or writer-wait
timings. `QUENCH6K_TIMING=1` also logs shorter operations. Pinned-memory allocation and cache restores
still run synchronously; background disk writes can overlap decoding.

## Steering and capture

A steering pack (`ninfer.steering/1`) removes directions from the attention-input activations:
y = x − a · Σₖ (x · dₖ) dₖ. With `--steering`, the server offers two models: `--name` (unsteered) and
`--steered-name` (strength `--strength`); a request can set its own strength. Strength 0 is
bit-identical to no pack.

A capture request writes `<name>.safetensors` to `--capture-dir`: per-layer activations and the top-20
log-probabilities at the last prompt token and at the last generated token, with the request as JSON
metadata. Capture requests use plain decoding.

## Inspection

`/inspect` is a page for asking why the model predicts a token; `POST /v1/inspect` is its API. It
takes `messages` (rendered with the generation prompt) or a raw `prompt`, an optional `target` (a token
id or text; default: the most likely next token), an optional `contrast` token (then everything splits
logit(target) − logit(contrast), e.g. why the model prefers one token over another) and `model` (the
variant). The answer splits the target's logit at the prompt's last token into what wrote it: the embedding, the PLE injection, each
layer's mixer (gated DeltaNet or attention) and MoE output, each routed expert and the shared expert,
and every expert neuron (an intermediate unit: activation times routing weight times its
down-projection column). The residual is a plain sum of these parts, each written into the four streams
with its injection weights, and the final mixer is linear in the residual once its gates and the
streams' RMS factors are held at the values this prompt gave, so the split is exact; the report shows
the checks (the residual rebuilt from its parts, the neurons' share of their experts). These are direct
effects: what a part changes through later layers is credited to those layers. For the largest parts
it also lists the tokens their outputs push up and down, and for each layer the experts the router
ranked next.

An inspection runs between steps on a free slot: the prompt is prefilled (cached prefixes reused, so
asking about another target of the same prompt is quick), then its last token runs as a single-token
step with activation capture. Image prompts are not supported. With `"edits_at": "last"`, an
inspection's edits apply to the inspected token only: the prefix comes from the unedited model's cache
(so a sweep over many edits of one long prompt costs about a second each), and the slot's state is
dropped afterwards.

### Edits

Any request (chat, completion or inspection) can carry model edits as `"edits"`, a list of:

| Edit | |
|---|---|
| `{"layer": L, "expert": E, "unit": U, "scale": S, "add": A}` | A routed expert's neuron: activation × S + A (defaults 1 and 0), wherever a token routes to the expert |
| `{"layer": L, "shared": true, "unit": U, ...}` | The same for the shared expert's neuron |
| `{"layer": L, "expert": E, "ban": true}` | The router never picks the expert |
| `{"layer": L, "expert": E, "router_bias": B}` | Adds B to the expert's router logit |

Scale 0 switches a neuron off, −1 flips it, and `add` pushes it. A request with edits decodes alone
and without speculation (one eager step per token), so it never affects other requests; its cached
states are kept apart by the edit set and reused only by requests with the same edits. An edit that
changes nothing (scale 1) gives the same output as no edit. The inspector page adds edits from its
neuron and expert rows, shows the prediction before and after, and compares greedy continuations.

### Neuron atlas

`/atlas` builds atlases: the model runs over a corpus while every expert neuron's activations are
recorded on the GPU (the sum of |activation|, the strongest activation with its token, and how often
each expert is routed). The corpus is an uploaded text, a file under `--atlas-dir` (the only way to
name server files), or text the model writes about a topic (`samples` texts of up to `max_tokens`, at
temperature 1). Scans run in 2048-token segments taking turns with running requests, like prefill.

| Endpoint | |
|---|---|
| `POST /v1/atlas` | `{"name", "topic", "samples", "max_tokens"}`, `{"name", "text"}` or `{"name", "file"}`; `"model"` picks the variant |
| `GET /v1/atlas`, `DELETE /v1/atlas/{name}` | List (with progress), delete |
| `GET /v1/atlas/{name}/neurons?base=&by=mean\|peak&layer=&limit=` | Neurons by mean \|activation\| per token or by peak, minus a baseline atlas's; each with its strongest example in context |
| `GET /v1/atlas/{name}/experts?base=` | Experts by routing rate, minus the baseline's |
| `POST /v1/lens` | `{"neurons": [{"layer", "expert" or "shared": true, "unit"}]}`: tokens each neuron's output raises and lowers |

A baseline atlas of ordinary text (the model writing about anything) turns "fires often" into "fires on
this topic". Atlases live in memory (at most 6, about 190 MB each). The page sends neurons and experts to
the inspector's edits.

## Environment variables

| Variable | |
|---|---|
| `QUENCH6K_TRACE=FILE` | Appends each request's prompt token IDs and generated tokens to FILE (JSON lines) |
| `QUENCH6K_TIMING=1` | Logs chat-template rendering and tokenization times |
| `QUENCH6K_SAMPLED_DRAFTS=P` | Sampled requests draw drafts at P% of the temperature (default 100; 0: greedy drafts) |
| `QUENCH6K_PLE_WARM_THREADS=N` | Threads that read embedding rows ahead of prefill (default 64) |
| `QUENCH6K_HY4_FP8=LIST` | Hy4-flash: BF16 projections quantized to FP8 at load, as name fragments (`all`, `none`; default `shared_experts,routed_expert,lm_head`) |
| `QUENCH6K_HY4_ENGRAM_PRELOAD=1` | Hy4-flash: read the Engram tables into the page cache at load |
| `QUENCH6K_HY4_GRAPHS=0` | Hy4-flash: run small steps eagerly instead of as CUDA graphs |
| `QUENCH6K_HY4_W4A4=1` | Hy4-flash: prefill experts with NVFP4 activations on the FP4 tensor cores (faster; fails the gate) |
| `QUENCH6K_HY4_DUMP=DIR`, `QUENCH6K_HY4_TIMING=1` | Hy4-flash debugging: per-layer hidden states, per-step host timings |
