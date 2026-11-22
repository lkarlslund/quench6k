# quench6k documentation

## Commands

| Command | Purpose |
|---|---|
| `quench6k serve` | HTTP server (OpenAI and Anthropic APIs) |
| `quench6k serve-models` | One HTTP endpoint for several models, each loaded when a request names it ([Several models](#several-models)) |
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

`quench6k serve --model DIR` detects the model and serves up to `--slots` requests at once (see
[Slots for Inkling 2, Hy4-flash and Flint](#slots-for-inkling-2-hy4-flash-and-flint); prefix reuse,
drafted decoding, tool calls, reasoning efforts `none`/`high`/`max`); steering, adapters, vision,
edits and stored contexts are not available for it. About 87 GiB of weights load onto the GPU; the
shared experts, the latent projections around the routed experts and the LM head are quantized to FP8 at
load (`QUENCH6K_HY4_FP8`). The four Engram tables (63 GB) stay on the host, memory-mapped; 64 random
rows per token are gathered from them, so keep them in the page cache (`QUENCH6K_HY4_ENGRAM_PRELOAD=1`
reads them in at load) and on uncompressed storage, as for the PLE table below. The vision tower is not
supported yet.

The checkpoint's DFly2 drafter (5 layers) drafts five tokens after each step in one pass: the accepted
token and four mask tokens attend through the target's caches of layers 25 to 41 (and the key lists
their indexers chose), then a hidden correction picks the drafts in order through the target's LM head.
Each step verifies as many of them as are expected to pay off, from running estimates of each draft
position's acceptance and of step times by width (`QUENCH6K_HY4_DRAFTS=N` fixes the count in `quench6k
hy4`; requests with `"mtp": false` decode without drafts; `QUENCH6K_HY4_DFLY=0` does not load the
drafter, and prompt-lookup drafts are used instead). Greedy and sampled decoding stay exact: a draft is
kept only when it is the token the target picks.

The hyper-connection lanes (the residual streams) are kept in FP32, where the transformers model keeps
them in BF16: on the gate's fixtures the generated positions' log-probabilities are closer to an FP32
forward that way (mean |dlogp| 5 to 30% lower), at no measurable cost.

Its correctness gate compares teacher-forced distributions against `tools/ref/hy4_ref.py`, an
independent PyTorch implementation run on the same weights in BF16 (reference A; `--fp32 lanes` keeps the
lanes in FP32 as the engine does, `--fp32 moe_sum` the routed experts' sum) and FP32 (B):

```
python3 -I tools/ref/hy4_fixtures.py --model $M --qwen $SWIFT --out out/hy4-fixtures
python3 -I tools/ref/hy4_ref.py --model $M --fixture out/hy4-fixtures/* --out out/hy4-ref --suffix=-bf16
python3 -I tools/ref/hy4_ref.py --model $M --dtype fp32 --fixture out/hy4-fixtures/* --out out/hy4-ref --suffix=-fp32
# copy each <fixture>-bf16/-fp32 pair into its fixture directory as ref-bf16 / ref-fp32, then
quench6k hy4 score --model $M --fixture out/hy4-fixtures/* --out out/hy4-eng
quench6k gate --fixtures out/hy4-fixtures --candidates out/hy4-eng --ref-a ref-bf16 --ref-b ref-fp32
```

`quench6k hy4 generate` and `quench6k hy4 bench` run greedy generation and throughput measurements
(`--dfly`: drafted decoding).

The engine steps up to 8 sequences at once, up to 8 tokens each (a pending token and its drafts), and
drafts for them in one pass; each sequence's results (logits, caches, drafts) are bit-identical to
running it alone. Small steps use only kernels whose arithmetic for a row does not depend on the step's
other rows (GEMVs, per-head GEMVs, routed experts per token-expert pair, attention with a fixed split of
each row's keys), so a token's bits also do not depend on how many drafts are verified with it, and
drafted greedy decoding gives the plain greedy tokens. `quench6k hy4 width-check` compares one-token
steps with wider ones, `quench6k hy4 batch-check` sequences alone and batched (teacher-forced logits and
caches, greedy decoding with drafts) bitwise, and with `--bench-ctx` measures the aggregate decode
throughput of 1, 2, 4 and 8 batched sequences. Kernels find each row's sequence through per-row
descriptors and per-sequence address tables in device memory, so the CUDA graphs of small steps belong
to the workspace and serve any sequences.

A sequence's indexed layers keep 17,664 bytes per position (cache row, token key and pooled key of 12
layers), in address ranges reserved for its capacity (up to the model's 1,048,576 positions) and backed
by memory in 2 MiB pieces as the sequence grows (`State::reserve`, released by `State::trim`); the
sliding layers' rings and the prompt-end snapshot take a fixed 106 MB. Sequences share what the weights
(87 GiB) and the workspace (1.85 GiB with 2048-token prefill chunks) leave: about 4.4 GiB on a 96 GB GPU,
some 260K positions for one sequence or 27K each for eight.

### Inkling 2

Thinking Machines' Inkling 2 Mini (`inkling2_loop`: 24 layers of which 6 .. 17 run three times, 128
experts per layer, sliding-window and full attention with a learned relative-position bias instead of
rotary embeddings, causal convolutions on keys, values and both sublayer outputs, and "block attention
residuals" that mix earlier block sums into each sublayer's input) runs from its original release:

| Checkpoint | |
|---|---|
| [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) | NVFP4 experts (MXFP8 in the last four layers), BF16 elsewhere, DSpark drafter in `mtp.safetensors` (20 GB) |

`quench6k serve --model DIR` detects the model and serves up to `--slots` requests at once (see
[Slots for Inkling 2, Hy4-flash and Flint](#slots-for-inkling-2-hy4-flash-and-flint); prefix reuse while
the rings still hold it, drafted decoding, tool calls, reasoning efforts `none` .. `max`); steering,
adapters, vision, edits and stored contexts are not available for it, and the vision tower is not
supported yet. The model's message tokens (`<|message_model|>`, `<|content_thinking|>`,
`<|content_invoke_tool_json|>`, ...) are decoded to the `<think>` and `<tool_call>` text the server parses,
so `reasoning_content` and `tool_calls` work as for the Qwen models. About 18 GiB of weights load onto
the GPU; the LM head is quantized to FP8 at load (`QUENCH6K_INK_FP8`): it does not move the gate's
distributions measurably, while FP8 layer projections move them by about half the BF16 noise floor.

The residual stream and the block sums are kept in FP32, where the transformers model keeps them in
BF16. The looped model amplifies rounding: its BF16 and FP32 references differ by 0.1 to 0.2 nats per
position on average (Qwen and Hy4-flash: under 0.04), and the engine's generated positions are at least
as close to the FP32 reference as the BF16 reference is. With differences that large, a fixture of a
few hundred positions can fail the gate on a single position where the two references themselves
disagree by nats.

The DSpark drafter (3 dense layers and a recurrent "Markov" head) drafts seven tokens after each step:
the accepted token and seven mask tokens attend, bidirectionally within the block, to context keys and
values computed from four target features of every earlier position (the inputs of layer visits 5, 23
and 41 and the final norm's output, mixed per drafter layer); then the Markov head picks the drafts in
order, biasing the LM head's logits of each block row by a state carried from the previous pick (among
each 256-token block's four best tokens by the LM head alone). The head also states each draft's
probability; the draft policy calibrates those online against what verification keeps, verifies the
number of drafts with the most expected tokens per second, and runs the drafter only while drafted steps
beat plain ones (`QUENCH6K_INK_DRAFTS=N` fixes the count in `quench6k inkling`; `QUENCH6K_INK_DRAFT=0`
does not load the drafter). Decoding stays exact.

Sequences share the GPU's memory by length: a sequence's full-attention caches (8 of the 48 layer
visits, and the drafter's one full-attention layer: 36 KiB per position in all) are reserved as virtual
memory for its capacity (up to the model's 1,048,576 positions) and mapped in pieces as positions are
used (`State::reserve` / `State::trim`); its rings (the sliding layers' keys and values and the
convolution inputs, 4096 positions each, about 3.1 GiB) are allocated with it.

Prefill is canonical, so a prompt's results depend only on its tokens, not on what the sequence held
before: prompts are computed in chunks that end at multiples of 2048 positions (`CANON_CHUNK`,
`InklingModel::prefill`) and at the prompt's end, and `State::rewind` reuses a cached prefix only up to a
multiple of 2048 at most the common prefix and the sequence's canonical frontier (the boundary up to
which every position came from such whole chunks; decode steps, drafts and partial chunks never advance
it, writes below it lower it). Everything after that boundary is recomputed, earlier replies included,
so a turn recomputes up to 2047 cached prompt tokens plus the previous reply. The rings at a boundary are
either still live (within 3056 positions) or the boundary snapshot each whole chunk takes (the last 1040
positions of the sliding keys and values and the last 4 of the convolution inputs, about 0.17 GiB);
`State::snapshot` does nothing now. Up to 8 sequences decode together
(`InklingModel::forward_batch`, 1 to 16 tokens each, at most 64 rows; `draft_batch` drafts for them
in one pass): every kernel of a small step finds each row's sequence, position, caches and rings in a
small per-step descriptor, so one CUDA graph per row count serves any sequences (each sequence also keeps
its last step's drafter features, so its drafts do not depend on other sequences' steps in between), and
each row's results
do not depend on the step's other rows (skinny tensor-core GEMM over up to 64 rows, the GEMV in launches
of at most 16, per-pair experts, per-row attention splits). A sequence's logits, caches and drafts are
bit-identical alone and batched (`quench6k inkling batch-check` compares them and measures batched
steps). One sequence of 1,042,346 tokens prefilled at 1.7K tokens/s (450 W) and decoded at 34 tokens/s
with 38.9 GiB of caches (256K: 4.6K and 97 tokens/s, 11.9 GiB; 512K: 2.9K and 60 tokens/s, 21.1 GiB).

Its correctness gate compares teacher-forced distributions against `tools/ref/inkling_ref.py`, which
drives the transformers implementation's decoder layers (transformers 5.24 or later) one layer at a time
on the GPU, in BF16 (reference A) and FP32 (B):

```
python3 -I tools/ref/inkling_fixtures.py --model $M --qwen $SWIFT --out out/inkling-fixtures
python3 -I tools/ref/inkling_ref.py --model $M --fixture out/inkling-fixtures/* --out out/inkling-ref --suffix=-bf16
python3 -I tools/ref/inkling_ref.py --model $M --dtype fp32 --fixture out/inkling-fixtures/* --out out/inkling-ref --suffix=-fp32
# copy each <fixture>-bf16/-fp32 pair into its fixture directory as ref-bf16 / ref-fp32, then
quench6k inkling score --model $M --fixture out/inkling-fixtures/* --out out/inkling-eng
quench6k gate --fixtures out/inkling-fixtures --candidates out/inkling-eng --ref-a ref-bf16 --ref-b ref-fp32
```

`quench6k inkling generate` and `quench6k inkling bench` run greedy generation and throughput
measurements (`--draft`: drafted decoding).

### Recursive Flint

Recursive Flint (`FlintForCausalLM`, 40B-A2.5B: 49 layers, 14 of full attention and 35 of 513-token
sliding-window attention, 256 experts per layer) runs from the vendor's NVFP4 GGUF ([README](README.md#recursive-flint)).
Each sequence's full-attention caches (28 KiB per position with BF16 KV, 14.4 KiB with FP8) are
reserved for the sequence's capacity (at most the model's 262,144 positions) but backed by device memory
only as positions are reached, so concurrent sequences share the memory by their actual lengths; the
sliding layers keep rings of 8,192 positions (up to 560 MiB per sequence, and as much again for the
copy taken at a prompt's end for prefix reuse). Up to 8 sequences with up to 8 tokens each decode in one
step; every row is computed exactly as a one-token step of its sequence alone, so a sequence's tokens do
not depend on which sequences share its steps or on its draft lengths (speculative greedy output equals
plain greedy output). `quench6k flint generate`, `ppl`, `bench` and `batch-check` are described in
[DEVELOPMENT.md](DEVELOPMENT.md#recursive-flint).

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

### Slots for Inkling 2, Hy4-flash and Flint

These models run in a scheduler of their own over their engines (`crates/cli/src/serve/slots.rs`), with
the same policies:

- Up to `--slots` requests at once, each slot with its own sequence state and cached prefix. A request
  goes to the free slot whose conversation it continues (the slot holds the most of its prompt, and at
  least half of what the slot holds is that prompt's beginning), else to the least recently used one:
  a conversation's slot is not taken by another that only shares its beginning (a system prompt, a
  template header). `--context` is each slot's capacity (prompt + completion).
- One prompt is prefilled at a time, in the engine's chunks; while other requests decode, in slices of
  about 4096 tokens that take turns with their steps by `--prefill-share`.
- The decoding requests step together, each with its own drafts (the checkpoint's drafter as each
  request's own draft policy decides, or prompt lookup; request field `mtp`) and sampling. A request's
  tokens never depend on which others share its steps: concurrent greedy requests give the same text
  as alone.
- Drafts under batching: a batched step's time is shared out by what each sequence costs it (the
  scheduler fits step time = a + b x rows over recent steps: each sequence pays the fixed part split
  evenly plus its own rows), which is the time its draft policy sees; and while several sequences step
  together, a sequence verifies no more drafts than pay for their rows at the batch's plain rate
  (seqs / (a + b seqs) tokens per second), from its own acceptance rate. Alone, the draft policy
  decides as before.
- Memory: a request starts when its plan fits next to the running requests' plans in free device
  memory and what idle slots hold; when the device is full, idle slots give memory up, least recently
  used first (memory mapped past what a slot holds, then its whole context: its cached prefix is lost;
  these models have no context store). A request that does not fit waits for running ones to finish.
  With growable states the per-request limit is device memory as well as `--context`: a prompt that
  cannot fit even alone fails at once, and a request that runs out of memory with nothing else
  holding it fails, both with the positions one sequence fits on the device. Cancellation, stop
  tokens and budgets are per request, and an error in one slot resets only that slot.
- All three engines run one batched step (`forward_batch`) for the decoding sequences (and Inkling's
  and Hy4-flash's drafters one pass, `draft_batch`), with growable states: a sequence's full-length
  caches are mapped as it grows, so slots share the device memory by their actual lengths.
  - Hy4-flash: about 17 KB per position plus about 106 MB per sequence. With the weights loaded, about
    4.5 GB remain on a 96 GB card: some 260K positions for one sequence, or 27K each for eight.
  - Inkling 2: 36 KB per position plus 3.1 GB per sequence (rings) and a boundary snapshot; with eight
    sequences about 51 GB stay for positions (some 1.4M in all: one of 1M, or eight of 128K).
    Prompts prefill canonically (chunks at multiples of 2048 positions), so output depends only on a
    conversation's tokens, not on what was cached.
  - Flint: 28 KB per position (BF16 caches; about half with `--kv fp8`) plus the sliding rings (up to
    560 MB per sequence).

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

### Several models

`quench6k serve-models --catalog FILE --listen ADDR [--settle SECS]` serves every model of a catalog
from one endpoint and loads none at start. Each model runs as its own `quench6k serve` process on a
local port, one at a time; requests (streams included) pass through to it.

```json
{"settle_secs": 60,
 "models": [
   {"args": ["--model", "/models/qwen", "--name", "qwen", "--context", "262144", "--kv", "fp8"]},
   {"args": ["--model", "/models/inkling", "--name", "inkling", "--context", "131072"], "env": {}}
 ]}
```

`args` are `serve` arguments without `--listen`; `env` adds environment variables. A model's names (its
`--name`, and the adapted or steered name with `--adapter` or `--steering`) and context come from them.

- `GET /v1/models` and `/v1/models/{id}` answer from the catalog without loading anything: names,
  `context_length` / `max_model_len`, and `loaded`. `/health` reports the loaded model and its state.
- A request names its model (the body's `model`). Nothing loaded: that model loads and the request
  waits for it. Another model loaded: the server switches (stopping that model's process, which frees its
  memory, then loading the requested one) only when the loaded model has been idle, with no request
  running, for the settle period (default 60 s). Until then the request is rejected with HTTP 503, code
  `model_switch_blocked` (Anthropic paths: `overloaded_error`) and a `Retry-After` header, so clients
  asking for different models cannot make the server load and unload in turns.
- Reads (`GET`, such as `/`, `/metrics`, `/stats`) go to the loaded model and never load one: with none
  loaded, or with `?model=` naming another model than the loaded one, they get HTTP 404 (what clients
  probing for endpoints such as llama.cpp's `/props` expect from a server without them). Other requests
  without a model go to the loaded model; with none loaded they get HTTP 400 `no_model_loaded`.
  `POST /v1/unload` stops the loaded model when no request is running.
- A failed load is reported to the waiting requests; the next request tries again.

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
| `QUENCH6K_HY4_DFLY=0` | Hy4-flash: do not load the DFly2 drafter (the server then drafts by prompt lookup) |
| `QUENCH6K_HY4_DRAFTS=N` | Hy4-flash, `quench6k hy4 ... --dfly`: verify N drafts per step instead of choosing the count |
| `QUENCH6K_HY4_W4A4=1` | Hy4-flash: prefill experts with NVFP4 activations on the FP4 tensor cores (faster; fails the gate) |
| `QUENCH6K_HY4_DUMP=DIR`, `QUENCH6K_HY4_TIMING=1` | Hy4-flash debugging: per-layer hidden states, per-step host timings |
| `QUENCH6K_INK_INT8=LIST` | Inkling 2: BF16 projections quantized to INT8 (a scale per 64 weights) at load, as name fragments (`all`, `none`; default the layers' and the drafter's attention and MLP projections) |
| `QUENCH6K_INK_FP8=LIST` | Inkling 2: BF16 projections quantized to FP8 at load, as name fragments (`all`, `none`; default `model.llm.unembed`, the LM head) |
| `QUENCH6K_INK_FP4_TC=1`, `QUENCH6K_INK_FP4_TERMS=N` | Inkling 2: prefill experts of the NVFP4 layers with NVFP4 activations of N terms (default 2; 3 for the experts' input) on the FP4 tensor cores (faster; fails the gate) |
| `QUENCH6K_INK_SKINNY=N` | Inkling 2: small steps of N or more tokens run the INT8 projections as a tensor-core GEMM (default 1: every step, so a token's result never depends on the step's width; 0: the GEMV) |
| `QUENCH6K_INK_MARKOV_CAND=N` | Inkling 2: the drafter's Markov head picks among each 256-token block's N best tokens (default 4; 0: the whole vocabulary) |
| `QUENCH6K_INK_PREFETCH=0`, `QUENCH6K_INK_CLUSTER=0` | Inkling 2: small steps without the L2 prefetch of the next visit's weights / without the 8-block cluster residual mixture |
| `QUENCH6K_INK_GUESS=0`, `QUENCH6K_INK_GUESS_K=k1,k2,..` | Inkling 2: steps of up to 4 tokens without the guessed experts (the next visit's router on this visit's MLP input; the best guesses' gate/up weights requested into L2 after the next visit's output projection) / the guesses per row for steps of 1, 2, .. tokens (default 3,2,2,1) |
| `QUENCH6K_INK_GRAPHS=0` | Inkling 2: run small steps and the drafter eagerly instead of as CUDA graphs |
| `QUENCH6K_INK_EXPERT_GROUPS=N` | Inkling 2: small steps of N or more rows (default 16; 0: never) run the routed experts in groups of up to 4 pairs of the same expert (each expert's rows read and decoded once per group; the same bits as per pair) |
| `QUENCH6K_INK_DRAFT=0` | Inkling 2: do not load the DSpark drafter (the server then drafts by prompt lookup) |
| `QUENCH6K_INK_DRAFTS=N` | Inkling 2, `quench6k inkling ... --draft`: verify N drafts per step instead of choosing the count |
| `QUENCH6K_INK_DUMP=DIR`, `QUENCH6K_INK_DUMP_DRAFT=DIR`, `QUENCH6K_INK_TIMING=1` | Inkling 2 debugging: the stream after each layer visit, the drafter's inputs and outputs, per-step timings |
| `QUENCH6K_INK_PROFILE_STEPS=N`, `QUENCH6K_INK_PROFILE_WIDTH=W`, `QUENCH6K_INK_CONF_LOG=FILE` | Inkling 2 profiling (`inkling bench`): N decode steps of W tokens in a profiler capture range; (`inkling generate --draft`) each judged draft's position, stated probability and outcome |
| `QUENCH6K_PDL=0` | Launch kernels without programmatic dependent launch (checks whether a result depends on PDL ordering) |
