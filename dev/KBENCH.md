# Fast kernel iteration: `quench6k kbench`

Goal: try many kernel variants (tuning parameters, fused kernels) for the decode path in seconds each,
with real weights and real activations, a bit-exact correctness check, and timings that resolve 1-2%,
without loading the 72 GiB model or running the full engine for every idea.

## Why the engine loop is too slow

- A full engine run loads 72.5 GiB (about 30 s), prefills, then benchmarks: 1-2 minutes per idea.
- Single benchmark runs vary by about 3% (the GPU is power-capped at 600 W), so gains under 3% are
  invisible without many repeats.
- Isolated microbenchmarks with synthetic weights (`crates/cuda/tests/micro.rs`) are fast but
  misleading: no real routing, no real value ranges (exactness hinges on ties), and kernels timed
  outside the decode graph lose the overlap of programmatic dependent launch (PDL). The HyperConnection
  merge looked like a win in isolation and lost inside the graph.

## Approach

### 1. Truncated model

Load only the first K layers (default 4: three GDN layers and one attention layer, one full period of
the layer pattern) plus embedding and final norm: about 6 GiB instead of 72.5, a few seconds from the
page cache. The engine's own code runs unchanged: the same weight formats, kernels, CUDA graphs and PDL
chains as the full model, so per-layer timings and numerics are the real ones. Decode time per token
scales with the layer count (48 / K), plus the LM head.

### 2. Real state, replayed

Prefill a real context (the benchmark corpus, default 8192 tokens) into the K layers, then snapshot the
sequence state (GDN recurrent and conv state, PLE conv state). A decode step is captured as one CUDA
graph over R rows (R = 1: plain decode; R = 4: an MTP verify round) and replayed N times from the
snapshot. Replays process the same tokens at the same positions, so the KV rows they write are the same
each time and the GDN state evolves deterministically: every replay sequence is reproducible.

### 3. Runtime kernel variants

Kernels read tuning parameters (rows per block, warps, prefetch depth, which steps are fused) from a
small variant table set from Rust (`quench6k_cuda::variants::set("gemv.warps", 4)`), at launch time. All
variants are compiled into the binary as template instantiations, so a sweep needs no rebuild: kbench
sets a configuration, re-captures the decode graph, and measures. A variant that is not compiled in is
rejected at launch, not silently replaced. Every key defaults to the engine's current behaviour.

### 4. Correctness: bitwise first

For exact changes (our rule for everything that is not a measured precision change), a candidate must
reproduce the baseline bit for bit: after every replay the final hidden state (`ws.out`) and the
residual stream are hashed, and the sequence of hashes must equal the baseline's. This takes
milliseconds and checks every layer type, every kernel in the chain and the state carried between
steps. Candidates that pass go on to the slow tiers only when they are finalists:

1. kbench bitwise check (seconds, every candidate)
2. kernel unit tests (`cargo test -p quench6k-cuda --test kernels`)
3. engine: `quench6k generate` on all fixtures, `quench6k gate`, `quench6k bench` greedy check (finalists)

### 5. Timing that resolves small differences

- Baseline and candidate graphs alternate in pairs (A B B A ...), each pair from the same restored
  snapshot, so drift in clocks and temperature cancels.
- The result is the median ratio candidate / baseline over the pairs, with a spread (25-75%), not a
  single absolute number.
- The working set is the real one (weights for K layers, about 0.5 GiB per decode step), larger than
  the 128 MB L2, so weights stream from DRAM as in the full model.

### 6. Sweeps

`quench6k kbench --sweep gemv.warps=4,8 --sweep gemv.rows=1,2` runs the cartesian product, prints one
line per configuration (median ratio, spread, bitwise yes/no) and the best exact configuration.
Variant kernels are templates over these keys; new fused kernels are added as further values of a key
(for example `hc.mode=split|fused`), so a hand-written fusion is swept like any parameter.

## Limits

- K layers time the per-layer kernels; the LM head and the MTP drafter are outside the truncated model
  and are measured by the full engine.
- The final numbers (tok/s) still come from `quench6k bench` on the full model; kbench ranks candidates
  and proves exactness.
- Prefill kernels need captured 8K-token batches (about 1 GB of activations per layer); the same
  structure applies, with R = chunk rows, but the first version covers decode.

## Measurement (2026-10-27)

### Experiment ledger

`bench/kbench.jsonl`: one JSON record per measured configuration, so that "was this tried?" is a grep,
not a read through the dated findings below. `kbench --log bench/kbench.jsonl --note "what is tried"`
appends the baseline and every swept configuration:

| Field | |
|---|---|
| `date`, `commit` | UTC date; the checkout's HEAD, `+dirty` when tracked files differ from it |
| `layers`, `ctx`, `rows`, `steps`, `pairs` | The kbench configuration |
| `fixed`, `variant` | Keys set for baseline and candidate (`--variant`), and the candidate's own keys (`{}`: the baseline) |
| `us`, `floor_us` (baseline) | Microseconds per step, and the step's weight-streaming floor |
| `base_us`, `ratio`, `q25`, `q75`, `exact`, `finite` (candidates) | Median candidate / baseline time over the pairs, its spread, bitwise equality with the baseline, all values finite |
| `rejected` | The launch error, when a configuration is not compiled in |
| `gpu` | As in bench/README.md, with `power_limit_w` (600 for comparisons) |
| `sass` | Results of `--sass` checks |
| `note` | `--note` |
| `verdict`, `why` | Filled in by hand after the run: `kept`, `dropped`, `neutral` or `baseline`, and one line of reason |

Records with `"backfill": true` and a `source` are the earlier rounds' experiments, taken from the
findings below and from ROADMAP.md (no commit or configuration; `ratio` where a number was recorded).
The ledger holds numbers and settings only: no prompts or outputs.

### SASS check

A hint the compiler ignored changes no output and possibly no timing, so a neutral result does not
tell "did not help" from "is not there". `quench6k sass --file gemv [--kernel REGEX] [--top N]` lists
the kernels of `kernels/gemv.cu` in the binary (demangled), their instruction counts and local loads
and stores (register spills), and optionally the most frequent opcodes. `--check FILE/KERNEL=INSTR`
(also kbench's `--sass`, recorded in the ledger) requires every kernel matching KERNEL to contain an
instruction matching INSTR, and prints the first match's position:

    quench6k sass --check 'gemv/gemv_kernel<\(WFmt\)3, \(int\)4, __nv_bfloat16, \(int\)2, \(int\)5, \(int\)8>=LDG\.E\.NA'

A verdict then has three outcomes: effective (present, faster), ineffective (present, within noise),
not implemented (absent). One file takes ~8 s; all of them ~20 s. ncu cannot attach on this driver
(see 2026-10-19), so `--clock-control` advice for ncu does not apply here; nsys does.

First use: the GEMV instances the engine runs (INT8G, 5-deep prefetch, 8 warps) spill nothing; 43 of
the variant-only instances do (BF16 with 16 warps or 8-deep prefetch: up to 944 local loads).

### Which step to rank on

The service runs sampled MTP on one slot at 55K tokens (median) up to 190K: the 4-row verify step at
long context, not plain decode at 8K. Rank candidates at `--rows 4`, and at `--ctx 65536` and `163840`
as well as 8192. The verify step barely depends on context (4 layers: 662 / 656 / 677 us at 8K / 64K
/ 160K; 8 layers: 1232 / 1238 / 1256 us), so the per-layer chain, not attention, is where long-context
time goes; the full model adds the drafter and LM head (MTP 561 -> 520 tok/s from 8K to 260K).

kbench checks every replay for non-finite values in the final hidden state and the residual; a
candidate that produces any is marked `NON-FINITE` and never "best".

### Real conversations as benchmark prompts

`bench/context.ids` is tiled text: greedy MTP accepts 6.9 tokens per round on it, real traffic 2.2-3.6,
so its MTP numbers overstate real speed by ~1.7x and weigh the draft path wrongly. The server's
context files (`--cache-dir`) hold real conversations:

    quench6k context-ids ~/.cache/quench6k/<id>.ctx --model M --out ~/bench-private/<id>.ids
    QUENCH6K_BENCH_IDS=~/bench-private/<id>.ids quench6k bench --model M --ctx <its length> --mtp 6 --adaptive --sampled

`context-ids` cuts after the last `<|im_start|>assistant\n`, so `bench` generates a real assistant
turn. It refuses contexts with images (their spans hold negative content keys and need the pixels)
and refuses to write inside a git checkout. **These files are private conversation data: keep them
outside the repository, never commit them or records that quote them, and do not put their names or
content in the ledger or in any document.** Pick conversations that share only the system prompt.

Measured 2026-10-27 (600 W, three conversations of 20K / 91K / 170K tokens, 512 tokens generated):
sampled MTP 277 / 368 / 346 tok/s at 2.3 / 3.6 / 3.4 tokens per round, greedy 275 / 488 / 341. The
service log shows the same range (`quench6k traffic`: 305-336 tok/s at 3.1-3.6 tokens per round).
Single conversations vary a lot; compare candidates on the same set and alternate.

### Real traffic

`journalctl --user -u quench6k -o short-iso | quench6k traffic [--weekly] [--since YYYY-MM-DD]`
summarizes the server's per-request log lines per period and context range: requests, generated
tokens, decode tok/s (generated / decode time), MTP tokens per round, cache reuse, the share of time in
prefill, and prefill tok/s over requests that prefilled at least 4,096 new tokens. It caught what no
benchmark did: cold prefill went 3-5K -> 9-10K tok/s and decode ~225 -> 305-336 tok/s when the PLE
table was stored uncompressed (2026-10-26).

### Serving load

`tools/loadgen.py` drives a running server from the outside with synthetic agent-like conversations
(public filler only: the repository's tracked files or generated text, never real conversations):
several at once, each turn appending the reply and a new task with filler so the contexts grow towards
a target. `sweep --levels 1,2,4,8` gives aggregate and per-conversation throughput and latency per
concurrency, with the wait for a slot measured separately from prefill (a one-slot model shows its
queue as waits that grow with the level while aggregate tok/s stays flat); `consistency` compares the
greedy texts of each conversation alone and all at once, turn by turn (each conversation with a
system prompt of its own, the model unloaded between the phases, so both reuse the same cached
prefixes: prefill bits depend on where a chunk starts, so a different reuse is a different
computation, not a batching effect); `fill` grows N conversations to
a target each (e.g. 4 x 256K) and reports per turn how much of the prompt the server reused, then
revisits each to show what was evicted. Qwen picks MTP draft widths from timing and its numerics
depend on the width, so its greedy text varies with load: compare with the server started under
`QUENCH6K_POLICY=fixed=N`, where alone and together must match exactly. See tools/README.md.

### Census

`tools/census.py` credits each kernel of an nsys trace (the command is in its docstring) with its
critical-path contribution: its end minus the latest earlier end. Raw durations under PDL include
the wait on the predecessor and add up to ~2x the busy time; the credits add up to it exactly. Weight
loads issued before `griddepcontrol.wait` are credited to the predecessor, so a kernel's credit is
what it costs after its inputs are ready. `--shape` splits by grid and stream (which GEMV is which);
`--side REGEX` keeps side-stream kernels (L2 prefetches) out of the chain.

`--record NAME -- <bench command>` runs nsys itself with node-level graph tracing and GPU metrics
sampled at 200 kHz (5 us), then reports. With the samples each kernel's credited time also gets the
DRAM read bandwidth measured during it (percent of peak), and a summary line gives its distribution
over the busy time. Decode is a memory-bound chain with latency-bound links: the summary says how much
of the step leaves DRAM idle, and the column says which links do. Node tracing adds a few percent per
round; the bench's own number is the one without it.

### A/B

`tools/ab.py` runs variants of a bench command interleaved (A B B A per round, `--runs` rounds) and
prints each throughput's median and range per variant with the change against the first. Variants are
environment settings (`-v off:QUENCH6K_INK_ATTN_FUSED=0`) or another binary (`--bin old=PATH`).
Interleaving cancels clock and temperature drift: Inkling decode repeats within 0.1-0.2%, so 1%
differences are real, where a single A-then-B pair could not tell them from drift.

2026-10-27, full model, fp8 KV, 64K context, MTP 3 (4-row verify), 20 rounds, 600 W: 8.87 ms busy
per round (verify ~8.1, draft ~0.77), 957 kernels, 0.23 ms idle.

| Group | Per round | Share | Per call | Note |
|---|---|---|---|---|
| Routed experts (gate/up + down, 49 layers) | 3.17 ms | 36% | 49 + 16 us | ~1.7 TB/s for ~110 MB per layer: at the floor |
| Dense GEMVs (212) | 2.90 ms | 33% | | |
| - GDN in-projection (36) | 1.11 ms | 13% | 31.7 us | 1.37 TB/s, the one GEMV below ~1.6 |
| - output projections (95) | 0.67 ms | 7.5% | 7.0 us | weights mostly preloaded under the predecessor |
| - LM head (1) | 0.41 ms | 4.6% | 411 us | 1.59 TB/s: at the floor |
| - attention QKV (12), router + shared gate/up (47) | 0.49 ms | 5.5% | 21 / 4.9 us | the 113-block router GEMV at ~0.9 TB/s |
| HyperConnection (106 x 3 kernels) | 1.12 ms | 12.6% | 5.0 + 1.7 + 3.9 us | down, finish, up-mix |
| GDN recurrence (36) | 0.44 ms | 4.9% | 12.2 us | |
| QSA attention split + merge, select, index (15) | 0.68 ms | 7.7% | 25 + 4.5 + 8 + 8 us | a faster reduction is a numeric change |
| Routing, combine, SwiGLU, small kernels | ~0.3 ms | ~3% | 0.7-2 us | |

What it says about fusion (the 2026-10-27 kernel survey estimated ~2.5 us of ramp and tail per removed
kernel, ~480 us per token for fusing the HC chain and the MoE chain):

- The ramp is already hidden: every decode kernel loads its weights before the PDL wait, and the
  kernels that fusion would remove are credited 0.7-2 us each (finish 1.7, routing 0.7, SwiGLU 0.9,
  combine 2.0). Removing all of them outright would save ~0.4 ms per round (~4.5%); a fused kernel
  keeps their work.
- The HC's cost is not its kernel count: down and up-mix are separated by two grid-wide reductions
  (sums of squares over 20 splits, dot products over 20 splits), and a fused kernel needs the same
  barriers in-kernel. Every version measured lost: whole rows per block +6%, finish in up-mix
  +3-5%, finish by the last block +2-5%, per-row-block tickets +0.6-1.7% (2026-10-16 to 10-19). The
  most a perfect fusion could take is hc_finish's 1.7 us x 106 = 0.18 ms per round (2%).
- The real remaining gaps against bandwidth (worked on below): the GDN in-projection (1.37 TB/s;
  at 1.6 TB/s it would save ~0.16 ms per round), the HC's ~0.6 TB/s for 6.6 MB per call (post-wait
  latency of three dependent phases, not bandwidth), and the QSA split + merge.

## 2026-10-27: the three census targets

kbench, 4 layers, fp8 KV, 64K, 4 rows, 600 W; each line is in `bench/kbench.jsonl`. ncu attaches on
this driver now: `ncu --cache-control none --clock-control none -k regex:<kernel> --set full` around
`kbench --steps 1 --pairs 1` (kernels run alone there, without PDL overlap, so use it for where a
kernel's time goes, not for its in-graph cost). `gemv.rows/N`, `gemv.warps/N` and `gemv.pre/N` set
one GEMV shape (N output rows) without touching the others (`sq_variant_shape`).

- **QSA decode split: kept, exact, +1.4% end to end.** ncu: SMs 85% active, ~2,950 instructions per
  warp, mostly shared-memory loads of q and of the probabilities: issue-bound, not memory- or
  occupancy-bound (staging the FP8 K/V raw for three blocks per SM, one wave, alone: +-0). The new
  kernel (`attention_split_raw8_kernel`, `qsa.raw8`, default 1) scores each warp's four keys together
  (q read once per column instead of once per key), reduces the 48 dot products with two
  `warp_sum_slots`, and reads probabilities four keys at a time in the PV loop; every sum keeps its
  operands and order. 23 -> 11 us standalone, -1.27% per kbench step. Full model (MTP 3,
  alternating): verify 8.20 -> 8.09 ms per round, draft 0.772 -> 0.749 ms, 446 -> 452 tok/s at 64K,
  437 -> 443 at 160K. `score` byte-identical on all fixtures, spec-check (code, long-context) and
  batch-check exact. The merge spread over 2 / 4 blocks per head: +-0.
- **GDN in-projection: unchanged.** ncu: 1.26 TB/s alone, long-scoreboard stalls; every block issues
  all its weight loads at once, so the 2.7 waves reach DRAM as bursts with gaps between them. Every
  rows x warps shape for N = 16480 alone is slower (+0.8..+4.2%). A persistent version (two CTAs per
  SM, each warp loads the next row before computing the current one; exact) is +1.9%: it preloads
  ~1 MB before the PDL wait instead of the first wave's ~15 MB and keeps too little in flight; with
  three CTAs per SM it spills (+2.8%).
- **HyperConnection: unchanged.** ncu (warm L2): down / finish / up-mix 3.7-4.4 / 2.7 / 4.7 us alone,
  stalls on the loads of z and the residual, SMs ~40% idle within each kernel: latency of dependent
  round trips. In the trace each HC kernel's credit does not depend on its predecessor (hc_down 5.1
  us after the combine, 4.7 after a GEMV), so block residency is not the limit; two row tasks per
  hc_down warp (half the blocks) is +1.3%.

## Findings (2026-10-16)

- Method check: a sweep over an unread key gives 1.000 +-0.1%; achievable DRAM read rate on this card
  (600 W) is 1.647 TB/s; a dependent kernel boundary costs 0.45-0.54 us with PDL (0.68 us without).
- Decode floor: a 4-layer step reads 318 MB of static weights (80 MB per GDN layer, 75 per attention
  layer) plus about 110 MB of routed experts: 260 us at 1.647 TB/s against 490 us measured (53%). Per
  layer of the full model: about 102 us against a 65 us floor.
- The gap is not kernel boundaries (about 6.5 us per layer for 13 dependent phases) and not idle DRAM
  (L2 prefetch of the next layer's weights gains at most 1%, and hurts above 32 MiB): it is the latency
  of each phase's dependent part (fresh activations from L2, compute, reduction, store), 2-3 us per
  phase whatever its size.
- Variants (default off): `gemv.pre/warps/rows` (four-row steps now default to 2 rows per warp, -1.2%),
  `moe.t1_prefetch` (+1.8%: the early down blocks slow gate/up), `l2.prefetch` (MiB per layer; -1% at 1
  row, +0.7% at 4 rows).
- What would close the gap: fewer dependent phases per layer, or a persistent kernel whose phases wait
  on per-tile flags instead of whole-kernel completion, so each consumer starts as soon as its own
  inputs exist and keeps streaming weights while it waits.

## Findings (2026-10-16, later): verify cost, prompt lookup, megakernel

- Decode step time against rows (kbench, 4 layers, 8K context, relative to one row): 2 rows 1.28,
  4 rows 1.62, 8 rows 2.23, 12 rows 3.32, 16 rows 4.03. Every extra row routes to its own 10 experts,
  so a verify step costs more weight traffic per row; the step from 8 to 12 rows is out of line.
- Prompt-lookup drafting (n-gram match in the context, verified with the MTP verify step), simulated on
  the service trace (6,439 requests, 150K generated tokens) with these step costs: lookup could supply
  17-23% of the tokens, but on top of MTP (413 tok/s) it gains at most +2-3% overall (+8% on
  generations over 1,000 tokens), and loses 6-30% without gating it on the drafter's first token. MTP
  already takes most of the predictable tokens, and long verify steps are expensive. Not implemented.
- Megakernel prototype (synthetic, one GDN decode layer's 12 phases with their real byte counts and
  all-to-all dependencies between phases, 113 MB per layer, floor 68.5 us):
  - one PDL kernel per phase, weights loaded before the dependency wait (the engine's design):
    103.6 us per layer, matching the engine's GDN layers;
  - persistent kernel, one CTA per SM, a producer warp streaming the CTA's weights through a 5 x 16 KB
    shared-memory ring with TMA bulk copies, consumers synchronized by completion counters: 101 us
    (with the activation reads idealized away), 81.6 us with no dependency waits at all;
  - persistent kernel with per-thread or TMA bulk L2 lookahead prefetch: 110-128 us.
  - an all-to-all completion-counter sync costs 0.55-0.6 us per phase (188 CTAs), against 0.46 us for
    a PDL kernel boundary: flags are not cheaper than PDL.
  The engine already gets the megakernel's main benefit (weights streaming while the previous phase
  finishes) from PDL and pre-wait weight loads; what remains is about 1.6 us per all-to-all dependency,
  whatever the synchronization mechanism. Published megakernel gains (1.5x and more) are measured
  against engines without that overlap. A full megakernel is not worth its cost here; fewer dependent
  phases per layer would be.
- Verify-step follow-up: GEMV for 9-15 rows now runs in one pass instead of 8 rows plus the rest
  (exact; little gain: those steps are dominated by expert traffic and dense GEMV compute). The main
  step block now uses the layout of its own token count, so a decode step uploads about 5 KB instead of
  136 KB (token and position areas sized for 8,192 tokens): -28 us per plain step and -44 us per 4-row
  verify step at this host's PCIe link (Gen 3 x4, 3.6 GB/s measured; exact). Tried and dropped: a
  register cap for the 4-row GEMV (`__launch_bounds__` min blocks 2-4: 1.4-15% slower) and the fused
  shared-expert tail for multi-row steps (2.5% slower at 4 rows: every block recomputes the SwiGLU input).

## Findings (2026-10-16, night): precision, host overhead, prefill shapes

- Lower-precision dense weights do not fit the gate. Fake-quantized at load
  (`QUENCH6K_FAKE_FP4_W=classes`, `QUENCH6K_FAKE_BITS=b` for integers with a scale per 64; classes qkv, z,
  gdnout, attnin, attno, shared, hc, router, plus mtp), each scored and gated against out/g7:
  NVFP4 for every large dense class fails 11 of 12 checks; alone, each of z, qkv, gdnout, attno, hc,
  shared and attnin fails at least one completion bound (z and qkv only narrowly); INT6 and INT5 for
  all dense classes fail 3 and 5 completion checks. The current configuration's margins are small
  (multiturn completion p99 0.300 against a bound of 0.340). Dense weights stay 8-bit.
- Host waits: `cudaDeviceScheduleSpin` makes no difference (the default already spins). Launching the
  second half of a decode or verify step before the host gathers its PLE rows (a device wait on a
  pinned flag) is correct but not faster: unprofiled, the launch already overlaps layer 0 (the gaps
  seen under nsys come from its API tracing).
- Prefill PLE rows (5 KB per token over the 3.6 GB/s link) are now gathered and uploaded in
  1,024-token pieces on the side stream (+0.5-1% prefill).
- Context-store switching between conversations costs about 0.4 s per switch at 22K tokens (stash:
  0.08 s pinned allocation + 0.146 s copy of 498 MiB; load: 0.146 s), all PCIe; rare in the service
  trace (5 of 107 requests over 8K switch back to an earlier conversation).
- Prefill shapes at 8K: BF16 GEMMs 26% (HC down N=324 at about 168 TFLOPS; 64-column tiles for small
  grids are 1% slower overall), FP4 expert GEMMs 15% (bounded below by reading every expert's weights
  once per chunk: 1.4 GB per layer), GDN chunked 11%, MXFP8 in-proj 9%, attention 8%, HC combine+norm
  13% (at the memory floor of the FP32 residual). A 16K chunk is 3.7% faster at 32K but needs 6.4 GB
  more device memory than the service has.
- Long-context decode costs little: 5.23 ms per token at 8K, 5.58 ms at 250K.
- Drafting only from a frequent-token subset of the vocabulary would cut the draft head's 1-bit
  sketch pass (238 us of an 839 us draft graph) but covers only 94% of generated tokens; not worth the
  lost acceptance.
- Sampled rounds (koder's default: temperature 1.0, top_p 0.95, top_k 20) had serial tails: top-64
  of 248K logits per verified row with one block per row (105 us for 4 rows), the sampled draft's
  proposal in FP64 on one lane (45 us per draft), the verifier's sampling in FP64 on one thread
  (41-85 us per round). Now: top-k in 32 slices then a merge, with the radix digit found by a block
  scan (17 us; exact, test against a CPU reference); the draft proposal warp-parallel in FP32 (15 us;
  any proposal keeps sampling exact since the verifier uses its recorded probabilities; MTP and plain
  draws agree, z within +-1.3); the verifier's exponentials on 32 lanes (bit-identical outputs).
- Prompt tokenization: 26 ms per request at 26K tokens, now cached per `<|im_start|>` segment
  (0.4 ms for a re-sent conversation, exact). Streaming detokenization is O(1) per token.
- Single-TF32 GDN prefill is 4% faster but fails the tool-call completion bound (also with TF32
  only in the per-chunk products); 3xTF32 stays.
- HC up-mix loads its first token group's residual, pending update and RMS factor together with z
  (-0.15% / -0.29% per step at 1 / 4 rows, exact). Deeper GEMV preloads for long K, 64-column GEMM
  tiles and the INT8 wide kernel for the HC down-projection in prefill were all slower.
- Context store: a stash allocates a fresh pinned buffer (0.07-0.08 s for 500 MiB) and evictions write
  to disk synchronously inside the insert; both only cost on conversation switches, which are rare.
- Full model after this session (FP8 KV): 8K prefill 14.7K tok/s, plain decode 190 tok/s, greedy MTP
  416 tok/s; 32K: 14.4K / 189 / 408 tok/s. Sampled MTP (temperature 1.0, top_p 0.95, top_k 20) on
  the service: 236-350 tok/s at 2.3-3.7 tokens per round, depending on the text.
- Small-batch MoE routing uses warp-shuffle scans for its expert offsets and active list (-0.34% per
  4-row verify step, exact).
- Candidates left (larger work): verify-row attention sharing K/V gathers across the rows (the
  prefill multi-query kernel does this for 4 queries, but verify rows must keep plain decode's per-row
  arithmetic); a split multi-query attention for decode; per-expert dependencies between the expert
  gate/up and down kernels (the 1-row version was 1.8% slower); the prefill multi-query attention
  with 8 queries per block needs about 122 KB of shared memory, over the 99 KB limit.
- Fusing hc_down and hc_finish for one token with whole rows per block (4 rows x 4 streams, the same
  split partials and sums, so bitwise identical) is 6% slower per decode step: 81 blocks instead of
  420 and a serial 4-thread finish (with 162 two-row blocks: 5.6% slower; every block stages the
  whole 40 KB residual, where hc_down stages 2 KB per block). The three-kernel HC stays.
- Draft length policy: forced fixed lengths (2, 4, 6 drafts) are 7-10% slower than the adaptive
  policy on sampled requests; its decay (0.75-0.95) makes no measurable difference.

## Findings (2026-10-19): decode kernels after the megakernel survey

Measurement: `QUENCH6K_DUP=op` (an op launched twice inside the graph) gives each op's marginal cost
with PDL overlap; ops whose weights fit in L2 show only their latency on the second launch, and
differences under ~5 us are run-to-run noise. ncu cannot attach on this driver; nsys is not installed.
Per 4-layer step at 8K, 1 row / 4 rows: experts +60 / +146 us, hc_small (9 calls) +60 / +71,
gdn proj + out +54 / +77, qsa_attention_split +22 / +25, gdn_fused_small +23 / +19, each small
attention-layer kernel about 5-10.

Kept (all bit-identical: `score` outputs byte-identical, spec-check and batch-check exact):
- Chained small experts (`moe.chain`, default 1): the down kernel launches while gate/up runs, loads
  its weights at once and waits on a per-slot counter of finished gate/up blocks (zeroed by the
  routing kernel) instead of the whole kernel. kbench -3.5% / -2.6% / -0.75% at 2 / 4 / 8 rows.
- Weight loads that do not allocate in L1 (`ld.global.nc.L1::no_allocate`): activations re-read for
  every weight chunk stay in L1. Experts -2.1% / -2.4% at 4 / 8 rows (`moe.nl1`, default 1); every
  other decode weight load (dense GEMV, HC, single-token experts, NVFP4 dense) -0.5% / -1.9% at 1 / 4.
- L2 evict_first on those weight loads (createpolicy + .L2::cache_hint), so activations, KV, GDN state
  and the MTP draft sketch stay in L2: kbench -10% / -6% (overstated: the 4-layer model's working set
  fits in L2), full model +5% on MTP rounds (greedy MTP 512 / 8K: 327 -> 343 / 537 -> 566 tok/s over
  the L1 change alone). Plain 1-row decode is unchanged on the full model (5.33 ms per token).
- Together, full model: greedy MTP 314 -> ~343 tok/s at 512, 521 -> ~564 at 8K (verify 11.7 -> 10.9
  ms, draft 1.61 -> 1.37 ms per round); batched plain steps of two sequences 276 -> 312 tok/s.

Tried and dropped:
- hc_finish fused into hc_up_mix (every block recomputes z): +3-5% slower. hc_finish done by the
  down kernel's last block (a ticket): +2-5% slower. A kernel boundary costs ~0.5 us; a serial tail
  or redundant per-block work costs more. The HC's three dependent phases (~8.6 us at 1 row against a
  ~4 us weight floor) stay.
- Split attention with 64 / 128 keys per block: +2 / +5% at 1 row; 16 keys: +1%. 32 stays.
- Draft sketch loads with evict_first: draft graph slower (the 99 MB sketch partly stays in L2 across
  the drafts of a round); with evict_last: no change. GDN state loads/stores with evict_last: no change.
- GDN state prefetch: the state is already loaded before the PDL wait. Splitting the 48-block GDN
  kernel across more SMs would shorten only the pre-wait part.
- Smaller MTP draft vocabulary (frequent tokens + sequence tokens): the draft head is 48 us per call
  over the full vocabulary (microbenchmark), so a 32K subset would save ~1.5% of a round.
- hc_finish done per 16-row block by the last of its 20 split blocks (tickets, waiting for row block
  0's sums of squares): +0.6% / +1.7% / +1.1% at 1 / 4 / 8 rows. In-kernel cross-block handoffs
  (fence, atomic, spin) cost more than a PDL boundary unless they let work of different waves overlap,
  which is why chaining helps the experts (40 x ~37 gate/up blocks, several waves) and nothing else.
- Chaining a GEMV's columns into a consumer (router columns -> routing, gdn proj -> gdn_fused, index
  columns -> the QSA indexer chain): the producers run in one wave (moe proj ~225 blocks) or the
  consumer could overlap only the last wave; not attempted.
- Tensor cores for 9-16-row GEMVs (pair rounds, where the GEMV drops to 0.85 TB/s): not attempted;
  pair rounds are rare in the service's traffic.

Where the time is now: at 4-8 rows the routed experts run at their bandwidth floor (the duplicated
cost matches ~60 unique experts per layer at 8 rows). The remainder grows with rows: about 96 / 107
/ 138 us per layer at 1 / 4 / 8 rows against ~48 us of static weights, in the HC kernels, the GDN
recurrence and attention. The generic GEMV holds 1.4-1.6 TB/s up to 8 rows and drops to 0.85 TB/s at
16 rows (pair rounds).

## 2026-10-20: verify row counts and the draft policy

Sampled serving traffic (8 coding and prose prompts, temperature 0.6 and 1.0, 2 seeds, 600 tokens,
`--kv fp8`, steering, 600 W): rounds commit ~2.4 tokens, and a round with 4 drafts (5 verified rows)
cost 1.6 ms more than one with 3, against ~0.8 ms for the neighbouring steps.

- The small-batch launchers instantiated 1, 2, 4, 8 and 16 rows, so 5-7 rows ran the 8-row
  instances. Exact instances per launcher (kbench `rows.gu`, `rows.dn`, `rows.sc`, `rows.hc`: bit T
  set = exact instance for T rows), full model, 8K, fp8:
  - experts gate/up: 5 rows -3.5%; 3 rows +2.1%, 6 rows +-0, 7 rows +0.5%.
  - HC: 5 rows -2.2%, 6 rows -1.1%, 7 rows +0.2%. No 3-row instance: groups of 3 tokens give 6 / 12
    reduction slots and `warp_sum_slots` needs a power of two (spec-check failed at token 4).
  - experts down, shared expert combine: no gain at any count.
  - Kept gate/up 5 and HC 5, 6 (defaults of `rows.gu` / `rows.hc`): -5.6% per 5-row step, -1.2% per
    6-row step; exact.
- Beyond that, each row from the fifth on adds ~0.3 ms of gate/up and ~0.15 ms of down: about five
  newly routed experts per layer per token, real traffic.
- Draft policy (`QUENCH6K_POLICY`): decay 0.8 / 0.95, prior 0.6 and draft temperature 70% / 85% are
  all within the run-to-run spread of the adaptive baseline (279-294 tok/s); fixed 3 drafts 292-296.
  The first configuration after a build or a long idle measured 4-8% low on every request; compare
  configurations only after a warm-up run. The policy stays as it is.
- DFlash on the same sampled traffic: 2.49 tokens per round, the same as MTP (2.44-2.49), at 258 tok/s
  against 285-293 (drafting 1.4 ms per round against MTP's ~0.6). Its acceptance lead on the earlier
  single chat prompt (3.50 vs 3.18) does not carry over, so cheaper drafting could at best tie MTP.
  The MTP drafter's sketch head for DFlash's 7 rows (sketch shortlist, NVFP4 rescoring, top 16):
  0.27 ms against 0.46 ms for the LM head and top-k, but 2.34 tokens per round and 250-257 tok/s:
  dropped, as is a 4-bit drafter.

## 2026-10-21: windowed draft attention and draft trees (both dropped on measurement)

Prompted by EAGLE-4 (sliding-window draft layer) and Kapitsa's KFlash (best-first trees over a block
drafter's candidates). MTP 3 drafts, `--kv fp8`, greedy corpus, 600 W:

- Draft per round is 0.745 ms at 8K, 0.773 at 64K, 0.836 at 256K (verify 8.00 / 8.31 / 8.60). The
  0.58 -> 1.6 ms growth measured on 2026-10-15 predates the sketch head. In the eager profile the
  indexer scores + top-512 select of one QSA layer cost 14 us at 8K and 36 us at 256K, so a window
  in the MTP layer would save at most ~0.09 ms of a 9.4 ms round (~1%), at some acceptance risk.
- Verify per round by rows (8K): 3 rows 7.29 ms, 4 rows 8.00, 5 rows 9.00, 6 rows 9.99: each extra
  row costs 0.7-1.0 ms (~10% of a round), the weights of the experts it newly routes to. A tree row
  costs the same. On sampled traffic the first draft is rejected in ~27% of rounds (fixed 1 draft:
  1.73 tokens per round); even if the MTP's second candidate matched half of those, a sibling row
  adds <= 0.14 tokens per round (~5.5%) for ~10% more verify time, and it is worth less than the
  third chain draft it would replace (fixed 2 -> 3: +0.33 tokens). Exact-match acceptance for tree
  rows would also be stricter than the ratio test. KFlash measured the same shape on a MoE target
  (+0.6% average on Qwen3.6-35B-A3B, -5% on long agent turns). Trees stay off for this model.
- What does grow with context is the target's own QSA selection: 12 layers x ~22 us more at 256K
  than at 8K (~0.26 ms of the 0.6 ms verify growth); qsa_select alone is 20 us per call at 65K groups.

## 2026-10-21: launch gaps (nsys) and the shared expert on a side stream

Nsight Systems ships inside Nsight Compute here (`/usr/lib/nsight-compute/host/target-linux-x64/nsys`):
`nsys profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi` around `bench
--ncu-steps N`, then `nsys stats -r cuda_gpu_trace -f csv` gives every graph node's start and duration.

- MTP 3, 20 rounds, fp8: the GPU is idle 0.9% of the time at 8K and 1.2% at 256K (union of kernel
  intervals). Of that, ~50-70 us per round are host round trips between the draft and verify graphs
  (inflated by nsys's API tracing). Launch gaps are not worth a persistent kernel; per-kernel durations
  are mostly PDL overlap (their sum is ~1.9x the busy time).
- The shared expert's SwiGLU and down projection (small batches) ran after the routed experts' down
  kernel, adding ~4 us per MoE layer to the critical path, while the routing kernel (one block)
  left the GPU idle. They depend only on the MoE projection, so they now run on a side stream forked
  before the routing and joined before the combine (`moe.shared_side`, default 1; same kernels, so the
  output is bit-identical: spec-check bf16/fp8 x code/multiturn x 3/6 drafts and eager, batch-check).
  MTP 3, fp8, 600 W, alternating: verify 8.00 -> 7.88 ms at 8K, 8.60 -> 8.48 at 256K; 455 -> 462 and
  424 -> 429 tok/s (+1.4%).
- Still on the 4-row critical path per layer: routed experts ~56 us (at their bandwidth floor), each
  HyperConnection ~14 us (3 dependent kernels, ~6.6 MB of weights), combine ~2 us, routing ~4 us after
  the projection. QSA attention (split + merge) costs 25 us per QSA layer in-graph (kbench
  `QUENCH6K_DUP=qsa_attention_split`, 4 rows: 662 -> 687 us per 4-layer step), ~3.7% of a verify step,
  for ~2 MB of FP8 K/V; selection and index scores ~1-1.5 us each. The split kernel's 65 chunks of 32
  keys write 12 x 256 partial floats each (6.4 MB per layer at 4 rows) and the merge reads them back in a
  65-step serial loop. A faster attention changes the summation order: a numeric change (score + gate).

## 2026-10-22: Recursive Flint decode

Flint (40B-A2.5B, vendor NVFP4 GGUF) shares no kernels with the Qwen path except the grouped W4A16
expert GEMM (prefill). Decode at 8K went 317 -> 443 tok/s (450 W, BF16 KV) through these steps, each
measured with nsys per-kernel traces (`QUENCH6K_VARIANTS=flint.pdl=0` so that durations are not
inflated by PDL waits):

- **PDL and `__restrict__`.** With PDL, the first version crashed (illegal address) only without
  compute-sanitizer: a dense layer's expert kernel read the slot table that the preceding kernel writes
  through a `const __restrict__` pointer, and read-only loads may be hoisted above `griddepcontrol.wait`
  (it read the previous step's MoE slot ids against a 12-entry dense table). Kernels that read what the
  previous kernel wrote must not mark those pointers `__restrict__`; `flint.cu` uses none. Bisected
  with the `flint.pdl` bitmask (PDL per kernel) and `QUENCH6K_FLINT_SYNC=op`.
- **GEMV shape.** A warp per row with 4-byte code loads reached ~0.5 TB/s. Keeping x in registers per
  lane (no shared memory) was worse: every warp re-reads the whole x row, 20 MB of L2 traffic for 6 MB of
  QKV weights. What works: x staged once per CTA in shared memory with 16-byte chunks XOR-swizzled
  (`j ^ ((j >> 3) & 7)`, conflict-free for lanes reading 32 consecutive values), 16-byte code loads (32
  weights and two scale bytes per lane), 2-4 rows per warp. The LM head runs at 1.58 TB/s.
- **RMSNorm.** Normalizing per CTA in the GEMV prologue (global -> shared -> block sum -> sync) cost
  ~4 us per call and lost; folding it into the output instead (stage bf16(h * w), sum of squares in the
  same pass, scale the dot products by rsqrt at the end) gains ~2% over a separate RMSNorm kernel. The
  router does the same per warp (each warp reads the whole row anyway) and routes in its last CTA.
- **Experts.** Six routed experts plus the shared expert as two 512-wide halves are 8 slots per token
  (dense layers: 12 slices of the 6144-wide MLP). The down projection with a warp per slot and
  `warp_sum_slots` over 8 rows per CTA (256 CTAs at T = 1) halved its time; 32 rows per CTA (64 CTAs)
  was slower than the original.
- **Attention merge.** Merging in the last split's CTA (ticket) was slower than a separate merge kernel,
  even with the merge spread over warps and lanes: one CTA per (row tile, KV head) is too little
  parallelism. Kept separate (`flint.attn_fused_merge=1` selects the fused one). The split count is
  derived on the device from the position (>= 4 key tiles per split), so CUDA graphs stay valid.
- **Decode attention.** At T = 1 only 8 of a CTA's 64 query rows are real. `flint.attn_dec` (default)
  lets the four warps share 16 rows and split each 64-key tile instead: identical greedy output, +3% at
  8K, more with FP8 KV.
- **FP8 KV.** Converting E4M3 tiles to BF16 with their scales in shared memory made decode slower (255
  vs 289 tok/s at 64K). Converting to FP16 unscaled (one cvt per two values; E4M3 is exact in FP16),
  FP16 MMAs, the key scale applied to the score columns and the value scale folded into P: 333 vs 300
  tok/s at 64K, 188 vs 146 at 260K (with decode-mode attention). PPL 24.98 vs 24.93 (2K), 15.20 vs
  14.95 (16K). Prefill is 5-20% slower with FP8 (the conversion is not hidden there).

Not done: prefill attention at long context is compute-bound at ~136 TFLOPS effective (260K prompt in
49 s at 600 W); larger tiles or FP8 MMAs would help. Single-token GEMVs (QKV 6.6 us, O 6.3 us) are at
their latency floor; fusing more of the per-layer chain (9 kernels) is the remaining decode lever.

Later the same day:

- **Prefill.** At 64K, prefill time splits into the grouped W4A16 GEMMs (45%), attention (41%) and a
  standalone residual add (5%, now folded into the FFN RMSNorm). Attention runs at ~170 TFLOPS
  effective, near what BF16 MMAs with FP32 accumulation give. The engine's W4A4 expert path (the Qwen
  prefill's FP4 tensor-core kernels; activation rows quantized after gathering since the FP4 gather is
  fixed to 10 slots) gives +21% at 8K and +13% at 64K, but PPL 15.61 vs 14.95 at 16K, worse than
  llama.cpp's 14.71: kept off (`flint.prefill_fp4`). W4A8 would be the accurate way.
- **Prompt-lookup speculation** (server). Drafts are the tokens after the latest earlier occurrence of
  the last 4, 3 or 2 tokens; [pending, drafts] (<= 8 tokens) run as one graph step (graphs per token
  count), each row is sampled and accepted while it equals the draft (exact for deterministic drafts),
  and the sequence is truncated to the accepted tokens (all caches are position-indexed). A
  26K-token agent edit request: 376 -> 713 tok/s sampled (3.88 tokens/step). The batched verification
  kernels (M-row GEMV, non-decode attention) round differently from the single-token ones, so greedy
  output diverges from plain greedy at near-ties (after ~40 tokens on one prompt).
- **Adaptive drafts.** With matches of 2 tokens and fixed 7-token drafts, free-form text lost 14%
  (drafts rejected, larger steps). Matches of at least 3 tokens and a draft cap from the request's
  acceptance rate (EMA; caps 1/2/4/7): free-form 491 vs 494 tok/s, the agent edit 738 vs 382.
- **Prefix reuse.** Reusing a prefix of length p only needs the sliding rings to hold [p - 512, p)
  (a rewritten chunk of <= 2048 positions never overwrites a key its own queries need), so the limit is
  p >= cached - 3583, not the 1535 first used. The rings are also copied at every prompt's end (35
  layers, 287 MB, a device copy): a follow-up that diverges anywhere after that point (reasoning dropped
  from the history, re-rendered whitespace) restores them and keeps the whole earlier prompt (26,085 of
  26,107 tokens reused after 3,000 generated, instead of a full prefill).
- **Wide prefill attention.** Steps of more than 2 tokens use 128-row tiles (8 warps, 16 tokens x 8
  heads) with 32-key tiles (`flint.attn_wide`, default on; 70 KB of shared memory): K/V tile traffic
  and FP8 conversion per FLOP halve. Prefill at 64K +6.5% (BF16 KV) / +6% (FP8). PPL moves by rounding
  only (2K chunks within 0.5% either way: 24.83 vs 24.93 overall; tiled corpus 1.0390 vs 1.0391).
- **Two-term FP4 activations** (tried, reverted): NVFP4 of x plus NVFP4 of its residual under a
  shared row scale, summed in one W4A4 GEMM over repeated weight k-tiles (K = 2 kw). PPL 25.18 / 15.18
  vs 24.83 / 15.03 (W4A16), but prefill 17.1K vs 17.9K tok/s at 8K: with ~64 rows per expert per
  2048-token chunk the FP4 kernel's 128-row tiles run half empty.
- **Prefill chunk.** 4096-token chunks (sliding ring 8192 so a chunk plus the window fits) instead of
  2048: more rows per expert per GEMM. Prefill 17.9K -> 21.3K tok/s at 8K, +8% at 64K, +23% at 131K;
  decode unchanged; the reuse reach grows to 7679 positions. 8192-token chunks (ring 16384) add only
  2-3% for ~1.5 GB more workspace.
- **L2 prefetch of the chosen experts** (tried, reverted): the router's last CTA issuing
  `cp.async.bulk.prefetch.L2` for the 8 slots' gate/up/down weights (216 pieces of 64 KiB) made decode
  45% slower (276 vs 501 tok/s at 512): issuing them serializes the router's tail, which the expert
  kernels wait for.
- **Decode splits.** An FP8 decode-attention CTA takes ~85 KB of shared memory (one per SM), so 64
  splits x 4 KV heads = 256 CTAs leave a partial second wave on 188 SMs. A cap of 94 (two full waves)
  with the split count recounted from the tiles per split (empty splits dropped; 8K keeps 64 x 2 tiles):
  FP8 decode at 64K 340 -> 357 tok/s, 8K unchanged (`flint.splits`).

## 2026-11-22: Recursive Flint, several sequences per step

Up to 8 sequences x up to 8 tokens per step (`FlintModel::forward_batch`), every row bitwise a
one-token step of its sequence (`quench6k flint batch-check` checks it, and that drafted greedy
output equals plain greedy output). How each kernel keeps a row's bits independent of the step's width:

- **GEMVs** (QKV, O, LM head): a row-wise twin of the single-token NVFP4 GEMV (`fp4_gemvb_kernel`): the
  same lane layout and product order per row, weights held in registers and read once for all rows,
  rows staged through shared memory 8 (K = 2048) or 4 (K = 4096) at a time. Normalized inputs are
  staged once per step by `prenorm_kernel` (bf16(h * w) and the RMS factor, summed exactly as the
  single-token kernel's prologue). The first version took 248 registers (one CTA per SM, two waves):
  22.8 us for QKV at 8 rows vs 6.5 us at 1 row; eight weights at a time and batched staging loads
  brought it to 127.
- **Attention**: each row runs as a one-token decode-mode CTA on its own sequence's caches (device
  table of per-layer cache addresses, so a CUDA graph per row count serves any sequences). Tried
  (removed): two consecutive rows of one sequence sharing a CTA in full-attention layers when their
  key splits have the same tiles per split (K/V read once, the later row's extra tile all masked for
  the earlier one). Bitwise up to 3 tiles per split (13K keys), but at 4 (20K) the later row differed
  reproducibly; cause not found.
- **Router and experts** were already per token; the router takes 2 tokens per CTA row (8 per row:
  +5% step time at 8 rows).

Cost: one-token decode runs the same kernels (474 vs 480 tok/s at 2K, 449 vs 449 at 8K, 600 W);
verification steps of 2..8 tokens cost within +-5% of the old width-dependent kernels. Aggregate
decode at 600 W, 2K / 8K prompts: 1 / 2 / 4 / 8 sequences 484 / 735 / 1042 / 1323 and 451 / 676 /
928 / 1165 tok/s. At 8 rows (nsys, 2K, before the register fix) expert gate/up took 25% of the step,
expert down 15%, the three GEMVs 36%, attention 14%: the experts of 8 tokens are mostly different.

**PDL stale tables.** A fresh server's greedy output was not reproducible with PDL (0-1 of 8 prompts
identical between runs, once an illegal address); `QUENCH6K_PDL=0` or `flint.pdl=511` (Flint's
planning kernels without PDL) fixed it. The grouped NVFP4 GEMM (`moe.cu`) read the tile tables that
Flint's planning kernels rewrite in place through `__restrict__` (non-coherent) loads; `__ldcg` fixed
it (8/8, several fresh servers). Reading Flint's activations through L2 as well (defensive) cost 14%
of a decode step (rows of h, x and act are reused through L1 by the CTAs of an SM) and was reverted.

## 2026-11-12: prefill after rtp-llm's Qwen4 PLE/GDN work

Alibaba's rtp-llm PRs [#1673](https://github.com/alibaba/rtp-llm/pull/1673) and
[#1681](https://github.com/alibaba/rtp-llm/pull/1681) tune `qwen4_exp` (this architecture) for H20 TP2 at
batch 16-64: one GEMM for the PLE key and value projections, the PLE convolution, history roll and
residual add in one kernel, Q/K normalization fused into the GDN prefill scatter, GDN decode
convolution fused with the gate, and narrower value tiles (BV16) in the GDN chunk-state kernel. Our
engine already fuses the PLE and GDN decode work (`ple_gate` + `ple_conv`, `gdn_fused_small`); PLE is
0.7% of a decode step and 0.5% of prefill here, so packing its projections is not worth it.

`bench --profile` now also times one eager prefill per op. 8K prefill at 600 W (15.3K tok/s): GDN
chunked 14.4%, routed experts 15.6% (gate/up 9.9%, down 5.7%), QSA attention 8.8% (11.4% at 64K), the
four HC ops 25%, GDN in/out projections 13.5%, FP4 activation quantization 4.1%, QSA projections 5.8%.

- **GDN state walk** (kept, exact): the chunk-state kernel is 72% of `gdn_chunked_tc` (1.51 of
  2.10 ms per layer at 8K; prep 0.55 ms). Without its MMAs it takes 0.58 ms, so it is bound by the
  3xTF32 MMA chains, on 144 of 188 SMs (one block per SM by shared memory; a fourth column slab per
  head would be 192 blocks, a second wave). BV16 does not apply: narrower slabs mean more blocks than
  SMs. Two phases per chunk instead of four (W S, Q S and Vn = U - W S by the same warps; then O and
  the state update), four block barriers instead of eight, and `mma_pair` interleaving two tiles' MMA
  chains (W S and Q S share the S fragments): 2.107 -> 1.999 ms per layer (-5.1%), outputs and state
  bit-identical (`micro gdn_chunked_tc_8k` hashes them).
- **FP4 activation quantization** (kept, exact): 214 + 277 us per layer at 8K (the 8192 x 2560 expert
  input and the 81920 x 640 down input), about 7x its memory floor: the least-error block scale tries 5
  scales x 16 IEEE divisions per block. The E2M1 code and value now come from `rintf` (step 0.5 below
  2, 1 up to 4, then 4 or 6; ties to even, which is what the thresholds of `e2m1_code` give, checked
  case by case at every threshold), and the winning candidate's codes are kept instead of dividing the
  block a sixth time: 171 + 225 us (-19%), bytes identical (`micro fp4_quant_rows_prefill`). Division
  stays IEEE: a reciprocal-based code needs an exact fallback near every threshold, which costs more
  than the division.
- Full model, `score --decode` on the six oracle fixtures: every `.bin` byte-identical to main.
  Prefill, two interleaved base/new pairs at 600 W (FP8 KV; the GPU warms from pair to pair): 8K
  +0.35% / +1.2%, 64K +0.09% / +0.9%, 131K -0.12% / +0.8%. Expected from the kernels: ~1.5% at 8K.
- **Routed experts** (not attempted): gate/up run at ~490 TFLOPS and down at ~430 TFLOPS, about a
  quarter of FP4 peak. (The guess here that half-empty 128-row tiles were the cause was wrong; see
  2026-11-14: both kernels move only their compulsory DRAM traffic.)
- `kernels` test `expert_gate_up_small_matches_cpu` fails on main too (t 2 row 0 feature 0: -0.150 vs
  -0.118); its hand-built offsets probably no longer match `moe_gate_up_small`'s layout, and in parallel
  runs the out-of-bounds reads poison the shared context, so other tests fail with error 700. Decode
  sanity checks and oracle scores are unaffected. Not fixed yet.

## 2026-11-14: routed-expert FP4 GEMMs under ncu

`fp4_expert_gemm_prefill` in
crates/cuda/tests/micro.rs runs one 8K chunk's routed GEMMs (512 experts, top 10, 81,920 rows; a skewed
synthetic routing with 872 tiles, 73% full, 43 to 755 rows per expert).

- gate_up with SwiGLU (N 640 x 2, K 2560): 988 us, 543 TFLOPS. down with BF16 output (N 2560, K 640):
  628 us, 427 TFLOPS. Live profiles agree (~490 and ~430 TFLOPS).
- **Both are bound by DRAM traffic they cannot avoid.** gate_up moves 1.16 GB from DRAM (65% of peak).
  Its compulsory traffic is 1.17 GB: every expert's gate and up codes and scales (944 MB), the
  quantized rows (118 MB) and the output (105 MB). down moves 0.88 GB at 83% of peak: 420 MB of
  weights and 420 MB of BF16 output. Tile fill doesn't matter because tensor throughput is 38% and 32%.
  Only a larger chunk lowers the weight bytes per token: see the 2026-10-16 findings, where a 16K chunk was +3.7% at
  32K but needed 6.4 GB more device memory.
- gate_up stalls: MIO throttle 34% of samples, math pipe 15%, barrier 12%, wait 13%, long scoreboard
  9%. Achieved occupancy is 28% (2 CTAs per SM, limited by registers and shared memory).
- Tried, no gain (all bit-identical):
  - 64-value k-tiles in 4 or 5 stages, to keep more reads in flight: gate_up 1606 us (+63%), down
    +2%. Per-step barriers and copy instructions cost more than the extra bytes in flight gain.
  - Both k64 steps' block scales in one 8-byte shared load: no change.
- The BF16-output down kernel spills 4 bytes per thread (ptxas), which is negligible.
- What is left would need a different kernel design to raise gate_up's DRAM efficiency: warp
  specialization with TMA bulk copies, fewer registers for 3 CTAs per SM. The bound is at most about
  0.2 ms of 1.0 ms per layer, or about 2% of prefill.

## 2026-11-22: Hy4-flash batched steps

`quench6k hy4 batch-check` steps several sequences alone and batched and compares them bitwise
(teacher-forced logits rows and cache checksums; greedy decoding with drafts: tokens, drafts, logits;
plain greedy against drafted), and with `--bench-ctx` times batched decode steps. `quench6k hy4
width-check` compares a fixture's positions run in one-token steps with wider steps. Extra batch-check
cases: `QUENCH6K_HY4_INTERLEAVE=1` (one sequence's short step after another's one-token steps),
`QUENCH6K_HY4_STAGGER=1` (short raw prompts started between running sequences' steps, as a server runs
them), `QUENCH6K_HY4_PICK=topk` (greedy picks by the top-k kernel, as the server's sampler, instead of
argmax). `QUENCH6K_TOPK_ATOMIC_TIES=1` restores top-k's old timing-dependent choice among tied values,
to tell tie effects from other differences.

- Width invariance before this work: steps of 1 to 6 tokens gave the one-token bits, 7 and more did not
  (the attention split each row's keys by the step's width). With a fixed split (16 for sliding, 32 for
  indexed layers, what one-token steps used) 1 to 8 tokens give the same bits; 9 to 16 still differ (the
  routed experts switch to the grouped tensor-core GEMM there, single-sequence steps only).
- The indexer's top-k picked tied groups by thread timing (many groups score exactly 0): drafted greedy
  differed from plain greedy on long prompts. Ties now go to the lowest group indices (topk.cu).
- Aggregate decode throughput, 600 W, prompts from the long-context fixture (tokens per second over all
  sequences; drafted: 3 drafts per sequence per step):

| ctx | sequences | plain ms/step | plain tok/s | drafted ms/step | drafted tok/s |
|---|---|---|---|---|---|
| 512 | 1 | 9.8 | 102 | 18.6 | 103 |
| 512 | 2 | 11.7 | 171 | 25.3 | 198 |
| 512 | 4 | 15.9 | 251 | 41.2 | 306 |
| 512 | 8 | 23.7 | 338 | 72.8 | 333 |
| 8192 | 1 | 10.6 | 94 | 19.3 | 102 |
| 8192 | 2 | 12.6 | 158 | 27.0 | 153 |
| 8192 | 4 | 16.8 | 238 | 43.3 | 229 |
| 8192 | 8 | 24.3 | 329 | 76.2 | 246 |

  Plain steps scale well (8 sequences in 2.4x the time of one); verified drafts stop paying at about 4
  sequences: a batched step's cost grows with its rows (every row is a GEMV row and 16 expert pairs), so
  with many sequences fewer drafts per sequence should be verified.
- Dropped: routed experts per run of pairs of one expert (each work item loading its expert's weight rows
  once for up to 8 pairs, the same arithmetic): 4 x 4-row steps 171 ms instead of 41 ms.
- PDL and served short prompts (2026-11-22): a server with 8 slots, drafts on, gave 1 in 12-24 short prompts
  (9-16 tokens, run as one CUDA graph step with the grouped NVFP4 expert GEMM) a different output 8 at once
  than alone. Cause: the grouped GEMM read its tile tables, which the planning kernels rewrite in place just
  before, through the non-coherent path (the Flint fix in moe.cu, already merged; the server builds
  predated it). With that fix reverted, `scratch` trials (58 prompts alone, then shuffled rounds 8 at once,
  120 tokens each) showed 1 of 258 different; with it 0 of 606. The Hy4 kernels' own in-place tables and
  small per-step buffers (routing, pair order, tile planning, key lists and their lengths, attention split
  partials, hyper-connection weights, the drafter's kept lists) are now read through L2 as well: the same
  bits (teacher-forced scores in 12-token steps byte-identical), decode 97.5 / 98.1 tok/s against 97.5 / 99.8.
  `QUENCH6K_HY4_STAGGER_DRAFTS=N` in batch-check runs short prompts started between N running sequences'
  verified steps and drafter passes, the server's order.

## 2026-11-22: Inkling 2, several sequences per step

`quench6k inkling batch-check --model M --text CORPUS --ctx 512 8192 32768 --seqs 1 2 4 8` first decodes
each sequence alone (its own stretch of the corpus, 37 tokens shorter per sequence; plain steps, then
drafted steps whose widths vary by sequence and step) and records every step's fed tokens, drafts, draft
probabilities and logits rows (FNV hashes of the F32 bytes), then replays the same steps in lockstep
batches (`forward_batch`, `draft_batch`) and compares bitwise. It then times batched steps from the
prompts' ends: plain steps (one greedy token per sequence, argmax and download included), verify steps
of 4 tokens per sequence (the corpus's next tokens as drafts, one kept) and drafter passes.

600 W, max clock 3090 MHz, aggregate tokens per second of plain decode (step time); 32K measured before
the grouped expert kernels (plain steps of up to 8 rows do not use them):

| context | 1 | 2 | 4 | 8 sequences |
|---|---|---|---|---|
| 512 | 233 (4.30 ms) | 356 (5.62) | 526 (7.60) | 672 (11.91) |
| 8192 | 227 (4.41) | 337 (5.93) | 468 (8.55) | 559 (14.30) |
| 32768 | 197 (5.09) | 276 (7.23) | 368 (10.88) | 428 (18.69) |

Verify steps of 4 tokens per sequence: 550 / 667 / 800 / 911 rows/s at 512 (7.3 / 12.0 / 20.0 / 35.1 ms),
520 / 610 / 713 / 797 at 8K; 32K (before grouping) 428 / 507 / 545 / 563. Drafter passes for 1 / 2 / 4 /
8 sequences: 0.86 / 1.18 / 2.07 / 3.96 ms at 512, 1.3 / 2.1 / 4.0 / 7.9 ms at 32K. A plain step's extra
sequence costs about 1.1 ms at 512: its 8 experts per layer visit (48 visits x 8 x 4 MB of NVFP4 weights,
about 1.5 GB) are new DRAM traffic. Verify steps grow with their pairs (the per-pair expert kernels
re-read and re-decode an expert's rows for every pair routed to it), not with the experts read: in an nsys
capture of these runs (`--cuda-graph-trace=node`) the two expert kernels take a third of all GPU time.
Grouping up to 4 pairs of an expert per warp item (rows loaded and decoded once, each pair's sums as
before: the same bits) gives 8 sequences' verify steps 39.0 -> 35.2 ms and 4 sequences' 20.4 -> 20.0 ms,
but 8-row steps 11.9 -> 13.3 ms, so it starts at 16 rows. The FMAs per pair stay; a tensor-core expert
kernel (tokens as the MMA's N, so a row's sums still do not depend on the others) would remove that, but
its sums would differ from today's kernels in order, so it needs the gate, not a bitwise check.

Long contexts, one sequence, `inkling generate --prompt-file` (a needle sentence 40% into repository text,
then a question about it), 450 W: 253,964 tokens prefilled in 55 s (4.6K tokens/s), decode 97 tokens/s,
11.9 GiB of caches; 523,508 tokens in 182 s (2.9K tokens/s), 60 tokens/s, 21.1 GiB; 1,042,346 tokens in
626 s (1.7K tokens/s), 34 tokens/s, 38.9 GiB (58.4 GiB on the device in all). The model quoted the needle
at 256K and 512K, and was quoting it at 1M when the run's 40 tokens ended. At 1M a decode step reads each full-attention visit's 4 GiB of keys and values.

Canonical prefill (`quench6k inkling canon-check --model M --text CORPUS`): a conversation of 3 turns (a
5000-token system prompt, user turns of 600 to 800 tokens, 24 greedy reply tokens) decoded cold (a fresh
sequence per turn), warm (one sequence) and cross (the sequence first serves another conversation with
the same system prompt, decodes batched with another sequence, and generates 3104 more tokens before the
last turn, so that turn restores the boundary snapshot): every turn's prompt logits, reply logits and
tokens are bit-identical in the three. Reuse stops at the last 2048 boundary the previous turns
prefilled whole: warm turn 2 reused 4096 of 5623 cached tokens (1527 recomputed beyond the new ones),
turn 3 6144 of 6344 (200).

## 2026-11-22: canonical prefill for Hy4-flash and Recursive Flint

`quench6k hy4 canon-check` / `quench6k flint canon-check --model M --text DOCS.md dev/KBENCH.md`: three
conversations of three turns (a shared system prompt of 3109 / 6181 tokens, user turns of 500 to 950
tokens from the text, 12 greedy reply tokens per turn). Cold computes every turn on a fresh sequence and
defines the replies; warm runs each conversation on one sequence, decoding in verification steps of up to
3 tokens whose last draft is wrong (so rows past the kept position are written and dropped), and before
the last turn feeds RING + CANON_CHUNK tokens in unaligned chunks (a discarded generation), so that turn
restores the boundary snapshot; cross runs the turns turn by turn on two sequences, so most turns land on
a sequence that held another conversation (reuse up to the shared system prompt's boundary, from a live
ring or a lower snapshot), with a third sequence stepping after every prefill chunk and decoding batched
with the turn. Every turn's rows (the prompt's last and each reply token's: FNV/SipHash of the F32 logits)
and tokens are compared with cold.

Result (450 W): all 18 warm and cross turns bit-identical to cold for both models. Reuse and extra prefill
(tokens recomputed beyond what reusing the whole cached common prefix would have recomputed):

| | turn 1 (warm) | turn 2 (warm, snapshot restored) | cross turns |
|---|---|---|---|
| Hy4 (C = 2048) | reused 2048 of 3620..3882, +1572..1834 | reused 4096 of 4229..4753, +133..657 | reused 2048 of 3109, +1061 |
| Flint (C = 4096) | reused 4096 of 6692..6954, +2596..2858 | reused 4096 of 7301..7825, +3205..3729 | reused 4096 of 6181, +2085 |

The extra cost per turn is the cached tokens past the last whole chunk: up to C - 1 prompt tokens plus the
previous reply, on average about C / 2 (Hy4 about 0.26 s at 3.9K tokens/s, Flint about 0.1 s at 19.7K).
Halving the chunk would halve it but slows all prefill: 4096 Hy4 tokens prefill at 3922 tokens/s in chunks
of 2048 and 3212 in chunks of 1024 (-18%); 8192 Flint tokens at 19,660 in chunks of 4096 and 17,306 in
chunks of 2048 (-12%). So C stays the largest chunk each engine runs (Hy4's ring holds a 2048-token chunk
plus the window).

A row's bits do not depend on its chunk's length in either engine: positions 0..776 and 2048..2824 (Flint:
4096..4872) computed as a 776-token chunk or as the start of a full chunk (then truncated) give a
bit-identical next-token row. Only the chunk's start matters (attention splits and selection depend on
where a chunk begins), which canonical geometry fixes.

Snapshot costs: Hy4 copies 30 sliding layers x 512 rows x 1152 bytes (17.7 MB) per 2048-token chunk, Flint
35 x 2 x 513 x 1 KiB (35 MiB) per 4096-token chunk, both as device copies on the prefill's stream (tens of
microseconds of copy per chunk of hundreds of milliseconds); the prompt-end snapshots they replace copied
17.7 MB (Hy4) and up to 560 MiB (Flint) per prompt.

## 2026-11-23: Recursive Flint, several conversations at once

Where a batched step of 8 sequences at 2K went (nsys, PDL off, 6.1 ms): expert gate/up 1.67 ms and down
0.97 ms (both at about the DRAM read ceiling for the experts the 8 tokens pick), the three row GEMVs 1.91
ms (QKV 0.79, O 0.82, LM head 0.30: 2.5-2.8x their one-row time for the same weights), attention 0.80 +
0.12 merge, router 0.34. The server added 0.27 ms per step: a top-k launch, download and synchronization
per sequence after the step. All changes below are bit-identical (batch-check digests at 2K and 16K and
rows vs alone, canon-check, served texts greedy and sampled):

- **Row GEMVs** (`fp4_gemvc_kernel`): each staged input value is read from shared memory and converted
  once per warp, not once per weight row; the R x G sums reduce together (`warp_sum_slots`). Every GEMV
  kernel also picked its matrix by indexing the `Mats` parameter with a run-time index, which copied the
  struct to local memory (88 bytes of stack per thread); selects instead (QKV at one row 6.4 -> 4.8 us).
- **Fixed-grid GEMV** (`fp4_gemvp_kernel`, K = 2048, up to 8 rows; the LM head also 10..16): two CTAs per
  SM stage the rows once and walk the column blocks, prefetching the next block's weights. The per-block
  kernels staged the rows once per block: 9384 times for the LM head. Micro bench (cold weights, 8 rows):
  QKV 12.8 -> 8.1 us, LM head 291 -> 151 us; O (K = 4096) 12.8 -> 11.6 us (gemvc; nothing tried beat it:
  register-capped, M-split and L1-direct variants were slower).
- **Decode attention, 4 or more rows**: one K/V tile buffer and the 16 query rows it reads (39 instead of
  87 KB of shared memory, two CTAs per SM). +3% at 8 rows; -4% at one row, so one row keeps the old one
  (`flint.attn_sb` = smallest row count). Tried: stopping the grid of sliding layers at the window's 9
  splits (the others exit at once): one row 504 -> 470 tok/s, reverted.
- **Server sampling** (`serve/flint.rs`): a batched step's top-k candidates in one launch per 8 rows and one
  download, enqueued right after the step (each row's candidates are those of a launch for its sequence
  alone); a sequence wanting another pool fetches its own. Gap between steps 270 -> 176 us.

Batched decode (batch-check, 2K, 600 W), 1 / 2 / 4 / 8 sequences: 484 / 731 / 1039 / 1341 -> 501 / 777 /
1145 / 1539 tok/s. Served (loadgen sweep, 1 turn towards 2K, 1024 tokens, 8 slots, 600 W, two sessions
each): generated tok/s 391 / 688 / 825 / 1085 -> 402 / 744 / 884 / 1202; at 450 W, 8 at once 958 -> 1064.
Prompt-lookup drafts under load (same workload, on vs off): +2% at 1, +5-10% at 2 and 4, +1-2% at 8 (the
step-cost cap already keeps drafts to about 1.05 tokens per step at 8): kept on at every load.

## 2026-11-23: Inkling 2 batched small steps (same bits)

Where a batched step's time went (inkling batch-check, 8 sequences at 512 tokens, kernels without PDL
overlap): plain 8-row step 12.9 ms, of it routed experts 7.6 ms (per-pair kernels at 1.45 TB/s on the 52
distinct experts' bytes: the floor is the experts the rows touch, 6.5x one row's), projections 1.3, attention
1.0, mixtures 0.7, message gates 0.75; 32-row verify step 36 ms (grouped experts 25 ms); a drafter pass for
8 sequences 4.9 ms, 3.8 of it the FP8 LM head as the GEMV in 16-row launches (0.46 GB re-read per launch,
compute-bound at 16 rows). Served at 8 at once the sampler's top-k took 0.7 ms a step: Inkling's vocabulary
(200058) is not a multiple of 32, so every row ran the one-block path (84 us).

Changes, each bitwise against the path it replaces (batch-check digests of tokens, drafts, stated
probabilities and every logits row; PDL on and off): experts read their input rows from shared memory
(staged once per warp, swizzled) instead of 64-byte chunks per lane from global memory per item; the LM head
of verify steps and drafter passes in one pass over the weights (4 rows a warp, inputs in tiles of 8, the
GEMV's per-row chunk order; drafter blocks without their unused first rows); small-step attention loads a
sequence's key tiles once for the rows sharing a split start (verify rows, drafter blocks); top-k of long
rows of any length through 32 slices.

batch-check, 600 W, step ms (512 tokens: A B B A medians; 32768: one run each), before -> after:

| tokens | sequences | plain | verify-4 | drafter pass |
|---|---|---|---|---|
| 512 | 1 | 4.24 -> 4.16 | 7.16 -> 7.02 | 0.88 -> 0.81 |
| 512 | 2 | 5.56 -> 5.58 | 12.12 -> 10.55 | 1.20 -> 1.08 |
| 512 | 4 | 7.68 -> 7.52 | 20.79 -> 16.55 | 2.17 -> 1.84 |
| 512 | 8 | 12.31 -> 11.60 | 36.78 -> 29.45 | 4.22 -> 3.22 |
| 32768 | 1 | 5.93 -> 5.99 | 9.79 -> 9.15 | 1.39 -> 1.26 |
| 32768 | 8 | 19.02 -> 17.55 | 54.07 -> 42.58 | 8.23 -> 5.48 |

Served (tools/loadgen.py sweep, 1 turn towards 800 tokens, max_tokens 400, greedy; the same texts before
and after, drafts on and off), summed decode tok/s at 1 / 2 / 4 / 8 at once, 600 W: drafts on 322 / 443 /
582 / 569 -> 350 / 480 / 653 / 669, drafts off 245 / 377 / 514 / 595 -> 254 / 391 / 547 / 667; 450 W, 8 at
once (two runs each): 524, 536 -> 632, 612. At 8 at once drafting still only breaks even: the scheduler's
cap admits few drafts while a drafter pass for 8 sequences costs 3.2 ms and computes all 7 drafts per
sequence.

## 2026-11-23: Hy4-flash small steps, fewer launches (same bits)

nsys (graph node tracing, `QUENCH6K_HY4_PROFILE=1` puts batch-check's timed steps in a profiler range;
tools/census.py critical-path credits), one-token steps at ctx 1024: 1477 kernels per step at 1 and at 8
sequences. GPU busy per step 9.69 ms (1 sequence) / 20.95 ms (8); the experts were 2.0 / 10.4 ms of it (at 8
sequences about 113 distinct experts per layer, at their DRAM floor). The rest was mostly kernel boundaries:
per sublayer boundary rms_rows (one block per row) -> hc_post -> hc_mix_partial -> hc_pre (one block per row)
cost 9-15 us; the attention split merge 6-9 us (its 16 or 32 loads per thread one after another); the router
5.6 us (16 rounds of a block-wide argmax); q_a and kv_a, and the three MLP input projections, separate GEMVs
of 48-128 CTAs at 8 rows.

Changes (every one against the launches it replaces, the same arithmetic per output in the same order):
- One launch per sublayer boundary (hc_next): each of a token's 24 slice blocks takes the row norm as rms_rows
  does, updates its 512 lane values as hc_post does and writes hc_mix_partial's slice sums; the token's last
  block to finish (arrival counter) runs hc_pre, its 512-thread loops and block sum emulated by two virtual
  threads per thread. The weights load before the dependency wait. 13 us at first (hc_pre's loop of six
  dependent loads became twelve with 256 threads), 5.6 us (1 row) / 7.8 us (8 rows) with the loads unrolled.
- Stacked projections loaded as one matrix (views keep the parts for prefill): [q_a | kv_a | tk | hw],
  [down_lat | shared_gate | shared_up] (then the routed input and the shared activation in one launch), the
  dense and drafter gate/up pairs.
- Wide GEMV forms for 3 to 8 rows by matrix shape (hy4_micro, isolated, 600 W; e.g. o_proj at 4 rows 49.4 ->
  34.2 us, the stacked MLP input at 8 rows 11.1 us against 3 x 7.4): 4 x 1 / 4 x 2 at 3-4 rows, 8 x 4 / 8 x 2
  / 8 x 1 at 5-8 rows; 8 / R weight chunks before the dependency wait. The router keeps its 16-row launches
  (faster in a step than the 8 x 2 form the micro preferred).
- The split merge issues all its loads at once (templated on 16 / 32 splits), the indexer's pooling tiles its
  weights through shared memory, its prep loop is unrolled; the router sorts (bitonic in warps, 16-entry
  lists merged pairwise): hy4_route_sort checks it against the argmax rounds with tied values.
- `QUENCH6K_HY4_FUSE=0` runs the previous launches; `QUENCH6K_HY4_AB=fuse` times both in one process.

Kernels per one-token step 1477 -> 1009; GPU busy per step (nsys, same process) 9.69 -> 8.86 ms at 1
sequence, 20.95 -> 19.46 ms at 8. Dropped on measurement: weight loads before the dependency wait in the
per-head GEMV (W_uv 14 -> 18-21 us: the extra registers cost occupancy); the routed experts' first weights
loaded during the stacked MLP GEMV (that GEMV triggering its dependents late, the pair order first): gate/up
40 -> 74 us at 1 row, fewer resident blocks.

batch-check bench, ctx 1024, 24 steps, `QUENCH6K_HY4_AB=fuse` (new old old new in one process), tok/s:

| sequences | plain 600 W | 3 drafts 600 W |
|---|---|---|
| 1 | 102.9 / 102.9 -> 112.8 / 114.6 | 148.7 / 149.8 -> 162.5 / 166.1 |
| 2 | 172.7 / 172.6 -> 191.2 / 194.5 | 212.7 / 213.2 -> 229.6 / 230.8 |
| 4 | 257.2 / 256.6 -> 281.6 / 282.3 | 304.7 / 302.3 -> 296.3 / 318.6 |
| 8 | 334.6 / 359.6 -> 361.2 / 379.2 | 362.9 / 372.0 -> 367.0 / 381.6 |

450 W, 8 sequences: plain 367.8 / 368.4 -> 382.0 / 392.7, 3 drafts 327.2 / 327.2 -> 327.0 / 329.1 (power bound:
the experts' DRAM traffic). The first arm of a case includes its graph captures.

## 2026-11-23: several conversations at once, drafts planned for the whole step (served)

Served workload (a scratch client; tools/loadgen.py's replies are short, so its levels are mostly
prefill): L requests started together, each its own ~1K-token prompt asking for a long explanation, 320
or 480 tokens each (all reach max_tokens), greedy, prompts 0..L-1 at every level (so prompt 0 alone and 8
at once are compared by text). `QUENCH6K_SLOT_STATS=1` makes the server print, per number of sequences
in a step: steps, rows per step, step and drafter times, drafts verified and accepted per sequence and
step, and tok/s. Rates below are those of steps with exactly that many sequences, 600 W.

Before (d731c05), where the time went:

- Hy4-flash, 8 at once: drafts on 323-345 tok/s, off 362-368. Steps of 9.0 rows: the cap forced one
  draft every 8th step on sequences capped at none, so with 8 sequences almost every step carried a lone
  draft, and 40-60% of steps a drafter pass of 2.6-3.1 ms for 1.6-2.6 sequences; 1.09 tokens per
  sequence and step. A 9-row step costs 24.1-24.4 ms against 21.7 for 8 rows. 2 at once: 172-180 (off
  171), 4: 235-250 (off 251).
- Inkling 2, 8 at once: on 643-659, off 666-669. 8.2-9.2 rows per step (0.05-0.15 drafts per sequence
  and step), but a drafter pass in 55-80% of steps for 1.7-1.8 sequences (1.0-1.1 ms), most of which
  kept no draft: the request's DraftPolicy truncated stated probabilities by its cost table, whose wider
  steps were last measured alone (5 ms and more) while its one-token steps were measured as their share
  of the batched step (about 1.2 ms).

Changes (every check bit-identical, every text identical):

- A batched step's drafts are planned for the whole step (serve/slots.rs StepCost::plan): drafts added
  in order of their chance (per draft position and request), the prefix with the most expected tokens per
  second kept, a drafter pass counted once (a fixed part plus a part per sequence), step time learnt per
  number of sequences and rows (recent plain step, recent excess per row count, a line where neither was
  seen; the first step of a shape, a CUDA graph capture, not learnt from). Stated probabilities
  (Inkling): a request's passes are planned by what they recently kept and had accepted, and a pass's
  drafts are kept while their calibrated chance is worth a row. Without drafts for 32 steps (then 64, ..
  1024) all sequences try one in one pass; prompt lookup (no pass) tries one every 8th step.
- Inkling's drafter computes only the drafts the requests can use (draft_batch_some; 8 sequences: 3.28 ms
  for all 7, 2.46 for three, 2.04 for one).
- Inkling and Hy4 sample from one top-k launch per 8 rows and one download per step (Flint's
  StepCandidates, now shared): Inkling drafts off, 8 at once 666-669 -> 675-679.

tok/s by sequences per step, drafts on (A B B A sessions; Hy4 one session each, Engram pages warmed):

| model | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Inkling 2, before | 395-402 | 585 | 596-615 | 643-659 |
| Inkling 2, after | 392-404 | - | 617-618 | 749-772 |
| Hy4-flash, before | 122 | 172 | 245 | 338 |
| Hy4-flash, after | 126 | 170 | 252 | 366 |
| Flint, before | 543-544 | 972-974 | 1241-1259 | 1586-1591 |
| Flint, after | 543-545 | 964-974 | 1259-1260 | 1588-1589 |

Drafts off (after): Inkling 246 / 371 / 516 / 675-679, Hy4 103 / 171 / 253 / 370, Flint 515 / 806 / 1183 /
1596. The requests' summed decode tok/s (client side, Inkling, 480 tokens) at 1 / 2 / 4 / 8 at once:
399-405 / 657-665 / 710-764 / 670-688 -> 395-407 / 726-744 / 715-724 / 879-897. 450 W, 8 at once
(loadgen sweep, 1 turn towards 800 tokens, 400 max, A B B A, two seeds each): Inkling generated tok/s
470, 449, 466, 458 -> 493, 467, 480, 456 (server 8-sequence steps 582-615 -> 622-672); Hy4 (the client
above, 320 tokens) 332 -> 346 tok/s in 8-sequence steps, 325 -> 336 summed.

Hy4's drafts at 2 and more at once still pay little: its drafter pass costs 2.3 ms alone and 3 ms for 8
sequences, a verified row about 2 ms (more for uneven widths), and its drafts are accepted at about 0.57
given the one before, so the plan drafts alone and rarely with others; at 8 it now matches plain steps
instead of losing 6-11%. Inkling's drafts are kept where likely (0.95 of them accepted at 8 at once),
0.5-0.7 per sequence and step.

Checks: scheduler unit tests (draft_plans: plans against learnt costs, a step shape that costs more,
stated-probability passes); Inkling batch-check with each step's drafts only, against the whole pass
alone (600 and 3000 tokens, 1 / 4 / 8 sequences, PDL on and off: bit-identical, digests unchanged);
inkling_lm_rows (micro test) with 3 of each block's rows: bitwise; served texts identical alone and 8 at
once, drafts on and off, greedy and sampled (temperature 0.8, seeded), for all three models; loadgen
`consistency --shared-system --between none` (4 conversations, 2 turns, greedy and sampled) 4/4 at
--slots 1 and 4 for Inkling 2, Hy4-flash and Flint.

## 2026-11-26: Nemotron-H decode, graphs and small-step forms (same bits)

Where a plain step goes (nsys, `QUENCH6K_NEM_PROFILE=decode` puts bench decode steps 9.. in the
capture range, tools/census.py, 31 steps at 8K, 600 W):

| | 1 sequence | 8 sequences |
|---|---|---|
| step (first start to last end) | 9.29 ms, busy 9.23, 966 kernels | 22.2 ms, busy 21.8 |
| GEMVs (305) | 4.69 ms, ~7.6 GB: at the DRAM rate | 7.42 ms: the same bytes, ALU-bound at 8 rows |
| routed experts (up + down, 41 layers) | 2.16 ms, 2.8 GB (1.29 TB/s) | 10.6 ms, at 84-91% of DRAM read |
| block norms (94) | 0.64 ms, 6.8 us each | 0.60 ms |
| router (41) | 0.31 ms, 7.5 us each | 0.30 ms |
| Mamba conv, commit, scan, gated norm (40) | 0.60 ms | 1.87 ms (scan 1.24) |
| attention decode + combine (9) | 0.42 ms | 0.65 ms |

Speed of light (tools/sol.py of the sm120-cuda skill, 1.65 TB/s): a plain step of one sequence reads
about 11.0 GB (dense projections 7.3 GiB, 22 of 512 experts per MoE layer 2.6 GiB, Mamba state read and
written 0.3 GiB, cache), 6.7 ms or 150 tok/s; 113.5 tok/s after the changes below is 76% of it. Eight
sequences read about 29 GB with independently routed tokens (about 150 distinct experts per layer),
17.8 ms or 450 tok/s; 377 tok/s is 82%. The GPU was idle 0.6% of a plain step, so graphs could only win
the drafted path's host turnarounds.

Changes (each a feature bit, `forward::set_features`; `QUENCH6K_NEM_AB=maskA,maskB` times masks
A B B A in one bench process, and the tokens of every arm must be equal; medians, 600 W):

| change | 1 seq 8K plain | 1 seq 8K 3 drafts | 8 seq 2K plain | 8 seq 2K 3 drafts |
|---|---|---|---|---|
| CUDA graphs (steps and MTP passes) | 109.1 -> 110.5 | 242.4 -> 246.7 | 355.7 -> 358.3 | 439.0 -> 437.6 (captures in every run) |
| Mamba conv + commit in one launch | 110.5 -> 111.5 | 246.7 -> 248.9 | 358.3 -> 364.2 | |
| wide GEMV (INT8G added), 2-64 rows | | | 358.3 -> 367.5 | 437.6 -> 487.1 |
| attention combine per query head | 110.0 -> 110.9 (32K: 105.6 -> 109.5) | 252.3 -> 254.3 | | |
| router by sorting | 110.9 -> 113.5 | 254.3 -> 258.0 | 373.4 -> 377.0 | 491.4 -> 494.9 |

Dropped on measurement: the residual add in the next block norm's launch (fewer launches): first -1.4%
(its stores came before its loads), then +0.4% / +0.2%, within noise; the routed experts' next item's
weights loaded during an item (up kernel 0%, down -0.8% at 1 sequence, -1.1% at 8).

Every new form is checked bit for bit against the launches it replaces (crates/cuda/tests/nemotron.rs
decode_forms_equal); batch-check, verify-check and spec-check passed with graphs (4 sequences).
Not done in this round: a null arm and clock / temperature logs next to these numbers.

## 2026-11-26: Nemotron-H decode, work beside the router and the scan (same bits)

Census of main's decode (nsys, tools/census.py, 600 W): 1 sequence at 8K, 9.00 ms per traced step,
DRAM read 72% of peak on average and under 20% for 9% of the step. GEMVs 4.62 ms (at the DRAM rate),
routed experts 2.14 ms (up 31.3 + down 20.9 us per layer), block norms 0.63 ms (6.7 us each, DRAM 36%),
attention 0.31 ms (35 us per layer at 8K, DRAM 17%), scan, combine, gated norm, relu², conv, router
0.10-0.26 ms each. 8 sequences at 32K: 22.0 ms; wide GEMVs 5.93 ms (DRAM 63%), experts 10.5 ms (~87%),
attention 1.70 ms (189 us per layer, 1.42 TB/s, 83%), scan 1.06 ms, router-side small GEMVs 0.97 ms.

The levers proposed last round, checked first:
- Attention at 8 x 32K was not 2.96 ms per layer in the engine but 189 us at 83% of DRAM peak (at most
  ~1% of the step left). An L2 prefetch of key tiles helped where the grid is smaller than the GPU and
  cost 0.57% at 8 x 32K, so it runs only on small grids.
- The up / down expert kernels at 1 sequence (ncu on the micro test, 2.81 GHz): 24.9-26.0 us and
  26.0-26.4 us for 34 MB each, 77-80% of DRAM peak: equal; the census' 31 vs 21 us split is PDL credit
  (down loads its first weights before its wait). Folding the rounding and grouping launches into the
  expert kernels: micro 43.0 -> 41.6 us at 1 row, engine +0.17% plain, -0.30% drafted and at 8
  sequences: dropped.
- GEMV forms per shape: +1.75% at 8 rows; below 5 rows the isolated winners lost in the engine (fewer
  weights loaded before the dependency wait), so the table starts at 5 rows.
- The small latency-bound kernels: instead of fusing them, independent work now runs beside them on a
  side stream (fork / join events, also inside the CUDA graphs).

Changes (feature bits, `QUENCH6K_NEM_AB` in-process A/B, 3 rounds A B .. B A after a warm-up run of
each arm, one prefill restored before every run, a null arm in every A/B; median ratio and range;
600 W, SM clock median 2.6-2.8 GHz, 61-82 °C, throttle reasons power cap only):

| change | 1 seq 8K plain | 1 seq 8K 3 drafts | 8 seq 2K plain | 8 seq 2K 3 drafts |
|---|---|---|---|---|
| null arm (typical band) | 0.9999-1.0005 | 0.9988-1.0017 | 0.9995-1.0004 | 0.9980-1.0001 |
| norm weights before the wait (F_RMS) | +0.59% | +0.36% | +0.19% | |
| attention L2 prefetch, small grids (F_ATTN_PF) | +0.56% | +0.12% | +0.22% | |
| shared expert up + relu² beside the router (F_SIDE) | +1.97% | +0.12% | +0.34% | |
| fc1 beside the router too (F_SIDE_FC1) | +4.57% | +2.71% | | |
| in_proj's z rows beside conv + scan (F_SIDE_Z) | +1.66% | +0.68% | +2.11% | +0.17% |
| GEMV forms per shape, 5+ rows (F_FORMS) | 0 | 0 | +1.75% | +0.25% |
| all against main's path (0x197 vs 0x7f97) | +7.35% (1.0721-1.0756) | +4.11% (1.0313-1.0415) | +6.08% (1.0607-1.0621) | +2.14% (1.0201-1.0218) |

Rates in that last A/B: 119.9 -> 128.7 tok/s, 267.7 -> 278.7, 387.0 -> 410.5, 501.9 -> 512.3. Runs
that decode after a restore read higher than the plain bench's (fresh prefill before every run: 113.3
tok/s for main's 1 seq 8K plain against 120.0 here): compare ratios, not these rates with older tables.

Checks: decode_forms_equal (also prefetch distances 0, 1, 3, 5), digests unchanged, batch-check 8 x 12,
verify 7, spec-check 8 sequences with chat, fixture scores of all 7 fixtures byte-identical to main's
build (the .json files differ only in their wall-clock seconds).
