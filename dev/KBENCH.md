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
