# Fast kernel iteration: `squash kbench`

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
small variant table set from Rust (`squash_cuda::variants::set("gemv.warps", 4)`), at launch time. All
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
2. kernel unit tests (`cargo test -p squash-cuda --test kernels`)
3. engine: `squash generate` on all fixtures, `squash gate`, `squash bench` greedy check (finalists)

### 5. Timing that resolves small differences

- Baseline and candidate graphs alternate in pairs (A B B A ...), each pair from the same restored
  snapshot, so drift in clocks and temperature cancels.
- The result is the median ratio candidate / baseline over the pairs, with a spread (25-75%), not a
  single absolute number.
- The working set is the real one (weights for K layers, about 0.5 GiB per decode step), larger than
  the 128 MB L2, so weights stream from DRAM as in the full model.

### 6. Sweeps

`squash kbench --sweep gemv.warps=4,8 --sweep gemv.rows=1,2` runs the cartesian product, prints one
line per configuration (median ratio, spread, bitwise yes/no) and the best exact configuration.
Variant kernels are templates over these keys; new fused kernels are added as further values of a key
(for example `hc.mode=split|fused`), so a hand-written fusion is swept like any parameter.

## Limits

- K layers time the per-layer kernels; the LM head and the MTP drafter are outside the truncated model
  and are measured by the full engine.
- The final numbers (tok/s) still come from `squash bench` on the full model; kbench ranks candidates
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
