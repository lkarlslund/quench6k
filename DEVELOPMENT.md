# Development

## Layout

| Path | |
|---|---|
| `crates/cuda` | CUDA kernels (`kernels/*.cu`, compiled by `build.rs` with nvcc for `sm_120a`) and their Rust launchers |
| `crates/engine` | Checkpoint loading, forward pass, CUDA graphs, speculative decoding, sessions, context store, vision |
| `crates/cli` | The `quench6k` binary: server, chat template and tokenizer, benchmarks, checks |
| `crates/benchplot` | Renders `bench/*.jsonl` as `bench/results.svg` and the tables in BENCHMARK.md |
| `oracle/` | Reference distributions for the correctness gate |
| `tools/` | Python reference implementations (development only, not part of the binary) |
| `ci/`, `.github/` | Release builds |
| `dev/` | Design documents and history (below) |

## Build and test

```
cargo build --release
cargo test --release
```

Kernel tests (`cargo test -p quench6k-cuda --test kernels`) compare every kernel with a CPU
implementation and need the GPU's memory: stop a running server first.

## Correctness

The engine runs a quantized model, so the bar is statistical: on every oracle fixture, quench6k may
not deviate from the nearer of two llama.cpp references (NVFP4 W4A4, and the same weights with 8-bit
activations) by more than twice those references deviate from each other ([dev/GOAL.md](dev/GOAL.md) §1).

```
M=/path/to/Swift-1.5-Qwen3.8-Flash-Next-NVFP4
quench6k score --model $M --fixture oracle/fixtures/* --out out/run --decode
quench6k gate --candidates out/run
quench6k generate --model $M --fixture oracle/fixtures/*
quench6k spec-check --model $M --fixture oracle/fixtures/code --n 192 --graphs
quench6k batch-check --model $M --fixture oracle/fixtures/code oracle/fixtures/multiturn
quench6k vision-check --model $M --reference out/vision/shapes
```

Changes that are meant to be exact must leave the `score` outputs byte-identical; changes to
numerics must pass `gate`. `spec-check` requires speculative greedy decoding to match plain greedy
decoding token for token; `batch-check` requires the same of batched plain steps and of speculative
pair rounds (two sequences).

## Kernel iteration

`quench6k kbench` loads the first few layers only, replays decode steps from a snapshot, checks
candidates bit for bit against the baseline and times them in alternating pairs. Kernel variants are
selected at run time (`--variant key=value`, `--sweep key=v1,v2`). See [dev/KBENCH.md](dev/KBENCH.md).

## Benchmarks

```
quench6k bench --model $M --ctx 512 1024 8192 65536 131072 260000 --json bench/results.jsonl
quench6k bench --model $M --ctx ... --mtp 6 --adaptive --json bench/results.jsonl
quench6k bench --model $M --ctx ... --mtp 6 --adaptive --sampled --json bench/results.jsonl
cargo run --release -p quench6k-benchplot -- bench/ninfer-6000.jsonl bench/results.jsonl \
    --svg bench/results.svg --markdown BENCHMARK.md
```

`bench/README.md` describes the corpus and the record format. Before timing, `bench` checks 64 greedy
tokens of the benchmarked path against the code fixture's reference sequence, or, for a checkpoint
other than the oracle's, against eager decoding of the same checkpoint.

## Recursive Flint

`crates/engine/src/flint.rs` (model and forward pass), `crates/engine/src/gguf.rs` (GGUF reader) and
`crates/cuda/kernels/flint.cu`. The reference is llama.cpp's Flint implementation with the same GGUF:

```
F=/path/to/flint-nvfp4-gguf
quench6k flint generate --model $F --raw --prompt 'def fibonacci(n):' --n 48   # vs llama-completion --temp 0
quench6k flint ppl --model $F --file text.txt --ctx 2048 --chunks 6           # vs llama-perplexity -c 2048
quench6k [--kv fp8] flint bench --model $F --text corpus.txt --ctx 512 8192 65536 131072 260000
```

`ppl` scores like llama-perplexity (second half of each chunk), so the numbers compare directly; `bench`
checks graph against eager decoding and reports how much of the corpus continuation greedy decoding
reproduces.

## Releases

`.github/workflows/release.yml` runs daily and on demand. When `main` has commits that are not
released yet, it builds Linux x86_64, Linux arm64 and Windows x86_64 with CUDA from NVIDIA's
redistributable archives (`ci/fetch-cuda.sh`) and publishes release `rNNNN`, NNNN being the number of
commits on `main`. The CUDA runtime is linked statically.

## Experiment switches

| Variable | |
|---|---|
| `QUENCH6K_VARIANTS=key=v,...` | Kernel variants (as kbench's `--variant`) |
| `QUENCH6K_FAKE_BITS=b`, `QUENCH6K_FAKE_FP4_W=classes`, `QUENCH6K_FAKE_FP4`, `QUENCH6K_FAKE_MXFP8[_SET]` | Fake-quantize weights to test lower precisions with `score` + `gate` |
| `QUENCH6K_PREFILL_FP4`, `QUENCH6K_GDN_MXFP8`, `QUENCH6K_GDN_TC`, `QUENCH6K_DEQUANT_GEMM` | Turn off (0) individual prefill precision or kernel choices |
| `QUENCH6K_HC_UNFUSED`, `QUENCH6K_YD_BF16`, `QUENCH6K_SCORE_FP32` | Alternative code paths for comparisons |
| `QUENCH6K_CHECK_ENCODE`, `QUENCH6K_CHECK_ROUTE` | Cross-check cached tokenization and MoE routing |
| `QUENCH6K_TEST_TOOL_RETRY` | Treats the first tool call as invalid (exercises regeneration) |
| `QUENCH6K_BENCH_IDS=FILE` | Benchmark corpus other than `bench/context.ids` |
| `QUENCH6K_DUP=op` | Launches one op (its profiler name) twice: its marginal in-graph cost (kbench; results are wrong) |
| `QUENCH6K_KBENCH_PROFILE=N` | kbench: N eager steps timed per op before benchmarking |
| `QUENCH6K_DFLASH_BF16=1` | DFlash drafter projections in BF16 instead of INT8 |
| `QUENCH6K_DFLASH_PROFILE=N` | Times round N's DFlash drafting per op (eagerly, 10 times) |
| `QUENCH6K_POLICY=decay=..,prior=..,fixed=N` | Adaptive draft length policy settings; `fixed=N` always drafts N |
| `QUENCH6K_FLINT_SYNC=1` or `=op` | Flint: synchronize after every op (or one named op) and report the failing one |
| Variants `flint.pdl` (bitmask), `flint.fuse_norm`, `flint.attn_dec`, `flint.attn_fused_merge`, `flint.attn_min_tiles`, `flint.attn_wide`, `flint.splits`, `flint.gemv`, `flint.gemv_r`, `flint.moe_v1` | Flint kernel choices (see dev/KBENCH.md) |
| Variant `flint.prefill_fp4=1` | Flint prefill experts in W4A4 (FP4 activations): +13-21% prefill, PPL 15.61 vs 14.95 at 16K, so off |
| Variant `flint.lookup=0` | Flint server: no prompt-lookup speculation |

## History

- [dev/GOAL.md](dev/GOAL.md): the project goal, correctness gate and the NInfer 6000 baseline.
- [dev/ROADMAP.md](dev/ROADMAP.md): milestones and the dated progress log.
- [dev/SPEC.md](dev/SPEC.md): the model specification written before the implementation.
- [dev/KBENCH.md](dev/KBENCH.md): the kernel benchmark design and the findings of each round of
  decode optimization, including rejected experiments.
- [dev/RESULTS-2026-10-16.md](dev/RESULTS-2026-10-16.md): an earlier benchmark with notes on the
  prefill work.
