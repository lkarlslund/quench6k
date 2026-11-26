# Tools

Development tooling. None of it is part of the `quench6k` binary, which is Rust host code plus CUDA
kernels only.

- `ref/ref.py`: an independent PyTorch implementation of [dev/SPEC.md](../dev/SPEC.md), used as a dev-only
  oracle. It scores a fixture sequence teacher-forced, streaming one layer of weights at a time,
  and can dump per-op tensors of chosen layers as golden data for kernel tests:

  ```
  python3 -I tools/ref/ref.py --model <checkpoint> --fixture oracle/fixtures/code --out out/ref/code-a16
  quench6k gate --candidates out/ref --suffix=-a16
  ```

  `--act-quant` emulates NVFP4 W4A4 expert activations; `--fp64` computes in FP64.
- `census.py`: per-kernel critical-path time of a decode or MTP round from an Nsight Systems trace, with
  the DRAM bandwidth sampled during each kernel; `--record` takes the trace too (dev/KBENCH.md "Census").
- `ab.py`: interleaved A/B runs of a bench command, medians and ranges per variant (dev/KBENCH.md "A/B").
- `loadgen.py`: synthetic multi-conversation load for an OpenAI-compatible server (Python standard
  library only; dev/KBENCH.md "Serving load"). Each conversation is agent-like: turns of a task plus a
  block of filler (the repository's tracked source and docs, or `--filler synthetic` generated text),
  the reply appended before the next turn, so the prompt grows towards `--target` tokens at the last
  turn. Prompts follow from `--seed` and the conversation's number; requests stream greedily
  (temperature 0, thinking off) with `--max-tokens` per turn. Per request it reports the wait for a slot
  (the first `return_progress` chunk), prompt tokens and how many were reused (`cached_tokens`), prefill
  tok/s of the rest, time to first token, decode tok/s and the total; 503s (`model_switch_blocked` from
  serve-models) are retried after their Retry-After, other errors (`context_length_exceeded`) end the
  conversation. Servers that send progress only after prefill chunks (the one-sequence worker loops
  today) get the wait extrapolated back from the chunks' rate (shown `~`), or with a single chunk none
  (the wait is then part of the prefill), and a note says how many; decode tok/s needs at least 16
  generated tokens (speculative decoding streams in bursts). Three modes:

  ```
  python3 tools/loadgen.py --url http://127.0.0.1:8000 --turns 3 --target 8192 sweep --levels 1,2,4,8
  python3 tools/loadgen.py --url http://127.0.0.1:8000 --turns 3 --target 8192 consistency --conversations 8
  python3 tools/loadgen.py --url http://127.0.0.1:8000 --turns 8 --target 262144 --max-tokens 32 fill --conversations 4
  ```

  `sweep` runs the same workload shape at each concurrency (fresh prompts per level, so one level's
  cache does not serve the next) and prints a table of aggregate generated and prefill tok/s, median
  decode tok/s per request, generated tok/s per conversation over its whole life, wait and TTFT
  percentiles, and the reused share of prompt tokens. `consistency` runs each conversation alone, one
  after another, then all at once, and compares the texts turn by turn: the first differing turn and
  character with both versions around it, exit status 1 on any difference (batch invariance end to
  end). Each conversation has a system prompt of its own and the model is unloaded between the
  phases (`--shared-system`, `--between none` turn that off), so both phases start from an empty cache
  and each turn reuses the same cached prefix: give the server at least as many slots as
  conversations; turns served from cache differently in the two phases are counted. `--order
  together-first` swaps the phases. `fill` grows `--conversations` concurrently to `--target` tokens each, a
  row per request (prompt, reused, new, wait, prefill, decode), then one short revisit turn per
  conversation, one at a time, showing how much of each the server kept. `--json FILE` appends JSON
  lines (a header with the corpus digest, one record per request, the summaries); `-v` prints a row
  per request in every mode; `--extra '{"mtp": false}'` merges fields into every request.
  Tests (a fake server, no GPU): `python3 -m unittest tools/test_loadgen.py`.
- `ref/hy4_ref.py`, `ref/hy4_fixtures.py`, `ref/hy4_compare.py`: the Hy4-flash reference (an independent
  PyTorch implementation after transformers' `hy_v4_flash`, streaming one layer of weights at a time and
  scoring several fixtures per pass), the Hy4-flash versions of the oracle fixtures, and a per-layer
  comparison of hidden states with the engine's `QUENCH6K_HY4_DUMP` output (see DOCS.md, Hy4-flash).
- `ref/nemotron_ref.py`, `ref/nemotron_fixtures.py`: the Nemotron-H reference (Nemotron 3 Super, Munin
  2.0): an independent PyTorch implementation after the checkpoint's `modeling_nemotron_h.py`, on the CPU
  by default (`--device cuda` later), streaming one layer at a time and dequantizing the checkpoint's own
  NVFP4 / FP8 / BF16 weights (only the experts the tokens route to). Mamba-2 runs as the plain sequential
  recurrence with an FP32 state (`--scan fp64` for the floor); `--dtype bf16` rounds where the model does
  under its CUDA kernels, `--dtype fp32` gives the noise floor; `--fp32 residual` keeps the residual stream
  in FP32, `--kv-fp8` emulates an FP8 KV cache, `--torch-dt-clamp` the modeling file's torch_forward dt
  clamp; `--dump-layers` writes the residual stream after chosen layers (`seq<j>/lanes-NN.npy`), and
  `--prompt TEXT` prints the next-token candidates. The fixtures are the oracle conversations (and a
  Danish one) in the checkpoint's chat template and tokenizer, written to out/ (see DEVELOPMENT.md).
