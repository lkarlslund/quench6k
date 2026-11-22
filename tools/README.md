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
- `ref/hy4_ref.py`, `ref/hy4_fixtures.py`, `ref/hy4_compare.py`: the Hy4-flash reference (an independent
  PyTorch implementation after transformers' `hy_v4_flash`, streaming one layer of weights at a time and
  scoring several fixtures per pass), the Hy4-flash versions of the oracle fixtures, and a per-layer
  comparison of hidden states with the engine's `QUENCH6K_HY4_DUMP` output (see DOCS.md, Hy4-flash).
