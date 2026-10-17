# Tools

Development tooling. None of it is part of the `quench6k` binary, which is Rust host code plus CUDA
kernels only.

- `ref/ref.py`: an independent PyTorch implementation of [SPEC.md](../SPEC.md), used as a dev-only
  oracle. It scores a fixture sequence teacher-forced, streaming one layer of weights at a time,
  and can dump per-op tensors of chosen layers as golden data for kernel tests:

  ```
  python3 -I tools/ref/ref.py --model <checkpoint> --fixture oracle/fixtures/code --out out/ref/code-a16
  quench6k gate --candidates out/ref --suffix=-a16
  ```

  `--act-quant` emulates NVFP4 W4A4 expert activations; `--fp64` computes in FP64.
