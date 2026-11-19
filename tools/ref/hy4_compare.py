#!/usr/bin/env python3
"""Per-layer comparison of Hy4-flash hidden lanes: tools/ref/hy4_ref.py --dump-dir output (lanes-NN.npy,
F32 [T, 4, 3072]) against the engine's QUENCH6K_HY4_DUMP output (lanes-NN.bin, BF16 [T, 4, 3072]).

    python3 -I tools/ref/hy4_compare.py out/hy4-ref-dump out/hy4-engine-dump
"""
import pathlib
import sys

import numpy as np


def bf16(path):
    raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32)


def main():
    ref, eng = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    for r in sorted(ref.glob("lanes-*.npy")):
        e = eng / (r.stem + ".bin")
        if not e.exists():
            continue
        a = np.load(r).astype(np.float64)
        b = bf16(e).astype(np.float64)[: a.size].reshape(a.shape)
        diff = a - b
        rel = np.sqrt((diff ** 2).sum() / (a ** 2).sum())
        per_tok = np.sqrt((diff ** 2).sum(axis=(1, 2)) / (a ** 2).sum(axis=(1, 2)))
        worst = int(per_tok.argmax())
        print(f"{r.stem}: rel {rel:.2e}  worst token {worst} rel {per_tok[worst]:.2e}  max|d| {np.abs(diff).max():.3g}")


if __name__ == "__main__":
    main()
