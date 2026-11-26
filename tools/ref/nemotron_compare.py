#!/usr/bin/env python3
"""Per-layer comparison of Nemotron-H hidden states: tools/ref/nemotron_ref.py --dump-dir output of one
sequence (lanes-NN.npy, F32 [T, 4096], after layer NN's residual add; final-<fixture>.npy after norm_f)
against the engine's dump (`quench6k nemotron score|generate --dump-dir`: lanes-NN.bin and final.bin,
raw F32 [T, 4096]).

    python3 -I tools/ref/nemotron_compare.py out/nem-ref-dump/seq0 out/nem-engine-dump/<fixture> \
        [--final out/nem-ref-dump/final-<fixture>.npy]

Prints, per layer, the relative L2 difference over all tokens, the worst token's, and the largest absolute
difference; the first layer whose difference jumps is where to look.
"""
import argparse
import pathlib

import numpy as np


def compare(name, a, raw):
    a = a.astype(np.float64)
    if raw.size < a.size:
        print(f"{name}: engine dump has {raw.size // a.shape[-1]} rows, reference {a.shape[0]}")
        return
    b = raw.astype(np.float64)[: a.size].reshape(a.shape)
    diff = a - b
    rel = np.sqrt((diff ** 2).sum() / max((a ** 2).sum(), 1e-30))
    per_tok = np.sqrt((diff ** 2).sum(axis=-1) / np.maximum((a ** 2).sum(axis=-1), 1e-30))
    worst = int(per_tok.argmax())
    print(f"{name}: rel {rel:.2e}  worst token {worst} rel {per_tok[worst]:.2e}  max|d| {np.abs(diff).max():.3g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", help="reference dump directory of one sequence (lanes-NN.npy)")
    ap.add_argument("engine", help="engine dump directory (lanes-NN.bin)")
    ap.add_argument("--final", help="the reference's final-<fixture>.npy (after norm_f)")
    args = ap.parse_args()
    ref, eng = pathlib.Path(args.ref), pathlib.Path(args.engine)
    for r in sorted(ref.glob("lanes-*.npy")):
        e = eng / (r.stem + ".bin")
        if e.exists():
            compare(r.stem, np.load(r), np.fromfile(e, dtype=np.float32))
    if args.final and (eng / "final.bin").exists():
        compare("final", np.load(args.final), np.fromfile(eng / "final.bin", dtype=np.float32))


if __name__ == "__main__":
    main()
