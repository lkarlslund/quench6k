"""Interleaved A/B timing of a command that prints throughput (dev/KBENCH.md "A/B").

    python3 tools/ab.py --runs 3 -v on: -v off:QUENCH6K_INK_ATTN_FUSED=0 -- \\
        quench6k inkling bench --model $M --ctx 512 8192 --generate 128

Variants are NAME:[VAR=VALUE[,VAR=VALUE...]] (environment added to the command's), and --bin NAME=PATH
runs another binary for that variant (e.g. a build of the previous commit). Each round runs the variants
in order and then in reverse (A B B A), so clock and temperature drift over a round cancels; --runs
rounds. Every "<name> <number> tok/s" of a line is a metric, labelled with the line's text before its
first colon ("ctx 512: prefill 6625 tok/s, decode 228.0 tok/s" gives "ctx 512 prefill" and
"ctx 512 decode"); --metric REGEX (named groups `value` and optionally `name`) takes other numbers.
Prints each metric's median and range per variant and the median's ratio to the first variant.
"""

import argparse
import collections
import os
import re
import statistics
import subprocess
import sys

TOKS = re.compile(r"(?P<name>[A-Za-z_][\w-]*)\s+(?P<value>\d+(?:\.\d+)?)\s*tok/s")


def metrics(text, pattern):
    out = {}
    for line in text.splitlines():
        label = " ".join(line.split(":", 1)[0].split()) if ":" in line else ""
        for m in pattern.finditer(line):
            name = m.groupdict().get("name") or "value"
            key = f"{label} {name}".strip()
            while key in out:  # repeated labels keep their order
                key += "'"
            out[key] = float(m.group("value"))
    return out


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: ab.py [options] -v NAME:[ENV=V,...] -v ... -- command ...")
    cut = argv.index("--")
    argv, cmd = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-v", "--variant", action="append", required=True, help="NAME:[VAR=VALUE[,VAR=VALUE...]]")
    ap.add_argument("--bin", action="append", default=[], help="NAME=PATH: the command's first word for that variant")
    ap.add_argument("--runs", type=int, default=2, help="rounds of A B ... B A")
    ap.add_argument("--metric", help="regex with a named group `value` (and optionally `name`)")
    ap.add_argument("--quiet", action="store_true", help="no line per run")
    args = ap.parse_args(argv)
    if not cmd:
        ap.error("no command after `--`")
    pattern = re.compile(args.metric) if args.metric else TOKS
    bins = dict(b.split("=", 1) for b in args.bin)
    variants = []
    for v in args.variant:
        name, _, env = v.partition(":")
        # VAR=VALUE pairs split at commas that start a new VAR= (values may hold commas).
        extra = dict(kv.split("=", 1) for kv in re.split(r",(?=[A-Za-z_][A-Za-z0-9_]*=)", env) if kv)
        variants.append((name, extra))
    names = [n for n, _ in variants]
    if len(set(names)) != len(names):
        ap.error("variant names must differ")

    results = collections.defaultdict(lambda: collections.defaultdict(list))
    order = []
    for r in range(args.runs):
        for name, extra in variants + variants[::-1]:
            run = [bins.get(name, cmd[0]), *cmd[1:]]
            p = subprocess.run(run, env={**os.environ, **extra}, capture_output=True, text=True)
            got = metrics(p.stdout + p.stderr, pattern)
            if not got:
                sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
                sys.exit(f"{name}: no metric in the output (exit status {p.returncode})")
            for k, x in got.items():
                if k not in order:
                    order.append(k)
                results[k][name].append(x)
            if not args.quiet:
                print(f"round {r + 1} {name}: " + ", ".join(f"{k} {x:g}" for k, x in got.items()), flush=True)

    base = names[0]
    width = max(len(k) for k in order)
    print(f"\n{'metric':<{width}}  " + "  ".join(f"{n:>24}" for n in names))
    for k in order:
        cells = []
        b = statistics.median(results[k][base]) if results[k][base] else None
        for n in names:
            xs = results[k][n]
            if not xs:
                cells.append(f"{'-':>24}")
                continue
            med = statistics.median(xs)
            ratio = f" {100 * (med / b - 1):+.1f}%" if b and n != base else ""
            cells.append(f"{f'{med:g} [{min(xs):g}-{max(xs):g}]{ratio}':>24}")
        print(f"{k:<{width}}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
