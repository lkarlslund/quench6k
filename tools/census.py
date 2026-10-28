"""Kernel census of a decode or MTP round from an Nsight Systems trace (dev/KBENCH.md "Census").

    nsys profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi -o run \\
        quench6k --kv fp8 bench --model $M --ctx 65536 --mtp 3 --generate 128 --ncu-steps 20
    python3 tools/census.py run.nsys-rep --rounds 20 [--shape]

With programmatic dependent launch a kernel starts before its predecessor ends and waits, so raw
durations overlap (their sum is ~2x the busy time). Each kernel is credited with its critical-path
contribution instead: its end minus the latest end of any kernel before it, over all kernels sorted by
end time, so the credits add up to the busy time. Weight loads a kernel issues before its dependency
wait are hidden behind the predecessor and credited to it.
"""

import argparse
import collections
import csv
import pathlib
import re
import subprocess

NSYS = "/usr/lib/nsight-compute/host/target-linux-x64/nsys"


def trace_rows(path):
    path = pathlib.Path(path)
    if path.suffix == ".nsys-rep":
        out = path.with_name(path.stem)
        subprocess.run([NSYS, "stats", "-q", "-r", "cuda_gpu_trace", "-f", "csv", "-o", str(out), "--force-export=true", str(path)],
                       check=True, stdout=subprocess.DEVNULL)
        path = out.with_name(out.name + "_cuda_gpu_trace.csv")
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["Name"].startswith("[CUDA"):
                continue
            start = int(r["Start (ns)"])
            yield start, start + int(r["Duration (ns)"]), r


def kernel_name(name):
    name = name.replace("<unnamed>::", "").removeprefix("void ")
    return re.sub(r"[<(].*$", "", name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace", help=".nsys-rep, or the CSV of `nsys stats -r cuda_gpu_trace -f csv`")
    ap.add_argument("--rounds", type=int, required=True, help="rounds (or decode steps) in the capture range")
    ap.add_argument("--shape", action="store_true", help="one line per kernel and launch shape (grid, stream)")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    rows = sorted(trace_rows(args.trace), key=lambda x: x[1])
    crit, raw, calls = collections.Counter(), collections.Counter(), collections.Counter()
    last = min(s for s, _, _ in rows)
    first, idle = last, 0
    for s, e, r in rows:
        key = kernel_name(r["Name"])
        if args.shape:
            key = f"{key} {r['GrdX']}x{r['GrdY']} s{r['Strm']}"
        idle += max(0, s - last)
        crit[key] += max(0, e - max(last, s))
        raw[key] += e - s
        calls[key] += 1
        last = max(last, e)
    n, busy = args.rounds, sum(crit.values())
    print(f"per round: {(last - first) / 1e3 / n:.1f} us from first start to last end, {busy / 1e3 / n:.1f} us busy, "
          f"{idle / 1e3 / n:.1f} us idle, {len(rows) / n:.0f} kernels")
    print(f"{'kernel':<56} {'calls':>6} {'crit us':>8} {'share':>6} {'crit/call':>9} {'raw/call':>9}")
    for key, c in crit.most_common(args.top):
        print(f"{key[:56]:<56} {calls[key] / n:>6.0f} {c / 1e3 / n:>8.1f} {100 * c / busy:>5.1f}% {c / 1e3 / calls[key]:>9.2f} "
              f"{raw[key] / 1e3 / calls[key]:>9.2f}")


if __name__ == "__main__":
    main()
