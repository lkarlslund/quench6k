"""Kernel census of decode or MTP rounds from an Nsight Systems trace (dev/KBENCH.md "Census").

Record a trace and report it (node-level graph tracing plus GPU metrics sampling):

    python3 tools/census.py --record run --rounds 20 -- \\
        quench6k --kv fp8 bench --model $M --ctx 65536 --mtp 3 --generate 128 --ncu-steps 20

Report an existing trace (.nsys-rep, its .sqlite export, or the CSV of
`nsys stats -r cuda_gpu_trace -f csv`):

    python3 tools/census.py run.nsys-rep --rounds 20 [--shape] [--side REGEX]

With programmatic dependent launch a kernel starts before its predecessor ends and waits, so raw
durations overlap (their sum is ~2x the busy time). Each kernel is credited with its critical-path
contribution instead: its end minus the latest end of any kernel before it, over all kernels sorted by
end time, so the credits add up to the busy time. Weight loads a kernel issues before its dependency
wait are hidden behind the predecessor and credited to it. Kernels matching --side (e.g. L2 prefetches
on a side stream) stay out of the chain.

When the trace has GPU metrics (--record, or `nsys profile --gpu-metrics-devices=...`), each kernel's
credited time also gets the DRAM read bandwidth sampled during it (percent of peak), and a summary gives
its distribution over the busy time: how much of the chain leaves memory idle. Node-level tracing adds a
few percent per round; the per-round time without it is the bench's own number.
"""

import argparse
import bisect
import collections
import csv
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import sys

NSYS = os.environ.get("NSYS") or shutil.which("nsys") or "/usr/lib/nsight-compute/host/target-linux-x64/nsys"


def kernel_name(name):
    name = name.replace("<unnamed>::", "").removeprefix("void ")
    return re.sub(r"[<(].*$", "", name)


def export_sqlite(rep):
    db = rep.with_suffix(".sqlite")
    if not db.exists() or db.stat().st_mtime < rep.stat().st_mtime:
        subprocess.run([NSYS, "export", "-t", "sqlite", "-o", str(db), "--force-overwrite", "true", str(rep)], check=True,
                       stdout=subprocess.DEVNULL)
    return db


def csv_rows(path):
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["Name"].startswith("[CUDA"):
                continue
            start = int(r["Start (ns)"])
            yield start, start + int(r["Duration (ns)"]), kernel_name(r["Name"]), f"{r['GrdX']}x{r['GrdY']} s{r['Strm']}"


def sqlite_rows(db):
    q = ("select k.start, k.end, s.value, k.gridX, k.gridY, k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k "
         "join StringIds s on s.id = k.demangledName")
    for start, end, name, gx, gy, stream in db.execute(q):
        yield start, end, kernel_name(name), f"{gx}x{gy} s{stream}"


def dram_samples(db):
    """DRAM read bandwidth samples (timestamps, percent of peak), or None without GPU metrics."""
    try:
        ids = [i for (i,) in db.execute("select metricId from TARGET_INFO_GPU_METRICS where metricName like 'DRAM Read Bandwidth%'")]
    except sqlite3.OperationalError:
        return None
    if not ids:
        return None
    rows = db.execute(f"select timestamp, value from GPU_METRICS where metricId = {ids[0]} order by timestamp").fetchall()
    return ([t for t, _ in rows], [v for _, v in rows]) if rows else None


class Sampled:
    """Time-weighted mean of a sampled counter over an interval; sample k covers (t[k-1], t[k]]."""

    def __init__(self, samples):
        self.t, self.v = samples

    def mean(self, a, b):
        t, v = self.t, self.v
        k = bisect.bisect_left(t, a)
        acc, lo = 0.0, a
        while k < len(t) and lo < b:
            hi = min(b, t[k])
            acc += v[k] * (hi - lo)
            lo = hi
            k += 1
        return acc / (lo - a) if lo > a else None


def record(args, cmd):
    rep = pathlib.Path(args.record).with_suffix(".nsys-rep")
    rep.unlink(missing_ok=True)
    # nsys ends the application when the capture range stops (exit status 143): the report decides.
    subprocess.run([NSYS, "profile", "-t", "cuda", "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
                    "--gpu-metrics-devices=cuda-visible", f"--gpu-metrics-frequency={args.frequency}", "-o", str(rep.with_suffix("")),
                    "--force-overwrite", "true", *cmd])
    if not rep.exists():
        sys.exit(f"no report: {rep}")
    return rep


def main():
    argv = sys.argv[1:]
    cmd = []
    if "--" in argv:
        cut = argv.index("--")
        argv, cmd = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace", nargs="?", help=".nsys-rep, its .sqlite export, or the CSV of `nsys stats -r cuda_gpu_trace -f csv`")
    ap.add_argument("--rounds", type=int, required=True, help="rounds (or decode steps) in the capture range")
    ap.add_argument("--record", metavar="NAME", help="profile the command after `--` into NAME.nsys-rep first")
    ap.add_argument("--frequency", type=int, default=200000, help="GPU metrics samples per second with --record")
    ap.add_argument("--shape", action="store_true", help="one line per kernel and launch shape (grid, stream)")
    ap.add_argument("--side", metavar="REGEX", help="kernels kept out of the critical path (side-stream work)")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args(argv)
    if args.record:
        if not cmd:
            ap.error("--record needs the command after `--`")
        path = record(args, cmd)
    elif args.trace:
        path = pathlib.Path(args.trace)
    else:
        ap.error("a trace or --record")

    dram = None
    if path.suffix == ".csv":
        rows = list(csv_rows(path))
    else:
        db = sqlite3.connect(export_sqlite(path) if path.suffix == ".nsys-rep" else path)
        rows = list(sqlite_rows(db))
        samples = dram_samples(db)
        dram = Sampled(samples) if samples else None
    side = re.compile(args.side) if args.side else None
    side_raw, side_calls = collections.Counter(), collections.Counter()
    chain = []
    for s, e, name, shape in rows:
        key = f"{name} {shape}" if args.shape else name
        if side and side.search(name):
            side_raw[key] += e - s
            side_calls[key] += 1
        else:
            chain.append((s, e, key))
    chain.sort(key=lambda x: x[1])

    crit, raw, calls = collections.Counter(), collections.Counter(), collections.Counter()
    rd, rd_time = collections.Counter(), collections.Counter()
    buckets = [0.0] * 4  # busy time with DRAM read < 20, < 50, < 80, >= 80 % of peak
    last = min(s for s, _, _ in chain)
    first, idle = last, 0
    for s, e, key in chain:
        idle += max(0, s - last)
        a = max(last, s)
        crit[key] += max(0, e - a)
        raw[key] += e - s
        calls[key] += 1
        if dram and e > a:
            m = dram.mean(a, e)
            if m is not None:
                rd[key] += m * (e - a)
                rd_time[key] += e - a
                buckets[0 if m < 20 else 1 if m < 50 else 2 if m < 80 else 3] += e - a
        last = max(last, e)
    n, busy = args.rounds, sum(crit.values())
    print(f"per round: {(last - first) / 1e3 / n:.1f} us from first start to last end, {busy / 1e3 / n:.1f} us busy, "
          f"{idle / 1e3 / n:.1f} us idle, {len(chain) / n:.0f} kernels")
    if sum(buckets):
        tb = sum(buckets)
        print(f"DRAM read over the busy time: mean {sum(rd.values()) / sum(rd_time.values()):.1f}% of peak; "
              f"<20%: {100 * buckets[0] / tb:.0f}% of the time, 20-50%: {100 * buckets[1] / tb:.0f}%, "
              f"50-80%: {100 * buckets[2] / tb:.0f}%, >=80%: {100 * buckets[3] / tb:.0f}%")
    dcol = dram is not None
    print(f"{'kernel':<56} {'calls':>6} {'crit us':>8} {'share':>6} {'crit/call':>9} {'raw/call':>9}" + (f" {'DRAM rd':>7}" if dcol else ""))
    for key, c in crit.most_common(args.top):
        line = (f"{key[:56]:<56} {calls[key] / n:>6.0f} {c / 1e3 / n:>8.1f} {100 * c / busy:>5.1f}% {c / 1e3 / calls[key]:>9.2f} "
                f"{raw[key] / 1e3 / calls[key]:>9.2f}")
        if dcol:
            line += f" {rd[key] / rd_time[key]:>6.0f}%" if rd_time[key] else f" {'-':>7}"
        print(line)
    for key, r in side_raw.most_common():
        print(f"{'(side) ' + key[:49]:<56} {side_calls[key] / n:>6.0f} {'':>8} {'':>6} {'':>9} {r / 1e3 / side_calls[key]:>9.2f}")


if __name__ == "__main__":
    main()
