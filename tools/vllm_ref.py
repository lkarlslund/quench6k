"""Teacher-forced reference distributions from vLLM (a qwen4_exp build) for oracle fixtures.

Writes <fixture>/vllm.{json,bin} in the oracle.distributions/1 format; vLLM reports log-probabilities,
so lse is 0 and the "logits" are log-probabilities (the gate compares log-probabilities).

  PYTHONPATH=<vllm build>/dist-packages python tools/vllm_ref.py --model M oracle/fixtures/*
"""

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

TOP_K = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="vllm")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("fixtures", nargs="+")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=args.model, max_model_len=args.max_model_len, max_logprobs=TOP_K, enforce_eager=True,
              enable_prefix_caching=False, gpu_memory_utilization=0.9, limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=TOP_K)
    for f in map(Path, args.fixtures):
        seq = json.loads((f / "sequence.json").read_text())
        tokens, prompt_len = seq["tokens"], seq["prompt_len"]
        t0 = time.time()
        out = llm.generate([TokensPrompt(prompt_token_ids=tokens)], sp, use_tqdm=False)[0]
        secs = time.time() - t0
        plp = out.prompt_logprobs  # entry i: distribution of token i given tokens < i (entry 0 is None)
        positions = len(tokens) - 1
        lse, target, ids, logits = [], [], [], []
        for j in range(positions):
            d = plp[j + 1]
            want = tokens[j + 1]
            top = sorted(((tid, lp.logprob) for tid, lp in d.items()), key=lambda x: -x[1])
            # The sequence token is always included; drop it from the top list when it is not top-k.
            ranked = [x for x in top if d[x[0]].rank is not None and d[x[0]].rank <= TOP_K][:TOP_K]
            while len(ranked) < TOP_K:
                ranked.append((-1, -math.inf))
            lse.append(0.0)
            target.append(d[want].logprob)
            ids.extend(t for t, _ in ranked)
            logits.extend(l for _, l in ranked)
        stem = f / args.name
        with open(stem.with_suffix(".bin"), "wb") as fh:
            fh.write(struct.pack(f"<{positions}f", *lse))
            fh.write(struct.pack(f"<{positions}f", *target))
            fh.write(struct.pack(f"<{positions * TOP_K}i", *ids))
            fh.write(struct.pack(f"<{positions * TOP_K}f", *logits))
        meta = {"engine": "vllm", "format": "oracle.distributions/1", "mode": "teacher_forced", "model": args.model,
                "positions": positions, "prompt_len": prompt_len, "seconds": secs, "top_k": TOP_K}
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"{f.name}: {len(tokens)} tokens in {secs:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
