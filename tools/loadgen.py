"""Synthetic multi-conversation load for an OpenAI-compatible server (tools/README.md, dev/KBENCH.md).

    python3 tools/loadgen.py --url http://127.0.0.1:8000 --turns 3 --target 8192 sweep --levels 1,2,4,8
    python3 tools/loadgen.py --url http://127.0.0.1:8000 --turns 3 consistency --conversations 8
    python3 tools/loadgen.py --url http://127.0.0.1:8000 --turns 8 --target 262144 fill --conversations 4

(Options common to the modes go before the mode.)

Each conversation is agent-like: a system prompt, then turns of a user message carrying a task and a
block of filler text, the assistant's reply appended to the history before the next turn, so the
prompt grows towards --target tokens at the last turn. Prompts follow from --seed and the
conversation's number alone (the filler comes from the repository's tracked source files and docs, or
is generated with --filler synthetic): the same seed gives the same prompts as long as the replies
are the same. Requests stream (SSE) with temperature 0 (greedy) unless --temperature says otherwise.

Per request it measures the wait for a slot (up to the server's prefill-start progress chunk; quench6k
and llama.cpp send them for "return_progress"; see derive() for servers that report only after
chunks), the prefill of the prompt tokens not served from cache
(usage.prompt_tokens_details.cached_tokens), time to first token, decode tok/s and the total. A 503
(e.g. model_switch_blocked from serve-models) is retried after its Retry-After; other errors such as
400 context_length_exceeded end that conversation.

Modes:
  sweep        the same workload shape at concurrency 1, 2, 4, 8 (--levels): aggregate and
               per-conversation throughput and latency per level (fresh prompts per level, so one
               level's cache does not serve the next).
  consistency  each conversation alone, one after another, then all at once; compares the texts turn
               by turn and reports the first differing turn and character (batch invariance end to
               end). Exit status 1 when anything differs or fails. To compare like with like, each
               conversation gets a system prompt of its own (--shared-system: one for all) and the
               model is unloaded between the phases (--between none: not), so both phases start from
               an empty cache and each turn reuses the same cached prefix; give the server at least
               as many slots as conversations. (A server that keeps contexts across an unload, such
               as quench6k's Qwen session with its context store, still serves the second phase from
               them.) Turns served from cache differently in the two phases are counted.
  fill         --conversations grown concurrently to --target each, a row per request (prompt, reused,
               new, timing), then one short revisit turn per conversation to show what stayed cached.

--json FILE appends JSON lines: a header (arguments, model, corpus digest), one per request, and the
summaries.
"""

import argparse
import hashlib
import http.client
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse

M64 = (1 << 64) - 1
MIN_DECODE = 16  # generated tokens a request needs for its decode tok/s


class Rng:
    """splitmix64 seeded from a hash of the keys: the same keys give the same stream everywhere."""

    def __init__(self, *keys):
        self.s = int.from_bytes(hashlib.sha256(repr(keys).encode()).digest()[:8], "little")

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n

    def pick(self, seq):
        return seq[self.below(len(seq))]


# ---- filler ---------------------------------------------------------------------------------------

TEXT_EXT = {".rs", ".py", ".md", ".cu", ".cuh", ".toml", ".sh", ".js"}

WORDS = (
    "the a of to in and for with on by from at as kernel buffer slot request token context cache "
    "layer expert router batch queue stream thread block warp memory latency throughput prefill decode "
    "draft verify accept reject window shard page tile matrix vector scale weight bias norm gate value "
    "state step round chunk prefix suffix store evict reuse load parse merge split check report table "
    "quickly slowly carefully every each some most all none first last next previous"
).split()


class Corpus:
    """Filler text: the repository's tracked text files (`repo`) or generated text (`synthetic`)."""

    def __init__(self, kind, root=None):
        self.kind = kind
        self.text, self.starts, self.paths = "", [], []
        if kind == "repo":
            out = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True, check=True).stdout
            parts = []
            pos = 0
            for p in sorted(x for x in out.decode().split("\0") if os.path.splitext(x)[1] in TEXT_EXT):
                full = os.path.join(root, p)
                if not os.path.isfile(full) or os.path.getsize(full) > 2 << 20:
                    continue
                with open(full, encoding="utf-8", errors="replace") as f:
                    body = f"\n==> {p} <==\n" + f.read()
                self.starts.append(pos)
                self.paths.append(p)
                parts.append(body)
                pos += len(body)
            self.text = "".join(parts)
            if len(self.text) < 4096:
                raise SystemExit(f"loadgen: too little text under {root} for --filler repo; use --filler synthetic")
            self.digest = hashlib.sha256(self.text.encode()).hexdigest()[:16]
        else:
            self.digest = "synthetic-v1"

    def chunk(self, rng, n):
        """(label, about n characters of text) drawn with rng."""
        if self.kind == "synthetic":
            return f"notes-{rng.below(10000):04d}.txt", synthetic(rng, n)
        t = self.text
        off = rng.below(len(t))
        nl = t.find("\n", off)
        off = nl + 1 if 0 <= nl < len(t) - 1 else 0
        s = t[off:off + n]
        while len(s) < n:
            s += t[: n - len(s)]
        cut = s.rfind("\n")
        if cut > n // 2:
            s = s[: cut + 1]
        lo, hi = 0, len(self.starts)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.starts[mid] <= off:
                lo = mid
            else:
                hi = mid
        return self.paths[lo] if self.paths else "corpus", s


def synthetic(rng, n):
    """Procedural text: prose, log lines and code-like lines, about n characters."""
    out, size = [], 0
    while size < n:
        k = rng.below(3)
        if k == 0:
            line = " ".join(rng.pick(WORDS) for _ in range(8 + rng.below(10))).capitalize() + "."
        elif k == 1:
            line = (f"2031-{1 + rng.below(12):02d}-{1 + rng.below(28):02d} {rng.below(24):02d}:{rng.below(60):02d}:"
                    f"{rng.below(60):02d} node-{rng.below(64)} {rng.pick(['INFO', 'WARN', 'DEBUG'])} "
                    f"{rng.pick(WORDS)} {rng.pick(WORDS)} took {rng.below(5000)} ms")
        else:
            line = (f"let {rng.pick(WORDS)}_{rng.below(100)} = {rng.pick(WORDS)}({rng.below(4096)}, "
                    f"{rng.pick(WORDS)}_{rng.below(100)}); // {rng.pick(WORDS)} {rng.pick(WORDS)}")
        out.append(line)
        size += len(line) + 1
    return "\n".join(out) + "\n"


SYSTEM = ("You are a careful software engineering assistant working in a code repository. The user "
          "shows you files or notes and gives you a task about them. Answer the task directly and "
          "briefly, in plain prose.")

TASKS = [
    "Summarize what the following text does in two sentences.",
    "Name the most important identifier in the following text and say what it is for.",
    "Point out one thing in the following text that could be a bug or a mistake, and why.",
    "Suggest one test that the following code or text would need.",
    "Explain the following text to a new team member in three short bullet points.",
    "What would you rename in the following text, and to what?",
    "List the external things the following text depends on.",
    "Write a one-line commit message for a change that adds the following text.",
    "Which part of the following text is the hardest to understand, and why?",
    "Describe the data flow in the following text in one paragraph.",
    "Does the following text agree with what we discussed earlier? Answer in two sentences.",
    "Give the following text a title of at most eight words.",
]


class Conversation:
    """One synthetic conversation: its messages and how much filler the next turn gets."""

    def __init__(self, cid, key, corpus, turns, target, chars_per_token, min_filler, unique_system=False):
        self.cid, self.corpus, self.turns, self.target = cid, corpus, turns, target
        self.rng = Rng("conversation", key, cid)
        self.ratio = chars_per_token
        self.min_filler = min_filler
        # A system prompt of its own: no other conversation shares a cached beginning with it.
        system = f"Session {key[0]}-{cid}. {SYSTEM}" if unique_system else SYSTEM
        self.messages = [{"role": "system", "content": system}]
        self.last_prompt = None  # prompt tokens and completion tokens of the last request
        self.last_completion = 0
        self.sent_chars = 0
        self.done = 0  # turns completed

    def chars(self):
        return sum(len(m["content"]) for m in self.messages)

    def next_user(self, turn):
        """Appends the user message of `turn` (0-based); its filler aims the prompt at the plan."""
        want = self.target * (turn + 1) // self.turns
        if self.last_prompt:
            have = self.last_prompt + self.last_completion + 8
        else:
            have = int(self.chars() / self.ratio) + 16
        task = self.rng.pick(TASKS)
        need = max(self.min_filler, int((want - have - 40) * self.ratio))
        label, text = self.corpus.chunk(self.rng, need)
        content = f"Task {turn + 1}: {task}\n\n<file name=\"{label}\">\n{text}</file>"
        self.messages.append({"role": "user", "content": content})
        self.sent_chars = self.chars()
        return content

    def add_reply(self, rec):
        """Appends the assistant's reply and recalibrates characters per token from the usage."""
        if rec.get("prompt_tokens"):
            self.last_prompt = rec["prompt_tokens"]
            self.last_completion = rec.get("completion_tokens") or 0
            self.ratio = max(1.0, self.sent_chars / self.last_prompt)
        self.messages.append({"role": "assistant", "content": rec.get("content") or "(no answer)"})


# ---- HTTP / SSE -----------------------------------------------------------------------------------

def iter_sse(fp):
    """Yields the data of each server-sent event read from a binary file-like object."""
    data = []
    while True:
        line = fp.readline()
        if not line:
            break
        line = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)


class Client:
    def __init__(self, url, model, args):
        u = urllib.parse.urlsplit(url if "://" in url else "http://" + url)
        self.host, self.port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
        self.https = u.scheme == "https"
        self.prefix = u.path.rstrip("/")
        if self.prefix.endswith("/v1"):
            self.prefix = self.prefix[:-3]
        self.model, self.args = model, args

    def conn(self, timeout):
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout)

    def post_json(self, path, timeout=600):
        c = self.conn(timeout)
        try:
            c.request("POST", self.prefix + path, body=b"{}", headers={"Content-Type": "application/json"})
            r = c.getresponse()
            body = r.read()
            if r.status != 200:
                raise RuntimeError(f"POST {path}: HTTP {r.status}: {body[:200]!r}")
            return json.loads(body)
        finally:
            c.close()

    def get_json(self, path):
        c = self.conn(30)
        try:
            c.request("GET", self.prefix + path)
            r = c.getresponse()
            body = r.read()
            if r.status != 200:
                raise RuntimeError(f"GET {path}: HTTP {r.status}: {body[:200]!r}")
            return json.loads(body)
        finally:
            c.close()

    def body(self, messages, seed):
        a = self.args
        b = {"model": self.model, "messages": messages, "stream": True,
             "stream_options": {"include_usage": True}, "return_progress": True,
             "max_tokens": a.max_tokens, "temperature": a.temperature, "seed": seed}
        if a.thinking is not None:
            b["chat_template_kwargs"] = {"enable_thinking": a.thinking}
        if a.extra:
            b.update(a.extra)
        return b

    def chat(self, messages, seed):
        """One streamed chat completion, retried on 503 after Retry-After. Returns a record."""
        rec = {"retries": 0, "retry_wait_s": 0.0}
        payload = json.dumps(self.body(messages, seed)).encode()
        t_start = time.monotonic()
        while True:
            one = self._once(payload)
            if one.get("status") == 503 and rec["retries"] < self.args.retries:
                wait = one.get("retry_after")
                wait = 2.0 if wait is None else wait
                rec["retries"] += 1
                rec["retry_wait_s"] += wait
                time.sleep(wait)
                continue
            break
        rec.update(one)
        rec["total_s"] = time.monotonic() - t_start
        return rec

    def _once(self, payload):
        r = {"status": None, "error": None, "content": "", "reasoning": "", "finish": None}
        t0 = time.monotonic()
        c = self.conn(self.args.request_timeout)
        t_first = None
        progress = []  # (seconds since sending, processed, cache, total, server time_ms)
        try:
            c.request("POST", self.prefix + "/v1/chat/completions", body=payload,
                      headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            resp = c.getresponse()
            r["status"] = resp.status
            if resp.status != 200:
                raw = resp.read()
                try:
                    err = json.loads(raw).get("error", {})
                    err = err if isinstance(err, dict) else {"message": str(err)}
                except ValueError:
                    err = {"message": raw[:300].decode("utf-8", errors="replace")}
                r["error"] = err.get("code") or err.get("type") or f"http_{resp.status}"
                r["error_message"] = err.get("message", "")
                ra = resp.getheader("Retry-After")
                if ra is not None:
                    try:
                        r["retry_after"] = float(ra)
                    except ValueError:
                        pass
                return r
            content, reasoning = [], []
            for data in iter_sse(resp):
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                now = time.monotonic() - t0
                if "error" in ev:
                    e = ev["error"]
                    r["error"] = "stream_error"
                    r["error_message"] = e.get("message", str(e)) if isinstance(e, dict) else str(e)
                    break
                pp = ev.get("prompt_progress")
                if pp and pp.get("processed") is not None:
                    progress.append((now, pp["processed"], pp.get("cache") or 0, pp.get("total") or 0,
                                     pp.get("time_ms")))
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    for key, sink in (("content", content), ("reasoning_content", reasoning)):
                        if d.get(key):
                            if t_first is None:
                                t_first = now
                            sink.append(d[key])
                    if d.get("tool_calls") and t_first is None:
                        t_first = now
                    if ch.get("finish_reason"):
                        r["finish"] = ch["finish_reason"]
                u = ev.get("usage")
                if u:
                    r["prompt_tokens"] = u.get("prompt_tokens")
                    r["completion_tokens"] = u.get("completion_tokens")
                    r["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
            r["content"], r["reasoning"] = "".join(content), "".join(reasoning)
            if r["error"] is None and r["finish"] is None:
                r["error"] = "incomplete_stream"
        except (OSError, http.client.HTTPException) as e:
            r["error"] = type(e).__name__
            r["error_message"] = str(e)
        finally:
            c.close()
        t_end = time.monotonic() - t0
        derive(r, progress, t_first, t_end)
        return r


def derive(r, progress, t_first, t_end):
    """Fills in the timing fields of a record from the stream's event times (seconds from sending).

    The wait for a slot ends at the prefill's start: a first progress chunk with nothing computed yet
    (processed == cache) marks it exactly ("admission": "exact"). Servers that report only after a
    prefill chunk get it extrapolated back at the rate between their progress chunks ("estimated"),
    or, with a single chunk, not at all (None: the wait is then part of the prefill time)."""
    admit = how = prefilled = None
    if progress:
        t0, done0, cache0 = progress[0][:3]
        if done0 <= cache0:
            admit, how = t0, "exact"
        else:
            t1, done1 = progress[-1][:2]
            if done1 > done0 and t1 > t0:
                admit, how = max(0.0, t0 - (done0 - cache0) * (t1 - t0) / (done1 - done0)), "estimated"
        prefilled = next((p[0] for p in progress if p[1] >= p[3]), None)
    r["admission"] = how
    r["wait_s"] = admit
    ms = progress[0][4] if how == "exact" else None
    r["server_wait_s"] = None if ms is None else ms / 1000
    r["ttft_s"] = t_first
    r["stream_s"] = t_end
    prompt, cached = r.get("prompt_tokens"), r.get("cached_tokens") or 0
    r["new_tokens"] = None if prompt is None else prompt - cached
    end_prefill = prefilled if prefilled is not None else t_first
    r["prefill_s"] = None if end_prefill is None else end_prefill - (admit or 0.0)
    r["prefill_tps"] = (r["new_tokens"] / r["prefill_s"]
                        if r["new_tokens"] and r["prefill_s"] and r["prefill_s"] > 0 else None)
    n = r.get("completion_tokens")
    r["decode_s"] = None if t_first is None else t_end - t_first
    # Speculative decoding streams tokens in bursts: a rate over a few tokens says nothing.
    r["decode_tps"] = (n - 1) / r["decode_s"] if n and n >= MIN_DECODE and r["decode_s"] and r["decode_s"] > 0 else None


# ---- running --------------------------------------------------------------------------------------

def run(client, convs, concurrency, args, emit, phase):
    """Runs the conversations, at most `concurrency` at a time, each turn after the previous reply.
    Returns (records, wall seconds)."""
    records, lock = [], threading.Lock()
    todo = list(convs)
    if not todo:
        return records, 0.0
    start = threading.Barrier(min(concurrency, len(convs)))

    def turns(conv):
        while conv.done < conv.turns:
            turn = conv.done
            conv.next_user(turn)
            rec = client.chat(list(conv.messages), seed=args.seed + conv.cid)
            rec.update(phase=phase, conv=conv.cid, turn=turn)
            with lock:
                records.append(rec)
                emit(rec)
            if rec["error"]:
                return
            conv.add_reply(rec)
            conv.done += 1
            if args.think_time:
                time.sleep(args.think_time)

    def worker():
        start.wait()
        while True:
            with lock:
                if not todo:
                    return
                conv = todo.pop(0)
            turns(conv)

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(concurrency, len(convs)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return records, time.monotonic() - t0


def pct(values, q):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def summarize(records, wall, conc):
    ok = [r for r in records if not r["error"]]
    gen = sum(r.get("completion_tokens") or 0 for r in ok)
    prompt = sum(r.get("prompt_tokens") or 0 for r in ok)
    cached = sum(r.get("cached_tokens") or 0 for r in ok)
    new = sum(r.get("new_tokens") or 0 for r in ok)
    per_conv = {}
    for r in ok:
        c = per_conv.setdefault(r["conv"], [0, 0.0])
        c[0] += r.get("completion_tokens") or 0
        c[1] += r["total_s"]
    errors = {}
    for r in records:
        if r["error"]:
            errors[r["error"]] = errors.get(r["error"], 0) + 1
    return {
        "type": "summary", "concurrency": conc, "requests": len(records), "ok": len(ok), "errors": errors,
        "retries": sum(r["retries"] for r in records), "wall_s": wall,
        "generated": gen, "gen_tps": gen / wall if wall > 0 else None,
        "prompt_tokens": prompt, "cached_tokens": cached, "new_tokens": new,
        "prefill_tps_agg": new / wall if wall > 0 else None,
        "reuse": cached / prompt if prompt else None,
        "decode_tps_p50": pct([r["decode_tps"] for r in ok], 0.5),
        "conv_tps_mean": statistics.mean(g / t for g, t in per_conv.values() if t > 0) if per_conv else None,
        "prefill_tps_p50": pct([r["prefill_tps"] for r in ok], 0.5),
        "wait_p50": pct([r["wait_s"] for r in ok], 0.5), "wait_max": pct([r["wait_s"] for r in ok], 1.0),
        "ttft_p50": pct([r["ttft_s"] for r in ok], 0.5), "ttft_p90": pct([r["ttft_s"] for r in ok], 0.9),
        "total_p50": pct([r["total_s"] for r in ok], 0.5),
    }


def f(v, spec="{:.1f}", none="-"):
    return none if v is None else spec.format(v)


def table(rows, header):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    lines = ["  ".join(str(x).rjust(w) for x, w in zip(header, widths))]
    lines += ["  ".join(str(x).rjust(w) for x, w in zip(row, widths)) for row in rows]
    return "\n".join(lines)


SUMMARY_HEAD = ["conc", "reqs", "err", "wall s", "gen tok/s", "prefill tok/s", "dec tok/s p50", "conv tok/s",
                "wait p50/max s", "ttft p50/p90 s", "total p50 s", "reuse%"]


def summary_row(label, s):
    err = sum(s["errors"].values())
    return [label, s["requests"], f"{err}" + (f" r{s['retries']}" if s["retries"] else ""), f(s["wall_s"]),
            f(s["gen_tps"]), f(s["prefill_tps_agg"], "{:.0f}"), f(s["decode_tps_p50"]), f(s["conv_tps_mean"]),
            f"{f(s['wait_p50'], '{:.2f}')}/{f(s['wait_max'], '{:.2f}')}",
            f"{f(s['ttft_p50'], '{:.2f}')}/{f(s['ttft_p90'], '{:.2f}')}", f(s["total_p50"], "{:.2f}"),
            f(None if s["reuse"] is None else 100 * s["reuse"], "{:.0f}")]


REQ_HEAD = ["phase", "conv", "turn", "prompt", "reused", "new", "wait s", "prefill s", "prefill tok/s", "gen",
            "ttft s", "dec tok/s", "total s", "finish"]


def request_row(r):
    finish = r["error"] if r["error"] else r["finish"]
    if r["retries"]:
        finish = f"{finish} (r{r['retries']})"
    return [r["phase"], r["conv"], r["turn"] + 1, f(r.get("prompt_tokens"), "{}"), f(r.get("cached_tokens"), "{}"),
            f(r.get("new_tokens"), "{}"),
            ("~" if r.get("admission") == "estimated" else "") + f(r.get("wait_s"), "{:.2f}"), f(r.get("prefill_s"), "{:.2f}"),
            f(r.get("prefill_tps"), "{:.0f}"), f(r.get("completion_tokens"), "{}"), f(r.get("ttft_s"), "{:.2f}"),
            f(r.get("decode_tps")), f(r.get("total_s"), "{:.2f}"), finish]


def admission_note(records):
    """A line on requests whose wait for a slot the stream did not show exactly, or ''."""
    ok = [r for r in records if not r["error"]]
    est = sum(r.get("admission") == "estimated" for r in ok)
    none = sum(r.get("admission") is None for r in ok)
    parts = []
    if est:
        parts.append(f"{est} waits (~) extrapolated from prefill progress")
    if none:
        parts.append(f"{none} requests without a prefill-start progress event: their wait counts as prefill")
    return ("note: the server reports progress only after prefill chunks: " + "; ".join(parts)) if parts else ""


def first_diff(a, b):
    """Index of the first differing character, or None when equal."""
    if a == b:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def compare(alone, together):
    """Per conversation: (conversation, None) when the texts match turn by turn, else (conversation,
    description of the first difference)."""
    out = []
    key = lambda recs: {(r["conv"], r["turn"]): r for r in recs}
    a_by, b_by = key(alone), key(together)
    for cid in sorted({r["conv"] for r in alone + together}):
        turns = sorted({t for (c, t) in list(a_by) + list(b_by) if c == cid})
        diff = None
        for t in turns:
            a, b = a_by.get((cid, t)), b_by.get((cid, t))
            if a is None or b is None or a["error"] or b["error"]:
                which = "alone" if a is None or a["error"] else "together"
                r = a if which == "alone" else b
                diff = {"turn": t, "kind": "error", "detail": f"{which}: {r['error'] if r else 'missing'}"}
                break
            for field in ("reasoning", "content"):
                i = first_diff(a[field], b[field])
                if i is not None:
                    diff = {"turn": t, "kind": field, "char": i,
                            "alone": a[field][max(0, i - 30): i + 50], "together": b[field][max(0, i - 30): i + 50],
                            "alone_tokens": a.get("completion_tokens"), "together_tokens": b.get("completion_tokens"),
                            "alone_cached": a.get("cached_tokens"), "together_cached": b.get("cached_tokens")}
                    break
            if diff:
                break
            if a.get("completion_tokens") != b.get("completion_tokens") or a["finish"] != b["finish"]:
                diff = {"turn": t, "kind": "tokens",
                        "detail": f"alone {a.get('completion_tokens')} tokens {a['finish']}, "
                                  f"together {b.get('completion_tokens')} tokens {b['finish']}"}
                break
        out.append((cid, diff, len(turns)))
    return out


# ---- modes ----------------------------------------------------------------------------------------

def make_convs(args, corpus, key, n, target, turns=None, unique_system=False):
    return [Conversation(i, key, corpus, turns or args.turns, target, args.chars_per_token, args.min_filler,
                         unique_system)
            for i in range(n)]


def mode_sweep(args, client, corpus, target, out):
    rows = []
    for level in args.levels:
        key = (args.seed, "sweep", level) if not args.same_prompts else (args.seed, "sweep")
        convs = make_convs(args, corpus, key, level, target)
        recs, wall = run(client, convs, level, args, out.request, f"c{level}")
        s = summarize(recs, wall, level)
        out.summary(s, phase=f"c{level}")
        rows.append(summary_row(level, s))
        print(f"concurrency {level}: {s['requests']} requests, {sum(s['errors'].values())} errors, "
              f"{f(s['gen_tps'])} generated tok/s, wall {f(s['wall_s'])} s", flush=True)
    print()
    print(table(rows, SUMMARY_HEAD))
    errs = [r for r in out.records if r["error"]]
    for r in errs[:10]:
        print(f"error: {r['phase']} conv {r['conv']} turn {r['turn'] + 1}: {r['error']}: {r.get('error_message', '')}")
    note = admission_note(out.records)
    if note:
        print(note)
    return 0


def mode_consistency(args, client, corpus, target, out):
    key = (args.seed, "consistency")
    n = args.conversations
    phases = {}
    order = ["alone", "together"] if args.order == "alone-first" else ["together", "alone"]
    for k, phase in enumerate(order):
        if k > 0 and args.between == "unload":
            # The second phase starts from an empty cache, as the first did (the server keeps no context
            # across an unload; the next request loads the model again).
            try:
                client.post_json("/v1/unload")
                print("between phases: model unloaded (cached contexts dropped)", flush=True)
            except (OSError, RuntimeError, ValueError, http.client.HTTPException) as e:
                print(f"between phases: could not unload ({e}); the second phase may reuse the first's cache",
                      flush=True)
        convs = make_convs(args, corpus, key, n, target, unique_system=args.unique_system)
        recs, wall = run(client, convs, 1 if phase == "alone" else n, args, out.request, phase)
        s = summarize(recs, wall, 1 if phase == "alone" else n)
        out.summary(s, phase=phase)
        phases[phase] = (recs, s)
    print(table([summary_row(p, phases[p][1]) for p in order], ["phase"] + SUMMARY_HEAD[1:]))
    print()
    same = 0
    for cid, diff, nturns in compare(phases["alone"][0], phases["together"][0]):
        out.write({"type": "consistency", "conv": cid, "turns": nturns, "diff": diff})
        if diff is None:
            same += 1
            print(f"conv {cid}: identical over {nturns} turns")
        elif diff["kind"] in ("error", "tokens"):
            print(f"conv {cid}: turn {diff['turn'] + 1}: {diff['detail']}")
        else:
            print(f"conv {cid}: turn {diff['turn'] + 1} {diff['kind']} differs at character {diff['char']} "
                  f"(alone {diff['alone_tokens']} tokens, together {diff['together_tokens']})")
            print(f"    alone:    {diff['alone']!r}")
            print(f"    together: {diff['together']!r}")
            print(f"    prompt tokens served from cache: alone {diff['alone_cached']}, together {diff['together_cached']}")
    note = admission_note(out.records)
    if note:
        print(note)
    # Turns whose prompts were served from cache differently in the two phases (prefill chunks then start
    # elsewhere, and cached reply tokens come from decode steps rather than prefill): how many differ.
    a_by = {(r["conv"], r["turn"]): r for r in phases["alone"][0]}
    b_by = {(r["conv"], r["turn"]): r for r in phases["together"][0]}
    both = [(a_by[k], b_by[k]) for k in a_by if k in b_by and not a_by[k]["error"] and not b_by[k]["error"]]
    other = [(a, b) for a, b in both if (a.get("cached_tokens") or 0) != (b.get("cached_tokens") or 0)]
    if other:
        changed = sum(a["content"] != b["content"] or a["reasoning"] != b["reasoning"] for a, b in other)
        print(f"turns served from cache differently in the two phases: {len(other)} of {len(both)}, "
              f"{changed} of them with different text")
    print(f"\nconsistency: {same}/{n} conversations identical alone and {n} at once")
    return 0 if same == n else 1


def mode_fill(args, client, corpus, target, out):
    key = (args.seed, "fill")
    convs = make_convs(args, corpus, key, args.conversations, target)
    print(table([], REQ_HEAD), flush=True)

    def live(rec):
        out.request(rec)
        print("  ".join(str(x) for x in request_row(rec)), flush=True)

    recs, wall = run(client, convs, args.conversations, args, live, "grow")
    s = summarize(recs, wall, args.conversations)
    out.summary(s, phase="grow")
    rows = [request_row(r) for r in recs]
    if args.revisit:
        # One short turn per conversation, one at a time: how much of each is still cached.
        live_convs = [c for c in convs if c.done == c.turns]
        for c in live_convs:
            c.turns += 1
            c.target = 0  # the least filler
        rv, rwall = run(client, live_convs, 1, args, live, "revisit")
        out.summary(summarize(rv, rwall, 1), phase="revisit")
        rows += [request_row(r) for r in rv]
    print()
    print(table(rows, REQ_HEAD))
    print()
    print(table([summary_row(args.conversations, s)], SUMMARY_HEAD))
    for c in convs:
        print(f"conv {c.cid}: reached {c.last_prompt or 0} prompt tokens")
    note = admission_note(out.records)
    if note:
        print(note)
    return 0


class Output:
    def __init__(self, path, verbose):
        self.fp = open(path, "a") if path else None
        self.verbose = verbose
        self.records = []

    def write(self, obj):
        if self.fp:
            self.fp.write(json.dumps(obj) + "\n")
            self.fp.flush()

    def request(self, rec):
        self.records.append(rec)
        self.write(dict(rec, type="request"))
        if self.verbose:
            print("  ".join(str(x) for x in request_row(rec)), flush=True)

    def summary(self, s, phase):
        self.write(dict(s, phase=phase))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL (with or without /v1)")
    p.add_argument("--model", help="model name to request (default: the first of /v1/models)")
    p.add_argument("--seed", type=int, default=1, help="workload seed (prompts follow from it)")
    p.add_argument("--turns", type=int, default=3, help="turns per conversation")
    p.add_argument("--target", type=int, default=4096,
                   help="prompt tokens at the last turn (capped below the server's context)")
    p.add_argument("--max-tokens", type=int, default=64, help="max_tokens per turn")
    p.add_argument("--temperature", type=float, default=0.0, help="0 is greedy")
    think = p.add_mutually_exclusive_group()
    think.add_argument("--thinking", dest="thinking", action="store_true", default=False,
                       help="let the model think (default: chat_template_kwargs.enable_thinking false)")
    think.add_argument("--server-thinking", dest="thinking", action="store_const", const=None,
                       help="leave thinking to the server's default")
    p.add_argument("--extra", type=json.loads, help="JSON object merged into every request body")
    p.add_argument("--filler", choices=["repo", "synthetic"], default="repo",
                   help="filler from the repository's tracked text files, or generated text")
    p.add_argument("--corpus-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   help="git checkout for --filler repo (default: this one)")
    p.add_argument("--chars-per-token", type=float, default=3.5,
                   help="first guess for sizing filler; later turns use the server's prompt_tokens")
    p.add_argument("--min-filler", type=int, default=200, help="filler characters per turn at least")
    p.add_argument("--think-time", type=float, default=0.0, help="seconds between a reply and the next turn")
    p.add_argument("--retries", type=int, default=30, help="503 retries per request (after Retry-After)")
    p.add_argument("--request-timeout", type=float, default=900, help="socket timeout per request, seconds")
    p.add_argument("--json", help="append JSON lines (header, requests, summaries) to this file")
    p.add_argument("-v", "--verbose", action="store_true", help="print a row per request as it completes")
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sweep", help="throughput and latency at several concurrencies")
    s.add_argument("--levels", type=lambda v: [int(x) for x in v.split(",")], default=[1, 2, 4, 8])
    s.add_argument("--same-prompts", action="store_true",
                   help="reuse the same prompts at every level (cache hits across levels)")
    c = sub.add_parser("consistency", help="texts alone vs all at once, turn by turn")
    c.add_argument("--conversations", type=int, default=8)
    c.add_argument("--order", choices=["alone-first", "together-first"], default="alone-first")
    c.add_argument("--shared-system", dest="unique_system", action="store_false",
                   help="one system prompt for all conversations (default: one each, so no conversation "
                        "reuses another's cached beginning)")
    c.add_argument("--between", choices=["unload", "none"], default="unload",
                   help="between the phases, POST /v1/unload so the second starts from an empty cache as "
                        "the first did (the model loads again), or nothing")
    fl = sub.add_parser("fill", help="grow conversations to a target context each")
    fl.add_argument("--conversations", type=int, default=4)
    fl.add_argument("--no-revisit", dest="revisit", action="store_false",
                    help="skip the final short turn per conversation")
    args = p.parse_args(argv)
    if args.turns < 1:
        p.error("--turns must be at least 1")

    client = Client(args.url, None, args)
    try:
        models = client.get_json("/v1/models").get("data") or []
    except (OSError, RuntimeError, ValueError, http.client.HTTPException) as e:
        print(f"loadgen: cannot list models at {args.url}: {e}", file=sys.stderr)
        return 2
    entry = next((m for m in models if m.get("id") == args.model), None) if args.model else (models[0] if models else None)
    if args.model and entry is None:
        print(f"loadgen: warning: {args.model!r} is not in /v1/models ({[m.get('id') for m in models]})",
              file=sys.stderr)
    client.model = args.model or (entry or {}).get("id") or "default"
    context = (entry or {}).get("max_model_len") or (entry or {}).get("context_length")
    target = args.target
    if context:
        cap = context - args.max_tokens - 256
        if target > cap:
            print(f"loadgen: --target {target} capped to {cap} (context {context}, max_tokens {args.max_tokens})")
            target = cap
    try:
        corpus = Corpus(args.filler, args.corpus_root)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"loadgen: cannot read the repository's files ({e}); use --filler synthetic", file=sys.stderr)
        return 2
    out = Output(args.json, args.verbose)
    conv_count = getattr(args, "conversations", None)
    print(f"model {client.model}, context {context or '?'}, {args.mode}: {args.turns} turns towards {target} tokens, "
          f"max_tokens {args.max_tokens}, temperature {args.temperature}, filler {args.filler} ({corpus.digest}), "
          f"seed {args.seed}" + (f", {conv_count} conversations" if conv_count else ""), flush=True)
    out.write({"type": "header", "mode": args.mode, "model": client.model, "context": context, "target": target,
               "corpus": corpus.digest, "argv": sys.argv[1:] if argv is None else argv})
    if args.verbose and args.mode != "fill":
        print(table([], REQ_HEAD), flush=True)
    try:
        return {"sweep": mode_sweep, "consistency": mode_consistency, "fill": mode_fill}[args.mode](
            args, client, corpus, target, out)
    finally:
        if out.fp:
            out.fp.close()


if __name__ == "__main__":
    sys.exit(main())
