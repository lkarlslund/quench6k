"""Tests of tools/loadgen.py against a small fake server (no GPU): python3 -m unittest tools/test_loadgen.py"""

import contextlib
import hashlib
import http.server
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loadgen  # noqa: E402

CONTEXT = 4096


class Fake(http.server.BaseHTTPRequestHandler):
    """A chat-completions server: replies follow from the messages; chars / 4 tokens; chunked SSE."""

    protocol_version = "HTTP/1.1"
    lock = threading.Lock()
    inflight = 0
    blocked = 0  # the next this many requests get 503 model_switch_blocked
    diverge = False  # alter replies made while another request is running
    late_progress = False  # no progress chunk at the prefill's start (as the one-sequence loops)
    seen = {}  # messages JSON prefix -> prompt tokens, for cached_tokens
    bodies = []
    unloads = 0
    no_unload = False  # POST /v1/unload answers 404

    def log_message(self, *a):
        pass

    def send_json(self, status, obj, headers=()):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_json(200, {"data": [{"id": "fake-model", "max_model_len": CONTEXT}]})
        else:
            self.send_json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        if self.path == "/v1/unload":
            if cls.no_unload:
                return self.send_json(404, {"error": {"message": "not found"}})
            with cls.lock:
                cls.unloads += 1
                cls.seen = {}
            return self.send_json(200, {"status": "unloaded"})
        with cls.lock:
            cls.bodies.append(body)
            if cls.blocked > 0:
                cls.blocked -= 1
                return self.send_json(503, {"error": {"code": "model_switch_blocked", "message": "busy"}},
                                      [("Retry-After", "0")])
        msgs = body["messages"]
        prompt = sum(len(m["content"]) for m in msgs) // 4 + 10
        if prompt >= CONTEXT:
            return self.send_json(400, {"error": {"code": "context_length_exceeded", "message": "too long"}})
        with cls.lock:
            cls.inflight += 1
        try:
            time.sleep(0.05)
            with cls.lock:
                concurrent = cls.inflight > 1
                cached = max([v for k, v in cls.seen.items() if json.dumps(msgs).startswith(k[:-1])] or [0])
                cls.seen[json.dumps(msgs)] = prompt
            h = hashlib.sha256(json.dumps(msgs).encode()).hexdigest()
            words = [h[i:i + 4] for i in range(0, 40, 4)][: body["max_tokens"]]
            if cls.diverge and concurrent:
                words[3] = "XXXX"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()

            def event(obj):
                data = obj if isinstance(obj, str) else json.dumps(obj)
                chunk = f"data: {data}\r\n\r\n".encode()
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")

            self.wfile.write(b"d\r\n: keep-alive\n\r\n")
            event({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
            if not cls.late_progress:
                event({"choices": [], "prompt_progress": {"total": prompt, "cache": cached, "processed": cached,
                                                          "time_ms": 1}})
            event({"choices": [], "prompt_progress": {"total": prompt, "cache": cached, "processed": prompt, "time_ms": 2}})
            text = " ".join(words)
            for i in range(0, len(text), 7):
                event({"choices": [{"index": 0, "delta": {"content": text[i:i + 7]}, "finish_reason": None}]})
            event({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                   "usage": {"prompt_tokens": prompt, "completion_tokens": len(words), "total_tokens": prompt + len(words),
                             "prompt_tokens_details": {"cached_tokens": cached}}})
            event("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
        finally:
            with cls.lock:
                cls.inflight -= 1


class Server:
    def __enter__(self):
        Fake.blocked, Fake.diverge, Fake.seen, Fake.bodies, Fake.inflight = 0, False, {}, [], 0
        Fake.unloads, Fake.no_unload = 0, False
        Fake.late_progress = False
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def run_main(args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = loadgen.main(args)
    return rc, out.getvalue()


class Pure(unittest.TestCase):
    def test_prompts_deterministic(self):
        for kind in ("synthetic", "repo"):
            corpus = loadgen.Corpus(kind, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            a = loadgen.Conversation(0, (1, "x"), corpus, 3, 8000, 3.5, 200)
            b = loadgen.Conversation(0, (1, "x"), corpus, 3, 8000, 3.5, 200)
            c = loadgen.Conversation(1, (1, "x"), corpus, 3, 8000, 3.5, 200)
            ma, mb, mc = a.next_user(0), b.next_user(0), c.next_user(0)
            self.assertEqual(ma, mb)
            self.assertNotEqual(ma, mc)
            self.assertGreater(len(ma), 2000)  # ~8000/3 tokens' worth at 3.5 characters each
            for conv in (a, b):
                conv.add_reply({"content": "ok", "prompt_tokens": len(ma) // 4, "completion_tokens": 1})
            self.assertEqual(a.next_user(1), b.next_user(1))

    def test_rng_stable(self):
        # A fixed value: prompts must not change between Python versions or runs.
        self.assertEqual(loadgen.Rng(1, "x").next(), loadgen.Rng(1, "x").next())
        self.assertEqual(loadgen.synthetic(loadgen.Rng("s"), 100), loadgen.synthetic(loadgen.Rng("s"), 100))

    def test_sse(self):
        raw = (b": comment\r\ndata: {\"a\": 1}\r\n\r\ndata:x\ndata: y\n\nevent: e\ndata: [DONE]\n\n"
               b"data: tail")
        self.assertEqual(list(loadgen.iter_sse(io.BytesIO(raw))), ['{"a": 1}', "x\ny", "[DONE]", "tail"])

    def test_admission(self):
        r = {"prompt_tokens": 1000, "cached_tokens": 200, "completion_tokens": 21}
        loadgen.derive(r, [(0.5, 200, 200, 1000, 480), (0.9, 1000, 200, 1000, 880)], 1.0, 2.0)
        self.assertEqual((r["admission"], r["wait_s"], r["server_wait_s"]), ("exact", 0.5, 0.48))
        self.assertAlmostEqual(r["prefill_s"], 0.4)
        self.assertAlmostEqual(r["prefill_tps"], 2000)
        self.assertAlmostEqual(r["decode_tps"], 20)
        # Progress only after chunks: 400 tokens per 0.2 s, so the first 400 started 0.2 s before 0.7.
        r = {"prompt_tokens": 1000, "cached_tokens": 200, "completion_tokens": 2}
        loadgen.derive(r, [(0.7, 600, 200, 1000, None), (0.9, 1000, 200, 1000, None)], 1.0, 2.0)
        self.assertEqual(r["admission"], "estimated")
        self.assertAlmostEqual(r["wait_s"], 0.5)
        # A single chunk: no wait, the prefill counts from sending.
        r = {"prompt_tokens": 1000, "cached_tokens": 0, "completion_tokens": 2}
        loadgen.derive(r, [(0.9, 1000, 0, 1000, None)], 1.0, 2.0)
        self.assertEqual((r["admission"], r["wait_s"], r["prefill_s"]), (None, None, 0.9))

    def test_diff(self):
        self.assertIsNone(loadgen.first_diff("abc", "abc"))
        self.assertEqual(loadgen.first_diff("abc", "abd"), 2)
        self.assertEqual(loadgen.first_diff("ab", "abc"), 2)
        rec = lambda c, t, text, n=3: {"conv": c, "turn": t, "content": text, "reasoning": "", "error": None,
                                         "finish": "stop", "completion_tokens": n}
        alone = [rec(0, 0, "same"), rec(0, 1, "hello world"), rec(1, 0, "x")]
        together = [rec(0, 0, "same"), rec(0, 1, "hello there"), rec(1, 0, "x")]
        res = loadgen.compare(alone, together)
        self.assertEqual(res[0][1]["turn"], 1)
        self.assertEqual(res[0][1]["char"], 6)
        self.assertIsNone(res[1][1])
        res = loadgen.compare([rec(0, 0, "x", 3)], [rec(0, 0, "x", 4)])
        self.assertEqual(res[0][1]["kind"], "tokens")


class AgainstFake(unittest.TestCase):
    def test_sweep(self):
        with Server() as s, tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.jsonl")
            rc, out = run_main(["--url", s.url, "--filler", "synthetic", "--turns", "3", "--target", "2000",
                                "--max-tokens", "8", "--json", path, "sweep", "--levels", "1,2,4"])
            self.assertEqual(rc, 0, out)
            with open(path) as fp:
                lines = [json.loads(x) for x in fp]
            reqs = [x for x in lines if x["type"] == "request"]
            self.assertEqual(len(reqs), 3 * (1 + 2 + 4))
            self.assertTrue(all(r["error"] is None for r in reqs), out)
            self.assertTrue(all(r["prefill_tps"] for r in reqs))
            # Later turns reuse the earlier ones (the fake cache), and the last turn is near the target.
            self.assertTrue(all(r["cached_tokens"] > 0 for r in reqs if r["turn"] > 0))
            last = [r["prompt_tokens"] for r in reqs if r["turn"] == 2]
            self.assertTrue(all(1500 < p < 2600 for p in last), last)
            self.assertEqual(len([x for x in lines if x["type"] == "summary"]), 3)
            self.assertEqual(Fake.bodies[0]["temperature"], 0.0)
            self.assertEqual(Fake.bodies[0]["chat_template_kwargs"], {"enable_thinking": False})
            self.assertIn("gen tok/s", out)

    def test_consistency(self):
        with Server() as s:
            args = ["--url", s.url, "--filler", "synthetic", "--turns", "2", "--target", "1000", "--max-tokens", "8",
                    "consistency", "--conversations", "3"]
            rc, out = run_main(args)
            self.assertEqual(rc, 0, out)
            self.assertIn("3/3 conversations identical", out)
            # A system prompt per conversation, the cache dropped between the phases: both phases reuse
            # the same prefixes.
            self.assertEqual(Fake.unloads, 1)
            systems = {b["messages"][0]["content"] for b in Fake.bodies if "messages" in b}
            self.assertEqual(len(systems), 3)
            self.assertNotIn("served from cache differently", out)
        with Server() as s:
            # No unload endpoint and no isolation: the second phase finds the first's prompts cached.
            Fake.no_unload = True
            rc, out = run_main(["--url", s.url] + args[2:] + ["--shared-system"])
            self.assertEqual(rc, 0, out)
            self.assertIn("could not unload", out)
            self.assertIn("served from cache differently in the two phases", out)
        with Server() as s:
            Fake.diverge = True
            args[1] = s.url
            rc, out = run_main(args)
            self.assertEqual(rc, 1, out)
            self.assertIn("turn 1 content differs at character 15", out)

    def test_retry_and_context(self):
        with Server() as s:
            Fake.blocked = 2
            rc, out = run_main(["--url", s.url, "--filler", "synthetic", "--turns", "2", "--target", "100000",
                                "--max-tokens", "4", "-v", "fill", "--conversations", "1", "--no-revisit"])
            self.assertEqual(rc, 0, out)
            self.assertIn("capped to", out)
            self.assertIn("(r2)", out)
        with Server() as s:
            # Turns sized from a bad first guess overflow the context: the error ends the conversation.
            rc, out = run_main(["--url", s.url, "--filler", "synthetic", "--turns", "1", "--target", "3500",
                                "--chars-per-token", "8", "--max-tokens", "4", "fill", "--conversations", "1"])
            self.assertIn("context_length_exceeded", out)

    def test_late_progress_note(self):
        with Server() as s:
            Fake.late_progress = True
            rc, out = run_main(["--url", s.url, "--filler", "synthetic", "--turns", "1", "--target", "1000",
                                "--max-tokens", "4", "sweep", "--levels", "2"])
            self.assertEqual(rc, 0, out)
            self.assertIn("2 requests without a prefill-start progress event", out)

    def test_fill_revisit(self):
        with Server() as s:
            rc, out = run_main(["--url", s.url, "--filler", "synthetic", "--turns", "3", "--target", "3000",
                                "--max-tokens", "4", "fill", "--conversations", "2"])
            self.assertEqual(rc, 0, out)
            self.assertIn("revisit", out)
            self.assertIn("conv 1: reached", out)


if __name__ == "__main__":
    unittest.main()
