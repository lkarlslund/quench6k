#!/usr/bin/env python3
"""Nemotron-H (NemotronHForCausalLM: Nemotron 3 Super, Munin 2.0) text reference: an independent PyTorch
implementation after the checkpoint's modeling_nemotron_h.py, scoring token sequences teacher-forced while
streaming one layer of weights at a time. It runs on the CPU (default) or a GPU (`--device cuda`).

It reads the checkpoint's own NVFP4 (routed experts), FP8-per-tensor and BF16 weights and dequantizes them
(only the experts a layer's tokens route to), so it judges an engine on the same weights. `--dtype bf16`
(default) rounds activations to BF16 where the model does when it runs with its CUDA kernels (mamba_ssm,
causal_conv1d; the path every serving stack takes) and rounds dequantized weights to BF16; `--dtype fp32`
keeps the same math in FP32 and exact weights, and gives the noise floor. Matmuls accumulate in FP32.

Per block: x + mixer(RMSNorm(x)), the mixer one of
- Mamba-2: in_proj -> z | xBC | dt; causal depthwise conv (kernel 4, bias) + SiLU on xBC, rounded once;
  dt = softplus(dt + dt_bias) in FP32, clamped to config time_step_limit (default (0, inf): no clamp, as
  the CUDA path; the modeling file's torch_forward instead clamps at time_step_min, `--torch-dt-clamp`);
  A = -exp(A_log); then the plain sequential recurrence, token by token, with an FP32 state (`--scan
  fp64` for the floor): h = exp(dt A) h + dt x B^T, y = h C + D x. This is exact, not the chunked
  (SSD) form the kernels and torch_forward use; both compute the same recurrence. y is rounded to BF16
  (the chunk-scan kernel's output dtype; `--fp32 scan_out` keeps it FP32 as torch_forward does), then the
  gated RMSNorm over groups of intermediate/n_groups with norm_before_gate=False: rmsnorm(y * silu(z)) * w,
  in FP32, rounded once, and out_proj.
- attention: q/k/v projections, GQA, no positional encoding (the modeling file applies none), causal
  softmax over head_dim^-0.5-scaled scores (eager roundings: BF16 scores and probabilities, FP32 softmax),
  o_proj. `--kv-fp8` rounds K and V through E4M3 with the checkpoint's static k_scale / v_scale, as an
  engine with an FP8 cache would.
- MoE: router sigmoid(x W^T) in FP32, choice by sigmoid + e_score_correction_bias, weights from the plain
  sigmoid scores normalized over the chosen experts times routed_scaling_factor; the routed experts run in
  the latent space (fc1_latent_proj before, fc2_latent_proj after) as down(relu(up(x))^2), their weighted
  sum accumulated in FP32 and rounded once before fc2_latent_proj; plus the shared expert relu^2 MLP on
  the block input; routed + shared in BF16.
Final norm_f, then FP32 logits from the BF16 lm_head.

The residual stream between blocks is BF16 as in the modeling file (residual_in_fp32 false); `--fp32
residual` keeps it in FP32 and normalizes it from FP32, as an engine with an FP32 residual stream does.

Outputs the oracle.distributions/1 format of oracle/fixtures (see oracle/README.md), can dump the residual
stream after chosen layers' residual adds (`--dump-layers`: <dump-dir>/seq<j>/lanes-NN.npy, F32 [T, hidden])
and the final hidden states after norm_f (<dump-dir>/final-<fixture>.npy) for engine debugging, and with
`--prompt` prints the next-token candidates of a text.

    python3 -I tools/ref/nemotron_ref.py --model $M --fixture out/nemotron-fixtures/* --out out/nemotron-ref --suffix=-bf16
    python3 -I tools/ref/nemotron_ref.py --model $M --prompt 'The capital of Denmark is'
"""
import argparse
import json
import math
import pathlib
import resource
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
          "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8, "I8": torch.int8, "I32": torch.int32,
          "I64": torch.int64}
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
# Byte -> (low nibble, high nibble) values: element 2i of a row is the low nibble of byte i.
NVFP4_LUT = torch.tensor([[E2M1[b & 15], E2M1[b >> 4]] for b in range(256)], dtype=torch.float32)
FP8_MAX = 448.0


class Checkpoint:
    """Tensors read with plain file reads (no mmap, so the process's RSS is what it holds), after checking
    that every shard of the index is complete."""

    def __init__(self, root: pathlib.Path):
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.tensors, self.files = {}, {}
        for shard in sorted(set(index["weight_map"].values())):
            path = root / shard
            if not path.exists():
                raise SystemExit(f"{path} is missing (download not finished?)")
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            header.pop("__metadata__", None)
            end = max(t["data_offsets"][1] for t in header.values())
            if path.stat().st_size != 8 + n + end:
                raise SystemExit(f"{path}: {path.stat().st_size} bytes, the header says {8 + n + end}")
            for name, t in header.items():
                a, b = t["data_offsets"]
                self.tensors[name] = (path, 8 + n + a, b - a, t["dtype"], t["shape"])
        missing = set(index["weight_map"]) - set(self.tensors)
        if missing:
            raise SystemExit(f"{len(missing)} tensors of the index are in no shard, e.g. {sorted(missing)[0]}")

    def has(self, name):
        return name in self.tensors

    def raw(self, name):
        path, off, nbytes, dtype, shape = self.tensors[name]
        if path not in self.files:
            self.files[path] = open(path, "rb", buffering=0)
        f = self.files[path]
        buf = bytearray(nbytes)
        f.seek(off)
        view, got = memoryview(buf), 0
        while got < nbytes:
            r = f.readinto(view[got:])
            if not r:
                raise SystemExit(f"{path}: short read of {name}")
            got += r
        t = torch.frombuffer(buf, dtype=torch.uint8) if nbytes else torch.empty(0, dtype=torch.uint8)
        return t.view(DTYPES[dtype]).reshape(shape)

    def weight(self, base, device):
        """`base`.weight dequantized to FP32 on `device`: NVFP4 (U8 codes, E4M3 scale per 16, F32
        weight_scale_2), FP8 E4M3 with one F32 weight_scale, or a plain float tensor."""
        w = self.raw(base + ".weight")
        if w.dtype == torch.uint8:
            codes = w.to(device)
            scale = self.raw(base + ".weight_scale").to(device).float()
            scale2 = self.raw(base + ".weight_scale_2").to(device).float()
            o, half = codes.shape
            v = NVFP4_LUT.to(device)[codes.long()].view(o, half * 2 // 16, 16) * scale[..., None]
            return v.view(o, half * 2) * scale2
        if w.dtype == torch.float8_e4m3fn:
            scale = self.raw(base + ".weight_scale").to(device).float()
            assert scale.numel() == 1, f"{base}: FP8 with {scale.numel()} scales"
            return w.to(device).float() * scale.reshape(())
        return w.to(device).float()


def describe(ck, base):
    w = ck.tensors[base + ".weight"][3]
    return {"U8": "NVFP4", "F8_E4M3": "FP8"}.get(w, w)


class Reference:
    def __init__(self, ck, cfg, args):
        self.ck, self.cfg = ck, cfg
        self.dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
        self.dev = torch.device(args.device)
        self.fp32 = set(args.fp32)
        self.res_dt = torch.float32 if "residual" in self.fp32 else self.dt
        self.scan_dt = {"fp32": torch.float32, "fp64": torch.float64}[args.scan]
        self.kv_fp8 = args.kv_fp8
        self.dump_layers = set(args.dump_layers)
        self.dump_dir = pathlib.Path(args.dump_dir) if args.dump_dir else None
        self.D = cfg["hidden_size"]
        self.eps = cfg.get("layer_norm_epsilon", cfg.get("norm_eps", 1e-5))
        kinds = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
        self.kinds = [kinds[c] for c in cfg["hybrid_override_pattern"]]
        assert len(self.kinds) == cfg["num_hidden_layers"]
        lim = cfg.get("time_step_limit") or (0.0, math.inf)
        self.dt_lo, self.dt_hi = float(lim[0]), float(lim[1])
        if args.torch_dt_clamp:
            self.dt_lo = cfg["time_step_min"]
        self.memo = {}
        self.stats = {}

    # -- helpers -------------------------------------------------------------------------------------------
    def w(self, name):
        """FP32 weight `name` (.weight), dequantized once per layer; rounded to BF16 under --dtype bf16."""
        if name not in self.memo:
            w = self.ck.weight(name, self.dev)
            self.memo[name] = w.to(self.dt).float() if self.dt != torch.float32 else w
        return self.memo[name]

    def p(self, name):
        """A parameter tensor (norm weights, biases, A_log, ...) as FP32."""
        if name not in self.memo:
            self.memo[name] = self.ck.raw(name).to(self.dev).float()
        return self.memo[name]

    def expert(self, name):
        w = self.ck.weight(name, self.dev)
        return w.to(self.dt).float() if self.dt != torch.float32 else w

    def lin(self, x, w):
        """x @ w^T with FP32 accumulation, rounded once to the activation dtype."""
        return F.linear(x.float(), w).to(self.dt)

    def rms(self, x, weight):
        """NemotronHRMSNorm: FP32 normalization, times the FP32 weight, rounded to the activation dtype."""
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (weight * y).to(self.dt)

    def relu2(self, x):
        return F.relu(x.float()).square().to(self.dt)

    # -- mixers --------------------------------------------------------------------------------------------
    def mamba(self, h, p, layer):
        c = self.cfg
        T = h.shape[0]
        H, P, N, G = c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"], c["n_groups"]
        I = H * P
        K = c["conv_kernel"]
        m = p + ".mixer"
        proj = self.lin(h, self.w(m + ".in_proj"))
        conv_dim = I + 2 * G * N
        assert proj.shape[1] == I + conv_dim + H, proj.shape
        z, xbc, dt_raw = proj.split([I, conv_dim, H], dim=-1)

        # Causal depthwise conv over time (weight [C, 1, K]: tap K-1 is the current token) + bias + SiLU,
        # in FP32, rounded once (causal_conv1d_fn).
        cw = self.p(m + ".conv1d.weight").reshape(conv_dim, K)
        xf = torch.cat((xbc.new_zeros(K - 1, conv_dim).float(), xbc.float()), dim=0)
        conv = self.p(m + ".conv1d.bias").expand(T, conv_dim).clone() if self.ck.has(m + ".conv1d.bias") \
            else torch.zeros(T, conv_dim, device=self.dev)
        for k in range(K):
            conv += xf[k:k + T] * cw[:, k]
        xbc = F.silu(conv).to(self.dt)
        x, B, C = xbc.split([I, G * N, G * N], dim=-1)

        sd = self.scan_dt
        dt = F.softplus(dt_raw.float() + self.p(m + ".dt_bias"))
        self.stats.setdefault("dt_below_min", []).append((dt < c["time_step_min"]).float().mean().item())
        if self.dt_lo > 0.0 or self.dt_hi < math.inf:
            dt = dt.clamp(self.dt_lo, self.dt_hi)
        dt = dt.to(sd)
        A = -torch.exp(self.p(m + ".A_log").to(sd))
        Dskip = self.p(m + ".D").to(sd)
        xs = x.reshape(T, H, P).to(sd)
        rep = H // G  # head h reads group h // rep
        Bs = B.reshape(T, G, N).to(sd).repeat_interleave(rep, dim=1)  # [T, H, N]
        Cs = C.reshape(T, G, N).to(sd).repeat_interleave(rep, dim=1)
        decay = torch.exp(dt * A)  # [T, H]
        dtx = (dt[..., None] * xs)[..., None]  # [T, H, P, 1]
        state = torch.zeros(H, P, N, dtype=sd, device=self.dev)
        y = torch.empty(T, H, P, dtype=sd, device=self.dev)
        for t in range(T):
            state.mul_(decay[t][:, None, None]).baddbmm_(dtx[t], Bs[t][:, None, :])
            torch.bmm(state, Cs[t][:, :, None], out=y[t][:, :, None])
        y = y + Dskip[:, None] * xs
        y = y.reshape(T, I).float()
        if "scan_out" not in self.fp32:
            y = y.to(self.dt).float()

        # Gated RMSNorm, norm_before_gate=False: normalize y * silu(z) over groups of I / G.
        g = (y * F.silu(z.float())).view(T, G, I // G)
        g = g * torch.rsqrt(g.square().mean(-1, keepdim=True) + self.eps)
        out = (g.reshape(T, I) * self.p(m + ".norm.weight")).to(self.dt)
        return self.lin(out, self.w(m + ".out_proj"))

    def attention(self, h, p):
        c = self.cfg
        T = h.shape[0]
        Hq, Hk, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        m = p + ".mixer"
        q = self.lin(h, self.w(m + ".q_proj")).view(T, Hq, d)
        k = self.lin(h, self.w(m + ".k_proj")).view(T, Hk, d)
        v = self.lin(h, self.w(m + ".v_proj")).view(T, Hk, d)
        if self.kv_fp8:
            def fp8(t, scale):
                s = self.ck.raw(scale).float().reshape(()).to(self.dev)
                q8 = (t.float() / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                return (q8.float() * s).to(self.dt)
            k, v = fp8(k, m + ".k_proj.k_scale"), fp8(v, m + ".v_proj.v_scale")
        rep = Hq // Hk  # q head i reads kv head i // rep
        Kt = k.float().repeat_interleave(rep, dim=1).permute(1, 2, 0)  # [Hq, d, T]
        Vh = v.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)  # [Hq, T, d]
        Qh = q.float().permute(1, 0, 2)  # [Hq, T, d]
        scaling = d ** -0.5
        out = torch.empty(T, Hq, d, dtype=self.dt, device=self.dev)
        kpos = torch.arange(T, device=self.dev)
        step = 256
        for s in range(0, T, step):
            e = min(T, s + step)
            sc = (torch.matmul(Qh[:, s:e], Kt[:, :, :e]).to(self.dt) * scaling).float()
            mask = kpos[None, :e] <= torch.arange(s, e, device=self.dev)[:, None]
            sc = sc.masked_fill(~mask[None], float("-inf"))
            probs = torch.softmax(sc, dim=-1).to(self.dt).float()
            out[s:e] = torch.matmul(probs, Vh[:, :e]).to(self.dt).permute(1, 0, 2)
        return self.lin(out.reshape(T, Hq * d), self.w(m + ".o_proj"))

    def mlp(self, h, base):
        return self.lin(self.relu2(self.lin(h, self.w(base + ".up_proj"))), self.w(base + ".down_proj"))

    def moe(self, h, p):
        c = self.cfg
        T = h.shape[0]
        m = p + ".mixer"
        E, k = c["n_routed_experts"], c["num_experts_per_tok"]
        scores = torch.sigmoid(h.float() @ self.p(m + ".gate.weight").T)
        choice = scores + self.p(m + ".gate.e_score_correction_bias")
        ng, tg = c.get("n_group", 1), c.get("topk_group", 1)
        if ng > 1:
            gs = choice.view(T, ng, E // ng).topk(2, dim=-1).values.sum(-1)
            keep = torch.zeros_like(gs).scatter_(1, gs.topk(tg, dim=-1).indices, 1.0).bool()
            choice = choice.masked_fill(~keep.repeat_interleave(E // ng, dim=1), float("-inf"))
        top_i = choice.topk(k, dim=-1).indices
        top_w = scores.gather(1, top_i)
        if c.get("norm_topk_prob", True):
            top_w = top_w / (top_w.sum(-1, keepdim=True) + 1e-20)
        top_w = top_w * c["routed_scaling_factor"]

        lat = self.lin(h, self.w(m + ".fc1_latent_proj")) if self.ck.has(m + ".fc1_latent_proj.weight") else h
        acc = torch.zeros(T, lat.shape[1], dtype=torch.float32, device=self.dev)
        hit = torch.unique(top_i).tolist()
        self.stats.setdefault("experts_hit", []).append(len(hit))
        for e in hit:
            tok, pos = torch.where(top_i == e)
            base = f"{m}.experts.{e}"
            a = self.relu2(self.lin(lat[tok], self.expert(base + ".up_proj")))
            o = self.lin(a, self.expert(base + ".down_proj"))
            acc.index_add_(0, tok, o.float() * top_w[tok, pos, None])
        routed = acc.to(self.dt)
        if self.ck.has(m + ".fc2_latent_proj.weight"):
            routed = self.lin(routed, self.w(m + ".fc2_latent_proj"))
        return (routed.float() + self.mlp(h, m + ".shared_experts").float()).to(self.dt)

    # -- model ---------------------------------------------------------------------------------------------
    def layer(self, x, i):
        p = f"backbone.layers.{i}"
        h = self.rms(x, self.p(p + ".norm.weight"))
        kind = self.kinds[i]
        if kind == "mamba":
            y = self.mamba(h, p, i)
        elif kind == "attention":
            y = self.attention(h, p)
        elif kind == "moe":
            y = self.moe(h, p)
        else:
            y = self.mlp(h, p + ".mixer")
        return (x.float() + y.float()).to(self.res_dt)

    def forward(self, sequences):
        """Final hidden states (after norm_f) of each token sequence, layer-major."""
        emb = self.ck.raw("backbone.embeddings.weight")
        xs = [emb[t].to(self.dev).to(self.dt).to(self.res_dt) for t in sequences]
        del emb
        for i, kind in enumerate(self.kinds):
            t0 = time.time()
            self.stats = {}
            if kind in ("moe", "mlp") and len(xs) > 1:
                # Per-token blocks: all sequences' tokens at once, so each expert is dequantized once.
                lens = [x.shape[0] for x in xs]
                xs = list(self.layer(torch.cat(xs), i).split(lens))
            else:
                xs = [self.layer(x, i) for x in xs]
            for j in range(len(xs)):
                if i in self.dump_layers and self.dump_dir:
                    d = self.dump_dir / f"seq{j}"
                    d.mkdir(parents=True, exist_ok=True)
                    np.save(d / f"lanes-{i:02d}.npy", xs[j].float().cpu().numpy())
            self.memo.clear()
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
            extra = ""
            if "dt_below_min" in self.stats:
                extra = f" dt<min {100 * max(self.stats['dt_below_min']):.2f}%"
            if "experts_hit" in self.stats:
                extra = f" experts {max(self.stats['experts_hit'])}"
            print(f"layer {i:2d} {kind:9s} {time.time() - t0:6.1f}s{extra}", flush=True)
        out = [self.rms(x, self.p("backbone.norm_f.weight")) for x in xs]
        self.memo.clear()
        return out


def peak_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def load_tokenizer(root):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(root / "tokenizer.json"))


def run(args):
    root = pathlib.Path(args.model)
    cfg = json.loads((root / "config.json").read_text())
    cfg.pop("quantization_config", None)
    if args.threads:
        torch.set_num_threads(args.threads)
    ck = Checkpoint(root)
    if args.layers:
        cfg["hybrid_override_pattern"] = cfg["hybrid_override_pattern"][:args.layers]
        cfg["num_hidden_layers"] = args.layers
    ref = Reference(ck, cfg, args)
    lm = "lm_head.weight"
    tok = None
    if args.prompt is not None:
        tok = load_tokenizer(root)
        ids = tok.encode(args.prompt).ids  # the tokenizer adds no BOS (add_bos_token false)
        fixtures, seqs, prompt_lens = [None], [torch.tensor(ids, dtype=torch.long)], [len(ids)]
    else:
        fixtures = [pathlib.Path(f) for f in args.fixture]
        seqs, prompt_lens = [], []
        for f in fixtures:
            seq = json.loads((f / "sequence.json").read_text())
            tokens = torch.tensor(seq["tokens"], dtype=torch.long)
            if args.limit:
                tokens = tokens[:args.limit]
            seqs.append(tokens)
            prompt_lens.append(min(seq["prompt_len"], len(tokens)))
    n_tok = sum(len(s) for s in seqs)
    print(f"{len(seqs)} sequence(s), {n_tok} tokens, dtype {args.dtype}, device {args.device}, "
          f"{torch.get_num_threads()} threads", flush=True)
    t0 = time.time()
    with torch.no_grad():
        outs = ref.forward(seqs)
        head = ck.raw(lm).to(ref.dev).float()
        t_fwd = time.time() - t0
        for f, tokens, out, prompt_len in zip(fixtures, seqs, outs, prompt_lens):
            if f is None:
                logits = out.float() @ head.T
                lp = torch.log_softmax(logits.double(), dim=-1)
                print("position  token -> argmax")
                for t in range(len(tokens)):
                    a = int(logits[t].argmax())
                    print(f"{t:4d} {tok.decode([int(tokens[t])])!r:>18} -> {tok.decode([a])!r}")
                v, i = lp[-1].topk(10)
                print("next token after the prompt:")
                for vv, ii in zip(v.tolist(), i.tolist()):
                    print(f"  {math.exp(vv):7.4f} {ii:6d} {tok.decode([ii])!r}")
                continue
            if args.dump_dir:
                np.save(pathlib.Path(args.dump_dir) / f"final-{f.name}.npy", out.float().cpu().numpy())
            P = len(tokens) - 1
            lse = torch.empty(P, dtype=torch.float64)
            target = torch.empty(P, dtype=torch.float64)
            top_ids = torch.empty(P, args.top_k, dtype=torch.int32)
            top_logits = torch.empty(P, args.top_k, dtype=torch.float64)
            tk = tokens.to(out.device)
            for s in range(0, P, 512):
                e = min(P, s + 512)
                logits = (out[s:e].float() @ head.T).double()
                lse[s:e] = torch.logsumexp(logits, dim=-1).cpu()
                target[s:e] = logits.gather(1, tk[s + 1:e + 1, None]).squeeze(1).cpu()
                v, i = logits.topk(args.top_k, dim=-1)
                top_ids[s:e], top_logits[s:e] = i.int().cpu(), v.cpu()
            out_path = pathlib.Path(args.out) / f"{f.name}{args.suffix}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path.with_suffix(".bin"), "wb") as fh:
                fh.write(lse.float().numpy().tobytes())
                fh.write(target.float().numpy().tobytes())
                fh.write(top_ids.numpy().astype("<i4").tobytes())
                fh.write(top_logits.float().numpy().tobytes())
            meta = {"engine": "quench6k-nemotron-ref", "format": "oracle.distributions/1", "mode": "teacher_forced",
                    "positions": P, "prompt_len": prompt_len, "top_k": args.top_k, "dtype": args.dtype,
                    "scan": args.scan, "fp32": sorted(ref.fp32), "kv_fp8": args.kv_fp8,
                    "dt_clamp": [ref.dt_lo, ref.dt_hi if ref.dt_hi < math.inf else None],
                    "device": args.device, "seconds": round(time.time() - t0, 1)}
            out_path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
            nll = (lse - target).mean().item()
            print(f"{f.name}: {P} positions, mean NLL {nll:.4f}")
    total = time.time() - t0
    print(f"forward {t_fwd:.1f}s ({t_fwd / n_tok:.2f}s per token), total {total:.1f}s, "
          f"peak RSS {peak_rss_gib():.1f} GiB", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixture", nargs="+", help="fixture directories (each with sequence.json)")
    src.add_argument("--prompt", help="print the next-token candidates of this text instead")
    ap.add_argument("--out", help="output directory; writes <fixture><suffix>.{bin,json}")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--fp32", nargs="*", default=[], choices=("scan_out", "residual"),
                    help="parts kept in FP32 under --dtype bf16")
    ap.add_argument("--scan", choices=("fp32", "fp64"), default="fp32", help="Mamba state and recurrence dtype")
    ap.add_argument("--torch-dt-clamp", action="store_true",
                    help="clamp dt at time_step_min as the modeling file's torch_forward does")
    ap.add_argument("--kv-fp8", action="store_true", help="round K and V through FP8 with k_scale / v_scale")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N tokens")
    ap.add_argument("--layers", type=int, default=0, help="run only the first N layers (debugging)")
    ap.add_argument("--dump-layers", type=int, nargs="*", default=[])
    ap.add_argument("--dump-dir")
    args = ap.parse_args()
    if args.fixture and not args.out:
        ap.error("--fixture needs --out")
    run(args)


if __name__ == "__main__":
    main()
