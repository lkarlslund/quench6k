#!/usr/bin/env python3
"""Independent PyTorch reference of SPEC.md for Qwen3.8 Flash-Next (text model).

It runs a whole token sequence through the model one layer at a time (teacher forcing), streaming
each layer's weights from the safetensors checkpoint, so the GPU only ever holds one layer. All
arithmetic is FP32 (optionally FP64) on BF16 weights; NVFP4 experts are dequantized exactly, and
`--act-quant` additionally quantizes expert inputs to NVFP4 the way a W4A4 engine does.

Outputs the oracle.distributions/1 format of oracle/fixtures (see oracle/README.md), and can dump
per-op tensors of selected layers as golden data for kernel tests.
"""
import argparse
import json
import math
import pathlib
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

EPS = 1e-6
D, S, H = 2560, 4, 10240
EOS = 248044
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


class Checkpoint:
    def __init__(self, root: pathlib.Path, device, dtype):
        self.root, self.device, self.dtype = root, device, dtype
        self.where = {}
        self.headers = {}
        for path in sorted(root.glob("*.safetensors")):
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            header.pop("__metadata__", None)
            self.headers[path] = (8 + n, header)
            for name in header:
                self.where[name] = path
        self.handles = {}

    def raw(self, name):
        path = self.where[name]
        if path not in self.handles:
            self.handles[path] = safe_open(str(path), framework="pt", device="cpu")
        return self.handles[path].get_tensor(name)

    def get(self, name):
        """A floating-point weight, upcast to the compute dtype on the device."""
        return self.raw(name).to(self.device).to(self.dtype)

    def ple_table(self):
        """Row lookup into the 128-shard PLE table via numpy memmaps of each shard."""
        prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
        shards = {}
        for name, path in self.where.items():
            if name.startswith(prefix):
                data_start, header = self.headers[path]
                t = header[name]
                rows, width = t["shape"]
                shards[int(name[len(prefix):].split(".")[0])] = np.memmap(
                    path, dtype=np.uint16, mode="r", offset=data_start + t["data_offsets"][0], shape=(rows, width))
        return [shards[i] for i in range(len(shards))]


def rms(x, w=None, plus_one=True):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    if w is None:
        return y
    return y * (1 + w) if plus_one else y * w


def grouped_rms(x, w):
    """x [..., H] normalized per 2560-wide stream; w [H] with the (1 + w) convention."""
    shape = x.shape
    y = rms(x.reshape(*shape[:-1], S, D))
    return y.reshape(shape) * (1 + w)


def rope(x, cos, sin):
    """Rotate-half RoPE on the first 64 dims of the last axis. cos/sin: [T, 64]."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    while cos.dim() < xr.dim():
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    half = r // 2
    rotated = torch.cat((-xr[..., half:], xr[..., :half]), dim=-1)
    return torch.cat((xr * cos + rotated * sin, xp), dim=-1)


def rope_tables(positions, device, dtype):
    inv_freq = 1.0 / (10_000_000 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
    freqs = positions.to(torch.float64)[:, None] * inv_freq[None, :]
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos().to(device, dtype), freqs.sin().to(device, dtype)


def nvfp4_dequant(codes, scales, scale2):
    """codes U8 [N, K/2] (low nibble first), scales F8 [N, K/16], scale2 F32 [] -> [N, K] float32."""
    lo, hi = codes & 0xF, codes >> 4
    nib = torch.stack((lo, hi), dim=-1).reshape(codes.shape[0], -1).long()
    table = E2M1.to(codes.device)
    val = table[nib & 7] * torch.where(nib & 8 != 0, -1.0, 1.0)
    s = scales.float().repeat_interleave(16, dim=1)
    return val * s * scale2.float()


def nvfp4_round(x):
    """Round to the nearest E2M1 value (ties to even code), saturating at +/-6."""
    a = x.abs().clamp(max=6.0)
    table = E2M1.to(x.device, x.dtype)
    idx = torch.bucketize(a, (table[1:] + table[:-1]) / 2)
    # bucketize puts exact midpoints in the lower bucket; move them up when the upper code is even
    mid = (table[1:] + table[:-1]) / 2
    at_mid = (idx < 7) & (a == mid[idx.clamp(max=6)])
    idx = torch.where(at_mid & (idx % 2 == 1), idx + 1, idx)
    return table[idx] * torch.sign(x)


def nvfp4_act_quant(x, input_scale):
    """Per-16-block NVFP4 quantize-dequantize of activations x [T, K] with a static global scale."""
    t, k = x.shape
    blocks = x.reshape(t, k // 16, 16)
    amax = blocks.abs().amax(-1, keepdim=True)
    scale = (amax / 6.0 / input_scale).clamp(max=448.0).to(torch.float8_e4m3fn).to(x.dtype)
    denom = scale * input_scale
    q = nvfp4_round(torch.where(denom > 0, blocks / denom, torch.zeros_like(blocks)))
    return (q * denom).reshape(t, k)


class Reference:
    def __init__(self, ckpt: Checkpoint, act_quant: bool, dump_layers=(), dump_dir=None):
        self.ck, self.act_quant = ckpt, act_quant
        self.dev, self.dt = ckpt.device, ckpt.dtype
        self.dump_layers, self.dump_dir = set(dump_layers), dump_dir
        cfg = json.loads((ckpt.root / "config.json").read_text())["text_config"]
        self.layer_types = cfg["layer_types"]

    def dump(self, layer, name, t):
        if layer in self.dump_layers and self.dump_dir is not None:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.dump_dir / f"L{layer:02d}.{name}.npy", t.detach().float().cpu().numpy())

    # --- HyperConnection -------------------------------------------------------------------
    def hc(self, prefix, h, inject=True):
        g = self.ck.get
        n = grouped_rms(h, g(f"{prefix}.hc_norm.weight"))
        z = F.silu((n @ g(f"{prefix}.input_mix_weight_down.weight").T) / S)
        gates = torch.sigmoid(z @ g(f"{prefix}.input_mix_weight_up.weight").T)
        m = (gates * n).reshape(-1, S, D).mean(dim=1)
        if not inject:
            return m, None
        inj = 2 * torch.sigmoid((n @ g(f"{prefix}.block_inject_weight.weight").T) / S)
        return m, inj

    # --- Gated DeltaNet --------------------------------------------------------------------
    def gdn(self, p, m, layer):
        g = self.ck.get
        T = m.shape[0]
        qkv = m @ g(f"{p}.in_proj_qkv.weight").T                      # [T, 10240]
        w = g(f"{p}.conv1d.weight").reshape(H, 4)                    # [C, 4]
        xp = F.pad(qkv.T, (3, 0))                                    # [C, T+3]
        conv = sum(w[:, k:k + 1] * xp[:, k:k + T] for k in range(4))
        qkv = F.silu(conv.T)
        q, k, v = qkv[:, :2048], qkv[:, 2048:4096], qkv[:, 4096:]
        q = q.reshape(T, 16, 128).repeat_interleave(3, dim=1)
        k = k.reshape(T, 16, 128).repeat_interleave(3, dim=1)
        v = v.reshape(T, 48, 128)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + EPS) / math.sqrt(128)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + EPS)
        beta = torch.sigmoid(m @ g(f"{p}.in_proj_b.weight").T)       # [T, 48]
        a = m @ g(f"{p}.in_proj_a.weight").T
        decay = -g(f"{p}.A_log").exp() * F.softplus(a + g(f"{p}.dt_bias"))  # [T, 48], <= 0
        o, state = chunked_delta_rule(q, k, v, decay, beta)
        self.dump(layer, "gdn_state", state)
        z = (m @ g(f"{p}.in_proj_z.weight").T).reshape(T, 48, 128)
        o = rms(o, g(f"{p}.norm.weight"), plus_one=False) * torch.sigmoid(z)
        return o.reshape(T, 6144) @ g(f"{p}.out_proj.weight").T

    # --- QSA attention ---------------------------------------------------------------------
    def qsa(self, p, m, cos, sin, layer):
        g = self.ck.get
        T = m.shape[0]
        qg = (m @ g(f"{p}.q_proj.weight").T).reshape(T, 24, 512)
        q, gate = qg[..., :256], qg[..., 256:].reshape(T, 6144)
        q = rope(rms(q, g(f"{p}.q_norm.weight")), cos, sin)
        k = rope(rms((m @ g(f"{p}.k_proj.weight").T).reshape(T, 2, 256), g(f"{p}.k_norm.weight")), cos, sin)
        v = (m @ g(f"{p}.v_proj.weight").T).reshape(T, 2, 256)

        mask = self.indexer_mask(f"{p}.indexer", m, cos, sin)
        self.dump(layer, "qsa_mask_rows", mask.sum(-1))
        out = torch.empty(T, 24, 256, device=self.dev, dtype=self.dt)
        kk = k.repeat_interleave(12, dim=1).transpose(0, 1)          # [24, T, 256]
        vv = v.repeat_interleave(12, dim=1).transpose(0, 1)
        for s in range(0, T, 1024):
            e = min(T, s + 1024)
            scores = torch.einsum("qhd,hkd->hqk", q[s:e], kk) / 16.0
            scores = scores.masked_fill(~mask[s:e].unsqueeze(0), float("-inf"))
            out[s:e] = torch.einsum("hqk,hkd->qhd", torch.softmax(scores, dim=-1), vv)
        o = out.reshape(T, 6144) * torch.sigmoid(gate)
        return o @ g(f"{p}.o_proj.weight").T

    def indexer_mask(self, p, m, cos, sin):
        """Bool [T, T]: which keys each query attends to after group selection (SPEC §6.2)."""
        g = self.ck.get
        T = m.shape[0]
        qk = m @ g(f"{p}.index_qk_proj.weight").T                    # [T, 640]
        iq = rope(rms(qk[:, :512].reshape(T, 4, 128), g(f"{p}.q_layernorm.weight")), cos, sin)
        raw = qk[:, 512:]
        if self.dt != torch.float64:
            raw = raw.to(torch.bfloat16).to(self.dt)                 # raw keys are stored as BF16
        G = T // 4
        idx = torch.arange(T, device=self.dev)
        causal = idx[None, :] <= idx[:, None]
        if G == 0:
            return causal
        pooled = raw[: G * 4].reshape(G, 4, 128).mean(dim=1)
        if self.dt != torch.float64:
            pooled = pooled.to(torch.bfloat16).to(self.dt)
        starts = torch.arange(G, device=self.dev) * 4
        pk = rope(rms(pooled, g(f"{p}.k_layernorm.weight")), cos[starts], sin[starts])  # [G, 128]
        scores = torch.relu(torch.einsum("thd,gd->tgh", iq, pk)).sum(-1) / math.sqrt(128)  # [T, G]
        complete = (starts[None, :] + 3) <= idx[:, None]             # group fully visible to the query
        scores = scores.masked_fill(~complete, float("-inf"))
        n_complete = complete.sum(-1)
        k_sel = min(512, G)
        top = scores.topk(k_sel, dim=-1).indices                     # [T, k_sel]
        rank_ok = torch.arange(k_sel, device=self.dev)[None, :] < n_complete[:, None].clamp(max=512)
        sel_groups = torch.zeros(T, G, dtype=torch.bool, device=self.dev)
        sel_groups.scatter_(1, top, rank_ok)
        sel_groups &= complete
        key_group = idx // 4
        mask = torch.zeros(T, T, dtype=torch.bool, device=self.dev)
        in_group = key_group < G
        mask[:, in_group] = sel_groups[:, key_group[in_group]]
        tail_start = (idx + 1) // 4 * 4                              # keys at or after this are the tail
        mask |= (idx[None, :] >= tail_start[:, None]) & causal
        return mask & causal

    # --- MoE -------------------------------------------------------------------------------
    def moe(self, p, m, layer):
        g = self.ck.get
        logits = m @ g(f"{p}.gate.weight").T
        probs = torch.softmax(logits, dim=-1)
        top_w, top_i = probs.topk(10, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)
        self.dump(layer, "moe_top_i", top_i)
        out = torch.zeros_like(m)
        for e in torch.unique(top_i).tolist():
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            x = m[rows]
            ep = f"{p}.experts.{e}"
            y = None
            ws = {}
            for proj in ("gate_proj", "up_proj", "down_proj"):
                raw = self.ck.raw
                ws[proj] = nvfp4_dequant(raw(f"{ep}.{proj}.weight").to(self.dev), raw(f"{ep}.{proj}.weight_scale").to(self.dev),
                                         raw(f"{ep}.{proj}.weight_scale_2").to(self.dev)).to(self.dt)
                ws[proj + ".in"] = float(raw(f"{ep}.{proj}.input_scale"))
            xg = nvfp4_act_quant(x, ws["gate_proj.in"]) if self.act_quant else x
            xu = nvfp4_act_quant(x, ws["up_proj.in"]) if self.act_quant else x
            hdn = F.silu(xg @ ws["gate_proj"].T) * (xu @ ws["up_proj"].T)
            if self.act_quant:
                hdn = nvfp4_act_quant(hdn, ws["down_proj.in"])
            y = hdn @ ws["down_proj"].T
            out.index_add_(0, rows, y * top_w[rows, slot].unsqueeze(-1))
        sh = F.silu(m @ g(f"{p}.shared_expert.gate_proj.weight").T) * (m @ g(f"{p}.shared_expert.up_proj.weight").T)
        sh = sh @ g(f"{p}.shared_expert.down_proj.weight").T
        return out + torch.sigmoid(m @ g(f"{p}.shared_expert_gate.weight").T) * sh

    # --- PLE -------------------------------------------------------------------------------
    def ple(self, p, h, tokens):
        g, raw = self.ck.get, self.ck.raw
        T = tokens.shape[0]
        mult = raw(f"{p}.ple_embedding.layer_multipliers").long()
        sizes = raw(f"{p}.ple_embedding.ngram_heads_vocab_sizes").long()
        offs = raw(f"{p}.ple_embedding.ngram_heads_offsets").long()
        toks = tokens.long().cpu()
        # s_k: token k back within the EOS-delimited segment, else EOS (SPEC §8.1)
        seg_start = torch.zeros(T, dtype=torch.long)
        start = 0
        for t in range(T):
            seg_start[t] = start
            if toks[t] == EOS:
                start = t + 1
        pos = torch.arange(T)
        s = []
        for k in range(3):
            src = (pos - k).clamp(min=0)
            s.append(torch.where((pos - k) >= seg_start, toks[src], torch.full_like(toks, EOS)))
        bigram = (s[0] * mult[0]) ^ (s[1] * mult[1])
        trigram = bigram ^ (s[2] * mult[2])
        rows = torch.cat((torch.remainder(bigram[:, None], sizes[None, :8]) + offs[None, :8],
                          torch.remainder(trigram[:, None], sizes[None, 8:]) + offs[None, 8:]), dim=1)  # [T, 16]
        shards = self.ck.ple_table()
        per = shards[0].shape[0]
        r = rows.numpy()
        e = np.empty((T, 16, 160), dtype=np.uint16)
        for j in range(16):
            for t in range(T):
                e[t, j] = shards[r[t, j] // per][r[t, j] % per]
        e = torch.from_numpy(e.view(np.int16).astype(np.int32) << 16).view(torch.float32)
        e = e.reshape(T, 2560).to(self.dev, self.dt)

        kn = grouped_rms(e @ g(f"{p}.key_proj.weight").T, g(f"{p}.norm_key.weight")).reshape(T, S, D)
        qn = grouped_rms(h, g(f"{p}.norm_query.weight")).reshape(T, S, D)
        v = e @ g(f"{p}.value_proj.weight").T
        a = (kn * qn).sum(-1) / math.sqrt(D)
        a = a.abs().clamp(min=1e-6).sqrt() * a.sign()
        gv = (torch.sigmoid(a)[..., None] * v[:, None, :]).reshape(T, H)
        xn = grouped_rms(gv, g(f"{p}.norm_conv.weight"))
        w = g(f"{p}.conv1d.weight").reshape(H, 4)
        xp = F.pad(xn.T, (9, 0))
        conv = sum(w[:, k:k + 1] * xp[:, 3 * k:3 * k + T] for k in range(4))
        return gv + F.silu(conv.T)

    # --- whole model -----------------------------------------------------------------------
    def forward(self, tokens):
        g = self.ck.get
        T = tokens.shape[0]
        tok = tokens.to(self.dev)
        x0 = self.ck.raw("model.language_model.embed_tokens.weight")[tokens.cpu()].to(self.dev, self.dt)
        h = x0.repeat(1, S)
        cos, sin = rope_tables(torch.arange(T), self.dev, self.dt)
        for i in range(48):
            t0 = time.time()
            p = f"model.language_model.layers.{i}"
            if i == 1:
                h = h + self.ple(f"{p}.ple", h, tok)
                self.dump(i, "after_ple", h)
            m, inj = self.hc(f"{p}.attn_hyper_connection", h)
            self.dump(i, "attn_in", m)
            if self.layer_types[i] == "linear_attention":
                y = self.gdn(f"{p}.linear_attn", m, i)
            else:
                y = self.qsa(f"{p}.self_attn", m, cos, sin, i)
            self.dump(i, "attn_out", y)
            h = h + (y[:, None, :] * inj[:, :, None]).reshape(T, H)
            m, inj = self.hc(f"{p}.mlp_hyper_connection", h)
            self.dump(i, "moe_in", m)
            y = self.moe(f"{p}.mlp", m, i)
            self.dump(i, "moe_out", y)
            h = h + (y[:, None, :] * inj[:, :, None]).reshape(T, H)
            self.dump(i, "residual", h)
            self.ck.handles.clear()
            print(f"  layer {i:2d} {self.layer_types[i][:6]} {time.time() - t0:6.2f}s", flush=True)
        out, _ = self.hc("model.language_model.hyper_connection_mixer", h, inject=False)
        return out


def chunked_delta_rule(q, k, v, decay, beta, chunk=64):
    """Gated delta rule over a whole sequence (SPEC §5), chunked WY form, from zero state.

    q, k: [T, Hh, 128] (q already scaled), v: [T, Hh, 128], decay, beta: [T, Hh]. Returns o [T, Hh, 128]
    and the final state [Hh, 128(k), 128(v)].
    """
    T, Hh, dk = k.shape
    dv = v.shape[-1]
    pad = (-T) % chunk
    q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
    decay, beta = (F.pad(x, (0, 0, 0, pad)) for x in (decay, beta))
    n = (T + pad) // chunk
    q, k, v = (x.reshape(n, chunk, Hh, -1).permute(2, 0, 1, 3) for x in (q, k, v))    # [Hh, n, C, d]
    decay, beta = (x.reshape(n, chunk, Hh).permute(2, 0, 1) for x in (decay, beta))  # [Hh, n, C]
    cum = decay.cumsum(-1)
    lower = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
    strict = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device), -1)
    diff = (cum[..., :, None] - cum[..., None, :]).masked_fill(~lower, float("-inf"))
    pdecay = diff.exp()                                                # gamma_i / gamma_j for j <= i
    kb = k * beta[..., None]
    # (I + A) solves: A_ij = beta_i (gamma_i/gamma_j) k_i.k_j for j < i
    A = (kb @ k.transpose(-1, -2)) * pdecay
    A = A.masked_fill(~strict, 0.0)
    eye = torch.eye(chunk, dtype=q.dtype, device=q.device)
    Tinv = torch.linalg.solve_triangular(eye + A, eye.expand_as(A).contiguous(), upper=False, unitriangular=True)
    U = Tinv @ (v * beta[..., None])                                   # [Hh, n, C, dv]
    W = Tinv @ (kb * cum.exp()[..., None])                             # [Hh, n, C, dk]
    attn = (q @ k.transpose(-1, -2)) * pdecay                          # includes the diagonal
    state = torch.zeros(Hh, dk, dv, dtype=q.dtype, device=q.device)
    out = torch.empty_like(v)
    for c in range(n):
        delta = U[:, c] - W[:, c] @ state                              # [Hh, C, dv]
        out[:, c] = (q[:, c] * cum[:, c].exp()[..., None]) @ state + attn[:, c] @ delta
        last = cum[:, c, -1]
        kd = k[:, c] * (last[:, None] - cum[:, c]).exp()[..., None]
        state = state * last.exp()[:, None, None] + kd.transpose(-1, -2) @ delta
    out = out.permute(1, 2, 0, 3).reshape(n * chunk, Hh, dv)[:T]
    return out, state


def recurrent_delta_rule(q, k, v, decay, beta):
    """Token-by-token form of SPEC §5; the definition the chunked form must match."""
    T, Hh, dk = k.shape
    state = torch.zeros(Hh, dk, v.shape[-1], dtype=q.dtype, device=q.device)
    out = torch.empty_like(v)
    for t in range(T):
        state = state * decay[t].exp()[:, None, None]
        delta = beta[t][:, None] * (v[t] - torch.einsum("hkv,hk->hv", state, k[t]))
        state = state + k[t][:, :, None] * delta[:, None, :]
        out[t] = torch.einsum("hkv,hk->hv", state, q[t])
    return out, state


def score(args):
    dtype = torch.float64 if args.fp64 else torch.float32
    ckpt = Checkpoint(pathlib.Path(args.model), torch.device("cuda"), dtype)
    seq = json.loads((pathlib.Path(args.fixture) / "sequence.json").read_text())
    tokens = torch.tensor(seq["tokens"], dtype=torch.long)
    ref = Reference(ckpt, args.act_quant, args.dump_layers, pathlib.Path(args.dump_dir) if args.dump_dir else None)
    t0 = time.time()
    with torch.no_grad():
        out = ref.forward(tokens)
        P = len(tokens) - 1
        head = ckpt.get("lm_head.weight")
        lse = torch.empty(P, dtype=torch.float64)
        target = torch.empty(P, dtype=torch.float64)
        top_ids = torch.empty(P, args.top_k, dtype=torch.int32)
        top_logits = torch.empty(P, args.top_k, dtype=torch.float64)
        for s in range(0, P, 512):
            e = min(P, s + 512)
            logits = (out[s:e] @ head.T).double()
            lse[s:e] = torch.logsumexp(logits, dim=-1).cpu()
            target[s:e] = logits.gather(1, tokens[s + 1:e + 1, None].to(logits.device)).squeeze(1).cpu()
            v, i = logits.topk(args.top_k, dim=-1)
            top_ids[s:e], top_logits[s:e] = i.int().cpu(), v.cpu()
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.with_suffix(".bin"), "wb") as f:
        f.write(lse.float().numpy().tobytes())
        f.write(target.float().numpy().tobytes())
        f.write(top_ids.numpy().astype("<i4").tobytes())
        f.write(top_logits.float().numpy().tobytes())
    meta = {"engine": "squash-ref", "format": "oracle.distributions/1", "mode": "teacher_forced",
            "positions": P, "prompt_len": seq["prompt_len"], "top_k": args.top_k,
            "act_quant": args.act_quant, "dtype": str(dtype), "seconds": time.time() - t0}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True, help="oracle fixture directory (uses sequence.json)")
    ap.add_argument("--out", required=True, help="output stem; writes .bin and .json")
    ap.add_argument("--act-quant", action="store_true", help="emulate NVFP4 W4A4 expert activations")
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--dump-layers", type=int, nargs="*", default=[])
    ap.add_argument("--dump-dir")
    score(ap.parse_args())


if __name__ == "__main__":
    main()
