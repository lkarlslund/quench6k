#!/usr/bin/env python3
"""Hy4-flash (hy_v4_flash) text reference: an independent PyTorch implementation after
modeling_hy_v4_flash.py of transformers PR 50903, scoring a token sequence teacher-forced while streaming
one layer of weights at a time to the GPU.

It reads the NVFP4 / block-FP8 / FP8-per-row layout of Avifenesh/Hy4-flash-NVFP4 and dequantizes on the
GPU, so it judges an engine on the same weights. `--dtype bf16` (default) rounds activations where the
transformers model does (BF16 hidden streams, BF16 RoPE tables, BF16 accumulation of the routed experts);
`--dtype fp32` keeps the same math in FP32 and gives the noise floor.

Outputs the oracle.distributions/1 format of oracle/fixtures (see oracle/README.md), and can dump the
hidden streams after chosen layers for engine debugging.
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

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# nn.RMSNorm(eps=None) on BF16 inputs (the Engram norms) uses BF16's machine epsilon.
TORCH_RMS_EPS = torch.finfo(torch.bfloat16).eps


class Checkpoint:
    def __init__(self, root: pathlib.Path, device):
        self.root, self.device = root, device
        self.where, self.headers, self.handles = {}, {}, {}
        for path in sorted(root.glob("*.safetensors")):
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            header.pop("__metadata__", None)
            self.headers[path] = (8 + n, header)
            for name in header:
                self.where[name] = path

    def has(self, name):
        return name in self.where

    def raw(self, name):
        path = self.where[name]
        if path not in self.handles:
            self.handles[path] = safe_open(str(path), framework="pt", device="cpu")
        return self.handles[path].get_tensor(name)

    def get(self, name, dtype):
        """A weight dequantized on the device and cast to `dtype`."""
        w = self.raw(name)
        if self.has(name + "_scale_inv"):  # block-128 FP8
            s = self.raw(name + "_scale_inv").to(self.device)
            w = w.to(self.device).float()
            o, i = w.shape
            s = s.repeat_interleave(128, 0)[:o].repeat_interleave(128, 1)[:, :i]
            return (w * s).to(dtype)
        if w.dtype == torch.uint8 and self.has(name + "_scale"):  # NVFP4, two-level
            scale = self.raw(name + "_scale")
            scale2 = self.raw(name + "_scale_2").float()
            return nvfp4_dequant(w, scale, scale2, self.device, dtype)
        return w.to(self.device).to(dtype)

    def memmap(self, name, np_dtype):
        path = self.where[name]
        start, header = self.headers[path]
        t = header[name]
        return np.memmap(path, dtype=np_dtype, mode="r", offset=start + t["data_offsets"][0],
                         shape=tuple(t["shape"]))


def nvfp4_dequant(codes, scale, scale2, device, dtype, chunk=64):
    """codes U8 [.., out, in/2] (low nibble first), scale E4M3 [.., out, in/16], scale2 F32 [E] or []."""
    lut = E2M1.to(device)
    stacked = codes.dim() == 3
    if not stacked:
        codes, scale, scale2 = codes[None], scale[None], scale2.reshape(1)
    e, o, half = codes.shape
    out = torch.empty(e, o, 2 * half, dtype=dtype, device=device)
    for s in range(0, e, chunk):
        c = codes[s:s + chunk].to(device)
        mag = torch.stack((c & 7, (c >> 4) & 7), dim=-1).flatten(-2).long()
        sign = torch.stack((c & 8, c & 128), dim=-1).flatten(-2) != 0
        v = lut[mag]
        v = torch.where(sign, -v, v)
        sc = scale[s:s + chunk].to(device).float().repeat_interleave(16, dim=-1)
        out[s:s + chunk] = (v * sc * scale2[s:s + chunk].to(device)[:, None, None]).to(dtype)
    return out if stacked else out[0]


class Reference:
    def __init__(self, ckpt: Checkpoint, cfg: dict, dtype, dump_layers=(), dump_dir=None):
        self.ckpt, self.cfg, self.dt = ckpt, cfg, dtype
        self.dev = ckpt.device
        self.dump_layers, self.dump_dir = set(dump_layers), dump_dir
        self.D = cfg["hidden_size"]
        self.eps = cfg["rms_norm_eps"]
        self.hc_eps = cfg["hc_eps"]
        self.hc = cfg["hc_mult"]
        self.layer_types = [("sliding" if t == "sliding_attention" else "indexed") for t in cfg["layer_types"]]
        self.memo = {}

    # -- helpers -------------------------------------------------------------------------------------------
    def w(self, name, dtype=None):
        """A weight, dequantized once per layer (the memo is cleared between layers)."""
        key = (name, dtype or self.dt)
        if key not in self.memo:
            self.memo[key] = self.ckpt.get(name, dtype or self.dt)
        return self.memo[key]

    def rms(self, x, weight, eps=None):
        """HYV4FlashRMSNorm: normalize in FP32, cast back to the input dtype, then scale."""
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + (self.eps if eps is None else eps))
        return weight * y.to(x.dtype)

    def torch_rms(self, x, weight):
        """nn.RMSNorm(eps=None) as the Engram layer uses it, with BF16's epsilon."""
        return F.rms_norm(x, (x.shape[-1],), weight, eps=TORCH_RMS_EPS)

    def lin(self, x, w, b=None):
        return F.linear(x, w, b)

    def rope_tables(self, positions, theta, dim=64):
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=self.dev) / dim))
        freqs = positions.float()[:, None] * inv_freq[None]
        # The transformers model casts the tables to the activation dtype; then each frequency covers a
        # channel pair (2i, 2i+1).
        cos = freqs.cos().to(self.dt).repeat_interleave(2, dim=-1)
        sin = freqs.sin().to(self.dt).repeat_interleave(2, dim=-1)
        return cos, sin

    def rope(self, x, cos, sin):
        """Interleaved-pair RoPE on the last dim; cos/sin [T, dim] broadcast over middle dims."""
        shape = [cos.shape[0]] + [1] * (x.dim() - 2) + [cos.shape[1]]
        c, s = cos.reshape(shape).float(), sin.reshape(shape).float()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return (x.float() * c + rot.float() * s).to(x.dtype)

    # -- blocks --------------------------------------------------------------------------------------------
    def hyper(self, lanes, prefix, head=False):
        """Hyper-connection pre-mix (and post weights unless `head`), in FP32."""
        T = lanes.shape[0]
        flat = lanes.reshape(T, -1).float()
        if head:
            fn = self.w(prefix + ".hc_head_fn", torch.float32)
            base = self.w(prefix + ".hc_head_base", torch.float32)
            scale = self.w(prefix + ".hc_head_scale", torch.float32)
        else:
            fn = self.w(prefix + ".hc_pre.hc_fn", torch.float32)
            base = self.w(prefix + ".hc_pre.hc_base", torch.float32)
            scale = self.w(prefix + ".hc_pre.hc_scale", torch.float32)
        mixes = (flat @ fn.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        if head:
            pre = torch.sigmoid(mixes * scale + base) + self.hc_eps
            return (pre[..., None] * lanes.float()).sum(1).to(self.dt)
        n = self.hc
        pre = torch.sigmoid(mixes[:, :n] * scale[0] + base[:n]) + self.hc_eps
        post = self.cfg["hc_magnitude"] * torch.sigmoid(mixes[:, n:] * scale[1] + base[n:]) + self.hc_eps
        out = (pre[..., None] * lanes.float()).sum(1)
        return post, out.to(self.dt)

    def mlp(self, x, prefix, clamp=None):
        g = self.lin(x, self.w(prefix + ".gate_proj.weight"))
        u = self.lin(x, self.w(prefix + ".up_proj.weight"))
        return self.lin(F.silu(g) * u, self.w(prefix + ".down_proj.weight"))

    def moe(self, x, p):
        c = self.cfg
        T = x.shape[0]
        router = self.w(p + ".gate.weight", torch.float32)
        bias = self.w(p + ".gate.e_score_correction_bias", torch.float32)
        scores = torch.sigmoid(x.float() @ router.T)
        top_i = torch.topk(scores + bias, k=c["num_experts_per_tok"], dim=-1).indices
        top_w = scores.gather(1, top_i)
        top_w = top_w / (top_w.sum(-1, keepdim=True) + 1e-20) * c["routed_scaling_factor"]

        routed_in = self.lin(x, self.w(p + ".routed_expert_down_proj.weight"))
        gate = self.w(p + ".experts.gate_proj.weight")
        up = self.w(p + ".experts.up_proj.weight")
        down = self.w(p + ".experts.down_proj.weight")
        lim = c["swiglu_limit"]
        final = torch.zeros_like(routed_in)
        hit = torch.unique(top_i).tolist()  # ascending, as the transformers loop visits them
        for e in hit:
            tok, pos = torch.where(top_i == e)
            cur = routed_in[tok]
            g = self.lin(cur, gate[e]).clamp(max=lim)
            u = self.lin(cur, up[e]).clamp(min=-lim, max=lim)
            cur = self.lin(F.silu(g) * u, down[e]) * top_w[tok, pos, None]
            final.index_add_(0, tok, cur.to(final.dtype))
        del gate, up, down
        routed_out = self.lin(final, self.w(p + ".routed_expert_up_proj.weight"))
        h = routed_out + self.mlp(x, p + ".shared_experts")
        return self.rms(h, self.w(p + ".combine_output_norm.weight"))

    def indexer_allowed(self, h, q_resid, p, positions):
        """[T, T] bool: which keys the compressed indexer lets each query see (before the causal mask)."""
        c = self.cfg
        T = h.shape[0]
        ratio = c["compressed_indexer_compress_ratio"]
        nh, hd = c["compressed_indexer_num_attention_heads"], c["compressed_indexer_head_dim"]
        ip = p + ".compressed_indexer"
        tk = self.rms(self.lin(h, self.w(ip + ".token_key_projection.weight")), self.w(ip + ".token_key_norm.weight"))
        G = T // ratio
        grouped = tk[:G * ratio].reshape(G, ratio, hd).float()
        values = grouped @ self.w(ip + ".key_value_projection.weight", torch.float32).T
        gates = grouped @ self.w(ip + ".key_gate_projection.weight", torch.float32).T
        gates = gates + self.w(ip + ".intra_group_ape", torch.float32)
        pooled = (values * gates.softmax(dim=1)).sum(1)
        gk = self.rms(pooled, self.w(ip + ".key_norm.weight", torch.float32)).to(self.dt)

        theta = c["rope_parameters"]["deepseek_sparse_attention"]["rope_theta"]
        rd = c["qk_rope_head_dim"]
        cos, sin = self.rope_tables(positions, theta, rd)
        q = self.lin(q_resid, self.w(ip + ".query_projection.weight")).reshape(T, nh, hd)
        q = torch.cat((q[..., :hd - rd], self.rope(q[..., hd - rd:], cos, sin)), dim=-1)
        gpos = positions[:G * ratio:ratio]
        gcos, gsin = self.rope_tables(gpos, theta, rd)
        gk = torch.cat((gk[..., :hd - rd], self.rope(gk[..., hd - rd:], gcos, gsin)), dim=-1)
        hw = self.lin(h, self.w(ip + ".head_weight_projection.weight")).float() * (nh ** -0.5 * hd ** -0.5)

        allowed = torch.zeros(T, T, dtype=torch.bool, device=self.dev)
        k_top = min(c["compressed_indexer_group_topk"] - 1, G)
        gids = torch.arange(G, device=self.dev)
        offsets = torch.arange(ratio, device=self.dev)
        for s in range(0, T, 512):
            e = min(T, s + 512)
            scores = F.relu(torch.einsum("thd,gd->thg", q[s:e].float(), gk.float()))
            logits = torch.einsum("th,thg->tg", hw[s:e], scores)
            qg = torch.arange(s, e, device=self.dev) // ratio
            logits = logits.masked_fill(gids[None] >= qg[:, None], float("-inf"))
            recalled = torch.cat((logits.topk(k_top, dim=-1).indices, qg[:, None]), dim=-1)
            tokens = (recalled[..., None] * ratio + offsets).flatten(-2).clamp(max=T - 1)
            allowed[s:e].scatter_(1, tokens, True)
        return allowed

    def attention(self, h, p, kind, positions):
        c = self.cfg
        T = h.shape[0]
        sliding = kind == "sliding"
        H = c["swa_num_attention_heads"] if sliding else c["num_attention_heads"]
        nope, rd, vd = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
        kvr = c["kv_lora_rank"]
        scaling = (nope + rd) ** -0.5
        a = p + ".self_attn"
        gate = self.lin(h, self.w(a + ".linear_gate.weight")).reshape(T, H, vd)
        q_resid = self.rms(self.lin(h, self.w(a + ".q_a_proj.weight")), self.w(a + ".q_a_layernorm.weight"))
        q = self.lin(q_resid, self.w(a + ".q_b_proj.weight")).reshape(T, H, nope + rd)
        ckv = self.lin(h, self.w(a + ".kv_a_proj_with_mqa.weight"))
        k_pass = self.rms(ckv[:, :kvr], self.w(a + ".kv_a_layernorm.weight"))
        theta = c["rope_parameters"]["sliding_attention" if sliding else "deepseek_sparse_attention"]["rope_theta"]
        cos, sin = self.rope_tables(positions, theta, rd)
        q_rot = self.rope(q[..., nope:], cos, sin)
        k_rot = self.rope(ckv[:, kvr:], cos, sin)
        kv = self.lin(k_pass, self.w(a + ".kv_b_proj.weight")).reshape(T, H, nope + vd)
        K = torch.cat((kv[..., :nope], k_rot[:, None].expand(T, H, rd)), dim=-1)
        V = kv[..., nope:]
        Q = torch.cat((q[..., :nope], q_rot), dim=-1)
        sinks = self.w(a + ".learnable_sink_param", torch.float32)

        allowed = None
        if not sliding and T > c["compressed_indexer_group_topk"] * c["compressed_indexer_compress_ratio"]:
            allowed = self.indexer_allowed(h, q_resid, a, positions)
        out = torch.empty(T, H, vd, dtype=self.dt, device=self.dev)
        kpos = torch.arange(T, device=self.dev)
        Kt = K.permute(1, 2, 0)  # [H, 256, T]
        Vh = V.permute(1, 0, 2)  # [H, T, vd]
        for s in range(0, T, 256):
            e = min(T, s + 256)
            qpos = torch.arange(s, e, device=self.dev)
            mask = kpos[None] <= qpos[:, None]
            if sliding:
                mask &= kpos[None] > qpos[:, None] - c["sliding_window"]
            if allowed is not None:
                mask &= allowed[s:e]
            w = torch.matmul(Q[s:e].permute(1, 0, 2).float(), Kt.float()).to(self.dt) * scaling
            w = w.float().masked_fill(~mask[None], float("-inf"))
            w = torch.cat((w, sinks[:, None, None].expand(H, e - s, 1)), dim=-1)
            probs = torch.softmax(w, dim=-1)[..., :-1]
            o = torch.matmul(probs.to(self.dt).float(), Vh.float()).to(self.dt)  # [H, q, vd]
            out[s:e] = o.permute(1, 0, 2)
        out = out * torch.sigmoid(gate)
        o = self.lin(out.reshape(T, H * vd), self.w(a + ".o_proj.weight"))
        return self.rms(o, self.w(a + ".post_proj_norm.weight"))

    def engram(self, lanes, tokens, layer_idx):
        c = self.cfg
        p = f"model.layers.{layer_idx}.engram_layer"
        layer_id = layer_idx + 1
        T = tokens.shape[0]
        n_max, nh = c["engram_max_ngram_size"], c["engram_n_head_per_ngram"]
        pad = c["engram_pad_id"]
        sizes = [s for order in c["engram_layer_head_vocab_sizes"][str(layer_id)] for s in order]

        lt = self.ckpt.raw(p + ".lookup_table").numpy().astype(np.int64)
        cv = int(lt.max()) + 1
        half_bound = max(1, np.iinfo(np.int64).max // cv // 2)
        rng = np.random.default_rng(c["engram_seed"] + 10007 * layer_id)
        mult = rng.integers(0, half_bound, size=n_max, dtype=np.int64) * 2 + 1

        tok = tokens.cpu().numpy().astype(np.int64)
        shifted = [tok]
        for off in range(1, n_max):
            prev = np.concatenate((np.full(off, pad, np.int64), tok[:-off]))[:T]
            shifted.append(prev)
        ng = np.stack(shifted, axis=-1)  # [T, n_max]: token, previous, the one before
        comp = np.where(ng < lt.shape[0], lt[np.minimum(ng, lt.shape[0] - 1)], lt[pad])
        with np.errstate(over="ignore"):
            mixed = comp * mult
        ids = []
        for n in range(2, n_max + 1):
            mix = mixed[:, 0]
            for o in range(1, n):
                mix = np.bitwise_xor(mix, mixed[:, o])
            start = (n - 2) * nh
            ids.append(mix[:, None] % np.array(sizes[start:start + nh], dtype=np.int64))
        ids = np.concatenate(ids, axis=-1)  # [T, heads]
        offsets = np.concatenate(([0], np.cumsum(sizes[:-1])))
        rows = ids + offsets

        table = self.ckpt.memmap(p + ".multi_head_embedding.embedding.weight", np.uint8)
        tscale = self.ckpt.memmap(p + ".multi_head_embedding.embedding.weight_scale", np.float32)
        flat = rows.reshape(-1)
        q8 = torch.from_numpy(np.ascontiguousarray(table[flat])).to(self.dev).view(torch.float8_e4m3fn).float()
        sc = torch.from_numpy(np.ascontiguousarray(tscale[flat])).to(self.dev)
        emb = (q8 * sc[:, None]).to(self.dt).reshape(T, len(sizes), -1)
        if c["engram_fp8_embedding"]:
            fmax = torch.finfo(torch.float8_e4m3fn).max
            scale = emb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fmax
            emb = (emb / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn).to(emb.dtype) * scale
        emb = torch.stack([self.torch_rms(emb[:, i], self.w(f"{p}.lookup_per_head_norms.{i}.weight"))
                           for i in range(len(sizes))], dim=1).reshape(T, -1)

        D = self.D
        gates = []
        for ci in range(self.hc):
            key = self.lin(emb, self.w(f"{p}.key_projs.{ci}.weight"), self.w(f"{p}.key_projs.{ci}.bias"))
            key = self.torch_rms(key, self.w(f"{p}.norm_key.{ci}.weight"))
            query = self.torch_rms(lanes[:, ci], self.w(f"{p}.norm_query.{ci}.weight"))
            g = (key * query).sum(-1) * D ** -0.5
            g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
            gates.append(g.sigmoid())
        gates = torch.stack(gates, dim=1)[..., None]
        value = self.lin(emb, self.w(p + ".value_proj.weight"), self.w(p + ".value_proj.bias"))
        main = self.w(p + ".unified_pre_gamma") * F.rms_norm(value, (D,), None, eps=TORCH_RMS_EPS)
        branch = torch.stack([self.w(f"{p}.unified_per_hc_scales.{ci}") * main for ci in range(self.hc)], dim=1)
        out = gates * main[:, None] + branch
        keep = torch.ones(T, dtype=torch.bool, device=self.dev)
        for t in (pad, self.cfg.get("image_token_id"), self.cfg.get("video_token_id")):
            if t is not None:
                keep &= tokens.to(self.dev) != t
        return out.masked_fill(~keep[:, None, None], 0.0)

    def layer(self, lanes, tokens, i, positions):
        c = self.cfg
        p = f"model.layers.{i}"
        table = f"model.layers.{i}.engram_layer.multi_head_embedding.embedding.weight"
        if i + 1 in (c.get("engram_layer_ids") or []):
            if self.ckpt.has(table):
                lanes = lanes + self.engram(lanes, tokens, i)
            elif not getattr(self, f"_warned{i}", False):
                setattr(self, f"_warned{i}", True)
                print(f"WARNING: {table} is missing; layer {i} runs without its Engram memory", flush=True)
        post, h = self.hyper(lanes, p + ".hc_attn_layer")
        h = self.rms(h, self.w(p + ".input_layernorm.weight"))
        y = self.attention(h, p, self.layer_types[i], positions)
        lanes = (post[..., None] * y[:, None].float() + lanes.float()).to(self.dt)
        post, h = self.hyper(lanes, p + ".hc_mlp_layer")
        h = self.rms(h, self.w(p + ".post_attention_layernorm.weight"))
        if c["mlp_layer_types"][i] == "sparse":
            y = self.moe(h, p + ".mlp")
        else:
            y = self.rms(self.mlp(h, p + ".mlp"), self.w(p + ".mlp.post_fc2_norm.weight"))
        return (post[..., None] * y[:, None].float() + lanes.float()).to(self.dt)

    def forward(self, sequences):
        """Final hidden states (after the head's norm) of each token sequence, layer-major."""
        lanes, positions = [], []
        for tokens in sequences:
            T = tokens.shape[0]
            positions.append(torch.arange(T, device=self.dev))
            emb = self.ckpt.raw("model.embed_tokens.weight")[tokens].to(self.dev).to(self.dt)
            lanes.append(emb[:, None].expand(T, self.hc, self.D).contiguous().cpu())
        for i in range(self.cfg["num_hidden_layers"]):
            t0 = time.time()
            for j, tokens in enumerate(sequences):
                x = self.layer(lanes[j].to(self.dev), tokens, i, positions[j])
                lanes[j] = x.cpu()
                if i in self.dump_layers and self.dump_dir:
                    d = self.dump_dir / f"seq{j}"
                    d.mkdir(parents=True, exist_ok=True)
                    np.save(d / f"lanes-{i:02d}.npy", x.float().cpu().numpy())
            self.memo.clear()
            torch.cuda.empty_cache()
            print(f"layer {i:2d} {self.layer_types[i]:8s} {time.time() - t0:5.1f}s", flush=True)
        out = []
        for x in lanes:
            h = self.hyper(x.to(self.dev), "model.hc_head", head=True)
            out.append(self.rms(h, self.w("model.norm.weight")))
        self.memo.clear()
        return out


def score(args):
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    root = pathlib.Path(args.model)
    cfg_all = json.loads((root / "config.json").read_text())
    cfg = dict(cfg_all["text_config"])
    for k in ("image_token_id", "video_token_id"):
        cfg[k] = cfg_all.get(k)
    ckpt = Checkpoint(root, torch.device("cuda"))
    fixtures = [pathlib.Path(f) for f in args.fixture]
    seqs, prompt_lens = [], []
    for f in fixtures:
        seq = json.loads((f / "sequence.json").read_text())
        tokens = torch.tensor(seq["tokens"], dtype=torch.long)
        if args.limit:
            tokens = tokens[:args.limit]
        seqs.append(tokens)
        prompt_lens.append(min(seq["prompt_len"], len(tokens)))
    ref = Reference(ckpt, cfg, dtype, args.dump_layers, pathlib.Path(args.dump_dir) if args.dump_dir else None)
    t0 = time.time()
    with torch.no_grad():
        outs = ref.forward(seqs)
        head = ckpt.get("lm_head.weight", torch.float32)
        for f, tokens, out, prompt_len in zip(fixtures, seqs, outs, prompt_lens):
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
            meta = {"engine": "quench6k-hy4-ref", "format": "oracle.distributions/1", "mode": "teacher_forced",
                    "positions": P, "prompt_len": prompt_len, "top_k": args.top_k, "dtype": args.dtype,
                    "seconds": round(time.time() - t0, 1)}
            out_path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
            nll = (lse - target).mean().item()
            print(f"{f.name}: {P} positions, mean NLL {nll:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True, nargs="+", help="fixture directories (each with sequence.json)")
    ap.add_argument("--out", required=True, help="output directory; writes <fixture><suffix>.{bin,json}")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N tokens")
    ap.add_argument("--dump-layers", type=int, nargs="*", default=[])
    ap.add_argument("--dump-dir")
    score(ap.parse_args())


if __name__ == "__main__":
    main()
