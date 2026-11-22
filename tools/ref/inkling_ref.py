#!/usr/bin/env python3
"""Inkling 2 Mini (inkling2_loop) text reference: scores token sequences teacher-forced with the
transformers implementation of the decoder layers (transformers >= 5.24), streaming one layer's weights
at a time to the GPU, so it runs beside a loaded engine.

The decoder layers, their projections, convolutions, attention residuals, message gates and MoE are the
transformers modules; this script only drives them (the loop over layers, passes and block states, as
Inkling2LoopTextModel.forward) and replaces the attention core with an equivalent one chunked over queries
(the transformers eager path materializes [heads, T, T] scores, too large for long fixtures). The experts
are dequantized from the checkpoint's NVFP4 / MXFP8 as transformers does (BF16 weights, BF16 activations).

`--dtype bf16` (default) rounds where the transformers model does; `--dtype fp32` runs the same modules in
FP32 and gives the noise floor. Outputs the oracle.distributions/1 format of oracle/fixtures (see
oracle/README.md), and can dump the hidden states after chosen layer visits (0 .. 47) for engine debugging.

    python -I tools/ref/inkling_ref.py --model $M --fixture out/inkling-fixtures/* --out out/inkling-ref --suffix=-bf16
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
from transformers.models.inkling2_loop import modeling_inkling2_loop as mod
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
PREFIX = "model.llm."


class Checkpoint:
    def __init__(self, root: pathlib.Path):
        self.where, self.handles = {}, {}
        for path in sorted(root.glob("model*.safetensors")):
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            header.pop("__metadata__", None)
            for name in header:
                self.where[name] = path

    def has(self, name):
        return name in self.where

    def raw(self, name):
        path = self.where[name]
        if path not in self.handles:
            self.handles[path] = safe_open(str(path), framework="pt", device="cpu")
        return self.handles[path].get_tensor(name)


def dequant_experts(ck, base, dev):
    """BF16 [E, rows, cols] of an expert stack (NVFP4 with amax, or MXFP8), as transformers dequantizes."""
    w = ck.raw(base).to(dev)
    sc = ck.raw(base + "_scale").to(dev)
    if ck.has(base + "_amax"):
        factor = ck.raw(base + "_amax").to(dev).float() / ck.raw(base + "_outer_scale_denominator").to(dev).float()
        e2m1 = E2M1.to(dev)
        byte = torch.arange(256, device=dev)
        lut = torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)
        E, rows = w.shape[:2]
        out = torch.empty(E, rows, w.shape[2] * 2, dtype=torch.bfloat16, device=dev)
        for e in range(0, E, 8):
            v = lut[w[e:e + 8].long()].view(*sc[e:e + 8].shape, 16) * sc[e:e + 8].float()[..., None]
            out[e:e + 8] = (v * factor[e:e + 8, None, None, None]).reshape(-1, rows, out.shape[2]).bfloat16()
        return out
    E, rows, cols = w.shape
    out = torch.empty(E, rows, cols, dtype=torch.bfloat16, device=dev)
    for e in range(0, E, 8):
        v = w[e:e + 8].float().view(*sc[e:e + 8].shape, 32) * torch.exp2(sc[e:e + 8].float() - 127)[..., None]
        out[e:e + 8] = v.reshape(-1, rows, cols).bfloat16()
    return out


def chunked_attention(self, hidden_states, attention_mask=None, relative_distance=None, conv_mask=None, cache=None,
                      key_hidden_states=None, value_hidden_states=None, cache_offset=0, **kwargs):
    """Inkling2LoopSlidingAttention / FullAttention.forward without a cache, with the scores of up to
    CHUNK queries at a time (same roundings as eager_attention_forward: BF16 scores, bias and mask added in
    the activation dtype, FP32 softmax over the keys and the sink)."""
    pos = kwargs["position_ids"]
    kh = hidden_states if key_hidden_states is None else key_hidden_states
    vh = hidden_states if value_hidden_states is None else value_hidden_states
    q = self.q_proj(hidden_states, pos)
    k = mod.repeat_kv(self.k_proj(kh), self.num_key_value_groups)
    v = mod.repeat_kv(self.v_proj(vh), self.num_key_value_groups)
    rb = self.relative_position_bias
    B, T = hidden_states.shape[:2]
    rs = rb.norm(rb.linear(hidden_states).view(B, T, rb.num_heads, -1), pos)
    rel = (rs @ rb.proj).transpose(1, 2)  # [B, H, T, extent]
    sinks = self.sink_proj(hidden_states).transpose(1, 2).unsqueeze(-1)
    extent = rb.rel_extent
    dt = q.dtype
    out = torch.empty_like(q)
    window = self.sliding_window
    for q0 in range(0, T, CHUNK):
        q1 = min(T, q0 + CHUNK)
        lo = max(0, q0 - window + 1) if window else 0
        dist = (torch.arange(q0, q1, device=q.device)[:, None] - torch.arange(lo, q1, device=q.device)[None, :])
        s = torch.matmul(q[:, :, q0:q1], k[:, :, lo:q1].transpose(2, 3)) * self.scaling
        idx = dist.clamp(0, extent - 1).expand(B, rel.shape[1], -1, -1)
        bias = rel[:, :, q0:q1].gather(-1, idx).masked_fill((dist < 0) | (dist >= extent), 0.0)
        s = s + bias
        masked = dist < 0
        if window:
            masked = masked | (dist >= window)
        s = s + torch.zeros(dist.shape, dtype=dt, device=q.device).masked_fill(masked, torch.finfo(dt).min)
        p = F.softmax(torch.cat([s, sinks[:, :, q0:q1]], dim=-1).float(), dim=-1)[..., :-1].to(dt)
        out[:, :, q0:q1] = torch.matmul(p, v[:, :, lo:q1])
    out = out.transpose(1, 2).reshape(B, T, -1).contiguous()
    return self.o_proj(out, hidden_states), None


CHUNK = 1024
mod.Inkling2LoopSlidingAttention.forward = chunked_attention
mod.Inkling2LoopFullAttention.forward = chunked_attention


class Reference:
    def __init__(self, root, dtype, dump_layers, dump_dir):
        self.root, self.dt = root, dtype
        self.dev = torch.device("cuda")
        self.ck = Checkpoint(root)
        tc = json.loads((root / "config.json").read_text())["text_config"]
        tc = {k: v for k, v in tc.items() if k not in ("model_type", "torch_dtype", "quantization_config")}
        self.cfg = Inkling2LoopTextConfig(**tc)
        self.cfg._attn_implementation = "eager"
        try:
            self.cfg._experts_implementation = "eager"
        except AttributeError:
            pass
        c = self.cfg
        self.loop = (c.loop_layer_index_start, c.loop_layer_index_end, c.loop_num_passes)
        self.dump_layers, self.dump_dir = set(dump_layers), dump_dir

    def w(self, name):
        return self.ck.raw(PREFIX + name)

    def layer(self, i):
        """Decoder layer i on the GPU with the checkpoint's weights."""
        c = self.cfg
        sliding = c.layer_types[i] == "hybrid_sliding"
        looped = self.loop[0] <= i < self.loop[1]
        cls = {(False, False): mod.Inkling2LoopFullDecoderLayer, (True, False): mod.Inkling2LoopSlidingDecoderLayer,
               (False, True): mod.Inkling2LoopFullLoopDecoderLayer, (True, True): mod.Inkling2LoopSlidingLoopDecoderLayer}
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            layer = cls[(sliding, looped)](c, i)
        torch.set_default_dtype(prev)
        layer = layer.to_empty(device=self.dev)
        p = f"layers.{i}."
        a = p + "attn."
        names = {
            "self_attn.q_proj.linear.weight": a + "wq_du.weight",
            "self_attn.k_proj.linear.weight": a + "wk_dv.weight",
            "self_attn.v_proj.linear.weight": a + "wv_dv.weight",
            "self_attn.q_proj.norm.scale": a + "q_l2_clip.scale",
            "self_attn.relative_position_bias.norm.scale": a + "rel_l2_clip.scale",
            "self_attn.relative_position_bias.proj": a + "rel_logits_proj.proj",
            "self_attn.relative_position_bias.linear.weight": a + "wr_du.weight",
            "self_attn.o_proj.linear.weight": a + "wo_ud.weight",
            "self_attn.o_proj.gate_proj.weight": a + "gate_attn.weight",
            "self_attn.sink_proj.weight": a + "attn_sink_proj.weight",
            "post_attention_convolution.conv1d.weight": p + "attn_sconv.weight",
            "post_mlp_convolution.conv1d.weight": p + "mlp_sconv.weight",
            "post_attention_layernorm.weight": p + "mlp_norm.weight",
            "mlp.gate.weight": p + "mlp.gate.weight",
            "mlp.gate.e_score_correction_bias": p + "mlp.gate.bias",
            "mlp_input.query": f"attn_res.mlp_queries.{i}",
        }
        if i > 0:
            names["attention_input.query"] = f"attn_res.attn_queries.{i - 1}"
        if sliding:
            names["input_layernorm.weight"] = p + "attn_norm.weight"
            for x in "kv":
                names[f"self_attn.{x}_proj.sconv.conv1d.weight"] = a + f"{x}_sconv.weight"
        else:
            for x in "qkv":
                names[f"{x}_input_layernorm.weight"] = p + f"{x}_input_norm.weight"
            for x in "kv":
                names[f"self_attn.{x}_proj.sconv.static_weight"] = a + f"{x}_sconv.static_w"
                names[f"self_attn.{x}_proj.sconv.weight_proj.weight"] = a + f"{x}_sconv.weight_proj"
        if looped:
            for ours, theirs in (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
                for x, y in (("down_proj", "down_weight"), ("forget_gate_proj", "forget_weight"),
                             ("message_gate_proj", "gain_weight")):
                    names[f"{ours}_message_gate.{x}.weight"] = p + f"controllers.{theirs}.{y}"
        params = dict(layer.named_parameters())
        params.update(dict(layer.named_buffers()))
        with torch.no_grad():
            for ours, theirs in names.items():
                t = params[ours]
                t.copy_(self.w(theirs).to(self.dev).to(t.dtype).view(t.shape))
            gu = dequant_experts(self.ck, PREFIX + p + "mlp.experts.w13_weight", self.dev)
            E, rows, cols = gu.shape
            gu = gu.view(E, rows // 2, 2, cols).transpose(1, 2).reshape(E, rows, cols)  # [gate; up]
            params["mlp.experts.gate_up_proj"].copy_(gu)
            del gu
            params["mlp.experts.down_proj"].copy_(dequant_experts(self.ck, PREFIX + p + "mlp.experts.w2_weight", self.dev))
            missing = set(params) - set(names) - {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
            assert not missing, f"layer {i}: parameters without weights: {sorted(missing)}"
        if self.dt == torch.float32:
            layer = layer.float()
        return layer

    def rms(self, x, w):
        """Inkling2LoopRMSNorm: FP32 normalization, weight product, back to the input dtype."""
        xf = x.float()
        return (w.to(self.dev).float() * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.cfg.rms_norm_eps))).to(x.dtype)

    def block_residual(self, query):
        m = mod.Inkling2LoopBlockAttentionResidual(self.cfg, num_queries=1).to(self.dev)
        with torch.no_grad():
            m.query.copy_(query.to(self.dev).float())
        return m

    def forward(self, sequences):
        c = self.cfg
        dev, dt = self.dev, self.dt
        ls, le, passes = self.loop
        emb_w = self.w("embed.weight")
        norm_w = self.w("embed_norm.weight")
        st = []
        for tokens in sequences:
            x = self.rms(emb_w[tokens].to(dev).to(dt), norm_w)
            T = tokens.shape[0]
            st.append({"h": x[None], "blocks": x.new_zeros(1, T, 0, c.hidden_size),
                       "pos": torch.arange(T, device=dev)[None]})
        mq = torch.cat([self.w("middle_cycle_cross_pass_queries").view(-1, 2, c.hidden_size),
                        self.w("middle_cycle_cross_pass_kv_queries").view(-1, 2, c.hidden_size)], dim=1)
        message_q = mq[:, [0, 2, 3, 1]]  # per loop layer: query, key, value, mlp
        msg = mod.Inkling2LoopMessageAttentionResidual(c).to(dev)
        with torch.no_grad():
            msg.query.copy_(message_q.to(dev).float())
        order = list(range(ls)) + [i for p in range(passes) for i in range(ls, le)] + list(range(le, c.num_hidden_layers))
        visit_pass = [0] * ls + [p for p in range(passes) for _ in range(ls, le)] + [0] * (c.num_hidden_layers - le)
        cur = None
        for visit, (i, pss) in enumerate(zip(order, visit_pass)):
            t0 = time.time()
            if cur is None or cur[0] != i:
                cur = (i, self.layer(i))
            layer = cur[1]
            looped = ls <= i < le
            for j, s in enumerate(st):
                if i == ls:
                    if pss == 0:
                        s["pre"] = (s["h"], s["blocks"], s["blocks"].shape[-2])
                    else:
                        npre = s["pre"][2]
                        s["prev"] = (s["blocks"][..., npre + 1:, :], s["h"])
                        s["h"], s["blocks"] = s["pre"][0], s["pre"][1]
                kw = dict(attention_mask=None, relative_distance=None, conv_mask=None, cache=None, position_ids=s["pos"])
                if looped:
                    messages = None
                    if pss > 0:
                        messages = msg(s["prev"][0], s["prev"][1], i - ls)
                    new = layer(s["h"], s["blocks"], messages, **kw)
                else:
                    new = layer(s["h"], s["blocks"], **kw)
                if layer.is_block_boundary:
                    s["blocks"] = torch.cat([s["blocks"], s["h"].unsqueeze(-2)], dim=-2)
                s["h"] = new
                if visit in self.dump_layers and self.dump_dir:
                    d = self.dump_dir / f"seq{j}"
                    d.mkdir(parents=True, exist_ok=True)
                    np.save(d / f"hidden-{visit:02d}.npy", new[0].float().cpu().numpy())
            if visit + 1 == len(order) or order[visit + 1] != i:
                cur = None
                torch.cuda.empty_cache()
            print(f"visit {visit:2d} layer {i:2d} pass {pss} {time.time() - t0:5.1f}s", flush=True)
        final = self.block_residual(self.w("attn_res.output_query"))
        out = []
        for s in st:
            h = final(s["blocks"], s["h"])[0]
            out.append(self.rms(h, self.w("norm.weight")).to(dt)[0])
        return out


def score(args):
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    root = pathlib.Path(args.model)
    fixtures = [pathlib.Path(f) for f in args.fixture]
    seqs, prompt_lens = [], []
    for f in fixtures:
        seq = json.loads((f / "sequence.json").read_text())
        tokens = torch.tensor(seq["tokens"], dtype=torch.long)
        if args.limit:
            tokens = tokens[:args.limit]
        seqs.append(tokens)
        prompt_lens.append(min(seq["prompt_len"], len(tokens)))
    ref = Reference(root, dtype, args.dump_layers, pathlib.Path(args.dump_dir) if args.dump_dir else None)
    c = ref.cfg
    t0 = time.time()
    with torch.no_grad():
        outs = ref.forward(seqs)
        head = ref.w("unembed.weight")[:c.unpadded_vocab_size].to(ref.dev).to(dtype)
        cap = c.final_logit_softcapping
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
                logits = (out[s:e] @ head.T).float()
                logits = (torch.tanh(logits / cap) * cap).double()
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
            meta = {"engine": "quench6k-inkling-ref", "format": "oracle.distributions/1", "mode": "teacher_forced",
                    "positions": P, "prompt_len": prompt_len, "top_k": args.top_k, "dtype": args.dtype,
                    "seconds": round(time.time() - t0, 1)}
            out_path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
            nll = (lse - target).mean().item()
            print(f"{f.name}: {P} positions, mean NLL {nll:.4f}")


def main():
    global CHUNK
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True, nargs="+", help="fixture directories (each with sequence.json)")
    ap.add_argument("--out", required=True, help="output directory; writes <fixture><suffix>.{bin,json}")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N tokens")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="queries per attention chunk")
    ap.add_argument("--dump-layers", type=int, nargs="*", default=[], help="layer visits (0 .. 47) to dump")
    ap.add_argument("--dump-dir")
    args = ap.parse_args()
    CHUNK = args.chunk
    score(args)


if __name__ == "__main__":
    main()
