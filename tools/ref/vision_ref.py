#!/usr/bin/env python3
"""HF transformers reference for the vision tower (qwen4_exp): preprocesses an image with the
checkpoint's image processor and runs Qwen4ExpVisionModel in FP32 on the checkpoint's BF16 weights.

Writes to --out:
  pixels.f32    FP32 [patches, 1536] processor output (what the tower consumes)
  embeds.f32    FP32 [patches / 4, 2560] merger output (what replaces the image tokens)
  meta.json     {"grid_thw": [t, h, w], "patches": n, "tokens": n / 4, "image": path}
"""
import argparse
import json
import pathlib

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=pathlib.Path)
    ap.add_argument("--image", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--max-pixels", type=int, default=None, help="cap the resized image area")
    ap.add_argument("--bf16", action="store_true", help="also run the tower in BF16 (writes embeds_bf16.f32)")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoImageProcessor
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    config = AutoConfig.from_pretrained(args.model)
    proc = AutoImageProcessor.from_pretrained(args.model)
    kw = {}
    if args.max_pixels:
        kw["size"] = {"shortest_edge": proc.size["shortest_edge"], "longest_edge": args.max_pixels}
    img = Image.open(args.image).convert("RGB")
    batch = proc(images=[img], return_tensors="pt", **kw)
    pixels, grid = batch["pixel_values"].float(), batch["image_grid_thw"]

    vc = config.vision_config
    vc._attn_implementation = "eager"
    model = Qwen4ExpVisionModel(vc).float()
    state = {}
    with safe_open(str(args.model / "model-tail.safetensors"), "pt") as f:
        for name in f.keys():
            if name.startswith("model.visual."):
                state[name[len("model.visual."):]] = f.get_tensor(name).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all("inv_freq" in m for m in missing), missing
    model = model.cuda().eval()
    with torch.no_grad():
        out = model(pixels.cuda(), grid_thw=grid.cuda(), return_dict=True)
    embeds = out.pooler_output.float().cpu().numpy()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.bf16:
        mb = model.to(torch.bfloat16)
        with torch.no_grad():
            eb = mb(pixels.cuda().to(torch.bfloat16), grid_thw=grid.cuda(), return_dict=True).pooler_output.float().cpu().numpy()
        eb.astype(np.float32).tofile(args.out / "embeds_bf16.f32")
        cos = (eb * embeds).sum(1) / np.linalg.norm(eb, axis=1) / np.linalg.norm(embeds, axis=1)
        print("HF BF16 vs FP32: |err| max %.5f mean %.6f; cosine min %.6f mean %.6f"
              % (np.abs(eb - embeds).max(), np.abs(eb - embeds).mean(), cos.min(), cos.mean()))
    pixels.numpy().astype(np.float32).tofile(args.out / "pixels.f32")
    embeds.astype(np.float32).tofile(args.out / "embeds.f32")
    meta = {"grid_thw": grid[0].tolist(), "patches": int(pixels.shape[0]), "tokens": int(embeds.shape[0]), "image": str(args.image)}
    (args.out / "meta.json").write_text(json.dumps(meta) + "\n")
    print(meta, "embeds", embeds.shape, "absmax", float(np.abs(embeds).max()))


if __name__ == "__main__":
    main()
