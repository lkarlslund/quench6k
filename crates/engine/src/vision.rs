//! Vision tower (SPEC §14): patch embedding, interpolated learned position embedding, 27 ViT blocks
//! with 2D RoPE and the 2x2 patch merger into the text model's 2560-wide embedding space.
//!
//! Input patches arrive in spatial-merge-block order (block row, block column, in-block row, in-block
//! column per frame), each patch flattened as (channel, frame, 16, 16) — the layout of the HF
//! Qwen2-VL image processor — so that every 4 consecutive patches form one merged token.

use anyhow::{Result, ensure};
use squash_cuda::kernels::{self as k, VitAct, WeightFormat};
use squash_cuda::{DeviceBuffer, Stream};

use crate::checkpoint::Checkpoint;
use crate::loader::{self, DeviceWeights, LoadOptions};

pub const PATCH: usize = 16;
pub const TEMPORAL: usize = 2;
pub const MERGE: usize = 2;
pub const PATCH_DIM: usize = 3 * TEMPORAL * PATCH * PATCH;
const W: usize = 1152;
const HEAD_DIM: usize = 72;
const FF: usize = 4304;
/// FF rounded up to the GEMM's K tile (fc2's weight is zero-padded to it).
const FF_PAD: usize = 4320;
const DEPTH: usize = 27;
const OUT: usize = 2560;
const GRID_SIDE: usize = 48;
const ROPE_THETA: f64 = 10000.0;

pub struct VisionModel {
    weights: DeviceWeights,
    fc2: Vec<DeviceBuffer>,
    pos_embed: Vec<f32>,
}

/// One image (t = 1) or video clip: patches in merge-block order and the patch grid (t, h, w).
pub struct VisionInput {
    pub pixels: Vec<f32>,
    pub grid: [usize; 3],
}

impl VisionInput {
    pub fn patches(&self) -> usize {
        self.grid.iter().product()
    }

    /// Merged tokens the image contributes to the text sequence.
    pub fn tokens(&self) -> usize {
        self.patches() / (MERGE * MERGE)
    }
}

/// (row, column) of each patch of one frame, in merge-block order.
fn patch_coords(h: usize, w: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(h * w);
    for br in 0..h / MERGE {
        for bc in 0..w / MERGE {
            for ir in 0..MERGE {
                for ic in 0..MERGE {
                    out.push((br * MERGE + ir, bc * MERGE + ic));
                }
            }
        }
    }
    out
}

/// Bilinear (align_corners) taps of target index `i` on an axis of `size` into the 48-entry grid.
fn taps(i: usize, size: usize) -> [(usize, f32); 2] {
    let src = i as f32 * (GRID_SIDE - 1) as f32 / (size.max(2) - 1) as f32;
    let src = if size == 1 { 0.0 } else { src };
    let f = src.floor();
    let mut out = [(0, 0.0); 2];
    for (o, slot) in out.iter_mut().enumerate() {
        let raw = f as i64 + o as i64;
        let tap = raw.clamp(0, GRID_SIDE as i64 - 1) as usize;
        let dist = (src - f - o as f32).abs();
        *slot = (tap, (1.0 - dist).max(0.0));
    }
    out
}

impl VisionModel {
    pub fn load(ckpt: &Checkpoint, opts: LoadOptions, s: &Stream) -> Result<Self> {
        let (weights, _) = loader::upload(ckpt, crate::selection::vision, opts)?;
        let mut fc2 = Vec::with_capacity(DEPTH);
        for b in 0..DEPTH {
            let t = weights.get(&format!("model.visual.blocks.{b}.mlp.linear_fc2.weight"))?;
            ensure!(t.shape == [W, FF], "unexpected fc2 shape {:?}", t.shape);
            let buf = DeviceBuffer::new(W * FF_PAD * 2)?;
            k::pad_cols(s, t.addr, W, FF, buf.addr(), FF_PAD)?;
            fc2.push(buf);
        }
        let pe = weights.get("model.visual.pos_embed.weight")?;
        let raw: Vec<u16> = s.download(pe.addr, GRID_SIDE * GRID_SIDE * W)?;
        let pos_embed = raw.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
        s.synchronize()?;
        Ok(Self { weights, fc2, pos_embed })
    }

    fn w(&self, name: &str) -> Result<u64> {
        Ok(self.weights.get(&format!("model.visual.{name}"))?.addr)
    }

    /// Learned position embedding resampled to the grid ([patches, 1152]) and the 2D RoPE tables
    /// (cos, sin [patches, 72]).
    fn tables(&self, grid: [usize; 3]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let [t, h, w] = grid;
        let coords = patch_coords(h, w);
        let n = t * h * w;
        let mut pos = vec![0f32; n * W];
        let (mut cos, mut sin) = (vec![0f32; n * HEAD_DIM], vec![0f32; n * HEAD_DIM]);
        let inv: Vec<f64> = (0..HEAD_DIM / 4).map(|j| 1.0 / ROPE_THETA.powf((2 * j) as f64 / (HEAD_DIM / 2) as f64)).collect();
        for i in 0..n {
            let (r, c) = coords[i % (h * w)];
            let row = &mut pos[i * W..(i + 1) * W];
            for (hy, wy) in taps(r, h) {
                for (hx, wx) in taps(c, w) {
                    let wt = wy * wx;
                    let src = &self.pos_embed[(hy * GRID_SIDE + hx) * W..(hy * GRID_SIDE + hx + 1) * W];
                    for (o, &v) in row.iter_mut().zip(src) {
                        *o += wt * v;
                    }
                }
            }
            for d in 0..HEAD_DIM {
                let j = d % (HEAD_DIM / 2);
                let angle = if j < HEAD_DIM / 4 { r as f64 * inv[j] } else { c as f64 * inv[j - HEAD_DIM / 4] };
                // FP32 angle, as the reference computes positions x inv_freq in FP32.
                let a = (angle as f32) as f64;
                cos[i * HEAD_DIM + d] = a.cos() as f32;
                sin[i * HEAD_DIM + d] = a.sin() as f32;
            }
        }
        (pos, cos, sin)
    }

    /// Encodes one image or clip into its merged token embeddings, FP32 [tokens, 2560] on the device.
    pub fn encode(&self, s: &Stream, input: &VisionInput) -> Result<DeviceBuffer> {
        let [_, h, w] = input.grid;
        ensure!(h % MERGE == 0 && w % MERGE == 0, "patch grid {h}x{w} is not a multiple of the merge size");
        let n = input.patches();
        ensure!(input.pixels.len() == n * PATCH_DIM, "{} pixel values for {n} patches", input.pixels.len());
        let (pos, cos, sin) = self.tables(input.grid);
        let f32buf = |count: usize| DeviceBuffer::new(count * 4);
        let pix = f32buf(n * PATCH_DIM)?;
        s.upload(&pix, &input.pixels)?;
        let (pos_d, cos_d, sin_d) = (f32buf(n * W)?, f32buf(n * HEAD_DIM)?, f32buf(n * HEAD_DIM)?);
        s.upload(&pos_d, &pos)?;
        s.upload(&cos_d, &cos)?;
        s.upload(&sin_d, &sin)?;
        let xb = DeviceBuffer::new(n * PATCH_DIM.max(FF_PAD) * 2)?;
        let mb = DeviceBuffer::zeroed(n * FF_PAD * 2)?;
        let hbuf = f32buf(n * W)?;
        let y = f32buf(n * 3 * W.max(FF))?;
        let (q, kk, v) = (f32buf(n * W)?, f32buf(n * W)?, f32buf(n * W)?);
        let att = DeviceBuffer::new(n * W * 2)?;
        let gemm = |x: u64, wt: u64, out: u64, m: usize, nn: usize, kd: usize| k::gemm_f32(s, x, kd, WeightFormat::Bf16, wt, kd, 0, out, nn, m, nn, kd, 1.0, false);

        // Patch embedding (Conv3d with stride = kernel, i.e. a linear map of each patch) + position.
        k::to_bf16(s, pix.addr(), n * PATCH_DIM, xb.addr())?;
        gemm(xb.addr(), self.w("patch_embed.proj.weight")?, y.addr(), n, W, PATCH_DIM)?;
        k::vit_residual(s, hbuf.addr(), y.addr(), self.w("patch_embed.proj.bias")?, pos_d.addr(), n, W, true)?;

        let seg = h * w;
        for b in 0..DEPTH {
            let p = |name: &str| self.w(&format!("blocks.{b}.{name}"));
            k::vit_layernorm(s, hbuf.addr(), n, W, p("norm1.weight")?, p("norm1.bias")?, xb.addr())?;
            gemm(xb.addr(), p("attn.qkv.weight")?, y.addr(), n, 3 * W, W)?;
            k::vit_qkv_rope(s, y.addr(), p("attn.qkv.bias")?, cos_d.addr(), sin_d.addr(), n, q.addr(), kk.addr(), v.addr())?;
            k::vit_attention(s, q.addr(), kk.addr(), v.addr(), n, seg, att.addr())?;
            gemm(att.addr(), p("attn.proj.weight")?, y.addr(), n, W, W)?;
            k::vit_residual(s, hbuf.addr(), y.addr(), p("attn.proj.bias")?, 0, n, W, false)?;
            k::vit_layernorm(s, hbuf.addr(), n, W, p("norm2.weight")?, p("norm2.bias")?, xb.addr())?;
            gemm(xb.addr(), p("mlp.linear_fc1.weight")?, y.addr(), n, FF, W)?;
            k::vit_bias_act(s, y.addr(), n, FF, p("mlp.linear_fc1.bias")?, VitAct::GeluTanh, mb.addr(), FF_PAD)?;
            gemm(mb.addr(), self.fc2[b].addr(), y.addr(), n, W, FF_PAD)?;
            k::vit_residual(s, hbuf.addr(), y.addr(), p("mlp.linear_fc2.bias")?, 0, n, W, false)?;
        }

        // Merger: LayerNorm per patch, then 4 consecutive patches form one 4608-wide row.
        let tokens = n / (MERGE * MERGE);
        let mw = W * MERGE * MERGE;
        k::vit_layernorm(s, hbuf.addr(), n, W, self.w("merger.norm.weight")?, self.w("merger.norm.bias")?, xb.addr())?;
        gemm(xb.addr(), self.w("merger.linear_fc1.weight")?, y.addr(), tokens, mw, mw)?;
        k::vit_bias_act(s, y.addr(), tokens, mw, self.w("merger.linear_fc1.bias")?, VitAct::GeluErf, mb.addr(), mw)?;
        gemm(mb.addr(), self.w("merger.linear_fc2.weight")?, y.addr(), tokens, OUT, mw)?;
        let out = f32buf(tokens * OUT)?;
        k::vit_residual(s, out.addr(), y.addr(), self.w("merger.linear_fc2.bias")?, 0, tokens, OUT, true)?;
        s.synchronize()?;
        Ok(out)
    }
}
