//! Vision tower (SPEC §14): patch embedding, interpolated learned position embedding, 27 ViT blocks
//! with 2D RoPE and the 2x2 patch merger into the text model's 2560-wide embedding space.
//!
//! Input patches arrive in spatial-merge-block order (block row, block column, in-block row, in-block
//! column per frame), each patch flattened as (channel, frame, 16, 16) — the layout of the HF
//! Qwen2-VL image processor — so that every 4 consecutive patches form one merged token.

use anyhow::{Result, ensure};
use quench6k_cuda::kernels::{self as k, VitAct, WeightFormat};
use quench6k_cuda::{DeviceBuffer, Stream};

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
        let pos_embed = raw
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect();
        s.synchronize()?;
        Ok(Self {
            weights,
            fc2,
            pos_embed,
        })
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
        let inv: Vec<f64> = (0..HEAD_DIM / 4)
            .map(|j| 1.0 / ROPE_THETA.powf((2 * j) as f64 / (HEAD_DIM / 2) as f64))
            .collect();
        for i in 0..n {
            let (r, c) = coords[i % (h * w)];
            let row = &mut pos[i * W..(i + 1) * W];
            for (hy, wy) in taps(r, h) {
                for (hx, wx) in taps(c, w) {
                    let wt = wy * wx;
                    let src =
                        &self.pos_embed[(hy * GRID_SIDE + hx) * W..(hy * GRID_SIDE + hx + 1) * W];
                    for (o, &v) in row.iter_mut().zip(src) {
                        *o += wt * v;
                    }
                }
            }
            for d in 0..HEAD_DIM {
                let j = d % (HEAD_DIM / 2);
                let angle = if j < HEAD_DIM / 4 {
                    r as f64 * inv[j]
                } else {
                    c as f64 * inv[j - HEAD_DIM / 4]
                };
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
        ensure!(
            h % MERGE == 0 && w % MERGE == 0,
            "patch grid {h}x{w} is not a multiple of the merge size"
        );
        let n = input.patches();
        ensure!(
            input.pixels.len() == n * PATCH_DIM,
            "{} pixel values for {n} patches",
            input.pixels.len()
        );
        let (pos, cos, sin) = self.tables(input.grid);
        // Only the residual stream, q / k / v, the attention output and the RoPE tables are per patch
        // (about 21 KB each: 350 MB for the largest default image); every row-wise step runs in blocks of
        // ROWS patches through fixed buffers. Each output row's arithmetic is the same either way.
        const ROWS: usize = 2048;
        let f32buf = |count: usize| DeviceBuffer::new(count * 4);
        let hbuf = f32buf(n * W)?;
        let (q, kk, v) = (f32buf(n * W)?, f32buf(n * W)?, f32buf(n * W)?);
        let att = DeviceBuffer::new(n * W * 2)?;
        let (cos_d, sin_d) = (f32buf(n * HEAD_DIM)?, f32buf(n * HEAD_DIM)?);
        s.upload(&cos_d, &cos)?;
        s.upload(&sin_d, &sin)?;
        let r = ROWS.min(n);
        let xb = DeviceBuffer::new(r * PATCH_DIM.max(W) * 2)?;
        // GEMM outputs: qkv [r, 3W], fc1 [r, FF], and smaller.
        let y = f32buf(r * (3 * W).max(FF))?;
        let mb = DeviceBuffer::zeroed(r * FF_PAD * 2)?;
        let gemm = |x: u64, wt: u64, out: u64, m: usize, nn: usize, kd: usize| {
            k::gemm_f32(
                s,
                x,
                kd,
                WeightFormat::Bf16,
                wt,
                kd,
                0,
                out,
                nn,
                m,
                nn,
                kd,
                1.0,
                false,
            )
        };
        let blocks = || (0..n).step_by(ROWS).map(move |r0| (r0, ROWS.min(n - r0)));
        let f32_at =
            |b: &DeviceBuffer, row: usize, width: usize| b.addr() + (row * width * 4) as u64;

        // Patch embedding (Conv3d with stride = kernel, i.e. a linear map of each patch) + position.
        {
            let (pix, pos_d) = (f32buf(r * PATCH_DIM)?, f32buf(r * W)?);
            for (r0, m) in blocks() {
                s.upload(&pix, &input.pixels[r0 * PATCH_DIM..(r0 + m) * PATCH_DIM])?;
                s.upload(&pos_d, &pos[r0 * W..(r0 + m) * W])?;
                k::to_bf16(s, pix.addr(), m * PATCH_DIM, xb.addr())?;
                gemm(
                    xb.addr(),
                    self.w("patch_embed.proj.weight")?,
                    y.addr(),
                    m,
                    W,
                    PATCH_DIM,
                )?;
                k::vit_residual(
                    s,
                    f32_at(&hbuf, r0, W),
                    y.addr(),
                    self.w("patch_embed.proj.bias")?,
                    pos_d.addr(),
                    m,
                    W,
                    true,
                )?;
            }
            // The uploads read host memory asynchronously: finish before the buffers are freed.
            s.synchronize()?;
        }

        let seg = h * w;
        for b in 0..DEPTH {
            let p = |name: &str| self.w(&format!("blocks.{b}.{name}"));
            for (r0, m) in blocks() {
                k::vit_layernorm(
                    s,
                    f32_at(&hbuf, r0, W),
                    m,
                    W,
                    p("norm1.weight")?,
                    p("norm1.bias")?,
                    xb.addr(),
                )?;
                gemm(xb.addr(), p("attn.qkv.weight")?, y.addr(), m, 3 * W, W)?;
                k::vit_qkv_rope(
                    s,
                    y.addr(),
                    p("attn.qkv.bias")?,
                    f32_at(&cos_d, r0, HEAD_DIM),
                    f32_at(&sin_d, r0, HEAD_DIM),
                    m,
                    f32_at(&q, r0, W),
                    f32_at(&kk, r0, W),
                    f32_at(&v, r0, W),
                )?;
            }
            k::vit_attention(s, q.addr(), kk.addr(), v.addr(), n, seg, att.addr())?;
            for (r0, m) in blocks() {
                let hrow = f32_at(&hbuf, r0, W);
                gemm(
                    att.addr() + (r0 * W * 2) as u64,
                    p("attn.proj.weight")?,
                    y.addr(),
                    m,
                    W,
                    W,
                )?;
                k::vit_residual(s, hrow, y.addr(), p("attn.proj.bias")?, 0, m, W, false)?;
                k::vit_layernorm(
                    s,
                    hrow,
                    m,
                    W,
                    p("norm2.weight")?,
                    p("norm2.bias")?,
                    xb.addr(),
                )?;
                gemm(xb.addr(), p("mlp.linear_fc1.weight")?, y.addr(), m, FF, W)?;
                k::vit_bias_act(
                    s,
                    y.addr(),
                    m,
                    FF,
                    p("mlp.linear_fc1.bias")?,
                    VitAct::GeluTanh,
                    mb.addr(),
                    FF_PAD,
                )?;
                gemm(mb.addr(), self.fc2[b].addr(), y.addr(), m, W, FF_PAD)?;
                k::vit_residual(s, hrow, y.addr(), p("mlp.linear_fc2.bias")?, 0, m, W, false)?;
            }
        }

        // Merger: LayerNorm per patch, then 4 consecutive patches form one 4608-wide row (blocks are
        // multiples of 4 patches).
        let tokens = n / (MERGE * MERGE);
        let mw = W * MERGE * MERGE;
        let out = f32buf(tokens * OUT)?;
        for (r0, m) in blocks() {
            let (t0, tm) = (r0 / (MERGE * MERGE), m / (MERGE * MERGE));
            k::vit_layernorm(
                s,
                f32_at(&hbuf, r0, W),
                m,
                W,
                self.w("merger.norm.weight")?,
                self.w("merger.norm.bias")?,
                xb.addr(),
            )?;
            gemm(
                xb.addr(),
                self.w("merger.linear_fc1.weight")?,
                y.addr(),
                tm,
                mw,
                mw,
            )?;
            k::vit_bias_act(
                s,
                y.addr(),
                tm,
                mw,
                self.w("merger.linear_fc1.bias")?,
                VitAct::GeluErf,
                mb.addr(),
                mw,
            )?;
            gemm(
                mb.addr(),
                self.w("merger.linear_fc2.weight")?,
                y.addr(),
                tm,
                OUT,
                mw,
            )?;
            k::vit_residual(
                s,
                f32_at(&out, t0, OUT),
                y.addr(),
                self.w("merger.linear_fc2.bias")?,
                0,
                tm,
                OUT,
                true,
            )?;
        }
        s.synchronize()?;
        Ok(out)
    }
}

/// Qwen2-VL `smart_resize`: dimensions rounded to multiples of 32 (patch x merge), area kept within
/// [min_pixels, max_pixels], aspect ratio approximately preserved. Python's round-half-to-even.
pub fn smart_resize(
    height: usize,
    width: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    let factor = (PATCH * MERGE) as f64;
    let (h, w) = (height as f64, width as f64);
    ensure!(h.max(w) / h.min(w) <= 200.0, "image aspect ratio over 200");
    let round = |x: f64| x.round_ties_even();
    let (mut hb, mut wb) = (round(h / factor) * factor, round(w / factor) * factor);
    if hb * wb > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        hb = factor.max((h / beta / factor).floor() * factor);
        wb = factor.max((w / beta / factor).floor() * factor);
    } else if hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        hb = (h * beta / factor).ceil() * factor;
        wb = (w * beta / factor).ceil() * factor;
    }
    Ok((hb as usize, wb as usize))
}

/// The checkpoint's image processor: decode, `smart_resize` with bicubic resampling (to 8-bit, as
/// PIL does), scale to [-1, 1] (mean 0.5, std 0.5), duplicate into the 2 temporal frames and cut
/// 16x16 patches in merge-block order.
pub fn preprocess_image(bytes: &[u8], min_pixels: usize, max_pixels: usize) -> Result<VisionInput> {
    let img = image::load_from_memory(bytes)?.to_rgb8();
    let (h, w) = smart_resize(
        img.height() as usize,
        img.width() as usize,
        min_pixels,
        max_pixels,
    )?;
    let img = image::imageops::resize(
        &img,
        w as u32,
        h as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let (gh, gw) = (h / PATCH, w / PATCH);
    let coords = patch_coords(gh, gw);
    let mut pixels = Vec::with_capacity(gh * gw * PATCH_DIM);
    for &(r, c) in &coords {
        for ch in 0..3 {
            for _frame in 0..TEMPORAL {
                for y in 0..PATCH {
                    for x in 0..PATCH {
                        let v = img.get_pixel((c * PATCH + x) as u32, (r * PATCH + y) as u32)[ch];
                        pixels.push((v as f32 / 255.0 - 0.5) / 0.5);
                    }
                }
            }
        }
    }
    Ok(VisionInput {
        pixels,
        grid: [1, gh, gw],
    })
}

impl VisionModel {
    /// Encodes `input` into one span of placeholder tokens per temporal group (an image has one),
    /// starting at the given sequence indices (each group is its own MRoPE grid with t = 1).
    /// The image's spans in the text sequence. It is encoded only when a span reaches position `from`
    /// or later (earlier ones lie in a restored prefix and need only their geometry, for positions).
    pub fn spans(
        &self,
        s: &Stream,
        input: &VisionInput,
        starts: &[usize],
        from: usize,
    ) -> Result<Vec<crate::forward::MmSpan>> {
        let [t, h, w] = input.grid;
        ensure!(
            starts.len() == t,
            "{} span starts for {t} temporal groups",
            starts.len()
        );
        let (gh, gw) = (h / MERGE, w / MERGE);
        let needed = starts.iter().any(|&st| st + gh * gw > from);
        let emb = if needed {
            Some(std::sync::Arc::new(self.encode(s, input)?))
        } else {
            None
        };
        let offsets: Vec<[i32; 3]> = (0..gh)
            .flat_map(|r| (0..gw).map(move |c| [0, r as i32, c as i32]))
            .collect();
        Ok(starts
            .iter()
            .enumerate()
            .map(|(g, &start)| crate::forward::MmSpan {
                start,
                emb: emb.clone(),
                row: g * gh * gw,
                offsets: offsets.clone(),
                advance: gh.max(gw) as i32,
            })
            .collect())
    }
}

/// A sampled, preprocessed video: temporal groups of 2 frames and each group's timestamp (seconds).
pub struct VideoClip {
    pub input: VisionInput,
    pub timestamps: Vec<f64>,
}

/// The Qwen3-VL video `smart_resize`: like images, but the area budget covers all frames.
fn video_resize(
    frames: usize,
    height: usize,
    width: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    let factor = (PATCH * MERGE) as f64;
    let (mut h, mut w) = (height as f64, width as f64);
    if h < factor || w < factor {
        let scale = (factor / h).max(factor / w);
        h = (h * scale).floor();
        w = (w * scale).floor();
    }
    ensure!(h.max(w) / h.min(w) <= 200.0, "video aspect ratio over 200");
    let t = ((frames as f64) / TEMPORAL as f64).round_ties_even() * TEMPORAL as f64;
    let (mut hb, mut wb) = (
        (h / factor).round_ties_even() * factor,
        (w / factor).round_ties_even() * factor,
    );
    let n = frames as f64;
    if t * hb * wb > max_pixels as f64 {
        let beta = (n * h * w / max_pixels as f64).sqrt();
        hb = factor.max((h / beta / factor).floor() * factor);
        wb = factor.max((w / beta / factor).floor() * factor);
    } else if t * hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (n * h * w)).sqrt();
        hb = (h * beta / factor).ceil() * factor;
        wb = (w * beta / factor).ceil() * factor;
    }
    Ok((hb as usize, wb as usize))
}

/// The checkpoint's video processor, with ffmpeg decoding: samples 2 frames per second (4 to
/// `max_frames`, at most every frame) at evenly spaced indices, resizes them all to one size, pairs
/// frames into temporal patches (repeating the last frame to an even count) and cuts patches like
/// images. Timestamps average each pair's frame times, as the HF processor's prompt does.
pub fn preprocess_video(
    path: &std::path::Path,
    min_pixels: usize,
    max_pixels: usize,
    max_frames: usize,
) -> Result<VideoClip> {
    use anyhow::Context;
    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .context("running ffprobe")?;
    ensure!(
        probe.status.success(),
        "ffprobe: {}",
        String::from_utf8_lossy(&probe.stderr)
    );
    let info: serde_json::Value = serde_json::from_slice(&probe.stdout)?;
    let st = &info["streams"][0];
    let (width, height) = (
        st["width"].as_u64().context("video width")? as usize,
        st["height"].as_u64().context("video height")? as usize,
    );
    let rate = |v: &serde_json::Value| -> Option<f64> {
        let (a, b) = v.as_str()?.split_once('/')?;
        let (a, b) = (a.parse::<f64>().ok()?, b.parse::<f64>().ok()?);
        (b > 0.0 && a > 0.0).then_some(a / b)
    };
    let fps = rate(&st["avg_frame_rate"])
        .or_else(|| rate(&st["r_frame_rate"]))
        .unwrap_or(24.0);
    let total = match st["nb_frames"]
        .as_str()
        .and_then(|n| n.parse::<usize>().ok())
    {
        Some(n) => n,
        None => (st["duration"]
            .as_str()
            .and_then(|d| d.parse::<f64>().ok())
            .context("video length")?
            * fps)
            .floor() as usize,
    };
    ensure!(total > 0, "video has no frames");
    let wanted = ((total as f64 / fps * 2.0) as usize)
        .max(4)
        .min(max_frames)
        .min(total);
    let indices: Vec<usize> = (0..wanted)
        .map(|i| {
            if wanted == 1 {
                0.0
            } else {
                i as f64 * (total - 1) as f64 / (wanted - 1) as f64
            }
        })
        .map(|x| x.round_ties_even() as usize)
        .collect();
    let select = indices
        .iter()
        .map(|i| format!("eq(n\\,{i})"))
        .collect::<Vec<_>>()
        .join("+");
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-vf",
            &format!("select='{select}'"),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .output()
        .context("running ffmpeg")?;
    ensure!(
        out.status.success(),
        "ffmpeg: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let frame_bytes = width * height * 3;
    let got = out.stdout.len() / frame_bytes;
    ensure!(got > 0, "ffmpeg returned no frames");
    let (h, w) = video_resize(got, height, width, min_pixels, max_pixels)?;
    let mut frames: Vec<image::RgbImage> = out
        .stdout
        .chunks_exact(frame_bytes)
        .map(|raw| {
            let img = image::RgbImage::from_raw(width as u32, height as u32, raw.to_vec())
                .expect("frame size");
            image::imageops::resize(
                &img,
                w as u32,
                h as u32,
                image::imageops::FilterType::CatmullRom,
            )
        })
        .collect();
    let mut idx = indices[..got.min(indices.len())].to_vec();
    while frames.len() % TEMPORAL != 0 {
        frames.push(frames.last().unwrap().clone());
        idx.push(*idx.last().unwrap());
    }
    let (gh, gw) = (h / PATCH, w / PATCH);
    let coords = patch_coords(gh, gw);
    let groups = frames.len() / TEMPORAL;
    let mut pixels = Vec::with_capacity(groups * gh * gw * PATCH_DIM);
    for g in 0..groups {
        for &(r, c) in &coords {
            for ch in 0..3 {
                for f in &frames[g * TEMPORAL..(g + 1) * TEMPORAL] {
                    for y in 0..PATCH {
                        for x in 0..PATCH {
                            let v = f.get_pixel((c * PATCH + x) as u32, (r * PATCH + y) as u32)[ch];
                            pixels.push((v as f32 / 255.0 - 0.5) / 0.5);
                        }
                    }
                }
            }
        }
    }
    let timestamps = idx
        .chunks(TEMPORAL)
        .map(|p| (p[0] as f64 / fps + p[TEMPORAL - 1] as f64 / fps) / 2.0)
        .collect();
    Ok(VideoClip {
        input: VisionInput {
            pixels,
            grid: [groups, gh, gw],
        },
        timestamps,
    })
}
