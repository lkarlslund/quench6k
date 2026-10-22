//! Recursive Flint (FlintForCausalLM) from the vendor's NVFP4 GGUF: weights on the device and the forward
//! pass over a chunk of new tokens of one sequence.
//!
//! Every FFN runs as 512-wide slots per token (see kernels/flint.cu): a MoE layer's six routed experts
//! plus the shared expert's two halves, a dense layer's 6144-wide MLP as twelve slices. The expert tables
//! of a layer list each slot's gate, up and down matrices.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use quench6k_cuda::flint::{self as fk, Fp4, Table};
use quench6k_cuda::kernels::{self as k, WeightFormat};
use quench6k_cuda::{DeviceBuffer, PinnedBuffer, Stream};

use crate::gguf::{GgmlType, Gguf};

pub const D: usize = 2048;
pub const LAYERS: usize = 49;
pub const HQ: usize = 32;
pub const HKV: usize = 4;
pub const HD: usize = 128;
pub const QKV: usize = (HQ + 2 * HKV) * HD;
pub const VOCAB: usize = 150_144;
pub const EW: usize = 512;
pub const EXPERTS: usize = 256;
pub const WINDOW: usize = 513;
/// Sliding layers keep the last SWA_RING positions (power of two; at least WINDOW + the largest chunk).
pub const SWA_RING: usize = 4096;
pub const MAX_CHUNK: usize = 2048;
/// Most tokens of one graph step (a speculative verification).
pub const MAX_STEP: usize = SMALL;
const EPS: f32 = 1e-6;
const MAX_SPLITS: usize = 64;

pub struct Layer {
    pub attn_norm: u64,
    pub ffn_norm: u64,
    pub q: Fp4,
    pub k: Fp4,
    pub v: Fp4,
    pub o: Fp4,
    /// Single-entry grouped-GEMM tables of q, k, v, o (prefill).
    pub q_t: Table,
    pub k_t: Table,
    pub v_t: Table,
    pub o_t: Table,
    pub q_norm: u64,
    pub k_norm: u64,
    pub sinks: u64,
    pub sliding: bool,
    pub rope: bool,
    /// BF16 [256, 2048] for MoE layers.
    pub router: Option<u64>,
    /// Slots per token and expert-table entries.
    pub slots: usize,
    pub entries: usize,
    pub gate: Table,
    pub up: Table,
    pub down: Table,
}

pub struct FlintModel {
    pub layers: Vec<Layer>,
    pub embed: u64,
    pub out_norm: u64,
    pub lm_head: Fp4,
    pub lm_t: Table,
    pub stop_tokens: Vec<i32>,
    _buffers: Vec<DeviceBuffer>,
}

struct Loader<'a> {
    g: &'a Gguf,
    s: &'a Stream,
    buffers: Vec<DeviceBuffer>,
    pinned: PinnedBuffer,
    staging: DeviceBuffer,
    bytes: usize,
}

const STAGING: usize = 64 << 20;

impl Loader<'_> {
    fn alloc(&mut self, bytes: usize) -> Result<u64> {
        let b = DeviceBuffer::new(bytes)?;
        let a = b.addr();
        self.buffers.push(b);
        self.bytes += bytes;
        Ok(a)
    }

    fn upload<T: Copy>(&mut self, v: &[T]) -> Result<u64> {
        let a = self.alloc(std::mem::size_of_val(v))?;
        self.s.upload_to(a, v)?;
        Ok(a)
    }

    fn f32s(&mut self, name: &str) -> Result<u64> {
        let v = self.g.f32s(name)?;
        self.upload(&v)
    }

    /// An NVFP4 tensor (rows = product of the outer dims) as codes/scales; `pb` 64-value blocks per piece row.
    fn fp4(&mut self, name: &str, pb: Option<usize>) -> Result<(u64, u64, usize, usize)> {
        let t = self.g.get(name)?.clone();
        ensure!(t.ty == GgmlType::Nvfp4, "{name}: expected NVFP4, found {:?}", t.ty);
        let kdim = t.dims[0];
        let rows: usize = t.dims[1..].iter().product();
        ensure!(kdim % 256 == 0, "{name}: K = {kdim} is not a multiple of 256");
        let kblocks = kdim / 64;
        // Pieces split only the input dimension of a single matrix.
        let pb = pb.unwrap_or(kblocks);
        let codes = self.alloc(rows * kdim / 2)?;
        let scales = self.alloc(rows * kdim / 16)?;
        let data = self.g.data(&t);
        let total = rows * kblocks;
        let per = STAGING / 36;
        let mut first = 0;
        while first < total {
            let n = per.min(total - first);
            let src = &data[first * 36..(first + n) * 36];
            self.pinned.as_mut_slice()[..src.len()].copy_from_slice(src);
            unsafe { self.s.copy_h2d(self.staging.addr(), self.pinned.as_ptr(), src.len())? };
            fk::repack(self.s, self.staging.addr(), n, first, kblocks, rows, pb, codes, scales)?;
            self.s.synchronize()?;
            first += n;
        }
        Ok((codes, scales, rows, kdim))
    }

    fn scale(&self, name: &str) -> Result<Vec<f32>> {
        self.g.f32s(&name.replace(".weight", ".scale"))
    }

    /// A single matrix with its global scale on the device and its single-entry table.
    fn matrix(&mut self, name: &str) -> Result<(Fp4, Table)> {
        let (codes, scales, n, kdim) = self.fp4(name, None)?;
        let s2 = self.scale(name)?;
        ensure!(s2.len() == 1, "{name}: expected one global scale");
        let t = self.table(&[(codes, scales, s2[0])])?;
        Ok((Fp4 { codes, scales, s2: t.2, n, k: kdim }, t))
    }

    fn table(&mut self, entries: &[(u64, u64, f32)]) -> Result<Table> {
        let c: Vec<u64> = entries.iter().map(|e| e.0).collect();
        let s: Vec<u64> = entries.iter().map(|e| e.1).collect();
        let w: Vec<f32> = entries.iter().map(|e| e.2).collect();
        Ok((self.upload(&c)?, self.upload(&s)?, self.upload(&w)?))
    }
}

impl FlintModel {
    /// `dir`: a directory with one `*.gguf` file (or the file itself).
    pub fn gguf_path(dir: &Path) -> Result<std::path::PathBuf> {
        if dir.is_file() {
            return Ok(dir.to_path_buf());
        }
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
            .collect();
        files.sort();
        ensure!(files.len() == 1, "{}: expected exactly one .gguf file, found {}", dir.display(), files.len());
        Ok(files.remove(0))
    }

    pub fn is_flint(dir: &Path) -> bool {
        Self::gguf_path(dir).is_ok_and(|p| Gguf::open(&p).is_ok_and(|g| g.meta.get("general.architecture").and_then(|v| v.as_str()) == Some("flint")))
    }

    pub fn load(dir: &Path, s: &Stream) -> Result<Self> {
        let path = Self::gguf_path(dir)?;
        let g = Gguf::open(&path)?;
        let arch = g.meta.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("");
        ensure!(arch == "flint", "{}: architecture {arch:?} is not flint", path.display());
        ensure!(g.meta_i64("flint.block_count")? as usize == LAYERS, "unexpected layer count");
        ensure!(g.meta_i64("flint.embedding_length")? as usize == D, "unexpected hidden size");
        ensure!(g.meta_i64("flint.attention.sliding_window")? as usize == WINDOW, "unexpected sliding window");
        let pattern = |key: &str| -> Result<Vec<bool>> {
            match g.meta.get(key) {
                Some(crate::gguf::Value::Array(a)) => Ok(a.iter().map(|v| matches!(v, crate::gguf::Value::Bool(true))).collect()),
                _ => anyhow::bail!("metadata {key} missing"),
            }
        };
        let sliding = pattern("flint.attention.sliding_window_pattern")?;
        let rope = pattern("flint.attention.rope_pattern")?;
        let dense_lead = g.meta_i64("flint.leading_dense_block_count")? as usize;
        let mut stop_tokens = Vec::new();
        if let Some(crate::gguf::Value::Array(a)) = g.meta.get("tokenizer.ggml.eog_token_ids") {
            stop_tokens.extend(a.iter().filter_map(|v| v.as_i64()).map(|v| v as i32));
        }

        let mut l = Loader { g: &g, s, buffers: Vec::new(), pinned: PinnedBuffer::new(STAGING)?, staging: DeviceBuffer::new(STAGING)?, bytes: 0 };
        let t0 = std::time::Instant::now();
        let emb = g.get("token_embd.weight")?.clone();
        ensure!(emb.ty == GgmlType::Bf16 && emb.dims == [D, VOCAB], "token_embd: unexpected layout");
        let embed = l.alloc(emb.bytes)?;
        for (i, chunk) in g.data(&emb).chunks(STAGING).enumerate() {
            l.pinned.as_mut_slice()[..chunk.len()].copy_from_slice(chunk);
            unsafe { s.copy_h2d(embed + (i * STAGING) as u64, l.pinned.as_ptr(), chunk.len())? };
            s.synchronize()?;
        }
        let out_norm = l.f32s("output_norm.weight")?;
        let (lm_head, lm_t) = l.matrix("output.weight")?;
        ensure!(lm_head.n == VOCAB && lm_head.k == D, "output: unexpected shape");

        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let n = |x: &str| format!("blk.{i}.{x}");
            let attn_norm = l.f32s(&n("attn_norm.weight"))?;
            let ffn_norm = l.f32s(&n("ffn_norm.weight"))?;
            let (q, q_t) = l.matrix(&n("attn_q.weight"))?;
            let (kk, k_t) = l.matrix(&n("attn_k.weight"))?;
            let (v, v_t) = l.matrix(&n("attn_v.weight"))?;
            let (o, o_t) = l.matrix(&n("attn_output.weight"))?;
            ensure!(q.n == HQ * HD && kk.n == HKV * HD && v.n == HKV * HD && o.n == D && o.k == HQ * HD, "layer {i}: unexpected attention shapes");
            let q_norm = l.f32s(&n("attn_q_norm.weight"))?;
            let k_norm = l.f32s(&n("attn_k_norm.weight"))?;
            let sinks = l.f32s(&n("attn_sinks.weight"))?;
            ensure!(!g.tensors.contains_key(&n("ffn_exp_probs_b.bias")), "router expert bias is not supported");
            let (router, slots, gate, up, down);
            let mut entries_g = Vec::new();
            let mut entries_u = Vec::new();
            let mut entries_d = Vec::new();
            if i < dense_lead {
                let (gc, gs, rows, _) = l.fp4(&n("ffn_gate.weight"), None)?;
                let (uc, us, _, _) = l.fp4(&n("ffn_up.weight"), None)?;
                let (dc, ds, drows, dk) = l.fp4(&n("ffn_down.weight"), Some(EW / 64))?;
                ensure!(drows == D && dk == rows && rows % EW == 0, "layer {i}: unexpected dense MLP shapes");
                let (g2, u2, d2) = (l.scale(&n("ffn_gate.weight"))?[0], l.scale(&n("ffn_up.weight"))?[0], l.scale(&n("ffn_down.weight"))?[0]);
                for p in 0..rows / EW {
                    let off = (p * EW) as u64;
                    entries_g.push((gc + off * (D / 2) as u64, gs + off * (D / 16) as u64, g2));
                    entries_u.push((uc + off * (D / 2) as u64, us + off * (D / 16) as u64, u2));
                    let piece = (p * D) as u64;
                    entries_d.push((dc + piece * (EW / 2) as u64, ds + piece * (EW / 16) as u64, d2));
                }
                router = None;
                slots = rows / EW;
            } else {
                let rt = g.get(&n("ffn_gate_inp.weight"))?.clone();
                ensure!(rt.ty == GgmlType::Bf16 && rt.dims == [D, EXPERTS], "layer {i}: unexpected router");
                let data = g.data(&rt).to_vec();
                router = Some(l.upload(&data)?);
                let (gc, gs, grows, _) = l.fp4(&n("ffn_gate_exps.weight"), None)?;
                let (uc, us, _, _) = l.fp4(&n("ffn_up_exps.weight"), None)?;
                let (dc, ds, drows, dk) = l.fp4(&n("ffn_down_exps.weight"), None)?;
                ensure!(grows == EXPERTS * EW && drows == EXPERTS * D && dk == EW, "layer {i}: unexpected expert shapes");
                let (g2, u2, d2) = (l.scale(&n("ffn_gate_exps.weight"))?, l.scale(&n("ffn_up_exps.weight"))?, l.scale(&n("ffn_down_exps.weight"))?);
                for e in 0..EXPERTS {
                    let r = (e * EW) as u64;
                    entries_g.push((gc + r * (D / 2) as u64, gs + r * (D / 16) as u64, g2[e]));
                    entries_u.push((uc + r * (D / 2) as u64, us + r * (D / 16) as u64, u2[e]));
                    let r = (e * D) as u64;
                    entries_d.push((dc + r * (EW / 2) as u64, ds + r * (EW / 16) as u64, d2[e]));
                }
                let (sgc, sgs, srows, _) = l.fp4(&n("ffn_gate_shexp.weight"), None)?;
                let (suc, sus, _, _) = l.fp4(&n("ffn_up_shexp.weight"), None)?;
                let (sdc, sds, sdrows, sdk) = l.fp4(&n("ffn_down_shexp.weight"), Some(EW / 64))?;
                ensure!(srows == 2 * EW && sdrows == D && sdk == 2 * EW, "layer {i}: unexpected shared expert shapes");
                let (sg2, su2, sd2) = (l.scale(&n("ffn_gate_shexp.weight"))?[0], l.scale(&n("ffn_up_shexp.weight"))?[0], l.scale(&n("ffn_down_shexp.weight"))?[0]);
                for p in 0..2 {
                    let off = (p * EW) as u64;
                    entries_g.push((sgc + off * (D / 2) as u64, sgs + off * (D / 16) as u64, sg2));
                    entries_u.push((suc + off * (D / 2) as u64, sus + off * (D / 16) as u64, su2));
                    let piece = (p * D) as u64;
                    entries_d.push((sdc + piece * (EW / 2) as u64, sds + piece * (EW / 16) as u64, sd2));
                }
                slots = 8;
            }
            gate = l.table(&entries_g)?;
            up = l.table(&entries_u)?;
            down = l.table(&entries_d)?;
            layers.push(Layer {
                attn_norm,
                ffn_norm,
                q,
                k: kk,
                v,
                o,
                q_t,
                k_t,
                v_t,
                o_t,
                q_norm,
                k_norm,
                sinks,
                sliding: sliding[i],
                rope: rope[i],
                router,
                slots,
                entries: entries_g.len(),
                gate,
                up,
                down,
            });
        }
        eprintln!("flint: loaded {} in {:.1}s ({:.1} GiB on the device)", path.display(), t0.elapsed().as_secs_f64(), l.bytes as f64 / (1u64 << 30) as f64);
        let Loader { buffers, .. } = l;
        Ok(Self { layers, embed, out_norm, lm_head, lm_t, stop_tokens, _buffers: buffers })
    }
}

/// A sequence's KV caches: full layers [capacity, 4, 128] BF16, sliding layers a ring of SWA_RING rows.
pub struct FlintSeq {
    pub capacity: usize,
    /// Full-attention caches are FP8 (E4M3 with a scale per position and head) instead of BF16.
    pub fp8: bool,
    pub tokens: Vec<i32>,
    pub k: Vec<DeviceBuffer>,
    pub v: Vec<DeviceBuffer>,
    /// Captured steps of this sequence (its caches are baked in), by token count 1..=SMALL.
    graphs: Vec<Option<quench6k_cuda::Graph>>,
    /// The sliding layers' rings as they were after the first `pos` tokens (taken at a prompt's end),
    /// so that a follow-up prompt diverging anywhere after it needs no full prefill.
    snap: Option<(usize, DeviceBuffer)>,
    sliding: Vec<bool>,
}

impl FlintSeq {
    pub fn new(model: &FlintModel, capacity: usize) -> Result<Self> {
        let fp8 = crate::forward::KV_FP8.load(std::sync::atomic::Ordering::Relaxed);
        let mut k = Vec::with_capacity(LAYERS);
        let mut v = Vec::with_capacity(LAYERS);
        for layer in &model.layers {
            let bytes = if layer.sliding {
                SWA_RING * HKV * HD * 2
            } else if fp8 {
                capacity * HKV * (HD + 4)
            } else {
                capacity * HKV * HD * 2
            };
            k.push(DeviceBuffer::new(bytes)?);
            v.push(DeviceBuffer::new(bytes)?);
        }
        Ok(Self { capacity, fp8, tokens: Vec::new(), k, v, graphs: (0..=SMALL).map(|_| None).collect(), snap: None, sliding: model.layers.iter().map(|l| l.sliding).collect() })
    }

    /// Keeps the longest prefix of `prompt` whose state is available and drops the rest of the cached
    /// tokens; returns its length. The common prefix with the cached tokens is usable while the rings
    /// still hold the WINDOW positions before it (rewriting a chunk of <= MAX_CHUNK positions never
    /// overwrites a key its queries need); otherwise the ring snapshot is restored when it covers it.
    pub fn rewind(&mut self, prompt: &[i32], s: &Stream) -> Result<usize> {
        const REACH: usize = SWA_RING - WINDOW;
        let common = self.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let p = common.min(prompt.len().saturating_sub(1));
        let live = if self.tokens.len() <= p + REACH { p } else { 0 };
        let snap = match &self.snap {
            Some((e, _)) if *e <= self.tokens.len() && p + REACH >= *e => p.min(*e),
            _ => 0,
        };
        let keep = if snap > live {
            self.restore(s)?;
            snap
        } else {
            live
        };
        if self.snap.as_ref().is_some_and(|(e, _)| keep < *e) {
            self.snap = None;
        }
        self.tokens.truncate(keep);
        Ok(keep)
    }

    /// Remembers the rings at the current position (after a prompt).
    pub fn snapshot(&mut self, model: &FlintModel, s: &Stream) -> Result<()> {
        let ring = SWA_RING * HKV * HD * 2;
        let n = model.layers.iter().filter(|l| l.sliding).count();
        let buf = match self.snap.take() {
            Some((_, b)) => b,
            None => DeviceBuffer::new(2 * n * ring)?,
        };
        let mut at = 0;
        for (i, layer) in model.layers.iter().enumerate() {
            if layer.sliding {
                s.copy_d2d(buf.addr() + at as u64, self.k[i].addr(), ring)?;
                s.copy_d2d(buf.addr() + (at + ring) as u64, self.v[i].addr(), ring)?;
                at += 2 * ring;
            }
        }
        self.snap = Some((self.tokens.len(), buf));
        Ok(())
    }

    fn restore(&mut self, s: &Stream) -> Result<()> {
        let Some((_, buf)) = &self.snap else { return Ok(()) };
        let ring = SWA_RING * HKV * HD * 2;
        let mut at = 0;
        for i in 0..self.k.len() {
            if self.sliding[i] {
                s.copy_d2d(self.k[i].addr(), buf.addr() + at as u64, ring)?;
                s.copy_d2d(self.v[i].addr(), buf.addr() + (at + ring) as u64, ring)?;
                at += 2 * ring;
            }
        }
        Ok(())
    }
}

/// Scratch for steps of up to MAX_CHUNK tokens.
pub struct Workspace {
    _bufs: Vec<DeviceBuffer>,
    tok: u64,
    start: u64,
    h: u64,
    xn: u64,
    qkv: u64,
    qf: u64,
    kf: u64,
    vf: u64,
    qb: u64,
    attn: u64,
    of: u64,
    part_o: u64,
    part_ml: u64,
    rlogits: u64,
    sel_e: u64,
    sel_w: u64,
    act: u64,
    counts: u64,
    cursor: u64,
    offsets: u64,
    tile_expert: u64,
    tile_row: u64,
    num_tiles: u64,
    pair_row: u64,
    d_offsets: u64,
    d_tile_expert: u64,
    d_tile_row: u64,
    d_num_tiles: u64,
    xs: u64,
    yg: u64,
    yu: u64,
    actb: u64,
    yd: u64,
    /// NVFP4-quantized rows for the W4A4 expert path (flint.prefill_fp4): token rows, then activations.
    q4: u64,
    q4b: u64,
    pub logits: u64,
    logit_rows: usize,
    argmax_scratch: u64,
    pub argmax_out: u64,
    ticket: u64,
    /// Attention merge tickets, one per (64-row tile, KV head).
    attn_tickets: u64,
    /// Host staging of a decode step's token and position.
    io: PinnedBuffer,
}

const SMALL: usize = 8;

impl Workspace {
    /// `logit_rows`: rows of logits a step can return (1 for generation).
    pub fn new(logit_rows: usize) -> Result<Self> {
        let t = MAX_CHUNK;
        let pairs = t * 12;
        let bm = k::moe_tile_rows();
        let max_tiles = pairs.div_ceil(bm) + EXPERTS + 2;
        let mut bufs = Vec::new();
        let mut a = |bytes: usize| -> Result<u64> {
            let b = DeviceBuffer::new(bytes)?;
            let addr = b.addr();
            bufs.push(b);
            Ok(addr)
        };
        let ws = Self {
            tok: a(t * 4)?,
            start: a(16)?,
            h: a(t * D * 4)?,
            xn: a(t * D * 2)?,
            qkv: a(SMALL * QKV * 4)?,
            qf: a(t * HQ * HD * 4)?,
            kf: a(t * HKV * HD * 4)?,
            vf: a(t * HKV * HD * 4)?,
            qb: a(t * HQ * HD * 2)?,
            attn: a(t * HQ * HD * 2)?,
            of: a(t * D * 4)?,
            part_o: a(t * HQ * MAX_SPLITS.min(8).max(1) * HD * 4 + SMALL * HQ * MAX_SPLITS * HD * 4)?,
            part_ml: a(t * HQ * MAX_SPLITS * 2 * 4)?,
            rlogits: a(t * EXPERTS * 4)?,
            sel_e: a(pairs * 4)?,
            sel_w: a(pairs * 4)?,
            act: a(SMALL * 12 * EW * 4)?,
            counts: a(512 * 4)?,
            cursor: a(512 * 4)?,
            offsets: a(513 * 4)?,
            tile_expert: a(max_tiles * 4)?,
            tile_row: a(max_tiles * 4)?,
            num_tiles: a(4)?,
            pair_row: a(pairs * 4)?,
            d_offsets: a(8)?,
            d_tile_expert: a(t.div_ceil(bm) * 4)?,
            d_tile_row: a(t.div_ceil(bm) * 4)?,
            d_num_tiles: a(4)?,
            xs: a(pairs * D * 2)?,
            yg: a(pairs * EW * 4)?,
            yu: a(pairs * EW * 4)?,
            actb: a(pairs * EW * 2)?,
            yd: a(pairs * D * 4)?,
            q4: a(k::fp4_rows_bytes(pairs, D))?,
            q4b: a(k::fp4_rows_bytes(pairs, EW))?,
            logits: a(logit_rows * VOCAB * 4)?,
            logit_rows,
            argmax_scratch: a(k::argmax_scratch(logit_rows, VOCAB))?,
            argmax_out: a(logit_rows * 4)?,
            ticket: {
                let b = DeviceBuffer::zeroed(16)?;
                let addr = b.addr();
                bufs.push(b);
                addr
            },
            attn_tickets: {
                let b = DeviceBuffer::zeroed((t / 8 + 1) * HKV * 4)?;
                let addr = b.addr();
                bufs.push(b);
                addr
            },
            io: PinnedBuffer::new(64)?,
            _bufs: Vec::new(),
        };
        Ok(Self { _bufs: bufs, ..ws })
    }
}

/// Splits the attention of a step of `t` tokens may use (bounded by the partial buffers).
fn splits_for(t: usize, start: usize, window: usize) -> usize {
    let s = fk::attn_splits(t, start, window).min(MAX_SPLITS);
    if t > SMALL { s.min(8.max(1)).min((MAX_CHUNK * 8) / t.max(1)).max(1) } else { s }
}

/// QUENCH6K_FLINT_SYNC=1: synchronize after every op and name the failing one (debugging).
fn checkpoint(s: &Stream, what: &str) -> Result<()> {
    static ON: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    if let Some(v) = ON.get_or_init(|| std::env::var("QUENCH6K_FLINT_SYNC").ok()).as_deref().filter(|v| *v == "1" || *v == what) {
        let _ = v;
        s.synchronize().with_context(|| format!("after {what}"))?;
    }
    Ok(())
}

/// Runs `tokens` (at positions seq.tokens.len() ..) through the model, appending them to the sequence.
/// With `all_logits`, logits of every token are written (t <= logit_rows); otherwise of the last token.
pub fn forward(model: &FlintModel, ws: &Workspace, seq: &mut FlintSeq, tokens: &[i32], all_logits: bool, s: &Stream) -> Result<()> {
    let t = tokens.len();
    ensure!(t > 0 && t <= MAX_CHUNK, "chunk of {t} tokens");
    let start = seq.tokens.len();
    ensure!(start + t <= seq.capacity, "sequence capacity {} exceeded", seq.capacity);
    s.upload_to(ws.tok, tokens)?;
    s.upload_to(ws.start, &[start as i32])?;
    body(model, ws, seq, t, Some(start), all_logits, s)?;
    seq.tokens.extend_from_slice(tokens);
    Ok(())
}

/// One decode token through the sequence's CUDA graph (captured on first use); logits in `ws.logits`.
pub fn decode(model: &FlintModel, ws: &mut Workspace, seq: &mut FlintSeq, token: i32, s: &Stream) -> Result<()> {
    step(model, ws, seq, &[token], s)
}

/// 1..=SMALL tokens (a speculative verification) through the sequence's CUDA graph for that count;
/// logits of every token in `ws.logits` (the workspace must hold that many rows).
pub fn step(model: &FlintModel, ws: &mut Workspace, seq: &mut FlintSeq, tokens: &[i32], s: &Stream) -> Result<()> {
    let t = tokens.len();
    ensure!((1..=SMALL).contains(&t) && t <= ws.logit_rows, "step of {t} tokens");
    let start = seq.tokens.len();
    ensure!(start + t <= seq.capacity, "sequence capacity {} exceeded", seq.capacity);
    let io = ws.io.as_mut_slice();
    for (i, tok) in tokens.iter().enumerate() {
        io[4 * i..4 * i + 4].copy_from_slice(&tok.to_le_bytes());
    }
    io[48..52].copy_from_slice(&(start as i32).to_le_bytes());
    unsafe {
        s.copy_h2d(ws.tok, ws.io.as_ptr(), 4 * t)?;
        s.copy_h2d(ws.start, ws.io.as_ptr().add(48), 4)?;
    }
    if seq.graphs[t].is_none() {
        let g = quench6k_cuda::Graph::capture(s, |s| body(model, ws, seq, t, None, true, s))?;
        seq.graphs[t] = Some(g);
    }
    seq.graphs[t].as_ref().unwrap().launch(s)?;
    seq.tokens.extend_from_slice(tokens);
    Ok(())
}

/// The step's kernels for `t` tokens whose ids and start position are already on the device. `start`:
/// the host position for choosing attention splits, or None for a position-independent (graph) step.
fn body(model: &FlintModel, ws: &Workspace, seq: &FlintSeq, t: usize, start: Option<usize>, all_logits: bool, s: &Stream) -> Result<()> {
    fk::embed(s, ws.tok, t, model.embed, ws.h)?;
    let bm = k::moe_tile_rows();
    let small = t <= SMALL;
    if !small {
        fk::dense_plan(s, t, bm, ws.d_offsets, ws.d_tile_expert, ws.d_tile_row, ws.d_num_tiles)?;
    }
    let dense_gemm = |tb: Table, x: u64, y: u64, n: usize, kdim: usize| -> Result<()> {
        k::moe_nvfp4_gemm(s, x, tb.0, tb.1, tb.2, ws.d_offsets, ws.d_tile_expert, ws.d_tile_row, ws.d_num_tiles, t.div_ceil(bm), y, n, kdim)
    };
    for (li, layer) in model.layers.iter().enumerate() {
        let fuse = t == 1 && quench6k_cuda::variants::get("flint.fuse_norm", 1) != 0;
        if !fuse {
            fk::rmsnorm(s, ws.h, layer.attn_norm, EPS, ws.xn, t)?;
        }
        let (qi, ki, vi) = if small {
            if fuse {
                fk::gemv(s, ws.h, Some(layer.attn_norm), 1, &[layer.q, layer.k, layer.v], ws.qkv, QKV, false)?;
            } else {
                fk::gemv(s, ws.xn, None, t, &[layer.q, layer.k, layer.v], ws.qkv, QKV, false)?;
            }
            ((ws.qkv, QKV), (ws.qkv + (HQ * HD * 4) as u64, QKV), (ws.qkv + ((HQ + HKV) * HD * 4) as u64, QKV))
        } else {
            dense_gemm(layer.q_t, ws.xn, ws.qf, HQ * HD, D)?;
            dense_gemm(layer.k_t, ws.xn, ws.kf, HKV * HD, D)?;
            dense_gemm(layer.v_t, ws.xn, ws.vf, HKV * HD, D)?;
            ((ws.qf, HQ * HD), (ws.kf, HKV * HD), (ws.vf, HKV * HD))
        };
        let (mask, window) = if layer.sliding { ((SWA_RING - 1) as i32, WINDOW) } else { (i32::MAX, 0) };
        let f8_rows = if seq.fp8 && !layer.sliding { seq.capacity } else { 0 };
        checkpoint(s, "qkv")?;
        fk::qk_prep(s, qi, ki, vi, layer.q_norm, layer.k_norm, layer.rope, ws.start, mask, f8_rows, t, ws.qb, seq.k[li].addr(), seq.v[li].addr())?;
        checkpoint(s, "qk_prep")?;
        // Steps of few tokens always split the keys MAX_SPLITS ways, so that eager and graph steps
        // sum in the same order (empty splits exit at once).
        let splits = match start {
            Some(start) if !small => splits_for(t, start, window),
            _ => MAX_SPLITS,
        };
        fk::attention(s, ws.qb, seq.k[li].addr(), seq.v[li].addr(), t, ws.start, window, mask, splits, layer.sinks, ws.part_o, ws.part_ml, ws.attn_tickets, ws.attn, f8_rows)?;
        checkpoint(s, "attention")?;
        if small {
            fk::gemv(s, ws.attn, None, t, &[layer.o], ws.h, D, true)?;
        } else {
            // The residual add happens in the FFN norm below.
            dense_gemm(layer.o_t, ws.attn, ws.of, D, HQ * HD)?;
        }
        let o_add = if small { 0 } else { ws.of };

        checkpoint(s, "o")?;
        match layer.router {
            Some(router) if small => fk::router(s, ws.h, layer.ffn_norm, router, t, ws.rlogits, ws.xn, ws.ticket, ws.sel_e, ws.sel_w)?,
            Some(router) => {
                fk::rmsnorm_add(s, ws.h, o_add, layer.ffn_norm, EPS, ws.xn, t)?;
                if t <= k::gemv_max_m() {
                    k::gemv(s, t, ws.xn, D, WeightFormat::Bf16, router, 0, ws.rlogits, EXPERTS, 0, EXPERTS, D, 1, 1.0, false)?;
                } else {
                    k::gemm_f32(s, ws.xn, D, WeightFormat::Bf16, router, D, 0, ws.rlogits, EXPERTS, t, EXPERTS, D, 1.0, false)?;
                }
                fk::route(s, ws.rlogits, t, ws.sel_e, ws.sel_w)?;
            }
            None => {
                fk::rmsnorm_add(s, ws.h, o_add, layer.ffn_norm, EPS, ws.xn, t)?;
                fk::dense_slots(s, t, layer.slots, ws.sel_e, ws.sel_w)?
            }
        }
        checkpoint(s, "route")?;
        let slots = layer.slots;
        if small {
            fk::moe_small(s, ws.xn, t, slots, ws.sel_e, ws.sel_w, layer.gate, layer.up, layer.down, ws.act, ws.h)?;
            checkpoint(s, "moe_small")?;
        } else if quench6k_cuda::variants::get("flint.prefill_fp4", 0) != 0 {
            // W4A4: activations quantized to NVFP4 per row, FP4 tensor cores (as the Qwen prefill).
            let pairs = t * slots;
            let bm4 = k::fp4_tile_rows();
            let max_tiles = pairs.div_ceil(bm4) + layer.entries;
            fk::moe_plan(s, ws.sel_e, pairs, layer.entries, bm4, ws.counts, ws.cursor, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.pair_row)?;
            fk::gather(s, ws.xn, ws.pair_row, pairs, slots, ws.xs)?;
            k::fp4_quant_rows(s, ws.xs, pairs, D, ws.q4)?;
            k::moe_fp4_gate_up(s, ws.q4, pairs, layer.gate, layer.up, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, max_tiles, ws.actb, EW, D)?;
            k::fp4_quant_rows(s, ws.actb, pairs, EW, ws.q4b)?;
            k::moe_fp4_gemm(s, ws.q4b, pairs, layer.down.0, layer.down.1, layer.down.2, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, max_tiles, ws.yd, D, EW)?;
            fk::combine(s, ws.yd, ws.pair_row, ws.sel_w, t, slots, ws.h)?;
        } else {
            let pairs = t * slots;
            let max_tiles = pairs.div_ceil(bm) + layer.entries;
            fk::moe_plan(s, ws.sel_e, pairs, layer.entries, bm, ws.counts, ws.cursor, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.pair_row)?;
            fk::gather(s, ws.xn, ws.pair_row, pairs, slots, ws.xs)?;
            let gemm = |tb: Table, x: u64, y: u64, n: usize, kdim: usize| {
                k::moe_nvfp4_gemm(s, x, tb.0, tb.1, tb.2, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, max_tiles, y, n, kdim)
            };
            gemm(layer.gate, ws.xs, ws.yg, EW, D)?;
            gemm(layer.up, ws.xs, ws.yu, EW, D)?;
            fk::swiglu(s, ws.yg, ws.yu, pairs * EW, ws.actb)?;
            gemm(layer.down, ws.actb, ws.yd, D, EW)?;
            fk::combine(s, ws.yd, ws.pair_row, ws.sel_w, t, slots, ws.h)?;
        }
    }
    // Final norm and LM head.
    let rows = if all_logits { t } else { 1 };
    ensure!(rows <= ws.logit_rows, "{rows} logit rows requested, workspace holds {}", ws.logit_rows);
    let first = t - rows;
    let mut done = 0;
    while done < rows {
        let m = (rows - done).min(fk::gemv_max_m());
        let h = ws.h + ((first + done) * D * 4) as u64;
        if m == 1 && quench6k_cuda::variants::get("flint.fuse_norm", 1) != 0 {
            fk::gemv(s, h, Some(model.out_norm), 1, &[model.lm_head], ws.logits + (done * VOCAB * 4) as u64, VOCAB, false)?;
        } else {
            fk::rmsnorm(s, h, model.out_norm, EPS, ws.xn, m)?;
            fk::gemv(s, ws.xn, None, m, &[model.lm_head], ws.logits + (done * VOCAB * 4) as u64, VOCAB, false)?;
        }
        done += m;
    }
    Ok(())
}

/// Greedy next token of each of the `rows` logit rows.
pub fn argmax(ws: &Workspace, rows: usize, s: &Stream) -> Result<Vec<i32>> {
    k::argmax(s, ws.logits, rows, VOCAB, VOCAB, ws.argmax_scratch, ws.argmax_out)?;
    s.download::<i32>(ws.argmax_out, rows)
}

/// Prefills `prompt` beyond the reusable prefix in chunks; logits of the last token are left in `ws`.
pub fn prefill(model: &FlintModel, ws: &Workspace, seq: &mut FlintSeq, prompt: &[i32], s: &Stream) -> Result<usize> {
    let keep = seq.rewind(prompt, s)?;
    let mut at = keep;
    while at < prompt.len() {
        let end = (at + MAX_CHUNK).min(prompt.len());
        forward(model, ws, seq, &prompt[at..end], false, s).with_context(|| format!("prefill chunk at {at}"))?;
        at = end;
    }
    Ok(keep)
}
