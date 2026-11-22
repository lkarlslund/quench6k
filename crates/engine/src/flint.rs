//! Recursive Flint (FlintForCausalLM) from the vendor's NVFP4 GGUF: weights on the device and the forward
//! pass over a chunk of new tokens of one sequence.
//!
//! Every FFN runs as 512-wide slots per token (see kernels/flint.cu): a MoE layer's six routed experts
//! plus the shared expert's two halves, a dense layer's 6144-wide MLP as twelve slices. The expert tables
//! of a layer list each slot's gate, up and down matrices.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use quench6k_cuda::flint::{self as fk, Fp4, KvCache, Rows, Table};
use quench6k_cuda::kernels::{self as k, WeightFormat};
use quench6k_cuda::{DeviceBuffer, Event, Graph, PinnedBuffer, Stream, VmBuffer};

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
pub const SWA_RING: usize = 8192;
pub const MAX_CHUNK: usize = 4096;
/// Canonical prefill: every prompt position is computed in chunks of CANON_CHUNK tokens aligned to
/// multiples of CANON_CHUNK from position 0 (the last chunk ends at the prompt's end), so a prompt's
/// chunk geometry depends only on its length, and a cached prefix is reused only up to such a boundary
/// (see [`FlintState::rewind`]): a sequence's bits then depend only on its tokens, not on what was
/// cached before.
pub const CANON_CHUNK: usize = MAX_CHUNK;
/// Boundary snapshots a sequence keeps: the latest boundary and the lowest SNAPS - 1 others.
const SNAPS: usize = 8;
/// Most tokens of one sequence in a graph step (a speculative verification).
pub const MAX_STEP: usize = SMALL;
/// Most sequences of one batched step ([`FlintModel::forward_batch`]).
pub const MAX_SEQS: usize = 8;
/// Most rows (tokens of all sequences) of one batched step.
pub const MAX_ROWS: usize = MAX_SEQS * MAX_STEP;
const EPS: f32 = 1e-6;
/// Buffer bound on attention splits; steps of few tokens use `decode_splits()` (<= this).
const MAX_SPLITS: usize = 128;

/// Key splits of steps of few tokens (variant flint.splits; 94 = two waves of the 188 SMs over the four
/// KV heads when the context is long enough).
fn decode_splits() -> usize {
    static S: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        (quench6k_cuda::variants::get("flint.splits", 94).max(1) as usize).min(MAX_SPLITS)
    })
}

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
    /// The longest sequence the model was trained for (GGUF flint.context_length).
    pub max_context: usize,
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
        ensure!(
            t.ty == GgmlType::Nvfp4,
            "{name}: expected NVFP4, found {:?}",
            t.ty
        );
        let kdim = t.dims[0];
        let rows: usize = t.dims[1..].iter().product();
        ensure!(
            kdim % 256 == 0,
            "{name}: K = {kdim} is not a multiple of 256"
        );
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
            unsafe {
                self.s
                    .copy_h2d(self.staging.addr(), self.pinned.as_ptr(), src.len())?
            };
            fk::repack(
                self.s,
                self.staging.addr(),
                n,
                first,
                kblocks,
                rows,
                pb,
                codes,
                scales,
            )?;
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
        Ok((
            Fp4 {
                codes,
                scales,
                s2: t.2,
                n,
                k: kdim,
            },
            t,
        ))
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
        ensure!(
            files.len() == 1,
            "{}: expected exactly one .gguf file, found {}",
            dir.display(),
            files.len()
        );
        Ok(files.remove(0))
    }

    pub fn is_flint(dir: &Path) -> bool {
        Self::gguf_path(dir).is_ok_and(|p| {
            Gguf::open(&p).is_ok_and(|g| {
                g.meta.get("general.architecture").and_then(|v| v.as_str()) == Some("flint")
            })
        })
    }

    pub fn load(dir: &Path, s: &Stream) -> Result<Self> {
        let path = Self::gguf_path(dir)?;
        let g = Gguf::open(&path)?;
        let arch = g
            .meta
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        ensure!(
            arch == "flint",
            "{}: architecture {arch:?} is not flint",
            path.display()
        );
        ensure!(
            g.meta_i64("flint.block_count")? as usize == LAYERS,
            "unexpected layer count"
        );
        ensure!(
            g.meta_i64("flint.embedding_length")? as usize == D,
            "unexpected hidden size"
        );
        ensure!(
            g.meta_i64("flint.attention.sliding_window")? as usize == WINDOW,
            "unexpected sliding window"
        );
        let pattern = |key: &str| -> Result<Vec<bool>> {
            match g.meta.get(key) {
                Some(crate::gguf::Value::Array(a)) => Ok(a
                    .iter()
                    .map(|v| matches!(v, crate::gguf::Value::Bool(true)))
                    .collect()),
                _ => anyhow::bail!("metadata {key} missing"),
            }
        };
        let sliding = pattern("flint.attention.sliding_window_pattern")?;
        let rope = pattern("flint.attention.rope_pattern")?;
        let dense_lead = g.meta_i64("flint.leading_dense_block_count")? as usize;
        let max_context = g.meta_i64("flint.context_length")? as usize;
        ensure!(
            max_context > 0 && max_context < (1 << 30),
            "unexpected context length {max_context}"
        );
        let mut stop_tokens = Vec::new();
        if let Some(crate::gguf::Value::Array(a)) = g.meta.get("tokenizer.ggml.eog_token_ids") {
            stop_tokens.extend(a.iter().filter_map(|v| v.as_i64()).map(|v| v as i32));
        }

        let mut l = Loader {
            g: &g,
            s,
            buffers: Vec::new(),
            pinned: PinnedBuffer::new(STAGING)?,
            staging: DeviceBuffer::new(STAGING)?,
            bytes: 0,
        };
        let t0 = std::time::Instant::now();
        let emb = g.get("token_embd.weight")?.clone();
        ensure!(
            emb.ty == GgmlType::Bf16 && emb.dims == [D, VOCAB],
            "token_embd: unexpected layout"
        );
        let embed = l.alloc(emb.bytes)?;
        for (i, chunk) in g.data(&emb).chunks(STAGING).enumerate() {
            l.pinned.as_mut_slice()[..chunk.len()].copy_from_slice(chunk);
            unsafe { s.copy_h2d(embed + (i * STAGING) as u64, l.pinned.as_ptr(), chunk.len())? };
            s.synchronize()?;
        }
        let out_norm = l.f32s("output_norm.weight")?;
        let (lm_head, lm_t) = l.matrix("output.weight")?;
        ensure!(
            lm_head.n == VOCAB && lm_head.k == D,
            "output: unexpected shape"
        );

        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let n = |x: &str| format!("blk.{i}.{x}");
            let attn_norm = l.f32s(&n("attn_norm.weight"))?;
            let ffn_norm = l.f32s(&n("ffn_norm.weight"))?;
            let (q, q_t) = l.matrix(&n("attn_q.weight"))?;
            let (kk, k_t) = l.matrix(&n("attn_k.weight"))?;
            let (v, v_t) = l.matrix(&n("attn_v.weight"))?;
            let (o, o_t) = l.matrix(&n("attn_output.weight"))?;
            ensure!(
                q.n == HQ * HD && kk.n == HKV * HD && v.n == HKV * HD && o.n == D && o.k == HQ * HD,
                "layer {i}: unexpected attention shapes"
            );
            let q_norm = l.f32s(&n("attn_q_norm.weight"))?;
            let k_norm = l.f32s(&n("attn_k_norm.weight"))?;
            let sinks = l.f32s(&n("attn_sinks.weight"))?;
            ensure!(
                !g.tensors.contains_key(&n("ffn_exp_probs_b.bias")),
                "router expert bias is not supported"
            );
            let (router, slots, gate, up, down);
            let mut entries_g = Vec::new();
            let mut entries_u = Vec::new();
            let mut entries_d = Vec::new();
            if i < dense_lead {
                let (gc, gs, rows, _) = l.fp4(&n("ffn_gate.weight"), None)?;
                let (uc, us, _, _) = l.fp4(&n("ffn_up.weight"), None)?;
                let (dc, ds, drows, dk) = l.fp4(&n("ffn_down.weight"), Some(EW / 64))?;
                ensure!(
                    drows == D && dk == rows && rows % EW == 0,
                    "layer {i}: unexpected dense MLP shapes"
                );
                let (g2, u2, d2) = (
                    l.scale(&n("ffn_gate.weight"))?[0],
                    l.scale(&n("ffn_up.weight"))?[0],
                    l.scale(&n("ffn_down.weight"))?[0],
                );
                for p in 0..rows / EW {
                    let off = (p * EW) as u64;
                    entries_g.push((gc + off * (D / 2) as u64, gs + off * (D / 16) as u64, g2));
                    entries_u.push((uc + off * (D / 2) as u64, us + off * (D / 16) as u64, u2));
                    let piece = (p * D) as u64;
                    entries_d.push((
                        dc + piece * (EW / 2) as u64,
                        ds + piece * (EW / 16) as u64,
                        d2,
                    ));
                }
                router = None;
                slots = rows / EW;
            } else {
                let rt = g.get(&n("ffn_gate_inp.weight"))?.clone();
                ensure!(
                    rt.ty == GgmlType::Bf16 && rt.dims == [D, EXPERTS],
                    "layer {i}: unexpected router"
                );
                let data = g.data(&rt).to_vec();
                router = Some(l.upload(&data)?);
                let (gc, gs, grows, _) = l.fp4(&n("ffn_gate_exps.weight"), None)?;
                let (uc, us, _, _) = l.fp4(&n("ffn_up_exps.weight"), None)?;
                let (dc, ds, drows, dk) = l.fp4(&n("ffn_down_exps.weight"), None)?;
                ensure!(
                    grows == EXPERTS * EW && drows == EXPERTS * D && dk == EW,
                    "layer {i}: unexpected expert shapes"
                );
                let (g2, u2, d2) = (
                    l.scale(&n("ffn_gate_exps.weight"))?,
                    l.scale(&n("ffn_up_exps.weight"))?,
                    l.scale(&n("ffn_down_exps.weight"))?,
                );
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
                ensure!(
                    srows == 2 * EW && sdrows == D && sdk == 2 * EW,
                    "layer {i}: unexpected shared expert shapes"
                );
                let (sg2, su2, sd2) = (
                    l.scale(&n("ffn_gate_shexp.weight"))?[0],
                    l.scale(&n("ffn_up_shexp.weight"))?[0],
                    l.scale(&n("ffn_down_shexp.weight"))?[0],
                );
                for p in 0..2 {
                    let off = (p * EW) as u64;
                    entries_g.push((sgc + off * (D / 2) as u64, sgs + off * (D / 16) as u64, sg2));
                    entries_u.push((suc + off * (D / 2) as u64, sus + off * (D / 16) as u64, su2));
                    let piece = (p * D) as u64;
                    entries_d.push((
                        sdc + piece * (EW / 2) as u64,
                        sds + piece * (EW / 16) as u64,
                        sd2,
                    ));
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
        eprintln!(
            "flint: loaded {} in {:.1}s ({:.1} GiB on the device)",
            path.display(),
            t0.elapsed().as_secs_f64(),
            l.bytes as f64 / (1u64 << 30) as f64
        );
        let Loader { buffers, .. } = l;
        Ok(Self {
            layers,
            embed,
            out_norm,
            lm_head,
            lm_t,
            stop_tokens,
            max_context,
            _buffers: buffers,
        })
    }
}

/// Bytes per position of a sliding layer's ring (K or V, BF16 [HKV, HD]).
const RING_ROW: usize = HKV * HD * 2;
/// Bytes per position of an FP8 cache's scales (K or V, FP32 [HKV]).
const SCALE_ROW: usize = HKV * 4;

fn full_row(fp8: bool) -> usize {
    if fp8 { HKV * HD } else { HKV * HD * 2 }
}

/// One layer's caches of a sequence. Full layers: K and V (and FP8 scales) for `capacity` positions,
/// reserved as virtual memory and mapped as positions are reached. Sliding layers: a ring of SWA_RING
/// BF16 rows (position p at row p % SWA_RING), mapped up to the rows reached.
struct LayerCache {
    sliding: bool,
    k: VmBuffer,
    v: VmBuffer,
    ks: Option<VmBuffer>,
    vs: Option<VmBuffer>,
    /// Bytes from the start of k, v, ks, vs known to be zeroed or written (newly mapped memory is
    /// zeroed before a step uses it).
    clean: [usize; 4],
}

impl LayerCache {
    fn kv(&self) -> KvCache {
        KvCache {
            k: self.k.addr(),
            v: self.v.addr(),
            ks: self.ks.as_ref().map_or(0, |b| b.addr()),
            vs: self.vs.as_ref().map_or(0, |b| b.addr()),
        }
    }

    /// Each buffer with its bytes per position.
    fn parts(&mut self, fp8: bool) -> impl Iterator<Item = (&mut VmBuffer, usize)> {
        let row = if self.sliding { RING_ROW } else { full_row(fp8) };
        [(&mut self.k, row), (&mut self.v, row)]
            .into_iter()
            .chain(self.ks.iter_mut().map(|b| (b, SCALE_ROW)))
            .chain(self.vs.iter_mut().map(|b| (b, SCALE_ROW)))
    }
}

/// The sliding layers' rows of the WINDOW positions before boundary `pos` (a multiple of CANON_CHUNK;
/// fewer when pos < WINDOW): per sliding layer K rows, then V rows.
struct Snap {
    pos: usize,
    buf: DeviceBuffer,
}

/// Bytes of a boundary snapshot of `n` sliding layers.
fn snap_bytes(n: usize) -> usize {
    2 * n * WINDOW * RING_ROW
}

/// One sequence's state: its tokens and KV caches. Full-attention caches grow with the sequence (device
/// memory is mapped as positions are reached, see [`FlintState::reserve`], and released by
/// [`FlintState::trim`]), so sequences share the device memory by their actual lengths.
pub struct FlintState {
    /// Most positions this sequence may reach.
    pub capacity: usize,
    /// Full-attention caches are FP8 (E4M3 with a scale per position and head) instead of BF16.
    pub fp8: bool,
    /// The tokens whose K/V the caches hold, in order.
    pub tokens: Vec<i32>,
    layers: Vec<LayerCache>,
    /// Positions backed by device memory in every cache.
    mapped: usize,
    /// One past the highest position ever written into the rings since they were last empty: rows of
    /// positions below `high - SWA_RING` may have been overwritten.
    high: usize,
    /// The rings hold the last-written rows of positions [ring_lo, pos()).
    ring_lo: usize,
    /// The canonical frontier: a multiple of CANON_CHUNK below which every position was computed by
    /// an aligned prefill chunk of CANON_CHUNK tokens (never by a smaller or unaligned step) and has
    /// not been rewritten since. Every snapshot's boundary is at most the frontier.
    canon: usize,
    /// Boundary snapshots (at most SNAPS), taken when a canonical chunk reaches a boundary.
    snaps: Vec<Snap>,
    /// Snapshot buffers no longer used (kept for reuse: freeing device memory waits for the device).
    spare: Vec<DeviceBuffer>,
}

impl FlintState {
    /// `capacity`: most positions this sequence may reach (at most the model's maximum). Reserves
    /// addresses only; device memory is mapped as positions are reached.
    pub fn new(model: &FlintModel, capacity: usize, _s: &Stream) -> Result<Self> {
        Self::create(model, capacity)
    }

    fn create(model: &FlintModel, capacity: usize) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= model.max_context,
            "sequence capacity {capacity}: the model supports 1..={}",
            model.max_context
        );
        let fp8 = crate::forward::KV_FP8.load(std::sync::atomic::Ordering::Relaxed);
        let mut layers = Vec::with_capacity(LAYERS);
        for layer in &model.layers {
            let (rows, row) = if layer.sliding {
                (SWA_RING, RING_ROW)
            } else {
                (capacity, full_row(fp8))
            };
            let scales = || -> Result<Option<VmBuffer>> {
                Ok(if fp8 && !layer.sliding {
                    Some(VmBuffer::reserve(capacity * SCALE_ROW)?)
                } else {
                    None
                })
            };
            layers.push(LayerCache {
                sliding: layer.sliding,
                k: VmBuffer::reserve(rows * row)?,
                v: VmBuffer::reserve(rows * row)?,
                ks: scales()?,
                vs: scales()?,
                clean: [0; 4],
            });
        }
        Ok(Self {
            capacity,
            fp8,
            tokens: Vec::new(),
            layers,
            mapped: 0,
            high: 0,
            ring_lo: 0,
            canon: 0,
            snaps: Vec::new(),
            spare: Vec::new(),
        })
    }

    /// Positions of cached tokens.
    pub fn pos(&self) -> usize {
        self.tokens.len()
    }

    /// Device bytes of the growable (full-attention) caches per position, for the current KV format.
    pub fn bytes_per_position(model: &FlintModel) -> usize {
        let fp8 = crate::forward::KV_FP8.load(std::sync::atomic::Ordering::Relaxed);
        let full = model.layers.iter().filter(|l| !l.sliding).count();
        full * 2 * (full_row(fp8) + if fp8 { SCALE_ROW } else { 0 })
    }

    /// Device bytes the sliding rings take once a sequence reaches SWA_RING positions (below that,
    /// RING_ROW per position and layer). Boundary snapshots take [`FlintState::snapshot_bytes`] each.
    pub fn ring_bytes(model: &FlintModel) -> usize {
        model.layers.iter().filter(|l| l.sliding).count() * 2 * SWA_RING * RING_ROW
    }

    /// Device bytes of one boundary snapshot (a sequence keeps up to 8, allocated as boundaries are
    /// reached).
    pub fn snapshot_bytes(model: &FlintModel) -> usize {
        snap_bytes(model.layers.iter().filter(|l| l.sliding).count())
    }

    /// Device memory this sequence holds: mapped caches and its ring snapshots.
    pub fn device_bytes(&self) -> usize {
        let caches: usize = self
            .layers
            .iter()
            .map(|l| {
                l.k.mapped()
                    + l.v.mapped()
                    + l.ks.as_ref().map_or(0, |b| b.mapped())
                    + l.vs.as_ref().map_or(0, |b| b.mapped())
            })
            .sum();
        caches
            + self.snaps.iter().map(|s| s.buf.bytes()).sum::<usize>()
            + self.spare.iter().map(|b| b.bytes()).sum::<usize>()
    }

    /// Positions [0, n) are backed in every cache (from what is mapped).
    fn mapped_positions(&mut self) -> usize {
        let fp8 = self.fp8;
        let mut n = self.capacity;
        for l in &mut self.layers {
            let sliding = l.sliding;
            for (b, row) in l.parts(fp8) {
                let rows = b.mapped() / row;
                if !(sliding && rows >= SWA_RING) {
                    n = n.min(rows);
                }
            }
        }
        n
    }

    /// Maps device memory for positions < `positions` (at most the capacity). Out of device memory:
    /// an error whose message contains "out of memory", and nothing changed.
    pub fn reserve(&mut self, positions: usize) -> Result<()> {
        ensure!(
            positions <= self.capacity,
            "{positions} positions exceed the sequence capacity {}",
            self.capacity
        );
        if positions <= self.mapped {
            return Ok(());
        }
        let fp8 = self.fp8;
        let before: Vec<Vec<usize>> = self
            .layers
            .iter_mut()
            .map(|l| l.parts(fp8).map(|(b, _)| b.mapped()).collect())
            .collect();
        let mut result = Ok(());
        'map: for l in &mut self.layers {
            let rows = if l.sliding {
                positions.min(SWA_RING)
            } else {
                positions
            };
            for (b, row) in l.parts(fp8) {
                if let Err(e) = b.map_to(rows * row) {
                    result = Err(e);
                    break 'map;
                }
            }
        }
        if let Err(e) = result {
            for (l, before) in self.layers.iter_mut().zip(&before) {
                for ((b, _), &m) in l.parts(fp8).zip(before) {
                    b.unmap_to(m)?;
                }
            }
            return Err(e);
        }
        self.mapped = self.mapped_positions();
        Ok(())
    }

    /// Forgets the cached tokens from `positions` on and releases the full-attention cache memory
    /// past them (the rings too while they have not wrapped). The device must be done with this
    /// sequence's earlier steps (synchronize the stream first).
    pub fn trim(&mut self, positions: usize) {
        self.truncate(positions);
        let fp8 = self.fp8;
        // Rings that never wrapped hold nothing past `high`; once wrapped they stay (positions below
        // the kept ones may sit in any row).
        let rings = positions == 0 || self.high <= SWA_RING;
        if rings {
            self.high = self.high.min(positions);
        }
        for l in &mut self.layers {
            let sliding = l.sliding;
            if sliding && !rings {
                continue;
            }
            let rows = if sliding {
                positions.min(SWA_RING)
            } else {
                positions
            };
            let mut clean = l.clean;
            for (i, (b, row)) in l.parts(fp8).enumerate() {
                // Unmapping only fails on a broken context; the memory stays mapped then.
                let _ = b.unmap_to(rows * row);
                clean[i] = clean[i].min(b.mapped());
            }
            l.clean = clean;
        }
        self.mapped = self.mapped_positions();
        if positions == 0 {
            self.snaps.clear();
            self.spare.clear();
        }
    }

    /// Zeroes cache memory mapped since the last call (before any step writes or reads it), so that
    /// no cache row ever holds stale bits from other allocations.
    fn clear_new(&mut self, s: &Stream) -> Result<()> {
        let fp8 = self.fp8;
        for l in &mut self.layers {
            let mut clean = l.clean;
            for (i, (b, _)) in l.parts(fp8).enumerate() {
                let m = b.mapped();
                if m > clean[i] {
                    s.memset(b.addr() + clean[i] as u64, 0, m - clean[i])?;
                }
                clean[i] = m;
            }
            l.clean = clean;
        }
        Ok(())
    }

    /// Forgets the cached tokens from `positions` on (their memory stays mapped).
    pub fn truncate(&mut self, positions: usize) {
        self.tokens.truncate(positions);
        self.lower(self.tokens.len());
    }

    /// Forgets everything and releases all cache memory.
    pub fn reset(&mut self) {
        self.trim(0);
    }

    /// The canonical frontier (see [`FlintState::rewind`]).
    pub fn frontier(&self) -> usize {
        self.canon
    }

    /// Boundaries of the kept ring snapshots, ascending.
    pub fn snapshot_boundaries(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.snaps.iter().map(|s| s.pos).collect();
        v.sort_unstable();
        v
    }

    /// Positions from `pos` on are (about to be) rewritten: lowers the frontier to the boundary at or
    /// below `pos` and forgets the snapshots above it.
    fn lower(&mut self, pos: usize) {
        self.canon = self.canon.min(pos / CANON_CHUNK * CANON_CHUNK);
        let canon = self.canon;
        let mut i = 0;
        while i < self.snaps.len() {
            if self.snaps[i].pos > canon {
                let sn = self.snaps.swap_remove(i);
                self.spare.push(sn.buf);
            } else {
                i += 1;
            }
        }
    }

    /// Keeps the longest cached prefix of `prompt` that canonical prefill allows and drops the rest
    /// of the cached tokens; returns its length B (0: start over): a multiple of CANON_CHUNK at most
    /// the common prefix with the cached tokens, below the prompt's length (its last token is run
    /// again for its logits) and at most the canonical frontier, for which the rings' WINDOW positions
    /// before B are still in the rings or in a boundary snapshot (restored then). The caller computes
    /// the prompt from B on in chunks of CANON_CHUNK tokens (the last one ends at the prompt's end),
    /// so every position's bits are those of a cold prefill.
    pub fn rewind(&mut self, prompt: &[i32], s: &Stream) -> Result<usize> {
        self.lower(self.tokens.len());
        let common = self
            .tokens
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        let limit =
            common.min(prompt.len().saturating_sub(1)).min(self.canon) / CANON_CHUNK * CANON_CHUNK;
        let mut b = limit;
        while b > 0 {
            if b <= self.tokens.len() && self.ring_lo <= b - b.min(WINDOW) {
                break;
            }
            if let Some(i) = self.snaps.iter().position(|sn| sn.pos == b) {
                self.restore(i, s)?;
                break;
            }
            b -= CANON_CHUNK;
        }
        self.truncate(b);
        Ok(b)
    }

    /// Prefix reuse needs no prompt-end snapshot any more (boundary snapshots are taken as canonical
    /// chunks reach them); kept for the scheduler's API.
    pub fn snapshot(&mut self, _model: &FlintModel, _s: &Stream) -> Result<()> {
        Ok(())
    }

    /// The ring rows of the last `n` positions before `e`: (first row, count) runs in position order.
    fn ring_runs(e: usize, n: usize) -> Vec<(usize, usize)> {
        if n == 0 {
            return Vec::new();
        }
        let first = (e - n) % SWA_RING;
        if first + n <= SWA_RING {
            vec![(first, n)]
        } else {
            vec![(first, SWA_RING - first), (0, n - (SWA_RING - first))]
        }
    }

    /// Copies the rings' rows of the WINDOW positions before `e` into `buf` (`save`) or back.
    fn copy_window(&self, e: usize, buf: u64, save: bool, s: &Stream) -> Result<()> {
        let n = e.min(WINDOW);
        let mut at = buf;
        for l in self.layers.iter().filter(|l| l.sliding) {
            for ring in [l.k.addr(), l.v.addr()] {
                let mut part = at;
                for (first, count) in Self::ring_runs(e, n) {
                    let r = ring + (first * RING_ROW) as u64;
                    let bytes = count * RING_ROW;
                    if save {
                        s.copy_d2d(part, r, bytes)?;
                    } else {
                        s.copy_d2d(r, part, bytes)?;
                    }
                    part += bytes as u64;
                }
                at += (WINDOW * RING_ROW) as u64;
            }
        }
        Ok(())
    }

    /// Restores the rings' rows before snapshot `i`'s boundary.
    fn restore(&mut self, i: usize, s: &Stream) -> Result<()> {
        let b = self.snaps[i].pos;
        let rows = b.min(SWA_RING) * RING_ROW;
        for l in self.layers.iter_mut().filter(|l| l.sliding) {
            if l.k.mapped() < rows {
                l.k.map_to(rows)?;
            }
            if l.v.mapped() < rows {
                l.v.map_to(rows)?;
            }
        }
        self.clear_new(s)?;
        self.copy_window(b, self.snaps[i].buf.addr(), false, s)?;
        self.ring_lo = b - b.min(WINDOW);
        self.high = self.high.max(b);
        Ok(())
    }

    /// Positions [start, end) were written (tokens fed at `start`; rows past the kept tokens count
    /// too): advances the frontier when this was an aligned chunk of CANON_CHUNK tokens right at it,
    /// and then snapshots the rings at the new boundary (skipped when no memory is left for it).
    fn wrote(&mut self, start: usize, end: usize, s: &Stream) -> Result<()> {
        self.high = self.high.max(end);
        self.ring_lo = self.ring_lo.min(start).max(end.saturating_sub(SWA_RING));
        if start == self.canon && start.is_multiple_of(CANON_CHUNK) && end - start == CANON_CHUNK {
            self.canon = end;
            let n = self.layers.iter().filter(|l| l.sliding).count();
            // A spare or new buffer, else the highest boundary's (the previous latest; the lowest
            // ones stay).
            let buf = if self.snaps.len() >= SNAPS {
                let i = (0..self.snaps.len()).max_by_key(|&i| self.snaps[i].pos).unwrap();
                Some(self.snaps.swap_remove(i).buf)
            } else if let Some(b) = self.spare.pop() {
                Some(b)
            } else {
                DeviceBuffer::new(snap_bytes(n)).ok()
            };
            if let Some(buf) = buf {
                self.copy_window(end, buf.addr(), true, s)?;
                self.snaps.push(Snap { pos: end, buf });
            }
        }
        Ok(())
    }

    /// The caches of layer `li`.
    fn kv(&self, li: usize) -> KvCache {
        self.layers[li].kv()
    }

}

/// The single-sequence server loop's handle on a [`FlintState`] (its constructor without a stream).
pub struct FlintSeq(pub FlintState);

impl FlintSeq {
    pub fn new(model: &FlintModel, capacity: usize) -> Result<Self> {
        Ok(Self(FlintState::create(model, capacity)?))
    }
}

impl std::ops::Deref for FlintSeq {
    type Target = FlintState;
    fn deref(&self) -> &FlintState {
        &self.0
    }
}

impl std::ops::DerefMut for FlintSeq {
    fn deref_mut(&mut self) -> &mut FlintState {
        &mut self.0
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
    /// Router tickets, one per group of tokens (at most one per token).
    ticket: u64,
    /// Attention merge tickets, one per (64-row tile, KV head).
    attn_tickets: u64,
    /// A batched step's rows: tokens, positions and sequence slots ([MAX_ROWS] i32 each).
    rows: u64,
    /// Per layer and sequence slot: (k, v, k scales, v scales) of the step's sequences.
    kvtab: u64,
    /// Host staging of `rows` and `kvtab`; `io_done` marks when their last upload has completed.
    io: PinnedBuffer,
    kv_io: PinnedBuffer,
    io_done: Event,
    /// The table last uploaded to `kvtab`.
    kv_host: Vec<u64>,
    /// Batched-step graphs by (rows, FP8 caches).
    graphs: std::collections::HashMap<(usize, bool), Graph>,
}

const KVTAB: usize = LAYERS * MAX_SEQS * 4;

const SMALL: usize = 8;

impl Workspace {
    /// `logit_rows`: rows of logits a step can return (1 for generation of one sequence, MAX_ROWS
    /// for any batched step).
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
            qkv: a(MAX_ROWS * QKV * 4)?,
            qf: a(t * HQ * HD * 4)?,
            kf: a(t * HKV * HD * 4)?,
            vf: a(t * HKV * HD * 4)?,
            qb: a(t * HQ * HD * 2)?,
            attn: a(t * HQ * HD * 2)?,
            of: a(t * D * 4)?,
            // Chunks of more than SMALL tokens use at most 8 splits; smaller steps MAX_SPLITS.
            part_o: a((t * HQ * 8).max(MAX_ROWS * HQ * MAX_SPLITS) * HD * 4)?,
            part_ml: a((t * HQ * 8).max(MAX_ROWS * HQ * MAX_SPLITS) * 2 * 4)?,
            rlogits: a(t * EXPERTS * 4)?,
            sel_e: a(pairs * 4)?,
            sel_w: a(pairs * 4)?,
            act: a(MAX_ROWS * 12 * EW * 4)?,
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
            rows: a(3 * MAX_ROWS * 4)?,
            ticket: {
                let b = DeviceBuffer::zeroed(MAX_ROWS * 4)?;
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
            kvtab: {
                let b = DeviceBuffer::zeroed(KVTAB * 8)?;
                let addr = b.addr();
                bufs.push(b);
                addr
            },
            io: PinnedBuffer::new(3 * MAX_ROWS * 4)?,
            kv_io: PinnedBuffer::new(KVTAB * 8)?,
            io_done: Event::new()?,
            kv_host: vec![0; KVTAB],
            graphs: std::collections::HashMap::new(),
            _bufs: Vec::new(),
        };
        Ok(Self { _bufs: bufs, ..ws })
    }
}

/// Splits the attention of a step of `t` tokens may use (bounded by the partial buffers).
fn splits_for(t: usize, start: usize, window: usize) -> usize {
    let s = fk::attn_splits(t, start, window).min(64);
    if t > SMALL {
        s.min(8.max(1)).min((MAX_CHUNK * 8) / t.max(1)).max(1)
    } else {
        s
    }
}

/// QUENCH6K_FLINT_SYNC=1: synchronize after every op and name the failing one (debugging).
fn checkpoint(s: &Stream, what: &str) -> Result<()> {
    static ON: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    if let Some(v) = ON
        .get_or_init(|| std::env::var("QUENCH6K_FLINT_SYNC").ok())
        .as_deref()
        .filter(|v| *v == "1" || *v == what)
    {
        let _ = v;
        s.synchronize().with_context(|| format!("after {what}"))?;
    }
    Ok(())
}

/// Runs `tokens` (at positions seq.tokens.len() ..) through the model, appending them to the sequence.
/// With `all_logits`, logits of every token are written (t <= logit_rows); otherwise of the last token.
pub fn forward(
    model: &FlintModel,
    ws: &Workspace,
    seq: &mut FlintState,
    tokens: &[i32],
    all_logits: bool,
    s: &Stream,
) -> Result<()> {
    let t = tokens.len();
    ensure!(t > 0 && t <= MAX_CHUNK, "chunk of {t} tokens");
    ensure!(
        tokens.iter().all(|&t| (0..VOCAB as i32).contains(&t)),
        "token id out of range"
    );
    let start = seq.tokens.len();
    ensure!(
        start + t <= seq.capacity,
        "sequence capacity {} exceeded",
        seq.capacity
    );
    seq.reserve(start + t)?;
    seq.clear_new(s)?;
    seq.lower(start);
    s.upload_to(ws.tok, tokens)?;
    s.upload_to(ws.start, &[start as i32])?;
    body(model, ws, seq, t, Some(start), all_logits, s)?;
    seq.tokens.extend_from_slice(tokens);
    seq.wrote(start, start + t, s)
}

/// One decode token (a batched step of one sequence); logits in `ws.logits`.
pub fn decode(
    model: &FlintModel,
    ws: &mut Workspace,
    seq: &mut FlintState,
    token: i32,
    s: &Stream,
) -> Result<()> {
    step(model, ws, seq, &[token], s)
}

/// 1..=MAX_STEP tokens of one sequence (a speculative verification) as a batched step; logits of
/// every token in `ws.logits` (the workspace must hold that many rows). Every row's logits are
/// bitwise those of feeding its token alone.
pub fn step(
    model: &FlintModel,
    ws: &mut Workspace,
    seq: &mut FlintState,
    tokens: &[i32],
    s: &Stream,
) -> Result<()> {
    model.forward_batch(&mut [seq], ws, s, &[tokens])
}

impl FlintModel {
    /// One step for 1..=MAX_SEQS sequences with 1..=MAX_STEP tokens each, appended to each sequence;
    /// logits of every fed token in `ws.logits`, sequence by sequence, row by row. Every row is
    /// computed as a one-token step of its sequence alone, so each sequence's results are bitwise
    /// the same as running it alone (with any widths and companions). Maps the memory the step needs
    /// first (an "out of memory" error before any work). Runs as a CUDA graph per (rows, KV format)
    /// that reads the rows' positions and caches from device tables. The caller reads the results
    /// (synchronizes) before the next step.
    pub fn forward_batch(
        &self,
        states: &mut [&mut FlintState],
        ws: &mut Workspace,
        s: &Stream,
        tokens: &[&[i32]],
    ) -> Result<()> {
        let n = states.len();
        ensure!(
            (1..=MAX_SEQS).contains(&n) && tokens.len() == n,
            "batched step of {n} sequences and {} token lists",
            tokens.len()
        );
        ensure!(
            tokens.iter().all(|t| (1..=MAX_STEP).contains(&t.len())),
            "batched step with a sequence of no or more than {MAX_STEP} tokens"
        );
        ensure!(
            tokens.iter().flat_map(|t| t.iter()).all(|&t| (0..VOCAB as i32).contains(&t)),
            "token id out of range"
        );
        let rows: usize = tokens.iter().map(|t| t.len()).sum();
        ensure!(
            rows <= ws.logit_rows,
            "{rows} rows in a step, the workspace holds {}",
            ws.logit_rows
        );
        let fp8 = states[0].fp8;
        ensure!(
            states.iter().all(|st| st.fp8 == fp8),
            "a batched step mixes KV formats"
        );
        for (st, t) in states.iter_mut().zip(tokens) {
            ensure!(
                st.tokens.len() + t.len() <= st.capacity,
                "sequence capacity {} exceeded",
                st.capacity
            );
            st.reserve(st.tokens.len() + t.len())?;
        }
        for st in states.iter_mut() {
            st.clear_new(s)?;
            let pos = st.tokens.len();
            st.lower(pos);
        }
        // Rows and cache tables into pinned staging (after the previous upload has been consumed).
        ws.io_done.synchronize()?;
        let mut kv = vec![0u64; KVTAB];
        for (b, st) in states.iter().enumerate() {
            for li in 0..LAYERS {
                let c = st.kv(li);
                kv[(li * MAX_SEQS + b) * 4..][..4].copy_from_slice(&[c.k, c.v, c.ks, c.vs]);
            }
        }
        {
            let io = ws.io.as_mut_slice();
            let mut r = 0;
            for (b, (st, t)) in states.iter().zip(tokens).enumerate() {
                for (i, &tok) in t.iter().enumerate() {
                    let pos = (st.tokens.len() + i) as i32;
                    for (field, v) in [tok, pos, b as i32].into_iter().enumerate() {
                        let at = (field * MAX_ROWS + r) * 4;
                        io[at..at + 4].copy_from_slice(&v.to_le_bytes());
                    }
                    r += 1;
                }
            }
        }
        unsafe { s.copy_h2d(ws.rows, ws.io.as_ptr(), 3 * MAX_ROWS * 4)? };
        if kv != ws.kv_host {
            for (dst, v) in ws.kv_io.as_mut_slice().chunks_exact_mut(8).zip(&kv) {
                dst.copy_from_slice(&v.to_le_bytes());
            }
            unsafe { s.copy_h2d(ws.kvtab, ws.kv_io.as_ptr(), KVTAB * 8)? };
            ws.kv_host = kv;
        }
        ws.io_done.record(s)?;
        if !ws.graphs.contains_key(&(rows, fp8)) {
            let g = Graph::capture(s, |s| body_rows(self, ws, rows, fp8, s))?;
            ws.graphs.insert((rows, fp8), g);
        }
        ws.graphs[&(rows, fp8)].launch(s)?;
        for (st, t) in states.iter_mut().zip(tokens) {
            let start = st.tokens.len();
            st.tokens.extend_from_slice(t);
            st.wrote(start, start + t.len(), s)?;
        }
        Ok(())
    }
}

/// The kernels of a batched step of `rows` rows whose tokens, positions, sequences and cache tables
/// are on the device (`ws.rows`, `ws.kvtab`). Every kernel computes each row as a one-token step does.
fn body_rows(model: &FlintModel, ws: &Workspace, rows: usize, fp8: bool, s: &Stream) -> Result<()> {
    let (tok, pos, seq) = (
        ws.rows,
        ws.rows + (MAX_ROWS * 4) as u64,
        ws.rows + (2 * MAX_ROWS * 4) as u64,
    );
    fk::embed(s, tok, rows, model.embed, ws.h)?;
    let fuse = quench6k_cuda::variants::get("flint.fuse_norm", 1) != 0;
    // Normalized inputs staged once per step (flint.prenorm) instead of in every CTA: BF16 rows in
    // `qb` (free until qk_prep), RMS factors in `rlogits` (free until the router).
    let prenorm = if quench6k_cuda::variants::get("flint.prenorm", 1) != 0 {
        (ws.qb, ws.rlogits)
    } else {
        (0, 0)
    };
    // The single-token GEMV for one row, else its row-wise twin (bitwise the same per row).
    let gemv = |x: u64, norm: Option<u64>, mats: &[Fp4], y: u64, ldy: usize, acc: bool| {
        if rows == 1 {
            fk::gemv(s, x, norm, 1, mats, y, ldy, acc)
        } else {
            fk::gemv_rows(s, x, norm, rows, mats, y, ldy, acc, prenorm)
        }
    };
    // A GEMV over normalized rows: RMSNorm folded into it (flint.fuse_norm) or a separate kernel.
    let gemv_norm = |h: u64, w: u64, mats: &[Fp4], y: u64, ldy: usize| -> Result<()> {
        if fuse {
            gemv(h, Some(w), mats, y, ldy, false)
        } else {
            fk::rmsnorm(s, h, w, EPS, ws.xn, rows)?;
            gemv(ws.xn, None, mats, y, ldy, false)
        }
    };
    for (li, layer) in model.layers.iter().enumerate() {
        gemv_norm(
            ws.h,
            layer.attn_norm,
            &[layer.q, layer.k, layer.v],
            ws.qkv,
            QKV,
        )?;
        let r = Rows {
            n: rows,
            pos,
            seq,
            kvtab: ws.kvtab + (li * MAX_SEQS * 4 * 8) as u64,
        };
        let (mask, window) = if layer.sliding {
            ((SWA_RING - 1) as i32, WINDOW)
        } else {
            (i32::MAX, 0)
        };
        fk::qk_prep_rows(
            s,
            (ws.qkv, QKV),
            (ws.qkv + (HQ * HD * 4) as u64, QKV),
            (ws.qkv + ((HQ + HKV) * HD * 4) as u64, QKV),
            layer.q_norm,
            layer.k_norm,
            layer.rope,
            mask,
            r,
            ws.qb,
        )?;
        checkpoint(s, "qk_prep")?;
        fk::attention_rows(
            s,
            ws.qb,
            r,
            fp8 && !layer.sliding,
            window,
            mask,
            decode_splits(),
            layer.sinks,
            ws.part_o,
            ws.part_ml,
            ws.attn,
        )?;
        checkpoint(s, "attention")?;
        gemv(ws.attn, None, &[layer.o], ws.h, D, true)?;
        match layer.router {
            Some(router) => fk::router(
                s,
                ws.h,
                layer.ffn_norm,
                router,
                rows,
                ws.rlogits,
                ws.xn,
                ws.ticket,
                ws.sel_e,
                ws.sel_w,
            )?,
            None => {
                fk::rmsnorm_add(s, ws.h, 0, layer.ffn_norm, EPS, ws.xn, rows)?;
                fk::dense_slots(s, rows, layer.slots, ws.sel_e, ws.sel_w)?
            }
        }
        fk::moe_small(
            s,
            ws.xn,
            rows,
            layer.slots,
            ws.sel_e,
            ws.sel_w,
            layer.gate,
            layer.up,
            layer.down,
            ws.act,
            ws.h,
        )?;
        checkpoint(s, "moe_small")?;
    }
    gemv_norm(ws.h, model.out_norm, &[model.lm_head], ws.logits, VOCAB)
}

/// The step's kernels for `t` tokens whose ids and start position are already on the device. `start`:
/// the host position for choosing attention splits, or None for a position-independent (graph) step.
fn body(
    model: &FlintModel,
    ws: &Workspace,
    seq: &FlintState,
    t: usize,
    start: Option<usize>,
    all_logits: bool,
    s: &Stream,
) -> Result<()> {
    fk::embed(s, ws.tok, t, model.embed, ws.h)?;
    let bm = k::moe_tile_rows();
    let small = t <= SMALL;
    if !small {
        fk::dense_plan(
            s,
            t,
            bm,
            ws.d_offsets,
            ws.d_tile_expert,
            ws.d_tile_row,
            ws.d_num_tiles,
        )?;
    }
    let dense_gemm = |tb: Table, x: u64, y: u64, n: usize, kdim: usize| -> Result<()> {
        k::moe_nvfp4_gemm(
            s,
            x,
            tb.0,
            tb.1,
            tb.2,
            ws.d_offsets,
            ws.d_tile_expert,
            ws.d_tile_row,
            ws.d_num_tiles,
            t.div_ceil(bm),
            y,
            n,
            kdim,
        )
    };
    for (li, layer) in model.layers.iter().enumerate() {
        let fuse = t == 1 && quench6k_cuda::variants::get("flint.fuse_norm", 1) != 0;
        if !fuse {
            fk::rmsnorm(s, ws.h, layer.attn_norm, EPS, ws.xn, t)?;
        }
        let (qi, ki, vi) = if small {
            if fuse {
                fk::gemv(
                    s,
                    ws.h,
                    Some(layer.attn_norm),
                    1,
                    &[layer.q, layer.k, layer.v],
                    ws.qkv,
                    QKV,
                    false,
                )?;
            } else {
                fk::gemv(
                    s,
                    ws.xn,
                    None,
                    t,
                    &[layer.q, layer.k, layer.v],
                    ws.qkv,
                    QKV,
                    false,
                )?;
            }
            (
                (ws.qkv, QKV),
                (ws.qkv + (HQ * HD * 4) as u64, QKV),
                (ws.qkv + ((HQ + HKV) * HD * 4) as u64, QKV),
            )
        } else {
            dense_gemm(layer.q_t, ws.xn, ws.qf, HQ * HD, D)?;
            dense_gemm(layer.k_t, ws.xn, ws.kf, HKV * HD, D)?;
            dense_gemm(layer.v_t, ws.xn, ws.vf, HKV * HD, D)?;
            ((ws.qf, HQ * HD), (ws.kf, HKV * HD), (ws.vf, HKV * HD))
        };
        let (mask, window) = if layer.sliding {
            ((SWA_RING - 1) as i32, WINDOW)
        } else {
            (i32::MAX, 0)
        };
        let cache = seq.kv(li);
        checkpoint(s, "qkv")?;
        fk::qk_prep(
            s,
            qi,
            ki,
            vi,
            layer.q_norm,
            layer.k_norm,
            layer.rope,
            ws.start,
            mask,
            t,
            ws.qb,
            cache,
        )?;
        checkpoint(s, "qk_prep")?;
        // Steps of few tokens always split the keys MAX_SPLITS ways, so that eager and graph steps
        // sum in the same order (empty splits exit at once).
        let splits = match start {
            Some(start) if !small => splits_for(t, start, window),
            _ => decode_splits(),
        };
        fk::attention(
            s,
            ws.qb,
            cache,
            t,
            ws.start,
            window,
            mask,
            splits,
            layer.sinks,
            ws.part_o,
            ws.part_ml,
            ws.attn_tickets,
            ws.attn,
        )?;
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
            Some(router) if small => fk::router(
                s,
                ws.h,
                layer.ffn_norm,
                router,
                t,
                ws.rlogits,
                ws.xn,
                ws.ticket,
                ws.sel_e,
                ws.sel_w,
            )?,
            Some(router) => {
                fk::rmsnorm_add(s, ws.h, o_add, layer.ffn_norm, EPS, ws.xn, t)?;
                if t <= k::gemv_max_m() {
                    k::gemv(
                        s,
                        t,
                        ws.xn,
                        D,
                        WeightFormat::Bf16,
                        router,
                        0,
                        ws.rlogits,
                        EXPERTS,
                        0,
                        EXPERTS,
                        D,
                        1,
                        1.0,
                        false,
                    )?;
                } else {
                    k::gemm_f32(
                        s,
                        ws.xn,
                        D,
                        WeightFormat::Bf16,
                        router,
                        D,
                        0,
                        ws.rlogits,
                        EXPERTS,
                        t,
                        EXPERTS,
                        D,
                        1.0,
                        false,
                    )?;
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
            fk::moe_small(
                s, ws.xn, t, slots, ws.sel_e, ws.sel_w, layer.gate, layer.up, layer.down, ws.act,
                ws.h,
            )?;
            checkpoint(s, "moe_small")?;
        } else if quench6k_cuda::variants::get("flint.prefill_fp4", 0) != 0 {
            // W4A4: activations quantized to NVFP4 per row, FP4 tensor cores (as the Qwen prefill).
            let pairs = t * slots;
            let bm4 = k::fp4_tile_rows();
            let max_tiles = pairs.div_ceil(bm4) + layer.entries;
            fk::moe_plan(
                s,
                ws.sel_e,
                pairs,
                layer.entries,
                bm4,
                ws.counts,
                ws.cursor,
                ws.offsets,
                ws.tile_expert,
                ws.tile_row,
                ws.num_tiles,
                ws.pair_row,
            )?;
            fk::gather(s, ws.xn, ws.pair_row, pairs, slots, ws.xs)?;
            k::fp4_quant_rows(s, ws.xs, pairs, D, ws.q4)?;
            k::moe_fp4_gate_up(
                s,
                ws.q4,
                pairs,
                layer.gate,
                layer.up,
                ws.offsets,
                ws.tile_expert,
                ws.tile_row,
                ws.num_tiles,
                max_tiles,
                ws.actb,
                EW,
                D,
            )?;
            k::fp4_quant_rows(s, ws.actb, pairs, EW, ws.q4b)?;
            k::moe_fp4_gemm(
                s,
                ws.q4b,
                pairs,
                layer.down.0,
                layer.down.1,
                layer.down.2,
                ws.offsets,
                ws.tile_expert,
                ws.tile_row,
                ws.num_tiles,
                max_tiles,
                ws.yd,
                D,
                EW,
            )?;
            fk::combine(s, ws.yd, ws.pair_row, ws.sel_w, t, slots, ws.h)?;
        } else {
            let pairs = t * slots;
            let max_tiles = pairs.div_ceil(bm) + layer.entries;
            fk::moe_plan(
                s,
                ws.sel_e,
                pairs,
                layer.entries,
                bm,
                ws.counts,
                ws.cursor,
                ws.offsets,
                ws.tile_expert,
                ws.tile_row,
                ws.num_tiles,
                ws.pair_row,
            )?;
            fk::gather(s, ws.xn, ws.pair_row, pairs, slots, ws.xs)?;
            let gemm = |tb: Table, x: u64, y: u64, n: usize, kdim: usize| {
                k::moe_nvfp4_gemm(
                    s,
                    x,
                    tb.0,
                    tb.1,
                    tb.2,
                    ws.offsets,
                    ws.tile_expert,
                    ws.tile_row,
                    ws.num_tiles,
                    max_tiles,
                    y,
                    n,
                    kdim,
                )
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
    ensure!(
        rows <= ws.logit_rows,
        "{rows} logit rows requested, workspace holds {}",
        ws.logit_rows
    );
    let first = t - rows;
    let mut done = 0;
    while done < rows {
        let m = (rows - done).min(fk::gemv_max_m());
        let h = ws.h + ((first + done) * D * 4) as u64;
        if m == 1 && quench6k_cuda::variants::get("flint.fuse_norm", 1) != 0 {
            fk::gemv(
                s,
                h,
                Some(model.out_norm),
                1,
                &[model.lm_head],
                ws.logits + (done * VOCAB * 4) as u64,
                VOCAB,
                false,
            )?;
        } else {
            fk::rmsnorm(s, h, model.out_norm, EPS, ws.xn, m)?;
            fk::gemv(
                s,
                ws.xn,
                None,
                m,
                &[model.lm_head],
                ws.logits + (done * VOCAB * 4) as u64,
                VOCAB,
                false,
            )?;
        }
        done += m;
    }
    Ok(())
}

/// Greedy next token of each of the `rows` logit rows.
pub fn argmax(ws: &Workspace, rows: usize, s: &Stream) -> Result<Vec<i32>> {
    k::argmax(
        s,
        ws.logits,
        rows,
        VOCAB,
        VOCAB,
        ws.argmax_scratch,
        ws.argmax_out,
    )?;
    s.download::<i32>(ws.argmax_out, rows)
}

/// Rewinds `seq` to the canonical reuse point of `prompt` ([`FlintState::rewind`]) and prefills the
/// rest ([`FlintModel::prefill`]); logits of the last token are left in `ws`. Returns the reuse point.
pub fn prefill(
    model: &FlintModel,
    ws: &Workspace,
    seq: &mut FlintState,
    prompt: &[i32],
    s: &Stream,
) -> Result<usize> {
    let keep = seq.rewind(prompt, s)?;
    model.prefill(seq, ws, s, &prompt[keep..])?;
    Ok(keep)
}

impl FlintModel {
    /// Canonical prefill of `tokens` at seq.pos() (after [`FlintState::rewind`]): chunks end at
    /// multiples of [`CANON_CHUNK`] (absolute positions) and at the tokens' end; logits of the last
    /// token are left in `ws` (row 0).
    pub fn prefill(&self, seq: &mut FlintState, ws: &Workspace, s: &Stream, tokens: &[i32]) -> Result<()> {
        let mut at = 0;
        while at < tokens.len() {
            let start = seq.pos();
            let n = (CANON_CHUNK - start % CANON_CHUNK).min(tokens.len() - at);
            forward(self, ws, seq, &tokens[at..at + n], false, s)
                .with_context(|| format!("prefill chunk at {start}"))?;
            at += n;
        }
        Ok(())
    }
}
