//! Eager forward pass of the text model over a chunk of new tokens for one sequence.

use anyhow::{Result, ensure};
use squash_cuda::{DeviceBuffer, PinnedBuffer, Stream, kernels as k};

use crate::profile::Profiler;

/// Runs `$call` (a `Result<()>` expression) on stream `$s`, timed under `$name` when profiling.
macro_rules! op {
    ($prof:ident, $name:expr, $s:expr, $call:expr) => {
        match $prof.as_deref_mut() {
            Some(p) => p.time($name, $s, || $call)?,
            None => $call?,
        }
    };
}

use crate::model::{Block, D, EOS, EXPERT_WIDTH, EXPERTS, H, HC_RANK, Hc, LAYERS, Linear, Model, Proj, S, TOP_K, VOCAB, is_attention};

const QSA_HEADS: usize = 24;
const QSA_KV: usize = 2;
const QSA_DIM: usize = 256;
const INDEX_DIM: usize = 128;
const TOP_GROUPS: usize = 512;
const GDN_CONV: usize = 10240;
const GDN_HEADS: usize = 48;
const GDN_DIM: usize = 128;
const GDN_Z: usize = GDN_HEADS * GDN_DIM;
const PLE_HIST: usize = 9;
/// Widest projection group: GDN in_proj (10240 + 6144 + 48 + 48).
const PROJ_WIDTH: usize = GDN_CONV + GDN_Z + 2 * GDN_HEADS;

pub struct QsaCache {
    pub k: DeviceBuffer,
    pub v: DeviceBuffer,
    pub raw: DeviceBuffer,
    pub pooled: DeviceBuffer,
    /// K and V rows are FP8 E4M3 with a per-(position, KV head) scale (272 bytes) instead of BF16.
    pub fp8: bool,
}

/// KV cache format for sequences created from now on (false: BF16, true: FP8 E4M3 with scales).
pub static KV_FP8: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Bytes per position of one K or V cache.
pub fn kv_row_bytes(fp8: bool) -> usize {
    if fp8 { QSA_KV * (QSA_DIM + 16) } else { QSA_KV * QSA_DIM * 2 }
}

pub struct GdnState {
    pub conv: DeviceBuffer,
    pub recurrent: DeviceBuffer,
}

/// Everything a sequence carries between steps (SPEC §13).
pub struct SequenceState {
    pub capacity: usize,
    pub tokens: Vec<i32>,
    pub pos3: DeviceBuffer,
    pub qsa: Vec<Option<QsaCache>>,
    pub gdn: Vec<Option<GdnState>>,
    pub ple_conv: DeviceBuffer,
    /// The MTP drafter's own attention cache, when drafting.
    pub mtp_qsa: Option<QsaCache>,
    /// Speculative decoding: recurrent state is committed one step behind (see `Spec`).
    pub spec: Option<Spec>,
    /// Device f32: strength of the model's steering pack for this sequence (0: unsteered).
    pub steer_strength: DeviceBuffer,
    /// Image / video spans of the prompt, in order (their embeddings replace the placeholder tokens).
    pub mm: Vec<std::sync::Arc<MmSpan>>,
}

/// A run of placeholder tokens [start, start + len) whose embeddings come from the vision tower.
pub struct MmSpan {
    pub start: usize,
    /// Merger output, FP32 [rows, 2560]; this span's rows start at `row`.
    pub emb: std::sync::Arc<DeviceBuffer>,
    pub row: usize,
    /// (t, h, w) offsets of each token from the span's base position.
    pub offsets: Vec<[i32; 3]>,
    /// Positions the span occupies in the text sequence (max(h, w) of the merged grid).
    pub advance: i32,
}

impl MmSpan {
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

/// Recurrent-state bookkeeping for speculative decoding. A verification step processes drafts that may
/// be rejected, so GDN and PLE-conv kernels record each token's raw inputs instead of committing; the
/// next step first replays the accepted prefix (`ctl` = {tokens to replay, record parity}) onto the
/// committed state. Records are double-buffered by parity.
pub struct Spec {
    pub ctl: DeviceBuffer,
    pub gdn_rec: Vec<Option<DeviceBuffer>>,
    pub ple_rec: DeviceBuffer,
    /// Host copy of the control word.
    pub replay: i32,
    pub parity: i32,
}

impl SequenceState {
    pub fn new(capacity: usize) -> Result<Self> {
        Self::with_mtp(capacity, false)
    }

    pub fn with_mtp(capacity: usize, mtp: bool) -> Result<Self> {
        let capacity = capacity.div_ceil(4) * 4;
        let fp8 = KV_FP8.load(std::sync::atomic::Ordering::Relaxed);
        let qsa_cache = || -> Result<QsaCache> {
            Ok(QsaCache {
                k: DeviceBuffer::new(capacity * kv_row_bytes(fp8))?,
                v: DeviceBuffer::new(capacity * kv_row_bytes(fp8))?,
                raw: DeviceBuffer::new(capacity * INDEX_DIM * 2)?,
                pooled: DeviceBuffer::new(capacity / 4 * INDEX_DIM * 2)?,
                fp8,
            })
        };
        let mut qsa = Vec::with_capacity(LAYERS);
        let mut gdn = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            if is_attention(layer) {
                qsa.push(Some(qsa_cache()?));
                gdn.push(None);
            } else {
                qsa.push(None);
                gdn.push(Some(GdnState {
                    conv: DeviceBuffer::zeroed((3 * GDN_CONV + k::gdn_conv_tickets()) * 4)?,
                    recurrent: DeviceBuffer::zeroed(GDN_HEADS * GDN_DIM * GDN_DIM * 4)?,
                }));
            }
        }
        Ok(Self {
            capacity,
            tokens: Vec::new(),
            pos3: DeviceBuffer::new(capacity * 3 * 4)?,
            qsa,
            gdn,
            ple_conv: DeviceBuffer::zeroed(PLE_HIST * H * 4)?,
            mtp_qsa: if mtp { Some(qsa_cache()?) } else { None },
            spec: if mtp {
                Some(Spec {
                    ctl: DeviceBuffer::zeroed(8)?,
                    gdn_rec: (0..LAYERS).map(|l| if is_attention(l) { Ok(None) } else { DeviceBuffer::new(k::gdn_record_floats() * 4).map(Some) }).collect::<Result<_>>()?,
                    ple_rec: DeviceBuffer::new(k::ple_record_floats() * 4)?,
                    replay: 0,
                    parity: 0,
                })
            } else {
                None
            },
            steer_strength: DeviceBuffer::zeroed(4)?,
            mm: Vec::new(),
        })
    }

    /// Empties the sequence for a new prompt: recurrent and convolution state zeroed, speculative
    /// bookkeeping cleared (attention caches are position-indexed and need nothing). Buffer addresses
    /// stay, so graphs captured for this state remain valid.
    pub fn reset(&mut self, s: &Stream) -> Result<()> {
        self.tokens.clear();
        self.mm.clear();
        for g in self.gdn.iter().flatten() {
            s.memset(g.conv.addr(), 0, g.conv.bytes())?;
            s.memset(g.recurrent.addr(), 0, g.recurrent.bytes())?;
        }
        s.memset(self.ple_conv.addr(), 0, self.ple_conv.bytes())?;
        if let Some(sp) = &mut self.spec {
            s.memset(sp.ctl.addr(), 0, sp.ctl.bytes())?;
            sp.replay = 0;
            sp.parity = 0;
        }
        s.synchronize()
    }

    /// Sets this sequence's steering strength (read on the device by every later step).
    pub fn set_steering_strength(&self, s: &Stream, strength: f32) -> Result<()> {
        s.upload(&self.steer_strength, &[strength])?;
        s.synchronize()
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// MRoPE position (t, h, w) of sequence index `p`: text tokens advance all three axes by one;
    /// a vision span occupies `advance` positions from its base, its tokens at base + their offsets.
    pub fn position(&self, p: usize) -> [i32; 3] {
        let mut delta = 0i32;
        for sp in &self.mm {
            if p < sp.start {
                break;
            }
            if p < sp.start + sp.len() {
                let base = sp.start as i32 + delta;
                let o = sp.offsets[p - sp.start];
                return [base + o[0], base + o[1], base + o[2]];
            }
            delta += sp.advance - sp.len() as i32;
        }
        let v = p as i32 + delta;
        [v, v, v]
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Scratch buffers for steps of up to `max_tokens` tokens.
pub struct Workspace {
    pub max_tokens: usize,
    /// Activation capture for the next eager steps: the row at absolute position `pos` is recorded into
    /// `buf` (see `CaptureLayout`). Not for captured graphs.
    pub capture: Option<CaptureTarget>,
    score_cols: usize,
    bufs: Vec<DeviceBuffer>,
    pub h: u64,
    /// Output of a projection group, FP32 [T, <= PROJ_WIDTH].
    proj: u64,
    n: u64,
    low_bf: u64,
    up: u64,
    inj: u64,
    pub m: u64,
    y: u64,
    // GDN
    conv: u64,
    /// Chunked GDN recurrence scratch (prefill workspaces).
    gdn_chunks: u64,
    /// MXFP8 activations for the FP8 projections: codes [T, D] then scale bytes [T, D / 32].
    xq: u64,
    xq_bytes: usize,
    beta: u64,
    g: u64,
    o: u64,
    o_bf: u64,
    // QSA
    q: u64,
    iq: u64,
    scores: u64,
    sel: u64,
    att: u64,
    // MoE
    top_i: u64,
    top_w: u64,
    counts: u64,
    cursor: u64,
    offsets: u64,
    tile_expert: u64,
    tile_row: u64,
    num_tiles: u64,
    pair_row: u64,
    xs: u64,
    eg: u64,
    eu: u64,
    act: u64,
    yd: u64,
    sact: u64,
    sh: u64,
    // PLE
    kp: u64,
    pv: u64,
    gv: u64,
    xn: u64,
    /// Final mixer output, BF16 [T, D].
    pub out: u64,
    partial: u64,
    active: u64,
    num_active: u64,
    row_token: u64,
    att_part: u64,
    hc_r: u64,
    /// MTP: normalized embedding rows (BF16 [T, 2560]) and an all-ones [T, 4] used to add a row to every stream.
    e_bf: u64,
    ones: u64,
    /// MTP step block (start, tokens, positions; StepLayout without PLE rows) and its pinned staging,
    /// and the MTP input rows (FP32 [T, 10240]).
    pub(crate) mtp_step: u64,
    mtp_step_host: PinnedBuffer,
    pub mtp_hin: u64,
    inj2: u64,
    hc_scratch: u64,
    /// Device copy of the step block, and its pinned host staging.
    step: u64,
    step_host: PinnedBuffer,
}

impl Workspace {
    pub fn new(max_tokens: usize, capacity: usize) -> Result<Self> {
        let t = max_tokens;
        let pairs = t * TOP_K;
        let max_tiles = pairs.div_ceil(k::moe_tile_rows()) + EXPERTS;
        let score_cols = (capacity / 4).max(TOP_GROUPS).div_ceil(64) * 64;
        let mut bufs = Vec::new();
        let mut alloc = |bytes: usize| -> Result<u64> {
            let b = DeviceBuffer::new(bytes)?;
            let a = b.addr();
            bufs.push(b);
            Ok(a)
        };
        Self {
            max_tokens,
            capture: None,
            score_cols,
            h: alloc(t * H * 4)?,
            proj: alloc(t * PROJ_WIDTH * 4)?,
            n: alloc(t * H * 2)?,
            low_bf: alloc(t * HC_RANK * 2)?,
            up: alloc(t * H * 4)?,
            inj: alloc(t * S * 4)?,
            m: alloc(t * D * 2)?,
            y: alloc(t * D * 4)?,
            conv: alloc(t * GDN_CONV * 4)?,
            beta: alloc(t * GDN_HEADS * 4)?,
            g: alloc(t * GDN_HEADS * 4)?,
            gdn_chunks: alloc(if t >= 64 { k::gdn_chunk_scratch_floats(t).max(k::gdn_tc_scratch_floats(t)) * 4 } else { 4 })?,
            o: alloc(t * GDN_Z * 4)?,
            o_bf: alloc(t * GDN_Z * 2)?,
            q: alloc(t * QSA_HEADS * QSA_DIM * 4)?,
            iq: alloc(t * 4 * INDEX_DIM * 4)?,
            scores: alloc(t * score_cols * 4)?,
            sel: alloc(t * TOP_GROUPS * 4)?,
            att: alloc(t * QSA_HEADS * QSA_DIM * 2)?,
            top_i: alloc(pairs * 4)?,
            top_w: alloc(pairs * 4)?,
            counts: alloc(EXPERTS * 4)?,
            cursor: alloc(EXPERTS * 4)?,
            offsets: alloc((EXPERTS + 1) * 4)?,
            tile_expert: alloc(max_tiles * 4)?,
            tile_row: alloc(max_tiles * 4)?,
            num_tiles: alloc(4)?,
            pair_row: alloc(pairs * 4)?,
            xs: alloc(pairs * D * 2)?,
            eg: alloc(pairs * EXPERT_WIDTH * 4)?,
            eu: alloc(pairs * EXPERT_WIDTH * 4)?,
            act: alloc(pairs * EXPERT_WIDTH * 2)?,
            yd: alloc(pairs * D * 4)?,
            sact: alloc(t * EXPERT_WIDTH * 2)?,
            sh: alloc(t * D * 4)?,
            kp: alloc(t * H * 4)?,
            pv: alloc(t * D * 4)?,
            gv: alloc(t * H * 4)?,
            xn: alloc(t * H * 4)?,
            out: alloc(t * D * 2)?,
            partial: alloc(PARTIAL_FLOATS * 4)?,
            xq: alloc((t * D).div_ceil(256) * 256 + t * D / 32)?,
            xq_bytes: t * D,
            active: alloc(EXPERTS * 4)?,
            num_active: alloc(4)?,
            row_token: alloc(k::moe_small_tokens() * TOP_K * 4)?,
            hc_r: alloc(t * S * 4)?,
            e_bf: alloc(t * D * 2)?,
            ones: alloc(t * S * 4)?,
            mtp_step: alloc(StepLayout::new(t).ple)?,
            mtp_step_host: PinnedBuffer::new(StepLayout::new(t).ple)?,
            mtp_hin: alloc(t * H * 4)?,
            inj2: alloc(t * S * 4)?,
            hc_scratch: alloc(k::hc_small_scratch_floats(t.min(k::gemv_max_m())) * 4)?,
            att_part: alloc(k::qsa_split_floats(k::gemv_max_m()) * 4)?,
            step: alloc(StepLayout::new(t).bytes)?,
            step_host: PinnedBuffer::new(StepLayout::new(t).bytes)?,
            bufs,
        }
        .init()
    }

    pub fn bytes(&self) -> usize {
        self.bufs.iter().map(|b| b.bytes()).sum()
    }

    fn init(self) -> Result<Self> {
        Stream::new()?.upload_to(self.ones, &vec![1f32; self.max_tokens * S])?;
        Ok(self)
    }

    fn max_tiles(&self) -> usize {
        (self.max_tokens * TOP_K).div_ceil(k::moe_tile_rows()) + EXPERTS
    }
}

/// SMs on the target GPU; skinny GEMMs aim for at least two CTAs per SM.
const TARGET_CTAS: usize = 2 * 188;
const PARTIAL_FLOATS: usize = 1 << 22;

/// y[t, n] = x[t, k] · Wᵀ with row strides `ldx`/`ldy`: skinny GEMM for decode-sized `t`, tensor-core
/// GEMM otherwise.
#[allow(clippy::too_many_arguments)]
fn linear_ld(s: &Stream, partial: u64, x: u64, ldx: usize, w: &Linear, y: u64, ldy: usize, t: usize) -> Result<()> {
    // A few rows over 16 (MTP catch-up projects 4 stream rows per token) stay on the GEMV path in
    // 16-row slices; each row's arithmetic is the same either way.
    if t > k::gemv_max_m() && t <= 4 * k::gemv_max_m() {
        let m = k::gemv_max_m();
        for r0 in (0..t).step_by(m) {
            linear_ld(s, partial, x + (r0 * ldx * 2) as u64, ldx, w, y + (r0 * ldy * 4) as u64, ldy, m.min(t - r0))?;
        }
        return Ok(());
    }
    if t > k::gemv_max_m() && std::env::var_os("SQUASH_FAKE_MXFP8").is_some() && w.k % 32 == 0 {
        k::fake_mxfp8(s, x, t, w.k, ldx)?;
    }
    if t <= k::gemv_max_m() {
        // Split K only when even one row per warp leaves fewer CTAs than SMs.
        let ctas = w.n.div_ceil(8);
        let splits = if ctas >= TARGET_CTAS / 2 { 1 } else { (TARGET_CTAS / 2).div_ceil(ctas).clamp(1, w.k.div_ceil(512)).min(PARTIAL_FLOATS / (t * w.n)).max(1) };
        k::gemv(s, t, x, ldx, w.fmt, w.addr, w.scale, y, ldy, partial, w.n, w.k, splits, 1.0, false)
    } else {
        k::gemm_f32(s, x, ldx, w.fmt, w.addr, w.k, w.scale, y, ldy, t, w.n, w.k, 1.0, false)
    }
}

fn linear(s: &Stream, ws: &Workspace, x: u64, w: &Linear, y: u64, t: usize) -> Result<()> {
    linear_ws(s, ws, x, w.k, w, y, w.n, t)
}

/// As [`linear_ld`], with MXFP8 weights on prefill-sized batches: the activations are quantized to MXFP8
/// in the workspace and multiplied on the FP8 tensor cores.
#[allow(clippy::too_many_arguments)]
fn linear_ws(s: &Stream, ws: &Workspace, x: u64, ldx: usize, w: &Linear, y: u64, ldy: usize, t: usize) -> Result<()> {
    if w.fmt == k::WeightFormat::Mxfp8 && t > 4 * k::gemv_max_m() {
        ensure!(t * w.k <= ws.xq_bytes, "MXFP8 activation buffer holds {} bytes", ws.xq_bytes);
        let scales = ws.xq + (t * w.k).div_ceil(256) as u64 * 256;
        k::mxfp8_rows(s, x, t, w.k, ldx, ws.xq, scales)?;
        return k::gemm_mx8(s, ws.xq, scales, w.addr, w.scale, y, ldy, t, w.n, w.k);
    }
    linear_ld(s, ws.partial, x, ldx, w, y, ldy, t)
}

/// Computes every member of `proj` for `t` rows of `x` into `out` ([t, proj.n], FP32).
fn project(s: &Stream, ws: &Workspace, x: u64, proj: &Proj, out: u64, t: usize) -> Result<()> {
    match &proj.fused {
        Some(w) => linear_ws(s, ws, x, w.k, w, out, proj.n, t),
        None => {
            for (w, col) in &proj.parts {
                linear_ws(s, ws, x, w.k, w, out + (*col * 4) as u64, proj.n, t)?;
            }
            Ok(())
        }
    }
}

/// Mixes the residual through a HyperConnection: writes the BF16 block input to `out` and, when the HC
/// injects, the injection weights to `ws.inj`.
/// `pending`: the previous block's output, still to be added to the residual with the injection
/// weights in `ws.inj` (fused with the norm).
fn hyper_connection(s: &Stream, ws: &Workspace, hc: &Hc, t: usize, pending: Option<u64>, out: u64, mut prof: Option<&mut Profiler>) -> Result<()> {
    let inj = if hc.injects { ws.inj } else { 0 };
    {
        match pending {
            Some(y) => op!(prof, "hc combine+norm", s, k::hc_combine_norm(s, ws.h, y, ws.inj, ws.n, t)),
            None => op!(prof, "hc norm", s, k::grouped_rmsnorm(s, ws.h, 0, ws.n, t)),
        }
        op!(prof, "hc down", s, project(s, ws, ws.n, &hc.down, ws.proj, t));
        op!(prof, "hc act", s, k::hc_act(s, ws.proj, hc.down.n, ws.low_bf, inj, t));
        if t > 4 * k::gemv_max_m() && std::env::var_os("SQUASH_HC_UNFUSED").is_none() {
            op!(prof, "hc up+mix", s, k::gemm_hc_up_mix(s, ws.low_bf, hc.up.fmt, hc.up.addr, hc.up.scale, ws.n, hc.norm, out, t, hc.up.k));
        } else {
            op!(prof, "hc up", s, linear(s, ws, ws.low_bf, &hc.up, ws.up, t));
            op!(prof, "hc mix", s, k::hc_mix(s, ws.up, ws.n, hc.norm, out, t));
        }
    }
    Ok(())
}

/// Small-batch HyperConnection: one cooperative kernel. `pending` is the previous block's output and
/// injection weights, applied to the residual first; the new injection weights go to `inj_out`.
#[allow(clippy::too_many_arguments)]
fn hyper_connection_small(s: &Stream, ws: &Workspace, hc: &Hc, t: usize, pending: Option<(u64, u64)>, inj_out: u64, out: u64) -> Result<()> {
    let weight = |l: &Linear| (l.fmt, l.addr, l.scale);
    let down = weight(&hc.down.parts[0].0);
    let inject = hc.injects.then(|| weight(&hc.down.parts[1].0));
    let (y, inj_prev) = pending.unwrap_or((0, ws.inj));
    k::hc_small(s, ws.h, t, y, inj_prev, down, inject, weight(&hc.up), hc.norm, ws.hc_scratch, ws.low_bf, ws.hc_r, inj_out, out)
}

/// PLE row indices of the token at position `p` given the full token history (SPEC §8.1).
/// Faults the PLE rows of `tokens[from..]` into the page cache from `threads` threads, in order (blocks
/// of 64 positions), until done or `stop` is set. Run alongside a prefill, it turns the host gather's
/// page faults (the table is larger than RAM) into parallel reads ahead of the GPU.
pub fn warm_ple(model: &Model, tokens: &[i32], from: usize, threads: usize, stop: &std::sync::atomic::AtomicBool) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let Some(ple) = model.layers[1].ple.as_ref() else { return };
    const BLOCK: usize = 64;
    let next = AtomicUsize::new(from);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut sink = 0u8;
                loop {
                    let p0 = next.fetch_add(BLOCK, Ordering::Relaxed);
                    if p0 >= tokens.len() || stop.load(Ordering::Relaxed) {
                        break;
                    }
                    for p in p0..(p0 + BLOCK).min(tokens.len()) {
                        for r in ple_rows(tokens, p, &ple.multipliers, &ple.sizes, &ple.offsets) {
                            sink ^= model.ple_table.touch(r);
                        }
                    }
                }
                std::hint::black_box(sink);
            });
        }
    });
}

pub fn ple_rows(tokens: &[i32], p: usize, mult: &[i64; 3], sizes: &[i64; 16], offsets: &[i64; 16]) -> [usize; 16] {
    let s0 = tokens[p] as i64;
    let s1 = if p >= 1 && tokens[p - 1] != EOS { tokens[p - 1] } else { EOS } as i64;
    let s2 = if p >= 2 && tokens[p - 1] != EOS && tokens[p - 2] != EOS { tokens[p - 2] } else { EOS } as i64;
    let bigram = s0.wrapping_mul(mult[0]) ^ s1.wrapping_mul(mult[1]);
    let trigram = bigram ^ s2.wrapping_mul(mult[2]);
    std::array::from_fn(|j| {
        let x = if j < 8 { bigram } else { trigram };
        (x.rem_euclid(sizes[j]) + offsets[j]) as usize
    })
}

/// Where to record activations: the token at absolute position `pos`, into `buf`
/// ([`CaptureLayout::BYTES`] per layer, 48 layers).
#[derive(Clone, Copy, Debug)]
pub struct CaptureTarget {
    pub pos: usize,
    pub buf: u64,
}

/// Per-layer record of an activation capture (byte offsets): the pre-mix lane state (F32 [4, 2560],
/// before the attention-input HyperConnection), the mixed block input (BF16 [2560]), the attention or
/// GDN block output (F32 [2560]), the MoE output (F32 [2560]), the router logits (F32 [512]) and the
/// routed experts (I32 [10]) with their weights (F32 [10]).
pub struct CaptureLayout;

impl CaptureLayout {
    pub const PREMIX: usize = 0;
    pub const MIX: usize = Self::PREMIX + H * 4;
    pub const BLOCK: usize = Self::MIX + D * 2;
    pub const MOE: usize = Self::BLOCK + D * 4;
    pub const ROUTER: usize = Self::MOE + D * 4;
    pub const TOP_I: usize = Self::ROUTER + EXPERTS * 4;
    pub const TOP_W: usize = Self::TOP_I + TOP_K * 4;
    pub const BYTES: usize = Self::TOP_W + TOP_K * 4;
}

/// Byte offsets inside the step block: [start: i32 | pad][tokens: i32 x max][pos3: i32 x 3 x max][PLE rows: BF16 x D x max].
struct StepLayout {
    tokens: usize,
    pos: usize,
    ple: usize,
    bytes: usize,
}

impl StepLayout {
    fn new(max_tokens: usize) -> Self {
        let tokens = 16;
        let pos = tokens + 4 * max_tokens;
        let ple = (pos + 12 * max_tokens).div_ceil(16) * 16;
        Self { tokens, pos, ple, bytes: ple + 2 * D * max_tokens }
    }

    /// Bytes to transfer for a step of `t` tokens.
    fn used(&self, t: usize) -> usize {
        self.ple + 2 * D * t
    }
}

/// Fills the pinned step block for `new_tokens` (appending them to `state`): start position, token
/// IDs, positions, and the PLE rows gathered from the host table. Returns the bytes to transfer.
pub fn prepare(model: &Model, state: &mut SequenceState, ws: &mut Workspace, new_tokens: &[i32]) -> Result<usize> {
    prepare_tokens(state, ws, new_tokens)?;
    prepare_ple(model, state, ws, new_tokens.len())
}

/// First half of [`prepare`]: start position, token IDs and positions (appends to `state`).
pub fn prepare_tokens(state: &mut SequenceState, ws: &mut Workspace, new_tokens: &[i32]) -> Result<()> {
    let t = new_tokens.len();
    let start = state.len();
    ensure!(t > 0 && t <= ws.max_tokens, "step of {t} tokens exceeds the workspace ({})", ws.max_tokens);
    ensure!(start + t <= state.capacity, "sequence would exceed its capacity of {}", state.capacity);
    state.tokens.extend_from_slice(new_tokens);
    let layout = StepLayout::new(ws.max_tokens);
    let block = ws.step_host.as_mut_slice();
    block[0..4].copy_from_slice(&(start as i32).to_le_bytes());
    for (j, tok) in new_tokens.iter().enumerate() {
        block[layout.tokens + 4 * j..layout.tokens + 4 * j + 4].copy_from_slice(&tok.to_le_bytes());
        let pos = state.position(start + j);
        for a in 0..3 {
            let o = layout.pos + 12 * j + 4 * a;
            block[o..o + 4].copy_from_slice(&pos[a].to_le_bytes());
        }
    }
    Ok(())
}

/// Second half of [`prepare`]: gathers the PLE rows of the last `t` tokens of `state`. Returns the bytes
/// of step block to transfer.
pub fn prepare_ple(model: &Model, state: &SequenceState, ws: &mut Workspace, t: usize) -> Result<usize> {
    let start = state.len() - t;
    let layout = StepLayout::new(ws.max_tokens);
    let block = ws.step_host.as_mut_slice();
    let ple = model.layers[1].ple.as_ref().expect("layer 1 has PLE");
    let gather = |j0: usize, out: &mut [u8]| {
        let n = out.len() / (2 * D);
        let row_ids: Vec<[usize; 16]> = (j0..j0 + n).map(|j| ple_rows(&state.tokens, start + j, &ple.multipliers, &ple.sizes, &ple.offsets)).collect();
        for ids in &row_ids {
            for &r in ids {
                model.ple_table.prefetch(r);
            }
        }
        let mut rows = Vec::with_capacity(D);
        for (j, ids) in row_ids.iter().enumerate() {
            rows.clear();
            for &r in ids {
                model.ple_table.gather(r, &mut rows);
            }
            let o = 2 * D * j;
            for (d, v) in rows.iter().enumerate() {
                out[o + 2 * d..o + 2 * d + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    };
    let region = &mut block[layout.ple..layout.ple + 2 * D * t];
    // Large steps (prefill chunks) gather on several threads, each a contiguous range of tokens.
    const THREADS: usize = 16;
    if t >= 256 {
        let per = t.div_ceil(THREADS);
        std::thread::scope(|sc| {
            for (i, part) in region.chunks_mut(2 * D * per).enumerate() {
                let gather = &gather;
                sc.spawn(move || gather(i * per, part));
            }
        });
    } else {
        gather(0, region);
    }
    Ok(layout.used(t))
}

/// Prepares a batched decode step: token `tokens[b]` for sequence `states[b]` (at most 4 sequences).
/// Writes each sequence's start at word b of the step block, then tokens, positions and PLE rows per row.
pub fn prepare_batch(model: &Model, states: &mut [&mut SequenceState], ws: &mut Workspace, tokens: &[i32]) -> Result<usize> {
    let b = states.len();
    ensure!(b >= 1 && b <= 4 && tokens.len() == b && b <= ws.max_tokens, "batch of {b} sequences");
    let layout = StepLayout::new(ws.max_tokens);
    let ple = model.layers[1].ple.as_ref().expect("layer 1 has PLE");
    let mut rows = Vec::with_capacity(D);
    for (j, (st, &tok)) in states.iter_mut().zip(tokens).enumerate() {
        let p = st.len();
        ensure!(p < st.capacity, "sequence would exceed its capacity of {}", st.capacity);
        st.tokens.push(tok);
        let block = ws.step_host.as_mut_slice();
        block[4 * j..4 * j + 4].copy_from_slice(&(p as i32).to_le_bytes());
        block[layout.tokens + 4 * j..layout.tokens + 4 * j + 4].copy_from_slice(&tok.to_le_bytes());
        let pos = st.position(p);
        for a in 0..3 {
            let o = layout.pos + 12 * j + 4 * a;
            block[o..o + 4].copy_from_slice(&pos[a].to_le_bytes());
        }
        let ids = ple_rows(&st.tokens, p, &ple.multipliers, &ple.sizes, &ple.offsets);
        rows.clear();
        for &r in &ids {
            model.ple_table.gather(r, &mut rows);
        }
        let o = layout.ple + 2 * D * j;
        for (d, v) in rows.iter().enumerate() {
            block[o + 2 * d..o + 2 * d + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    Ok(layout.used(b))
}

/// Runs a step prepared by [`prepare_batch`] on the device (capturable): one token per sequence, the
/// weight-bound operations batched over the rows, stateful ones per sequence. `ws.out` row b holds
/// sequence b's final mixer output.
pub fn forward_batch(model: &Model, states: &[&SequenceState], ws: &Workspace, s: &Stream, bytes: usize) -> Result<()> {
    let t = states.len();
    let layout = StepLayout::new(ws.max_tokens);
    unsafe { s.copy_h2d(ws.step, ws.step_host.as_ptr(), bytes)? };
    for (b, st) in states.iter().enumerate() {
        k::store_positions(s, ws.step + (layout.pos + 12 * b) as u64, st.pos3.addr(), ws.step + 4 * b as u64, 1)?;
    }
    k::embed(s, ws.step + layout.tokens as u64, t, model.embed, ws.h)?;
    let ctx = LayerCtx {
        ws,
        s,
        t,
        start: ws.step,
        pos3: 0,
        capacity: states[0].capacity,
        ple_rows: ws.step + layout.ple as u64,
        ple_state: 0,
        spec: None,
        steer: None,
        batch: Some(states),
        capture: None,
    };
    let mut carry = Carry::new(ws, false);
    for i in 0..model.layers.len() {
        layer_forward(i, &model.layers[i], None, None, None, &ctx, &mut carry, None)?;
    }
    final_mixer(&model.final_hc, &ctx, &mut carry, None)
}

/// Commits a speculative sequence's deferred recurrent state (the accepted tokens of its last round,
/// recorded but not yet applied) so that it can continue with plain steps.
pub fn flush_spec(model: &Model, state: &mut SequenceState, ws: &Workspace, s: &Stream) -> Result<()> {
    let Some(sp) = state.spec.as_ref() else { return Ok(()) };
    for (i, layer) in model.layers.iter().enumerate() {
        if let (Block::Gdn(g), Some(st), Some(rec)) = (&layer.block, state.gdn[i].as_ref(), sp.gdn_rec[i].as_ref()) {
            let flush = k::Deferral { rec: rec.addr(), ctl: sp.ctl.addr(), defer: false };
            k::gdn_fused_small(s, ws.proj, g.proj.n, 0, g.conv, st.conv.addr(), st.recurrent.addr(), g.a_log, g.dt_bias, g.norm, g.proj.col(1), g.proj.col(2), g.proj.col(3), ws.o_bf, Some(flush))?;
        }
        if let Some(ple) = &layer.ple {
            let flush = k::Deferral { rec: sp.ple_rec.addr(), ctl: sp.ctl.addr(), defer: false };
            k::ple_conv(s, ws.gv, ws.xn, ple.conv, state.ple_conv.addr(), ws.h, 0, Some(flush))?;
        }
    }
    // Nothing left to replay.
    s.memset(sp.ctl.addr(), 0, 4)?;
    s.synchronize()?;
    if let Some(sp) = state.spec.as_mut() {
        sp.replay = 0;
    }
    Ok(())
}

/// A captured batched decode step for fixed sequences (their buffers are baked in): the step block
/// is prepared on the host with [`prepare_batch`], then one graph runs the forward pass and the
/// greedy argmax of every row into `out` (i32 per sequence).
pub struct BatchGraph {
    graph: squash_cuda::Graph,
    bytes: usize,
    rows: usize,
}

impl BatchGraph {
    pub fn capture(model: &Model, states: &[&SequenceState], ws: &Workspace, hw: &HeadWorkspace, s: &Stream, out: u64) -> Result<Self> {
        let bytes = StepLayout::new(ws.max_tokens).used(states.len());
        let graph = squash_cuda::Graph::capture(s, |s| {
            forward_batch(model, states, ws, s, bytes)?;
            head_argmax_into(model, hw, s, ws.out, states.len(), out)
        })?;
        Ok(Self { graph, bytes, rows: states.len() })
    }

    /// One step: `tokens[b]` for `states[b]` (the captured sequences, in order); the next greedy tokens
    /// are in `out` after the stream synchronizes.
    pub fn step(&self, model: &Model, states: &mut [&mut SequenceState], ws: &mut Workspace, s: &Stream, tokens: &[i32]) -> Result<()> {
        ensure!(states.len() == self.rows, "graph captured for {} sequences", self.rows);
        let bytes = prepare_batch(model, states, ws, tokens)?;
        ensure!(bytes == self.bytes, "step layout changed");
        self.graph.launch(s)
    }
}

/// Runs the step prepared by [`prepare`] (`t` tokens, `bytes` of step block) entirely on the device:
/// no host synchronization, so it can be captured into a CUDA graph. Afterwards `ws.out` holds the final
/// mixer output, BF16 [T, D].
pub fn forward_device(model: &Model, state: &SequenceState, ws: &Workspace, s: &Stream, t: usize, bytes: usize, prof: Option<&mut Profiler>) -> Result<()> {
    forward_part(model, state, ws, s, t, bytes, Part::Whole, false, prof)
}

/// A slice of a step: the whole step, or the part before PLE (the step header, embedding and layer 0,
/// which need only token IDs) and the rest (PLE rows, layers 1..48, final mixer).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Whole,
    BeforePle,
    FromPle,
}

#[allow(clippy::too_many_arguments)]
/// With `defer` (speculative verification, needs `state.spec`), the recurrent layers record this step's
/// tokens instead of committing them; see [`Spec`].
pub fn forward_part(model: &Model, state: &SequenceState, ws: &Workspace, s: &Stream, t: usize, bytes: usize, part: Part, defer: bool, mut prof: Option<&mut Profiler>) -> Result<()> {
    let layout = StepLayout::new(ws.max_tokens);
    match part {
        Part::Whole => unsafe { s.copy_h2d(ws.step, ws.step_host.as_ptr(), bytes)? },
        Part::BeforePle => unsafe { s.copy_h2d(ws.step, ws.step_host.as_ptr(), layout.ple)? },
        Part::FromPle => unsafe { s.copy_h2d(ws.step + layout.ple as u64, ws.step_host.as_ptr().add(layout.ple), bytes - layout.ple)? },
    }
    let spec = state.spec.as_ref().map(|sp| (sp.ctl.addr(), sp.ple_rec.addr(), defer));
    let steer = model.steering.as_ref().map(|p| (p.directions.addr(), p.ranks.addr(), p.masks.addr(), p.norm_preserve, state.steer_strength.addr()));
    let first = state.len() - t;
    let capture = ws.capture.filter(|c| c.pos >= first && c.pos < first + t).map(|c| (c.buf, c.pos - first));
    let ctx = LayerCtx { ws, s, t, start: ws.step, pos3: state.pos3.addr(), capacity: state.capacity, ple_rows: ws.step + layout.ple as u64, ple_state: state.ple_conv.addr(), spec, steer, batch: None, capture };
    if part != Part::FromPle {
        op!(prof, "store_positions", s, k::store_positions(s, ws.step + layout.pos as u64, ctx.pos3, ctx.start, t));
        op!(prof, "embed", s, k::embed(s, ws.step + layout.tokens as u64, t, model.embed, ws.h));
        for sp in &state.mm {
            let (a, b) = (sp.start.max(first), (sp.start + sp.len()).min(first + t));
            if a < b {
                k::embed_rows(s, sp.emb.addr() + ((sp.row + a - sp.start) * D * 4) as u64, b - a, ws.h + ((a - first) * H * 4) as u64)?;
            }
        }
    }
    // After layer 0 of a small batch the deferred update is the MoE output in ws.y with weights in ws.inj.
    let mut carry = Carry::new(ws, part == Part::FromPle && t <= k::gemv_max_m());
    let layers = match part {
        Part::Whole => 0..model.layers.len(),
        Part::BeforePle => 0..1,
        Part::FromPle => 1..model.layers.len(),
    };
    for i in layers {
        let rec = state.spec.as_ref().and_then(|sp| sp.gdn_rec[i].as_ref().map(|b| b.addr()));
        layer_forward(i, &model.layers[i], state.gdn[i].as_ref(), rec, state.qsa[i].as_ref(), &ctx, &mut carry, prof.as_deref_mut())?;
    }
    if part == Part::BeforePle {
        return Ok(());
    }
    final_mixer(&model.final_hc, &ctx, &mut carry, prof)
}

/// What one step's layers share.
struct LayerCtx<'a> {
    ws: &'a Workspace,
    s: &'a Stream,
    t: usize,
    /// Device i32: position of the step's first token.
    start: u64,
    pos3: u64,
    capacity: usize,
    /// Gathered PLE rows (BF16 [t, 2560]) and the PLE conv state, for the layer that has PLE.
    ple_rows: u64,
    ple_state: u64,
    /// Speculative bookkeeping: control word, PLE records, and whether this step defers its commit.
    spec: Option<(u64, u64, bool)>,
    /// Steering: (directions, ranks, masks, norm_preserve, strength) when the model has a pack.
    steer: Option<(u64, u64, u64, bool, u64)>,
    /// Batched decode: row b is the next token of sequence b (with its start position at
    /// `start + 4 b`); stateful operations run per sequence.
    batch: Option<&'a [&'a SequenceState]>,
    /// Activation capture: (record buffer, row of this step).
    capture: Option<(u64, usize)>,
}

/// Residual bookkeeping carried from layer to layer. Small batches defer each block's residual update to
/// the next HyperConnection kernel; `inj` and `inj_next` alternate so that kernel reads the previous
/// injection weights while writing its own.
struct Carry {
    pending: Option<u64>,
    inj: u64,
    inj_next: u64,
}

impl Carry {
    fn new(ws: &Workspace, after_layer0: bool) -> Self {
        Self { pending: if after_layer0 { Some(ws.y) } else { None }, inj: ws.inj, inj_next: ws.inj2 }
    }
}

fn final_mixer(hc: &Hc, ctx: &LayerCtx, carry: &mut Carry, mut prof: Option<&mut Profiler>) -> Result<()> {
    let (ws, s, t) = (ctx.ws, ctx.s, ctx.t);
    if t <= k::gemv_max_m() {
        op!(prof, "hc_small", s, hyper_connection_small(s, ws, hc, t, carry.pending.take().map(|y| (y, carry.inj)), carry.inj_next, ws.out));
        Ok(())
    } else {
        hyper_connection(s, ws, hc, t, carry.pending.take(), ws.out, prof)
    }
}

/// One decoder layer (SPEC §3) over the step's tokens, with the layer's own GDN or QSA state.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn layer_forward(index: usize, layer: &crate::model::Layer, gdn: Option<&GdnState>, gdn_rec: Option<u64>, qsa: Option<&QsaCache>, ctx: &LayerCtx, carry: &mut Carry, mut prof: Option<&mut Profiler>) -> Result<()> {
    let (ws, s, t, start, pos3) = (ctx.ws, ctx.s, ctx.t, ctx.start, ctx.pos3);
    let small = t <= k::gemv_max_m();
    if let Some(ple) = &layer.ple {
        if let Some(y) = carry.pending.take() {
            op!(prof, "hc_combine", s, k::hc_combine(s, ws.h, y, carry.inj, t));
        }
        op!(prof, "ple.key", s, linear(s, ws, ctx.ple_rows, &ple.key, ws.kp, t));
        op!(prof, "ple.value", s, linear(s, ws, ctx.ple_rows, &ple.value, ws.pv, t));
        op!(prof, "ple_gate", s, k::ple_gate(s, ws.kp, ws.h, ws.pv, ple.norm_key, ple.norm_query, ple.norm_conv, ws.gv, ws.xn, t));
        let spec = ctx.spec.map(|(ctl, rec, defer)| k::Deferral { rec, ctl, defer });
        match ctx.batch {
            None => op!(prof, "ple_conv", s, k::ple_conv(s, ws.gv, ws.xn, ple.conv, ctx.ple_state, ws.h, t, spec)),
            Some(seqs) => {
                for (b, st) in seqs.iter().enumerate() {
                    let row = (b * H * 4) as u64;
                    op!(prof, "ple_conv", s, k::ple_conv(s, ws.gv + row, ws.xn + row, ple.conv, st.ple_conv.addr(), ws.h + row, 1, None));
                }
            }
        }
    }

    // Steering at the attention-input pre-mix point: the pending block output is committed first
    // (the same FMA the HC kernels would apply), so a zero strength leaves every value unchanged.
    if let Some((dirs, ranks, masks, norm_preserve, strength)) = ctx.steer.filter(|_| index < LAYERS) {
        if let Some(y) = carry.pending.take() {
            op!(prof, "hc_combine", s, k::hc_combine(s, ws.h, y, carry.inj, t));
        }
        op!(prof, "steer", s, k::steer(s, ws.h, t, dirs, ranks, masks, norm_preserve, strength, index));
    }
    let rec = ctx.capture.filter(|_| index < LAYERS).map(|(buf, row)| (buf + (index * CaptureLayout::BYTES) as u64, row));
    if let Some((buf, row)) = rec {
        if let Some(y) = carry.pending.take() {
            op!(prof, "hc_combine", s, k::hc_combine(s, ws.h, y, carry.inj, t));
        }
        s.copy_d2d(buf + CaptureLayout::PREMIX as u64, ws.h + (row * H * 4) as u64, H * 4)?;
    }
    if small {
        op!(prof, "hc_small", s, hyper_connection_small(s, ws, &layer.attn_hc, t, carry.pending.map(|y| (y, carry.inj)), carry.inj_next, ws.m));
        std::mem::swap(&mut carry.inj, &mut carry.inj_next);
    } else {
        hyper_connection(s, ws, &layer.attn_hc, t, carry.pending.take(), ws.m, prof.as_deref_mut())?;
    }
    if let Some((buf, row)) = rec {
        s.copy_d2d(buf + CaptureLayout::MIX as u64, ws.m + (row * D * 2) as u64, D * 2)?;
    }
    match &layer.block {
        Block::Gdn(g) => {
            let st = || gdn.expect("GDN layer state");
            let (ld, x) = (g.proj.n, ws.proj);
            let col = |j: usize| x + (g.proj.col(j) * 4) as u64;
            if t > k::gemv_max_m() && fake_set("gdn") {
                k::fake_mxfp8(s, ws.m, t, D, D)?;
            }
            op!(prof, "gdn proj", s, project(s, ws, ws.m, &g.proj, x, t));
            let spec = match (ctx.spec, gdn_rec) {
                (Some((ctl, _, defer)), Some(rec)) => Some(k::Deferral { rec, ctl, defer }),
                _ => None,
            };
            if let Some(seqs) = ctx.batch {
                for (b, sq) in seqs.iter().enumerate() {
                    let sg = sq.gdn[index].as_ref().expect("GDN layer state");
                    let (xr, yr) = (x + (b * ld * 4) as u64, ws.o_bf + (b * GDN_Z * 2) as u64);
                    op!(prof, "gdn_fused_small", s, k::gdn_fused_small(s, xr, ld, 1, g.conv, sg.conv.addr(), sg.recurrent.addr(), g.a_log, g.dt_bias, g.norm, g.proj.col(1), g.proj.col(2), g.proj.col(3), yr, None));
                }
            } else if t <= k::gemv_max_m() {
                op!(prof, "gdn_fused_small", s, k::gdn_fused_small(s, x, ld, t, g.conv, st().conv.addr(), st().recurrent.addr(), g.a_log, g.dt_bias, g.norm, g.proj.col(1), g.proj.col(2), g.proj.col(3), ws.o_bf, spec));
            } else {
                // The chunked prefill kernels work on committed state: replay any pending records first.
                if let Some(sp) = spec {
                    op!(prof, "gdn_flush", s, k::gdn_fused_small(s, x, ld, 0, g.conv, st().conv.addr(), st().recurrent.addr(), g.a_log, g.dt_bias, g.norm, g.proj.col(1), g.proj.col(2), g.proj.col(3), ws.o_bf, Some(k::Deferral { defer: false, ..sp })));
                }
                op!(prof, "gdn_conv", s, k::gdn_conv(s, col(0), ld, ws.conv, st().conv.addr(), g.conv, t));
                op!(prof, "gdn_prep", s, k::gdn_prep(s, ws.conv, col(2), col(3), ld, g.a_log, g.dt_bias, ws.beta, ws.g, t));
                if t >= 64 && gdn_tc() {
                    op!(prof, "gdn_chunked_tc", s, k::gdn_chunked_tc(s, ws.conv, ws.beta, ws.g, st().recurrent.addr(), ws.o, t, ws.gdn_chunks));
                } else {
                    op!(prof, "gdn_recurrent", s, k::gdn_recurrent(s, ws.conv, ws.beta, ws.g, st().recurrent.addr(), ws.o, t, ws.gdn_chunks));
                }
                op!(prof, "gdn_gated_norm", s, k::gdn_gated_norm(s, ws.o, col(1), ld, g.norm, ws.o_bf, t));
            }
            op!(prof, "gdn out", s, linear(s, ws, ws.o_bf, &g.out, ws.y, t));
        }
        Block::Qsa(q) => {
            let (ld, x) = (q.proj.n, ws.proj);
            let col = |j: usize| x + (q.proj.col(j) * 4) as u64;
            op!(prof, "qsa proj", s, project(s, ws, ws.m, &q.proj, x, t));
            if let Some(seqs) = ctx.batch {
                for (b, sq) in seqs.iter().enumerate() {
                    let c = sq.qsa[index].as_ref().expect("QSA layer cache");
                    let (row, start_b, pos3_b) = ((b * ld * 4) as u64, start + 4 * b as u64, sq.pos3.addr());
                    let colr = |j: usize| col(j) + row;
                    let (qr, iqr) = (ws.q + (b * QSA_HEADS * QSA_DIM * 4) as u64, ws.iq + (b * 4 * INDEX_DIM * 4) as u64);
                    let (scores, sel) = (ws.scores + (b * ws.score_cols * 4) as u64, ws.sel + (b * TOP_GROUPS * 4) as u64);
                    let att = ws.att + (b * QSA_HEADS * QSA_DIM * 2) as u64;
                    op!(prof, "qsa_qkv_prep", s, k::qsa_qkv_prep(s, colr(0), colr(1), ld, q.q_norm, q.k_norm, qr, c.k.addr(), c.v.addr(), pos3_b, start_b, 1, c.fp8));
                    op!(prof, "qsa_index_prep", s, k::qsa_index_prep(s, colr(3), ld, q.index_q_norm, iqr, c.raw.addr(), pos3_b, start_b, 1));
                    op!(prof, "qsa_pool", s, k::qsa_pool(s, c.raw.addr(), q.index_k_norm, c.pooled.addr(), pos3_b, start_b, 1));
                    if sq.capacity / 4 > TOP_GROUPS {
                        op!(prof, "qsa_index_scores", s, k::qsa_index_scores(s, iqr, c.pooled.addr(), scores, ws.score_cols, start_b, 1));
                        op!(prof, "qsa_select", s, k::qsa_select(s, scores, 1, ws.score_cols, start_b, sel));
                    }
                    op!(prof, "qsa_attention_split", s, k::qsa_attention_split(s, qr, colr(0), ld, c.k.addr(), c.v.addr(), sel, start_b, 1, c.fp8, ws.att_part, att));
                }
                op!(prof, "qsa o", s, linear(s, ws, ws.att, &q.o, ws.y, t));
                return finish_layer(layer, ctx, carry, prof, None);
            }
            let c = qsa.expect("QSA layer cache");
            op!(prof, "qsa_qkv_prep", s, k::qsa_qkv_prep(s, col(0), col(1), ld, q.q_norm, q.k_norm, ws.q, c.k.addr(), c.v.addr(), pos3, start, t, c.fp8));
            op!(prof, "qsa_index_prep", s, k::qsa_index_prep(s, col(3), ld, q.index_q_norm, ws.iq, c.raw.addr(), pos3, start, t));
            op!(prof, "qsa_pool", s, k::qsa_pool(s, c.raw.addr(), q.index_k_norm, c.pooled.addr(), pos3, start, t));
            // Rows whose query sees at most 512 complete groups attend densely and skip selection.
            if ctx.capacity / 4 > TOP_GROUPS {
                op!(prof, "qsa_index_scores", s, k::qsa_index_scores(s, ws.iq, c.pooled.addr(), ws.scores, ws.score_cols, start, t));
                op!(prof, "qsa_select", s, k::qsa_select(s, ws.scores, t, ws.score_cols, start, ws.sel));
            }
            if t <= k::gemv_max_m() {
                op!(prof, "qsa_attention_split", s, k::qsa_attention_split(s, ws.q, col(0), ld, c.k.addr(), c.v.addr(), ws.sel, start, t, c.fp8, ws.att_part, ws.att));
            } else {
                op!(prof, "qsa_attention", s, k::qsa_attention(s, ws.q, col(0), ld, c.k.addr(), c.v.addr(), ws.sel, start, t, c.fp8, ws.att));
            }
            op!(prof, "qsa o", s, linear(s, ws, ws.att, &q.o, ws.y, t));
        }
    }
    if let Some((buf, row)) = rec {
        s.copy_d2d(buf + CaptureLayout::BLOCK as u64, ws.y + (row * D * 4) as u64, D * 4)?;
    }
    finish_layer(layer, ctx, carry, prof, rec)
}

/// Prefill experts on the FP4 tensor cores (W4A4, activations quantized per row with least-error block
/// scales): the default (gate PASS); SQUASH_PREFILL_FP4=0 keeps BF16 activations.
/// Prefill GDN on the tensor cores (3xTF32; the default, SQUASH_GDN_TC=0 selects the CUDA-core kernels).
fn gdn_tc() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SQUASH_GDN_TC").map_or(true, |v| v != "0"))
}

/// Experiment: whether projection class `c` is fake-quantized to MXFP8 (SQUASH_FAKE_MXFP8_SET).
fn fake_set(c: &str) -> bool {
    std::env::var("SQUASH_FAKE_MXFP8_SET").is_ok_and(|v| v.split(',').any(|x| x == c))
}

pub fn fp4_prefill() -> bool {
    fp4_mode() != 0
}

/// 0: off, 1: gate/up and down on FP4, 2: gate/up only (down BF16 on fake-quantized input), 3: down
/// only, 4: gate/up only (down BF16).
fn fp4_mode() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SQUASH_PREFILL_FP4").ok().and_then(|v| v.parse().ok()).unwrap_or(1))
}

/// The second half of a layer: the MLP HyperConnection, the MoE block and its output.
fn finish_layer(layer: &crate::model::Layer, ctx: &LayerCtx, carry: &mut Carry, mut prof: Option<&mut Profiler>, rec: Option<(u64, usize)>) -> Result<()> {
    let (ws, s, t) = (ctx.ws, ctx.s, ctx.t);
    let small = t <= k::gemv_max_m();
    if small {
        op!(prof, "hc_small", s, hyper_connection_small(s, ws, &layer.mlp_hc, t, Some((ws.y, carry.inj)), carry.inj_next, ws.m));
        std::mem::swap(&mut carry.inj, &mut carry.inj_next);
    } else {
        hyper_connection(s, ws, &layer.mlp_hc, t, Some(ws.y), ws.m, prof.as_deref_mut())?;
    }
    let moe = &layer.moe;
    let (ld, x) = (moe.proj.n, ws.proj);
    let col = |j: usize| x + (moe.proj.col(j) * 4) as u64;
    op!(prof, "moe proj", s, project(s, ws, ws.m, &moe.proj, x, t));
    let tables = |p: &crate::model::ExpertProj| (p.weights, p.scales, p.scale2);
    // FP4 prefill: the routed experts' down projection is written in BF16 (SQUASH_YD_BF16=0: FP32).
    let yd_bf16 = t > k::moe_small_tokens() && matches!(fp4_mode(), 1 | 3) && std::env::var("SQUASH_YD_BF16").map_or(true, |v| v != "0");
    if t == 1 {
        op!(prof, "experts t1", s, k::moe_experts_t1(s, ws.m, col(0), tables(&moe.gate), tables(&moe.up), tables(&moe.down), ws.top_i, ws.top_w, ws.pair_row, ws.act, ws.yd));
    } else if t <= k::moe_small_tokens() {
        let max_active = (t * TOP_K).min(EXPERTS);
        op!(prof, "moe_route_small", s, k::moe_route_small(s, col(0), ld, t, ws.top_i, ws.top_w, ws.offsets, ws.pair_row, ws.row_token, ws.active, ws.num_active));
        op!(prof, "experts gate_up", s, k::moe_gate_up_small(s, ws.m, ws.row_token, tables(&moe.gate), tables(&moe.up), ws.offsets, ws.active, ws.num_active, max_active, t, ws.act));
        op!(prof, "experts down", s, k::moe_down_small(s, ws.act, tables(&moe.down), ws.offsets, ws.active, ws.num_active, max_active, t, ws.yd));
    } else {
        op!(prof, "moe_route", s, k::moe_route(s, col(0), ld, t, ws.top_i, ws.top_w));
        let tile_rows = if fp4_prefill() && fp4_mode() != 3 { k::fp4_tile_rows() } else { k::moe_tile_rows() };
        op!(prof, "moe_plan", s, k::moe_plan(s, ws.top_i, t, ws.counts, ws.cursor, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.pair_row, tile_rows));
        let tiles = ws.max_tiles().min((t * TOP_K).div_ceil(tile_rows) + EXPERTS);
        let grouped = |proj: &crate::model::ExpertProj, x: u64, y: u64, n: usize, kk: usize| {
            k::moe_nvfp4_gemm(s, x, proj.weights, proj.scales, proj.scale2, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, tiles, y, n, kk)
        };
        let fake_fp4 = std::env::var_os("SQUASH_FAKE_FP4").is_some();
        if fp4_prefill() {
            // W4A4: quantize each token once (into yd, free until the down projection), gather the
            // quantized rows by expert (into xs), FP4 tensor-core GEMMs; the SwiGLU output is
            // quantized into eg (free once SwiGLU has read it).
            let pairs = t * TOP_K;
            let mode = fp4_mode();
            if mode != 3 {
                op!(prof, "fp4 quant", s, k::fp4_quant_rows(s, ws.m, t, D, ws.yd));
                op!(prof, "fp4 gather", s, k::fp4_gather(s, ws.yd, t, D, ws.pair_row, ws.xs));
                op!(prof, "experts gate_up", s, k::moe_fp4_gate_up(s, ws.xs, pairs, tables(&moe.gate), tables(&moe.up), ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, tiles, ws.act, EXPERT_WIDTH, D));
            } else {
                k::moe_gather(s, ws.m, ws.pair_row, t, ws.xs)?;
                k::fake_fp4_act(s, ws.xs, pairs, D)?;
                grouped(&moe.gate, ws.xs, ws.eg, EXPERT_WIDTH, D)?;
                grouped(&moe.up, ws.xs, ws.eu, EXPERT_WIDTH, D)?;
            }
            if mode == 3 {
                op!(prof, "swiglu", s, k::swiglu(s, ws.eg, ws.eu, EXPERT_WIDTH, EXPERT_WIDTH, ws.act, pairs));
            }
            if mode == 1 || mode == 3 {
                op!(prof, "fp4 quant", s, k::fp4_quant_rows(s, ws.act, pairs, EXPERT_WIDTH, ws.eg));
                let down = &moe.down;
                op!(prof, "experts down", s, k::moe_fp4_gemm_out(s, ws.eg, pairs, down.weights, down.scales, down.scale2, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, tiles, ws.yd, D, EXPERT_WIDTH, yd_bf16));
            } else {
                if mode == 2 {
                    k::fake_fp4_act(s, ws.act, pairs, EXPERT_WIDTH)?;
                }
                op!(prof, "experts down", s, grouped(&moe.down, ws.act, ws.yd, D, EXPERT_WIDTH));
            }
        } else {
        op!(prof, "moe_gather", s, k::moe_gather(s, ws.m, ws.pair_row, t, ws.xs));
        if fake_fp4 {
            k::fake_fp4_act(s, ws.xs, t * TOP_K, D)?;
        }
        op!(prof, "experts gate", s, grouped(&moe.gate, ws.xs, ws.eg, EXPERT_WIDTH, D));
        op!(prof, "experts up", s, grouped(&moe.up, ws.xs, ws.eu, EXPERT_WIDTH, D));
        op!(prof, "swiglu", s, k::swiglu(s, ws.eg, ws.eu, EXPERT_WIDTH, EXPERT_WIDTH, ws.act, t * TOP_K));
        if fake_fp4 {
            k::fake_fp4_act(s, ws.act, t * TOP_K, EXPERT_WIDTH)?;
        }
        op!(prof, "experts down", s, grouped(&moe.down, ws.act, ws.yd, D, EXPERT_WIDTH));
        }
    }
    // One token: the shared-expert tail (SwiGLU, down projection, combine) in one kernel. Larger
    // batches are faster as three kernels; the arithmetic is the same either way.
    if t == 1 {
        let down = (moe.shared_down.fmt, moe.shared_down.addr, moe.shared_down.scale);
        op!(prof, "moe_shared_combine", s, k::moe_shared_combine(s, x, ld, t, down, ws.yd, ws.pair_row, ws.top_w, ws.y));
    } else {
        op!(prof, "shared swiglu", s, k::swiglu(s, col(1), col(2), ld, EXPERT_WIDTH, ws.sact, t));
        op!(prof, "shared down", s, linear(s, ws, ws.sact, &moe.shared_down, ws.sh, t));
        if yd_bf16 {
            op!(prof, "moe_combine", s, k::moe_combine_bf16(s, ws.yd, ws.pair_row, ws.top_w, col(3), ld, ws.sh, ws.y, t));
        } else {
            op!(prof, "moe_combine", s, k::moe_combine(s, ws.yd, ws.pair_row, ws.top_w, col(3), ld, ws.sh, ws.y, t));
        }
    }
    if let Some((buf, row)) = rec {
        s.copy_d2d(buf + CaptureLayout::MOE as u64, ws.y + (row * D * 4) as u64, D * 4)?;
        s.copy_d2d(buf + CaptureLayout::ROUTER as u64, col(0) + (row * ld * 4) as u64, EXPERTS * 4)?;
        s.copy_d2d(buf + CaptureLayout::TOP_I as u64, ws.top_i + (row * TOP_K * 4) as u64, TOP_K * 4)?;
        s.copy_d2d(buf + CaptureLayout::TOP_W as u64, ws.top_w + (row * TOP_K * 4) as u64, TOP_K * 4)?;
    }
    // The block output is added to the residual by whatever reads it next (the next HyperConnection
    // fuses it with its norm).
    carry.pending = Some(ws.y);
    Ok(())
}

/// Stages an MTP step: `tokens` at positions `start ..` (written to the position table by the step).
pub fn prepare_mtp(ws: &mut Workspace, start: usize, tokens: &[i32]) -> Result<()> {
    ensure!(!tokens.is_empty() && tokens.len() <= ws.max_tokens, "MTP step of {} tokens exceeds the workspace", tokens.len());
    let layout = StepLayout::new(ws.max_tokens);
    let block = ws.mtp_step_host.as_mut_slice();
    block[0..4].copy_from_slice(&(start as i32).to_le_bytes());
    for (j, tok) in tokens.iter().enumerate() {
        block[layout.tokens + 4 * j..layout.tokens + 4 * j + 4].copy_from_slice(&tok.to_le_bytes());
        for a in 0..3 {
            let o = layout.pos + 12 * j + 4 * a;
            block[o..o + 4].copy_from_slice(&((start + j) as i32).to_le_bytes());
        }
    }
    Ok(())
}

/// Runs the MTP drafter (SPEC §10) over the `t` tokens staged by [`prepare_mtp`], paired with the
/// preceding hidden rows `hprev` (FP32 [t, 10240], each the pre-mixer residual of the position before
/// its token). Afterwards `ws.out` holds the MTP mixer output (BF16 [t, 2560]) and `ws.h` the MTP
/// pre-mixer residual, the `hprev` of the next draft step.
pub fn forward_mtp(model: &Model, state: &SequenceState, ws: &Workspace, s: &Stream, t: usize, hprev: u64, prof: Option<&mut Profiler>) -> Result<()> {
    let layout = StepLayout::new(ws.max_tokens);
    unsafe { s.copy_h2d(ws.mtp_step, ws.mtp_step_host.as_ptr(), layout.ple)? };
    forward_mtp_staged(model, state, ws, s, t, hprev, prof)
}

/// Offsets of the token IDs and positions inside a step block.
pub fn step_offsets(ws: &Workspace) -> (usize, usize) {
    let l = StepLayout::new(ws.max_tokens);
    (l.tokens, l.pos)
}

/// As [`forward_mtp`], with the MTP step block already on the device at `ws.mtp_step`.
pub fn forward_mtp_staged(model: &Model, state: &SequenceState, ws: &Workspace, s: &Stream, t: usize, hprev: u64, mut prof: Option<&mut Profiler>) -> Result<()> {
    let mtp = model.mtp.as_ref().expect("model loaded without MTP");
    let cache = state.mtp_qsa.as_ref().expect("sequence without MTP cache");
    let layout = StepLayout::new(ws.max_tokens);
    let start = ws.mtp_step;
    op!(prof, "store_positions", s, k::store_positions(s, ws.mtp_step + layout.pos as u64, state.pos3.addr(), start, t));
    op!(prof, "mtp embed", s, k::embed(s, ws.mtp_step + layout.tokens as u64, t, model.embed, ws.h));
    op!(prof, "mtp norm e", s, k::rmsnorm_rows(s, ws.h, H, mtp.norm_embedding, ws.e_bf, t));
    op!(prof, "mtp norm h", s, k::grouped_rmsnorm(s, hprev, mtp.norm_hidden, ws.n, t));
    op!(prof, "mtp fc_e", s, linear(s, ws, ws.e_bf, &mtp.fc_embedding, ws.y, t));
    // fc_hidden maps each of the 4 streams: rows of the [t, 4, 2560] input are the 4t stream slices.
    op!(prof, "mtp fc_h", s, linear_ld(s, ws.partial, ws.n, D, &mtp.fc_hidden, ws.h, D, S * t));
    op!(prof, "mtp combine", s, k::hc_combine(s, ws.h, ws.y, ws.ones, t));
    let ctx = LayerCtx { ws, s, t, start, pos3: state.pos3.addr(), capacity: state.capacity, ple_rows: 0, ple_state: 0, spec: None, steer: None, batch: None, capture: None };
    let mut carry = Carry::new(ws, false);
    layer_forward(LAYERS, &mtp.layer, None, None, Some(cache), &ctx, &mut carry, prof.as_deref_mut())?;
    final_mixer(&mtp.final_hc, &ctx, &mut carry, prof)
}

/// Eager step with per-op GPU timing collected into `prof`.
pub fn forward_profiled(model: &Model, state: &mut SequenceState, ws: &mut Workspace, s: &Stream, new_tokens: &[i32], prof: &mut Profiler) -> Result<()> {
    let bytes = prepare(model, state, ws, new_tokens)?;
    forward_device(model, state, ws, s, new_tokens.len(), bytes, Some(prof))?;
    s.synchronize()?;
    prof.collect()
}

/// Bytes of step block a step of `t` tokens transfers.
pub fn step_bytes(ws: &Workspace, t: usize) -> usize {
    StepLayout::new(ws.max_tokens).used(t)
}

/// Eager step: [`prepare`] then [`forward_device`].
pub fn forward(model: &Model, state: &mut SequenceState, ws: &mut Workspace, s: &Stream, new_tokens: &[i32]) -> Result<()> {
    let t0 = std::time::Instant::now();
    let bytes = prepare(model, state, ws, new_tokens)?;
    let t1 = std::time::Instant::now();
    forward_device(model, state, ws, s, new_tokens.len(), bytes, None)?;
    if new_tokens.len() > 1000 && std::env::var_os("SQUASH_TIME_PREP").is_some() {
        s.synchronize()?;
        eprintln!("forward {} tokens: prepare {:.1} ms, device {:.1} ms", new_tokens.len(), (t1 - t0).as_secs_f64() * 1e3, t1.elapsed().as_secs_f64() * 1e3);
    }
    // The pinned step block may be overwritten by the next prepare only after this step consumed it.
    s.synchronize()
}

/// A one-token decode step captured as a CUDA graph for a given sequence: forward pass, LM head and
/// greedy argmax, with the chosen token copied to pinned host memory.
///
/// The step is split into two graphs around PLE: the first (embedding, layer 0) is launched as soon as
/// the token is known, and the host gathers the token's PLE rows while it runs.
pub struct DecodeGraph {
    before_ple: squash_cuda::Graph,
    from_ple: squash_cuda::Graph,
    bytes: usize,
}

impl DecodeGraph {
    pub fn capture(model: &Model, state: &SequenceState, ws: &Workspace, hw: &HeadWorkspace, s: &Stream) -> Result<Self> {
        let bytes = StepLayout::new(ws.max_tokens).used(1);
        let before_ple = squash_cuda::Graph::capture(s, |s| forward_part(model, state, ws, s, 1, bytes, Part::BeforePle, false, None))?;
        let from_ple = squash_cuda::Graph::capture(s, |s| {
            forward_part(model, state, ws, s, 1, bytes, Part::FromPle, false, None)?;
            head_argmax(model, hw, s, ws.out)
        })?;
        Ok(Self { before_ple, from_ple, bytes })
    }

    /// Decodes `token` (appended to `state`) and returns the greedy next token.
    pub fn step(&self, model: &Model, state: &mut SequenceState, ws: &mut Workspace, hw: &HeadWorkspace, s: &Stream, token: i32) -> Result<i32> {
        self.step_timed(model, state, ws, hw, s, token, &mut [0.0; 2])
    }

    /// As [`step`](Self::step), adding host preparation and graph execution seconds to `times`.
    #[allow(clippy::too_many_arguments)]
    pub fn step_timed(&self, model: &Model, state: &mut SequenceState, ws: &mut Workspace, hw: &HeadWorkspace, s: &Stream, token: i32, times: &mut [f64; 2]) -> Result<i32> {
        let t0 = std::time::Instant::now();
        prepare_tokens(state, ws, &[token])?;
        self.before_ple.launch(s)?;
        let bytes = prepare_ple(model, state, ws, 1)?;
        debug_assert_eq!(bytes, self.bytes);
        let t1 = std::time::Instant::now();
        self.from_ple.launch(s)?;
        s.synchronize()?;
        times[0] += (t1 - t0).as_secs_f64();
        times[1] += t1.elapsed().as_secs_f64();
        Ok(i32::from_le_bytes(hw.token_host.as_slice()[..4].try_into().unwrap()))
    }
}

/// Greedy tokens of `rows` final hidden states (BF16 [rows, 2560]) written to the device i32s at `out`.
pub fn head_argmax_into(model: &Model, hw: &HeadWorkspace, s: &Stream, hidden: u64, rows: usize, out: u64) -> Result<()> {
    ensure!(rows <= hw.rows, "head workspace holds {} rows", hw.rows);
    linear_ld(s, hw.partial.addr(), hidden, D, &model.lm_head, hw.logits.addr(), VOCAB, rows)?;
    k::argmax(s, hw.logits.addr(), rows, VOCAB, VOCAB, hw.argmax.addr(), out)
}

/// Greedy draft token for one hidden state, written to `out`: a 1-bit sketch of the LM head shortlists
/// 8 candidates per 8192 rows and the drafter's NVFP4 head rescores them (the NVFP4 argmax whenever
/// it is shortlisted, which is nearly always).
pub fn draft_head_argmax_into(model: &Model, hw: &HeadWorkspace, s: &Stream, hidden: u64, out: u64) -> Result<()> {
    let mtp = model.mtp.as_ref().expect("model loaded without MTP");
    k::draft_head_argmax(s, hidden, mtp.draft_sketch, mtp.draft_head, hw.logits.addr(), hw.pick.addr(), VOCAB, out)
}

/// LM-head logits of `rows` consecutive hidden rows into the head workspace.
pub fn head_logits_rows(model: &Model, hw: &HeadWorkspace, s: &Stream, hidden: u64, rows: usize) -> Result<()> {
    ensure!(rows <= hw.rows, "head workspace holds {} rows", hw.rows);
    linear_ld(s, hw.partial.addr(), hidden, D, &model.lm_head, hw.logits.addr(), VOCAB, rows)
}

/// LM-head logits of one hidden row into the head workspace (row 0).
pub fn head_logits(model: &Model, hw: &HeadWorkspace, s: &Stream, hidden: u64) -> Result<()> {
    linear_ld(s, hw.partial.addr(), hidden, D, &model.lm_head, hw.logits.addr(), VOCAB, 1)
}

/// Greedy next token for one hidden state: LM head, argmax, and a copy of the token to `hw.token_host`.
pub fn head_argmax(model: &Model, hw: &HeadWorkspace, s: &Stream, hidden: u64) -> Result<()> {
    linear_ld(s, hw.partial.addr(), hidden, D, &model.lm_head, hw.logits.addr(), VOCAB, 1)?;
    k::argmax(s, hw.logits.addr(), 1, VOCAB, VOCAB, hw.argmax.addr(), hw.top_ids.addr())?;
    unsafe { s.copy_d2h(hw.token_host.as_ptr(), hw.top_ids.addr(), 4) }
}

/// LM-head statistics for `rows` final hidden states at `hidden` (BF16 [rows, D]): log-sum-exp,
/// the logit of `targets[r]`, and the top-`top_k` (ids, logits) sorted by logit.
pub struct HeadStats {
    pub lse: Vec<f32>,
    pub target: Vec<f32>,
    pub top_ids: Vec<i32>,
    pub top_logits: Vec<f32>,
}

pub struct HeadWorkspace {
    rows: usize,
    partial: DeviceBuffer,
    argmax: DeviceBuffer,
    token_host: PinnedBuffer,
    logits: DeviceBuffer,
    targets: DeviceBuffer,
    lse: DeviceBuffer,
    target: DeviceBuffer,
    top_ids: DeviceBuffer,
    top_logits: DeviceBuffer,
    pick: DeviceBuffer,
}

impl HeadWorkspace {
    /// Device address of the logits ([rows, VOCAB] F32).
    pub fn logits_addr(&self) -> u64 {
        self.logits.addr()
    }

    pub fn new(rows: usize, top_k: usize) -> Result<Self> {
        Ok(Self {
            rows,
            partial: DeviceBuffer::new(PARTIAL_FLOATS * 4)?,
            argmax: DeviceBuffer::new(k::argmax_scratch(rows, VOCAB))?,
            token_host: PinnedBuffer::new(4)?,
            logits: DeviceBuffer::new(rows * VOCAB * 4)?,
            targets: DeviceBuffer::new(rows * 4)?,
            lse: DeviceBuffer::new(rows * 4)?,
            target: DeviceBuffer::new(rows * 4)?,
            top_ids: DeviceBuffer::new(rows * top_k * 4)?,
            top_logits: DeviceBuffer::new(rows * top_k * 4)?,
            pick: DeviceBuffer::new(k::draft_pick_words(VOCAB) * 4)?,
        })
    }
}

pub fn head_stats(model: &Model, hw: &HeadWorkspace, s: &Stream, hidden: u64, targets: &[i32], top_k: usize) -> Result<HeadStats> {
    let mut out = HeadStats { lse: vec![], target: vec![], top_ids: vec![], top_logits: vec![] };
    for r0 in (0..targets.len()).step_by(hw.rows) {
        let n = hw.rows.min(targets.len() - r0);
        linear_ld(s, hw.partial.addr(), hidden + (r0 * D * 2) as u64, D, &model.lm_head, hw.logits.addr(), VOCAB, n)?;
        s.upload_to(hw.targets.addr(), &targets[r0..r0 + n])?;
        k::row_lse(s, hw.logits.addr(), n, VOCAB, VOCAB, hw.targets.addr(), hw.lse.addr(), hw.target.addr())?;
        k::topk(s, hw.logits.addr(), n, VOCAB, VOCAB, top_k, false, hw.top_ids.addr(), hw.top_logits.addr())?;
        out.lse.extend(s.download::<f32>(hw.lse.addr(), n)?);
        out.target.extend(s.download::<f32>(hw.target.addr(), n)?);
        out.top_ids.extend(s.download::<i32>(hw.top_ids.addr(), n * top_k)?);
        out.top_logits.extend(s.download::<f32>(hw.top_logits.addr(), n * top_k)?);
    }
    Ok(out)
}
