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
        let qsa_cache = || -> Result<QsaCache> {
            Ok(QsaCache {
                k: DeviceBuffer::new(capacity * QSA_KV * QSA_DIM * 2)?,
                v: DeviceBuffer::new(capacity * QSA_KV * QSA_DIM * 2)?,
                raw: DeviceBuffer::new(capacity * INDEX_DIM * 2)?,
                pooled: DeviceBuffer::new(capacity / 4 * INDEX_DIM * 2)?,
            })
        };
        let mut qsa = Vec::with_capacity(LAYERS);
        let mut gdn = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            if is_attention(layer) {
                qsa.push(Some(QsaCache {
                    k: DeviceBuffer::new(capacity * QSA_KV * QSA_DIM * 2)?,
                    v: DeviceBuffer::new(capacity * QSA_KV * QSA_DIM * 2)?,
                    raw: DeviceBuffer::new(capacity * INDEX_DIM * 2)?,
                    pooled: DeviceBuffer::new(capacity / 4 * INDEX_DIM * 2)?,
                }));
                gdn.push(None);
            } else {
                qsa.push(None);
                gdn.push(Some(GdnState {
                    conv: DeviceBuffer::zeroed(3 * GDN_CONV * 4)?,
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
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Scratch buffers for steps of up to `max_tokens` tokens.
pub struct Workspace {
    pub max_tokens: usize,
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
    linear_ld(s, ws.partial, x, w.k, w, y, w.n, t)
}

/// Computes every member of `proj` for `t` rows of `x` into `out` ([t, proj.n], FP32).
fn project(s: &Stream, ws: &Workspace, x: u64, proj: &Proj, out: u64, t: usize) -> Result<()> {
    match &proj.fused {
        Some(w) => linear_ld(s, ws.partial, x, w.k, w, out, proj.n, t),
        None => {
            for (w, col) in &proj.parts {
                linear_ld(s, ws.partial, x, w.k, w, out + (*col * 4) as u64, proj.n, t)?;
            }
            Ok(())
        }
    }
}

/// Mixes the residual through a HyperConnection: writes the BF16 block input to `out` and, when the HC
/// injects, the injection weights to `ws.inj`.
fn hyper_connection(s: &Stream, ws: &Workspace, hc: &Hc, t: usize, out: u64, mut prof: Option<&mut Profiler>) -> Result<()> {
    let inj = if hc.injects { ws.inj } else { 0 };
    {
        op!(prof, "hc norm", s, k::grouped_rmsnorm(s, ws.h, 0, ws.n, t));
        op!(prof, "hc down", s, project(s, ws, ws.n, &hc.down, ws.proj, t));
        op!(prof, "hc act", s, k::hc_act(s, ws.proj, hc.down.n, ws.low_bf, inj, t));
        op!(prof, "hc up", s, linear(s, ws, ws.low_bf, &hc.up, ws.up, t));
        op!(prof, "hc mix", s, k::hc_mix(s, ws.up, ws.n, hc.norm, out, t));
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
        for a in 0..3 {
            let o = layout.pos + 12 * j + 4 * a;
            block[o..o + 4].copy_from_slice(&((start + j) as i32).to_le_bytes());
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
    let row_ids: Vec<[usize; 16]> = (0..t).map(|j| ple_rows(&state.tokens, start + j, &ple.multipliers, &ple.sizes, &ple.offsets)).collect();
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
        let o = layout.ple + 2 * D * j;
        for (d, v) in rows.iter().enumerate() {
            block[o + 2 * d..o + 2 * d + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    Ok(layout.used(t))
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
    let ctx = LayerCtx { ws, s, t, start: ws.step, pos3: state.pos3.addr(), capacity: state.capacity, ple_rows: ws.step + layout.ple as u64, ple_state: state.ple_conv.addr(), spec };
    if part != Part::FromPle {
        op!(prof, "store_positions", s, k::store_positions(s, ws.step + layout.pos as u64, ctx.pos3, ctx.start, t));
        op!(prof, "embed", s, k::embed(s, ws.step + layout.tokens as u64, t, model.embed, ws.h));
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
        layer_forward(&model.layers[i], state.gdn[i].as_ref(), rec, state.qsa[i].as_ref(), &ctx, &mut carry, prof.as_deref_mut())?;
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
        hyper_connection(s, ws, hc, t, ws.out, prof)
    }
}

/// One decoder layer (SPEC §3) over the step's tokens, with the layer's own GDN or QSA state.
#[allow(clippy::too_many_arguments)]
fn layer_forward(layer: &crate::model::Layer, gdn: Option<&GdnState>, gdn_rec: Option<u64>, qsa: Option<&QsaCache>, ctx: &LayerCtx, carry: &mut Carry, mut prof: Option<&mut Profiler>) -> Result<()> {
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
        op!(prof, "ple_conv", s, k::ple_conv(s, ws.gv, ws.xn, ple.conv, ctx.ple_state, ws.h, t, spec));
    }

    if small {
        op!(prof, "hc_small", s, hyper_connection_small(s, ws, &layer.attn_hc, t, carry.pending.map(|y| (y, carry.inj)), carry.inj_next, ws.m));
        std::mem::swap(&mut carry.inj, &mut carry.inj_next);
    } else {
        hyper_connection(s, ws, &layer.attn_hc, t, ws.m, prof.as_deref_mut())?;
    }
    match &layer.block {
        Block::Gdn(g) => {
            let st = gdn.expect("GDN layer state");
            let (ld, x) = (g.proj.n, ws.proj);
            let col = |j: usize| x + (g.proj.col(j) * 4) as u64;
            op!(prof, "gdn proj", s, project(s, ws, ws.m, &g.proj, x, t));
            let spec = match (ctx.spec, gdn_rec) {
                (Some((ctl, _, defer)), Some(rec)) => Some(k::Deferral { rec, ctl, defer }),
                _ => None,
            };
            if t <= k::gemv_max_m() {
                op!(prof, "gdn_fused_small", s, k::gdn_fused_small(s, x, ld, t, g.conv, st.conv.addr(), st.recurrent.addr(), g.a_log, g.dt_bias, g.norm, g.proj.col(1), g.proj.col(2), g.proj.col(3), ws.o_bf, spec));
            } else {
                // The chunked prefill kernels work on committed state: replay any pending records first.
                if let Some(sp) = spec {
                    op!(prof, "gdn_flush", s, k::gdn_fused_small(s, x, ld, 0, g.conv, st.conv.addr(), st.recurrent.addr(), g.a_log, g.dt_bias, g.norm, g.proj.col(1), g.proj.col(2), g.proj.col(3), ws.o_bf, Some(k::Deferral { defer: false, ..sp })));
                }
                op!(prof, "gdn_conv", s, k::gdn_conv(s, col(0), ld, ws.conv, st.conv.addr(), g.conv, t));
                op!(prof, "gdn_prep", s, k::gdn_prep(s, ws.conv, col(2), col(3), ld, g.a_log, g.dt_bias, ws.beta, ws.g, t));
                op!(prof, "gdn_recurrent", s, k::gdn_recurrent(s, ws.conv, ws.beta, ws.g, st.recurrent.addr(), ws.o, t));
                op!(prof, "gdn_gated_norm", s, k::gdn_gated_norm(s, ws.o, col(1), ld, g.norm, ws.o_bf, t));
            }
            op!(prof, "gdn out", s, linear(s, ws, ws.o_bf, &g.out, ws.y, t));
        }
        Block::Qsa(q) => {
            let c = qsa.expect("QSA layer cache");
            let (ld, x) = (q.proj.n, ws.proj);
            let col = |j: usize| x + (q.proj.col(j) * 4) as u64;
            op!(prof, "qsa proj", s, project(s, ws, ws.m, &q.proj, x, t));
            op!(prof, "qsa_qkv_prep", s, k::qsa_qkv_prep(s, col(0), col(1), ld, q.q_norm, q.k_norm, ws.q, c.k.addr(), c.v.addr(), pos3, start, t));
            op!(prof, "qsa_index_prep", s, k::qsa_index_prep(s, col(3), ld, q.index_q_norm, ws.iq, c.raw.addr(), pos3, start, t));
            op!(prof, "qsa_pool", s, k::qsa_pool(s, c.raw.addr(), q.index_k_norm, c.pooled.addr(), pos3, start, t));
            // Rows whose query sees at most 512 complete groups attend densely and skip selection.
            if ctx.capacity / 4 > TOP_GROUPS {
                op!(prof, "qsa_index_scores", s, k::qsa_index_scores(s, ws.iq, c.pooled.addr(), ws.scores, ws.score_cols, start, t));
                op!(prof, "qsa_select", s, k::qsa_select(s, ws.scores, t, ws.score_cols, start, ws.sel));
            }
            if t <= k::gemv_max_m() {
                op!(prof, "qsa_attention_split", s, k::qsa_attention_split(s, ws.q, col(0), ld, c.k.addr(), c.v.addr(), ws.sel, start, t, ws.att_part, ws.att));
            } else {
                op!(prof, "qsa_attention", s, k::qsa_attention(s, ws.q, col(0), ld, c.k.addr(), c.v.addr(), ws.sel, start, t, ws.att));
            }
            op!(prof, "qsa o", s, linear(s, ws, ws.att, &q.o, ws.y, t));
        }
    }
    if small {
        op!(prof, "hc_small", s, hyper_connection_small(s, ws, &layer.mlp_hc, t, Some((ws.y, carry.inj)), carry.inj_next, ws.m));
        std::mem::swap(&mut carry.inj, &mut carry.inj_next);
    } else {
        op!(prof, "hc_combine", s, k::hc_combine(s, ws.h, ws.y, ws.inj, t));
        hyper_connection(s, ws, &layer.mlp_hc, t, ws.m, prof.as_deref_mut())?;
    }
    let moe = &layer.moe;
    let (ld, x) = (moe.proj.n, ws.proj);
    let col = |j: usize| x + (moe.proj.col(j) * 4) as u64;
    op!(prof, "moe proj", s, project(s, ws, ws.m, &moe.proj, x, t));
    let tables = |p: &crate::model::ExpertProj| (p.weights, p.scales, p.scale2);
    if t == 1 {
        op!(prof, "experts t1", s, k::moe_experts_t1(s, ws.m, col(0), tables(&moe.gate), tables(&moe.up), tables(&moe.down), ws.top_i, ws.top_w, ws.pair_row, ws.act, ws.yd));
    } else if t <= k::moe_small_tokens() {
        let max_active = (t * TOP_K).min(EXPERTS);
        op!(prof, "moe_route_small", s, k::moe_route_small(s, col(0), ld, t, ws.top_i, ws.top_w, ws.offsets, ws.pair_row, ws.row_token, ws.active, ws.num_active));
        op!(prof, "experts gate_up", s, k::moe_gate_up_small(s, ws.m, ws.row_token, tables(&moe.gate), tables(&moe.up), ws.offsets, ws.active, ws.num_active, max_active, t, ws.act));
        op!(prof, "experts down", s, k::moe_down_small(s, ws.act, tables(&moe.down), ws.offsets, ws.active, ws.num_active, max_active, t, ws.yd));
    } else {
        op!(prof, "moe_route", s, k::moe_route(s, col(0), ld, t, ws.top_i, ws.top_w));
        op!(prof, "moe_plan", s, k::moe_plan(s, ws.top_i, t, ws.counts, ws.cursor, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.pair_row));
        op!(prof, "moe_gather", s, k::moe_gather(s, ws.m, ws.pair_row, t, ws.xs));
        let tiles = ws.max_tiles().min((t * TOP_K).div_ceil(k::moe_tile_rows()) + EXPERTS);
        let grouped = |proj: &crate::model::ExpertProj, x: u64, y: u64, n: usize, kk: usize| {
            k::moe_nvfp4_gemm(s, x, proj.weights, proj.scales, proj.scale2, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, tiles, y, n, kk)
        };
        op!(prof, "experts gate", s, grouped(&moe.gate, ws.xs, ws.eg, EXPERT_WIDTH, D));
        op!(prof, "experts up", s, grouped(&moe.up, ws.xs, ws.eu, EXPERT_WIDTH, D));
        op!(prof, "swiglu", s, k::swiglu(s, ws.eg, ws.eu, EXPERT_WIDTH, EXPERT_WIDTH, ws.act, t * TOP_K));
        op!(prof, "experts down", s, grouped(&moe.down, ws.act, ws.yd, D, EXPERT_WIDTH));
    }
    // The fused shared-expert tail recomputes the SwiGLU input per warp: only worth it for one token.
    if t == 1 {
        let down = (moe.shared_down.fmt, moe.shared_down.addr, moe.shared_down.scale);
        op!(prof, "moe_shared_combine", s, k::moe_shared_combine(s, x, ld, t, down, ws.yd, ws.pair_row, ws.top_w, ws.y));
    } else {
        op!(prof, "shared swiglu", s, k::swiglu(s, col(1), col(2), ld, EXPERT_WIDTH, ws.sact, t));
        op!(prof, "shared down", s, linear(s, ws, ws.sact, &moe.shared_down, ws.sh, t));
        op!(prof, "moe_combine", s, k::moe_combine(s, ws.yd, ws.pair_row, ws.top_w, col(3), ld, ws.sh, ws.y, t));
    }
    if small {
        carry.pending = Some(ws.y);
    } else {
        op!(prof, "hc_combine", s, k::hc_combine(s, ws.h, ws.y, ws.inj, t));
    }
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
    let ctx = LayerCtx { ws, s, t, start, pos3: state.pos3.addr(), capacity: state.capacity, ple_rows: 0, ple_state: 0, spec: None };
    let mut carry = Carry::new(ws, false);
    layer_forward(&mtp.layer, None, None, Some(cache), &ctx, &mut carry, prof.as_deref_mut())?;
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
    let bytes = prepare(model, state, ws, new_tokens)?;
    forward_device(model, state, ws, s, new_tokens.len(), bytes, None)?;
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

/// Greedy draft token for one hidden state through the drafter's NVFP4 LM head, written to `out`.
pub fn draft_head_argmax_into(model: &Model, hw: &HeadWorkspace, s: &Stream, hidden: u64, out: u64) -> Result<()> {
    let (codes, scales, scale2) = model.mtp.as_ref().expect("model loaded without MTP").draft_head;
    k::nvfp4_dense_gemv(s, hidden, codes, scales, scale2, hw.logits.addr(), VOCAB)?;
    k::argmax(s, hw.logits.addr(), 1, VOCAB, VOCAB, hw.argmax.addr(), out)
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
}

impl HeadWorkspace {
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
