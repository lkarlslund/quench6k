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
/// K splits of the small-batch HC down projection: 5 per 2560-wide stream (512 columns each).
const HC_SPLITS: usize = 20;
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
}

impl SequenceState {
    pub fn new(capacity: usize) -> Result<Self> {
        let capacity = capacity.div_ceil(4) * 4;
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
    /// Device copy of the step block, and its pinned host staging.
    step: u64,
    step_host: PinnedBuffer,
}

impl Workspace {
    pub fn new(max_tokens: usize, capacity: usize) -> Result<Self> {
        let t = max_tokens;
        let pairs = t * TOP_K;
        let max_tiles = pairs.div_ceil(k::moe_tile_rows()) + EXPERTS;
        let score_cols = (capacity / 4).max(TOP_GROUPS);
        let mut bufs = Vec::new();
        let mut alloc = |bytes: usize| -> Result<u64> {
            let b = DeviceBuffer::new(bytes)?;
            let a = b.addr();
            bufs.push(b);
            Ok(a)
        };
        Ok(Self {
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
            att_part: alloc(k::qsa_split_floats(k::gemv_max_m()) * 4)?,
            step: alloc(StepLayout::new(t).bytes)?,
            step_host: PinnedBuffer::new(StepLayout::new(t).bytes)?,
            bufs,
        })
    }

    pub fn bytes(&self) -> usize {
        self.bufs.iter().map(|b| b.bytes()).sum()
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
    if t <= k::gemv_max_m() {
        let ctas = w.n.div_ceil(16);
        let splits = TARGET_CTAS.div_ceil(ctas).clamp(1, w.k.div_ceil(256)).min(PARTIAL_FLOATS / (t * w.n)).max(1);
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
    if t <= k::gemv_max_m() {
        // Project the raw FP32 residual with stream-aligned K splits; the finish kernel applies each
        // stream's RMS factor while reducing them.
        op!(prof, "hc down", s, {
            let parts: Vec<(Linear, usize)> = match &hc.down.fused {
                Some(w) => vec![(*w, 0)],
                None => hc.down.parts.clone(),
            };
            parts.iter().try_for_each(|(w, col)| k::gemv_partials_f32(s, t, ws.h, H, w.fmt, w.addr, w.scale, ws.partial + (*col * 4) as u64, hc.down.n, w.n, H, HC_SPLITS))
        });
        op!(prof, "hc finish", s, k::hc_finish(s, ws.partial, HC_SPLITS, t, hc.down.n, ws.h, ws.low_bf, inj, ws.hc_r));
        op!(prof, "hc up", s, linear(s, ws, ws.low_bf, &hc.up, ws.up, t));
        op!(prof, "hc mix", s, k::hc_mix_small(s, ws.up, ws.h, ws.hc_r, hc.norm, out, t));
    } else {
        op!(prof, "hc norm", s, k::grouped_rmsnorm(s, ws.h, 0, ws.n, t));
        op!(prof, "hc down", s, project(s, ws, ws.n, &hc.down, ws.proj, t));
        op!(prof, "hc act", s, k::hc_act(s, ws.proj, hc.down.n, ws.low_bf, inj, t));
        op!(prof, "hc up", s, linear(s, ws, ws.low_bf, &hc.up, ws.up, t));
        op!(prof, "hc mix", s, k::hc_mix(s, ws.up, ws.n, hc.norm, out, t));
    }
    Ok(())
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
pub fn forward_device(model: &Model, state: &SequenceState, ws: &Workspace, s: &Stream, t: usize, bytes: usize, mut prof: Option<&mut Profiler>) -> Result<()> {
    let layout = StepLayout::new(ws.max_tokens);
    unsafe { s.copy_h2d(ws.step, ws.step_host.as_ptr(), bytes)? };
    let start = ws.step;
    let tokens = ws.step + layout.tokens as u64;
    let pos3 = state.pos3.addr();
    op!(prof, "store_positions", s, k::store_positions(s, ws.step + layout.pos as u64, pos3, start, t));

    op!(prof, "embed", s, k::embed(s, tokens, t, model.embed, ws.h));
    for (i, layer) in model.layers.iter().enumerate() {
        if let Some(ple) = &layer.ple {
            let e = ws.step + layout.ple as u64;
            op!(prof, "ple.key", s, linear(s, ws, e, &ple.key, ws.kp, t));
            op!(prof, "ple.value", s, linear(s, ws, e, &ple.value, ws.pv, t));
            op!(prof, "ple_gate", s, k::ple_gate(s, ws.kp, ws.h, ws.pv, ple.norm_key, ple.norm_query, ple.norm_conv, ws.gv, ws.xn, t));
            op!(prof, "ple_conv", s, k::ple_conv(s, ws.gv, ws.xn, ple.conv, state.ple_conv.addr(), ws.h, t));
        }

        hyper_connection(s, ws, &layer.attn_hc, t, ws.m, prof.as_deref_mut())?;
        match &layer.block {
            Block::Gdn(g) => {
                let st = state.gdn[i].as_ref().unwrap();
                let (ld, x) = (g.proj.n, ws.proj);
                let col = |j: usize| x + (g.proj.col(j) * 4) as u64;
                op!(prof, "gdn proj", s, project(s, ws, ws.m, &g.proj, x, t));
                if t <= k::gemv_max_m() {
                    op!(prof, "gdn_fused_small", s, k::gdn_fused_small(s, x, ld, t, g.conv, st.conv.addr(), st.recurrent.addr(), g.a_log, g.dt_bias, g.norm, g.proj.col(1), g.proj.col(2), g.proj.col(3), ws.o_bf));
                } else {
                    op!(prof, "gdn_conv", s, k::gdn_conv(s, col(0), ld, ws.conv, st.conv.addr(), g.conv, t));
                    op!(prof, "gdn_prep", s, k::gdn_prep(s, ws.conv, col(2), col(3), ld, g.a_log, g.dt_bias, ws.beta, ws.g, t));
                    op!(prof, "gdn_recurrent", s, k::gdn_recurrent(s, ws.conv, ws.beta, ws.g, st.recurrent.addr(), ws.o, t));
                    op!(prof, "gdn_gated_norm", s, k::gdn_gated_norm(s, ws.o, col(1), ld, g.norm, ws.o_bf, t));
                }
                op!(prof, "gdn out", s, linear(s, ws, ws.o_bf, &g.out, ws.y, t));
            }
            Block::Qsa(q) => {
                let c = state.qsa[i].as_ref().unwrap();
                let (ld, x) = (q.proj.n, ws.proj);
                let col = |j: usize| x + (q.proj.col(j) * 4) as u64;
                op!(prof, "qsa proj", s, project(s, ws, ws.m, &q.proj, x, t));
                op!(prof, "qsa_qkv_prep", s, k::qsa_qkv_prep(s, col(0), col(1), ld, q.q_norm, q.k_norm, ws.q, c.k.addr(), c.v.addr(), pos3, start, t));
                op!(prof, "qsa_index_prep", s, k::qsa_index_prep(s, col(3), ld, q.index_q_norm, ws.iq, c.raw.addr(), pos3, start, t));
                op!(prof, "qsa_pool", s, k::qsa_pool(s, c.raw.addr(), q.index_k_norm, c.pooled.addr(), pos3, start, t));
                // Rows whose query sees at most 512 complete groups attend densely and skip selection.
                if state.capacity / 4 > TOP_GROUPS {
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
        op!(prof, "hc_combine", s, k::hc_combine(s, ws.h, ws.y, ws.inj, t));

        hyper_connection(s, ws, &layer.mlp_hc, t, ws.m, prof.as_deref_mut())?;
        let moe = &layer.moe;
        let (ld, x) = (moe.proj.n, ws.proj);
        let col = |j: usize| x + (moe.proj.col(j) * 4) as u64;
        op!(prof, "moe proj", s, project(s, ws, ws.m, &moe.proj, x, t));
        if t <= k::moe_small_tokens() {
            let max_active = (t * TOP_K).min(EXPERTS);
            let tables = |p: &crate::model::ExpertProj| (p.weights, p.scales, p.scale2);
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
        op!(prof, "shared swiglu", s, k::swiglu(s, col(1), col(2), ld, EXPERT_WIDTH, ws.sact, t));
        op!(prof, "shared down", s, linear(s, ws, ws.sact, &moe.shared_down, ws.sh, t));
        op!(prof, "moe_combine", s, k::moe_combine(s, ws.yd, ws.pair_row, ws.top_w, col(3), ld, ws.sh, ws.y, t));
        op!(prof, "hc_combine", s, k::hc_combine(s, ws.h, ws.y, ws.inj, t));
    }
    hyper_connection(s, ws, &model.final_hc, t, ws.out, prof.as_deref_mut())?;
    Ok(())
}

/// Eager step with per-op GPU timing collected into `prof`.
pub fn forward_profiled(model: &Model, state: &mut SequenceState, ws: &mut Workspace, s: &Stream, new_tokens: &[i32], prof: &mut Profiler) -> Result<()> {
    let bytes = prepare(model, state, ws, new_tokens)?;
    forward_device(model, state, ws, s, new_tokens.len(), bytes, Some(prof))?;
    s.synchronize()?;
    prof.collect()
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
pub struct DecodeGraph {
    graph: squash_cuda::Graph,
    bytes: usize,
}

impl DecodeGraph {
    pub fn capture(model: &Model, state: &SequenceState, ws: &Workspace, hw: &HeadWorkspace, s: &Stream) -> Result<Self> {
        let bytes = StepLayout::new(ws.max_tokens).used(1);
        let graph = squash_cuda::Graph::capture(s, |s| {
            forward_device(model, state, ws, s, 1, bytes, None)?;
            head_argmax(model, hw, s, ws.out)
        })?;
        Ok(Self { graph, bytes })
    }

    /// Decodes `token` (appended to `state`) and returns the greedy next token.
    pub fn step(&self, model: &Model, state: &mut SequenceState, ws: &mut Workspace, hw: &HeadWorkspace, s: &Stream, token: i32) -> Result<i32> {
        self.step_timed(model, state, ws, hw, s, token, &mut [0.0; 2])
    }

    /// As [`step`](Self::step), adding host preparation and graph execution seconds to `times`.
    #[allow(clippy::too_many_arguments)]
    pub fn step_timed(&self, model: &Model, state: &mut SequenceState, ws: &mut Workspace, hw: &HeadWorkspace, s: &Stream, token: i32, times: &mut [f64; 2]) -> Result<i32> {
        let t0 = std::time::Instant::now();
        let bytes = prepare(model, state, ws, &[token])?;
        debug_assert_eq!(bytes, self.bytes);
        let t1 = std::time::Instant::now();
        self.graph.launch(s)?;
        s.synchronize()?;
        times[0] += (t1 - t0).as_secs_f64();
        times[1] += t1.elapsed().as_secs_f64();
        Ok(i32::from_le_bytes(hw.token_host.as_slice()[..4].try_into().unwrap()))
    }
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
