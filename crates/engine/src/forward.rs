//! Eager forward pass of the text model over a chunk of new tokens for one sequence.

use anyhow::{Result, ensure};
use squash_cuda::{DeviceBuffer, Stream, kernels as k};

use crate::model::{Block, D, EOS, EXPERT_WIDTH, EXPERTS, H, HC_RANK, Hc, LAYERS, Model, PleTable, S, TOP_K, VOCAB, is_attention};

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
    pub tokens: u64,
    pub h: u64,
    n: u64,
    low: u64,
    low_bf: u64,
    up: u64,
    inj: u64,
    pub m: u64,
    y: u64,
    // GDN
    qkv: u64,
    conv: u64,
    z: u64,
    ga: u64,
    gb: u64,
    beta: u64,
    g: u64,
    o: u64,
    o_bf: u64,
    // QSA
    qg: u64,
    kv: u64,
    q: u64,
    idx: u64,
    iq: u64,
    scores: u64,
    sel: u64,
    att: u64,
    // MoE
    logits: u64,
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
    sg: u64,
    sgate: u64,
    sup: u64,
    sact: u64,
    sh: u64,
    // PLE
    e: u64,
    kp: u64,
    pv: u64,
    gv: u64,
    xn: u64,
    /// Final mixer output, BF16 [T, D].
    pub out: u64,
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
            tokens: alloc(t * 4)?,
            h: alloc(t * H * 4)?,
            n: alloc(t * H * 2)?,
            low: alloc(t * HC_RANK * 4)?,
            low_bf: alloc(t * HC_RANK * 2)?,
            up: alloc(t * H * 4)?,
            inj: alloc(t * S * 4)?,
            m: alloc(t * D * 2)?,
            y: alloc(t * D * 4)?,
            qkv: alloc(t * GDN_CONV * 4)?,
            conv: alloc(t * GDN_CONV * 4)?,
            z: alloc(t * GDN_Z * 4)?,
            ga: alloc(t * GDN_HEADS * 4)?,
            gb: alloc(t * GDN_HEADS * 4)?,
            beta: alloc(t * GDN_HEADS * 4)?,
            g: alloc(t * GDN_HEADS * 4)?,
            o: alloc(t * GDN_Z * 4)?,
            o_bf: alloc(t * GDN_Z * 2)?,
            qg: alloc(t * QSA_HEADS * 2 * QSA_DIM * 4)?,
            kv: alloc(t * 2 * QSA_KV * QSA_DIM * 4)?,
            q: alloc(t * QSA_HEADS * QSA_DIM * 4)?,
            idx: alloc(t * 5 * INDEX_DIM * 4)?,
            iq: alloc(t * 4 * INDEX_DIM * 4)?,
            scores: alloc(t * score_cols * 4)?,
            sel: alloc(t * TOP_GROUPS * 4)?,
            att: alloc(t * QSA_HEADS * QSA_DIM * 2)?,
            logits: alloc(t * EXPERTS * 4)?,
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
            sg: alloc(t * 4)?,
            sgate: alloc(t * EXPERT_WIDTH * 4)?,
            sup: alloc(t * EXPERT_WIDTH * 4)?,
            sact: alloc(t * EXPERT_WIDTH * 2)?,
            sh: alloc(t * D * 4)?,
            e: alloc(t * D * 2)?,
            kp: alloc(t * H * 4)?,
            pv: alloc(t * D * 4)?,
            gv: alloc(t * H * 4)?,
            xn: alloc(t * H * 4)?,
            out: alloc(t * D * 2)?,
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

fn gemm(s: &Stream, x: u64, w: u64, y: u64, m: usize, n: usize, kk: usize) -> Result<()> {
    k::gemm_f32(s, x, kk, w, kk, y, n, m, n, kk, 1.0, false)
}

/// Mixes the residual through a HyperConnection: writes the BF16 block input to `ws.m` (or `out`)
/// and, when the HC injects, the injection weights to `ws.inj`.
fn hyper_connection(s: &Stream, ws: &Workspace, hc: &Hc, t: usize, out: u64) -> Result<()> {
    k::grouped_rmsnorm(s, ws.h, hc.norm, ws.n, t)?;
    gemm(s, ws.n, hc.down, ws.low, t, HC_RANK, H)?;
    k::hc_low_rank_act(s, ws.low, ws.low_bf, t * HC_RANK)?;
    gemm(s, ws.low_bf, hc.up, ws.up, t, H, HC_RANK)?;
    k::hc_mix(s, ws.up, ws.n, out, t)?;
    if let Some(inject) = hc.inject {
        k::hc_inject(s, ws.n, inject, ws.inj, t)?;
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

fn gather_ple(table: &PleTable, tokens: &[i32], start: usize, t: usize, ple: &crate::model::Ple) -> Vec<u16> {
    let mut e = Vec::with_capacity(t * D);
    for p in start..start + t {
        for r in ple_rows(tokens, p, &ple.multipliers, &ple.sizes, &ple.offsets) {
            table.gather(r, &mut e);
        }
    }
    e
}

/// Runs `new_tokens` (appended at the end of `state`) through the model. Afterwards `ws.out` holds the
/// final mixer output, BF16 [T, D], for the new tokens.
pub fn forward(model: &Model, state: &mut SequenceState, ws: &Workspace, s: &Stream, new_tokens: &[i32]) -> Result<()> {
    let t = new_tokens.len();
    let start = state.len();
    ensure!(t > 0 && t <= ws.max_tokens, "step of {t} tokens exceeds the workspace ({})", ws.max_tokens);
    ensure!(start + t <= state.capacity, "sequence would exceed its capacity of {}", state.capacity);
    state.tokens.extend_from_slice(new_tokens);

    s.upload_to(ws.tokens, new_tokens)?;
    let pos: Vec<i32> = (start..start + t).flat_map(|p| [p as i32; 3]).collect();
    s.upload_to(state.pos3.addr() + (start * 12) as u64, &pos)?;
    let pos3 = state.pos3.addr();

    k::embed(s, ws.tokens, t, model.embed, ws.h)?;
    for (i, layer) in model.layers.iter().enumerate() {
        if let Some(ple) = &layer.ple {
            let e = gather_ple(&model.ple_table, &state.tokens, start, t, ple);
            s.upload_to(ws.e, &e)?;
            gemm(s, ws.e, ple.key, ws.kp, t, H, D)?;
            gemm(s, ws.e, ple.value, ws.pv, t, D, D)?;
            k::ple_gate(s, ws.kp, ws.h, ws.pv, ple.norm_key, ple.norm_query, ple.norm_conv, ws.gv, ws.xn, t)?;
            k::ple_conv(s, ws.gv, ws.xn, ple.conv, state.ple_conv.addr(), ws.h, t)?;
        }

        hyper_connection(s, ws, &layer.attn_hc, t, ws.m)?;
        match &layer.block {
            Block::Gdn(g) => {
                let st = state.gdn[i].as_ref().unwrap();
                gemm(s, ws.m, g.qkv, ws.qkv, t, GDN_CONV, D)?;
                gemm(s, ws.m, g.z, ws.z, t, GDN_Z, D)?;
                gemm(s, ws.m, g.a, ws.ga, t, GDN_HEADS, D)?;
                gemm(s, ws.m, g.b, ws.gb, t, GDN_HEADS, D)?;
                k::gdn_conv(s, ws.qkv, ws.conv, st.conv.addr(), g.conv, t)?;
                k::gdn_prep(s, ws.conv, ws.ga, ws.gb, g.a_log, g.dt_bias, ws.beta, ws.g, t)?;
                k::gdn_recurrent(s, ws.conv, ws.beta, ws.g, st.recurrent.addr(), ws.o, t)?;
                k::gdn_gated_norm(s, ws.o, ws.z, g.norm, ws.o_bf, t)?;
                gemm(s, ws.o_bf, g.out, ws.y, t, D, GDN_Z)?;
            }
            Block::Qsa(q) => {
                let c = state.qsa[i].as_ref().unwrap();
                gemm(s, ws.m, q.q, ws.qg, t, QSA_HEADS * 2 * QSA_DIM, D)?;
                k::gemm_f32(s, ws.m, D, q.k, D, ws.kv, 2 * QSA_KV * QSA_DIM, t, QSA_KV * QSA_DIM, D, 1.0, false)?;
                k::gemm_f32(s, ws.m, D, q.v, D, ws.kv + (QSA_KV * QSA_DIM * 4) as u64, 2 * QSA_KV * QSA_DIM, t, QSA_KV * QSA_DIM, D, 1.0, false)?;
                k::qsa_qkv_prep(s, ws.qg, ws.kv, q.q_norm, q.k_norm, ws.q, c.k.addr(), c.v.addr(), pos3, start, t)?;
                gemm(s, ws.m, q.index_qk, ws.idx, t, 5 * INDEX_DIM, D)?;
                k::qsa_index_prep(s, ws.idx, q.index_q_norm, ws.iq, c.raw.addr(), pos3, start, t)?;
                k::qsa_pool(s, c.raw.addr(), q.index_k_norm, c.pooled.addr(), pos3, start, t)?;
                // Group selection is needed only once some query has more than 512 complete groups.
                let max_groups = (start + t) / 4;
                if max_groups > TOP_GROUPS {
                    ensure!(max_groups <= ws.score_cols, "context exceeds the workspace's score columns");
                    k::qsa_index_scores(s, ws.iq, c.pooled.addr(), ws.scores, max_groups, start, t)?;
                    k::topk(s, ws.scores, t, max_groups, max_groups, TOP_GROUPS, true, ws.sel, 0)?;
                }
                k::qsa_attention(s, ws.q, ws.qg, c.k.addr(), c.v.addr(), ws.sel, start, t, ws.att)?;
                gemm(s, ws.att, q.o, ws.y, t, D, QSA_HEADS * QSA_DIM)?;
            }
        }
        k::hc_combine(s, ws.h, ws.y, ws.inj, t)?;

        hyper_connection(s, ws, &layer.mlp_hc, t, ws.m)?;
        let moe = &layer.moe;
        gemm(s, ws.m, moe.router, ws.logits, t, EXPERTS, D)?;
        k::moe_route(s, ws.logits, t, ws.top_i, ws.top_w)?;
        k::moe_plan(s, ws.top_i, t, ws.counts, ws.cursor, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.pair_row)?;
        k::moe_gather(s, ws.m, ws.pair_row, t, ws.xs)?;
        let tiles = ws.max_tiles().min((t * TOP_K).div_ceil(k::moe_tile_rows()) + EXPERTS);
        let grouped = |proj: &crate::model::ExpertProj, x: u64, y: u64, n: usize, kk: usize| {
            k::moe_nvfp4_gemm(s, x, proj.weights, proj.scales, proj.scale2, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, tiles, y, n, kk)
        };
        grouped(&moe.gate, ws.xs, ws.eg, EXPERT_WIDTH, D)?;
        grouped(&moe.up, ws.xs, ws.eu, EXPERT_WIDTH, D)?;
        k::swiglu(s, ws.eg, ws.eu, ws.act, t * TOP_K * EXPERT_WIDTH)?;
        grouped(&moe.down, ws.act, ws.yd, D, EXPERT_WIDTH)?;
        gemm(s, ws.m, moe.shared_gate, ws.sgate, t, EXPERT_WIDTH, D)?;
        gemm(s, ws.m, moe.shared_up, ws.sup, t, EXPERT_WIDTH, D)?;
        k::swiglu(s, ws.sgate, ws.sup, ws.sact, t * EXPERT_WIDTH)?;
        gemm(s, ws.sact, moe.shared_down, ws.sh, t, D, EXPERT_WIDTH)?;
        gemm(s, ws.m, moe.shared_expert_gate, ws.sg, t, 1, D)?;
        k::moe_combine(s, ws.yd, ws.pair_row, ws.top_w, ws.sg, ws.sh, ws.y, t)?;
        k::hc_combine(s, ws.h, ws.y, ws.inj, t)?;
    }
    hyper_connection(s, ws, &model.final_hc, t, ws.out)?;
    Ok(())
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
        gemm(s, hidden + (r0 * D * 2) as u64, model.lm_head, hw.logits.addr(), n, VOCAB, D)?;
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
