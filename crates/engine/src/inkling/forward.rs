//! Inkling 2 forward pass over a chunk of new tokens of one sequence.
//!
//! Per layer visit (48: the looped layers once per pass) a sequence keeps the keys and values (every
//! position for full-attention layers, a ring of the last `RING` for sliding ones) and rings of the inputs
//! of its four causal convolutions (key and value projections, attention and MoE outputs). Chunks hold at
//! most `max_t` tokens (`max_t + WINDOW <= RING`).

use anyhow::{Context, Result, ensure};
use quench6k_cuda::inkling::{self as ik, ConvSrc, KvSide, RmsSet};
use quench6k_cuda::kernels as k;
use quench6k_cuda::{DeviceBuffer, PinnedBuffer, Stream};

use super::*;

pub const RING: usize = 4096;
/// Block-sum slots: embedding, blocks of the stream, and the previous loop pass's last block and partial.
pub const SLOTS: usize = 8;
const PREV_BLOCK: usize = SLOTS - 2;
const PREV_PARTIAL: usize = SLOTS - 1;
const SLIDING_SPLITS: usize = 16;
const FULL_SPLITS: usize = 32;
const MIN_KEYS: usize = 32;
/// Steps of more tokens run NVFP4 experts as grouped tensor-core GEMMs (tiles of MOE_BM rows).
const SMALL_T: usize = 16;
const MOE_BM: usize = 64;
pub const TOP_K: usize = 32;

struct Alloc {
    buffers: Vec<DeviceBuffer>,
    bytes: usize,
}

impl Alloc {
    fn new() -> Self {
        Self {
            buffers: Vec::new(),
            bytes: 0,
        }
    }
    fn get(&mut self, bytes: usize) -> Result<u64> {
        let b = DeviceBuffer::new(bytes.max(256))?;
        let a = b.addr();
        self.buffers.push(b);
        self.bytes += bytes;
        Ok(a)
    }
}

/// One layer visit's caches.
struct VisitCache {
    k: u64,
    v: u64,
    rows: usize,
    k_ring: u64,
    v_ring: u64,
    attn_ring: u64,
    mlp_ring: u64,
}

/// One sequence's caches and token history.
pub struct State {
    pub pos: usize,
    pub ctx: usize,
    caches: Vec<VisitCache>,
    pub history: Vec<i32>,
    pub device_bytes: usize,
    /// CUDA graphs of small steps by token count and logits kind; they capture this state's caches.
    graphs: Vec<Option<quench6k_cuda::Graph>>,
    /// The drafter's pass as a CUDA graph (it captures this state's caches).
    dgraph: Option<quench6k_cuda::Graph>,
    /// The drafter's keys and values per layer ([ctx, 1024] BF16 each): context rows from the target's
    /// features, block rows from its own passes.
    draft_kv: Vec<(u64, u64)>,
    pub last_t: usize,
    /// Every ring (sliding keys and values, convolution inputs) at snap_pos, the end of the last prompt.
    snap: Option<DeviceBuffer>,
    snap_pos: Option<usize>,
    _alloc: Alloc,
}

impl State {
    pub fn new(model: &InklingModel, ctx: usize, s: &Stream) -> Result<Self> {
        let mut a = Alloc::new();
        let mut caches = Vec::new();
        for (layer, _) in model.cfg.visits() {
            let rows = if model.cfg.sliding[layer] { RING } else { ctx };
            caches.push(VisitCache {
                k: a.get(rows * KVD * 2)?,
                v: a.get(rows * KVD * 2)?,
                rows,
                k_ring: a.get(RING * KVD * 2)?,
                v_ring: a.get(RING * KVD * 2)?,
                attn_ring: a.get(RING * D * 2)?,
                mlp_ring: a.get(RING * D * 2)?,
            });
        }
        let mut draft_kv = Vec::new();
        if let Some(d) = &model.drafter {
            for _ in &d.layers {
                draft_kv.push((a.get(ctx * KVD * 2)?, a.get(ctx * KVD * 2)?));
            }
        }
        for b in &a.buffers {
            s.memset(b.addr(), 0, b.bytes())?;
        }
        Ok(Self {
            pos: 0,
            ctx,
            caches,
            history: Vec::new(),
            device_bytes: a.bytes,
            graphs: Vec::new(),
            dgraph: None,
            draft_kv,
            last_t: 0,
            snap: None,
            snap_pos: None,
            _alloc: a,
        })
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        self.history.clear();
        self.snap_pos = None;
    }

    /// Drops positions from `pos` on. The rings keep the last RING positions, so `pos` must be within
    /// RING - WINDOW positions of the current one (see [`State::rewind`]).
    pub fn truncate(&mut self, pos: usize) {
        self.pos = self.pos.min(pos);
        self.history.truncate(self.pos);
    }

    /// Keeps the longest cached prefix of `prompt` that can be reused (at most prompt.len() - 1 tokens, so
    /// the last prompt token runs again for its logits) and returns its length: the common prefix with the
    /// history if the rings still hold its last WINDOW positions, else the snapshot's position (after
    /// restoring the rings) if it is within reach, else 0.
    pub fn rewind(&mut self, prompt: &[i32], s: &Stream) -> Result<usize> {
        const REACH: usize = RING - WINDOW;
        let common = self.history.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let p = common.min(prompt.len().saturating_sub(1));
        let live = if self.pos <= p + REACH { p } else { 0 };
        let snap = match self.snap_pos {
            Some(e) if e <= self.pos && e <= common + REACH => p.min(e),
            _ => 0,
        };
        let keep = if snap > live {
            self.restore(s)?;
            snap
        } else {
            live
        };
        if self.snap_pos.is_some_and(|e| keep < e) {
            self.snap_pos = None;
        }
        self.truncate(keep);
        Ok(keep)
    }

    /// The (address, bytes) of every ring: per visit the convolution inputs, and the sliding layers' keys
    /// and values.
    fn rings(&self) -> Vec<(u64, usize)> {
        let mut v = Vec::new();
        for c in &self.caches {
            v.push((c.k_ring, RING * KVD * 2));
            v.push((c.v_ring, RING * KVD * 2));
            v.push((c.attn_ring, RING * D * 2));
            v.push((c.mlp_ring, RING * D * 2));
            if c.rows == RING {
                v.push((c.k, RING * KVD * 2));
                v.push((c.v, RING * KVD * 2));
            }
        }
        v
    }

    /// Copies the rings at the current position (the end of a prompt), for [`State::rewind`].
    pub fn snapshot(&mut self, s: &Stream) -> Result<()> {
        let rings = self.rings();
        let total: usize = rings.iter().map(|r| r.1).sum();
        if self.snap.is_none() {
            self.snap = Some(DeviceBuffer::new(total)?);
        }
        let mut at = self.snap.as_ref().unwrap().addr();
        for (addr, bytes) in rings {
            s.copy_d2d(at, addr, bytes)?;
            at += bytes as u64;
        }
        self.snap_pos = Some(self.pos);
        Ok(())
    }

    fn restore(&mut self, s: &Stream) -> Result<()> {
        let mut at = self.snap.as_ref().context("no snapshot")?.addr();
        for (addr, bytes) in self.rings() {
            s.copy_d2d(addr, at, bytes)?;
            at += bytes as u64;
        }
        let e = self.snap_pos.context("no snapshot")?;
        self.pos = e;
        self.history.truncate(e);
        Ok(())
    }
}

/// Scratch buffers for chunks of up to `max_t` tokens.
pub struct Workspace {
    pub max_t: usize,
    pub logit_rows: usize,
    start: u64,
    tokens: u64,
    slots: [u64; SLOTS],
    h: [u64; 2],
    r: u64,
    res: [u64; 3],
    msgs: u64,
    cat: u64,
    z: u64,
    zf: u64,
    fg: u64,
    gated: [u64; 3],
    x: [u64; 3],
    xm: u64,
    lin_q: u64,
    lin_k: u64,
    lin_v: u64,
    q: u64,
    rel: u64,
    sink: u64,
    part: u64,
    attn: u64,
    o: u64,
    rlogits: u64,
    sel_e: u64,
    sel_w: u64,
    order: u64,
    act: u64,
    y: u64,
    moe: u64,
    // grouped experts
    offsets: u64,
    tile_expert: u64,
    tile_row: u64,
    num_tiles: u64,
    pair_row: u64,
    max_tiles: usize,
    xs: u64,
    /// Target features per token BF16 [max_t, 4, 2304]: the inputs of the drafter's visits and the final
    /// norm's output (tap 3, also the LM head's input).
    pub feats: u64,
    /// The drafter's buffers (when the model has one).
    draft: Option<DraftWs>,
    pub out_logits: u64,
    pub lse: u64,
    pub target: u64,
    pub targets: u64,
    pub top_idx: u64,
    pub top_val: u64,
    pub argmax: u64,
    pub argmax_scratch: u64,
    pinned: PinnedBuffer,
    /// The side stream of the L2 prefetches (forked from and joined into the step's stream).
    side: Stream,
    fork: quench6k_cuda::Event,
    join: quench6k_cuda::Event,
    pub device_bytes: usize,
    _alloc: Alloc,
}

impl Workspace {
    /// Chunks of up to `max_t` tokens; `logit_rows`: most rows of logits a step produces (1 for
    /// generation; max_t for scoring every position).
    pub fn new(model: &InklingModel, max_t: usize, logit_rows: usize) -> Result<Self> {
        ensure!(max_t + WINDOW <= RING, "chunks of {max_t} tokens exceed the rings");
        let mut a = Alloc::new();
        let t = max_t;
        let vocab = model.cfg.vocab;
        let pairs = t * TOPK;
        let td = t * D * 4;
        let mut slots = [0; SLOTS];
        for s in &mut slots {
            *s = a.get(td)?;
        }
        let logit_rows = logit_rows.max(16).min(t.max(16));
        // Residual-mixture outputs: up to 3 (full layers' query, key and value inputs) of t rows each.
        let res = a.get(3 * td)?;
        let ws = Self {
            max_t,
            logit_rows,
            start: a.get(16)?,
            tokens: a.get(t * 4)?,
            slots,
            h: [a.get(td)?, a.get(td)?],
            r: a.get(td)?,
            res: [res, res + td as u64, res + 2 * td as u64],
            msgs: a.get(4 * td)?,
            cat: a.get(3 * t * 2 * D * 2)?,
            z: a.get(3 * t * RANK * 2)?,
            zf: a.get(t * RANK * 4)?,
            fg: a.get(t * 2 * D * 4)?,
            gated: [a.get(td)?, a.get(td)?, a.get(td)?],
            x: [a.get(t * D * 2)?, a.get(t * D * 2)?, a.get(t * D * 2)?],
            xm: a.get(t * D * 2)?,
            lin_q: a.get(t * QG * 4)?,
            lin_k: a.get(t * (KVD + 32) * 4)?,
            lin_v: a.get(t * (KVD + 32) * 4)?,
            q: a.get(t * QD * 2)?,
            rel: a.get(t * H * DREL * 4)?,
            sink: a.get(t * H * 4)?,
            // Split-key partials of small steps (larger ones run the tensor-core kernel).
            part: a.get(ik::attn_part_floats(t.min(16), FULL_SPLITS) * 4)?,
            attn: a.get(t * QD * 2)?,
            o: a.get(td)?,
            rlogits: a.get(t * EXPERTS * 4)?,
            sel_e: a.get(pairs * 4)?,
            sel_w: a.get(pairs * 4)?,
            order: a.get(pairs * 4)?,
            act: a.get(pairs * FF * 2)?,
            y: a.get(pairs * D * 4)?,
            moe: a.get(td)?,
            offsets: a.get((EXPERTS + 1) * 4)?,
            tile_expert: a.get((pairs / MOE_BM + EXPERTS) * 4)?,
            tile_row: a.get((pairs / MOE_BM + EXPERTS) * 4)?,
            num_tiles: a.get(4)?,
            pair_row: a.get(pairs * 4)?,
            max_tiles: pairs / MOE_BM + EXPERTS,
            xs: a.get(pairs * D * 2)?,
            feats: a.get(t * 4 * D * 2)?,
            draft: None,
            out_logits: a.get(logit_rows * vocab * 4)?,
            lse: a.get(t * 4)?,
            target: a.get(t * 4)?,
            targets: a.get(t * 4)?,
            top_idx: a.get(t * TOP_K * 4)?,
            top_val: a.get(t * TOP_K * 4)?,
            argmax: a.get(t * 4)?,
            argmax_scratch: a.get(k::argmax_scratch(t, vocab))?,
            pinned: PinnedBuffer::new(t * 4 + 64)?,
            side: Stream::new()?,
            fork: quench6k_cuda::Event::new()?,
            join: quench6k_cuda::Event::new()?,
            device_bytes: 0,
            _alloc: Alloc::new(),
        };
        let draft = match &model.drafter {
            Some(d) => {
                let b = d.block;
                let ff2 = d.layers[0].gate_up.n;
                Some(DraftWs {
                    stage: a.get(16)?,
                    drafts: a.get(b * 4)?,
                    prev: a.get(16)?,
                    x: a.get(b * D * 4)?,
                    xn: a.get(b * D * 2)?,
                    qkv: a.get(b * d.layers[0].qkv.n * 4)?,
                    q: a.get(b * QD * 2)?,
                    attn: a.get(b * QD * 2)?,
                    gu: a.get(b * ff2 * 4)?,
                    act: a.get(b * ff2 / 2 * 2)?,
                    hf: a.get(b * 2 * D * 2)?,
                    base: a.get(b * vocab * 4)?,
                    ctx: a.get(b * d.markov_hf.n * 4)?,
                    state: a.get(2 * 256 * 2)?,
                    latent: a.get(256 * 2)?,
                    best_v: a.get(ik::markov_blocks(vocab) * 4)?,
                    best_i: a.get(ik::markov_blocks(vocab) * 4)?,
                    mixed: a.get(t * D * 2)?,
                    kv: a.get(t * 2 * KVD * 4)?,
                    pinned: PinnedBuffer::new(64 + b * 4)?,
                })
            }
            None => None,
        };
        let bytes = a.bytes;
        Ok(Self {
            draft,
            device_bytes: bytes,
            _alloc: a,
            ..ws
        })
    }
}

/// Buffers of the drafter's pass over its block (`block` rows) and of its context keys (max_t rows).
struct DraftWs {
    /// I32 device: the accepted token, its position, the feature row of the token before it.
    stage: u64,
    drafts: u64,
    prev: u64,
    x: u64,
    xn: u64,
    qkv: u64,
    q: u64,
    attn: u64,
    gu: u64,
    act: u64,
    /// BF16 [block, 4608]: final hidden state | f_rnn, the inputs of the Markov head's context.
    hf: u64,
    base: u64,
    ctx: u64,
    state: u64,
    latent: u64,
    best_v: u64,
    best_i: u64,
    mixed: u64,
    kv: u64,
    pinned: PinnedBuffer,
}

/// Y [m, l.n] F32 = X [m, l.k] BF16 . W^T.
fn linear(s: &Stream, x: u64, ldx: usize, m: usize, l: &Linear, y: u64, ldy: usize) -> Result<()> {
    if m <= k::gemv_max_m() {
        k::gemv(s, m, x, ldx, l.fmt, l.w, l.scale, y, ldy, 0, l.n, l.k, 1, 1.0, false)
    } else {
        k::gemm_f32(s, x, ldx, l.fmt, l.w, l.k, l.scale, y, ldy, m, l.n, l.k, 1.0, false)
    }
}

fn prefetch_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QUENCH6K_INK_PREFETCH").map_or(true, |v| v != "0"))
}

/// Which positions' logits a forward pass produces.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Logits {
    None,
    Last,
    All,
}

impl InklingModel {
    /// Runs `tokens` (at positions state.pos ..) through the model, appending them to the caches.
    /// Softcapped logits (F32 [rows, vocab]) land in ws.out_logits: one row for `Last`, one per token for
    /// `All`. Steps of at most 16 tokens run as a CUDA graph captured once per token count and logits kind
    /// (QUENCH6K_INK_GRAPHS=0: eager).
    pub fn forward(&self, state: &mut State, ws: &mut Workspace, s: &Stream, tokens: &[i32], want: Logits) -> Result<()> {
        let t = tokens.len();
        ensure!(t > 0 && t <= ws.max_t, "chunk of {t} tokens (max {})", ws.max_t);
        ensure!(state.pos + t <= state.ctx, "context full ({} + {t} > {})", state.pos, state.ctx);
        ensure!(want != Logits::All || t <= ws.logit_rows, "{t} rows of logits (workspace holds {})", ws.logit_rows);
        for &tok in tokens {
            ensure!(tok >= 0 && (tok as usize) < self.cfg.embed_rows, "token {tok} out of range");
        }
        // The pinned buffer may still feed the previous step's upload.
        s.synchronize()?;
        {
            let buf = ws.pinned.as_mut_slice();
            for (j, &tok) in tokens.iter().enumerate() {
                buf[j * 4..j * 4 + 4].copy_from_slice(&tok.to_le_bytes());
            }
            let at = t.next_multiple_of(4) * 4;
            buf[at..at + 4].copy_from_slice(&(state.pos as i32).to_le_bytes());
        }
        let dump = std::env::var_os("QUENCH6K_INK_DUMP").is_some();
        let graphs = std::env::var("QUENCH6K_INK_GRAPHS").map_or(true, |v| v != "0") && !dump;
        if graphs && t <= 16 {
            let slot = t + 17 * want as usize;
            if state.graphs.len() <= slot {
                state.graphs.resize_with(slot + 1, || None);
            }
            if state.graphs[slot].is_none() {
                let g = quench6k_cuda::Graph::capture(s, |s| self.enqueue(state, ws, s, t, want))?;
                state.graphs[slot] = Some(g);
            }
            state.graphs[slot].as_ref().unwrap().launch(s)?;
        } else {
            self.enqueue(state, ws, s, t, want)?;
        }
        state.pos += t;
        state.last_t = t;
        state.history.extend_from_slice(tokens);
        Ok(())
    }

    /// Drafts the tokens after `token`, the one the last step chose after its row `row` (the last accepted
    /// one; `token` is not in the caches yet and goes at position state.pos). Writes the block's drafter
    /// keys past state.pos (the next step overwrites them). Empty without a drafter or room in the context.
    pub fn draft(&self, state: &mut State, ws: &mut Workspace, s: &Stream, token: i32, row: usize) -> Result<Vec<i32>> {
        let (Some(d), Some(_)) = (&self.drafter, &ws.draft) else {
            return Ok(Vec::new());
        };
        let b = d.block;
        if state.pos + b > state.ctx || state.last_t == 0 {
            return Ok(Vec::new());
        }
        ensure!(row < state.last_t, "draft source row {row} of a {}-token step", state.last_t);
        let dw = ws.draft.as_mut().unwrap();
        {
            let buf = dw.pinned.as_mut_slice();
            for (i, v) in [token, state.pos as i32, row as i32].into_iter().enumerate() {
                buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let graphs = std::env::var("QUENCH6K_INK_GRAPHS").map_or(true, |v| v != "0");
        if graphs {
            if state.dgraph.is_none() {
                let g = quench6k_cuda::Graph::capture(s, |s| self.draft_enqueue(state, ws, s))?;
                state.dgraph = Some(g);
            }
            state.dgraph.as_ref().unwrap().launch(s)?;
        } else {
            self.draft_enqueue(state, ws, s)?;
        }
        s.synchronize()?;
        let dw = ws.draft.as_ref().unwrap();
        if let Some(dir) = std::env::var_os("QUENCH6K_INK_DUMP_DRAFT") {
            // Debugging: the step's features, the block's [hidden | f_rnn] rows and the base logits.
            let dir = std::path::Path::new(&dir);
            let bytes = |v: Vec<u16>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            std::fs::write(dir.join("feats.bin"), bytes(s.download::<u16>(ws.feats, state.last_t * 4 * D)?))?;
            std::fs::write(dir.join("hf.bin"), bytes(s.download::<u16>(dw.hf, (b - 1) * 2 * D)?))?;
            let base: Vec<f32> = s.download(dw.base, (b - 1) * self.cfg.vocab)?;
            std::fs::write(dir.join("base.bin"), base.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            let mut meta = dw.pinned.as_slice()[..12].to_vec();
            meta.extend_from_slice(&dw.pinned.as_slice()[64..64 + (b - 1) * 4]);
            std::fs::write(dir.join("stage.bin"), meta)?;
        }
        Ok(dw.pinned.as_slice()[64..64 + (b - 1) * 4]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    fn draft_enqueue(&self, state: &State, ws: &Workspace, s: &Stream) -> Result<()> {
        let cfg = &self.cfg;
        let d = self.drafter.as_ref().unwrap();
        let dw = ws.draft.as_ref().unwrap();
        let b = d.block;
        let (anchor, pos, row) = (dw.stage, dw.stage + 4, dw.stage + 8);
        unsafe { s.copy_h2d(dw.stage, dw.pinned.as_ptr(), 12)? };
        let acc = |x: u64, ldx: usize, l: &Linear, y: u64| k::gemv(s, b, x, ldx, l.fmt, l.w, l.scale, y, D, 0, l.n, l.k, 1, 1.0, true);
        ik::block_embed(s, anchor, self.embed, self.embed_norm, d.mask, b, dw.x)?;
        for (i, l) in d.layers.iter().enumerate() {
            let (kc, vc) = state.draft_kv[i];
            ik::rms(s, &[RmsSet { x: dw.x, w: l.in_norm, out: dw.xn, ldo: D }], b)?;
            linear(s, dw.xn, D, b, &l.qkv, dw.qkv, l.qkv.n)?;
            ik::draft_qkv(s, dw.qkv, l.q_norm, l.k_norm, d.theta, pos, b, dw.q, kc, vc)?;
            let (window, splits) = if l.local { (d.window + b, SLIDING_SPLITS) } else { (0, FULL_SPLITS) };
            ik::attention(s, dw.q, 0, 0, kc, vc, state.ctx, 0, window, b, pos, b, splits, MIN_KEYS, 0, 0, 0, ws.part, dw.attn)?;
            acc(dw.attn, QD, &l.o, dw.x)?;
            ik::rms(s, &[RmsSet { x: dw.x, w: l.post_norm, out: dw.xn, ldo: D }], b)?;
            linear(s, dw.xn, D, b, &l.gate_up, dw.gu, l.gate_up.n)?;
            ik::silu_mul(s, dw.gu, l.gate_up.n / 2, b, dw.act)?;
            acc(dw.act, l.down.k, &l.down, dw.x)?;
        }
        // Rows 1.. predict the drafts: their final hidden states, and the Markov head's context input
        // [hidden | f_rnn] (f_rnn: the features of the token before the accepted one).
        let n = b - 1;
        ik::rms(s, &[RmsSet { x: dw.x + (D * 4) as u64, w: d.norm, out: dw.hf, ldo: 2 * D }], n)?;
        ik::mix(s, ws.feats, row, d.alpha + (d.layers.len() * 16) as u64, d.markov_norm, dw.hf + (D * 2) as u64, 2 * D, n)?;
        linear(s, dw.hf, 2 * D, n, &self.lm_head, dw.base, cfg.vocab)?;
        linear(s, dw.hf, 2 * D, n, &d.markov_hf, dw.ctx, d.markov_hf.n)?;
        s.memset(dw.state, 0, 256 * 2)?;
        s.copy_d2d(dw.prev, anchor, 4)?;
        for i in 0..n {
            let (sin, sout) = (dw.state + ((i % 2) * 512) as u64, dw.state + (((i + 1) % 2) * 512) as u64);
            ik::markov(
                s,
                d.markov,
                dw.ctx + (i * d.markov_hf.n * 4) as u64,
                sin,
                dw.prev,
                sout,
                dw.latent,
                dw.base + (i * cfg.vocab * 4) as u64,
                cfg.vocab,
                cfg.logit_cap,
                dw.best_v,
                dw.best_i,
                dw.drafts + (i * 4) as u64,
            )?;
        }
        unsafe { s.copy_d2h(dw.pinned.as_ptr().add(64), dw.drafts, n * 4)? };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// Message gates `which` (0 query, 1 key, 2 value, 3 MLP) of `layer` on inputs `hs`, messages `m`, into
    /// ws.gated.
    fn gates(&self, ws: &Workspace, s: &Stream, layer: &Layer, which: &[usize], hs: &[u64], m: &[u64], t: usize) -> Result<()> {
        let g = layer.gates.as_ref().unwrap();
        if t > SMALL_T {
            // Prefill: the projections as GEMMs (the fused kernels re-read the weights per token).
            for (j, &i) in which.iter().enumerate() {
                let cat = ws.cat;
                ik::rms(s, &[RmsSet { x: hs[j], w: 0, out: cat, ldo: 2 * D }, RmsSet { x: m[j], w: 0, out: cat + (D * 2) as u64, ldo: 2 * D }], t)?;
                linear(s, cat, 2 * D, t, &g[i].down, ws.zf, RANK)?;
                k::f32_to_bf16(s, ws.zf, ws.z, t * RANK)?;
                linear(s, ws.z, RANK, t, &g[i].fg, ws.fg, 2 * D)?;
                ik::msg_apply_rows(s, hs[j], m[j], ws.fg, t, ws.gated[j])?;
            }
            return Ok(());
        }
        let v: Vec<ik::MsgGate> = which
            .iter()
            .enumerate()
            .map(|(j, &i)| ik::MsgGate {
                hs: hs[j],
                m: m[j],
                down: g[i].down.w,
                fg: g[i].fg.w,
                out: ws.gated[j],
            })
            .collect();
        ik::msg_gates(s, &v, t, ws.cat, ws.z)
    }

    fn enqueue(&self, state: &State, ws: &Workspace, s: &Stream, t: usize, want: Logits) -> Result<()> {
        let cfg = &self.cfg;
        let pin = ws.pinned.as_ptr();
        unsafe {
            s.copy_h2d(ws.tokens, pin, t * 4)?;
            s.copy_h2d(ws.start, pin.add(t.next_multiple_of(4) * 4), 4)?;
        }
        let td = (t * D * 4) as u64;
        let start = ws.start;
        let mut hi = 0;
        ik::embed(s, ws.tokens, t, self.embed, self.embed_norm, ws.h[0])?;
        let mut nb = 0usize;
        // The previous visit's MoE convolution (output ws.h[hi]), computed by the next consumer of the
        // stream (its input mixture) or on its own (`flush`).
        let mut pending: Option<ConvSrc> = None;
        let flush = |pending: &mut Option<ConvSrc>, out: u64| -> Result<()> {
            if let Some(c) = pending.take() {
                ik::hidden_conv(s, c.src, c.cy, c.cw, c.ring, c.ring_rows, c.conv, c.base, c.start, t, out)?;
            }
            Ok(())
        };
        let dump = std::env::var_os("QUENCH6K_INK_DUMP");
        let mut prefetched = false;
        for (v, &(li, pass)) in cfg.visits().iter().enumerate() {
            let layer = &self.layers[li];
            let c = &state.caches[v];
            let h = ws.h[hi];
            if li == cfg.loop_start && pass > 0 {
                flush(&mut pending, h)?;
                // Messages of this pass come from the previous one's last block and partial sum; the pass
                // restarts from the stream before the loop (the embedding and the first block's sum).
                s.copy_d2d(ws.slots[PREV_BLOCK], ws.slots[nb - 1], td as usize)?;
                s.copy_d2d(ws.slots[PREV_PARTIAL], h, td as usize)?;
                nb = cfg.loop_start / BLOCK;
                s.copy_d2d(h, ws.slots[nb], td as usize)?;
            }
            let msg = layer.looped && pass > 0;
            // Small steps: the next visit's dense weights into L2 while this visit runs (QUENCH6K_INK_PREFETCH=0: off).
            if t <= SMALL_T && prefetch_on() && let Some(&(nl, np)) = cfg.visits().get(v + 1) {
                // On a side branch: the prefetch kernel stays until its data arrives.
                let r = self.layer_ranges(nl, self.layers[nl].looped && np > 0);
                ws.fork.record(s)?;
                ws.side.wait(&ws.fork)?;
                for chunk in r.chunks(16) {
                    ik::prefetch(&ws.side, chunk)?;
                }
                prefetched = true;
            }
            if msg {
                let q = self.msg_queries + ((li - cfg.loop_start) * 4 * D * 4) as u64;
                ik::attn_res(s, &[ws.slots[PREV_BLOCK], ws.slots[PREV_PARTIAL]], q, 4, t, ws.msgs, &[], 0, None)?;
            }
            let msgs = |i: usize| ws.msgs + i as u64 * td;
            // Attention inputs.
            let mut src: Vec<u64> = ws.slots[..nb].to_vec();
            src.push(h);
            let set = |x: u64, w: u64, out: u64| RmsSet { x, w, out, ldo: D };
            let (xq, xk, xv) = if li == 0 {
                ik::rms_bf16(s, h, layer.in_norm[0], ws.x[0], t)?;
                (ws.x[0], ws.x[0], ws.x[0])
            } else {
                let full = !layer.sliding;
                let nq = if full { 3 } else { 1 };
                let r = |i: usize| if full { ws.res[0] + i as u64 * td } else { ws.res[0] };
                if msg {
                    ik::attn_res(s, &src, layer.attn_query, nq, t, ws.res[0], &[], 0, pending.take())?;
                    self.gates(ws, s, layer, &[0, 1, 2], &[r(0), r(1), r(2)], &[msgs(0), msgs(1), msgs(2)], t)?;
                    let sets: Vec<RmsSet> = (0..3).map(|i| set(ws.gated[i], layer.in_norm[i], ws.x[i])).collect();
                    ik::rms(s, &sets, t)?;
                    (ws.x[0], ws.x[1], ws.x[2])
                } else {
                    // The mixtures normalized straight into the projections' inputs.
                    let norms: Vec<(u64, u64)> = (0..nq).map(|i| (layer.in_norm[i], ws.x[i])).collect();
                    ik::attn_res(s, &src, layer.attn_query, nq, t, 0, &norms, D, pending.take())?;
                    if full { (ws.x[0], ws.x[1], ws.x[2]) } else { (ws.x[0], ws.x[0], ws.x[0]) }
                }
            };
            debug_assert!(pending.is_none());
            if let Some(d) = &self.drafter
                && let Some(tap) = d.visits.iter().position(|&x| x == v)
            {
                ik::to_bf16_rows(s, h, t, ws.feats + (tap * D * 2) as u64, 4 * D)?;
            }
            // Attention.
            let a = &layer.attn;
            let ldk = a.kg.n;
            linear(s, xq, D, t, &a.qg, ws.lin_q, QG)?;
            linear(s, xk, D, t, &a.kg, ws.lin_k, ldk)?;
            linear(s, xv, D, t, &a.vg, ws.lin_v, ldk)?;
            let side = |ring: u64, conv: u64, cache: u64| {
                if layer.sliding {
                    KvSide { ring, conv, static_w: 0, cache }
                } else {
                    KvSide { ring, conv: 0, static_w: conv, cache }
                }
            };
            ik::qkv_prep(
                s,
                ws.lin_q,
                QG,
                a.qscale,
                a.rscale,
                cfg.qk_clip,
                cfg.rel_clip,
                !layer.sliding,
                ws.q,
                ws.rel,
                ws.sink,
                ws.lin_k,
                ws.lin_v,
                ldk,
                side(c.k_ring, a.kconv, c.k),
                side(c.v_ring, a.vconv, c.v),
                RING,
                c.rows,
                start,
                t,
            )?;
            let splits = if layer.sliding { SLIDING_SPLITS } else { FULL_SPLITS };
            ik::attention(
                s,
                ws.q,
                ws.rel,
                ws.sink,
                c.k,
                c.v,
                c.rows,
                a.proj,
                if layer.sliding { WINDOW } else { 0 },
                0,
                start,
                t,
                splits,
                MIN_KEYS,
                ws.lin_q,
                QG,
                GATE_OFF,
                ws.part,
                ws.attn,
            )?;
            linear(s, ws.attn, QD, t, &a.o, ws.o, D)?;
            let boundary = li.is_multiple_of(BLOCK);
            if boundary {
                // The block ends: its sum joins the sources; the new block starts from the attention output.
                s.copy_d2d(ws.slots[nb], h, td as usize)?;
                nb += 1;
            }
            let attn_conv = ConvSrc {
                src: ws.o,
                ring: c.attn_ring,
                ring_rows: RING,
                conv: layer.attn_conv,
                base: if boundary { 0 } else { h },
                start,
                ..Default::default()
            };
            // MoE.
            let mut src: Vec<u64> = ws.slots[..nb].to_vec();
            src.push(ws.r);
            if msg {
                ik::attn_res(s, &src, layer.mlp_query, 1, t, ws.res[0], &[], 0, Some(attn_conv))?;
                self.gates(ws, s, layer, &[3], &[ws.res[0]], &[msgs(3)], t)?;
                ik::rms_bf16(s, ws.gated[0], layer.post_norm, ws.xm, t)?;
            } else {
                ik::attn_res(s, &src, layer.mlp_query, 1, t, 0, &[(layer.post_norm, ws.xm)], D, Some(attn_conv))?;
            }
            linear(s, ws.xm, D, t, &layer.router, ws.rlogits, EXPERTS)?;
            // Small steps route and order the pairs in one launch and fold the combine into the
            // convolution; larger ones group the pairs for the tensor-core GEMMs (NVFP4 layers).
            let small = t <= SMALL_T;
            ik::route(s, ws.rlogits, layer.bias, cfg.route_scale, t, ws.sel_e, ws.sel_w, if small { ws.order } else { 0 })?;
            if !small {
                ik::moe_plan(s, ws.sel_e, t, MOE_BM, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.pair_row)?;
                ik::moe_gather(s, ws.xm, ws.pair_row, t, ws.xs)?;
                let gemm = |x: u64, which: usize, n: usize, kk: usize| {
                    let st = if which == 0 { layer.gate_up } else { layer.down };
                    match &layer.tables {
                        Some(tb) => k::moe_nvfp4_gemm(
                            s, x, tb[which].0, tb[which].1, st.s2, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.max_tiles, ws.y, n, kk,
                        ),
                        None => ik::mx8_grouped_gemm(s, x, st, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.max_tiles, ws.y, n, kk),
                    }
                };
                gemm(ws.xs, 0, 2 * FF, D)?;
                ik::moe_act(s, ws.y, t * TOPK, cfg.gate_cap, cfg.up_cap, ws.act)?;
                gemm(ws.act, 1, D, FF)?;
                ik::moe_combine_rows(s, ws.y, ws.pair_row, ws.sel_w, t, ws.moe)?;
            } else {
            ik::experts(
                s,
                ws.xm,
                ws.sel_e,
                ws.sel_w,
                layer.fp4,
                layer.gate_up,
                layer.down,
                cfg.gate_cap,
                cfg.up_cap,
                t,
                ws.order,
                small,
                ws.act,
                ws.y,
                if small { 0 } else { ws.moe },
            )?;
            }
            hi ^= 1;
            let (src_x, cy, cw) = if small { (0, ws.y, ws.sel_w) } else { (ws.moe, 0, 0) };
            pending = Some(ConvSrc {
                src: src_x,
                cy,
                cw,
                ring: c.mlp_ring,
                ring_rows: RING,
                conv: layer.mlp_conv,
                base: ws.r,
                start,
            });
            if let Some(dir) = &dump {
                flush(&mut pending, ws.h[hi])?;
                // Debugging: the stream after each layer visit, appended per chunk (F32 [t, 2304]).
                use std::io::Write;
                let v32: Vec<f32> = s.download(ws.h[hi], t * D)?;
                let bytes: Vec<u8> = v32.iter().flat_map(|x| x.to_le_bytes()).collect();
                let path = std::path::Path::new(dir).join(format!("hidden-{v:02}.bin"));
                std::fs::OpenOptions::new().create(true).append(true).open(path)?.write_all(&bytes)?;
            }
        }
        let hidden = ws.feats + (3 * D * 2) as u64;
        if want != Logits::None || self.drafter.is_some() {
            let mut src: Vec<u64> = ws.slots[..nb].to_vec();
            src.push(ws.h[hi]);
            ik::attn_res(s, &src, self.final_query, 1, t, 0, &[(self.norm, hidden)], 4 * D, pending.take())?;
        }
        flush(&mut pending, ws.h[hi])?;
        if prefetched {
            ws.join.record(&ws.side)?;
            s.wait(&ws.join)?;
        }
        if want != Logits::None {
            let (rows, first) = if want == Logits::All { (t, 0) } else { (1, t - 1) };
            linear(s, hidden + (first * 4 * D * 2) as u64, 4 * D, rows, &self.lm_head, ws.out_logits, cfg.vocab)?;
            ik::softcap(s, ws.out_logits, rows, cfg.vocab, cfg.vocab, cfg.logit_cap)?;
        }
        if let (Some(d), Some(dw)) = (&self.drafter, &ws.draft) {
            // The drafter's context keys and values of this step's positions.
            for (i, l) in d.layers.iter().enumerate() {
                ik::mix(s, ws.feats, 0, d.alpha + (i * 16) as u64, l.mix_norm, dw.mixed, D, t)?;
                linear(s, dw.mixed, D, t, &l.kv_ctx, dw.kv, 2 * KVD)?;
                let (kc, vc) = state.draft_kv[i];
                ik::ctx_kv(s, dw.kv, l.k_norm_ctx, d.theta, start, t, kc, vc)?;
            }
        }
        Ok(())
    }
}
