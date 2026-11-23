//! Inkling 2 forward pass over a chunk of new tokens of one sequence, or a small step of several.
//!
//! Per layer visit (48: the looped layers once per pass) a sequence keeps the keys and values (every
//! position for full-attention layers, a ring of the last `RING` for sliding ones) and rings of the inputs
//! of its four causal convolutions (key and value projections, attention and MoE outputs). Chunks hold at
//! most `max_t` tokens (`max_t + WINDOW <= RING`).
//!
//! Full-length caches (full-attention visits, the drafter's full-attention layer) are reserved as virtual
//! memory for the sequence's capacity and backed by device memory as positions are used
//! ([`State::reserve`], [`State::trim`]), so sequences share the device's memory by length.
//!
//! The kernels find each row's sequence (its position, caches and rings) in a small device descriptor
//! (`ik::Seqs`, written per step) and the sequence's table of buffer addresses (written once), so a step's
//! CUDA graph is captured by shape (rows, logits) and serves any sequences. Small steps (at most 16 tokens of
//! one sequence, or [`InklingModel::forward_batch`] over several) run kernels whose per-row results do not
//! depend on the step's other rows: a sequence's bits are the same alone or batched.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use quench6k_cuda::inkling::{self as ik, ConvSrc, KvSide, RmsSet, Seqs};
use quench6k_cuda::kernels as k;
use quench6k_cuda::{DeviceBuffer, PinnedBuffer, Stream, VmBuffer};

use super::*;

pub const RING: usize = 4096;
/// Block-sum slots: embedding, blocks of the stream, and the previous loop pass's last block and partial.
pub const SLOTS: usize = 8;
const PREV_BLOCK: usize = SLOTS - 2;
const PREV_PARTIAL: usize = SLOTS - 1;
const SLIDING_SPLITS: usize = 16;
const FULL_SPLITS: usize = 32;
const MIN_KEYS: usize = 32;
/// Steps of more tokens (one sequence: prefill chunks) run NVFP4 experts as grouped tensor-core GEMMs
/// (tiles of MOE_BM rows), the attention on tensor cores and the projections as GEMMs; steps of at most
/// this many tokens per sequence run the small-step kernels (per-row results independent of the step).
pub const SMALL_T: usize = 16;
/// Most sequences of a batched step ([`InklingModel::forward_batch`], [`InklingModel::draft_batch`]).
pub const MAX_SEQS: usize = ik::MAX_SEQ;
/// Most rows of a batched step (all sequences' tokens).
pub const BATCH_ROWS: usize = 64;
/// Steps of at most this many tokens prefetch the next visit's weights: wider ones read so many experts
/// that the prefetch only competes with them (verify steps of 3 / 5 / 6 tokens: +4.5% / +2.6% / -1.6%).
const PREFETCH_T: usize = 5;
const MOE_BM: usize = 64;
pub const TOP_K: usize = 32;
/// Rewinds keep positions within this many of the current one: the sliding rings hold RING positions, a
/// resumed step needs the WINDOW before it, and the drafter's local layers (window 1024 + its block) also
/// hold a block written past the position.
const REACH: usize = RING - WINDOW - 16;
/// Rows of the sliding keys and values a snapshot keeps: the target's window (1024 positions before a
/// resumed step) and the drafter's (its window of 1024 + block ends at the block's last row).
const SNAP_ROWS: usize = WINDOW + 16;
/// Canonical prefill: prompts are computed in chunks aligned to multiples of this from position 0 (the
/// last one ends at the prompt's end), and reuse stops at such a boundary (see [`State::rewind`]), so a
/// prompt's bits depend only on its tokens, not on what the sequence held before.
pub const CANON_CHUNK: usize = 2048;

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
    /// Capacity: most positions this sequence may reach.
    pub ctx: usize,
    caches: Vec<VisitCache>,
    pub history: Vec<i32>,
    /// Device bytes of the fixed buffers (rings, sliding caches); [`State::device_bytes`] adds the mapped
    /// full-length caches and the snapshot.
    pub device_bytes: usize,
    /// The drafter's keys and values per layer (k, v, rows: the capacity for full-attention layers, RING
    /// for local ones): context rows from the target's features, block rows from its own passes.
    draft_kv: Vec<(u64, u64, usize)>,
    pub last_t: usize,
    /// The first row of this sequence's tokens in the last step (its rows of the workspace's features).
    last_row0: usize,
    /// The rings' rows that resuming at snap_pos (the end of the last prompt) reads: the last SNAP_ROWS
    /// positions of the sliding keys and values, the last CONVK of the convolution inputs.
    snap: Option<DeviceBuffer>,
    snap_pos: Option<usize>,
    /// Positions below this are not in the rings (after a restore): no rewind below it.
    floor: usize,
    /// See [`State::frontier`].
    frontier: usize,
    /// Buffer addresses per layer visit, then per drafter layer (ik::TAB u64 each; see ik::T_*).
    tab: DeviceBuffer,
    /// Full-length caches (virtual reservations for `ctx` positions, mapped as positions are used).
    grow: Vec<VmBuffer>,
    /// The drafter's features of the last step's last ik::feat_rows() rows (row t at t % feat_rows).
    _feat: DeviceBuffer,
    _alloc: Alloc,
}

/// Bytes of one position in one full-length cache (keys or values).
const ROW_BYTES: usize = KVD * 2;

impl State {
    /// A sequence of up to `capacity` positions (at most the model's context, 1,048,576). Full-length
    /// caches are reserved as virtual memory and mapped by [`State::reserve`] (forward and draft map what
    /// they write); rings and small state are allocated now.
    pub fn new(model: &InklingModel, capacity: usize, s: &Stream) -> Result<Self> {
        ik::check_layout()?;
        ensure!(capacity > 0 && capacity <= model.cfg.max_context, "capacity {capacity} (the model's context is {})", model.cfg.max_context);
        let ctx = capacity;
        let mut a = Alloc::new();
        let mut grow = Vec::new();
        let full = |grow: &mut Vec<VmBuffer>| -> Result<u64> {
            let b = VmBuffer::reserve(ctx * ROW_BYTES)?;
            let addr = b.addr();
            grow.push(b);
            Ok(addr)
        };
        let mut caches = Vec::new();
        for (layer, _) in model.cfg.visits() {
            let (k, v, rows) = if model.cfg.sliding[layer] {
                (a.get(RING * ROW_BYTES)?, a.get(RING * ROW_BYTES)?, RING)
            } else {
                (full(&mut grow)?, full(&mut grow)?, ctx)
            };
            caches.push(VisitCache {
                k,
                v,
                rows,
                k_ring: a.get(RING * KVD * 2)?,
                v_ring: a.get(RING * KVD * 2)?,
                attn_ring: a.get(RING * D * 2)?,
                mlp_ring: a.get(RING * D * 2)?,
            });
        }
        let mut draft_kv = Vec::new();
        if let Some(d) = &model.drafter {
            ensure!(d.window <= SNAP_ROWS && d.window + d.block <= RING - REACH, "drafter window {}", d.window);
            for l in &d.layers {
                draft_kv.push(if l.local {
                    (a.get(RING * ROW_BYTES)?, a.get(RING * ROW_BYTES)?, RING)
                } else {
                    (full(&mut grow)?, full(&mut grow)?, ctx)
                });
            }
        }
        for b in &a.buffers {
            s.memset(b.addr(), 0, b.bytes())?;
        }
        let mut t = vec![0u64; (caches.len() + draft_kv.len()) * ik::TAB];
        for (v, c) in caches.iter().enumerate() {
            let e = &mut t[v * ik::TAB..];
            e[ik::T_K] = c.k;
            e[ik::T_V] = c.v;
            e[ik::T_KRING] = c.k_ring;
            e[ik::T_VRING] = c.v_ring;
            e[ik::T_ARING] = c.attn_ring;
            e[ik::T_MRING] = c.mlp_ring;
            e[ik::T_ROWS] = c.rows as u64;
        }
        for (i, &(kc, vc, rows)) in draft_kv.iter().enumerate() {
            let e = &mut t[(caches.len() + i) * ik::TAB..];
            e[ik::T_K] = kc;
            e[ik::T_V] = vc;
            e[ik::T_ROWS] = rows as u64;
        }
        let feat = DeviceBuffer::zeroed(ik::feat_rows() * 4 * D * 2)?;
        t[ik::T_FEAT] = feat.addr();
        let tab = DeviceBuffer::new(t.len() * 8)?;
        s.upload(&tab, &t)?;
        s.synchronize()?;
        Ok(Self {
            pos: 0,
            ctx,
            caches,
            history: Vec::new(),
            device_bytes: a.bytes + tab.bytes() + feat.bytes(),
            draft_kv,
            last_t: 0,
            last_row0: 0,
            snap: None,
            snap_pos: None,
            floor: 0,
            frontier: 0,
            tab,
            grow,
            _feat: feat,
            _alloc: a,
        })
    }

    /// Device bytes per position of the full-length caches (all of them together).
    pub fn bytes_per_position(model: &InklingModel) -> usize {
        let visits = model.cfg.visits().iter().filter(|&&(l, _)| !model.cfg.sliding[l]).count();
        let draft = model.drafter.as_ref().map_or(0, |d| d.layers.iter().filter(|l| !l.local).count());
        (visits + draft) * 2 * ROW_BYTES
    }

    /// Positions backed by device memory in every full-length cache.
    pub fn mapped_positions(&self) -> usize {
        self.grow.iter().map(|b| b.mapped() / ROW_BYTES).min().unwrap_or(self.ctx).min(self.ctx)
    }

    /// Device memory of this sequence: fixed buffers, mapped full-length caches and the snapshot.
    pub fn device_bytes(&self) -> usize {
        self.device_bytes + self.grow.iter().map(VmBuffer::mapped).sum::<usize>() + self.snap.as_ref().map_or(0, DeviceBuffer::bytes)
    }

    /// Maps device memory for positions < `positions` (at most the capacity) in every full-length cache.
    /// Out of device memory: an error whose message contains "out of memory"; nothing changed.
    pub fn reserve(&mut self, positions: usize) -> Result<()> {
        let positions = positions.min(self.ctx);
        if positions <= self.mapped_positions() {
            return Ok(());
        }
        let before: Vec<usize> = self.grow.iter().map(VmBuffer::mapped).collect();
        for i in 0..self.grow.len() {
            if let Err(e) = self.grow[i].map_to(positions * ROW_BYTES) {
                for (b, &m) in self.grow.iter_mut().zip(&before) {
                    b.unmap_to(m)?;
                }
                return Err(e.context(format!("inkling: caches for {positions} positions")));
            }
        }
        Ok(())
    }

    /// Unmaps full-length cache memory beyond `positions` and forgets the cached tokens there: the
    /// sequence keeps its first `positions` tokens if the rings still hold the positions before them
    /// (within REACH of the current position), else none. Waits for the device first.
    pub fn trim(&mut self, positions: usize) {
        if positions < self.pos {
            let keep = if self.pos <= positions + REACH && positions >= self.floor { positions } else { 0 };
            self.truncate(keep);
        }
        self.lower_frontier(positions);
        if let Err(e) = quench6k_cuda::synchronize() {
            eprintln!("inkling: trim: {e:#}");
            return;
        }
        for b in &mut self.grow {
            if let Err(e) = b.unmap_to(positions * ROW_BYTES) {
                eprintln!("inkling: trim: {e:#}");
            }
        }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        self.history.clear();
        self.snap_pos = None;
        self.floor = 0;
        self.frontier = 0;
    }

    /// Drops positions from `pos` on. The rings keep the last RING positions, so `pos` must be within
    /// REACH positions of the current one (see [`State::rewind`]).
    pub fn truncate(&mut self, pos: usize) {
        self.pos = self.pos.min(pos);
        self.history.truncate(self.pos);
    }

    /// The canonical frontier: every position below it was computed by full prefill chunks aligned to
    /// [`CANON_CHUNK`] (a multiple of it).
    pub fn frontier(&self) -> usize {
        self.frontier
    }

    /// Positions from `p` on are rewritten (or dropped): the frontier falls to the boundary at or below `p`,
    /// and a snapshot above it is gone.
    fn lower_frontier(&mut self, p: usize) {
        if p < self.frontier {
            self.frontier = p / CANON_CHUNK * CANON_CHUNK;
        }
        if self.snap_pos.is_some_and(|e| e > self.frontier) {
            self.snap_pos = None;
        }
    }

    /// Keeps the cached prefix of `prompt` that canonical prefill reuses and returns its length B: the
    /// largest multiple of [`CANON_CHUNK`] at most the common prefix with the history, below the prompt's
    /// length and at most the canonical frontier, whose rings are still live (within REACH) or are the last
    /// boundary snapshot (restored); else 0. Everything from B on must be recomputed by
    /// [`InklingModel::prefill`] (earlier replies included), so a prompt's results depend only on its tokens.
    pub fn rewind(&mut self, prompt: &[i32], s: &Stream) -> Result<usize> {
        let common = self.history.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let most = common.min(prompt.len().saturating_sub(1)).min(self.frontier) / CANON_CHUNK * CANON_CHUNK;
        let live = if self.pos <= most + REACH && most >= self.floor { most } else { 0 };
        let snap = match self.snap_pos {
            Some(e) if e <= most && e <= self.pos => e,
            _ => 0,
        };
        let keep = if snap > live {
            self.restore(s)?;
            snap
        } else {
            live
        };
        self.truncate(keep);
        Ok(keep)
    }

    /// Every ring (RING rows each) as (address, row bytes, rows a resumed step reads before its position):
    /// per visit the convolution inputs, the sliding layers' keys and values; the drafter's local layers'.
    fn rings(&self) -> Vec<(u64, usize, usize)> {
        let mut v = Vec::new();
        for c in &self.caches {
            v.push((c.k_ring, KVD * 2, CONVK));
            v.push((c.v_ring, KVD * 2, CONVK));
            v.push((c.attn_ring, D * 2, CONVK));
            v.push((c.mlp_ring, D * 2, CONVK));
            if c.rows == RING {
                v.push((c.k, ROW_BYTES, SNAP_ROWS));
                v.push((c.v, ROW_BYTES, SNAP_ROWS));
            }
        }
        for &(kc, vc, rows) in &self.draft_kv {
            if rows == RING {
                v.push((kc, ROW_BYTES, SNAP_ROWS));
                v.push((vc, ROW_BYTES, SNAP_ROWS));
            }
        }
        v
    }

    /// Copies between each ring's rows of positions [e - keep, e) and the snapshot (in that order), in at
    /// most two runs per ring.
    fn snap_copy(&self, s: &Stream, e: usize, save: bool) -> Result<()> {
        let mut at = self.snap.as_ref().context("no snapshot")?.addr();
        for (addr, rb, keep) in self.rings() {
            let p0 = e.saturating_sub(keep);
            let mut p = p0;
            while p < e {
                let r = p % RING;
                let n = (e - p).min(RING - r);
                let (ring, snap) = (addr + (r * rb) as u64, at + ((p - p0) * rb) as u64);
                if save {
                    s.copy_d2d(snap, ring, n * rb)?;
                } else {
                    s.copy_d2d(ring, snap, n * rb)?;
                }
                p += n;
            }
            at += (keep * rb) as u64;
        }
        Ok(())
    }

    /// Kept for callers of the prompt-end snapshot: canonical prefill takes boundary snapshots itself
    /// (see [`State::rewind`]), so this does nothing.
    pub fn snapshot(&mut self, _s: &Stream) -> Result<()> {
        Ok(())
    }

    /// Copies what resuming at the current position (a canonical boundary) reads of the rings.
    fn boundary_snapshot(&mut self, s: &Stream) -> Result<()> {
        if self.snap.is_none() {
            let total: usize = self.rings().iter().map(|r| r.1 * r.2).sum();
            self.snap = Some(DeviceBuffer::new(total)?);
        }
        self.snap_copy(s, self.pos, true)?;
        self.snap_pos = Some(self.pos);
        Ok(())
    }

    fn restore(&mut self, s: &Stream) -> Result<()> {
        let e = self.snap_pos.context("no snapshot")?;
        self.snap_copy(s, e, false)?;
        self.pos = e;
        self.history.truncate(e);
        // The rings hold nothing usable before the snapshot's rows.
        self.floor = e;
        Ok(())
    }

    /// Bookkeeping after a step that fed `tokens` from row `row0` of the workspace: a full chunk aligned
    /// to CANON_CHUNK at the frontier advances it (and takes the boundary snapshot), any other write below
    /// it lowers it.
    fn advance(&mut self, tokens: &[i32], row0: usize, s: &Stream) -> Result<()> {
        let p0 = self.pos;
        self.lower_frontier(p0);
        self.pos += tokens.len();
        self.last_t = tokens.len();
        self.last_row0 = row0;
        self.history.extend_from_slice(tokens);
        if tokens.len() == CANON_CHUNK && p0 == self.frontier && p0.is_multiple_of(CANON_CHUNK) {
            self.frontier = self.pos;
            self.boundary_snapshot(s)?;
        }
        Ok(())
    }
}

/// Scratch buffers for chunks of up to `max_t` tokens.
pub struct Workspace {
    pub max_t: usize,
    pub logit_rows: usize,
    /// The step's sequences (ik::Seqs, device), written with the tokens from the pinned buffer.
    seqs: u64,
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
    /// F32 [SPLIT_K, t, 128]: partial sums of the narrow GEMMs (router, message gates' down).
    splitk: u64,
    attn: u64,
    o: u64,
    rlogits: u64,
    sel_e: u64,
    sel_w: u64,
    order: u64,
    /// Groups of same-expert pairs for the grouped expert kernels (1 + 2 BATCH_ROWS * 8 ints).
    groups: u64,
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
    /// Tokens (max_t I32) then the step's ik::Seqs at `seqs_at`.
    pinned: PinnedBuffer,
    seqs_at: usize,
    /// CUDA graphs of small steps by (rows, logits kind); they read the sequences from `seqs`.
    graphs: HashMap<(usize, u8), quench6k_cuda::Graph>,
    /// The drafter's pass as CUDA graphs by sequence count.
    /// Drafter graphs by (sequences, drafts computed per sequence).
    dgraphs: HashMap<(usize, usize), quench6k_cuda::Graph>,
    /// The side stream of the L2 prefetches (forked from and joined into the step's stream).
    side: Stream,
    fork: quench6k_cuda::Event,
    join: quench6k_cuda::Event,
    /// One-token steps: the next visit's experts guessed (its router on this visit's MLP input) and the
    /// best ones' gate/up weights requested into L2 after that visit's output projection, on a second
    /// side stream.
    guess: Stream,
    guess_x: quench6k_cuda::Event,
    guess_done: quench6k_cuda::Event,
    guess_moe: quench6k_cuda::Event,
    guess_join: quench6k_cuda::Event,
    guess_logits: u64,
    guess_e: u64,
    guess_w: u64,
    pub device_bytes: usize,
    _alloc: Alloc,
}

impl Workspace {
    /// Chunks of up to `max_t` tokens; `logit_rows`: most rows of logits a step produces (1 for
    /// generation; max_t for scoring every position; the rows of a batched step).
    pub fn new(model: &InklingModel, max_t: usize, logit_rows: usize) -> Result<Self> {
        ensure!(max_t + WINDOW <= RING, "chunks of {max_t} tokens exceed the rings");
        ensure!(ik::small_rows() == BATCH_ROWS, "inkling: kernels take {} rows a small step", ik::small_rows());
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
        let seqs_at = (t * 4).next_multiple_of(256);
        let ws = Self {
            max_t,
            logit_rows,
            seqs: a.get(std::mem::size_of::<Seqs>())?,
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
            // Also the three projections' outputs together [t, QG + 2 * ldk] (prefill, one input).
            lin_q: a.get(t * (QGP + 2 * (KVD + 32)) * 4)?,
            lin_k: a.get(t * (KVD + 32) * 4)?,
            lin_v: a.get(t * (KVD + 32) * 4)?,
            q: a.get(t * QD * 2)?,
            rel: a.get(t * H * DREL * 4)?,
            sink: a.get(t * H * 4)?,
            // Split-key partials of small steps (larger ones run the tensor-core kernel).
            part: a.get(ik::attn_part_floats(t.min(BATCH_ROWS), FULL_SPLITS) * 4)?,
            splitk: a.get(SPLIT_K * t * 128 * 4)?,
            attn: a.get(t * QD * 2)?,
            o: a.get(td)?,
            rlogits: a.get(t * EXPERTS * 4)?,
            sel_e: a.get(pairs * 4)?,
            sel_w: a.get(pairs * 4)?,
            order: a.get(pairs * 4)?,
            groups: a.get((1 + 2 * BATCH_ROWS * TOPK) * 4)?,
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
            pinned: PinnedBuffer::new(seqs_at + std::mem::size_of::<Seqs>())?,
            seqs_at,
            graphs: HashMap::new(),
            dgraphs: HashMap::new(),
            side: Stream::new()?,
            fork: quench6k_cuda::Event::new()?,
            join: quench6k_cuda::Event::new()?,
            guess: Stream::new()?,
            guess_x: quench6k_cuda::Event::new()?,
            guess_done: quench6k_cuda::Event::new()?,
            guess_moe: quench6k_cuda::Event::new()?,
            guess_join: quench6k_cuda::Event::new()?,
            guess_logits: a.get(PREFETCH_T * EXPERTS * 4)?,
            guess_e: a.get(PREFETCH_T * TOPK * 4)?,
            guess_w: a.get(PREFETCH_T * TOPK * 4)?,
            device_bytes: 0,
            _alloc: Alloc::new(),
        };
        let draft = match &model.drafter {
            Some(d) => {
                let b = d.block;
                // Rows of a batched pass: one block per sequence.
                let rb = MAX_SEQS * b;
                ensure!(rb <= BATCH_ROWS, "drafter blocks of {b} rows");
                let ff2 = d.layers[0].gate_up.n;
                let stage_bytes = DRAFT_SEQS_AT + std::mem::size_of::<Seqs>();
                Some(DraftWs {
                    stage: a.get(stage_bytes)?,
                    stage_bytes,
                    drafts: a.get(MAX_SEQS * 2 * b * 4)?,
                    prev: a.get(MAX_SEQS * 4)?,
                    x: a.get(rb * D * 4)?,
                    xn: a.get(rb * D * 2)?,
                    qkv: a.get(rb * d.layers[0].qkv.n * 4)?,
                    q: a.get(rb * QD * 2)?,
                    attn: a.get(rb * QD * 2)?,
                    gu: a.get(rb * ff2 * 4)?,
                    act: a.get(rb * ff2 / 2 * 2)?,
                    hf: a.get(rb * 2 * D * 2)?,
                    base: a.get(rb * vocab * 4)?,
                    ctx: a.get(rb * d.markov_hf.n * 4)?,
                    state: a.get(MAX_SEQS * 2 * 256 * 2)?,
                    latent: a.get(MAX_SEQS * 256 * 2)?,
                    best_v: a.get(MAX_SEQS * 2 * ik::markov_blocks(vocab) * 4)?,
                    best_i: a.get(MAX_SEQS * ik::markov_blocks(vocab) * 4)?,
                    cand: a.get(rb * ik::markov_cand_blocks(vocab) * MARKOV_CAND_MAX * 4)?,
                    mixed: a.get(t * D * 2)?,
                    kv: a.get(t * 2 * KVD * 4)?,
                    pinned: PinnedBuffer::new(stage_bytes.next_multiple_of(64) + MAX_SEQS * 2 * b * 4)?,
                    confs: Vec::new(),
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

/// Where the drafter's stage holds the step's ik::Seqs (after the anchors and feature rows, MAX_SEQS each).
const DRAFT_SEQS_AT: usize = 64;

impl Workspace {
    /// Writes a step's tokens (sequence by sequence) and sequences into the pinned buffer (the step's
    /// first copies read them; the stream must be idle).
    fn stage(&mut self, seqs: &[(&State, &[i32])]) {
        let mut sq = Seqs {
            nseq: seqs.len() as i32,
            ..Default::default()
        };
        let buf = self.pinned.as_mut_slice();
        let mut row = 0;
        for (j, (st, tokens)) in seqs.iter().enumerate() {
            sq.tab[j] = st.tab.addr();
            sq.start[j] = st.pos as i32;
            sq.row0[j] = row as i32;
            for &tok in *tokens {
                buf[row * 4..row * 4 + 4].copy_from_slice(&tok.to_le_bytes());
                row += 1;
            }
        }
        sq.row0[seqs.len()] = row as i32;
        buf[self.seqs_at..self.seqs_at + std::mem::size_of::<Seqs>()].copy_from_slice(sq.as_bytes());
    }
}

/// Buffers of the drafter's pass over its blocks (`block` rows per sequence) and of its context keys
/// (max_t rows).
struct DraftWs {
    /// Device: per sequence the accepted token (I32 [MAX_SEQS]), the feature row of the token before it
    /// (I32 [MAX_SEQS] at 32), then the pass's ik::Seqs at DRAFT_SEQS_AT; written from `pinned`.
    stage: u64,
    stage_bytes: usize,
    /// Per sequence: drafts (I32 [block - 1]) then their probabilities (F32 [block - 1]).
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
    /// I32 [block - 1][cand blocks * MARKOV_CAND_MAX]: the Markov head's candidate rows.
    cand: u64,
    mixed: u64,
    kv: u64,
    /// The stage (stage_bytes), then (at stage_bytes rounded up to 64) the drafts read back.
    pinned: PinnedBuffer,
    /// The last pass's per-draft probabilities, by sequence of the call.
    confs: Vec<Vec<f32>>,
}

/// Y [m, l.n] F32 = X [m, l.k] BF16 . W^T (accumulated into Y with `acc`). Small steps (`small`, up to
/// BATCH_ROWS rows) run the skinny tensor-core GEMM or the GEMV in launches of at most gemv_max_m rows:
/// either way a row's result does not depend on the other rows. Larger steps run the GEMM.
#[allow(clippy::too_many_arguments)]
fn linear_acc(s: &Stream, x: u64, ldx: usize, m: usize, l: &Linear, y: u64, ldy: usize, small: bool, acc: bool) -> Result<()> {
    if !acc && skinny(m, l, small) {
        return ik::skinny_i8(s, m, l.w, l.scale, ldx, &[(l.n, x, y, ldy)], l.k);
    }
    let g = k::gemv_max_m();
    if small || m <= g {
        for r0 in (0..m).step_by(g) {
            let mm = g.min(m - r0);
            k::gemv(s, mm, x + (r0 * ldx * 2) as u64, ldx, l.fmt, l.w, l.scale, y + (r0 * ldy * 4) as u64, ldy, 0, l.n, l.k, 1, 1.0, acc)?;
        }
        Ok(())
    } else {
        ensure!(!acc, "accumulating GEMM");
        k::gemm_f32(s, x, ldx, l.fmt, l.w, l.k, l.scale, y, ldy, m, l.n, l.k, 1.0, false)
    }
}

#[allow(clippy::too_many_arguments)]
fn linear(s: &Stream, x: u64, ldx: usize, m: usize, l: &Linear, y: u64, ldy: usize, small: bool) -> Result<()> {
    linear_acc(s, x, ldx, m, l, y, ldy, small, false)
}

/// K splits of the narrow GEMMs (outputs of at most 128 columns: one tile column, too few CTAs).
const SPLIT_K: usize = 8;

/// [`linear`] for outputs of at most 128 columns: GEMMs split along K.
#[allow(clippy::too_many_arguments)]
fn linear_narrow(s: &Stream, ws: &Workspace, x: u64, ldx: usize, m: usize, l: &Linear, y: u64, ldy: usize, small: bool) -> Result<()> {
    if small || m <= k::gemv_max_m() || l.n > 128 {
        return linear(s, x, ldx, m, l, y, ldy, small);
    }
    k::gemm_f32_split(s, x, ldx, l.fmt, l.w, l.k, l.scale, y, ldy, m, l.n, l.k, 1.0, false, ws.splitk, SPLIT_K)
}

/// QUENCH6K_INK_SKINNY: steps of at least this many tokens run INT8 projections as the skinny tensor-core
/// GEMM, smaller ones as the GEMV (default 1: every small step, so that a token's result does not depend
/// on the step's width and drafted decoding gives the plain tokens; 0: never).
fn skinny_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QUENCH6K_INK_SKINNY").ok().and_then(|v| v.parse().ok()).unwrap_or(1))
}

/// Whether a step of m rows runs `l` as the skinny tensor-core GEMM (INT8 group weights) instead of the
/// GEMV (small steps: up to BATCH_ROWS rows; else up to 16).
fn skinny(m: usize, l: &Linear, small: bool) -> bool {
    let min = skinny_min();
    let most = if small { BATCH_ROWS } else { 16 };
    min > 0 && m >= min && m <= most && l.fmt == WeightFormat::Int8Group && l.n.is_multiple_of(16) && l.k.is_multiple_of(64)
}

const MARKOV_CAND_MAX: usize = 8;

fn markov_cand() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QUENCH6K_INK_MARKOV_CAND").ok().and_then(|v| v.parse().ok()).unwrap_or(4).min(MARKOV_CAND_MAX))
}

fn fp4_tc() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QUENCH6K_INK_FP4_TC").is_ok_and(|v| v == "1"))
}

fn fp4_terms() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QUENCH6K_INK_FP4_TERMS").ok().and_then(|v| v.parse().ok()).unwrap_or(2))
}

/// Guessed experts prefetched per routing row of a step of t tokens (0: no guess; QUENCH6K_INK_GUESS=0: off;
/// QUENCH6K_INK_GUESS_K=k1,k2,..: per width, for tuning).
fn guess_k(t: usize) -> usize {
    static V: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    let ks = V.get_or_init(|| {
        if std::env::var("QUENCH6K_INK_GUESS").is_ok_and(|v| v == "0") {
            return Vec::new();
        }
        match std::env::var("QUENCH6K_INK_GUESS_K") {
            Ok(v) => v.split(',').filter_map(|x| x.parse().ok()).collect(),
            // Steps of 1 / 2 / 3 / 4 tokens (verify steps of 2 / 3 / 4: +3.4% / +3.4% / +1%).
            Err(_) => vec![3, 2, 2, 1],
        }
    });
    if t > PREFETCH_T { 0 } else { ks.get(t - 1).copied().unwrap_or(0) }
}

/// Small steps of at least this many rows run the grouped expert kernels (QUENCH6K_INK_EXPERT_GROUPS;
/// 0: never). Their results are bitwise the per-pair kernels': this only moves time.
fn expert_groups_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("QUENCH6K_INK_EXPERT_GROUPS").ok().and_then(|v| v.parse().ok()) {
        Some(0) => usize::MAX,
        Some(n) => n,
        // 8 rows: slower (8 sequences of 1 token: 11.9 -> 13.3 ms); 16: even; 32: 39.0 -> 35.2 ms.
        None => 16,
    })
}

/// Small steps of at least this many rows run the per-pair expert kernels with each block's inputs staged
/// in shared memory on NVFP4 layers (QUENCH6K_INK_EXPERT_STAGED, 0: never). Their results are bitwise the
/// per-pair kernels'.
fn expert_staged_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("QUENCH6K_INK_EXPERT_STAGED").ok().and_then(|v| v.parse().ok()) {
        Some(0) => usize::MAX,
        Some(n) => n,
        None => 1,
    })
}

/// Small steps of more rows than this run the per-pair kernels without staging (QUENCH6K_INK_EXPERT_STAGED_MAX).
fn expert_staged_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QUENCH6K_INK_EXPERT_STAGED_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(BATCH_ROWS))
}

/// Small steps of more rows than this run the LM head as ik::lm_rows (one pass over the weights for all
/// rows; the same bits as the GEMV, which takes at most 16 rows a pass and reaches the weights' bandwidth
/// up to 4). QUENCH6K_INK_LM_ROWS=0: the GEMV always.
const LM_ROWS_GEMV: usize = 4;

fn lm_rows_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QUENCH6K_INK_LM_ROWS").map_or(true, |v| v != "0"))
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
        self.check_tokens(tokens)?;
        state.reserve(state.pos + t)?;
        // The pinned buffer may still feed the previous step's upload.
        s.synchronize()?;
        ws.stage(&[(&*state, tokens)]);
        self.launch(ws, s, t, want, t <= SMALL_T)?;
        state.advance(tokens, 0, s)?;
        Ok(())
    }

    /// One small step of several sequences: `tokens[i]` (1 to 16 tokens, at most BATCH_ROWS in all) at
    /// positions states[i].pos ... Logits of every fed token land in ws.out_logits, sequence by sequence and
    /// row by row (the workspace must hold that many rows). Each sequence's results (caches, hidden states,
    /// logits, features for its drafts) are bit-identical to [`InklingModel::forward`] of its tokens alone
    /// with `Logits::All`. Graphs are captured by row count.
    pub fn forward_batch(&self, states: &mut [&mut State], ws: &mut Workspace, s: &Stream, tokens: &[&[i32]]) -> Result<()> {
        let n = states.len();
        ensure!((1..=MAX_SEQS).contains(&n) && tokens.len() == n, "a batch of {n} sequences ({} token lists; at most {MAX_SEQS})", tokens.len());
        let rows: usize = tokens.iter().map(|t| t.len()).sum();
        ensure!(rows <= BATCH_ROWS && rows <= ws.max_t, "a batch of {rows} rows (at most {BATCH_ROWS})");
        ensure!(rows <= ws.logit_rows, "{rows} rows of logits (workspace holds {})", ws.logit_rows);
        // Below that, a step's rows would choose skinny GEMM or GEMV by the step's width, not the sequence's.
        ensure!(n == 1 || skinny_min() <= 1, "QUENCH6K_INK_SKINNY above 1 is not width-invariant; batching needs 0 or 1");
        for (st, tok) in states.iter().zip(tokens) {
            ensure!(!tok.is_empty() && tok.len() <= SMALL_T, "{} tokens of one sequence in a batched step (1 to {SMALL_T})", tok.len());
            ensure!(st.pos + tok.len() <= st.ctx, "context full ({} + {} > {})", st.pos, tok.len(), st.ctx);
            self.check_tokens(tok)?;
        }
        for (st, tok) in states.iter_mut().zip(tokens) {
            st.reserve(st.pos + tok.len())?;
        }
        s.synchronize()?;
        let seqs: Vec<(&State, &[i32])> = states.iter().map(|st| &**st).zip(tokens.iter().copied()).collect();
        ws.stage(&seqs);
        self.launch(ws, s, rows, Logits::All, true)?;
        let mut row0 = 0;
        for (st, tok) in states.iter_mut().zip(tokens) {
            st.advance(tok, row0, s)?;
            row0 += tok.len();
        }
        Ok(())
    }

    /// Canonical prefill of `tokens` at state.pos (after [`State::rewind`]): chunks end at multiples of
    /// [`CANON_CHUNK`] (absolute positions) and at the tokens' end; the last chunk's last logits row lands
    /// in ws.out_logits. The workspace must take chunks of CANON_CHUNK tokens.
    pub fn prefill(&self, state: &mut State, ws: &mut Workspace, s: &Stream, tokens: &[i32]) -> Result<()> {
        ensure!(ws.max_t >= CANON_CHUNK, "canonical prefill needs chunks of {CANON_CHUNK} tokens (workspace: {})", ws.max_t);
        let mut at = 0;
        while at < tokens.len() {
            let n = (CANON_CHUNK - state.pos % CANON_CHUNK).min(tokens.len() - at);
            let want = if at + n == tokens.len() { Logits::Last } else { Logits::None };
            self.forward(state, ws, s, &tokens[at..at + n], want)?;
            at += n;
        }
        Ok(())
    }

    fn check_tokens(&self, tokens: &[i32]) -> Result<()> {
        for &tok in tokens {
            ensure!(tok >= 0 && (tok as usize) < self.cfg.embed_rows, "token {tok} out of range");
        }
        Ok(())
    }

    /// Runs the staged step: small steps as a CUDA graph by (rows, logits kind) unless QUENCH6K_INK_GRAPHS=0
    /// or QUENCH6K_INK_DUMP.
    fn launch(&self, ws: &mut Workspace, s: &Stream, rows: usize, want: Logits, small: bool) -> Result<()> {
        let dump = std::env::var_os("QUENCH6K_INK_DUMP").is_some();
        let graphs = std::env::var("QUENCH6K_INK_GRAPHS").map_or(true, |v| v != "0") && !dump;
        if graphs && small {
            let key = (rows, want as u8);
            if !ws.graphs.contains_key(&key) {
                let g = quench6k_cuda::Graph::capture(s, |s| self.enqueue(ws, s, rows, want, true))?;
                ws.graphs.insert(key, g);
            }
            ws.graphs[&key].launch(s)
        } else {
            self.enqueue(ws, s, rows, want, small)
        }
    }

    /// Drafts the tokens after `token`, the one the last step chose after its row `row` (the last accepted
    /// one; `token` is not in the caches yet and goes at position state.pos). Writes the block's drafter
    /// keys past state.pos (the next step overwrites them). Empty without a drafter or room in the context.
    pub fn draft(&self, state: &mut State, ws: &mut Workspace, s: &Stream, token: i32, row: usize) -> Result<Vec<i32>> {
        Ok(self.draft_batch(&mut [state], ws, s, &[token], &[row])?.pop().unwrap_or_default())
    }

    /// [`InklingModel::draft`] for several sequences in one pass (their drafts bit-identical to drafting
    /// each alone): `pending[i]` the token states[i]'s last step chose after its row `rows[i]` (one of its
    /// last 16). Each sequence keeps its last step's features, so other sequences' steps in between do
    /// not matter. Sequences without room for a block (or before any step) get no drafts.
    pub fn draft_batch(&self, states: &mut [&mut State], ws: &mut Workspace, s: &Stream, pending: &[i32], rows: &[usize]) -> Result<Vec<Vec<i32>>> {
        let all = vec![usize::MAX; states.len()];
        self.draft_batch_some(states, ws, s, pending, rows, &all)
    }

    /// [`InklingModel::draft_batch`] for the first `most[i]` drafts of sequence i only (at least one):
    /// the pass computes the drafter's head for the drafts up to the largest of them (every row of the
    /// block still goes through the drafter's layers). Each draft and its probability are bit-identical
    /// to the whole pass's.
    pub fn draft_batch_some(
        &self,
        states: &mut [&mut State],
        ws: &mut Workspace,
        s: &Stream,
        pending: &[i32],
        rows: &[usize],
        most: &[usize],
    ) -> Result<Vec<Vec<i32>>> {
        let n = states.len();
        ensure!(
            n <= MAX_SEQS && pending.len() == n && rows.len() == n && most.len() == n,
            "a draft batch of {n} sequences (at most {MAX_SEQS})"
        );
        let mut out = vec![Vec::new(); n];
        let (Some(d), Some(_)) = (&self.drafter, &ws.draft) else {
            return Ok(out);
        };
        let b = d.block;
        let live: Vec<usize> = (0..n).filter(|&i| states[i].pos + b <= states[i].ctx && states[i].last_t > 0).collect();
        for &i in &live {
            let lt = states[i].last_t;
            ensure!(rows[i] < lt && rows[i] + ik::feat_rows() >= lt, "draft source row {} of a {lt}-token step (its last {} rows)", rows[i], ik::feat_rows());
            self.check_tokens(&pending[i..i + 1])?;
        }
        for &i in &live {
            let p = states[i].pos;
            states[i].reserve(p + b)?;
            // The block's drafter keys go at p ..
            states[i].lower_frontier(p);
        }
        let dw = ws.draft.as_mut().unwrap();
        dw.confs.resize(n, Vec::new());
        if live.is_empty() {
            return Ok(out);
        }
        s.synchronize()?;
        let m = live.len();
        // Drafts computed per sequence.
        let nd = live.iter().map(|&i| most[i].clamp(1, b - 1)).max().unwrap();
        {
            let mut sq = Seqs {
                nseq: m as i32,
                ..Default::default()
            };
            let buf = dw.pinned.as_mut_slice();
            for (j, &i) in live.iter().enumerate() {
                let st = &states[i];
                buf[j * 4..j * 4 + 4].copy_from_slice(&pending[i].to_le_bytes());
                buf[32 + j * 4..32 + j * 4 + 4].copy_from_slice(&(rows[i] as i32).to_le_bytes());
                sq.tab[j] = st.tab.addr();
                sq.start[j] = st.pos as i32;
                sq.row0[j] = (j * b) as i32;
            }
            sq.row0[m] = (m * b) as i32;
            buf[DRAFT_SEQS_AT..DRAFT_SEQS_AT + std::mem::size_of::<Seqs>()].copy_from_slice(sq.as_bytes());
        }
        let graphs = std::env::var("QUENCH6K_INK_GRAPHS").map_or(true, |v| v != "0");
        if graphs {
            if !ws.dgraphs.contains_key(&(m, nd)) {
                let g = quench6k_cuda::Graph::capture(s, |s| self.draft_enqueue(ws, s, m, nd))?;
                ws.dgraphs.insert((m, nd), g);
            }
            ws.dgraphs[&(m, nd)].launch(s)?;
        } else {
            self.draft_enqueue(ws, s, m, nd)?;
        }
        s.synchronize()?;
        let dw = ws.draft.as_mut().unwrap();
        let all = b - 1;
        let back = dw.stage_bytes.next_multiple_of(64);
        for (j, &i) in live.iter().enumerate() {
            let words = &dw.pinned.as_slice()[back + j * 2 * all * 4..back + (j + 1) * 2 * all * 4];
            let w = |x: usize| <[u8; 4]>::try_from(&words[x * 4..x * 4 + 4]).unwrap();
            let k = most[i].clamp(1, nd);
            out[i] = (0..k).map(|x| i32::from_le_bytes(w(x))).collect();
            dw.confs[i] = (0..k).map(|x| f32::from_le_bytes(w(all + x))).collect();
        }
        if let Some(dir) = std::env::var_os("QUENCH6K_INK_DUMP_DRAFT")
            && m == 1
        {
            // Debugging (one sequence): the step's features, the block's [hidden | f_rnn] rows (from row 1)
            // and the base logits.
            let st = &states[live[0]];
            let dir = std::path::Path::new(&dir);
            let bytes = |v: Vec<u16>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            std::fs::write(dir.join("feats.bin"), bytes(s.download::<u16>(ws.feats + (st.last_row0 * 4 * D * 2) as u64, st.last_t * 4 * D)?))?;
            std::fs::write(dir.join("hf.bin"), bytes(s.download::<u16>(dw.hf + (2 * D * 2) as u64, nd * 2 * D)?))?;
            let base: Vec<f32> = s.download(dw.base + (self.cfg.vocab * 4) as u64, nd * self.cfg.vocab)?;
            std::fs::write(dir.join("base.bin"), base.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            let mut meta = Vec::new();
            for v in [pending[live[0]], st.pos as i32, rows[live[0]] as i32] {
                meta.extend_from_slice(&v.to_le_bytes());
            }
            meta.extend_from_slice(&dw.pinned.as_slice()[back..back + nd * 4]);
            std::fs::write(dir.join("stage.bin"), meta)?;
        }
        Ok(out)
    }

    /// The last draft pass's per-draft probabilities under the drafter's head (its first sequence; kept
    /// when the last call drafted nothing for it).
    pub fn draft_conf(&self, ws: &Workspace) -> Vec<f32> {
        ws.draft.as_ref().and_then(|dw| dw.confs.first().cloned()).unwrap_or_default()
    }

    /// [`InklingModel::draft_conf`] per sequence of the last [`InklingModel::draft_batch`] call.
    pub fn draft_conf_batch(&self, ws: &Workspace) -> Vec<Vec<f32>> {
        ws.draft.as_ref().map(|dw| dw.confs.clone()).unwrap_or_default()
    }

    /// The drafter's pass over `m` sequences' blocks (staged in dw.stage), computing the first `nd`
    /// drafts of each: block rows of sequence j are rows j * block ...; every kernel's row results are
    /// independent of the other rows.
    fn draft_enqueue(&self, ws: &Workspace, s: &Stream, m: usize, nd: usize) -> Result<()> {
        let cfg = &self.cfg;
        let d = self.drafter.as_ref().unwrap();
        let dw = ws.draft.as_ref().unwrap();
        let b = d.block;
        let r = m * b;
        let (anchor, rows_from, sq) = (dw.stage, dw.stage + 32, dw.stage + DRAFT_SEQS_AT as u64);
        unsafe { s.copy_h2d(dw.stage, dw.pinned.as_ptr(), dw.stage_bytes)? };
        let visits = cfg.visits().len();
        let acc = |x: u64, ldx: usize, l: &Linear, y: u64| linear_acc(s, x, ldx, r, l, y, D, true, true);
        ik::block_embed(s, anchor, sq, self.embed, self.embed_norm, d.mask, r, dw.x)?;
        for (i, l) in d.layers.iter().enumerate() {
            ik::rms(s, &[RmsSet { x: dw.x, w: l.in_norm, out: dw.xn, ldo: D }], r)?;
            linear(s, dw.xn, D, r, &l.qkv, dw.qkv, l.qkv.n, true)?;
            ik::draft_qkv(s, dw.qkv, l.q_norm, l.k_norm, d.theta, sq, visits + i, r, dw.q)?;
            let (window, splits) = if l.local { (d.window + b, SLIDING_SPLITS) } else { (0, FULL_SPLITS) };
            ik::attention(s, dw.q, 0, 0, sq, visits + i, 0, window, b, r, splits, MIN_KEYS, 0, 0, 0, ws.part, dw.attn, true)?;
            acc(dw.attn, QD, &l.o, dw.x)?;
            ik::rms(s, &[RmsSet { x: dw.x, w: l.post_norm, out: dw.xn, ldo: D }], r)?;
            linear(s, dw.xn, D, r, &l.gate_up, dw.gu, l.gate_up.n, true)?;
            ik::silu_mul(s, dw.gu, l.gate_up.n / 2, r, dw.act)?;
            acc(dw.act, l.down.k, &l.down, dw.x)?;
        }
        // Rows 1.. of each block predict the drafts: their final hidden states, and the Markov head's
        // context input [hidden | f_rnn] (f_rnn: the features of the token before the accepted one). Every
        // row is computed (row 0's are not used).
        let n = b - 1;
        ik::rms(s, &[RmsSet { x: dw.x, w: d.norm, out: dw.hf, ldo: 2 * D }], r)?;
        ik::mix(s, ws.feats, rows_from, sq, d.alpha + (d.layers.len() * 16) as u64, d.markov_norm, dw.hf + (D * 2) as u64, 2 * D, r)?;
        // The block rows that predict the drafts computed (1 ..= nd of each block; row 0's logits are not
        // used).
        if self.lm_head.fmt == WeightFormat::Fp8Block && lm_rows_on() {
            let l = &self.lm_head;
            ik::lm_rows(s, dw.hf, 2 * D, m * nd, b, nd, l.w, l.scale, dw.base, cfg.vocab, l.n, l.k)?;
        } else {
            linear(s, dw.hf, 2 * D, r, &self.lm_head, dw.base, cfg.vocab, true)?;
        }
        linear(s, dw.hf, 2 * D, r, &d.markov_hf, dw.ctx, d.markov_hf.n, true)?;
        // The head's picks among each vocabulary block's best rows by base + fold_b (QUENCH6K_INK_MARKOV_CAND:
        // rows per block, 0 = the whole vocabulary).
        let c = markov_cand();
        let per = ik::markov_cand_blocks(cfg.vocab) * c;
        if c > 0 {
            ik::markov_cand(s, dw.base, d.markov.fold_b, cfg.vocab, r, c, dw.cand)?;
        }
        s.memset(dw.state, 0, m * 2 * 256 * 2)?;
        s.copy_d2d(dw.prev, anchor, m * 4)?;
        let ss = ik::MarkovStrides {
            ctx: b * d.markov_hf.n,
            base: b * cfg.vocab,
            out: 2 * n,
            cand: b * per,
        };
        for i in 0..nd {
            let (sin, sout) = (dw.state + ((i % 2) * 512) as u64, dw.state + (((i + 1) % 2) * 512) as u64);
            ik::markov(
                s,
                m,
                d.markov,
                dw.ctx + ((1 + i) * d.markov_hf.n * 4) as u64,
                sin,
                dw.prev,
                sout,
                dw.latent,
                dw.base + ((1 + i) * cfg.vocab * 4) as u64,
                cfg.vocab,
                cfg.logit_cap,
                dw.best_v,
                dw.best_i,
                dw.drafts + (i * 4) as u64,
                dw.drafts + ((n + i) * 4) as u64,
                (c > 0).then(|| (dw.cand + ((1 + i) * per * 4) as u64, per)),
                ss,
            )?;
        }
        unsafe { s.copy_d2h(dw.pinned.as_ptr().add(dw.stage_bytes.next_multiple_of(64)), dw.drafts, m * 2 * n * 4)? };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// Message gates `which` (0 query, 1 key, 2 value, 3 MLP) of `layer` on inputs `hs`, messages `m`, into
    /// ws.gated.
    fn gates(&self, ws: &Workspace, s: &Stream, layer: &Layer, which: &[usize], hs: &[u64], m: &[u64], t: usize, small: bool) -> Result<()> {
        let g = layer.gates.as_ref().unwrap();
        if !small {
            // Prefill: the projections as GEMMs (the fused kernels re-read the weights per token).
            for (j, &i) in which.iter().enumerate() {
                let cat = ws.cat;
                ik::rms(s, &[RmsSet { x: hs[j], w: 0, out: cat, ldo: 2 * D }, RmsSet { x: m[j], w: 0, out: cat + (D * 2) as u64, ldo: 2 * D }], t)?;
                linear_narrow(s, ws, cat, 2 * D, t, &g[i].down, ws.zf, RANK, false)?;
                k::f32_to_bf16(s, ws.zf, ws.z, t * RANK)?;
                linear(s, ws.z, RANK, t, &g[i].fg, ws.fg, 2 * D, false)?;
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

    /// The staged step of `t` rows (ws.stage): `small` for small steps (per-row results independent of the
    /// step's rows), else a prefill chunk of one sequence.
    fn enqueue(&self, ws: &Workspace, s: &Stream, t: usize, want: Logits, small: bool) -> Result<()> {
        let cfg = &self.cfg;
        let pin = ws.pinned.as_ptr();
        unsafe {
            s.copy_h2d(ws.tokens, pin, t * 4)?;
            s.copy_h2d(ws.seqs, pin.add(ws.seqs_at), std::mem::size_of::<Seqs>())?;
        }
        let td = (t * D * 4) as u64;
        let sq = ws.seqs;
        let mut hi = 0;
        ik::embed(s, ws.tokens, t, self.embed, self.embed_norm, ws.h[0])?;
        let mut nb = 0usize;
        // The previous visit's MoE convolution (output ws.h[hi]), computed by the next consumer of the
        // stream (its input mixture) or on its own (`flush`).
        let mut pending: Option<ConvSrc> = None;
        let flush = |pending: &mut Option<ConvSrc>, out: u64| -> Result<()> {
            if let Some(c) = pending.take() {
                ik::hidden_conv(s, &c, t, out)?;
            }
            Ok(())
        };
        let dump = std::env::var_os("QUENCH6K_INK_DUMP");
        let mut prefetched = false;
        let mut guessed = false;
        let mut pending_guess: Option<&Layer> = None;
        for (v, &(li, pass)) in cfg.visits().iter().enumerate() {
            let layer = &self.layers[li];
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
            if t <= PREFETCH_T && prefetch_on() && let Some(&(nl, np)) = cfg.visits().get(v + 1) {
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
                ik::attn_res(s, &[ws.slots[PREV_BLOCK], ws.slots[PREV_PARTIAL]], q, 4, t, ws.msgs, &[], 0, None, small)?;
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
                    ik::attn_res(s, &src, layer.attn_query, nq, t, ws.res[0], &[], 0, pending.take(), small)?;
                    self.gates(ws, s, layer, &[0, 1, 2], &[r(0), r(1), r(2)], &[msgs(0), msgs(1), msgs(2)], t, small)?;
                    let sets: Vec<RmsSet> = (0..3).map(|i| set(ws.gated[i], layer.in_norm[i], ws.x[i])).collect();
                    ik::rms(s, &sets, t)?;
                    (ws.x[0], ws.x[1], ws.x[2])
                } else {
                    // The mixtures normalized straight into the projections' inputs.
                    let norms: Vec<(u64, u64)> = (0..nq).map(|i| (layer.in_norm[i], ws.x[i])).collect();
                    ik::attn_res(s, &src, layer.attn_query, nq, t, 0, &norms, D, pending.take(), small)?;
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
            let mut ldk = a.kg.n;
            let (mut ldq, mut lin_k, mut lin_v) = (QG, ws.lin_k, ws.lin_v);
            match &a.qkv {
                // Small steps: one launch over the three projections, each with its input (the GEMV in
                // launches of at most gemv_max_m rows).
                Some(m) if small => {
                    // Query-side rows (and the zero rows after them) with a row stride of QGP.
                    ldq = QGP;
                    if skinny(t, m, true) {
                        let segs = [(QGP, xq, ws.lin_q, QGP), (ldk, xk, ws.lin_k, ldk), (ldk, xv, ws.lin_v, ldk)];
                        ik::skinny_i8(s, t, m.w, m.scale, D, &segs, D)?;
                    } else {
                        let g = k::gemv_max_m();
                        for r0 in (0..t).step_by(g) {
                            let (xo, oq, ok) = ((r0 * D * 2) as u64, (r0 * QGP * 4) as u64, (r0 * ldk * 4) as u64);
                            let segs = [(QGP, xq + xo, ws.lin_q + oq, QGP), (ldk, xk + xo, ws.lin_k + ok, ldk), (ldk, xv + xo, ws.lin_v + ok, ldk)];
                            k::gemv_seg(s, g.min(t - r0), m.fmt, m.w, m.scale, D, &segs, D)?;
                        }
                    }
                }
                // One input (sliding layers outside the message gates): one GEMM, outputs side by side.
                Some(m) if xq == xk && xk == xv => {
                    linear(s, xq, D, t, m, ws.lin_q, m.n, small)?;
                    ldq = m.n;
                    lin_k = ws.lin_q + (QGP * 4) as u64;
                    lin_v = lin_k + (ldk * 4) as u64;
                    ldk = m.n;
                }
                _ => {
                    linear(s, xq, D, t, &a.qg, ws.lin_q, QG, small)?;
                    linear(s, xk, D, t, &a.kg, ws.lin_k, ldk, small)?;
                    linear(s, xv, D, t, &a.vg, ws.lin_v, ldk, small)?;
                }
            }
            let side = |conv: u64| {
                if layer.sliding {
                    KvSide { conv, static_w: 0 }
                } else {
                    KvSide { conv: 0, static_w: conv }
                }
            };
            ik::qkv_prep(
                s,
                ws.lin_q,
                ldq,
                a.qscale,
                a.rscale,
                cfg.qk_clip,
                cfg.rel_clip,
                !layer.sliding,
                ws.q,
                ws.rel,
                ws.sink,
                lin_k,
                lin_v,
                ldk,
                side(a.kconv),
                side(a.vconv),
                RING,
                sq,
                v,
                t,
            )?;
            let splits = if layer.sliding { SLIDING_SPLITS } else { FULL_SPLITS };
            ik::attention(
                s,
                ws.q,
                ws.rel,
                ws.sink,
                sq,
                v,
                a.proj,
                if layer.sliding { WINDOW } else { 0 },
                0,
                t,
                splits,
                MIN_KEYS,
                ws.lin_q,
                ldq,
                GATE_OFF,
                ws.part,
                ws.attn,
                small,
            )?;
            linear(s, ws.attn, QD, t, &a.o, ws.o, D, small)?;
            if let Some(g) = pending_guess.take() {
                // The guessed experts' gate/up weights into L2 between the projections and the experts: earlier
                // they compete with the projections' reads, more than three crowd out the prefetched weights
                // (decode +3% at 512 to 32K).
                ws.guess_moe.record(s)?;
                ws.guess.wait(&ws.guess_moe)?;
                let (wb, sb) = if g.fp4 { (2 * FF * D / 2, 2 * FF * D / 16) } else { (2 * FF * D, 2 * FF * D / 32) };
                ik::prefetch_experts(&ws.guess, ws.guess_e, t, guess_k(t), g.gate_up.w, wb, g.gate_up.sc, sb)?;
            }
            let boundary = li.is_multiple_of(BLOCK);
            if boundary {
                // The block ends: its sum joins the sources; the new block starts from the attention output.
                s.copy_d2d(ws.slots[nb], h, td as usize)?;
                nb += 1;
            }
            let attn_conv = ConvSrc {
                src: ws.o,
                sq,
                visit: v,
                slot: ik::T_ARING,
                ring_rows: RING,
                conv: layer.attn_conv,
                base: if boundary { 0 } else { h },
                ..Default::default()
            };
            // MoE.
            let mut src: Vec<u64> = ws.slots[..nb].to_vec();
            src.push(ws.r);
            if msg {
                ik::attn_res(s, &src, layer.mlp_query, 1, t, ws.res[0], &[], 0, Some(attn_conv), small)?;
                self.gates(ws, s, layer, &[3], &[ws.res[0]], &[msgs(3)], t, small)?;
                ik::rms_bf16(s, ws.gated[0], layer.post_norm, ws.xm, t)?;
            } else {
                ik::attn_res(s, &src, layer.mlp_query, 1, t, 0, &[(layer.post_norm, ws.xm)], D, Some(attn_conv), small)?;
            }
            // One-token steps: guess the next visit's experts beside this visit's routing (QUENCH6K_INK_GUESS=0: off).
            let next = cfg.visits().get(v + 1).map(|&(nl, _)| &self.layers[nl]).filter(|_| guess_k(t) > 0);
            if let Some(next) = next {
                ws.guess_x.record(s)?;
                ws.guess.wait(&ws.guess_x)?;
                linear(&ws.guess, ws.xm, D, t, &next.router, ws.guess_logits, EXPERTS, true)?;
                ik::route(&ws.guess, ws.guess_logits, next.bias, cfg.route_scale, t, ws.guess_e, ws.guess_w, 0, 0)?;
                ws.guess_done.record(&ws.guess)?;
                guessed = true;
            }
            linear_narrow(s, ws, ws.xm, D, t, &layer.router, ws.rlogits, EXPERTS, small)?;
            // Small steps route and order the pairs in one launch and fold the combine into the
            // convolution; larger ones group the pairs for the tensor-core GEMMs (NVFP4 layers).
            let staged = small && layer.fp4 && t >= expert_staged_min() && t <= expert_staged_max();
            let groups = if small && !staged && t >= expert_groups_min() { ws.groups } else { 0 };
            ik::route(s, ws.rlogits, layer.bias, cfg.route_scale, t, ws.sel_e, ws.sel_w, if small { ws.order } else { 0 }, groups)?;
            if !small {
                // QUENCH6K_INK_FP4_TC=1: NVFP4 layers on the FP4 tensor cores, activations quantized to NVFP4
                // per row (each token's row once, gathered into pair order; the activation rows in place) as
                // QUENCH6K_INK_FP4_TERMS terms (default 2: x ~ q1 + q2, the residual quantized at the same row
                // scale; 3 for the experts' input). Prefill +16% (2 terms) / +5% (3), but further from the FP32
                // reference than BF16 activations (gate distance 0.037 / 0.029, the short fixture fails).
                if let (Some(tb), true) = (&layer.tables, fp4_tc()) {
                    let pairs = t * TOPK;
                    ik::moe_plan(s, ws.sel_e, t, k::fp4_tile_rows(), ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.pair_row)?;
                    let (gu, dn) = (layer.gate_up, layer.down);
                    let (o, ty, tt, mt) = (ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles);
                    if fp4_terms() >= 2 {
                        // Two terms: the residual of the first quantization quantized again (same row scale).
                        // Second terms after the first ones' buffers (16-byte aligned for the GEMM's copies).
                        let at = |rows: usize, kk: usize| k::fp4_rows_bytes(rows, kk).next_multiple_of(256) as u64;
                        let (q2, g2) = (ws.act + at(t, D), ws.xs + at(pairs, D));
                        let three = fp4_terms() >= 3;
                        let (q3, g3) = if three { (q2 + at(t, D), g2 + at(pairs, D)) } else { (0, 0) };
                        k::fp4_quant_rows2(s, ws.xm, t, D, ws.act, q2, q3)?;
                        k::fp4_gather2(s, ws.act, q2, t, D, TOPK, ws.pair_row, ws.xs, g2)?;
                        if three {
                            // The third terms' codes and block scales (row scales already gathered).
                            k::fp4_gather2(s, q3, q3, t, D, TOPK, ws.pair_row, g3, g3)?;
                        }
                        k::moe_fp4_gemm2(s, ws.xs, g2, g3, pairs, tb[0].0, tb[0].1, gu.s2, o, ty, tt, mt, ws.max_tiles, ws.y, 2 * FF, D)?;
                        ik::moe_act(s, ws.y, pairs, cfg.gate_cap, cfg.up_cap, ws.act)?;
                        let a2 = ws.xs + at(pairs, FF);
                        k::fp4_quant_rows2(s, ws.act, pairs, FF, ws.xs, a2, 0)?;
                        k::moe_fp4_gemm2(s, ws.xs, a2, 0, pairs, tb[1].0, tb[1].1, dn.s2, o, ty, tt, mt, ws.max_tiles, ws.y, D, FF)?;
                    } else {
                        k::fp4_quant_rows(s, ws.xm, t, D, ws.act)?;
                        k::fp4_gather(s, ws.act, t, D, TOPK, ws.pair_row, ws.xs)?;
                        k::moe_fp4_gemm(s, ws.xs, pairs, tb[0].0, tb[0].1, gu.s2, o, ty, tt, mt, ws.max_tiles, ws.y, 2 * FF, D)?;
                        ik::moe_act(s, ws.y, pairs, cfg.gate_cap, cfg.up_cap, ws.act)?;
                        k::fp4_quant_rows(s, ws.act, pairs, FF, ws.xs)?;
                        k::moe_fp4_gemm(s, ws.xs, pairs, tb[1].0, tb[1].1, dn.s2, o, ty, tt, mt, ws.max_tiles, ws.y, D, FF)?;
                    }
                    ik::moe_combine_rows(s, ws.y, ws.pair_row, ws.sel_w, t, ws.moe)?;
                } else {
                // NVFP4 layers: tiles of 128 rows once the experts average 96 rows or more (fewer rows: more of
                // a tall tile would be padding), else 64.
                let bm = if layer.tables.is_some() && t * TOPK >= 96 * EXPERTS { 128 } else { MOE_BM };
                ik::moe_plan(s, ws.sel_e, t, bm, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.pair_row)?;
                ik::moe_gather(s, ws.xm, ws.pair_row, t, ws.xs)?;
                let gemm = |x: u64, which: usize, n: usize, kk: usize| {
                    let st = if which == 0 { layer.gate_up } else { layer.down };
                    match &layer.tables {
                        // Gate/up of NVFP4 layers: the activation from the GEMM's epilogue.
                        Some(tb) => ik::nv4_gemm(
                            s, x, tb[which].0, tb[which].1, st.s2, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.max_tiles, bm, ws.y, n, kk,
                            (which == 0).then_some((ws.act, cfg.gate_cap, cfg.up_cap)),
                        ),
                        None => ik::mx8_grouped_gemm(s, x, st, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.max_tiles, ws.y, n, kk),
                    }
                };
                gemm(ws.xs, 0, 2 * FF, D)?;
                if layer.tables.is_none() {
                    ik::moe_act(s, ws.y, t * TOPK, cfg.gate_cap, cfg.up_cap, ws.act)?;
                }
                gemm(ws.act, 1, D, FF)?;
                ik::moe_combine_rows(s, ws.y, ws.pair_row, ws.sel_w, t, ws.moe)?;
                }
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
                groups,
                staged,
                ws.act,
                ws.y,
                if small { 0 } else { ws.moe },
            )?;
            }
            if let Some(next) = next {
                // The next visit rewrites the MLP input only after the guess has read it; the guessed experts'
                // weights go into L2 after the next visit's output projection.
                pending_guess = Some(next);
                s.wait(&ws.guess_done)?;
            }
            hi ^= 1;
            let (src_x, cy, cw) = if small { (0, ws.y, ws.sel_w) } else { (ws.moe, 0, 0) };
            pending = Some(ConvSrc {
                src: src_x,
                cy,
                cw,
                sq,
                visit: v,
                slot: ik::T_MRING,
                ring_rows: RING,
                conv: layer.mlp_conv,
                base: ws.r,
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
            ik::attn_res(s, &src, self.final_query, 1, t, 0, &[(self.norm, hidden)], 4 * D, pending.take(), small)?;
        }
        flush(&mut pending, ws.h[hi])?;
        if prefetched {
            ws.join.record(&ws.side)?;
            s.wait(&ws.join)?;
        }
        if guessed {
            ws.guess_join.record(&ws.guess)?;
            s.wait(&ws.guess_join)?;
        }
        if want != Logits::None {
            let (rows, first) = if want == Logits::All { (t, 0) } else { (1, t - 1) };
            let x = hidden + (first * 4 * D * 2) as u64;
            let l = &self.lm_head;
            if small && rows > LM_ROWS_GEMV && l.fmt == WeightFormat::Fp8Block && lm_rows_on() {
                ik::lm_rows(s, x, 4 * D, rows, 0, 0, l.w, l.scale, ws.out_logits, cfg.vocab, l.n, l.k)?;
            } else {
                linear(s, x, 4 * D, rows, l, ws.out_logits, cfg.vocab, small)?;
            }
            ik::softcap(s, ws.out_logits, rows, cfg.vocab, cfg.vocab, cfg.logit_cap)?;
        }
        if let (Some(d), Some(dw)) = (&self.drafter, &ws.draft) {
            // Each sequence's features of its last rows, for its next drafts.
            ik::keep_feats(s, ws.feats, sq, t)?;
            // The drafter's context keys and values of this step's positions.
            let visits = cfg.visits().len();
            for (i, l) in d.layers.iter().enumerate() {
                ik::mix(s, ws.feats, 0, 0, d.alpha + (i * 16) as u64, l.mix_norm, dw.mixed, D, t)?;
                linear(s, dw.mixed, D, t, &l.kv_ctx, dw.kv, 2 * KVD, small)?;
                ik::ctx_kv(s, dw.kv, l.k_norm_ctx, d.theta, sq, visits + i, t)?;
            }
        }
        Ok(())
    }
}
