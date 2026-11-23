//! Hy4-flash forward pass: steps over new tokens of one sequence (prefill chunks of up to `max_t` tokens,
//! `max_t + 511 <= RING`) or of up to [`MAX_SEQS`] sequences at once (up to [`MAX_STEP`] tokens each).
//!
//! Per sequence the indexed layers keep every position's cache row (576 BF16), token key (128) and pooled
//! group key (128 per four positions), the indexed layers interleaved per position in three buffers whose
//! addresses are reserved for the sequence's capacity and backed by memory as it grows
//! ([`State::reserve`]); the sliding layers keep a ring of the last `RING` rows.
//!
//! Kernels find a row's sequence through per-row descriptors (position, sequence index) and per-sequence
//! address tables in device memory, so the CUDA graphs of small steps belong to the workspace and serve any
//! sequences. Small steps (at most 16 tokens per sequence) run every row through kernels whose arithmetic
//! does not depend on the step's other rows (GEMVs, per-head GEMVs, attention with a fixed split of each
//! row's keys), so a sequence gets the same bits alone or batched, at any width.

use std::collections::HashMap;

use anyhow::{Result, ensure};
use quench6k_cuda::hy4::{self as hk, AttnMode, SeqRows};
use quench6k_cuda::kernels as k;
use quench6k_cuda::{DeviceBuffer, PinnedBuffer, Stream, VmBuffer};

use super::*;

/// Rows of a sliding layer's ring: a prefill chunk (at most 2048 tokens) plus the window before it.
pub const RING: usize = 2560;
const MAX_SPLITS: usize = 32;
/// Small steps split each row's keys over a fixed number of blocks (sliding layers: 512 keys, indexed
/// layers: up to 2048 selected keys), merged in split order: a row's result does not depend on the step.
const SPLITS_SLIDING: usize = 16;
const SPLITS_INDEXED: usize = 32;
/// Rows per launch of the small steps' attention and per-head GEMVs (and of the GEMV for more rows).
const SMALL_ROWS: usize = 16;
/// Chunks of up to this many tokens run the routed experts per (token, expert) pair; larger ones group
/// the pairs by expert for the tensor-core GEMM.
const SMALL_T: usize = 8;
/// Tile rows of the grouped NVFP4 GEMM (moe.cu BM).
const MOE_BM: usize = 64;
/// Indexer score matrix (floats): rows of a step scored at once over its groups.
const SCORE_FLOATS: usize = 16 << 20;
/// Rows of a step whose key lists and hidden states the drafter can reuse: the last DFLY_KEEP (a
/// verification step's rows, or a prefill chunk's last).
pub const DFLY_KEEP: usize = 8;
/// Most tokens per sequence in a step of several sequences.
pub const MAX_STEP: usize = 8;
/// Most sequences in one step.
pub const MAX_SEQS: usize = 8;
/// Positions a sequence's cache mapping grows by at least (about 14 MB).
const RESERVE_STEP: usize = 1024;
/// Most CUDA graphs of steps a workspace keeps.
const MAX_GRAPHS: usize = 512;
/// Per-sequence address tables: indexed rows, token keys, pooled keys, rings, drafter buffers.
const TABS: usize = 5;
/// Canonical prefill: every prompt position is computed in chunks of CANON_CHUNK tokens aligned to
/// multiples of CANON_CHUNK from position 0 (the last chunk ends at the prompt's end), so a prompt's chunk
/// geometry depends only on its length, and a cached prefix is reused only up to such a boundary (see
/// [`State::rewind`]): a sequence's bits then depend only on its tokens, not on what was cached before.
pub const CANON_CHUNK: usize = 2048;
/// Boundary snapshots a state keeps: the latest boundary and the lowest SNAPS - 1 others.
const SNAPS: usize = 4;

/// Prefill chunks run the routed experts W4A4 (NVFP4 activations, FP4 tensor cores) when
/// QUENCH6K_HY4_W4A4 is set to 1.
fn w4a4() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QUENCH6K_HY4_W4A4").is_ok_and(|v| v == "1"))
}

fn graphs_enabled() -> bool {
    std::env::var("QUENCH6K_HY4_GRAPHS").map_or(true, |v| v != "0") && std::env::var_os("QUENCH6K_HY4_DUMP").is_none()
}

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
    fn zeroed(&mut self, bytes: usize) -> Result<u64> {
        let b = DeviceBuffer::zeroed(bytes.max(256))?;
        let a = b.addr();
        self.buffers.push(b);
        self.bytes += bytes;
        Ok(a)
    }
}

/// Byte layout of a sequence's drafter buffer: per target layer the drafter reuses, the key lists
/// ([DFLY_KEEP, sel_width] I32) and their lengths of the last step's last rows, then those rows' hidden
/// states (BF16 [DFLY_KEEP, 3072]).
struct KeepLayout {
    lists: usize,
    lens: usize,
    hidden: usize,
    bytes: usize,
}

impl KeepLayout {
    fn of(model: &Hy4Model) -> Self {
        let slots = model.dfly.as_ref().map_or(0, |f| f.index_layers.len());
        let sw = hk::sel_width();
        let lens = slots * DFLY_KEEP * sw * 4;
        let hidden = (lens + slots * DFLY_KEEP * 4).next_multiple_of(256);
        Self {
            lists: 0,
            lens,
            hidden,
            bytes: hidden + DFLY_KEEP * D * 2,
        }
    }
    /// I32 offsets of slot `k`'s key lists and lengths.
    fn ints(&self, k: usize) -> (usize, usize) {
        ((self.lists + k * DFLY_KEEP * hk::sel_width() * 4) / 4, (self.lens + k * DFLY_KEEP * 4) / 4)
    }
}

/// One sequence's caches and token history.
pub struct State {
    pub pos: usize,
    /// Most positions the sequence may reach.
    pub capacity: usize,
    pub history: Vec<i32>,
    /// Tokens of the last step.
    pub last_t: usize,
    /// Device bytes when created (the fixed buffers); [`State::device_bytes`] counts what is mapped now.
    pub device_bytes: usize,
    /// Whether the last step produced hidden states (the drafter's source).
    hidden: bool,
    /// Indexed layers' cache rows [capacity, n_indexed, 576], token keys [capacity, n_indexed, 128] and
    /// pooled keys [capacity / 4, n_indexed, 128] (BF16).
    rows: VmBuffer,
    tk: VmBuffer,
    pooled: VmBuffer,
    /// Sliding layers' rings [n_sliding, RING, 576].
    rings: DeviceBuffer,
    /// The drafter's key lists and hidden states (KeepLayout).
    keep: DeviceBuffer,
    /// Boundary snapshots: SNAPS slots of the rings' last WINDOW positions before a boundary (a multiple
    /// of CANON_CHUNK) at the frontier [SNAPS, n_sliding, WINDOW, 576]; `snap_at[i]`: slot i's boundary.
    snap: DeviceBuffer,
    snap_at: [Option<usize>; SNAPS],
    /// The canonical frontier: a multiple of CANON_CHUNK below which every position was computed by an
    /// aligned prefill chunk of CANON_CHUNK tokens (never by a smaller or unaligned step) and has not been
    /// rewritten since. Every snapshot's boundary is at most the frontier.
    canon: usize,
    /// The rings hold the last-written rows of positions [ring_lo, pos).
    ring_lo: usize,
    n_indexed: usize,
    n_sliding: usize,
}

impl State {
    /// A sequence of up to `capacity` positions (at most the model's maximum). Its full-length caches are
    /// reserved as addresses and mapped as positions are written ([`State::reserve`]); the rings and the
    /// drafter's buffers are allocated now.
    pub fn new(model: &Hy4Model, capacity: usize, s: &Stream) -> Result<Self> {
        let capacity = capacity.div_ceil(GROUP) * GROUP;
        ensure!(
            capacity <= model.cfg.max_positions.next_multiple_of(GROUP),
            "capacity {capacity} exceeds the model's {} positions",
            model.cfg.max_positions
        );
        let (ni, ns) = (model.n_indexed, model.n_sliding);
        let rows = VmBuffer::reserve(capacity * ni * KROW * 2)?;
        let tk = VmBuffer::reserve(capacity * ni * ID * 2)?;
        let pooled = VmBuffer::reserve(capacity / GROUP * ni * ID * 2)?;
        let rings = DeviceBuffer::new((ns * RING * KROW * 2).max(16))?;
        s.memset(rings.addr(), 0, rings.bytes())?;
        let keep = DeviceBuffer::new(KeepLayout::of(model).bytes)?;
        s.memset(keep.addr(), 0, keep.bytes())?;
        let snap = DeviceBuffer::new((SNAPS * ns * WINDOW * KROW * 2).max(16))?;
        let device_bytes = rings.bytes() + keep.bytes() + snap.bytes();
        Ok(Self {
            pos: 0,
            capacity,
            history: Vec::new(),
            last_t: 0,
            device_bytes,
            hidden: false,
            rows,
            tk,
            pooled,
            rings,
            keep,
            snap,
            snap_at: [None; SNAPS],
            canon: 0,
            ring_lo: 0,
            n_indexed: ni,
            n_sliding: ns,
        })
    }

    /// Device bytes per position of the full-length caches.
    pub fn bytes_per_position(model: &Hy4Model) -> usize {
        model.n_indexed * (KROW * 2 + ID * 2) + model.n_indexed * ID * 2 / GROUP
    }

    /// Device memory the sequence holds now: its fixed buffers and the mapped part of its caches.
    pub fn device_bytes(&self) -> usize {
        self.rings.bytes() + self.keep.bytes() + self.snap.bytes() + self.rows.mapped() + self.tk.mapped() + self.pooled.mapped()
    }

    /// Positions whose full-length cache memory is mapped.
    pub fn mapped_positions(&self) -> usize {
        let (ni, rb, tb) = (self.n_indexed.max(1), KROW * 2, ID * 2);
        (self.rows.mapped() / (ni * rb))
            .min(self.tk.mapped() / (ni * tb))
            .min(self.pooled.mapped() / (ni * tb) * GROUP)
            .min(self.capacity)
    }

    /// Maps memory for the full-length caches of positions < `positions` (at most the capacity; RESERVE_STEP
    /// positions ahead when that fits). Out of device memory: an error whose message contains "out of
    /// memory", and nothing changed.
    pub fn reserve(&mut self, positions: usize) -> Result<()> {
        let want = positions.min(self.capacity);
        let have = self.mapped_positions();
        if want <= have {
            return Ok(());
        }
        // Grow by at least RESERVE_STEP positions (fewer, larger mappings), or exactly when that fails.
        let ahead = want.max(have + RESERVE_STEP).min(self.capacity);
        match self.map(ahead) {
            Err(_) if ahead > want => self.map(want),
            r => r,
        }
    }

    fn map(&mut self, want: usize) -> Result<()> {
        let ni = self.n_indexed;
        let before = (self.rows.mapped(), self.tk.mapped(), self.pooled.mapped());
        let r = (|| -> Result<()> {
            self.rows.map_to(want * ni * KROW * 2)?;
            self.tk.map_to(want * ni * ID * 2)?;
            self.pooled.map_to(want.div_ceil(GROUP) * ni * ID * 2)
        })();
        if let Err(e) = r {
            // Chunks mapped by this call lie past the old ends and were never used.
            let _ = self.rows.unmap_to(before.0);
            let _ = self.tk.unmap_to(before.1);
            let _ = self.pooled.unmap_to(before.2);
            return Err(e.context(format!("caches for {want} positions")));
        }
        Ok(())
    }

    /// Unmaps the full-length cache memory beyond `positions` and forgets the cached tokens there. Waits
    /// for the device when it releases memory.
    pub fn trim(&mut self, positions: usize) {
        let positions = positions.min(self.capacity);
        if positions < self.pos {
            self.pos = positions;
            self.history.truncate(positions);
            self.last_t = 0;
            self.hidden = false;
        }
        self.lower(self.pos);
        let ni = self.n_indexed;
        let keep = (positions * ni * KROW * 2, positions * ni * ID * 2, positions.div_ceil(GROUP) * ni * ID * 2);
        if (self.rows.mapped() > keep.0 || self.tk.mapped() > keep.1 || self.pooled.mapped() > keep.2)
            && let Err(e) = quench6k_cuda::synchronize()
                .and_then(|_| self.rows.unmap_to(keep.0))
                .and_then(|_| self.tk.unmap_to(keep.1))
                .and_then(|_| self.pooled.unmap_to(keep.2))
        {
            eprintln!("hy4: releasing cache memory: {e:#}");
        }
    }

    /// Forgets the sequence (the caches are overwritten as positions are refilled).
    pub fn reset(&mut self) {
        self.pos = 0;
        self.history.clear();
        self.canon = 0;
        self.snap_at = [None; SNAPS];
        self.ring_lo = 0;
        self.last_t = 0;
        self.hidden = false;
    }

    /// Drops positions from `pos` on. The sliding layers keep the last RING rows, so `pos` must be within
    /// RING - WINDOW positions of the current one for steps to continue from it (see [`State::rewind`]).
    pub fn truncate(&mut self, pos: usize) {
        self.pos = self.pos.min(pos);
        self.history.truncate(self.pos);
        self.lower(self.pos);
    }

    /// The canonical frontier (see [`State::rewind`]).
    pub fn frontier(&self) -> usize {
        self.canon
    }

    /// Boundaries of the kept ring snapshots, ascending.
    pub fn snapshot_boundaries(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.snap_at.iter().flatten().copied().collect();
        v.sort_unstable();
        v
    }

    /// Positions from `pos` on are (about to be) rewritten: lowers the frontier to the boundary at or below
    /// `pos` and forgets the snapshots above it.
    fn lower(&mut self, pos: usize) {
        self.canon = self.canon.min(pos / CANON_CHUNK * CANON_CHUNK);
        let canon = self.canon;
        for b in &mut self.snap_at {
            if b.is_some_and(|b| b > canon) {
                *b = None;
            }
        }
    }

    /// A step wrote positions [start, end) (its tokens; the rings' rows of those positions): advances the
    /// frontier when it was an aligned chunk of CANON_CHUNK tokens right at it, and then snapshots the
    /// rings at the new boundary.
    fn wrote(&mut self, start: usize, end: usize, s: &Stream) -> Result<()> {
        self.ring_lo = self.ring_lo.min(start).max(end.saturating_sub(RING));
        if start == self.canon && start.is_multiple_of(CANON_CHUNK) && end - start == CANON_CHUNK {
            self.canon = end;
            // A free slot, else the highest boundary (the previous latest; the lowest ones stay).
            let slot = match self.snap_at.iter().position(|b| b.is_none()) {
                Some(i) => i,
                None => (0..SNAPS).max_by_key(|&i| self.snap_at[i]).unwrap(),
            };
            self.snap_at[slot] = None;
            self.copy_window(end, slot, true, s)?;
            self.snap_at[slot] = Some(end);
        }
        Ok(())
    }

    /// Keeps the longest cached prefix of `prompt` that canonical prefill allows and returns its length B
    /// (0: start over): a multiple of CANON_CHUNK at most the common prefix with the history, below the
    /// prompt's length (its last token is run again for its logits) and at most the canonical frontier,
    /// for which the sliding rings' WINDOW positions before B are still in the rings or in a boundary
    /// snapshot (restored then). The caller computes the prompt from B on in chunks of CANON_CHUNK tokens
    /// (the last one ends at the prompt's end), so every position's bits are those of a cold prefill.
    pub fn rewind(&mut self, prompt: &[i32], s: &Stream) -> Result<usize> {
        self.lower(self.pos);
        let common = self.history.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let limit = common.min(prompt.len().saturating_sub(1)).min(self.canon) / CANON_CHUNK * CANON_CHUNK;
        let mut b = limit;
        while b > 0 {
            if b <= self.pos && self.ring_lo <= b - b.min(WINDOW) {
                break;
            }
            if let Some(i) = self.snap_at.iter().position(|&x| x == Some(b)) {
                self.copy_window(b, i, false, s)?;
                self.ring_lo = b - b.min(WINDOW);
                break;
            }
            b -= CANON_CHUNK;
        }
        self.truncate(b);
        self.last_t = 0;
        self.hidden = false;
        Ok(b)
    }

    /// The ring slots of the last `n` positions before `e`: (first slot, count) runs in position order.
    fn ring_runs(e: usize, n: usize) -> Vec<(usize, usize)> {
        if n == 0 {
            return Vec::new();
        }
        let first = (e - n) % RING;
        if first + n <= RING { vec![(first, n)] } else { vec![(first, RING - first), (0, n - (RING - first))] }
    }

    /// Prefix reuse needs no prompt-end snapshot any more (boundary snapshots are taken as canonical chunks
    /// reach them); kept for the scheduler's API.
    pub fn snapshot(&mut self, _s: &Stream) -> Result<()> {
        Ok(())
    }

    /// Copies the rings' last WINDOW positions before `e` into snapshot slot `slot` (`save`), or back.
    fn copy_window(&self, e: usize, slot: usize, save: bool, s: &Stream) -> Result<()> {
        let n = e.min(WINDOW);
        let row = KROW * 2;
        for l in 0..self.n_sliding {
            let ring = self.rings.addr() + (l * RING * row) as u64;
            let mut at = self.snap.addr() + ((slot * self.n_sliding + l) * WINDOW * row) as u64;
            for (first, count) in Self::ring_runs(e, n) {
                let r = ring + (first * row) as u64;
                if save { s.copy_d2d(at, r, count * row)? } else { s.copy_d2d(r, at, count * row)? }
                at += (count * row) as u64;
            }
        }
        Ok(())
    }

    /// Checksums (quench6k_cuda::device_checksum) of the caches of positions < pos: indexed rows, token
    /// keys, pooled keys (complete groups), and the rings. For bitwise comparisons of sequences.
    pub fn checksums(&self, s: &Stream) -> Result<[u64; 4]> {
        let ni = self.n_indexed;
        let p = self.pos;
        let sum = |a: u64, bytes: usize| if bytes == 0 { Ok(0) } else { quench6k_cuda::device_checksum(a, bytes, s) };
        Ok([
            sum(self.rows.addr(), p * ni * KROW * 2)?,
            sum(self.tk.addr(), p * ni * ID * 2)?,
            sum(self.pooled.addr(), p / GROUP * ni * ID * 2)?,
            sum(self.rings.addr(), self.rings.bytes())?,
        ])
    }

    /// Device addresses of the per-sequence tables (rows, token keys, pooled keys, rings, drafter buffers).
    fn table(&self) -> [u64; TABS] {
        [self.rows.addr(), self.tk.addr(), self.pooled.addr(), self.rings.addr(), self.keep.addr()]
    }
}

/// Shapes of a step: what the enqueued work depends on besides device data.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Plan {
    /// Rows (tokens of all sequences).
    r: usize,
    /// Every sequence has at most 16 rows: row-wise kernels throughout.
    small: bool,
    /// Every sequence has at most SMALL_T rows: experts per (token, expert) pair.
    ex_small: bool,
    /// Indexer score columns, when any row selects its keys.
    groups: Option<usize>,
    /// Large steps (one sequence) without selection attend densely in the indexed layers.
    dense: bool,
    want: Logits,
}

/// Columns scored for rows whose sequences have completed up to `g` groups before their own: about 4 to 8
/// buckets per doubling (a CUDA graph per bucket).
fn group_bucket(g: usize) -> usize {
    let q = (g.next_power_of_two() / 4).max(TOPG);
    g.div_ceil(q) * q
}

/// Scratch buffers for steps of up to `max_t` tokens of one sequence, or of several sequences.
pub struct Workspace {
    pub max_t: usize,
    pub logit_rows: usize,
    /// Rows the step buffers hold: max(max_t, MAX_SEQS * MAX_STEP).
    rows_cap: usize,
    /// Row descriptors and tables: I32 pos [r], seq [r], keep slot [r], then (8-aligned) u64 [TABS, MAX_SEQS].
    desc: u64,
    emb: u64,
    lanes: u64,
    x: u64,
    post: u64,
    hc_part: u64,
    /// I32 [T] arrival counters of the fused sublayer boundaries (zero between launches).
    hc_count: u64,
    /// Floats each of f_a, f_b holds.
    f_floats: usize,
    f_a: u64, // F32 scratch [T, max(16384, 13824 ...)]
    f_b: u64,
    f_c: u64,
    bf_a: u64, // BF16 scratch
    bf_b: u64,
    qn: u64,
    qa: u64,
    olat: u64,
    y: u64,
    part: u64,
    // indexer
    iq: u64,
    hw: u64,
    scores: u64,
    picked: u64,
    sel: u64,
    sel_n: u64,
    // MoE
    logits: u64,
    sel_e: u64,
    sel_w: u64,
    rin: u64,
    xs: u64,
    eg: u64,
    eu: u64,
    act: u64,
    ed: u64,
    fin: u64,
    counts: u64,
    offsets: u64,
    cursor: u64,
    tile_expert: u64,
    tile_row: u64,
    num_tiles: u64,
    pair_row: u64,
    max_tiles: usize,
    /// NVFP4-quantized grouped rows for the W4A4 expert GEMM (fp4_rows_layout).
    fp4_rows: u64,
    // engram
    q8: u64,
    q8_sc: u64,
    mask: u64,
    eemb: u64,
    // output
    pub hidden: u64,
    pub out_logits: u64,
    pub lse: u64,
    pub target: u64,
    pub targets: u64,
    pub top_idx: u64,
    pub top_val: u64,
    pub argmax: u64,
    pub argmax_scratch: u64,
    pinned: PinnedBuffer,
    /// The drafter's buffers (when the model has one).
    dfly: Option<DflyWs>,
    /// CUDA graphs of small steps by plan, and of the drafter's pass by sequence count.
    graphs: HashMap<Plan, quench6k_cuda::Graph>,
    dgraphs: HashMap<usize, quench6k_cuda::Graph>,
    pub device_bytes: usize,
    _alloc: Alloc,
}

/// Buffers of the drafter's pass over the query blocks (`width` rows each) of up to MAX_SEQS sequences.
struct DflyWs {
    /// Staged inputs (DraftLayout).
    stage: u64,
    /// I32 [width - 1, sequences]: the drafted tokens.
    drafts: u64,
    sel: u64,
    sel_n: u64,
    /// F32 [rows, 3072] residual stream, BF16 [rows, 3072] final hidden states, BF16 [MAX_SEQS, 6144] norm pairs.
    x: u64,
    h: u64,
    cat: u64,
    /// BF16 [MAX_SEQS, 3072] corrected hidden states; F32 [MAX_SEQS, vocab] their logits.
    corr: u64,
    logits: u64,
    pinned: PinnedBuffer,
}

/// Byte offsets of the drafter's staged inputs for `n` sequences of `w`-row blocks: I32 tokens [n * w],
/// positions [n * w], sequences [n * w], first tokens [n], anchors [n], kept source rows [n], hidden rows of
/// the corrections [w - 1, n]; then (8-aligned) u64 tables [n] of cache rows, drafter buffers, and hidden
/// source rows. The drafts [w - 1, n] are read back after them.
struct DraftLayout {
    tok: usize,
    pos: usize,
    seq: usize,
    tok0: usize,
    anchor: usize,
    krow: usize,
    hidx: usize,
    rows_tab: usize,
    keep_tab: usize,
    hsrc_tab: usize,
    bytes: usize,
    out: usize,
}

impl DraftLayout {
    fn new(n: usize, w: usize) -> Self {
        let rows = n * w;
        let tok = 0;
        let pos = tok + rows * 4;
        let seq = pos + rows * 4;
        let tok0 = seq + rows * 4;
        let anchor = tok0 + n * 4;
        let krow = anchor + n * 4;
        let hidx = krow + n * 4;
        let rows_tab = (hidx + (w - 1) * n * 4).next_multiple_of(8);
        let keep_tab = rows_tab + n * 8;
        let hsrc_tab = keep_tab + n * 8;
        let bytes = hsrc_tab + n * 8;
        Self {
            tok,
            pos,
            seq,
            tok0,
            anchor,
            krow,
            hidx,
            rows_tab,
            keep_tab,
            hsrc_tab,
            bytes,
            out: bytes.next_multiple_of(16),
        }
    }
}

/// Byte offsets of a step's staged inputs for `r` rows: embedding rows, Engram rows (q8 [layers, r, 16, 96],
/// scales [layers, r, 16]), mask [r], then the descriptors (copied to `desc`): I32 pos [r], seq [r], keep
/// slot [r], (8-aligned) u64 tables [TABS, MAX_SEQS].
struct StepLayout {
    q8: usize,
    sc: usize,
    mask: usize,
    desc: usize,
    /// Within the descriptors.
    seq: usize,
    kslot: usize,
    tab: usize,
    desc_bytes: usize,
}

impl StepLayout {
    fn new(r: usize, n_eng: usize) -> Self {
        let q8 = r * D * 2;
        let sc = q8 + n_eng * r * ENGRAM_HEADS * ENGRAM_DIM;
        let mask = sc + n_eng * r * ENGRAM_HEADS * 4;
        let desc = (mask + r).next_multiple_of(16);
        let seq = r * 4;
        let kslot = seq + r * 4;
        let tab = (kslot + r * 4).next_multiple_of(8);
        Self {
            q8,
            sc,
            mask,
            desc,
            seq,
            kslot,
            tab,
            desc_bytes: tab + TABS * MAX_SEQS * 8,
        }
    }
}

pub const TOP_K: usize = 32;

impl Workspace {
    /// Steps of up to `max_t` tokens of one sequence (and of up to MAX_SEQS sequences of up to MAX_STEP
    /// tokens each); `logit_rows`: most rows of logits a step produces (1 for plain generation; the rows of
    /// a verification step, of all its sequences). `_ctx` is not used (contexts are the states' business).
    pub fn new(model: &Hy4Model, max_t: usize, _ctx: usize, logit_rows: usize) -> Result<Self> {
        ensure!(max_t + WINDOW <= RING, "chunks of {max_t} tokens exceed the sliding ring ({} at most)", RING - WINDOW);
        let mut a = Alloc::new();
        let t = max_t.max(MAX_SEQS * MAX_STEP);
        let hmax = model.cfg.heads_sliding.max(model.cfg.heads_indexed);
        let f_width = (hmax * QK).max(13824).max((HC + 1) * D);
        let pairs = t * TOPK;
        let max_tiles = pairs / MOE_BM + EXPERTS;
        let vocab = model.cfg.vocab;
        let part = hk::attn_part_floats(SMALL_ROWS, hmax, MAX_SPLITS);
        let n_eng = model.engram_layers.len();
        let lay = StepLayout::new(t, n_eng);
        let pinned_bytes = lay.desc + lay.desc_bytes;
        let ws = Self {
            max_t,
            logit_rows: logit_rows.max(16),
            rows_cap: t,
            desc: a.get(lay.desc_bytes)?,
            emb: a.get(t * D * 2)?,
            lanes: a.get(t * HC * D * 4)?,
            x: a.get(t * D * 2)?,
            post: a.get(t * HC * 4)?,
            hc_part: a.get(hk::hc_partial_floats(t) * 4)?,
            hc_count: a.zeroed(t * 4)?,
            f_floats: t * f_width,
            f_a: a.get(t * f_width * 4)?,
            f_b: a.get(t * f_width * 4)?,
            f_c: a.get(t * D * 4)?,
            bf_a: a.get(t * f_width * 2)?,
            bf_b: a.get(t * f_width * 2)?,
            qn: a.get(t * hmax * NOPE * 2)?,
            qa: a.get(t * hmax * KROW * 2)?,
            olat: a.get(t * hmax * KV_LORA * 2)?,
            y: a.get(t * D * 2)?,
            part: a.get(part * 4)?,
            iq: a.get(t * IH * ID * 2)?,
            hw: a.get(t * IH * 4)?,
            scores: a.get(SCORE_FLOATS * 4)?,
            picked: a.get(t * (TOPG - 1) * 4)?,
            sel: a.get(t * hk::sel_width() * 4)?,
            sel_n: a.get(t * 4)?,
            logits: a.get(t * EXPERTS * 4)?,
            sel_e: a.get(pairs * 4)?,
            sel_w: a.get(pairs * 4)?,
            rin: a.get(t * LAT * 2)?,
            xs: a.get(pairs * LAT * 2)?,
            eg: a.get(pairs * LAT * 4)?,
            eu: a.get(pairs * LAT * 4)?,
            act: a.get(pairs * LAT * 2)?,
            ed: a.get(pairs * LAT * 4)?,
            fin: a.get(t * LAT * 2)?,
            counts: a.get(EXPERTS * 4)?,
            offsets: a.get((EXPERTS + 1) * 4)?,
            cursor: a.get(EXPERTS * 4)?,
            tile_expert: a.get(max_tiles * 4)?,
            tile_row: a.get(max_tiles * 4)?,
            num_tiles: a.get(4)?,
            pair_row: a.get(pairs * 4)?,
            max_tiles,
            fp4_rows: a.get(k::fp4_rows_bytes(pairs, LAT))?,
            q8: a.get(n_eng.max(1) * t * ENGRAM_HEADS * ENGRAM_DIM)?,
            q8_sc: a.get(n_eng.max(1) * t * ENGRAM_HEADS * 4)?,
            mask: a.get(t)?,
            eemb: a.get(t * ENGRAM_W * 2)?,
            hidden: a.get(t * D * 2)?,
            out_logits: a.get(logit_rows.max(16) * vocab * 4)?,
            lse: a.get(t * 4)?,
            target: a.get(t * 4)?,
            targets: a.get(t * 4)?,
            top_idx: a.get(t * TOP_K * 4)?,
            top_val: a.get(t * TOP_K * 4)?,
            argmax: a.get(t * 4)?,
            argmax_scratch: a.get(k::argmax_scratch(t, vocab))?,
            pinned: PinnedBuffer::new(pinned_bytes)?,
            dfly: None,
            graphs: HashMap::new(),
            dgraphs: HashMap::new(),
            device_bytes: 0,
            _alloc: Alloc::new(),
        };
        let dfly = match &model.dfly {
            Some(f) => {
                let w = f.width;
                let rows = MAX_SEQS * w;
                let sw = hk::sel_width();
                let lay = DraftLayout::new(MAX_SEQS, w);
                Some(DflyWs {
                    stage: a.get(lay.bytes)?,
                    drafts: a.get(MAX_SEQS * w * 4)?,
                    sel: a.get(rows * sw * 4)?,
                    sel_n: a.get(rows * 4)?,
                    x: a.get(rows * D * 4)?,
                    h: a.get(rows * D * 2)?,
                    cat: a.get(MAX_SEQS * 2 * D * 2)?,
                    corr: a.get(MAX_SEQS * D * 2)?,
                    logits: a.get(MAX_SEQS * vocab * 4)?,
                    pinned: PinnedBuffer::new(lay.out + MAX_SEQS * w * 4)?,
                })
            }
            None => None,
        };
        ensure!(
            dfly.is_none() || MAX_SEQS * model.dfly.as_ref().unwrap().width <= t,
            "step buffers smaller than the drafter's blocks"
        );
        let bytes = a.bytes;
        Ok(Self {
            dfly,
            device_bytes: bytes,
            _alloc: a,
            ..ws
        })
    }

    /// Drops the captured graphs (they are captured again as steps need them).
    pub fn clear_graphs(&mut self) {
        self.graphs.clear();
        self.dgraphs.clear();
    }

    /// Device addresses of a step's descriptors for `r` rows: (pos, seq, keep slot, tables).
    fn desc_addrs(&self, r: usize) -> (u64, u64, u64, u64) {
        let lay = StepLayout::new(r, 0);
        let d = self.desc;
        (d, d + lay.seq as u64, d + lay.kslot as u64, d + lay.tab as u64)
    }
}

/// Several rows of a small step through the GEMV's batched forms (QUENCH6K_HY4_WIDE=0: launches of 16
/// rows only).
fn wide_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match WIDE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = std::env::var("QUENCH6K_HY4_WIDE").map_or(true, |v| v != "0");
            WIDE.store(on as u8, Ordering::Relaxed);
            on
        }
    }
}

static WIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// Benchmarks: turns the batched GEMV forms on or off (the same bits either way). Steps captured as
/// graphs keep the form they were captured with: clear them with [`Workspace::clear_graphs`].
pub fn set_batched_gemv(on: bool) {
    WIDE.store(on as u8, std::sync::atomic::Ordering::Relaxed);
}

/// Y [m, l.n] F32 = X [m, l.k] BF16 . W^T: the GEMV (per-row arithmetic independent of m) for small steps
/// and any m <= 16; the tensor-core GEMM otherwise. Several rows go through the GEMV's batched forms, every
/// row bitwise as alone: 5 to 8 rows four weight rows per warp with the weights read once (larger
/// matrices), more rows the wide GEMV (16 rows per warp; the largest matrices) or groups of 8 rows; small
/// matrices and single rows in launches of up to 16 rows.
#[allow(clippy::too_many_arguments)]
fn linear(s: &Stream, x: u64, ldx: usize, m: usize, l: &Linear, y: u64, ldy: usize, small: bool) -> Result<()> {
    let gm = k::gemv_max_m();
    if m <= gm || small {
        let size = l.n * l.k;
        if m > 1 && matches!(l.fmt, k::WeightFormat::Bf16 | k::WeightFormat::Fp8Block) && wide_enabled() {
            if let Some((mt, r)) = wide_form(l, m) {
                return hk::gemv_wide(s, m, mt, r, x, ldx, l.fmt, l.w, l.scale, y, ldy, l.n, l.k);
            }
            // The wide GEMV runs blocks of 16 rows: not for a last block of 8 rows or fewer.
            if size >= 12 << 20 && m > 8 && (m % 16 == 0 || m % 16 > 8) {
                return hk::gemv_wide(s, m, 16, 4, x, ldx, l.fmt, l.w, l.scale, y, ldy, l.n, l.k);
            }
            if (size >= 4 << 20 && m > 4) || m > 16 {
                return k::gemv_grouped(s, m, 8, 4, x, ldx, l.fmt, l.w, l.scale, y, ldy, l.n, l.k);
            }
        }
        for r0 in (0..m).step_by(gm) {
            let rows = gm.min(m - r0);
            k::gemv(s, rows, x + (r0 * ldx * 2) as u64, ldx, l.fmt, l.w, l.scale, y + (r0 * ldy * 4) as u64, ldy, 0, l.n, l.k, 1, 1.0, false)?;
        }
        Ok(())
    } else {
        k::gemm_f32(s, x, ldx, l.fmt, l.w, l.k, l.scale, y, ldy, m, l.n, l.k, 1.0, false)
    }
}

/// The wide GEMV's form (input rows, weight rows per warp) for 3 to 8 rows of `l`, where it beats the
/// launches the rules in [`linear`] pick (hy4_micro at 600 W, per matrix shape; every form gives each row
/// the bits of a one-row launch): None keeps those rules.
fn wide_form(l: &Linear, m: usize) -> Option<(usize, usize)> {
    if !fuse_enabled() {
        return None;
    }
    let size = l.n * l.k;
    let bf16 = l.fmt == k::WeightFormat::Bf16;
    match m {
        3..=4 if bf16 && size >= 12 << 20 => Some((4, 1)),
        3..=4 if !bf16 && size >= 8 << 20 => Some((4, if l.k >= 8192 { 1 } else { 2 })),
        5..=8 if bf16 && (3072..8192).contains(&l.k) => Some((8, if l.n >= 1024 { 1 } else { 2 })),
        5..=8 if !bf16 && l.k < 8192 && size >= 12 << 20 => Some((8, 4)),
        5..=8 if !bf16 && l.k < 8192 && size >= 8 << 20 => Some((8, 2)),
        _ => None,
    }
}

/// A sublayer's output before its norm: F32 rows [r, 3072] `a` (plus rows `b` added in BF16, when not 0),
/// and the norm's weight.
struct SubOut {
    a: u64,
    b: u64,
    norm: u64,
}

/// The small steps' fused launches: sublayer boundaries in one launch, stacked projections, the wide GEMV
/// forms of [`wide_form`] (the same bits either way). QUENCH6K_HY4_FUSE=0: separate launches.
fn fuse_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match FUSE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = std::env::var("QUENCH6K_HY4_FUSE").map_or(true, |v| v != "0") && std::env::var_os("QUENCH6K_HY4_DUMP").is_none();
            FUSE.store(on as u8, Ordering::Relaxed);
            hk::set_legacy(!on);
            on
        }
    }
}

static FUSE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// Benchmarks: turns the small steps' fused launches on or off (the same bits either way). Steps captured
/// as graphs keep the form they were captured with: clear them with [`Workspace::clear_graphs`].
pub fn set_fused(on: bool) {
    FUSE.store(on as u8, std::sync::atomic::Ordering::Relaxed);
    hk::set_legacy(!on);
}

/// act (BF16 [m, n]) = swiglu(gate(x), up(x)) without clamps: through the stacked [gate | up] matrix
/// (`stacked`, one GEMV into f_a) when given and its rows fit, otherwise gate into f_a and up into f_b.
#[allow(clippy::too_many_arguments)]
fn gate_up(ws: &Workspace, s: &Stream, x: u64, ldx: usize, m: usize, g: &Linear, u: &Linear, stacked: Option<&Linear>, act: u64, small: bool) -> Result<()> {
    let n = g.n;
    match stacked.filter(|st| fuse_enabled() && m * st.n <= ws.f_floats) {
        Some(st) => {
            linear(s, x, ldx, m, st, ws.f_a, st.n, small)?;
            hk::swiglu(s, ws.f_a, ws.f_a + (n * 4) as u64, st.n, n, 0.0, act, n, m)
        }
        None => {
            linear(s, x, ldx, m, g, ws.f_a, n, small)?;
            linear(s, x, ldx, m, u, ws.f_b, n, small)?;
            hk::swiglu(s, ws.f_a, ws.f_b, n, n, 0.0, act, n, m)
        }
    }
}

/// Which positions' logits a forward pass produces.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Logits {
    None,
    Last,
    All,
}

impl Hy4Model {
    /// Host side of the Engram layers for each sequence's tokens (at its positions pos ..), rows in sequence
    /// order: hashed rows gathered into the pinned buffer (per engram layer: q8 [r, 16, 96] then scales
    /// [r, 16]), and the mask.
    fn engram_inputs(&self, seqs: &[(&[i32], &[i32])], ws: &mut Workspace, lay: &StepLayout) {
        let r: usize = seqs.iter().map(|(_, t)| t.len()).sum();
        let pad = self.cfg.engram_pad;
        let n_layers = self.engram_layers.len();
        // Rows of every (layer, token, head), then read-ahead for all of them before any copy: the reads
        // of rows not in the page cache overlap.
        let mut rows: Vec<i64> = Vec::with_capacity(n_layers * r * ENGRAM_HEADS);
        for &layer in &self.engram_layers {
            let eng = self.layers[layer].engram.as_ref().unwrap();
            for (history, tokens) in seqs {
                let full: Vec<i64> = history.iter().chain(tokens.iter()).map(|&x| x as i64).collect();
                let off = history.len();
                for j in 0..tokens.len() {
                    let p = off + j;
                    let ngram: Vec<i64> = (0..self.cfg.engram_max_ngram)
                        .map(|o| if p >= o { full[p - o] } else { pad })
                        .collect();
                    eng.hash.rows(&ngram, &mut rows);
                }
            }
        }
        // Small steps (decode): read-ahead hints for the rows (pages hinted before get none), so cold rows
        // load concurrently; larger chunks fault rows in from the copy threads instead (a hint per row
        // costs a system call).
        let hint = r <= MAX_SEQS * MAX_STEP;
        let t0 = std::time::Instant::now();
        let buf = ws.pinned.as_mut_slice();
        let (head, rest) = buf.split_at_mut(lay.sc);
        let q8 = &mut head[lay.q8..];
        let sc = &mut rest[..n_layers * r * ENGRAM_HEADS * 4];
        let per_layer = r * ENGRAM_HEADS;
        let copy = |li: usize, idx: std::ops::Range<usize>, q8: &mut [u8], sc: &mut [u8]| {
            let eng = self.layers[self.engram_layers[li]].engram.as_ref().unwrap();
            if hint {
                for i in idx.clone() {
                    eng.prefetch(rows[li * per_layer + i] as usize);
                }
            }
            for (k, i) in idx.enumerate() {
                let (q, s) = eng.row(rows[li * per_layer + i] as usize);
                q8[k * ENGRAM_DIM..(k + 1) * ENGRAM_DIM].copy_from_slice(q);
                sc[k * 4..k * 4 + 4].copy_from_slice(&s.to_le_bytes());
            }
        };
        // Up to 16 threads over contiguous ranges of each layer's rows (hints and faults in parallel): one
        // per 64 rows of a small step, per 512 rows of a larger one.
        let threads = per_layer.div_ceil(if hint { 64 } else { 512 }).clamp(1, 16 / n_layers.max(1)).max(1);
        let span = per_layer.div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            let mut q8_rest = q8;
            let mut sc_rest = sc;
            for li in 0..n_layers {
                for b0 in (0..per_layer).step_by(span) {
                    let n = span.min(per_layer - b0);
                    let (qa, qb) = std::mem::take(&mut q8_rest).split_at_mut(n * ENGRAM_DIM);
                    let (sa, sb) = std::mem::take(&mut sc_rest).split_at_mut(n * 4);
                    q8_rest = qb;
                    sc_rest = sb;
                    if threads == 1 {
                        copy(li, b0..b0 + n, qa, sa);
                    } else {
                        scope.spawn(move || copy(li, b0..b0 + n, qa, sa));
                    }
                }
            }
        });
        if std::env::var_os("QUENCH6K_HY4_TIMING").is_some() {
            eprintln!("hy4 engram r={r}: rows {:.2} ms ({threads} threads per layer)", t0.elapsed().as_secs_f64() * 1e3);
        }
        let mut j = 0;
        for (_, tokens) in seqs {
            for &tok in tokens.iter() {
                let tok = tok as i64;
                let keep = tok != pad && Some(tok) != self.cfg.image_token && Some(tok) != self.cfg.video_token;
                buf[lay.mask + j] = keep as u8;
                j += 1;
            }
        }
    }

    /// Host inputs of a step in the pinned buffer: embedding rows, Engram rows and mask, row descriptors
    /// and the sequences' tables.
    fn stage(&self, states: &[&mut State], ws: &mut Workspace, tokens: &[&[i32]], lay: &StepLayout) -> Result<()> {
        let mut j = 0;
        for toks in tokens {
            for &tok in toks.iter() {
                ensure!(tok >= 0 && (tok as usize) < self.cfg.vocab, "token {tok} out of range");
                ws.pinned.as_mut_slice()[j * D * 2..(j + 1) * D * 2].copy_from_slice(self.embed_row(tok as usize));
                j += 1;
            }
        }
        let seqs: Vec<(&[i32], &[i32])> = states.iter().zip(tokens).map(|(st, t)| (st.history.as_slice(), *t)).collect();
        self.engram_inputs(&seqs, ws, lay);
        let buf = &mut ws.pinned.as_mut_slice()[lay.desc..lay.desc + lay.desc_bytes];
        let mut put = |at: usize, v: i32| buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let mut j = 0;
        for (si, (st, toks)) in states.iter().zip(tokens).enumerate() {
            let t = toks.len();
            let kept = t.min(DFLY_KEEP);
            for i in 0..t {
                put(j * 4, (st.pos + i) as i32);
                put(lay.seq + j * 4, si as i32);
                put(lay.kslot + j * 4, i as i32 - (t - kept) as i32);
                j += 1;
            }
        }
        for (si, st) in states.iter().enumerate() {
            for (ti, addr) in st.table().into_iter().enumerate() {
                let at = lay.tab + (ti * MAX_SEQS + si) * 8;
                buf[at..at + 8].copy_from_slice(&addr.to_le_bytes());
            }
        }
        Ok(())
    }

    /// Runs `tokens` (at positions state.pos ..) through the model, appending them to the caches.
    /// Logits (F32 [rows, vocab]) land in ws.out_logits: one row for `Last`, one per token for `All`.
    /// Steps of at most 16 tokens run as CUDA graphs captured once per shape (QUENCH6K_HY4_GRAPHS=0: eager).
    pub fn forward(&self, state: &mut State, ws: &mut Workspace, s: &Stream, tokens: &[i32], want: Logits) -> Result<()> {
        self.step(&mut [state], ws, s, &[tokens], want)
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

    /// One step for 1..=MAX_SEQS sequences, 1..=MAX_STEP tokens each; logits for every fed token land in
    /// ws.out_logits sequence by sequence, row by row. Each sequence's results (caches, hidden states,
    /// logits) are bit-identical to `forward(state_i, tokens_i, Logits::All)` run alone.
    pub fn forward_batch(&self, states: &mut [&mut State], ws: &mut Workspace, s: &Stream, tokens: &[&[i32]]) -> Result<()> {
        ensure!(
            states.len() == tokens.len() && (1..=MAX_SEQS).contains(&states.len()),
            "{} sequences, {} token lists (1 to {MAX_SEQS} sequences)",
            states.len(),
            tokens.len()
        );
        for t in tokens {
            ensure!((1..=MAX_STEP).contains(&t.len()), "{} tokens in a batched step (1 to {MAX_STEP})", t.len());
        }
        self.step(states, ws, s, tokens, Logits::All)
    }

    fn step(&self, states: &mut [&mut State], ws: &mut Workspace, s: &Stream, tokens: &[&[i32]], want: Logits) -> Result<()> {
        let nseq = states.len();
        ensure!(nseq == tokens.len() && (1..=MAX_SEQS).contains(&nseq), "{nseq} sequences");
        let r: usize = tokens.iter().map(|t| t.len()).sum();
        let tmax = tokens.iter().map(|t| t.len()).max().unwrap_or(0);
        ensure!(tokens.iter().all(|t| !t.is_empty()), "a sequence without tokens");
        let small = tmax <= hk::heads_gemv_max_t();
        ensure!(small || nseq == 1, "steps of several sequences take at most {} tokens each", hk::heads_gemv_max_t());
        ensure!(r <= ws.rows_cap && (small || r <= ws.max_t), "step of {r} tokens (max {})", ws.max_t);
        ensure!(want != Logits::Last || nseq == 1, "last-row logits of several sequences");
        ensure!(want != Logits::All || r <= ws.logit_rows, "{r} rows of logits (workspace holds {})", ws.logit_rows);
        for (st, t) in states.iter_mut().zip(tokens) {
            ensure!(st.pos + t.len() <= st.capacity, "context full ({} + {} > {})", st.pos, t.len(), st.capacity);
            st.reserve(st.pos + t.len())?;
            let pos = st.pos;
            st.lower(pos);
        }
        // The pinned buffer may still feed the previous step's uploads.
        let timing = std::env::var_os("QUENCH6K_HY4_TIMING").is_some();
        let t0 = std::time::Instant::now();
        s.synchronize()?;
        let t1 = std::time::Instant::now();
        let lay = StepLayout::new(r, self.engram_layers.len());
        self.stage(states, ws, tokens, &lay)?;
        if timing {
            eprintln!(
                "hy4 step r={r}: wait {:.1} ms, stage {:.1} ms",
                (t1 - t0).as_secs_f64() * 1e3,
                t1.elapsed().as_secs_f64() * 1e3
            );
        }
        let groups = if small {
            let qg = states.iter().zip(tokens).map(|(st, t)| (st.pos + t.len() - 1) / GROUP).max().unwrap_or(0);
            (qg >= TOPG).then(|| group_bucket(qg))
        } else {
            let end = states[0].pos + r;
            (end > TOPG * GROUP).then_some(end / GROUP)
        };
        let plan = Plan {
            r,
            small,
            ex_small: tmax <= SMALL_T,
            groups,
            dense: !small && groups.is_none(),
            want,
        };
        if small && graphs_enabled() {
            if !ws.graphs.contains_key(&plan) {
                // Shapes come and go with the sequences' widths and lengths: keep at most MAX_GRAPHS.
                if ws.graphs.len() >= MAX_GRAPHS {
                    ws.graphs.clear();
                }
                let g = quench6k_cuda::Graph::capture(s, |s| self.enqueue(ws, s, &plan))?;
                ws.graphs.insert(plan, g);
            }
            ws.graphs[&plan].launch(s)?;
        } else {
            self.enqueue(ws, s, &plan)?;
        }
        for (st, t) in states.iter_mut().zip(tokens) {
            st.wrote(st.pos, st.pos + t.len(), s)?;
            st.pos += t.len();
            st.last_t = t.len();
            st.hidden = want != Logits::None;
            st.history.extend_from_slice(t);
        }
        Ok(())
    }

    /// Drafts the tokens after `token`, the one the last step chose after its row `row` (the last accepted
    /// one; `token` itself is not in the caches yet and goes at position state.pos). Writes the query
    /// block's cache rows past state.pos (the next step overwrites them). Empty without a drafter or
    /// room in the context.
    pub fn draft(&self, state: &mut State, ws: &mut Workspace, s: &Stream, token: i32, row: usize) -> Result<Vec<i32>> {
        Ok(self.draft_batch(&mut [state], ws, s, &[token], &[row])?.pop().unwrap_or_default())
    }

    /// [`Hy4Model::draft`] for several sequences in one pass: each sequence's drafts are bit-identical to
    /// drafting it alone. `pending[i]` and `rows[i]` are sequence i's `token` and `row`.
    pub fn draft_batch(
        &self,
        states: &mut [&mut State],
        ws: &mut Workspace,
        s: &Stream,
        pending: &[i32],
        rows: &[usize],
    ) -> Result<Vec<Vec<i32>>> {
        ensure!(
            states.len() == pending.len() && states.len() == rows.len() && states.len() <= MAX_SEQS,
            "draft_batch: {} sequences, {} tokens, {} rows (at most {MAX_SEQS})",
            states.len(),
            pending.len(),
            rows.len()
        );
        let mut out = vec![Vec::new(); states.len()];
        let (Some(f), Some(_)) = (&self.dfly, &ws.dfly) else {
            return Ok(out);
        };
        let w = f.width;
        let mut live = Vec::new();
        for (i, st) in states.iter_mut().enumerate() {
            if st.pos + w > st.capacity || st.last_t == 0 || !st.hidden {
                continue;
            }
            let row = rows[i];
            ensure!(row < st.last_t, "draft source row {row} of a {}-token step", st.last_t);
            ensure!(
                row + DFLY_KEEP >= st.last_t,
                "draft source row {row} of a {}-token step: only the last {DFLY_KEEP} rows are kept",
                st.last_t
            );
            ensure!((pending[i] as usize) < self.cfg.vocab, "token {} out of range", pending[i]);
            st.reserve(st.pos + w)?;
            // The block's cache rows go to positions pos .. pos + w.
            let pos = st.pos;
            st.lower(pos);
            live.push(i);
        }
        if live.is_empty() {
            return Ok(out);
        }
        let n = live.len();
        let lay = DraftLayout::new(n, w);
        let kl = KeepLayout::of(self);
        s.synchronize()?;
        {
            let d = ws.dfly.as_mut().unwrap();
            let buf = d.pinned.as_mut_slice();
            let mut put = |at: usize, v: i32| buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
            for (si, &i) in live.iter().enumerate() {
                let st = &states[i];
                let kept = st.last_t.min(DFLY_KEEP);
                let krow = rows[i] - (st.last_t - kept);
                for j in 0..w {
                    let r = si * w + j;
                    put(lay.tok + r * 4, if j == 0 { pending[i] } else { f.mask });
                    put(lay.pos + r * 4, (st.pos + j) as i32);
                    put(lay.seq + r * 4, si as i32);
                }
                put(lay.tok0 + si * 4, pending[i]);
                put(lay.anchor + si * 4, st.pos as i32);
                put(lay.krow + si * 4, krow as i32);
                for j in 0..w - 1 {
                    put(lay.hidx + (j * n + si) * 4, (si * w + j + 1) as i32);
                }
            }
            for (si, &i) in live.iter().enumerate() {
                let st = &states[i];
                for (at, v) in [
                    (lay.rows_tab, st.rows.addr()),
                    (lay.keep_tab, st.keep.addr()),
                    (lay.hsrc_tab, st.keep.addr() + kl.hidden as u64),
                ] {
                    buf[at + si * 8..at + si * 8 + 8].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        if graphs_enabled() {
            if !ws.dgraphs.contains_key(&n) {
                let g = quench6k_cuda::Graph::capture(s, |s| self.draft_enqueue(ws, s, n))?;
                ws.dgraphs.insert(n, g);
            }
            ws.dgraphs[&n].launch(s)?;
        } else {
            self.draft_enqueue(ws, s, n)?;
        }
        s.synchronize()?;
        let d = ws.dfly.as_ref().unwrap();
        let drafts: Vec<i32> = d.pinned.as_slice()[lay.out..lay.out + (w - 1) * n * 4]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for (si, &i) in live.iter().enumerate() {
            out[i] = (0..w - 1).map(|j| drafts[j * n + si]).collect();
        }
        Ok(out)
    }

    fn draft_enqueue(&self, ws: &Workspace, s: &Stream, n: usize) -> Result<()> {
        let cfg = &self.cfg;
        let f = self.dfly.as_ref().unwrap();
        let d = ws.dfly.as_ref().unwrap();
        let w = f.width;
        let r = n * w;
        let lay = DraftLayout::new(n, w);
        let kl = KeepLayout::of(self);
        let at = |o: usize| d.stage + o as u64;
        unsafe { s.copy_h2d(d.stage, d.pinned.as_ptr(), lay.bytes)? };
        // The blocks' embeddings; each block's first fused with the target's hidden state of its source row.
        hk::dfly_embed(s, f.embed, at(lay.tok), r, d.x)?;
        hk::dfly_norm(s, f.embed, 0, D, at(lay.tok0), f.enorm, cfg.eps, d.cat, 2 * D, n)?;
        hk::dfly_norm(s, 0, at(lay.hsrc_tab), D, at(lay.krow), f.hnorm, cfg.eps, d.cat + (D * 2) as u64, 2 * D, n)?;
        linear(s, d.cat, 2 * D, n, &f.eh, ws.f_a, D, true)?;
        hk::dfly_add(s, d.x, w, 0, ws.f_a, 1, f.fuse_gate, 0, n)?;
        let (pos, seq) = (at(lay.pos), at(lay.seq));
        for (li, layer) in f.layers.iter().enumerate() {
            let target = f.kv_layers[li];
            anyhow::ensure!(!self.layers[target].attn.sliding, "drafter layer {li}: target layer {target} has no full cache");
            let cache = SeqRows::new(at(lay.rows_tab), self.cache_slot[target] * KROW, self.n_indexed * KROW, 0);
            hk::rms_rows(s, d.x, D, layer.in_norm, D, cfg.eps, ws.x, D, r)?;
            self.attn_project(ws, s, &layer.attn, cache, pos, seq, r, true)?;
            let slot = f.index_layers.iter().position(|&l| l == f.index_layers[li]).unwrap();
            let (ko, kno) = kl.ints(slot);
            hk::dfly_sel(s, at(lay.keep_tab), ko, kno, at(lay.anchor), at(lay.krow), n, w, d.sel, d.sel_n)?;
            self.attn_output(ws, s, &layer.attn, AttnMode::Selected, cache, d.sel, d.sel_n, pos, seq, r, true, true)?;
            hk::dfly_add(s, d.x, 1, ws.y, 0, 1, 0, 0, r)?;
            hk::rms_rows(s, d.x, D, layer.post_attn_norm, D, cfg.eps, ws.x, D, r)?;
            let nn = layer.gate.n;
            gate_up(ws, s, ws.x, D, r, &layer.gate, &layer.up, layer.gate_up.as_ref(), ws.bf_a, true)?;
            linear(s, ws.bf_a, nn, r, &layer.down, ws.f_c, D, true)?;
            hk::dfly_add(s, d.x, 1, 0, ws.f_c, 1, 0, 0, r)?;
        }
        hk::rms_rows(s, d.x, D, f.norm, D, cfg.eps, d.h, D, r)?;
        // The drafts in order: block row j + 1 corrected by the embedding of the token before it.
        for j in 0..w - 1 {
            let prev = if j == 0 { at(lay.tok0) } else { d.drafts + ((j - 1) * n * 4) as u64 };
            hk::dfly_norm(s, d.h, 0, D, at(lay.hidx + j * n * 4), f.c_hidden_norm, cfg.eps, d.cat, 2 * D, n)?;
            hk::dfly_norm(s, f.embed, 0, D, prev, f.c_embed_norm, cfg.eps, d.cat + (D * 2) as u64, 2 * D, n)?;
            let nn = f.c_gate.n;
            gate_up(ws, s, d.cat, 2 * D, n, &f.c_gate, &f.c_up, f.c_gate_up.as_ref(), ws.bf_a, true)?;
            linear(s, ws.bf_a, nn, n, &f.c_down, ws.f_c, D, true)?;
            hk::dfly_add(s, ws.f_c, 1, d.h + ((j + 1) * D * 2) as u64, 0, w, 0, d.corr, n)?;
            linear(s, d.corr, D, n, &self.lm_head, d.logits, cfg.vocab, true)?;
            k::argmax(s, d.logits, n, cfg.vocab, cfg.vocab, ws.argmax_scratch, d.drafts + (j * n * 4) as u64)?;
        }
        unsafe { s.copy_d2h(d.pinned.as_ptr().add(lay.out), d.drafts, (w - 1) * n * 4)? };
        Ok(())
    }

    /// The step's device work for the staged rows (`plan`).
    fn enqueue(&self, ws: &Workspace, s: &Stream, plan: &Plan) -> Result<()> {
        let cfg = &self.cfg;
        let r = plan.r;
        let n_eng = self.engram_layers.len();
        let lay = StepLayout::new(r, n_eng);
        let pin = ws.pinned.as_ptr();
        unsafe {
            s.copy_h2d(ws.emb, pin, r * D * 2)?;
            if n_eng > 0 {
                s.copy_h2d(ws.q8, pin.add(lay.q8), lay.sc - lay.q8)?;
                s.copy_h2d(ws.q8_sc, pin.add(lay.sc), lay.mask - lay.sc)?;
                s.copy_h2d(ws.mask, pin.add(lay.mask), r)?;
            }
            s.copy_h2d(ws.desc, pin.add(lay.desc), lay.desc_bytes)?;
        }
        hk::hc_expand(s, ws.emb, ws.lanes, r)?;

        // Small steps fuse each sublayer boundary (the output's norm, the lanes' post update and the next
        // pre-mix) into one launch (hc_next), the same bits; QUENCH6K_HY4_FUSE=0 runs them separately.
        let fuse = plan.small && fuse_enabled();
        // A sublayer's output not yet added to the lanes (fused steps).
        let mut pending: Option<SubOut> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(eng) = &layer.engram {
                if let Some(o) = pending.take() {
                    self.sub_post(ws, s, &o, r)?;
                }
                let li = self.engram_layers.iter().position(|&x| x == i).unwrap();
                hk::engram_embed(
                    s,
                    ws.q8 + (li * r * ENGRAM_HEADS * ENGRAM_DIM) as u64,
                    ws.q8_sc + (li * r * ENGRAM_HEADS * 4) as u64,
                    eng.head_norm,
                    r,
                    ws.eemb,
                )?;
                linear(s, ws.eemb, ENGRAM_W, r, &eng.kv, ws.f_a, eng.kv.n, plan.small)?;
                hk::engram_apply(
                    s,
                    ws.f_a,
                    eng.kv.n,
                    eng.bias,
                    eng.norm_key,
                    eng.norm_query,
                    eng.gamma,
                    eng.hc_scale,
                    ws.mask,
                    r,
                    ws.lanes,
                )?;
            }
            // Attention sublayer.
            self.sub_pre(ws, s, pending.take(), &layer.attn_hc, layer.in_norm, false, r)?;
            self.attention(ws, s, i, plan, !fuse)?;
            pending = Some(SubOut { a: ws.f_a, b: 0, norm: layer.attn.post_norm });
            if !fuse {
                hk::hc_post(s, ws.lanes, ws.post, ws.y, D, r)?;
                pending = None;
            }
            // MLP sublayer.
            self.sub_pre(ws, s, pending.take(), &layer.mlp_hc, layer.post_attn_norm, false, r)?;
            pending = Some(match &layer.mlp {
                Mlp::Dense(m) => {
                    self.dense(ws, s, m, plan, !fuse)?;
                    SubOut { a: ws.f_a, b: 0, norm: m.norm }
                }
                Mlp::Moe(m) => {
                    self.moe(ws, s, m, plan, !fuse)?;
                    SubOut { a: ws.f_c, b: ws.f_a, norm: m.norm }
                }
            });
            if !fuse {
                hk::hc_post(s, ws.lanes, ws.post, ws.y, D, r)?;
                pending = None;
            }
            if let Some(dir) = std::env::var_os("QUENCH6K_HY4_DUMP") {
                // Debugging (unfused steps): the lanes after each layer, appended per chunk (F32 [r, 4, 3072]).
                use std::io::Write;
                let v: Vec<f32> = s.download(ws.lanes, r * HC * D)?;
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                let path = std::path::Path::new(&dir).join(format!("lanes-{i:02}.bin"));
                std::fs::OpenOptions::new().create(true).append(true).open(path)?.write_all(&bytes)?;
            }
        }
        if plan.want == Logits::None {
            if let Some(o) = pending.take() {
                self.sub_post(ws, s, &o, r)?;
            }
        }
        if plan.want != Logits::None {
            self.sub_pre(ws, s, pending.take(), &self.hc_head, self.norm, true, r)?;
            if self.dfly.is_some() {
                // The last rows' hidden states, for the drafter.
                let (_, seq, kslot, tab) = ws.desc_addrs(r);
                let kl = KeepLayout::of(self);
                hk::seq_scatter(s, ws.hidden, D * 2, D * 2, 0, kslot, seq, r, tab + (4 * MAX_SEQS * 8) as u64, kl.hidden, D * 2)?;
            }
            let (rows, first) = if plan.want == Logits::All { (r, 0) } else { (1, r - 1) };
            linear(
                s,
                ws.hidden + (first * D * 2) as u64,
                D,
                rows,
                &self.lm_head,
                ws.out_logits,
                cfg.vocab,
                plan.small,
            )?;
        }
        Ok(())
    }

    /// The next sublayer's pre-mix of the lanes into ws.x and ws.post (the head's into ws.hidden), after
    /// adding `prev` (a fused step's pending sublayer output) to the lanes in the same launch.
    #[allow(clippy::too_many_arguments)]
    fn sub_pre(&self, ws: &Workspace, s: &Stream, prev: Option<SubOut>, hc: &Hc, norm_w: u64, head: bool, r: usize) -> Result<()> {
        let cfg = &self.cfg;
        let (out, mag) = if head { (ws.hidden, 1.0) } else { (ws.x, cfg.hc_mag) };
        match prev {
            Some(o) => hk::hc_next(
                s,
                o.a,
                D,
                o.b,
                D,
                o.norm,
                cfg.eps,
                ws.lanes,
                ws.post,
                (hc.fn_, hc.base, hc.scale),
                norm_w,
                cfg.hc_eps,
                mag,
                cfg.eps,
                head,
                r,
                ws.hc_part,
                ws.hc_count,
                out,
            ),
            None => hk::hc_pre(s, ws.lanes, hc.fn_, hc.base, hc.scale, norm_w, cfg.hc_eps, mag, cfg.eps, head, r, ws.hc_part, out, ws.post),
        }
    }

    /// Adds a pending sublayer output to the lanes: its norm into ws.y, then the post update.
    fn sub_post(&self, ws: &Workspace, s: &Stream, o: &SubOut, r: usize) -> Result<()> {
        let x = if o.b != 0 {
            hk::add_rows(s, o.a, D, o.b, D, D, r, ws.f_b, D)?;
            ws.f_b
        } else {
            o.a
        };
        hk::rms_rows(s, x, D, o.norm, D, self.cfg.eps, ws.y, D, r)?;
        hk::hc_post(s, ws.lanes, ws.post, ws.y, D, r)
    }

    fn attention(&self, ws: &Workspace, s: &Stream, i: usize, plan: &Plan, norm: bool) -> Result<()> {
        let cfg = &self.cfg;
        let r = plan.r;
        let a = &self.layers[i].attn;
        let slot = self.cache_slot[i];
        let (pos, seq, kslot, tab) = ws.desc_addrs(r);
        let tab_of = |k: usize| tab + (k * MAX_SEQS * 8) as u64;
        let cache = if a.sliding {
            SeqRows::new(tab_of(3), slot * RING * KROW, KROW, RING)
        } else {
            SeqRows::new(tab_of(0), slot * KROW, self.n_indexed * KROW, 0)
        };
        self.attn_project(ws, s, a, cache, pos, seq, r, plan.small)?;
        // Keys each query sees.
        let mut mode = if a.sliding { AttnMode::Sliding } else { AttnMode::Dense };
        if let Some(ix) = &a.indexer {
            let tk = SeqRows::new(tab_of(1), slot * ID, self.n_indexed * ID, 0);
            let pooled = SeqRows::new(tab_of(2), slot * ID, self.n_indexed * ID, 0);
            // Small steps: tk and hw came with q_a and kv_a (attn_project).
            let (tk_y, hw_y, ld) = match a.proj_in.as_ref().filter(|pi| plan.small && fuse_enabled() && pi.n == Q_LORA + KROW + ID + IH) {
                Some(pi) => (ws.f_c + ((Q_LORA + KROW) * 4) as u64, ws.f_c + ((Q_LORA + KROW + ID) * 4) as u64, pi.n),
                None => {
                    linear(s, ws.x, D, r, &ix.tk, ws.f_c + (r * KROW * 4) as u64, ID, plan.small)?;
                    linear(s, ws.x, D, r, &ix.hw, ws.eu, IH, plan.small)?;
                    (ws.f_c + (r * KROW * 4) as u64, ws.eu, 0)
                }
            };
            linear(s, ws.bf_a, Q_LORA, r, &ix.q, ws.eg, IH * ID, plan.small)?;
            hk::idx_prep(
                s,
                tk_y,
                if ld > 0 { ld } else { ID },
                ix.tk_norm,
                cfg.eps,
                ws.eg,
                IH * ID,
                hw_y,
                if ld > 0 { ld } else { IH },
                cfg.theta_indexed,
                pos,
                seq,
                r,
                tk,
                ws.iq,
                ws.hw,
            )?;
            hk::idx_pool(s, tk, ix.wv, ix.wg, ix.ape, ix.key_norm, cfg.eps, cfg.theta_indexed, pos, seq, r, pooled)?;
            // Large steps attend densely until the context exceeds 2048 positions, then select (scoring only
            // the complete groups); small steps always list each row's keys (densely while its context is
            // short), scoring a bucket of groups past the longest row's (the rest score -inf).
            if !plan.dense {
                if let Some(g) = plan.groups {
                    let per = (SCORE_FLOATS / g).min(r);
                    for r0 in (0..r).step_by(per) {
                        let rows = per.min(r - r0);
                        hk::idx_scores(
                            s,
                            ws.iq + (r0 * IH * ID * 2) as u64,
                            ws.hw + (r0 * IH * 4) as u64,
                            pooled,
                            pos + (r0 * 4) as u64,
                            seq + (r0 * 4) as u64,
                            rows,
                            g,
                            ws.scores,
                            g,
                        )?;
                        k::topk(s, ws.scores, rows, g, g, TOPG - 1, false, ws.picked + (r0 * (TOPG - 1) * 4) as u64, 0)?;
                    }
                }
                hk::idx_expand(s, ws.picked, TOPG - 1, pos, r, ws.sel, ws.sel_n)?;
                mode = AttnMode::Selected;
                // The drafter reuses the last rows' key lists of its target layers.
                if let Some(slot) = self.dfly.as_ref().and_then(|f| f.index_layers.iter().position(|&l| l == i)) {
                    let sw = hk::sel_width();
                    let (lists, lens) = KeepLayout::of(self).ints(slot);
                    hk::seq_scatter(s, ws.sel, sw * 4, 0, ws.sel_n, kslot, seq, r, tab_of(4), lists * 4, sw * 4)?;
                    hk::seq_scatter(s, ws.sel_n, 4, 4, 0, kslot, seq, r, tab_of(4), lens * 4, 4)?;
                }
            }
        }
        self.attn_output(ws, s, a, mode, cache, ws.sel, ws.sel_n, pos, seq, r, plan.small, norm)
    }

    /// Projections of an attention sublayer for the `t` rows of ws.x (positions and sequences `pos`, `seq`):
    /// queries with W_uk absorbed (ws.qa), the rows' cache entries (cache), q_resid (ws.bf_a), gate (ws.f_b).
    #[allow(clippy::too_many_arguments)]
    fn attn_project(&self, ws: &Workspace, s: &Stream, a: &Attn, cache: SeqRows, pos: u64, seq: u64, t: usize, small: bool) -> Result<()> {
        let cfg = &self.cfg;
        let h = a.heads;
        let theta = if a.sliding { cfg.theta_sliding } else { cfg.theta_indexed };
        // q_resid (BF16 [t, 1024]) in bf_a; q (F32 [t, h * 256]) in f_a; kv_a (F32 [t, 576]) in f_c; gate in f_b.
        // Small steps run q_a and kv_a (and the indexer's tk and hw) as one stacked GEMV into f_c (rows of
        // proj_in.n: [q_a | kv_a | tk | hw]).
        let stacked = a.proj_in.as_ref().filter(|_| small && fuse_enabled());
        let (qa_y, kv_y, ld) = match stacked {
            Some(pi) => {
                linear(s, ws.x, D, t, pi, ws.f_c, pi.n, small)?;
                (ws.f_c, ws.f_c + (Q_LORA * 4) as u64, pi.n)
            }
            None => {
                linear(s, ws.x, D, t, &a.q_a, ws.f_a, Q_LORA, small)?;
                (ws.f_a, ws.f_c, 0)
            }
        };
        hk::rms_rows(s, qa_y, if stacked.is_some() { ld } else { Q_LORA }, a.q_a_norm, Q_LORA, cfg.eps, ws.bf_a, Q_LORA, t)?;
        linear(s, ws.bf_a, Q_LORA, t, &a.q_b, ws.f_a, h * QK, small)?;
        if stacked.is_none() {
            linear(s, ws.x, D, t, &a.kv_a, ws.f_c, KROW, small)?;
        }
        linear(s, ws.x, D, t, &a.gate, ws.f_b, h * VD, small)?;
        let kv_ld = if stacked.is_some() { ld } else { KROW };
        hk::mla_prep(s, ws.f_a, h * QK, kv_y, kv_ld, a.kv_a_norm, cfg.eps, theta, h, pos, seq, cache, t, ws.qn, ws.qa)?;
        // Absorb W_uk into the queries: qa[t, h, 0..512) = W_uk[h]^T q_nope.
        if small {
            // All rows in one launch (blocks of 16 rows; a row's arithmetic does not depend on the others).
            hk::heads_gemv(s, ws.qn, h * NOPE, NOPE, a.wuk_t, KV_LORA * NOPE, KV_LORA, NOPE, h, t, 0, 0, ws.qa, h * KROW, KROW)
        } else {
            k::gemm_bf16_batched(
                s, ws.qn, h * NOPE, NOPE, a.wuk_t, NOPE, KV_LORA * NOPE, ws.qa, h * KROW, KROW, t, KV_LORA, NOPE, h,
            )
        }
    }

    /// Attention of the projected queries over the latent caches, then W_uv, the gate, o_proj and (with
    /// `norm`) the post norm: ws.y (BF16 [t, 3072]); without, o_proj's output in ws.f_a.
    #[allow(clippy::too_many_arguments)]
    fn attn_output(
        &self,
        ws: &Workspace,
        s: &Stream,
        a: &Attn,
        mode: AttnMode,
        cache: SeqRows,
        sel: u64,
        sel_n: u64,
        pos: u64,
        seq: u64,
        t: usize,
        small: bool,
        norm: bool,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let h = a.heads;
        let wuv = a.kv_b + (NOPE * KV_LORA * 2) as u64;
        if small {
            // Each row's keys in a fixed number of splits, 16 heads per block, launches of SMALL_ROWS rows.
            let splits = if a.sliding { SPLITS_SLIDING } else { SPLITS_INDEXED };
            let sw = hk::sel_width();
            for r0 in (0..t).step_by(SMALL_ROWS) {
                let rows = SMALL_ROWS.min(t - r0);
                hk::mla_attention(
                    s,
                    mode,
                    ws.qa + (r0 * h * KROW * 2) as u64,
                    h,
                    cache,
                    sel + (r0 * sw * 4) as u64,
                    sel_n + (r0 * 4) as u64,
                    a.sinks,
                    pos + (r0 * 4) as u64,
                    seq + (r0 * 4) as u64,
                    rows,
                    splits,
                    16,
                    ws.part,
                    ws.olat + (r0 * h * KV_LORA * 2) as u64,
                )?;
            }
            // Expand the output latents (W_uv) and gate them: bf_b [t, h * 256], all rows in one launch.
            hk::heads_gemv(
                s,
                ws.olat,
                h * KV_LORA,
                KV_LORA,
                wuv,
                (NOPE + VD) * KV_LORA,
                VD,
                KV_LORA,
                h,
                t,
                ws.f_b,
                h * VD,
                ws.bf_b,
                h * VD,
                VD,
            )?;
        } else {
            // Prefill: all of a token's heads (up to 64) share each key tile (variant hy4.attn_heads).
            let hpb = quench6k_cuda::variants::get("hy4.attn_heads", 16) as usize;
            hk::mla_attention(s, mode, ws.qa, h, cache, sel, sel_n, a.sinks, pos, seq, t, 1, hpb, ws.part, ws.olat)?;
            k::gemm_bf16_batched(
                s,
                ws.olat,
                h * KV_LORA,
                KV_LORA,
                wuv,
                KV_LORA,
                (NOPE + VD) * KV_LORA,
                ws.bf_b,
                h * VD,
                VD,
                t,
                VD,
                KV_LORA,
                h,
            )?;
            hk::gate_mul(s, ws.bf_b, ws.f_b, t * h * VD)?;
        }
        linear(s, ws.bf_b, h * VD, t, &a.o, ws.f_a, D, small)?;
        if norm {
            hk::rms_rows(s, ws.f_a, D, a.post_norm, D, cfg.eps, ws.y, D, t)?;
        }
        Ok(())
    }

    /// The dense MLP of ws.x; with `norm` its normed output in ws.y, otherwise its output in ws.f_a.
    fn dense(&self, ws: &Workspace, s: &Stream, m: &Dense, plan: &Plan, norm: bool) -> Result<()> {
        let (t, small) = (plan.r, plan.small);
        let n = m.gate.n;
        gate_up(ws, s, ws.x, D, t, &m.gate, &m.up, m.gate_up.as_ref().filter(|_| small), ws.bf_a, small)?;
        linear(s, ws.bf_a, n, t, &m.down, ws.f_a, D, small)?;
        if norm {
            hk::rms_rows(s, ws.f_a, D, m.norm, D, self.cfg.eps, ws.y, D, t)?;
        }
        Ok(())
    }

    /// The MoE MLP of ws.x; with `norm` its normed output in ws.y, otherwise its two parts (routed in ws.f_c,
    /// shared in ws.f_a) to be added in BF16 and normed.
    fn moe(&self, ws: &Workspace, s: &Stream, m: &Moe, plan: &Plan, norm: bool) -> Result<()> {
        let cfg = &self.cfg;
        let (t, small) = (plan.r, plan.small);
        linear(s, ws.x, D, t, &m.router, ws.logits, EXPERTS, small)?;
        hk::route(s, ws.logits, EXPERTS, m.bias, cfg.scaling, t, ws.sel_e, ws.sel_w)?;
        if std::env::var_os("QUENCH6K_HY4_EXPERT_STATS").is_some() && !graphs_enabled() {
            // Debugging (eager steps only): distinct experts of the step's rows.
            let sel: Vec<i32> = s.download(ws.sel_e, t * TOPK)?;
            let mut seen = [false; EXPERTS];
            sel.iter().for_each(|&e| seen[e as usize] = true);
            eprintln!("hy4 experts: {t} rows, {} pairs, {} distinct experts", t * TOPK, seen.iter().filter(|&&x| x).count());
        }
        // Small steps: down_lat, shared_gate and shared_up as one stacked GEMV into f_b, then the routed input
        // and the shared expert's activation in one launch.
        let stacked = m.mlp_in.as_ref().filter(|mi| small && plan.ex_small && fuse_enabled() && t * mi.n <= ws.f_floats);
        match stacked {
            Some(mi) => {
                linear(s, ws.x, D, t, mi, ws.f_b, mi.n, small)?;
                hk::lat_prep(s, ws.f_b, mi.n, t, ws.rin, ws.bf_a, LAT)?;
            }
            None => {
                linear(s, ws.x, D, t, &m.down_lat, ws.f_a, LAT, small)?;
                k::f32_to_bf16(s, ws.f_a, ws.rin, t * LAT)?;
            }
        }
        let ex = &m.experts;
        if plan.ex_small {
            // ws.pair_row (unused by small steps) holds the pairs' expert order.
            hk::experts_small(s, ws.rin, ws.sel_e, ex.gate, ex.up, ex.down, cfg.swiglu_limit, t, ws.pair_row, ws.act, ws.ed)?;
            hk::experts_combine(s, ws.ed, 0, ws.sel_w, t, ws.fin)?;
            return self.moe_tail(ws, s, m, plan, stacked.is_some(), norm);
        } else {
            let pairs = t * TOPK;
            let w4a4 = w4a4();
            s.memset(ws.counts, 0, EXPERTS * 4)?;
            hk::moe_plan(
                s,
                ws.sel_e,
                t,
                if w4a4 { k::fp4_tile_rows() } else { MOE_BM },
                ws.counts,
                ws.offsets,
                ws.cursor,
                ws.tile_expert,
                ws.tile_row,
                ws.num_tiles,
                ws.pair_row,
            )?;
            hk::moe_gather(s, ws.rin, ws.pair_row, t, ws.xs)?;
            if w4a4 {
                // NVFP4 activations on the FP4 tensor cores (per-row least-error block scales).
                let gemm4 = |table: (u64, u64), s2: u64, y: u64| {
                    k::moe_fp4_gemm(
                        s,
                        ws.fp4_rows,
                        pairs,
                        table.0,
                        table.1,
                        s2,
                        ws.offsets,
                        ws.tile_expert,
                        ws.tile_row,
                        ws.num_tiles,
                        ws.max_tiles,
                        y,
                        LAT,
                        LAT,
                    )
                };
                k::fp4_quant_rows(s, ws.xs, pairs, LAT, ws.fp4_rows)?;
                gemm4(ex.tables[0], ex.gate.s2, ws.eg)?;
                gemm4(ex.tables[1], ex.up.s2, ws.eu)?;
                hk::swiglu(s, ws.eg, ws.eu, LAT, LAT, cfg.swiglu_limit, ws.act, LAT, pairs)?;
                k::fp4_quant_rows(s, ws.act, pairs, LAT, ws.fp4_rows)?;
                gemm4(ex.tables[2], ex.down.s2, ws.ed)?;
                hk::experts_combine(s, ws.ed, ws.pair_row, ws.sel_w, t, ws.fin)?;
                return self.moe_tail(ws, s, m, plan, false, norm);
            }
            let gemm = |x: u64, table: (u64, u64), s2: u64, y: u64| {
                k::moe_nvfp4_gemm(
                    s,
                    x,
                    table.0,
                    table.1,
                    s2,
                    ws.offsets,
                    ws.tile_expert,
                    ws.tile_row,
                    ws.num_tiles,
                    ws.max_tiles,
                    y,
                    LAT,
                    LAT,
                )
            };
            gemm(ws.xs, ex.tables[0], ex.gate.s2, ws.eg)?;
            gemm(ws.xs, ex.tables[1], ex.up.s2, ws.eu)?;
            hk::swiglu(s, ws.eg, ws.eu, LAT, LAT, cfg.swiglu_limit, ws.act, LAT, pairs)?;
            gemm(ws.act, ex.tables[2], ex.down.s2, ws.ed)?;
            hk::experts_combine(s, ws.ed, ws.pair_row, ws.sel_w, t, ws.fin)?;
        }
        self.moe_tail(ws, s, m, plan, false, norm)
    }

    /// Back to the model width, plus the shared expert (its activation already in bf_a with `shared_act`),
    /// then (with `norm`) the block's norm.
    fn moe_tail(&self, ws: &Workspace, s: &Stream, m: &Moe, plan: &Plan, shared_act: bool, norm: bool) -> Result<()> {
        let cfg = &self.cfg;
        let (t, small) = (plan.r, plan.small);
        linear(s, ws.fin, LAT, t, &m.up_lat, ws.f_c, D, small)?;
        if !shared_act {
            linear(s, ws.x, D, t, &m.shared_gate, ws.f_a, LAT, small)?;
            linear(s, ws.x, D, t, &m.shared_up, ws.f_b, LAT, small)?;
            hk::swiglu(s, ws.f_a, ws.f_b, LAT, LAT, 0.0, ws.bf_a, LAT, t)?;
        }
        linear(s, ws.bf_a, LAT, t, &m.shared_down, ws.f_a, D, small)?;
        if norm {
            hk::add_rows(s, ws.f_c, D, ws.f_a, D, D, t, ws.f_b, D)?;
            hk::rms_rows(s, ws.f_b, D, m.norm, D, cfg.eps, ws.y, D, t)?;
        }
        Ok(())
    }
}
