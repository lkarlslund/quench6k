//! Nemotron-H forward pass: steps over new tokens of one sequence (prefill chunks of up to `max_t`
//! tokens) or of up to [`MAX_SEQS`] sequences at once (up to [`MAX_STEP`] tokens each).
//!
//! The residual stream is F32 (the reference keeps it in BF16, `residual_in_fp32` = false; F32 is the
//! more precise of the two): each block adds its mixer's F32 output to it, and the mixer input is the
//! RMSNorm of the stream rounded to BF16 (the matmul input precision).
//!
//! Per sequence: the attention layers' key/value cache (BF16, or FP8 with a scale per position and KV
//! head; all attention layers interleaved per position) in an address range reserved for the sequence's
//! capacity and backed by memory as it grows ([`State::reserve`]), and one block of Mamba state (per
//! Mamba layer the F32 SSM state, the convolution state and, for speculative steps, the step's records),
//! allocated up front.
//!
//! Kernels find each sequence's rows, start position and state through a small per-step descriptor
//! (`Seqs`) in device memory. Steps whose sequences have at most [`SMALL`] rows each run every projection
//! through the GEMV, whose arithmetic per row does not depend on the step's other rows, and every other
//! kernel row-wise too, so a sequence gets the same bits alone or batched; larger steps (one sequence)
//! use the tensor-core GEMM and the grouped expert GEMM.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, ensure};
use quench6k_cuda::kernels as k;
use quench6k_cuda::{DeviceBuffer, Event, Graph, PinnedBuffer, Stream, VmBuffer};

use super::*;

/// Most sequences in one step.
pub const MAX_SEQS: usize = 8;
/// Most tokens per sequence in a step of several sequences.
pub const MAX_STEP: usize = 16;
/// Steps whose sequences have at most this many rows each run row-wise kernels (batch invariant).
pub const SMALL: usize = 16;
/// Canonical prefill: prompts are computed in chunks aligned to multiples of CANON_CHUNK positions (the
/// last ends at the prompt's end), so a prompt's chunk geometry depends only on its length.
pub const CANON_CHUNK: usize = 2048;
/// Most drafts a verification step checks, and its most tokens per sequence (the pending token and the
/// drafts).
pub const MAX_DRAFTS: usize = 7;
pub const MAX_VERIFY: usize = MAX_DRAFTS + 1;
/// Rows of a verification (deferred) step the Mamba state records: until the verdict, the Mamba states
/// stay at the step's start and its rows are kept as records, of which [`NemotronModel::commit`] keeps the
/// accepted ones.
pub const REC_ROWS: usize = MAX_VERIFY;
/// Rows per pass of the grouped expert GEMM in large steps (bounds its buffers). A whole canonical chunk:
/// passes of 512 rows left about 22 rows per expert, a third of the GEMM's 64-row tiles, and took twice
/// as long.
const EX_ROWS: usize = CANON_CHUNK;
/// Positions a sequence's cache mapping grows by at least.
const RESERVE_STEP: usize = 4096;
/// Top logits per position kept by teacher-forced scoring.
pub const TOP_K: usize = 32;
/// Most CUDA graphs of steps (and of MTP passes) a workspace keeps.
const MAX_GRAPHS: usize = 256;
/// The small steps' attention grid and partials are sized for position groups rounded up to a multiple of
/// this (blocks of groups past a row's position exit at once): a step's CUDA graph serves
/// GROUP_BUCKET * ATTN_GROUP positions of growth.
const GROUP_BUCKET: usize = 4;

/// Optional parts of the step that A/B timings switch at run time ([`set_features`]; all on by default).
pub const F_GRAPHS: u32 = 1;
/// Projections of several rows in small steps through the wide GEMV forms ([`wide_form`]).
pub const F_WIDE: u32 = 2;
/// Also the matrices with 8192 columns at 5 to 8 rows.
pub const F_WIDE_K8: u32 = 4;
/// The Mamba convolution and its commit in one launch in small steps ([`nk::mamba_conv_small`]).
pub const F_CONV: u32 = 16;
/// The router by sorting ([`nk::moe_route_sort`]).
pub const F_ROUTE: u32 = 128;
/// The small steps' attention combine with a CTA per query head ([`nk::set_attn_combine`]).
pub const F_COMBINE: u32 = 256;
/// The block norms load their weights before the dependency wait ([`nk::set_rms_early`]).
pub const F_RMS: u32 = 2048;
/// The default: every part.
pub const F_ALL: u32 = F_GRAPHS | F_WIDE | F_WIDE_K8 | F_CONV | F_ROUTE | F_COMBINE | F_RMS;
static FEATURES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(F_ALL);

/// Switches the optional parts of the step (`F_*` bits; for A/B timings in one process; the CUDA graphs
/// are kept per setting).
pub fn set_features(mask: u32) {
    FEATURES.store(mask, std::sync::atomic::Ordering::Relaxed);
    nk::set_attn_combine(mask & F_COMBINE != 0);
    nk::set_rms_early(mask & F_RMS != 0);
}

fn features() -> u32 {
    FEATURES.load(std::sync::atomic::Ordering::Relaxed)
}

fn feature(bit: u32) -> bool {
    features() & bit != 0
}

/// Small steps and MTP passes run as CUDA graphs captured once per shape (QUENCH6K_NEM_GRAPHS=0: eager;
/// never while dumping or with debug waits).
fn graphs_enabled(ws: &Workspace) -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QUENCH6K_NEM_GRAPHS").map_or(true, |v| v != "0"))
        && feature(F_GRAPHS)
        && ws.dump.is_none()
        && DEBUG_SYNC.load(std::sync::atomic::Ordering::Relaxed) == 0
}

/// Position groups of the small steps' attention for rows up to position `end` - 1, bucketed.
fn group_bucket(end: usize) -> usize {
    end.div_ceil(nk::ATTN_GROUP).div_ceil(GROUP_BUCKET) * GROUP_BUCKET
}

/// Which positions' logits a forward pass produces.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Logits {
    None,
    Last,
    All,
}

/// Format of the attention cache.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kv {
    Bf16,
    /// FP8 E4M3 with a scale per (position, KV head).
    Fp8,
}

impl Kv {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "bf16" => Ok(Kv::Bf16),
            "fp8" => Ok(Kv::Fp8),
            other => anyhow::bail!("KV cache format {other:?}: bf16 or fp8"),
        }
    }

    /// Bytes of one attention layer's cache row of a position.
    pub fn row_bytes(self) -> usize {
        match self {
            Kv::Bf16 => nk::KV_ROW_BYTES_BF16,
            Kv::Fp8 => nk::KV_ROW_BYTES_FP8,
        }
    }
}

/// One sequence's caches, recurrent states and token history.
pub struct State {
    pub pos: usize,
    /// Most positions the sequence may reach.
    pub capacity: usize,
    pub history: Vec<i32>,
    /// Tokens of the last step.
    pub last_t: usize,
    pub kv_format: Kv,
    /// Attention caches [capacity, n_attn, layer row] (see `nk::KvRows`).
    kv: VmBuffer,
    /// The Mamba block (`nk::mamba_seq_bytes`).
    mamba: DeviceBuffer,
    n_attn: usize,
    /// The canonical frontier: a multiple of CANON_CHUNK below which every position was computed by an
    /// aligned prefill chunk of CANON_CHUNK tokens and has not been rewritten since.
    canon: usize,
    /// Copies of the Mamba block (and of `h_last`) at canonical boundaries (at most SNAPS: the lowest
    /// and the latest).
    snaps: Vec<(usize, DeviceBuffer)>,
    /// Whether the sequence runs the MTP layer (the model has one): its cache rows follow the main
    /// model's positions so the layer can draft.
    mtp: bool,
    /// The MTP layer has processed positions < mtp_pos (its input at position q is the main model's hidden
    /// state at q and the token at q + 1). mtp_pos is pos - 1 (then `h_last` holds the hidden state at pos -
    /// 1) or pos.
    mtp_pos: usize,
    /// The MTP input form of the main model's hidden state at pos - 1 (F32 [D]).
    h_last: DeviceBuffer,
    /// A verification step not committed yet: (its first position, its tokens).
    uncommitted: Option<(usize, usize)>,
}

/// Mamba block snapshots a state keeps (the lowest boundary, for a shared system prompt, and the
/// latest, for the next turn of the conversation).
const SNAPS: usize = 2;

impl State {
    /// A sequence of up to `capacity` positions: its attention caches are reserved as addresses and mapped
    /// as positions are written; the Mamba block is allocated (zeroed) now.
    pub fn new(model: &NemotronModel, capacity: usize, kv_format: Kv, s: &Stream) -> Result<Self> {
        let rows = model.n_attn + model.mtp.is_some() as usize;
        let kv = VmBuffer::reserve((capacity * rows * kv_format.row_bytes()).max(1))?;
        let mamba = DeviceBuffer::new(Self::state_bytes(&model.cfg))?;
        s.memset(mamba.addr(), 0, mamba.bytes())?;
        let h_last = DeviceBuffer::new(D * 4)?;
        Ok(Self {
            pos: 0,
            capacity,
            history: Vec::new(),
            last_t: 0,
            kv_format,
            kv,
            mamba,
            n_attn: rows,
            canon: 0,
            snaps: Vec::new(),
            mtp: model.mtp.is_some(),
            mtp_pos: 0,
            h_last,
            uncommitted: None,
        })
    }

    /// Attention cache bytes per position (all attention layers and the MTP layer's).
    pub fn position_bytes(cfg: &Config, kv: Kv) -> usize {
        (cfg.count(Kind::Attention) + cfg.mtp_on() as usize) * kv.row_bytes()
    }

    /// Bytes of a sequence's Mamba block (fixed, whatever its length).
    pub fn state_bytes(cfg: &Config) -> usize {
        nk::mamba_seq_bytes(cfg.count(Kind::Mamba), REC_ROWS)
    }

    /// Device memory the sequence holds now: its Mamba block and the mapped part of its caches.
    pub fn device_bytes(&self) -> usize {
        (self.mamba.bytes() + D * 4) * (1 + self.snaps.len()) + self.kv.mapped()
    }

    /// Device bytes of a state with `positions` positions: its Mamba block, the snapshots it takes on the
    /// way and the caches.
    pub fn plan_bytes(cfg: &Config, kv: Kv, positions: usize) -> usize {
        (Self::state_bytes(cfg) + D * 4) * (1 + (positions / CANON_CHUNK).min(SNAPS)) + positions * Self::position_bytes(cfg, kv)
    }

    /// The canonical frontier (see [`State::rewind`]).
    pub fn frontier(&self) -> usize {
        self.canon
    }

    /// Positions from `pos` on are (about to be) rewritten: lowers the frontier to the boundary at or below
    /// `pos` and forgets the snapshots above it.
    fn lower(&mut self, pos: usize) {
        self.canon = self.canon.min(pos / CANON_CHUNK * CANON_CHUNK);
        let canon = self.canon;
        self.snaps.retain(|(b, _)| *b <= canon);
    }

    /// Drops positions from `pos` on. The recurrent states cannot be rewound: going back resets the
    /// sequence (decode steps never feed more than they keep, so this only happens on errors).
    pub fn truncate(&mut self, pos: usize, s: &Stream) -> Result<()> {
        if pos < self.pos {
            eprintln!("nemotron: sequence truncated from {} to {pos}: starting over", self.pos);
            self.reset(s)?;
        }
        Ok(())
    }

    fn row_bytes(&self) -> usize {
        self.n_attn * self.kv_format.row_bytes()
    }

    /// Where attention layer `slot`'s rows are in the cache.
    fn kv_rows(&self, slot: usize) -> nk::KvRows {
        let fp8 = self.kv_format == Kv::Fp8;
        nk::KvRows {
            off: slot * self.kv_format.row_bytes(),
            ld: self.row_bytes(),
            fp8,
        }
    }

    /// Positions whose cache memory is mapped.
    pub fn mapped_positions(&self) -> usize {
        (self.kv.mapped() / self.row_bytes().max(1)).min(self.capacity)
    }

    /// Maps cache memory for positions < `positions` (at most the capacity; RESERVE_STEP positions ahead
    /// when that fits). Out of device memory: an error whose message contains "out of memory".
    pub fn reserve(&mut self, positions: usize) -> Result<()> {
        let want = positions.min(self.capacity);
        let have = self.mapped_positions();
        if want <= have || self.n_attn == 0 {
            return Ok(());
        }
        let ahead = want.max(have + RESERVE_STEP).min(self.capacity);
        let rb = self.row_bytes();
        match self.kv.map_to(ahead * rb) {
            Err(_) if ahead > want => self.kv.map_to(want * rb),
            r => r,
        }
        .map_err(|e| e.context(format!("attention caches for {want} positions")))
    }

    /// Unmaps the cache memory beyond `positions` (waits for the device when it releases memory). A
    /// sequence longer than that is forgotten: its recurrent states cannot be rewound.
    pub fn trim(&mut self, positions: usize, s: &Stream) -> Result<()> {
        let positions = positions.min(self.capacity);
        if positions < self.pos {
            self.reset(s)?;
        }
        self.lower(positions);
        let keep = positions * self.row_bytes();
        if self.kv.mapped() > keep {
            quench6k_cuda::synchronize()?;
            self.kv.unmap_to(keep)?;
        }
        Ok(())
    }

    /// Forgets the sequence: position 0, a zeroed Mamba block.
    pub fn reset(&mut self, s: &Stream) -> Result<()> {
        self.pos = 0;
        self.history.clear();
        self.last_t = 0;
        self.canon = 0;
        self.snaps.clear();
        self.mtp_pos = 0;
        self.uncommitted = None;
        s.memset(self.mamba.addr(), 0, self.mamba.bytes())
    }

    /// Keeps the longest cached prefix of `prompt` that canonical prefill allows and returns its length B
    /// (0: start over): a multiple of CANON_CHUNK at most the common prefix with the history, below the
    /// prompt's length (its last token is run again for its logits) and at most the canonical frontier, at
    /// which the Mamba block is either the live one (the sequence ends there) or a snapshot (restored).
    /// The caller computes the prompt from B on in chunks ending at multiples of CANON_CHUNK, so every
    /// position's bits are those of a cold prefill.
    pub fn rewind(&mut self, prompt: &[i32], s: &Stream) -> Result<usize> {
        self.lower(self.pos);
        let common = self.history.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let limit = common.min(prompt.len().saturating_sub(1)).min(self.canon) / CANON_CHUNK * CANON_CHUNK;
        let mut b = limit;
        while b > 0 {
            if b == self.pos {
                break;
            }
            if let Some((_, snap)) = self.snaps.iter().find(|(at, _)| *at == b) {
                s.copy_d2d(self.mamba.addr(), snap.addr(), self.mamba.bytes())?;
                s.copy_d2d(self.h_last.addr(), snap.addr() + self.mamba.bytes() as u64, D * 4)?;
                break;
            }
            b -= CANON_CHUNK;
        }
        if b == 0 {
            self.reset(s)?;
            return Ok(0);
        }
        self.pos = b;
        self.history.truncate(b);
        self.last_t = 0;
        // A whole chunk's step left the MTP layer one position behind, with h_last the chunk's last row.
        self.mtp_pos = b - 1;
        self.uncommitted = None;
        self.lower(b);
        Ok(b)
    }

    /// A step wrote positions [start, end): advances the frontier when it was an aligned chunk of
    /// CANON_CHUNK tokens right at it, and then snapshots the Mamba block (its pending rows committed).
    fn wrote(&mut self, model: &NemotronModel, ws: &Workspace, start: usize, end: usize, s: &Stream) -> Result<()> {
        if start != self.canon || !start.is_multiple_of(CANON_CHUNK) || end - start != CANON_CHUNK {
            return Ok(());
        }
        self.canon = end;
        // The step's descriptor (this sequence alone) is still in ws.seqs.
        for (slot, w) in model.mamba_weights().enumerate() {
            nk::mamba_flush(s, ws.seqs, 1, slot, REC_ROWS, w)?;
        }
        nk::mamba_set_ctl(s, ws.seqs, 1, 0, false)?;
        // A free slot, else the highest boundary's (the previous latest; the lowest stays).
        let buf = if self.snaps.len() < SNAPS {
            DeviceBuffer::new(self.mamba.bytes() + D * 4)?
        } else {
            let i = (0..self.snaps.len()).max_by_key(|&i| self.snaps[i].0).unwrap();
            self.snaps.swap_remove(i).1
        };
        s.copy_d2d(buf.addr(), self.mamba.addr(), self.mamba.bytes())?;
        s.copy_d2d(buf.addr() + self.mamba.bytes() as u64, self.h_last.addr(), D * 4)?;
        self.snaps.push((end, buf));
        Ok(())
    }

    /// Checksums (quench6k_cuda::device_checksum) of the caches of positions < pos and of the Mamba
    /// block: for bitwise comparisons of sequences.
    pub fn checksums(&self, s: &Stream) -> Result<[u64; 2]> {
        let kv = self.pos * self.row_bytes();
        Ok([
            if kv == 0 { 0 } else { quench6k_cuda::device_checksum(self.kv.addr(), kv, s)? },
            quench6k_cuda::device_checksum(self.mamba.addr(), self.mamba.bytes(), s)?,
        ])
    }

    /// A copy of the sequence's position, history and recurrent state (for benches that decode several
    /// times from one prefill: [`State::restore`]; the caches below `pos` are not copied, decoding does not
    /// rewrite them).
    pub fn save(&self, s: &Stream) -> Result<Saved> {
        ensure!(self.uncommitted.is_none(), "save with a verification step pending");
        let mb = self.mamba.bytes();
        let buf = DeviceBuffer::new(mb + D * 4)?;
        s.copy_d2d(buf.addr(), self.mamba.addr(), mb)?;
        s.copy_d2d(buf.addr() + mb as u64, self.h_last.addr(), D * 4)?;
        Ok(Saved { pos: self.pos, history: self.history.clone(), last_t: self.last_t, canon: self.canon, mtp_pos: self.mtp_pos, buf })
    }

    /// Back to a [`State::save`] of this sequence.
    pub fn restore(&mut self, v: &Saved, s: &Stream) -> Result<()> {
        let mb = self.mamba.bytes();
        s.copy_d2d(self.mamba.addr(), v.buf.addr(), mb)?;
        s.copy_d2d(self.h_last.addr(), v.buf.addr() + mb as u64, D * 4)?;
        self.pos = v.pos;
        self.history.clone_from(&v.history);
        self.last_t = v.last_t;
        self.canon = v.canon;
        self.mtp_pos = v.mtp_pos;
        self.uncommitted = None;
        Ok(())
    }
}

/// See [`State::save`].
pub struct Saved {
    pos: usize,
    history: Vec<i32>,
    last_t: usize,
    canon: usize,
    mtp_pos: usize,
    buf: DeviceBuffer,
}

struct Alloc {
    buffers: Vec<DeviceBuffer>,
    bytes: usize,
}

impl Alloc {
    fn get(&mut self, bytes: usize) -> Result<u64> {
        let b = DeviceBuffer::new(bytes.max(256))?;
        let a = b.addr();
        self.buffers.push(b);
        self.bytes += bytes;
        Ok(a)
    }
}

/// Scratch buffers for steps of up to `max_t` tokens of one sequence, or of several sequences.
pub struct Workspace {
    pub max_t: usize,
    pub logit_rows: usize,
    /// Rows the step buffers hold: max(max_t, MAX_SEQS * MAX_STEP).
    rows_cap: usize,
    /// The step's sequence descriptor (`nk::Seqs`).
    seqs: u64,
    /// Residual stream F32 [rows, D].
    h: u64,
    /// Normed mixer input BF16 [rows, D].
    xn: u64,
    /// Wide F32 outputs [rows, INP]: in_proj (z | xBC | dt), q/k/v, the shared expert's up projection.
    big: u64,
    /// The Mamba mixer's scratch (convolved xBC BF16, dt F32, scan output BF16).
    mamba: nk::MambaScratch,
    /// BF16 [rows, max(DI, SH)]: gated norm output, attention output, the shared expert's activation.
    act: u64,
    /// Mixer output F32 [rows, D].
    mix: u64,
    /// Router logits F32 [rows, EXPERTS] and choices I32 / weights F32 [rows, TOPK].
    logits: u64,
    sel_e: u64,
    sel_w: u64,
    /// fc1 output F32 and BF16 [rows, LAT]; the experts' combined output BF16 [rows, LAT].
    lat: u64,
    lat_bf: u64,
    ex_bf: u64,
    /// Expert pairs (at most EX_ROWS * TOPK): gathered inputs BF16 [pairs, LAT], up output / down output
    /// F32 [pairs, FF], activation BF16 [pairs, FF], plan buffers.
    xs: u64,
    ey: u64,
    eact: u64,
    counts: u64,
    offsets: u64,
    cursor: u64,
    tile_expert: u64,
    tile_row: u64,
    num_tiles: u64,
    pair_row: u64,
    max_tiles: usize,
    /// Queries BF16 [rows, H * HD].
    q: u64,
    /// Small steps' attention partial results (grown on demand).
    part: Option<DeviceBuffer>,
    /// Logits F32 [logit_rows, vocab].
    pub out_logits: u64,
    pub argmax_scratch: u64,
    pub argmax: u64,
    /// Teacher-forced scoring: targets I32, log-sum-exp and target logits F32, top-k ids and logits.
    pub targets: u64,
    pub lse: u64,
    pub target: u64,
    pub top_idx: u64,
    pub top_val: u64,
    /// The MTP layer's inputs: the main model's hidden states in the MTP's input form F32 [rows, D] (kept
    /// from a step to its commit), the next tokens' embeddings F32 [rows, D], the hidden inputs of a pass
    /// F32 [rows, D] and [enorm(embedding) | hnorm(hidden)] BF16 [rows, 2 D]; per drafting sequence the
    /// pass's last row F32 [MAX_SEQS, D] and the next draft's hidden input F32 [MAX_SEQS, D]; the pass's
    /// sequence descriptor; accepted rows I32 [MAX_SEQS].
    hid: u64,
    memb: u64,
    mhid: u64,
    cat: u64,
    dx: u64,
    dh: u64,
    mseqs: u64,
    accepted: u64,
    /// An MTP pass's row copies (`nk::copy_rows` pairs): its hidden inputs into `mhid`, with heads each
    /// sequence's last row into `dx`, then copies after the pass (`h_last`).
    mtab: u64,
    /// Embedding rows (F32) and the descriptor of a step, then those of an MTP pass, its row copies and
    /// accepted rows, staged for upload.
    pinned: PinnedBuffer,
    /// Recorded after the uploads from the MTP pass's and the accepted rows' parts of `pinned`.
    mtp_up: Event,
    acc_up: Event,
    /// CUDA graphs of small steps (by features, plan and whether the MTP hidden states are kept) and of MTP
    /// passes.
    graphs: HashMap<(u32, Plan, bool), Graph>,
    mgraphs: HashMap<MtpKey, Graph>,
    /// Debugging: hidden states after each layer appended to `lanes-NN.bin` (F32 [rows, D]) and the final
    /// norm's output to `final.bin` (QUENCH6K_NEM_DUMP or set by the caller).
    pub dump: Option<PathBuf>,
    pub device_bytes: usize,
    _alloc: Alloc,
}

const SEQS_BYTES: usize = std::mem::size_of::<nk::Seqs>();

impl Workspace {
    /// Device bytes of the buffers, in the order [`Workspace::new`] allocates them.
    fn sizes(cfg: &Config, max_t: usize, logit_rows: usize) -> (usize, [usize; 43]) {
        let cap = Self::cap(max_t);
        let lr = logit_rows.max(1);
        let vocab = cfg.vocab;
        let pairs = cap.clamp(nk::SMALL_ROWS, EX_ROWS) * TOPK;
        let max_tiles = pairs / nk::moe_prefill_tile_rows().max(1) + EXPERTS;
        (
            max_tiles,
            [
                SEQS_BYTES,
                cap * D * 4,
                cap * D * 2,
                cap * INP * 4,
                cap * CONV * 2,
                cap * MH * 4,
                cap * DI * 2,
                cap * DI.max(SH) * 2,
                cap * D * 4,
                cap * EXPERTS * 4,
                cap * TOPK * 4,
                cap * TOPK * 4,
                cap * LAT * 4,
                cap * LAT * 2,
                cap * LAT * 2,
                pairs * LAT * 2,
                pairs * FF * 4,
                pairs * FF * 2,
                EXPERTS * 4,
                (EXPERTS + 1) * 4,
                EXPERTS * 4,
                max_tiles * 4,
                max_tiles * 4,
                4,
                pairs * 4,
                cap * H * HD * 2,
                lr * vocab * 4,
                k::argmax_scratch(lr, vocab),
                // argmax, log-sum-exp and target logits in one buffer (one download for the drafts).
                3 * lr * 4,
                lr * 4,
                0,
                0,
                lr * TOP_K * 4,
                lr * TOP_K * 4,
                cap * D * 4,
                cap * D * 4,
                cap * D * 4,
                cap * D * 2 * 2,
                MAX_SEQS * D * 4,
                MAX_SEQS * D * 4,
                SEQS_BYTES,
                MAX_SEQS * 4,
                Self::tab_rows(cap) * 16,
            ],
        )
    }

    /// Rows the step buffers hold: a chunk or a batched step, plus one row per sequence for MTP passes.
    fn cap(max_t: usize) -> usize {
        max_t.max(MAX_SEQS * MAX_STEP) + MAX_SEQS
    }

    /// Entries of an MTP pass's row copy table: hidden inputs, heads' last rows, copies after the pass.
    fn tab_rows(cap: usize) -> usize {
        cap + 2 * MAX_SEQS
    }

    /// Byte offsets in the pinned buffer: a step's descriptor, an MTP pass's embeddings, its descriptor,
    /// accepted rows, the MTP pass's row copy table; total.
    fn pinned_layout(cap: usize) -> (usize, usize, usize, usize, usize, usize) {
        let seqs = cap * D * 4;
        let memb = seqs + SEQS_BYTES.next_multiple_of(256);
        let mseqs = memb + cap * D * 4;
        let acc = mseqs + SEQS_BYTES.next_multiple_of(256);
        let tab = acc + (MAX_SEQS * 4).next_multiple_of(256);
        (seqs, memb, mseqs, acc, tab, tab + Self::tab_rows(cap) * 16)
    }

    /// Device bytes a workspace for steps of `max_t` tokens and `logit_rows` logits rows takes (without
    /// the small steps' attention partials, which grow with the context).
    pub fn bytes(cfg: &Config, max_t: usize, logit_rows: usize) -> usize {
        Self::sizes(cfg, max_t, logit_rows).1.iter().sum()
    }

    pub fn new(model: &NemotronModel, max_t: usize, logit_rows: usize) -> Result<Self> {
        nk::check_layout()?;
        let (max_tiles, sz) = Self::sizes(&model.cfg, max_t, logit_rows);
        let cap = Self::cap(max_t);
        let mut a = Alloc { buffers: Vec::new(), bytes: 0 };
        let mut b = [0u64; 43];
        for (slot, &bytes) in b.iter_mut().zip(&sz) {
            *slot = a.get(bytes)?;
        }
        Ok(Self {
            max_t,
            logit_rows: logit_rows.max(1),
            rows_cap: cap,
            seqs: b[0],
            h: b[1],
            xn: b[2],
            big: b[3],
            mamba: nk::MambaScratch { xbc: b[4], dt: b[5], y: b[6] },
            act: b[7],
            mix: b[8],
            logits: b[9],
            sel_e: b[10],
            sel_w: b[11],
            lat: b[12],
            lat_bf: b[13],
            ex_bf: b[14],
            xs: b[15],
            ey: b[16],
            eact: b[17],
            counts: b[18],
            offsets: b[19],
            cursor: b[20],
            tile_expert: b[21],
            tile_row: b[22],
            num_tiles: b[23],
            pair_row: b[24],
            max_tiles,
            q: b[25],
            part: None,
            out_logits: b[26],
            argmax_scratch: b[27],
            argmax: b[28],
            targets: b[29],
            lse: b[28] + (logit_rows.max(1) * 4) as u64,
            target: b[28] + (logit_rows.max(1) * 8) as u64,
            top_idx: b[32],
            top_val: b[33],
            hid: b[34],
            memb: b[35],
            mhid: b[36],
            cat: b[37],
            dx: b[38],
            dh: b[39],
            mseqs: b[40],
            accepted: b[41],
            mtab: b[42],
            pinned: PinnedBuffer::new(Self::pinned_layout(cap).5)?,
            mtp_up: Event::new()?,
            acc_up: Event::new()?,
            graphs: HashMap::new(),
            mgraphs: HashMap::new(),
            dump: std::env::var_os("QUENCH6K_NEM_DUMP").map(PathBuf::from),
            device_bytes: a.bytes,
            _alloc: a,
        })
    }

    /// Drops the captured CUDA graphs (captured again as steps need them).
    pub fn clear_graphs(&mut self) {
        self.graphs.clear();
        self.mgraphs.clear();
    }

    /// The small steps' attention partials for `rows` rows below `groups` position groups.
    fn part(&mut self, rows: usize, groups: usize) -> Result<u64> {
        let bytes = nk::attn_part_floats(rows, groups) * 4;
        if self.part.as_ref().is_none_or(|p| p.bytes() < bytes) {
            // Room for twice the groups: fewer reallocations as sequences grow.
            let grown = nk::attn_part_floats(MAX_SEQS * MAX_STEP, groups * 2) * 4;
            // Graphs captured with the old buffer go with it.
            self.clear_graphs();
            self.part = None;
            self.part = Some(DeviceBuffer::new(grown.max(bytes).max(256))?);
        }
        Ok(self.part.as_ref().unwrap().addr())
    }
}

/// Y [m, l.n] F32 (+)= alpha X [m, l.k] BF16 . W^T: the GEMV (per-row arithmetic independent of m) for
/// small steps and any m up to the GEMV's rows, the tensor-core GEMM otherwise.
#[allow(clippy::too_many_arguments)]
fn linear(s: &Stream, x: u64, ldx: usize, m: usize, l: &Linear, y: u64, ldy: usize, small: bool, accumulate: bool) -> Result<()> {
    let gm = k::gemv_max_m();
    if m <= gm || small {
        if let Some((mt, r)) = wide_form(l, m) {
            return nk::gemv_wide(s, m, mt, r, x, ldx, l.fmt, l.w, l.scale, y, ldy, l.n, l.k, l.alpha, accumulate);
        }
        for r0 in (0..m).step_by(gm) {
            let rows = gm.min(m - r0);
            k::gemv(s, rows, x + (r0 * ldx * 2) as u64, ldx, l.fmt, l.w, l.scale, y + (r0 * ldy * 4) as u64, ldy, 0, l.n, l.k, 1, l.alpha, accumulate)?;
        }
        Ok(())
    } else {
        k::gemm_f32(s, x, ldx, l.fmt, l.w, l.k, l.scale, y, ldy, m, l.n, l.k, l.alpha, accumulate)
    }
}

/// The wide GEMV's form (input rows per warp, weight rows per warp) for `m` rows of `l`, where it beats the
/// shared GEMV's launches of up to 16 rows (each row's bits are the same either way); None keeps those.
fn wide_form(l: &Linear, m: usize) -> Option<(usize, usize)> {
    if m < 2 || !feature(F_WIDE) {
        return None;
    }
    let size = l.n * l.k;
    let bf16 = l.fmt == k::WeightFormat::Bf16;
    match m {
        2 if size >= 8 << 20 => Some((2, 2)),
        3..=4 if bf16 && size >= 12 << 20 => Some((4, 1)),
        3..=4 if !bf16 && size >= 8 << 20 => Some((4, if l.k >= 8192 { 1 } else { 2 })),
        5..=8 if bf16 && l.n >= 1024 && (3072..8192).contains(&l.k) => Some((8, 1)),
        5..=8 if !bf16 && l.k >= 8192 && !feature(F_WIDE_K8) => None,
        5..=8 if !bf16 && size >= 12 << 20 => Some((8, 4)),
        5..=8 if !bf16 && size >= 8 << 20 => Some((8, 2)),
        // (The BF16 16 x 4 form spills registers.)
        9.. if !bf16 && size >= 12 << 20 && (m.is_multiple_of(16) || m % 16 > 8) => Some((16, 4)),
        9.. if size >= 4 << 20 => Some((8, 4)),
        _ => None,
    }
}

fn append_f32(dir: &std::path::Path, name: &str, v: &[f32]) -> Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::OpenOptions::new().create(true).append(true).open(dir.join(name))?.write_all(&bytes)?;
    Ok(())
}

static DEBUG_SYNC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Debugging races: waits for the device before the launches selected by `mask` (1 Mamba mixer, 2 small
/// experts, 4 expert combine, 8 router, 16 attention prep, 32 attention, 64 relu², 128 block norms, 256
/// residual adds, 512 to_bf16).
pub fn set_debug_sync(mask: u32) {
    DEBUG_SYNC.store(mask, std::sync::atomic::Ordering::Relaxed);
}

fn dsync(s: &Stream, bit: u32) -> Result<()> {
    if DEBUG_SYNC.load(std::sync::atomic::Ordering::Relaxed) & bit != 0 {
        s.synchronize()?;
    }
    Ok(())
}

/// What the enqueued work of a step depends on besides device data (a step's CUDA graph is keyed by it).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Plan {
    r: usize,
    nseq: usize,
    small: bool,
    /// Position groups of the small steps' attention: ceil((last position + 1) / ATTN_GROUP) over rows,
    /// bucketed ([`group_bucket`]).
    groups: usize,
    /// Partials of the small steps' attention.
    part: u64,
    kv_fp8: bool,
    want: Logits,
    /// The step's sequence descriptor (ws.seqs, or ws.mseqs for an MTP pass).
    seqs: u64,
    /// A verification step: the Mamba states keep their rows as records until the commit.
    defer: bool,
}

impl NemotronModel {
    /// Runs `tokens` (at positions state.pos ..) through the model, appending them to the caches and
    /// states. Logits (F32 [rows, vocab]) land in ws.out_logits: one row for `Last`, one per token for
    /// `All`.
    pub fn forward(&self, state: &mut State, ws: &mut Workspace, s: &Stream, tokens: &[i32], want: Logits) -> Result<()> {
        self.step(&mut [state], ws, s, &[tokens], want, false)
    }

    /// Canonical prefill of `tokens` at state.pos: chunks end at multiples of [`CANON_CHUNK`] (absolute
    /// positions) and at the tokens' end; the last chunk's last logits row lands in ws.out_logits. The
    /// workspace must take chunks of CANON_CHUNK tokens.
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
    /// ws.out_logits sequence by sequence, row by row. Each sequence's results (caches, states, logits)
    /// are bit-identical to `forward(state_i, tokens_i, Logits::All)` run alone.
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
        self.step(states, ws, s, tokens, Logits::All, false)
    }

    /// A verification step: like [`NemotronModel::forward_batch`] (1..=MAX_VERIFY tokens per sequence:
    /// the pending token and its drafts), but the Mamba states keep the rows as records until
    /// [`NemotronModel::commit`], which must follow (with the same states in the same order) before any
    /// other step of these sequences. Each row's logits are bit-identical to a plain step's.
    pub fn verify(&self, states: &mut [&mut State], ws: &mut Workspace, s: &Stream, tokens: &[&[i32]]) -> Result<()> {
        ensure!(
            states.len() == tokens.len() && (1..=MAX_SEQS).contains(&states.len()),
            "{} sequences, {} token lists (1 to {MAX_SEQS} sequences)",
            states.len(),
            tokens.len()
        );
        for t in tokens {
            ensure!((1..=MAX_VERIFY).contains(&t.len()), "{} tokens in a verification step (1 to {MAX_VERIFY})", t.len());
        }
        self.step(states, ws, s, tokens, Logits::All, true)
    }

    /// Embedding rows (F32) and the sequence descriptor into the pinned buffer.
    fn stage(&self, states: &[&mut State], ws: &mut Workspace, tokens: &[&[i32]]) -> Result<()> {
        let emb_bytes = Workspace::pinned_layout(ws.rows_cap).0;
        let buf = ws.pinned.as_mut_slice();
        let mut j = 0;
        for toks in tokens {
            for &tok in toks.iter() {
                ensure!(tok >= 0 && (tok as usize) < self.cfg.vocab, "token {tok} out of range");
                let row = self.embed_row(tok as usize);
                let dst = &mut buf[j * D * 4..(j + 1) * D * 4];
                for (c, o) in row.chunks_exact(2).zip(dst.chunks_exact_mut(4)) {
                    let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                    o.copy_from_slice(&v.to_le_bytes());
                }
                j += 1;
            }
        }
        let mut d = nk::Seqs::default();
        let mut row = 0;
        for (si, (st, toks)) in states.iter().zip(tokens).enumerate() {
            d.mamba[si] = st.mamba.addr();
            d.kv[si] = st.kv.addr();
            d.start[si] = st.pos as i32;
            d.row0[si] = row as i32;
            row += toks.len();
        }
        d.row0[states.len()] = row as i32;
        d.nseq = states.len() as i32;
        buf[emb_bytes..emb_bytes + SEQS_BYTES].copy_from_slice(d.as_bytes());
        Ok(())
    }

    fn step(&self, states: &mut [&mut State], ws: &mut Workspace, s: &Stream, tokens: &[&[i32]], want: Logits, defer: bool) -> Result<()> {
        let nseq = states.len();
        ensure!(nseq == tokens.len() && (1..=MAX_SEQS).contains(&nseq), "{nseq} sequences");
        ensure!(tokens.iter().all(|t| !t.is_empty()), "a sequence without tokens");
        let r: usize = tokens.iter().map(|t| t.len()).sum();
        let tmax = tokens.iter().map(|t| t.len()).max().unwrap_or(0);
        let small = tmax <= SMALL;
        ensure!(small || nseq == 1, "steps of several sequences take at most {SMALL} tokens each");
        ensure!(r <= ws.rows_cap && (small || r <= ws.max_t), "step of {r} tokens (max {})", ws.max_t);
        ensure!(want != Logits::Last || nseq == 1, "last-row logits of several sequences");
        ensure!(want != Logits::All || r <= ws.logit_rows, "{r} rows of logits (workspace holds {})", ws.logit_rows);
        let kv_fp8 = states[0].kv_format == Kv::Fp8;
        ensure!(states.iter().all(|st| (st.kv_format == Kv::Fp8) == kv_fp8), "sequences with different KV cache formats in one step");
        for (st, t) in states.iter_mut().zip(tokens) {
            ensure!(st.uncommitted.is_none(), "a verification step of this sequence was not committed");
            ensure!(st.pos + t.len() <= st.capacity, "context full ({} + {} > {})", st.pos, t.len(), st.capacity);
            st.reserve(st.pos + t.len())?;
            let pos = st.pos;
            st.lower(pos);
        }
        let groups = states.iter().zip(tokens).map(|(st, t)| group_bucket(st.pos + t.len())).max().unwrap_or(1);
        let part = if small { ws.part(r, groups)? } else { 0 };
        // The pinned buffer may still feed the previous step's uploads.
        s.synchronize()?;
        self.stage(states, ws, tokens)?;
        let emb_bytes = Workspace::pinned_layout(ws.rows_cap).0;
        unsafe {
            s.copy_h2d(ws.h, ws.pinned.as_ptr(), r * D * 4)?;
            s.copy_h2d(ws.seqs, ws.pinned.as_ptr().add(emb_bytes), SEQS_BYTES)?;
        }
        let plan = Plan {
            r,
            nseq,
            small,
            groups,
            part,
            kv_fp8,
            want,
            seqs: ws.seqs,
            defer,
        };
        let kv: Vec<nk::KvRows> = (0..self.n_attn).map(|slot| states[0].kv_rows(slot)).collect();
        let mtp = states.iter().any(|st| st.mtp);
        if small && graphs_enabled(ws) {
            let key = (features(), plan, mtp);
            if !ws.graphs.contains_key(&key) {
                if ws.graphs.len() >= MAX_GRAPHS {
                    ws.graphs.clear();
                }
                let g = Graph::capture(s, |s| self.step_body(ws, s, &plan, &kv, mtp))?;
                ws.graphs.insert(key, g);
            }
            ws.graphs[&key].launch(s)?;
        } else {
            self.step_body(ws, s, &plan, &kv, mtp)?;
        }
        if mtp && !defer {
            self.mtp_catch_up(states, ws, s, tokens)?;
        }
        for (st, t) in states.iter_mut().zip(tokens) {
            if defer {
                st.uncommitted = Some((st.pos, t.len()));
            } else if nseq == 1 {
                st.wrote(self, ws, st.pos, st.pos + t.len(), s)?;
            }
            st.pos += t.len();
            st.last_t = t.len();
            st.history.extend_from_slice(t);
        }
        Ok(())
    }

    /// A step's work on the device after its uploads: the layers and logits, the Mamba states' control
    /// (plain steps) and the MTP layer's hidden inputs.
    fn step_body(&self, ws: &Workspace, s: &Stream, plan: &Plan, kv: &[nk::KvRows], mtp: bool) -> Result<()> {
        self.enqueue(ws, s, plan, kv)?;
        if !plan.defer {
            // Nothing of this step is pending in the Mamba states.
            nk::mamba_set_ctl(s, plan.seqs, plan.nseq, 0, false)?;
        }
        if mtp {
            self.mtp_hidden(ws, s, plan.r)?;
        }
        Ok(())
    }

    fn enqueue(&self, ws: &Workspace, s: &Stream, plan: &Plan, kv: &[nk::KvRows]) -> Result<()> {
        let (r, small, want) = (plan.r, plan.small, plan.want);
        let eps = self.cfg.eps;
        for (i, layer) in self.layers.iter().enumerate() {
            dsync(s, 128)?;
            nk::rms(s, ws.h, D, layer.norm, eps, r, ws.xn, D, false)?;
            match &layer.mixer {
                Mixer::Mamba(m) => self.mamba(ws, s, m, self.slot[i], plan)?,
                Mixer::Attn(a) => self.attention(ws, s, a, kv[self.slot[i]], plan)?,
                Mixer::Moe(m) => self.moe(ws, s, m, r, small)?,
            }
            dsync(s, 256)?;
            k::add_f32(s, ws.h, ws.mix, r * D)?;
            if let Some(dir) = &ws.dump {
                append_f32(dir, &format!("lanes-{i:02}.bin"), &s.download::<f32>(ws.h, r * D)?)?;
            }
        }
        // Final norm of the rows whose logits are wanted (of every row when dumping).
        let xn = match (&ws.dump, want) {
            (Some(dir), _) => {
                nk::rms(s, ws.h, D, self.norm_f, eps, r, ws.xn, D, false)?;
                let b: Vec<u16> = s.download(ws.xn, r * D)?;
                let v: Vec<f32> = b.iter().map(|&x| f32::from_bits((x as u32) << 16)).collect();
                append_f32(dir, "final.bin", &v)?;
                if want == Logits::Last { ws.xn + ((r - 1) * D * 2) as u64 } else { ws.xn }
            }
            (None, Logits::None) => return Ok(()),
            (None, Logits::Last) => {
                nk::rms(s, ws.h + ((r - 1) * D * 4) as u64, D, self.norm_f, eps, 1, ws.xn, D, false)?;
                ws.xn
            }
            (None, Logits::All) => {
                nk::rms(s, ws.h, D, self.norm_f, eps, r, ws.xn, D, false)?;
                ws.xn
            }
        };
        let n = match want {
            Logits::None => return Ok(()),
            Logits::Last => 1,
            Logits::All => r,
        };
        linear(s, xn, D, n, &self.lm_head, ws.out_logits, self.cfg.vocab, small, false)
    }

    /// Mamba-2 mixer: in_proj -> (z, xBC, dt); causal convolution + SiLU, selective scan, gated RMSNorm;
    /// out_proj into ws.mix.
    fn mamba(&self, ws: &Workspace, s: &Stream, m: &Mamba, slot: usize, plan: &Plan) -> Result<()> {
        let r = plan.r;
        linear(s, ws.xn, D, r, &m.in_proj, ws.big, INP, plan.small, false)?;
        dsync(s, 1)?;
        if plan.small && feature(F_CONV) {
            nk::mamba_mixer_small(s, ws.big, INP, r, plan.seqs, plan.nseq, slot, REC_ROWS, &m.w, plan.defer, &ws.mamba, ws.act)?;
        } else {
            nk::mamba_mixer(s, ws.big, INP, r, plan.seqs, plan.nseq, slot, REC_ROWS, &m.w, plan.defer, &ws.mamba, ws.act)?;
        }
        linear(s, ws.act, DI, r, &m.out_proj, ws.mix, D, plan.small, false)
    }

    /// Attention mixer: q/k/v; keys and values into the caches; causal GQA attention; o_proj into ws.mix.
    fn attention(&self, ws: &Workspace, s: &Stream, a: &Attn, kv: nk::KvRows, plan: &Plan) -> Result<()> {
        let r = plan.r;
        debug_assert_eq!(kv.fp8, plan.kv_fp8);
        linear(s, ws.xn, D, r, &a.qkv, ws.big, QKV, plan.small, false)?;
        dsync(s, 16)?;
        nk::attn_prep(s, ws.big, QKV, 0, H * HD, (H + KVH) * HD, r, plan.seqs, kv, ws.q)?;
        dsync(s, 32)?;
        if plan.small {
            nk::attention_small(s, ws.q, r, plan.seqs, kv, plan.groups, plan.part, ws.act)?;
        } else {
            nk::attention_prefill(s, ws.q, r, plan.seqs, kv, ws.act)?;
        }
        linear(s, ws.act, H * HD, r, &a.o, ws.mix, D, plan.small, false)
    }

    /// MoE mixer: router; routed experts in the latent space (fc1 -> experts -> combine -> fc2); the
    /// shared expert on the full width, added into ws.mix.
    fn moe(&self, ws: &Workspace, s: &Stream, m: &Moe, r: usize, small: bool) -> Result<()> {
        linear(s, ws.xn, D, r, &m.gate, ws.logits, EXPERTS, small, false)?;
        dsync(s, 8)?;
        if feature(F_ROUTE) {
            nk::moe_route_sort(s, ws.logits, EXPERTS, m.bias, self.cfg.routed_scaling, r, ws.sel_e, ws.sel_w)?;
        } else {
            nk::moe_route(s, ws.logits, EXPERTS, m.bias, self.cfg.routed_scaling, r, ws.sel_e, ws.sel_w)?;
        }
        linear(s, ws.xn, D, r, &m.fc1, ws.lat, LAT, small, false)?;
        dsync(s, 512)?;
        quench6k_cuda::hy4::to_bf16(s, ws.lat, ws.lat_bf, r * LAT)?;
        let ex = &m.experts;
        let at = |base: u64, row: usize, width: usize| base + (row * width) as u64;
        if small {
            // Per (row, expert) pair, in launches of SMALL_ROWS rows (row-wise: batch invariant).
            for r0 in (0..r).step_by(nk::SMALL_ROWS) {
                let n = nk::SMALL_ROWS.min(r - r0);
                let sel_e = at(ws.sel_e, r0, TOPK * 4);
                dsync(s, 2)?;
                nk::moe_experts_small(s, at(ws.lat_bf, r0, LAT * 2), sel_e, &ex.up, &ex.down, n, ws.pair_row, ws.eact, ws.ey)?;
                dsync(s, 4)?;
                nk::moe_combine(s, ws.ey, 0, at(ws.sel_w, r0, TOPK * 4), n, at(ws.ex_bf, r0, LAT * 2))?;
            }
        } else {
            // Pairs grouped by expert into the prefill expert kernel's tiles, EX_ROWS rows at a time.
            let tile = nk::moe_prefill_tile_rows();
            for r0 in (0..r).step_by(EX_ROWS) {
                let n = EX_ROWS.min(r - r0);
                let sel_e = at(ws.sel_e, r0, TOPK * 4);
                s.memset(ws.counts, 0, EXPERTS * 4)?;
                nk::moe_plan(s, sel_e, n, tile, ws.counts, ws.offsets, ws.cursor, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.pair_row)?;
                nk::moe_gather(s, at(ws.lat_bf, r0, LAT * 2), ws.pair_row, n, ws.xs)?;
                // relu²(up) BF16 in ws.eact, bf16(down) BF16 in ws.ey.
                nk::moe_experts_prefill(s, ws.xs, &ex.up, &ex.down, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.max_tiles, ws.eact, ws.ey)?;
                nk::moe_combine_bf16(s, ws.ey, ws.pair_row, at(ws.sel_w, r0, TOPK * 4), n, at(ws.ex_bf, r0, LAT * 2))?;
            }
        }
        linear(s, ws.ex_bf, LAT, r, &m.fc2, ws.mix, D, small, false)?;
        linear(s, ws.xn, D, r, &m.shared_up, ws.big, SH, small, false)?;
        dsync(s, 64)?;
        nk::relu2(s, ws.big, SH, SH, r, ws.act, SH)?;
        linear(s, ws.act, SH, r, &m.shared_down, ws.mix, D, small, true)
    }
}

impl NemotronModel {
    /// The Mamba layers' parameters in Mamba-layer order.
    fn mamba_weights(&self) -> impl Iterator<Item = &nk::MambaW> {
        self.layers.iter().filter_map(|l| match &l.mixer {
            Mixer::Mamba(m) => Some(&m.w),
            _ => None,
        })
    }
}

/// What the MTP layer is fed as the main model's hidden state: the last layer's output (`pre`) or the
/// final norm's (`post`, F32) (QUENCH6K_NEM_MTP_H; default post).
fn mtp_post() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QUENCH6K_NEM_MTP_H").map_or(true, |v| v != "pre"))
}

/// One sequence's rows of an MTP pass: positions start.., each row's hidden input (F32 rows: contiguous
/// runs of (address, rows)) and the token after its position.
struct MtpSeq {
    st: usize,
    start: usize,
    hid: Vec<(u64, usize)>,
    next: Vec<i32>,
}

/// An MTP pass's CUDA graph key: the features, its plan and whether it computes the heads.
type MtpKey = (u32, Plan, bool);

/// Drafts of a sequence and the MTP layer's stated probability of each.
#[derive(Clone, Debug, Default)]
pub struct Drafts {
    pub tokens: Vec<i32>,
    pub conf: Vec<f32>,
}

impl NemotronModel {
    /// The main model's hidden states of the step's `r` rows in the MTP layer's input form into ws.hid.
    fn mtp_hidden(&self, ws: &Workspace, s: &Stream, r: usize) -> Result<()> {
        if mtp_post() {
            nk::rms(s, ws.h, D, self.norm_f, self.cfg.eps, r, ws.hid, D, true)
        } else {
            s.copy_d2d(ws.hid, ws.h, r * D * 4)
        }
    }

    /// After a plain step (prefill chunk or decode): the MTP layer over the step's positions whose next
    /// token is known (from mtp_pos to the step's last but one), then h_last = the step's last row.
    fn mtp_catch_up(&self, states: &mut [&mut State], ws: &mut Workspace, s: &Stream, tokens: &[&[i32]]) -> Result<()> {
        let mut seqs = Vec::new();
        let mut after = Vec::new();
        let mut off = 0;
        for (i, (st, t)) in states.iter().zip(tokens).enumerate() {
            let (a, n) = (st.pos, t.len());
            if st.mtp {
                let mut m = MtpSeq { st: i, start: a, hid: Vec::new(), next: Vec::new() };
                if st.mtp_pos + 1 == a {
                    m.start = a - 1;
                    m.hid.push((st.h_last.addr(), 1));
                    m.next.push(t[0]);
                }
                if n > 1 {
                    m.hid.push((ws.hid + (off * D * 4) as u64, n - 1));
                    m.next.extend_from_slice(&t[1..]);
                }
                if !m.next.is_empty() {
                    seqs.push(m);
                }
                after.push((st.h_last.addr(), ws.hid + ((off + n - 1) * D * 4) as u64));
            }
            off += n;
        }
        self.mtp_pass(states, ws, s, &seqs, false, &after)?;
        for (st, t) in states.iter_mut().zip(tokens) {
            if st.mtp {
                st.mtp_pos = st.pos + t.len() - 1;
            }
        }
        Ok(())
    }

    /// One pass of the MTP layer over `seqs` (each sequence's rows contiguous, in position order): writes
    /// its cache rows; with `heads`, each sequence's last row's logits into ws.out_logits (row j for
    /// seqs[j]) and the next draft's hidden input into ws.dh row j. Then the row copies `after`
    /// ((destination, source) of D floats; the pass reads its hidden inputs before them).
    fn mtp_pass(&self, states: &[&mut State], ws: &mut Workspace, s: &Stream, seqs: &[MtpSeq], heads: bool, after: &[(u64, u64)]) -> Result<()> {
        if seqs.is_empty() {
            for &(dst, src) in after {
                s.copy_d2d(dst, src, D * 4)?;
            }
            return Ok(());
        }
        let nseq = seqs.len();
        let n: usize = seqs.iter().map(|m| m.next.len()).sum();
        ensure!(n <= ws.rows_cap && nseq <= MAX_SEQS && after.len() <= MAX_SEQS, "MTP pass of {n} rows");
        let small = seqs.iter().all(|m| m.next.len() <= SMALL);
        ensure!(small || nseq == 1, "MTP pass of several sequences with more than {SMALL} rows");
        let groups = seqs.iter().map(|m| group_bucket(m.start + m.next.len())).max().unwrap_or(1);
        let part = if small { ws.part(n, groups)? } else { 0 };
        // The staging area may still feed the previous pass's uploads.
        ws.mtp_up.synchronize()?;
        let (_, memb_at, mseqs_at, _, tab_at, _) = Workspace::pinned_layout(ws.rows_cap);
        let heads_at = n;
        let after_at = n + if heads { nseq } else { 0 };
        let mut tab: Vec<u64> = Vec::with_capacity(2 * (after_at + after.len()));
        {
            let buf = ws.pinned.as_mut_slice();
            let mut d = nk::Seqs::default();
            let mut j = 0;
            for (si, m) in seqs.iter().enumerate() {
                let st = &states[m.st];
                d.mamba[si] = st.mamba.addr();
                d.kv[si] = st.kv.addr();
                d.start[si] = m.start as i32;
                d.row0[si] = j as i32;
                for &(addr, rows) in &m.hid {
                    for i in 0..rows {
                        tab.push(ws.mhid + ((tab.len() / 2) * D * 4) as u64);
                        tab.push(addr + (i * D * 4) as u64);
                    }
                }
                for &tok in &m.next {
                    ensure!(tok >= 0 && (tok as usize) < self.cfg.vocab, "token {tok} out of range");
                    let row = self.embed_row(tok as usize);
                    let dst = &mut buf[memb_at + j * D * 4..memb_at + (j + 1) * D * 4];
                    for (c, o) in row.chunks_exact(2).zip(dst.chunks_exact_mut(4)) {
                        let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                        o.copy_from_slice(&v.to_le_bytes());
                    }
                    j += 1;
                }
                ensure!(tab.len() / 2 == j, "MTP pass: {} hidden rows for {j} tokens", tab.len() / 2);
            }
            d.row0[nseq] = j as i32;
            d.nseq = nseq as i32;
            buf[mseqs_at..mseqs_at + SEQS_BYTES].copy_from_slice(d.as_bytes());
            if heads {
                for si in 0..nseq {
                    let last = d.row0[si + 1] as usize - 1;
                    tab.push(ws.dx + (si * D * 4) as u64);
                    tab.push(ws.h + (last * D * 4) as u64);
                }
            }
            for &(dst, src) in after {
                tab.push(dst);
                tab.push(src);
            }
            for (i, v) in tab.iter().enumerate() {
                buf[tab_at + i * 8..tab_at + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
            }
        }
        unsafe {
            s.copy_h2d(ws.memb, ws.pinned.as_ptr().add(memb_at), n * D * 4)?;
            s.copy_h2d(ws.mseqs, ws.pinned.as_ptr().add(mseqs_at), SEQS_BYTES)?;
            s.copy_h2d(ws.mtab, ws.pinned.as_ptr().add(tab_at), tab.len() * 8)?;
        }
        ws.mtp_up.record(s)?;
        let plan = Plan {
            r: n,
            nseq,
            small,
            groups,
            part,
            kv_fp8: states[seqs[0].st].kv_format == Kv::Fp8,
            want: Logits::None,
            seqs: ws.mseqs,
            defer: false,
        };
        let kv = states[seqs[0].st].kv_rows(self.n_attn);
        if small && graphs_enabled(ws) {
            let key = (features(), plan, heads);
            if !ws.mgraphs.contains_key(&key) {
                if ws.mgraphs.len() >= MAX_GRAPHS {
                    ws.mgraphs.clear();
                }
                let g = Graph::capture(s, |s| self.mtp_body(ws, s, &plan, kv, heads))?;
                ws.mgraphs.insert(key, g);
            }
            ws.mgraphs[&key].launch(s)?;
        } else {
            self.mtp_body(ws, s, &plan, kv, heads)?;
        }
        debug_assert_eq!(heads_at + if heads { nseq } else { 0 }, after_at);
        if !after.is_empty() {
            nk::copy_rows(s, ws.mtab + (after_at * 16) as u64, after.len())?;
        }
        Ok(())
    }

    /// An MTP pass's work on the device after its uploads (see [`NemotronModel::mtp_pass`]).
    fn mtp_body(&self, ws: &Workspace, s: &Stream, plan: &Plan, kv: nk::KvRows, heads: bool) -> Result<()> {
        let mtp = self.mtp.as_ref().expect("MTP layer");
        let (n, nseq, small) = (plan.r, plan.nseq, plan.small);
        let eps = self.cfg.eps;
        // The hidden inputs into ws.mhid.
        nk::copy_rows(s, ws.mtab, n)?;
        nk::rms(s, ws.memb, D, mtp.enorm, eps, n, ws.cat, 2 * D, false)?;
        nk::rms(s, ws.mhid, D, mtp.hnorm, eps, n, ws.cat + (D * 2) as u64, 2 * D, false)?;
        linear(s, ws.cat, 2 * D, n, &mtp.eh, ws.h, D, small, false)?;
        for layer in &mtp.layers {
            nk::rms(s, ws.h, D, layer.norm, eps, n, ws.xn, D, false)?;
            match &layer.mixer {
                Mixer::Attn(a) => self.attention(ws, s, a, kv, plan)?,
                Mixer::Moe(m) => self.moe(ws, s, m, n, small)?,
                Mixer::Mamba(_) => anyhow::bail!("MTP layers have no Mamba mixer"),
            }
            k::add_f32(s, ws.h, ws.mix, n * D)?;
        }
        if !heads {
            return Ok(());
        }
        // Each sequence's last row into ws.dx.
        nk::copy_rows(s, ws.mtab + (n * 16) as u64, nseq)?;
        nk::rms(s, ws.dx, D, mtp.final_norm, eps, nseq, ws.xn, D, false)?;
        linear(s, ws.xn, D, nseq, &self.lm_head, ws.out_logits, self.cfg.vocab, true, false)?;
        if mtp_post() {
            nk::rms(s, ws.dx, D, mtp.final_norm, eps, nseq, ws.dh, D, true)
        } else {
            s.copy_d2d(ws.dh, ws.dx, nseq * D * 4)
        }
    }

    /// Ends a step for drafting: after [`NemotronModel::verify`] keeps the first `keep[i]` fed tokens of
    /// each sequence (1 ..= its tokens; the Mamba states commit those rows), and after a plain step does
    /// nothing to the main model. Then, for sequences whose next token is known (`next[i]`, the token
    /// chosen after the kept rows), runs the MTP layer up to it and drafts up to `want[i]` tokens after it
    /// (greedy, each with its stated probability). `states` must be the step's, in its order.
    pub fn commit(&self, states: &mut [&mut State], ws: &mut Workspace, s: &Stream, keep: &[usize], next: &[Option<i32>], want: &[usize]) -> Result<Vec<Drafts>> {
        let nseq = states.len();
        ensure!(keep.len() == nseq && next.len() == nseq && want.len() == nseq, "commit: one keep, next and want per sequence");
        // The verification step's rows of each sequence (its offset in the step, first position, kept).
        let mut fed = Vec::with_capacity(nseq);
        let mut off = 0;
        let mut deferred = false;
        let mut acc = vec![0i32; nseq];
        for (i, st) in states.iter_mut().enumerate() {
            match st.uncommitted.take() {
                Some((a, t)) => {
                    ensure!((1..=t).contains(&keep[i]), "commit: keep {} of {t} rows", keep[i]);
                    deferred = true;
                    acc[i] = keep[i] as i32;
                    st.pos = a + keep[i];
                    st.history.truncate(a + keep[i]);
                    st.last_t = keep[i];
                    fed.push(Some((off, a, keep[i])));
                    off += t;
                }
                None => fed.push(None),
            }
        }
        if deferred {
            ensure!(fed.iter().all(Option::is_some), "commit: a step mixing verified and plain sequences");
            let at = Workspace::pinned_layout(ws.rows_cap).3;
            // The accepted rows' staging area may still feed the previous commit's upload.
            ws.acc_up.synchronize()?;
            let buf = &mut ws.pinned.as_mut_slice()[at..at + nseq * 4];
            for (i, a) in acc.iter().enumerate() {
                buf[i * 4..i * 4 + 4].copy_from_slice(&a.to_le_bytes());
            }
            unsafe { s.copy_h2d(ws.accepted, ws.pinned.as_ptr().add(at), nseq * 4)? };
            ws.acc_up.record(s)?;
            nk::mamba_set_ctl(s, ws.seqs, nseq, ws.accepted, true)?;
        }
        let mut out = vec![Drafts::default(); nseq];
        if self.mtp.is_none() {
            return Ok(out);
        }
        // The MTP layer up to each sequence's last kept position (and the next token's row, which drafts).
        let mut seqs = Vec::new();
        let mut drafting = Vec::new();
        // h_last updates, after the pass (which may read the previous h_last).
        let mut after = Vec::new();
        for (i, st) in states.iter_mut().enumerate() {
            if !st.mtp {
                continue;
            }
            let (row0, a, kept) = fed[i].unwrap_or((0, st.pos, 0));
            // Rows from position mtp_pos on (contiguous): a - 1 (h_last), the verified rows but the last,
            // then the last kept position with the next token.
            if kept > 0 && st.mtp_pos + 1 < a {
                // Positions the MTP layer never saw (not expected): continue from the verified rows.
                st.mtp_pos = a;
            }
            let mut m = MtpSeq { st: i, start: st.mtp_pos, hid: Vec::new(), next: Vec::new() };
            // The hidden state at pos - 1 (h_last's value once this commit is done).
            let mut last = st.h_last.addr();
            if kept > 0 {
                if st.mtp_pos + 1 == a {
                    m.hid.push((st.h_last.addr(), 1));
                    m.next.push(st.history[a]);
                }
                // Verified rows a .. a + kept - 1: the hidden state of the last one becomes h_last.
                let rows = ws.hid + (row0 * D * 4) as u64;
                if kept > 1 {
                    m.hid.push((rows, kept - 1));
                    m.next.extend_from_slice(&st.history[a + 1..a + kept]);
                }
                last = rows + ((kept - 1) * D * 4) as u64;
                after.push((st.h_last.addr(), last));
                st.mtp_pos = a + kept - 1;
            }
            if let Some(tok) = next[i]
                && st.mtp_pos + 1 == st.pos
            {
                m.hid.push((last, 1));
                m.next.push(tok);
                st.mtp_pos = st.pos;
                let w = want[i].min(MAX_DRAFTS).min(st.capacity.saturating_sub(st.pos + 1));
                if w > 0 {
                    st.reserve(st.pos + w)?;
                    drafting.push((i, seqs.len(), w));
                }
            }
            if !m.next.is_empty() {
                seqs.push(m);
            }
        }
        let vocab = self.cfg.vocab;
        self.mtp_pass(states, ws, s, &seqs, !drafting.is_empty(), &after)?;
        // Drafts: from the drafting sequences' last rows of that pass, then one row each per further draft
        // (the previous draft's MTP output as the hidden input, the draft as the token).
        let mut active: Vec<(usize, usize, usize)> = drafting;
        let mut n_rows = seqs.len();
        while !active.is_empty() {
            k::argmax(s, ws.out_logits, n_rows, vocab, vocab, ws.argmax_scratch, ws.argmax)?;
            k::row_lse(s, ws.out_logits, n_rows, vocab, vocab, ws.argmax, ws.lse, ws.target)?;
            // Picks, log-sum-exps and target logits are consecutive (logit_rows each): one download.
            let lr = ws.logit_rows;
            let all = s.download::<u32>(ws.argmax, 3 * lr)?;
            let picks: Vec<i32> = all[..n_rows].iter().map(|&v| v as i32).collect();
            let lse: Vec<f32> = all[lr..lr + n_rows].iter().map(|&v| f32::from_bits(v)).collect();
            let tgt: Vec<f32> = all[2 * lr..2 * lr + n_rows].iter().map(|&v| f32::from_bits(v)).collect();
            let mut next_seqs = Vec::new();
            let mut next_active = Vec::new();
            for &(i, row, left) in &active {
                let tok = picks[row];
                out[i].tokens.push(tok);
                out[i].conf.push((tgt[row] - lse[row]).exp());
                if left > 1 {
                    let depth = out[i].tokens.len();
                    next_active.push((i, next_seqs.len(), left - 1));
                    next_seqs.push(MtpSeq { st: i, start: states[i].pos + depth - 1, hid: vec![(ws.dh + (row * D * 4) as u64, 1)], next: vec![tok] });
                }
            }
            if next_seqs.is_empty() {
                break;
            }
            // The pass copies its hidden inputs (rows of ws.dh) before it rewrites ws.dh.
            self.mtp_pass(states, ws, s, &next_seqs, true, &[])?;
            n_rows = next_seqs.len();
            active = next_active;
        }
        Ok(out)
    }
}
