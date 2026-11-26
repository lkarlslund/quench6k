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

use std::path::PathBuf;

use anyhow::{Result, ensure};
use quench6k_cuda::kernels as k;
use quench6k_cuda::{DeviceBuffer, PinnedBuffer, Stream, VmBuffer};

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
/// Rows of a speculative (deferred) step the Mamba state records (0: no speculative steps yet; MTP
/// drafting will set it).
pub const REC_ROWS: usize = 0;
/// Rows per pass of the grouped expert GEMM in large steps (bounds its buffers).
const EX_ROWS: usize = 512;
/// Positions a sequence's cache mapping grows by at least.
const RESERVE_STEP: usize = 4096;
/// Top logits per position kept by teacher-forced scoring.
pub const TOP_K: usize = 32;

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
}

impl State {
    /// A sequence of up to `capacity` positions: its attention caches are reserved as addresses and mapped
    /// as positions are written; the Mamba block is allocated (zeroed) now.
    pub fn new(model: &NemotronModel, capacity: usize, kv_format: Kv, s: &Stream) -> Result<Self> {
        let kv = VmBuffer::reserve((capacity * Self::position_bytes(&model.cfg, kv_format)).max(1))?;
        let mamba = DeviceBuffer::new(Self::state_bytes(&model.cfg))?;
        s.memset(mamba.addr(), 0, mamba.bytes())?;
        Ok(Self {
            pos: 0,
            capacity,
            history: Vec::new(),
            last_t: 0,
            kv_format,
            kv,
            mamba,
            n_attn: model.n_attn,
        })
    }

    /// Attention cache bytes per position (all attention layers).
    pub fn position_bytes(cfg: &Config, kv: Kv) -> usize {
        cfg.count(Kind::Attention) * kv.row_bytes()
    }

    /// Bytes of a sequence's Mamba block (fixed, whatever its length).
    pub fn state_bytes(cfg: &Config) -> usize {
        nk::mamba_seq_bytes(cfg.count(Kind::Mamba), REC_ROWS)
    }

    /// Device memory the sequence holds now: its Mamba block and the mapped part of its caches.
    pub fn device_bytes(&self) -> usize {
        self.mamba.bytes() + self.kv.mapped()
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
        s.memset(self.mamba.addr(), 0, self.mamba.bytes())
    }

    /// Keeps the cached prefix when `prompt` extends the sequence's history (the recurrent states hold
    /// exactly the history), otherwise starts over; returns the number of prompt tokens already in the
    /// state. The prompt's last token must still be run for its logits, so a prompt equal to the history
    /// starts over too.
    pub fn rewind(&mut self, prompt: &[i32], s: &Stream) -> Result<usize> {
        let extends = self.history.len() < prompt.len() && prompt.starts_with(&self.history);
        if !extends {
            self.reset(s)?;
        }
        Ok(self.pos)
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
    /// Embedding rows (F32) and the descriptor staged for upload.
    pinned: PinnedBuffer,
    /// Debugging: hidden states after each layer appended to `lanes-NN.bin` (F32 [rows, D]) and the final
    /// norm's output to `final.bin` (QUENCH6K_NEM_DUMP or set by the caller).
    pub dump: Option<PathBuf>,
    pub device_bytes: usize,
    _alloc: Alloc,
}

const SEQS_BYTES: usize = std::mem::size_of::<nk::Seqs>();

impl Workspace {
    /// Device bytes of the buffers, in the order [`Workspace::new`] allocates them.
    fn sizes(cfg: &Config, max_t: usize, logit_rows: usize) -> (usize, [usize; 34]) {
        let cap = max_t.max(MAX_SEQS * MAX_STEP);
        let lr = logit_rows.max(1);
        let vocab = cfg.vocab;
        let pairs = cap.clamp(nk::SMALL_ROWS, EX_ROWS) * TOPK;
        let max_tiles = pairs / k::moe_tile_rows().max(1) + EXPERTS;
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
                lr * 4,
                lr * 4,
                lr * 4,
                lr * 4,
                lr * TOP_K * 4,
                lr * TOP_K * 4,
            ],
        )
    }

    /// Device bytes a workspace for steps of `max_t` tokens and `logit_rows` logits rows takes (without
    /// the small steps' attention partials, which grow with the context).
    pub fn bytes(cfg: &Config, max_t: usize, logit_rows: usize) -> usize {
        Self::sizes(cfg, max_t, logit_rows).1.iter().sum()
    }

    pub fn new(model: &NemotronModel, max_t: usize, logit_rows: usize) -> Result<Self> {
        nk::check_layout()?;
        let (max_tiles, sz) = Self::sizes(&model.cfg, max_t, logit_rows);
        let cap = max_t.max(MAX_SEQS * MAX_STEP);
        let mut a = Alloc { buffers: Vec::new(), bytes: 0 };
        let mut b = [0u64; 34];
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
            lse: b[30],
            target: b[31],
            top_idx: b[32],
            top_val: b[33],
            pinned: PinnedBuffer::new(cap * D * 4 + SEQS_BYTES)?,
            dump: std::env::var_os("QUENCH6K_NEM_DUMP").map(PathBuf::from),
            device_bytes: a.bytes,
            _alloc: a,
        })
    }

    /// The small steps' attention partials for `rows` rows below `groups` position groups.
    fn part(&mut self, rows: usize, groups: usize) -> Result<u64> {
        let bytes = nk::attn_part_floats(rows, groups) * 4;
        if self.part.as_ref().is_none_or(|p| p.bytes() < bytes) {
            // Room for twice the groups: fewer reallocations as sequences grow.
            let grown = nk::attn_part_floats(MAX_SEQS * MAX_STEP, groups * 2) * 4;
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
        for r0 in (0..m).step_by(gm) {
            let rows = gm.min(m - r0);
            k::gemv(s, rows, x + (r0 * ldx * 2) as u64, ldx, l.fmt, l.w, l.scale, y + (r0 * ldy * 4) as u64, ldy, 0, l.n, l.k, 1, l.alpha, accumulate)?;
        }
        Ok(())
    } else {
        k::gemm_f32(s, x, ldx, l.fmt, l.w, l.k, l.scale, y, ldy, m, l.n, l.k, l.alpha, accumulate)
    }
}

fn append_f32(dir: &std::path::Path, name: &str, v: &[f32]) -> Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::OpenOptions::new().create(true).append(true).open(dir.join(name))?.write_all(&bytes)?;
    Ok(())
}

/// What the enqueued work of a step depends on besides device data.
#[derive(Clone, Copy)]
struct Plan {
    r: usize,
    nseq: usize,
    small: bool,
    /// Position groups of the small steps' attention: ceil((last position + 1) / ATTN_GROUP) over rows.
    groups: usize,
    /// Partials of the small steps' attention.
    part: u64,
    kv_fp8: bool,
    want: Logits,
}

impl NemotronModel {
    /// Runs `tokens` (at positions state.pos ..) through the model, appending them to the caches and
    /// states. Logits (F32 [rows, vocab]) land in ws.out_logits: one row for `Last`, one per token for
    /// `All`.
    pub fn forward(&self, state: &mut State, ws: &mut Workspace, s: &Stream, tokens: &[i32], want: Logits) -> Result<()> {
        self.step(&mut [state], ws, s, &[tokens], want)
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
        self.step(states, ws, s, tokens, Logits::All)
    }

    /// Embedding rows (F32) and the sequence descriptor into the pinned buffer.
    fn stage(&self, states: &[&mut State], ws: &mut Workspace, tokens: &[&[i32]]) -> Result<()> {
        let emb_bytes = ws.rows_cap * D * 4;
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

    fn step(&self, states: &mut [&mut State], ws: &mut Workspace, s: &Stream, tokens: &[&[i32]], want: Logits) -> Result<()> {
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
            ensure!(st.pos + t.len() <= st.capacity, "context full ({} + {} > {})", st.pos, t.len(), st.capacity);
            st.reserve(st.pos + t.len())?;
        }
        let groups = states.iter().zip(tokens).map(|(st, t)| (st.pos + t.len()).div_ceil(nk::ATTN_GROUP)).max().unwrap_or(1);
        let part = if small { ws.part(r, groups)? } else { 0 };
        // The pinned buffer may still feed the previous step's uploads.
        s.synchronize()?;
        self.stage(states, ws, tokens)?;
        let emb_bytes = ws.rows_cap * D * 4;
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
        };
        let kv: Vec<nk::KvRows> = (0..self.n_attn).map(|slot| states[0].kv_rows(slot)).collect();
        self.enqueue(ws, s, &plan, &kv)?;
        // Nothing of this step is pending in the Mamba states (no deferred steps yet).
        nk::mamba_set_ctl(s, ws.seqs, nseq, 0, false)?;
        for (st, t) in states.iter_mut().zip(tokens) {
            st.pos += t.len();
            st.last_t = t.len();
            st.history.extend_from_slice(t);
        }
        Ok(())
    }

    fn enqueue(&self, ws: &Workspace, s: &Stream, plan: &Plan, kv: &[nk::KvRows]) -> Result<()> {
        let (r, small, want) = (plan.r, plan.small, plan.want);
        let eps = self.cfg.eps;
        for (i, layer) in self.layers.iter().enumerate() {
            nk::rms(s, ws.h, D, layer.norm, eps, r, ws.xn, D, false)?;
            match &layer.mixer {
                Mixer::Mamba(m) => self.mamba(ws, s, m, self.slot[i], plan)?,
                Mixer::Attn(a) => self.attention(ws, s, a, kv[self.slot[i]], plan)?,
                Mixer::Moe(m) => self.moe(ws, s, m, r, small)?,
            }
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
        nk::mamba_mixer(s, ws.big, INP, r, ws.seqs, plan.nseq, slot, REC_ROWS, &m.w, false, &ws.mamba, ws.act)?;
        linear(s, ws.act, DI, r, &m.out_proj, ws.mix, D, plan.small, false)
    }

    /// Attention mixer: q/k/v; keys and values into the caches; causal GQA attention; o_proj into ws.mix.
    fn attention(&self, ws: &Workspace, s: &Stream, a: &Attn, kv: nk::KvRows, plan: &Plan) -> Result<()> {
        let r = plan.r;
        debug_assert_eq!(kv.fp8, plan.kv_fp8);
        linear(s, ws.xn, D, r, &a.qkv, ws.big, QKV, plan.small, false)?;
        nk::attn_prep(s, ws.big, QKV, 0, H * HD, (H + KVH) * HD, r, ws.seqs, kv, ws.q)?;
        if plan.small {
            nk::attention_small(s, ws.q, r, ws.seqs, kv, plan.groups, plan.part, ws.act)?;
        } else {
            nk::attention_prefill(s, ws.q, r, ws.seqs, kv, ws.act)?;
        }
        linear(s, ws.act, H * HD, r, &a.o, ws.mix, D, plan.small, false)
    }

    /// MoE mixer: router; routed experts in the latent space (fc1 -> experts -> combine -> fc2); the
    /// shared expert on the full width, added into ws.mix.
    fn moe(&self, ws: &Workspace, s: &Stream, m: &Moe, r: usize, small: bool) -> Result<()> {
        linear(s, ws.xn, D, r, &m.gate, ws.logits, EXPERTS, small, false)?;
        nk::moe_route(s, ws.logits, EXPERTS, m.bias, self.cfg.routed_scaling, r, ws.sel_e, ws.sel_w)?;
        linear(s, ws.xn, D, r, &m.fc1, ws.lat, LAT, small, false)?;
        quench6k_cuda::hy4::to_bf16(s, ws.lat, ws.lat_bf, r * LAT)?;
        let ex = &m.experts;
        let at = |base: u64, row: usize, width: usize| base + (row * width) as u64;
        if small {
            // Per (row, expert) pair, in launches of SMALL_ROWS rows (row-wise: batch invariant).
            for r0 in (0..r).step_by(nk::SMALL_ROWS) {
                let n = nk::SMALL_ROWS.min(r - r0);
                let sel_e = at(ws.sel_e, r0, TOPK * 4);
                nk::moe_experts_small(s, at(ws.lat_bf, r0, LAT * 2), sel_e, &ex.up, &ex.down, n, ws.pair_row, ws.eact, ws.ey)?;
                nk::moe_combine(s, ws.ey, 0, at(ws.sel_w, r0, TOPK * 4), n, at(ws.ex_bf, r0, LAT * 2))?;
            }
        } else {
            // Pairs grouped by expert for the grouped NVFP4 GEMM, EX_ROWS rows at a time.
            let tile = k::moe_tile_rows();
            for r0 in (0..r).step_by(EX_ROWS) {
                let n = EX_ROWS.min(r - r0);
                let sel_e = at(ws.sel_e, r0, TOPK * 4);
                s.memset(ws.counts, 0, EXPERTS * 4)?;
                nk::moe_plan(s, sel_e, n, tile, ws.counts, ws.offsets, ws.cursor, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.pair_row)?;
                nk::moe_gather(s, at(ws.lat_bf, r0, LAT * 2), ws.pair_row, n, ws.xs)?;
                let gemm = |x: u64, table: (u64, u64), s2: u64, y: u64, nn: usize, kk: usize| {
                    k::moe_nvfp4_gemm(s, x, table.0, table.1, s2, ws.offsets, ws.tile_expert, ws.tile_row, ws.num_tiles, ws.max_tiles, y, nn, kk)
                };
                let pairs = n * TOPK;
                gemm(ws.xs, ex.tables[0], ex.up.s2, ws.ey, FF, LAT)?;
                nk::relu2(s, ws.ey, FF, FF, pairs, ws.eact, FF)?;
                gemm(ws.eact, ex.tables[1], ex.down.s2, ws.ey, LAT, FF)?;
                nk::moe_combine(s, ws.ey, ws.pair_row, at(ws.sel_w, r0, TOPK * 4), n, at(ws.ex_bf, r0, LAT * 2))?;
            }
        }
        linear(s, ws.ex_bf, LAT, r, &m.fc2, ws.mix, D, small, false)?;
        linear(s, ws.xn, D, r, &m.shared_up, ws.big, SH, small, false)?;
        nk::relu2(s, ws.big, SH, SH, r, ws.act, SH)?;
        linear(s, ws.act, SH, r, &m.shared_down, ws.mix, D, small, true)
    }
}
