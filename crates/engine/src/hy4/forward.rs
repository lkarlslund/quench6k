//! Hy4-flash forward pass over a chunk of new tokens of one sequence.
//!
//! Per sequence the indexed layers keep every position's cache row (`[ctx, 576]` BF16), token key
//! (`[ctx, 128]`) and pooled group key (`[ctx / 4, 128]`); the sliding layers keep a ring of the last
//! `RING` rows. Chunks hold at most `max_t` tokens (`max_t + 511 <= RING`).

use anyhow::{Context, Result, ensure};
use quench6k_cuda::hy4::{self as hk, AttnMode};
use quench6k_cuda::kernels as k;
use quench6k_cuda::{DeviceBuffer, PinnedBuffer, Stream};

use super::*;

const RING: usize = 4096;
const MAX_SPLITS: usize = 32;
/// Chunks of up to this many tokens run the routed experts per (token, expert) pair; larger ones group
/// the pairs by expert for the tensor-core GEMM.
const SMALL_T: usize = 8;
/// Tile rows of the grouped NVFP4 GEMM (moe.cu BM).
const MOE_BM: usize = 64;

/// Prefill chunks run the routed experts W4A4 (NVFP4 activations, FP4 tensor cores) when
/// QUENCH6K_HY4_W4A4 is set to 1.
fn w4a4() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QUENCH6K_HY4_W4A4").is_ok_and(|v| v == "1"))
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
}

enum Cache {
    Sliding { rows: u64 },
    Indexed { rows: u64, tk: u64, pooled: u64 },
}

/// One sequence's caches and token history.
pub struct State {
    pub pos: usize,
    pub ctx: usize,
    caches: Vec<Cache>,
    pub history: Vec<i32>,
    pub device_bytes: usize,
    /// CUDA graphs of small steps by token count and logits kind; they capture this state's caches.
    graphs: Vec<Option<quench6k_cuda::Graph>>,
    /// Sliding rings at snap_pos (the end of the last prompt).
    snap: Option<DeviceBuffer>,
    snap_pos: Option<usize>,
    _alloc: Alloc,
}

impl State {
    pub fn new(model: &Hy4Model, ctx: usize, s: &Stream) -> Result<Self> {
        let ctx = ctx.div_ceil(GROUP) * GROUP;
        let mut a = Alloc::new();
        let mut caches = Vec::new();
        for layer in &model.layers {
            if layer.attn.sliding {
                caches.push(Cache::Sliding {
                    rows: a.get(RING * KROW * 2)?,
                });
            } else {
                let rows = a.get(ctx * KROW * 2)?;
                let tk = a.get(ctx * ID * 2)?;
                let pooled = a.get(ctx / GROUP * ID * 2)?;
                caches.push(Cache::Indexed { rows, tk, pooled });
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
            snap: None,
            snap_pos: None,
            _alloc: a,
        })
    }

    /// Forgets the sequence (the caches are overwritten as positions are refilled).
    pub fn reset(&mut self) {
        self.pos = 0;
        self.history.clear();
        self.snap_pos = None;
    }

    /// Drops positions from `pos` on. The sliding layers keep the last RING rows, so `pos` must be within
    /// REACH positions of the current one (see [`State::rewind`]).
    pub fn truncate(&mut self, pos: usize) {
        self.pos = self.pos.min(pos);
        self.history.truncate(self.pos);
    }

    /// Keeps the longest cached prefix of `prompt` that can be reused (at most prompt.len() - 1 tokens,
    /// so the last prompt token is run again for its logits) and returns its length: the common prefix
    /// with the history if the sliding rings still hold its last WINDOW positions, else the snapshot's
    /// position (after restoring the rings) if it is within reach, else 0.
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

    /// Copies the sliding rings at the current position (the end of a prompt), for [`State::rewind`].
    pub fn snapshot(&mut self, s: &Stream) -> Result<()> {
        let rings: Vec<u64> = self
            .caches
            .iter()
            .filter_map(|c| match c {
                Cache::Sliding { rows } => Some(*rows),
                _ => None,
            })
            .collect();
        let bytes = RING * KROW * 2;
        if self.snap.is_none() {
            self.snap = Some(DeviceBuffer::new(rings.len() * bytes)?);
        }
        let dst = self.snap.as_ref().unwrap().addr();
        for (i, r) in rings.iter().enumerate() {
            s.copy_d2d(dst + (i * bytes) as u64, *r, bytes)?;
        }
        self.snap_pos = Some(self.pos);
        Ok(())
    }

    fn restore(&mut self, s: &Stream) -> Result<()> {
        let src = self.snap.as_ref().context("no snapshot")?.addr();
        let bytes = RING * KROW * 2;
        let mut i = 0;
        for c in &self.caches {
            if let Cache::Sliding { rows } = c {
                s.copy_d2d(*rows, src + (i * bytes) as u64, bytes)?;
                i += 1;
            }
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
    /// Indexer groups (context / 4) the score rows hold.
    groups: usize,
    pub start: u64,
    emb: u64,
    lanes: u64,
    x: u64,
    post: u64,
    hc_part: u64,
    f_a: u64,  // F32 scratch [T, max(16384, 13824 ...)]
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
    pub device_bytes: usize,
    _alloc: Alloc,
}

pub const TOP_K: usize = 32;

impl Workspace {
    /// Chunks of up to `max_t` tokens over contexts of up to `ctx`; `logit_rows`: most rows of logits a
    /// step produces (1 for generation; max_t for scoring every position).
    pub fn new(model: &Hy4Model, max_t: usize, ctx: usize, logit_rows: usize) -> Result<Self> {
        ensure!(max_t + WINDOW <= RING, "chunks of {max_t} tokens exceed the sliding ring");
        let mut a = Alloc::new();
        let t = max_t;
        let hmax = model.cfg.heads_sliding.max(model.cfg.heads_indexed);
        let f_width = (hmax * QK).max(13824).max((HC + 1) * D);
        let pairs = t * TOPK;
        let groups = ctx.div_ceil(GROUP);
        let max_tiles = pairs / MOE_BM + EXPERTS;
        let vocab = model.cfg.vocab;
        let small = hk::heads_gemv_max_t().min(t);
        let part = hk::attn_part_floats(small, hmax, MAX_SPLITS);
        let pinned_bytes = t * D * 2 + model.engram_layers.len() * t * ENGRAM_HEADS * (ENGRAM_DIM + 4) + t + 64;
        let ws = Self {
            max_t,
            logit_rows: logit_rows.max(16),
            groups,
            start: a.get(16)?,
            emb: a.get(t * D * 2)?,
            lanes: a.get(t * HC * D * 2)?,
            x: a.get(t * D * 2)?,
            post: a.get(t * HC * 4)?,
            hc_part: a.get(hk::hc_partial_floats(t) * 4)?,
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
            scores: a.get(t * groups * 4)?,
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
            q8: a.get(model.engram_layers.len().max(1) * t * ENGRAM_HEADS * ENGRAM_DIM)?,
            q8_sc: a.get(model.engram_layers.len().max(1) * t * ENGRAM_HEADS * 4)?,
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
            device_bytes: 0,
            _alloc: Alloc::new(),
        };
        let bytes = a.bytes;
        Ok(Self {
            device_bytes: bytes,
            _alloc: a,
            ..ws
        })
    }
}

/// Y [m, l.n] F32 = X [m, l.k] BF16 . W^T.
fn linear(s: &Stream, x: u64, ldx: usize, m: usize, l: &Linear, y: u64, ldy: usize) -> Result<()> {
    if m <= k::gemv_max_m() {
        k::gemv(s, m, x, ldx, l.fmt, l.w, l.scale, y, ldy, 0, l.n, l.k, 1, 1.0, false)
    } else {
        k::gemm_f32(s, x, ldx, l.fmt, l.w, l.k, l.scale, y, ldy, m, l.n, l.k, 1.0, false)
    }
}

/// Which positions' logits a forward pass produces.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Logits {
    None,
    Last,
    All,
}

impl Hy4Model {
    /// Host side of the Engram layers for `tokens` at positions `state.pos ..`: hashed rows gathered
    /// into the pinned buffer (per engram layer: q8 [t, 16, 96] then scales [t, 16]), and the mask.
    fn engram_inputs(&self, state: &State, tokens: &[i32], ws: &mut Workspace) -> (usize, usize) {
        let t = tokens.len();
        let pad = self.cfg.engram_pad;
        let n_layers = self.engram_layers.len();
        let q8_bytes = n_layers * t * ENGRAM_HEADS * ENGRAM_DIM;
        let base = t * D * 2;
        let full: Vec<i64> = state.history.iter().chain(tokens).map(|&x| x as i64).collect();
        let off = state.history.len();
        // Rows of every (layer, token, head), then read-ahead for all of them before any copy: the reads
        // of rows not in the page cache overlap.
        let mut rows: Vec<i64> = Vec::with_capacity(n_layers * t * ENGRAM_HEADS);
        for &layer in &self.engram_layers {
            let eng = self.layers[layer].engram.as_ref().unwrap();
            for j in 0..t {
                let p = off + j;
                let ngram: Vec<i64> = (0..self.cfg.engram_max_ngram)
                    .map(|o| if p >= o { full[p - o] } else { pad })
                    .collect();
                eng.hash.rows(&ngram, &mut rows);
            }
        }
        // Small steps (decode): read-ahead hints for every row, so cold rows load concurrently. Larger
        // chunks fault rows in from the copy threads instead (a hint per row costs a system call).
        if t <= 16 {
            for (li, &layer) in self.engram_layers.iter().enumerate() {
                let eng = self.layers[layer].engram.as_ref().unwrap();
                for &r in &rows[li * t * ENGRAM_HEADS..(li + 1) * t * ENGRAM_HEADS] {
                    eng.prefetch(r as usize);
                }
            }
        }
        let buf = ws.pinned.as_mut_slice();
        let (head, rest) = buf.split_at_mut(base + q8_bytes);
        let q8 = &mut head[base..];
        let sc = &mut rest[..n_layers * t * ENGRAM_HEADS * 4];
        let per_layer = t * ENGRAM_HEADS;
        let copy = |li: usize, idx: std::ops::Range<usize>, q8: &mut [u8], sc: &mut [u8]| {
            let eng = self.layers[self.engram_layers[li]].engram.as_ref().unwrap();
            for (k, i) in idx.enumerate() {
                let (q, s) = eng.row(rows[li * per_layer + i] as usize);
                q8[k * ENGRAM_DIM..(k + 1) * ENGRAM_DIM].copy_from_slice(q);
                sc[k * 4..k * 4 + 4].copy_from_slice(&s.to_le_bytes());
            }
        };
        // Up to 16 threads over contiguous ranges of each layer's rows: cold rows fault in parallel.
        let threads = per_layer.div_ceil(512).clamp(1, 16 / n_layers.max(1)).max(1);
        let span = per_layer.div_ceil(threads);
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
        let mask_at = base + q8_bytes + n_layers * t * ENGRAM_HEADS * 4;
        for (j, &tok) in tokens.iter().enumerate() {
            let tok = tok as i64;
            let keep = tok != pad && Some(tok) != self.cfg.image_token && Some(tok) != self.cfg.video_token;
            buf[mask_at + j] = keep as u8;
        }
        (q8_bytes, mask_at)
    }

    /// Runs `tokens` (at positions state.pos ..) through the model, appending them to the caches.
    /// Logits (F32 [rows, vocab]) land in ws.out_logits: one row for `Last`, one per token for `All`.
    /// Steps of at most 16 tokens that want no more than the last logits run as a CUDA graph captured
    /// once per token count (QUENCH6K_HY4_GRAPHS=0: eager).
    pub fn forward(&self, state: &mut State, ws: &mut Workspace, s: &Stream, tokens: &[i32], want: Logits) -> Result<()> {
        let t = tokens.len();
        ensure!(t > 0 && t <= ws.max_t, "chunk of {t} tokens (max {})", ws.max_t);
        ensure!(state.pos + t <= state.ctx, "context full ({} + {t} > {})", state.pos, state.ctx);
        ensure!(state.ctx / GROUP <= ws.groups, "workspace sized for a smaller context than the state's");
        ensure!(want != Logits::All || t <= ws.logit_rows, "{t} rows of logits (workspace holds {})", ws.logit_rows);
        // The pinned buffer may still feed the previous step's uploads.
        let timing = std::env::var_os("QUENCH6K_HY4_TIMING").is_some();
        let t0 = std::time::Instant::now();
        s.synchronize()?;
        let t1 = std::time::Instant::now();
        self.stage(state, ws, tokens)?;
        if timing {
            eprintln!(
                "hy4 step t={t}: wait {:.1} ms, stage {:.1} ms",
                (t1 - t0).as_secs_f64() * 1e3,
                t1.elapsed().as_secs_f64() * 1e3
            );
        }
        let graphs = std::env::var("QUENCH6K_HY4_GRAPHS").map_or(true, |v| v != "0")
            && std::env::var_os("QUENCH6K_HY4_DUMP").is_none();
        if graphs && t <= hk::heads_gemv_max_t() && (want != Logits::All || t <= ws.logit_rows) {
            let slot = t + 17 * want as usize;
            if state.graphs.len() <= slot {
                state.graphs.resize_with(slot + 1, || None);
            }
            if state.graphs[slot].is_none() {
                let g = quench6k_cuda::Graph::capture(s, |s| self.enqueue(state, ws, s, t, None, want))?;
                state.graphs[slot] = Some(g);
            }
            state.graphs[slot].as_ref().unwrap().launch(s)?;
        } else {
            self.enqueue(state, ws, s, t, Some(state.pos), want)?;
        }
        state.pos += t;
        state.history.extend_from_slice(tokens);
        Ok(())
    }

    /// Host inputs of a step in the pinned buffer: embedding rows, Engram rows and mask, start position.
    fn stage(&self, state: &State, ws: &mut Workspace, tokens: &[i32]) -> Result<()> {
        let buf = ws.pinned.as_mut_slice();
        for (j, &tok) in tokens.iter().enumerate() {
            ensure!((tok as usize) < self.cfg.vocab, "token {tok} out of range");
            buf[j * D * 2..(j + 1) * D * 2].copy_from_slice(self.embed_row(tok as usize));
        }
        let (_, mask_at) = self.engram_inputs(state, tokens, ws);
        let start_at = mask_at + tokens.len().next_multiple_of(16);
        ws.pinned.as_mut_slice()[start_at..start_at + 4].copy_from_slice(&(state.pos as i32).to_le_bytes());
        Ok(())
    }

    /// The step's device work for `t` staged tokens. `start`: the host position (eager steps choose
    /// shapes by it), or None for a position-independent step (graph capture).
    fn enqueue(&self, state: &State, ws: &Workspace, s: &Stream, t: usize, start: Option<usize>, want: Logits) -> Result<()> {
        let cfg = &self.cfg;
        let n_eng = self.engram_layers.len();
        let q8_bytes = n_eng * t * ENGRAM_HEADS * ENGRAM_DIM;
        let mask_at = t * D * 2 + q8_bytes + n_eng * t * ENGRAM_HEADS * 4;
        let start_at = mask_at + t.next_multiple_of(16);
        let pin = ws.pinned.as_ptr();
        unsafe {
            s.copy_h2d(ws.emb, pin, t * D * 2)?;
            if n_eng > 0 {
                s.copy_h2d(ws.q8, pin.add(t * D * 2), q8_bytes)?;
                s.copy_h2d(ws.q8_sc, pin.add(t * D * 2 + q8_bytes), n_eng * t * ENGRAM_HEADS * 4)?;
                s.copy_h2d(ws.mask, pin.add(mask_at), t)?;
            }
            s.copy_h2d(ws.start, pin.add(start_at), 4)?;
        }
        hk::hc_expand(s, ws.emb, ws.lanes, t)?;

        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(eng) = &layer.engram {
                let li = self.engram_layers.iter().position(|&x| x == i).unwrap();
                hk::engram_embed(
                    s,
                    ws.q8 + (li * t * ENGRAM_HEADS * ENGRAM_DIM) as u64,
                    ws.q8_sc + (li * t * ENGRAM_HEADS * 4) as u64,
                    eng.head_norm,
                    t,
                    ws.eemb,
                )?;
                linear(s, ws.eemb, ENGRAM_W, t, &eng.kv, ws.f_a, eng.kv.n)?;
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
                    t,
                    ws.lanes,
                )?;
            }
            // Attention sublayer.
            hk::hc_pre(
                s,
                ws.lanes,
                layer.attn_hc.fn_,
                layer.attn_hc.base,
                layer.attn_hc.scale,
                layer.in_norm,
                cfg.hc_eps,
                cfg.hc_mag,
                cfg.eps,
                false,
                t,
                ws.hc_part,
                ws.x,
                ws.post,
            )?;
            self.attention(state, ws, s, i, t, start)?;
            hk::hc_post(s, ws.lanes, ws.post, ws.y, D, t)?;
            // MLP sublayer.
            hk::hc_pre(
                s,
                ws.lanes,
                layer.mlp_hc.fn_,
                layer.mlp_hc.base,
                layer.mlp_hc.scale,
                layer.post_attn_norm,
                cfg.hc_eps,
                cfg.hc_mag,
                cfg.eps,
                false,
                t,
                ws.hc_part,
                ws.x,
                ws.post,
            )?;
            match &layer.mlp {
                Mlp::Dense(m) => self.dense(ws, s, m, t)?,
                Mlp::Moe(m) => self.moe(ws, s, m, t)?,
            }
            hk::hc_post(s, ws.lanes, ws.post, ws.y, D, t)?;
            if let Some(dir) = std::env::var_os("QUENCH6K_HY4_DUMP") {
                // Debugging: the lanes after each layer, appended per chunk (BF16 [t, 4, 3072]).
                use std::io::Write;
                let v: Vec<u16> = s.download(ws.lanes, t * HC * D)?;
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                let path = std::path::Path::new(&dir).join(format!("lanes-{i:02}.bin"));
                std::fs::OpenOptions::new().create(true).append(true).open(path)?.write_all(&bytes)?;
            }
        }
        if want != Logits::None {
            hk::hc_pre(
                s,
                ws.lanes,
                self.hc_head.fn_,
                self.hc_head.base,
                self.hc_head.scale,
                self.norm,
                cfg.hc_eps,
                1.0,
                cfg.eps,
                true,
                t,
                ws.hc_part,
                ws.hidden,
                ws.post,
            )?;
            let (rows, first) = if want == Logits::All { (t, 0) } else { (1, t - 1) };
            linear(
                s,
                ws.hidden + (first * D * 2) as u64,
                D,
                rows,
                &self.lm_head,
                ws.out_logits,
                cfg.vocab,
            )?;
        }
        Ok(())
    }

    fn attention(&self, state: &State, ws: &Workspace, s: &Stream, i: usize, t: usize, start: Option<usize>) -> Result<()> {
        let cfg = &self.cfg;
        let a = &self.layers[i].attn;
        let h = a.heads;
        let theta = if a.sliding { cfg.theta_sliding } else { cfg.theta_indexed };
        // q_resid (BF16 [t, 1024]) in bf_a; q (F32 [t, h * 256]) in f_a; kv_a (F32 [t, 576]) in f_c; gate in f_b.
        linear(s, ws.x, D, t, &a.q_a, ws.f_a, Q_LORA)?;
        hk::rms_rows(s, ws.f_a, Q_LORA, a.q_a_norm, Q_LORA, cfg.eps, ws.bf_a, Q_LORA, t)?;
        linear(s, ws.bf_a, Q_LORA, t, &a.q_b, ws.f_a, h * QK)?;
        linear(s, ws.x, D, t, &a.kv_a, ws.f_c, KROW)?;
        linear(s, ws.x, D, t, &a.gate, ws.f_b, h * VD)?;
        let (cache_rows, ring) = match &state.caches[i] {
            Cache::Sliding { rows } => (*rows, RING),
            Cache::Indexed { rows, .. } => (*rows, 0),
        };
        hk::mla_prep(
            s, ws.f_a, h * QK, ws.f_c, KROW, a.kv_a_norm, cfg.eps, theta, h, ws.start, ring, t, ws.qn, ws.qa, cache_rows,
        )?;
        // Absorb W_uk into the queries: qa[t, h, 0..512) = W_uk[h]^T q_nope.
        if t <= hk::heads_gemv_max_t() {
            hk::heads_gemv(
                s, ws.qn, h * NOPE, NOPE, a.wuk_t, KV_LORA * NOPE, KV_LORA, NOPE, h, t, 0, 0, ws.qa, h * KROW, KROW,
            )?;
        } else {
            k::gemm_bf16_batched(
                s, ws.qn, h * NOPE, NOPE, a.wuk_t, NOPE, KV_LORA * NOPE, ws.qa, h * KROW, KROW, t, KV_LORA, NOPE, h,
            )?;
        }
        // Keys each query sees.
        let mut mode = if a.sliding { AttnMode::Sliding } else { AttnMode::Dense };
        if let (Cache::Indexed { tk, pooled, .. }, Some(ix)) = (&state.caches[i], &a.indexer) {
            linear(s, ws.x, D, t, &ix.tk, ws.f_c + (t * KROW * 4) as u64, ID)?;
            linear(s, ws.bf_a, Q_LORA, t, &ix.q, ws.eg, IH * ID)?;
            linear(s, ws.x, D, t, &ix.hw, ws.eu, IH)?;
            hk::idx_prep(
                s,
                ws.f_c + (t * KROW * 4) as u64,
                ID,
                ix.tk_norm,
                cfg.eps,
                ws.eg,
                IH * ID,
                ws.eu,
                IH,
                cfg.theta_indexed,
                ws.start,
                t,
                *tk,
                ws.iq,
                ws.hw,
            )?;
            hk::idx_pool(s, *tk, ix.wv, ix.wg, ix.ape, ix.key_norm, cfg.eps, cfg.theta_indexed, ws.start, t, *pooled)?;
            // Eager steps score only the complete groups, and only past 2048 positions; graph steps score
            // all of the context's groups (the kernels mask by the device start) and always list the
            // selected positions, densely while the context is short.
            let groups = match start {
                Some(p) if p + t > TOPG * GROUP => Some((p + t) / GROUP),
                Some(_) => None,
                None => Some(state.ctx / GROUP),
            };
            if let Some(g) = groups {
                if g >= TOPG {
                    hk::idx_scores(s, ws.iq, ws.hw, *pooled, ws.start, t, g, ws.scores, g)?;
                    k::topk(s, ws.scores, t, g, g, TOPG - 1, false, ws.picked, 0)?;
                }
                hk::idx_expand(s, ws.picked, TOPG - 1, ws.start, t, ws.sel, ws.sel_n)?;
                mode = AttnMode::Selected;
            }
        }
        // Attention over the latent cache.
        let ctas = t * (h / 16);
        // Splits of small steps: about two waves of CTAs, at most one per 32-key tile (sliding: 512 keys,
        // selected: 2048).
        let tiles = if a.sliding { WINDOW / 32 } else { TOPG * GROUP / 32 };
        let splits = if t <= hk::heads_gemv_max_t() {
            (376usize.div_ceil(ctas)).clamp(1, MAX_SPLITS.min(tiles))
        } else {
            1
        };
        // Small steps: 16 heads per block (more blocks for the splits); prefill: all of a token's heads (up
        // to 64) share each key tile (variant hy4.attn_heads).
        let hpb = if t <= hk::heads_gemv_max_t() { 16 } else { quench6k_cuda::variants::get("hy4.attn_heads", 16) as usize };
        hk::mla_attention(
            s, mode, ws.qa, h, cache_rows, ring, ws.sel, ws.sel_n, a.sinks, ws.start, t, splits, hpb, ws.part, ws.olat,
        )?;
        // Expand the output latents (W_uv) and gate them: bf_b [t, h * 256].
        let wuv = a.kv_b + (NOPE * KV_LORA * 2) as u64;
        if t <= hk::heads_gemv_max_t() {
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
        linear(s, ws.bf_b, h * VD, t, &a.o, ws.f_a, D)?;
        hk::rms_rows(s, ws.f_a, D, a.post_norm, D, cfg.eps, ws.y, D, t)
    }

    fn dense(&self, ws: &Workspace, s: &Stream, m: &Dense, t: usize) -> Result<()> {
        let n = m.gate.n;
        linear(s, ws.x, D, t, &m.gate, ws.f_a, n)?;
        linear(s, ws.x, D, t, &m.up, ws.f_b, n)?;
        hk::swiglu(s, ws.f_a, ws.f_b, n, n, 0.0, ws.bf_a, n, t)?;
        linear(s, ws.bf_a, n, t, &m.down, ws.f_a, D)?;
        hk::rms_rows(s, ws.f_a, D, m.norm, D, self.cfg.eps, ws.y, D, t)
    }

    fn moe(&self, ws: &Workspace, s: &Stream, m: &Moe, t: usize) -> Result<()> {
        let cfg = &self.cfg;
        linear(s, ws.x, D, t, &m.router, ws.logits, EXPERTS)?;
        hk::route(s, ws.logits, EXPERTS, m.bias, cfg.scaling, t, ws.sel_e, ws.sel_w)?;
        linear(s, ws.x, D, t, &m.down_lat, ws.f_a, LAT)?;
        k::f32_to_bf16(s, ws.f_a, ws.rin, t * LAT)?;
        let ex = &m.experts;
        if t <= SMALL_T {
            hk::experts_small(s, ws.rin, ws.sel_e, ex.gate, ex.up, ex.down, cfg.swiglu_limit, t, ws.act, ws.ed)?;
            hk::experts_combine(s, ws.ed, 0, ws.sel_w, t, ws.fin)?;
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
                return self.moe_tail(ws, s, m, t);
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
        self.moe_tail(ws, s, m, t)
    }

    /// Back to the model width, plus the shared expert, then the block's norm.
    fn moe_tail(&self, ws: &Workspace, s: &Stream, m: &Moe, t: usize) -> Result<()> {
        let cfg = &self.cfg;
        linear(s, ws.fin, LAT, t, &m.up_lat, ws.f_c, D)?;
        linear(s, ws.x, D, t, &m.shared_gate, ws.f_a, LAT)?;
        linear(s, ws.x, D, t, &m.shared_up, ws.f_b, LAT)?;
        hk::swiglu(s, ws.f_a, ws.f_b, LAT, LAT, 0.0, ws.bf_a, LAT, t)?;
        linear(s, ws.bf_a, LAT, t, &m.shared_down, ws.f_a, D)?;
        hk::add_rows(s, ws.f_c, D, ws.f_a, D, D, t, ws.f_b, D)?;
        hk::rms_rows(s, ws.f_b, D, m.norm, D, cfg.eps, ws.y, D, t)
    }
}
