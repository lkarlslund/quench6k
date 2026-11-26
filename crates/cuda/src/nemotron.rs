//! Launchers for the Nemotron-H kernels (`kernels/nemotron*.cu`): the block RMSNorm over the FP32
//! residual, the Mamba-2 mixer, the latent MoE (sigmoid router, NVFP4 relu² experts) and the GQA
//! attention over a BF16 or FP8 cache. CPU references: [`reference`]. Pointers are raw
//! device addresses (0 for an absent optional input); every launch is asynchronous on the given stream.
//!
//! # Conventions
//!
//! * A step's rows belong to up to [`MAX_SEQ`] sequences, each a run of consecutive rows in position
//!   order, described by a [`Seqs`] descriptor in device memory (`seqs` arguments): sequence `s` has rows
//!   `row0[s] .. row0[s + 1]` at positions `start[s] ..`, its Mamba state block at `mamba[s]` and its
//!   attention cache at `kv[s]`. Prefill is one sequence with many rows; a decode / verify step has
//!   1 + drafts rows per sequence. The host-side `nseq` argument only sizes grids: pass the descriptor's
//!   `nseq` or any larger bound (e.g. [`MAX_SEQ`]; blocks of absent sequences exit).
//! * Projection outputs are read as FP32 rows (what the GEMV/GEMM kernels write) and rounded to BF16
//!   on read wherever the checkpoint's model rounds a BF16 `Linear` output (in_proj's z / xBC / dt, the
//!   attention q / k / v). Every result row depends only on its own sequence's inputs and state, never on
//!   the other rows or sequences of the launch (batch invariance).
//! * Small parameter vectors (conv weight and bias, dt_bias, A_log, D, the gated norm's weight) are FP32
//!   in device memory; the loader converts the checkpoint's BF16/FP32 tensors (exact).
//!
//! # Mamba-2 mixer (see `scratch/API.md` for the full walk-through)
//!
//! in_proj output row (18560 wide, [`PROJ`]): `z` [8192] at column 0, `xBC` [10240] at [`XBC_COL`]
//! (x [8192], then B [8 groups x 128], then C [8 x 128]), `dt` [128] at [`DT_COL`]. Per layer:
//! [`mamba_conv`] -> [`mamba_conv_commit`] -> [`mamba_scan`] -> [`mamba_gated_norm`] (or [`mamba_mixer`]
//! for all four), then out_proj on the gated norm's BF16 output.
//!
//! Speculative steps (`defer = true`) leave the state at the start of the step and record the step's
//! rows; [`mamba_set_ctl`] (after verification, once per step, not per layer) stores how many of them were
//! accepted, and the next Mamba launch of each layer replays (commits) those rows before its own rows.
//! Plain steps and prefill use `defer = false` (everything committed). [`mamba_flush`] commits pending
//! rows without new ones (before copying a state, e.g. to a prefix cache).
//!
//! The scan is the exact sequential recurrence for every row count (no chunked SSD form): prefill and
//! decode give the same bits for a token, so any prefill split is canonical.

use std::ffi::{c_int, c_void};

use anyhow::Result;

use crate::{Stream, check};

pub mod reference;

/// Most sequences in one step's descriptor.
pub const MAX_SEQ: usize = 16;

// ---- Model constants -------------------------------------------------------------------------------

pub const HIDDEN: usize = 4096;
/// Mamba heads, channels per head, state size per head and channel, B/C groups (16 heads each).
pub const HEADS: usize = 128;
pub const HEAD_DIM: usize = 64;
pub const STATE: usize = 128;
pub const GROUPS: usize = 8;
/// Mamba inner width (heads x head_dim) and the convolved width (x, B, C).
pub const INNER: usize = HEADS * HEAD_DIM;
pub const CONV_DIM: usize = INNER + 2 * GROUPS * STATE;
pub const CONV_TAPS: usize = 4;
/// in_proj output width and the column of each part (z at 0).
pub const PROJ: usize = INNER + CONV_DIM + HEADS;
pub const XBC_COL: usize = INNER;
pub const DT_COL: usize = INNER + CONV_DIM;
/// Gated RMSNorm group width (INNER / GROUPS).
pub const NORM_GROUP: usize = INNER / GROUPS;

/// Routed experts, experts per token, latent width, routed expert intermediate width.
pub const EXPERTS: usize = 512;
pub const TOPK: usize = 22;
pub const LATENT: usize = 1024;
pub const MOE_INTER: usize = 2688;
/// Shared expert intermediate width.
pub const SHARED_INTER: usize = 5376;
/// Most rows of [`moe_experts_small`].
pub const SMALL_ROWS: usize = 16;

/// Attention: query heads, KV heads, head dim.
pub const QH: usize = 32;
pub const KVH: usize = 2;
pub const HD: usize = 128;
/// Bytes of one layer's cache row of a position: BF16 K [2][128] then V [2][128].
pub const KV_ROW_BYTES_BF16: usize = 2 * KVH * HD * 2;
/// Bytes of one layer's FP8 cache row of a position: E4M3 K [2][128], E4M3 V [2][128], then FP32 scales
/// k [2], v [2] (amax / 448 of the head's 128 BF16-rounded values; 1 when all are zero).
pub const KV_ROW_BYTES_FP8: usize = 2 * KVH * HD + 2 * KVH * 4;
/// Attention key tile (positions) and decode split group (positions); both absolute-aligned.
pub const ATTN_TILE: usize = 64;
pub const ATTN_GROUP: usize = 1024;

// ---- Descriptors -----------------------------------------------------------------------------------

/// Sequences of a step (device memory; upload with [`Seqs::as_bytes`]).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Seqs {
    /// Base of each sequence's Mamba state block ([`mamba_seq_bytes`]).
    pub mamba: [u64; MAX_SEQ],
    /// Base of each sequence's attention cache (position-major rows, see [`KvRows`]).
    pub kv: [u64; MAX_SEQ],
    /// Position of each sequence's first row in this step.
    pub start: [i32; MAX_SEQ],
    /// Rows of sequence s: row0[s] .. row0[s + 1] (row0[0] = 0).
    pub row0: [i32; MAX_SEQ + 1],
    pub nseq: i32,
}

impl Default for Seqs {
    fn default() -> Self {
        Self { mamba: [0; MAX_SEQ], kv: [0; MAX_SEQ], start: [0; MAX_SEQ], row0: [0; MAX_SEQ + 1], nseq: 0 }
    }
}

impl Seqs {
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts((self as *const Self).cast::<u8>(), std::mem::size_of::<Self>()) }
    }
}

/// One Mamba layer's parameters (FP32 device vectors) and the dt limits.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MambaW {
    /// conv1d.weight as FP32 [CONV_DIM][4] (tap 3 multiplies the current row) and conv1d.bias [CONV_DIM].
    pub conv_w: u64,
    pub conv_b: u64,
    /// dt_bias, A_log, D: FP32 [HEADS] each (A = -exp(A_log) is computed by the kernels).
    pub dt_bias: u64,
    pub a_log: u64,
    pub d: u64,
    /// The gated RMSNorm's weight, FP32 [INNER].
    pub norm_w: u64,
    /// dt = clamp(softplus(dt + dt_bias), dt_min, dt_max). The checkpoint's time_step_limit is (0, inf):
    /// pass 0.0 and f32::INFINITY (the fused CUDA path of the reference model; its torch-only fallback
    /// clamps at time_step_min = 0.001 instead).
    pub dt_min: f32,
    pub dt_max: f32,
}

/// One attention layer's rows in the per-sequence cache: position p of sequence s at byte
/// `Seqs.kv[s] + p * ld + off` (16-byte aligned); the row is [`KV_ROW_BYTES_BF16`] or, with `fp8`,
/// [`KV_ROW_BYTES_FP8`] bytes. E.g. ld = 9 rows (8 layers + MTP), off = layer * row bytes.
#[derive(Clone, Copy, Debug)]
pub struct KvRows {
    pub off: usize,
    pub ld: usize,
    pub fp8: bool,
}

/// One projection kind of a layer's routed experts, contiguous over the 512 experts: codes U8
/// [512][N][K / 2] (E2M1 pairs, low nibble first), scales E4M3 [512][N][K / 16], s2 FP32 [512]
/// (weight_scale_2). Up: N = 2688, K = 1024; down: N = 1024, K = 2688.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NvExperts {
    pub codes: u64,
    pub scales: u64,
    pub s2: u64,
}

// ---- Mamba state block layout ----------------------------------------------------------------------

/// Bytes of the per-sequence header at the start of the Mamba block: i32 replay count, i32 record
/// parity (see [`mamba_set_ctl`]), padding.
pub const MAMBA_HEADER_BYTES: usize = 256;
/// SSM state of one layer: FP32 [HEADS][HEAD_DIM][STATE] (the reference cache's layout).
pub const SSM_BYTES: usize = HEADS * HEAD_DIM * STATE * 4;
/// Conv state of one layer: FP32 [3][CONV_DIM], the last three raw (pre-conv, BF16-rounded) xBC rows,
/// oldest first.
pub const CONV_BYTES: usize = 3 * CONV_DIM * 4;
/// One record row: dt after softplus FP32 [HEADS], raw xBC BF16 [CONV_DIM], convolved xBC BF16 [CONV_DIM].
pub const REC_ROW_BYTES: usize = HEADS * 4 + 2 * CONV_DIM * 2;

/// Bytes of one layer's part of a Mamba block with records for `rec_rows` rows per step (two parities):
/// SSM state, conv state, records. `rec_rows` = the most rows of a deferred step (0: no deferred steps).
pub fn mamba_layer_bytes(rec_rows: usize) -> usize {
    SSM_BYTES + CONV_BYTES + 2 * rec_rows * REC_ROW_BYTES
}

/// Bytes of a sequence's Mamba block: header, then `layers` layer parts (Mamba layer i of the model at
/// `MAMBA_HEADER_BYTES + i * mamba_layer_bytes(rec_rows)`). A new sequence's block must be zeroed.
pub fn mamba_seq_bytes(layers: usize, rec_rows: usize) -> usize {
    MAMBA_HEADER_BYTES + layers * mamba_layer_bytes(rec_rows)
}

mod ffi {
    use super::*;
    use crate::ffi::{Error, Stream};

    unsafe extern "C" {
        pub fn quench6k_nem_seqs_bytes() -> usize;
        pub fn quench6k_nem_rms(
            x: *const f32,
            ldx: c_int,
            w: *const c_void,
            eps: f32,
            rows: c_int,
            out: *mut c_void,
            ldo: c_int,
            f32_out: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_nem_mamba_conv(
            proj: *const f32,
            ld: c_int,
            rows: c_int,
            seqs: *const c_void,
            nseq: c_int,
            layer: c_int,
            rec_rows: c_int,
            w: MambaW,
            xbc: *mut c_void,
            dt: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_nem_mamba_conv_commit(
            proj: *const f32,
            ld: c_int,
            rows: c_int,
            seqs: *const c_void,
            nseq: c_int,
            layer: c_int,
            rec_rows: c_int,
            xbc: *const c_void,
            dt: *const f32,
            defer: c_int,
            flush: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_nem_mamba_scan(
            xbc: *const c_void,
            dt: *const f32,
            rows: c_int,
            seqs: *const c_void,
            nseq: c_int,
            layer: c_int,
            rec_rows: c_int,
            w: MambaW,
            defer: c_int,
            flush: c_int,
            y: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_nem_mamba_gated_norm(
            y: *const c_void,
            proj: *const f32,
            ld: c_int,
            rows: c_int,
            norm_w: *const f32,
            eps: f32,
            out: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_nem_mamba_set_ctl(seqs: *const c_void, nseq: c_int, accepted: *const i32, deferred: c_int, st: Stream) -> Error;

        pub fn quench6k_nem_moe_route(
            logits: *const f32,
            ld: c_int,
            bias: *const f32,
            scaling: f32,
            rows: c_int,
            sel_e: *mut i32,
            sel_w: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_nem_moe_experts_small(
            x: *const c_void,
            sel_e: *const i32,
            up: NvExperts,
            down: NvExperts,
            rows: c_int,
            order: *mut i32,
            act: *mut c_void,
            y: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_nem_moe_plan(
            sel_e: *const i32,
            rows: c_int,
            tile_rows: c_int,
            counts: *mut i32,
            offsets: *mut i32,
            cursor: *mut i32,
            tile_expert: *mut i32,
            tile_row: *mut i32,
            num_tiles: *mut i32,
            pair_row: *mut i32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_nem_moe_prefill_tile_rows() -> c_int;
        pub fn quench6k_nem_moe_experts_prefill(
            xs: *const c_void,
            up: NvExperts,
            down: NvExperts,
            offsets: *const i32,
            tile_expert: *const i32,
            tile_row: *const i32,
            num_tiles: *const i32,
            max_tiles: c_int,
            act: *mut c_void,
            y: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_nem_moe_combine_bf16(y: *const c_void, pair_row: *const i32, sel_w: *const f32, rows: c_int, out: *mut c_void, st: Stream) -> Error;
        pub fn quench6k_nem_moe_gather(x: *const c_void, pair_row: *const i32, rows: c_int, xs: *mut c_void, st: Stream) -> Error;
        pub fn quench6k_nem_relu2(
            x: *const f32,
            ldx: c_int,
            width: c_int,
            rows: c_int,
            out: *mut c_void,
            ldo: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_nem_moe_combine(
            y: *const f32,
            pair_row: *const i32,
            sel_w: *const f32,
            rows: c_int,
            out: *mut c_void,
            st: Stream,
        ) -> Error;

        pub fn quench6k_nem_attn_prep(
            qkv: *const f32,
            ld: c_int,
            q_col: c_int,
            k_col: c_int,
            v_col: c_int,
            rows: c_int,
            seqs: *const c_void,
            kv_off: i64,
            kv_ld: i64,
            fp8: c_int,
            q: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_nem_attn_part_floats(rows: c_int, max_groups: c_int) -> usize;
        pub fn quench6k_nem_attention_small(
            q: *const c_void,
            rows: c_int,
            seqs: *const c_void,
            kv_off: i64,
            kv_ld: i64,
            fp8: c_int,
            max_groups: c_int,
            part: *mut f32,
            out: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_nem_attention_prefill(
            q: *const c_void,
            rows: c_int,
            seqs: *const c_void,
            kv_off: i64,
            kv_ld: i64,
            fp8: c_int,
            out: *mut c_void,
            st: Stream,
        ) -> Error;
    }
}

fn p(a: u64) -> *const c_void {
    a as *const c_void
}
fn pm(a: u64) -> *mut c_void {
    a as *mut c_void
}
fn f(a: u64) -> *const f32 {
    a as *const f32
}
fn fm(a: u64) -> *mut f32 {
    a as *mut f32
}
fn i(a: u64) -> *const i32 {
    a as *const i32
}
fn im(a: u64) -> *mut i32 {
    a as *mut i32
}
fn n(v: usize) -> c_int {
    v as c_int
}

macro_rules! go {
    ($name:literal, $call:expr) => {
        check(unsafe { $call }, $name)
    };
}

/// Checks that the kernels' [`Seqs`] layout matches this one.
pub fn check_layout() -> Result<()> {
    let b = unsafe { ffi::quench6k_nem_seqs_bytes() };
    anyhow::ensure!(b == std::mem::size_of::<Seqs>(), "nemotron: sequence descriptor layout mismatch ({b} bytes)");
    Ok(())
}

// ---- Block norms -----------------------------------------------------------------------------------

/// RMSNorm of `rows` FP32 residual rows ([`HIDDEN`] wide, stride `ldx`) with the BF16 weight `w` [4096]:
/// `out = w * (x * (1 / sqrt(mean(x²) + eps)))`, all FP32 (fixed-order sum), rounded once to BF16
/// (`f32_out` = false) or written as FP32. eps: 1e-5. One CTA per row (rows independent).
#[allow(clippy::too_many_arguments)]
pub fn rms(s: &Stream, x: u64, ldx: usize, w: u64, eps: f32, rows: usize, out: u64, ldo: usize, f32_out: bool) -> Result<()> {
    go!(
        "nem_rms",
        ffi::quench6k_nem_rms(f(x), n(ldx), p(w), eps, n(rows), pm(out), n(ldo), f32_out as c_int, s.raw())
    )
}

// ---- Mamba-2 ---------------------------------------------------------------------------------------

/// Causal depthwise conv (4 taps, bias, SiLU) of xBC and dt = clamp(softplus(dt + dt_bias)) for the
/// step's rows of Mamba layer `layer`.
///
/// `proj`: in_proj output FP32 [rows, ld] (xBC at [`XBC_COL`], dt at [`DT_COL`]; values rounded to BF16
/// on read). The history before a sequence's first row is its conv state plus its pending (replayed)
/// records. Outputs: `xbc` BF16 [rows, CONV_DIM] (the convolved x, B, C, rounded to BF16 as the model
/// does) and `dt` FP32 [rows, HEADS]. Reads the state; writes nothing in it ([`mamba_conv_commit`] does).
#[allow(clippy::too_many_arguments)]
pub fn mamba_conv(
    s: &Stream,
    proj: u64,
    ld: usize,
    rows: usize,
    seqs: u64,
    nseq: usize,
    layer: usize,
    rec_rows: usize,
    w: &MambaW,
    xbc: u64,
    dt: u64,
) -> Result<()> {
    go!(
        "nem_mamba_conv",
        ffi::quench6k_nem_mamba_conv(f(proj), n(ld), n(rows), p(seqs), n(nseq), n(layer), n(rec_rows), *w, pm(xbc), fm(dt), s.raw())
    )
}

/// Advances layer `layer`'s conv state after [`mamba_conv`] (same arguments): replays the pending
/// records, then with `defer` keeps the state there and records this step's rows (raw xBC, convolved xBC
/// and dt) in the other record parity; without `defer` moves the state past this step's rows.
#[allow(clippy::too_many_arguments)]
pub fn mamba_conv_commit(
    s: &Stream,
    proj: u64,
    ld: usize,
    rows: usize,
    seqs: u64,
    nseq: usize,
    layer: usize,
    rec_rows: usize,
    xbc: u64,
    dt: u64,
    defer: bool,
) -> Result<()> {
    go!(
        "nem_mamba_conv_commit",
        ffi::quench6k_nem_mamba_conv_commit(
            f(proj),
            n(ld),
            n(rows),
            p(seqs),
            n(nseq),
            n(layer),
            n(rec_rows),
            p(xbc),
            f(dt),
            defer as c_int,
            0,
            s.raw()
        )
    )
}

/// The selective-state recurrence of layer `layer` (after [`mamba_conv_commit`]): per sequence, replays
/// its pending records into the SSM state, then for each of its rows in order
/// `S[h,p,n] = S * exp(dt[h] * A[h]) + (B[g(h),n] * dt[h]) * x[h,p]`, `y[h,p] = sum_n S[h,p,n] * C[g(h),n]
/// + D[h] * x[h,p]` (FP32 state, y rounded to BF16). With `defer` the state is stored after the replay
/// (this step's rows are not committed), otherwise after the last row.
/// Inputs `xbc` BF16 [rows, CONV_DIM], `dt` FP32 [rows, HEADS]; output `y` BF16 [rows, INNER].
#[allow(clippy::too_many_arguments)]
pub fn mamba_scan(
    s: &Stream,
    xbc: u64,
    dt: u64,
    rows: usize,
    seqs: u64,
    nseq: usize,
    layer: usize,
    rec_rows: usize,
    w: &MambaW,
    defer: bool,
    y: u64,
) -> Result<()> {
    go!(
        "nem_mamba_scan",
        ffi::quench6k_nem_mamba_scan(
            p(xbc),
            f(dt),
            n(rows),
            p(seqs),
            n(nseq),
            n(layer),
            n(rec_rows),
            *w,
            defer as c_int,
            0,
            pm(y),
            s.raw()
        )
    )
}

/// Gated RMSNorm (norm_before_gate = False) per group of [`NORM_GROUP`] channels:
/// `v = y * silu(z)`, `out = v * rsqrt(mean_group(v²) + eps) * norm_w`, computed in FP32, out BF16
/// [rows, INNER] (out_proj's input). `y` BF16 [rows, INNER]; z = proj[:, 0..INNER] (rounded to BF16 on
/// read). eps: layer_norm_epsilon (1e-5).
#[allow(clippy::too_many_arguments)]
pub fn mamba_gated_norm(s: &Stream, y: u64, proj: u64, ld: usize, rows: usize, norm_w: u64, eps: f32, out: u64) -> Result<()> {
    go!(
        "nem_mamba_gated_norm",
        ffi::quench6k_nem_mamba_gated_norm(p(y), f(proj), n(ld), n(rows), f(norm_w), eps, pm(out), s.raw())
    )
}

/// Scratch of [`mamba_mixer`] for `rows` rows: xbc BF16 [rows, CONV_DIM], dt FP32 [rows, HEADS],
/// y BF16 [rows, INNER].
#[derive(Clone, Copy, Debug)]
pub struct MambaScratch {
    pub xbc: u64,
    pub dt: u64,
    pub y: u64,
}

/// The whole mixer between in_proj and out_proj: [`mamba_conv`], [`mamba_conv_commit`], [`mamba_scan`],
/// [`mamba_gated_norm`] (eps 1e-5). `out` BF16 [rows, INNER].
#[allow(clippy::too_many_arguments)]
pub fn mamba_mixer(
    s: &Stream,
    proj: u64,
    ld: usize,
    rows: usize,
    seqs: u64,
    nseq: usize,
    layer: usize,
    rec_rows: usize,
    w: &MambaW,
    defer: bool,
    scratch: &MambaScratch,
    out: u64,
) -> Result<()> {
    mamba_conv(s, proj, ld, rows, seqs, nseq, layer, rec_rows, w, scratch.xbc, scratch.dt)?;
    mamba_conv_commit(s, proj, ld, rows, seqs, nseq, layer, rec_rows, scratch.xbc, scratch.dt, defer)?;
    mamba_scan(s, scratch.xbc, scratch.dt, rows, seqs, nseq, layer, rec_rows, w, defer, scratch.y)?;
    mamba_gated_norm(s, scratch.y, proj, ld, rows, w.norm_w, 1e-5, out)
}

/// Commits layer `layer`'s pending records of every sequence in `seqs` (rows ignored: no new rows).
/// Run for every Mamba layer, then `mamba_set_ctl(.., deferred = false)`.
pub fn mamba_flush(s: &Stream, seqs: u64, nseq: usize, layer: usize, rec_rows: usize, w: &MambaW) -> Result<()> {
    go!(
        "nem_mamba_flush_conv",
        ffi::quench6k_nem_mamba_conv_commit(f(0), 0, 0, p(seqs), n(nseq), n(layer), n(rec_rows), p(0), f(0), 0, 1, s.raw())
    )?;
    go!(
        "nem_mamba_flush_scan",
        ffi::quench6k_nem_mamba_scan(p(0), f(0), 0, p(seqs), n(nseq), n(layer), n(rec_rows), *w, 0, 1, pm(0), s.raw())
    )
}

/// Once per step after every layer ran (and after verification): for each sequence s < nseq of `seqs`,
/// with `deferred` (the step ran with defer = true) the next launches replay `accepted[s]` rows of this
/// step (device i32 [nseq], 1 ..= the sequence's rows; 0 = all its rows) and the record parity flips;
/// without `deferred` nothing is pending (replay count 0).
pub fn mamba_set_ctl(s: &Stream, seqs: u64, nseq: usize, accepted: u64, deferred: bool) -> Result<()> {
    go!(
        "nem_mamba_set_ctl",
        ffi::quench6k_nem_mamba_set_ctl(p(seqs), n(nseq), i(accepted), deferred as c_int, s.raw())
    )
}

// ---- MoE -------------------------------------------------------------------------------------------

/// Router: score = sigmoid(logits) (FP32 [rows, ld], 512 used); the 22 experts with the largest
/// score + bias (FP32 [512], e_score_correction_bias; ties: lower expert) in ascending expert order to
/// `sel_e` i32 [rows, 22]; their weights score / (sum of the 22 scores + 1e-20) * scaling (5.0) to
/// `sel_w` FP32 [rows, 22].
#[allow(clippy::too_many_arguments)]
pub fn moe_route(s: &Stream, logits: u64, ld: usize, bias: u64, scaling: f32, rows: usize, sel_e: u64, sel_w: u64) -> Result<()> {
    go!(
        "nem_moe_route",
        ffi::quench6k_nem_moe_route(f(logits), n(ld), f(bias), scaling, n(rows), im(sel_e), fm(sel_w), s.raw())
    )
}

/// Routed experts for small steps (rows <= [`SMALL_ROWS`]), GEMV-style: `x` latent input BF16
/// [rows, 1024] (fc1_latent_proj's output, BF16-rounded); pair = row * 22 + k;
/// `act[pair]` = bf16(relu(bf16(up_e · x))²) BF16 [rows * 22, 2688]; `y[pair]` = down_e · act FP32
/// [rows * 22, 1024]. `order` i32 [rows * 22] scratch. Combine with [`moe_combine`] (pair_row = 0).
#[allow(clippy::too_many_arguments)]
pub fn moe_experts_small(
    s: &Stream,
    x: u64,
    sel_e: u64,
    up: &NvExperts,
    down: &NvExperts,
    rows: usize,
    order: u64,
    act: u64,
    y: u64,
) -> Result<()> {
    go!(
        "nem_moe_experts_small",
        ffi::quench6k_nem_moe_experts_small(p(x), i(sel_e), *up, *down, n(rows), im(order), pm(act), fm(y), s.raw())
    )
}

/// Prefill grouping of the rows * 22 (row, k) pairs by expert into tiles of `tile_rows` rows for
/// `kernels::moe_nvfp4_gemm` ([`crate::kernels::moe_tile_rows`]). Scratch: counts / cursor i32 [512],
/// offsets i32 [513], tile_expert / tile_row i32 [rows * 22 / tile_rows + 512], num_tiles i32 [1],
/// pair_row i32 [rows * 22] (the expert-grouped row of each pair).
#[allow(clippy::too_many_arguments)]
pub fn moe_plan(
    s: &Stream,
    sel_e: u64,
    rows: usize,
    tile_rows: usize,
    counts: u64,
    offsets: u64,
    cursor: u64,
    tile_expert: u64,
    tile_row: u64,
    num_tiles: u64,
    pair_row: u64,
) -> Result<()> {
    go!(
        "nem_moe_plan",
        ffi::quench6k_nem_moe_plan(
            i(sel_e),
            n(rows),
            n(tile_rows),
            im(counts),
            im(offsets),
            im(cursor),
            im(tile_expert),
            im(tile_row),
            im(num_tiles),
            im(pair_row),
            s.raw()
        )
    )
}

/// Tile rows of [`moe_experts_prefill`] (pass to [`moe_plan`] as `tile_rows` for that path).
pub fn moe_prefill_tile_rows() -> usize {
    unsafe { ffi::quench6k_nem_moe_prefill_tile_rows() as usize }
}

/// Routed experts of a prefill step (replaces the two `kernels::moe_nvfp4_gemm` calls and [`relu2`]):
/// after [`moe_plan`] with `tile_rows` = [`moe_prefill_tile_rows`] and [`moe_gather`] into `xs` (BF16
/// [pairs, 1024]), `act` = relu²(up) BF16 [pairs, 2688] (fused epilogue) and `y` = bf16(down) BF16
/// [pairs, 1024]; combine with [`moe_combine_bf16`]. Uses the contiguous expert arrays (no pointer tables).
/// Bit-identical to the unfused path. `max_tiles` as for the GEMM (pairs / tile_rows + 512).
#[allow(clippy::too_many_arguments)]
pub fn moe_experts_prefill(
    s: &Stream,
    xs: u64,
    up: &NvExperts,
    down: &NvExperts,
    offsets: u64,
    tile_expert: u64,
    tile_row: u64,
    num_tiles: u64,
    max_tiles: usize,
    act: u64,
    y: u64,
) -> Result<()> {
    go!(
        "nem_moe_experts_prefill",
        ffi::quench6k_nem_moe_experts_prefill(
            p(xs),
            *up,
            *down,
            i(offsets),
            i(tile_expert),
            i(tile_row),
            i(num_tiles),
            n(max_tiles),
            pm(act),
            pm(y),
            s.raw()
        )
    )
}

/// [`moe_combine`] reading BF16 expert outputs `y` [pairs, 1024] (as [`moe_experts_prefill`] writes them).
pub fn moe_combine_bf16(s: &Stream, y: u64, pair_row: u64, sel_w: u64, rows: usize, out: u64) -> Result<()> {
    go!(
        "nem_moe_combine_bf16",
        ffi::quench6k_nem_moe_combine_bf16(p(y), i(pair_row), f(sel_w), n(rows), pm(out), s.raw())
    )
}

/// xs[pair_row[pair]] = x[pair / 22] (BF16 rows of 1024).
pub fn moe_gather(s: &Stream, x: u64, pair_row: u64, rows: usize, xs: u64) -> Result<()> {
    go!("nem_moe_gather", ffi::quench6k_nem_moe_gather(p(x), i(pair_row), n(rows), pm(xs), s.raw()))
}

/// out[r, i] = bf16(relu(bf16(x[r * ldx + i]))²) for `rows` rows of `width` (the relu² of a BF16 Linear
/// output, as the model rounds it): routed experts in prefill, the shared expert always.
#[allow(clippy::too_many_arguments)]
pub fn relu2(s: &Stream, x: u64, ldx: usize, width: usize, rows: usize, out: u64, ldo: usize) -> Result<()> {
    go!("nem_relu2", ffi::quench6k_nem_relu2(f(x), n(ldx), n(width), n(rows), pm(out), n(ldo), s.raw()))
}

/// out[r, n] = bf16(sum over k in ascending expert order of bf16(y[row(r, k), n]) * sel_w[r, k]) (FP32
/// sum, as the reference's index_add over experts), BF16 [rows, 1024] for fc2_latent_proj. `y` FP32
/// [pairs, 1024]; row(r, k) = pair_row[r * 22 + k], or r * 22 + k when `pair_row` is 0.
pub fn moe_combine(s: &Stream, y: u64, pair_row: u64, sel_w: u64, rows: usize, out: u64) -> Result<()> {
    go!(
        "nem_moe_combine",
        ffi::quench6k_nem_moe_combine(f(y), i(pair_row), f(sel_w), n(rows), pm(out), s.raw())
    )
}

// ---- Attention -------------------------------------------------------------------------------------

/// Attention inputs of the step's rows: q = bf16(qkv[r, q_col ..]) BF16 [rows, 4096]; k and v
/// ([2][128] each at k_col / v_col, rounded to BF16) into the cache row of the row's position (start + row
/// offset in its sequence): BF16, or with `kv.fp8` E4M3(x / scale), scale = amax / 448 per (position, KV
/// head) (1 if all zero), as Qwen's and Flint's FP8 caches. No rotary embedding, no q/k norm. The
/// checkpoint's static k_scale / v_scale (1.0) are not used.
#[allow(clippy::too_many_arguments)]
pub fn attn_prep(s: &Stream, qkv: u64, ld: usize, q_col: usize, k_col: usize, v_col: usize, rows: usize, seqs: u64, kv: KvRows, q: u64) -> Result<()> {
    go!(
        "nem_attn_prep",
        ffi::quench6k_nem_attn_prep(
            f(qkv),
            n(ld),
            n(q_col),
            n(k_col),
            n(v_col),
            n(rows),
            p(seqs),
            kv.off as i64,
            kv.ld as i64,
            kv.fp8 as c_int,
            pm(q),
            s.raw()
        )
    )
}

/// Floats of [`attention_small`]'s partial results for `rows` rows whose positions are below
/// `max_groups * ATTN_GROUP`.
pub fn attn_part_floats(rows: usize, max_groups: usize) -> usize {
    unsafe { ffi::quench6k_nem_attn_part_floats(n(rows), n(max_groups)) }
}

/// Causal GQA attention (32 q heads over 2 KV heads, head dim 128, scale 1/sqrt(128)) of decode / verify
/// rows over their sequences' caches (row at position p sees positions 0 ..= p, after [`attn_prep`]).
/// `q` BF16 [rows, 4096]; `out` BF16 [rows, 4096] (o_proj's input). Every row's result is the canonical
/// one: online softmax over 64-position key tiles within absolute 1024-position groups, groups folded in
/// order; P (times the value's FP8 scale) rounded to BF16 for P·V as flash attention does. `max_groups`
/// bounds ceil((p + 1) / 1024) over the rows (it sizes the grid and `part`, [`attn_part_floats`]).
#[allow(clippy::too_many_arguments)]
pub fn attention_small(s: &Stream, q: u64, rows: usize, seqs: u64, kv: KvRows, max_groups: usize, part: u64, out: u64) -> Result<()> {
    go!(
        "nem_attention_small",
        ffi::quench6k_nem_attention_small(
            p(q),
            n(rows),
            p(seqs),
            kv.off as i64,
            kv.ld as i64,
            kv.fp8 as c_int,
            n(max_groups),
            fm(part),
            pm(out),
            s.raw()
        )
    )
}

/// As [`attention_small`] for prefill rows (any count; rows of one sequence share key tiles; no scratch):
/// bit-identical per row to [`attention_small`].
pub fn attention_prefill(s: &Stream, q: u64, rows: usize, seqs: u64, kv: KvRows, out: u64) -> Result<()> {
    go!(
        "nem_attention_prefill",
        ffi::quench6k_nem_attention_prefill(p(q), n(rows), p(seqs), kv.off as i64, kv.ld as i64, kv.fp8 as c_int, pm(out), s.raw())
    )
}

mod gemm_ext_ffi {
    use std::ffi::{c_int, c_void};

    use crate::ffi::{Error, Stream};

    unsafe extern "C" {
        pub fn quench6k_gemm_bf16_f32_ext(
            fmt: c_int,
            x: *const c_void,
            ldx: c_int,
            w: *const c_void,
            ldw: c_int,
            scale: *const f32,
            y: *mut f32,
            ldy: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            accumulate: c_int,
            flags: c_int,
            stream: Stream,
        ) -> Error;
    }
}

/// Prefill projection Y [m, n] F32 (+)= alpha X [m, k] BF16 . W^T (weight formats BF16, INT8 per 64, FP8
/// per 128 x 128 block): the tensor-core GEMM of [`crate::kernels::gemm_f32`] (the same bits); `flags` bit 1
/// takes 64-row tiles when n <= 1024.
#[allow(clippy::too_many_arguments)]
pub fn gemm_f32_ext(
    s: &Stream,
    x: u64,
    ldx: usize,
    fmt: crate::kernels::WeightFormat,
    w: u64,
    ldw: usize,
    scale: u64,
    y: u64,
    ldy: usize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    accumulate: bool,
    flags: u32,
) -> Result<()> {
    let err = unsafe {
        gemm_ext_ffi::quench6k_gemm_bf16_f32_ext(
            fmt as i32,
            x as *const std::ffi::c_void,
            ldx as i32,
            w as *const std::ffi::c_void,
            ldw as i32,
            if scale == 0 { std::ptr::null() } else { scale as *const f32 },
            y as *mut f32,
            ldy as i32,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            accumulate as i32,
            flags as i32,
            s.raw(),
        )
    };
    crate::check(err, "gemm_bf16_f32_ext")
}

