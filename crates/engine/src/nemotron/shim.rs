//! TEMPORARY: a stand-in for `quench6k_cuda::nemotron` (the Nemotron-H kernels' launchers, written on
//! another branch) with the same types, constants and signatures as its API commit, so this branch compiles
//! before that module is merged. Every launcher fails at run time. At the merge: delete this file and
//! import `quench6k_cuda::nemotron as nk` in forward.rs instead.
//!
//! Expected change, not in that commit yet: the attention cache is BF16 (default) or FP8 with a dynamic
//! scale per (position, KV head) instead of the checkpoint's static scales (which are 1.0); here
//! [`KvRows::fp8`] selects the format and `attn_prep` / `attention_*` take no scales.

#![allow(dead_code)]

use anyhow::{Result, bail};
use quench6k_cuda::Stream;

pub const MAX_SEQ: usize = 16;
pub const HIDDEN: usize = 4096;
pub const HEADS: usize = 128;
pub const HEAD_DIM: usize = 64;
pub const STATE: usize = 128;
pub const GROUPS: usize = 8;
pub const INNER: usize = HEADS * HEAD_DIM;
pub const CONV_DIM: usize = INNER + 2 * GROUPS * STATE;
pub const CONV_TAPS: usize = 4;
pub const PROJ: usize = INNER + CONV_DIM + HEADS;
pub const XBC_COL: usize = INNER;
pub const DT_COL: usize = INNER + CONV_DIM;
pub const NORM_GROUP: usize = INNER / GROUPS;
pub const EXPERTS: usize = 512;
pub const TOPK: usize = 22;
pub const LATENT: usize = 1024;
pub const MOE_INTER: usize = 2688;
pub const SHARED_INTER: usize = 5376;
pub const SMALL_ROWS: usize = 16;
pub const QH: usize = 32;
pub const KVH: usize = 2;
pub const HD: usize = 128;
pub const ATTN_TILE: usize = 64;
pub const ATTN_GROUP: usize = 1024;

/// Sequences of a step (device memory; upload with [`Seqs::as_bytes`]).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Seqs {
    pub mamba: [u64; MAX_SEQ],
    pub kv: [u64; MAX_SEQ],
    pub start: [i32; MAX_SEQ],
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

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MambaW {
    pub conv_w: u64,
    pub conv_b: u64,
    pub dt_bias: u64,
    pub a_log: u64,
    pub d: u64,
    pub norm_w: u64,
    pub dt_min: f32,
    pub dt_max: f32,
}

/// One attention layer's rows in the per-sequence cache: position p of sequence s at byte
/// `Seqs.kv[s] + p * ld + off`.
#[derive(Clone, Copy, Debug)]
pub struct KvRows {
    pub off: usize,
    pub ld: usize,
    /// FP8 codes with a scale per (position, KV head) instead of BF16 (expected API change).
    pub fp8: bool,
}

/// Bytes of one layer's cache row of a position (expected API change): K and V of the 2 heads, BF16, or
/// FP8 codes followed by the four F32 scales.
pub fn kv_layer_row_bytes(fp8: bool) -> usize {
    if fp8 { 2 * KVH * HD + 2 * KVH * 4 } else { 2 * KVH * HD * 2 }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NvExperts {
    pub codes: u64,
    pub scales: u64,
    pub s2: u64,
}

pub const MAMBA_HEADER_BYTES: usize = 256;
pub const SSM_BYTES: usize = HEADS * HEAD_DIM * STATE * 4;
pub const CONV_BYTES: usize = 3 * CONV_DIM * 4;
pub const REC_ROW_BYTES: usize = HEADS * 4 + 2 * CONV_DIM * 2;

pub fn mamba_layer_bytes(rec_rows: usize) -> usize {
    SSM_BYTES + CONV_BYTES + 2 * rec_rows * REC_ROW_BYTES
}

pub fn mamba_seq_bytes(layers: usize, rec_rows: usize) -> usize {
    MAMBA_HEADER_BYTES + layers * mamba_layer_bytes(rec_rows)
}

fn pending(what: &str) -> Result<()> {
    bail!("nemotron: kernel {what} is not merged into this build yet (quench6k_cuda::nemotron)")
}

pub fn check_layout() -> Result<()> {
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct MambaScratch {
    pub xbc: u64,
    pub dt: u64,
    pub y: u64,
}

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
    let _ = (s, proj, ld, rows, seqs, nseq, layer, rec_rows, w, defer, scratch, out);
    pending("mamba_mixer")
}

pub fn mamba_set_ctl(s: &Stream, seqs: u64, nseq: usize, accepted: u64, deferred: bool) -> Result<()> {
    let _ = (s, seqs, nseq, accepted, deferred);
    pending("mamba_set_ctl")
}

#[allow(clippy::too_many_arguments)]
pub fn moe_route(s: &Stream, logits: u64, ld: usize, bias: u64, scaling: f32, rows: usize, sel_e: u64, sel_w: u64) -> Result<()> {
    let _ = (s, logits, ld, bias, scaling, rows, sel_e, sel_w);
    pending("moe_route")
}

#[allow(clippy::too_many_arguments)]
pub fn moe_experts_small(s: &Stream, x: u64, sel_e: u64, up: &NvExperts, down: &NvExperts, rows: usize, order: u64, act: u64, y: u64) -> Result<()> {
    let _ = (s, x, sel_e, up, down, rows, order, act, y);
    pending("moe_experts_small")
}

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
    let _ = (s, sel_e, rows, tile_rows, counts, offsets, cursor, tile_expert, tile_row, num_tiles, pair_row);
    pending("moe_plan")
}

pub fn moe_gather(s: &Stream, x: u64, pair_row: u64, rows: usize, xs: u64) -> Result<()> {
    let _ = (s, x, pair_row, rows, xs);
    pending("moe_gather")
}

#[allow(clippy::too_many_arguments)]
pub fn relu2(s: &Stream, x: u64, ldx: usize, width: usize, rows: usize, out: u64, ldo: usize) -> Result<()> {
    let _ = (s, x, ldx, width, rows, out, ldo);
    pending("relu2")
}

pub fn moe_combine(s: &Stream, y: u64, pair_row: u64, sel_w: u64, rows: usize, out: u64) -> Result<()> {
    let _ = (s, y, pair_row, sel_w, rows, out);
    pending("moe_combine")
}

#[allow(clippy::too_many_arguments)]
pub fn attn_prep(s: &Stream, qkv: u64, ld: usize, q_col: usize, k_col: usize, v_col: usize, rows: usize, seqs: u64, kv: KvRows, q: u64) -> Result<()> {
    let _ = (s, qkv, ld, q_col, k_col, v_col, rows, seqs, kv, q);
    pending("attn_prep")
}

pub fn attn_part_floats(rows: usize, max_groups: usize) -> usize {
    rows * QH * max_groups * (HD + 2)
}

#[allow(clippy::too_many_arguments)]
pub fn attention_small(s: &Stream, q: u64, rows: usize, seqs: u64, kv: KvRows, max_groups: usize, part: u64, out: u64) -> Result<()> {
    let _ = (s, q, rows, seqs, kv, max_groups, part, out);
    pending("attention_small")
}

pub fn attention_prefill(s: &Stream, q: u64, rows: usize, seqs: u64, kv: KvRows, out: u64) -> Result<()> {
    let _ = (s, q, rows, seqs, kv, out);
    pending("attention_prefill")
}
