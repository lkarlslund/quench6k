//! Typed launchers for squash's CUDA kernels. Pointers are raw device addresses (`u64`); every
//! launch is asynchronous on the given stream.

use std::ffi::c_void;

use anyhow::Result;

use crate::{Stream, check, ffi};

fn p(addr: u64) -> *const c_void {
    addr as *const c_void
}

fn pm(addr: u64) -> *mut c_void {
    addr as *mut c_void
}

/// Projection weight storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFormat {
    Bf16 = 0,
    /// FP8 E4M3 with a per-row FP32 scale.
    Fp8 = 1,
    /// INT8 with a per-row FP32 scale.
    Int8 = 2,
    /// INT8 with an FP32 scale per [`int8_group`] consecutive weights of a row.
    Int8Group = 3,
}

pub fn int8_group() -> usize {
    unsafe { ffi::squash_int8_group() as usize }
}

/// BF16 [N, K] -> INT8 [N, K] with scales [N, K / group] = group absmax / 127.
pub fn quantize_int8_groups(s: &Stream, w: u64, n: usize, k: usize, q: u64, scale: u64) -> Result<()> {
    check(unsafe { ffi::squash_quantize_int8_groups(p(w), n as i32, k as i32, pm(q), scale as *mut f32, s.raw()) }, "quantize_int8_groups")
}

fn opt_f(addr: u64) -> *const f32 {
    if addr == 0 { std::ptr::null() } else { addr as *const f32 }
}

/// Tensor-core GEMM: Y[M, N] = alpha · X[M, K] · W[N, K]ᵀ (or `+=` with `accumulate`), BF16
/// activations, FP32 output; 8-bit weight formats use the per-row scales at `scale`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_f32(s: &Stream, x: u64, ldx: usize, fmt: WeightFormat, w: u64, ldw: usize, scale: u64, y: u64, ldy: usize, m: usize, n: usize, k: usize, alpha: f32, accumulate: bool) -> Result<()> {
    let err = unsafe { ffi::squash_gemm_bf16_f32(fmt as i32, p(x), ldx as i32, p(w), ldw as i32, opt_f(scale), y as *mut f32, ldy as i32, m as i32, n as i32, k as i32, alpha, accumulate as i32, s.raw()) };
    check(err, "gemm_bf16_f32")
}

/// As [`gemm_f32`] with a BF16 output.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16(s: &Stream, x: u64, ldx: usize, fmt: WeightFormat, w: u64, ldw: usize, scale: u64, y: u64, ldy: usize, m: usize, n: usize, k: usize, alpha: f32, accumulate: bool) -> Result<()> {
    let err = unsafe { ffi::squash_gemm_bf16_bf16(fmt as i32, p(x), ldx as i32, p(w), ldw as i32, opt_f(scale), pm(y), ldy as i32, m as i32, n as i32, k as i32, alpha, accumulate as i32, s.raw()) };
    check(err, "gemm_bf16_bf16")
}

pub fn gemv_max_m() -> usize {
    unsafe { ffi::squash_gemv_max_m() as usize }
}

/// Skinny GEMM for M <= 16 rows, FP32 output; formats as in [`gemm_f32`]. With `splits` > 1 the K
/// range is split across CTAs and `partial` must hold `splits * M * N` floats.
#[allow(clippy::too_many_arguments)]
pub fn gemv(s: &Stream, m: usize, x: u64, ldx: usize, fmt: WeightFormat, w: u64, scale: u64, y: u64, ldy: usize, partial: u64, n: usize, k: usize, splits: usize, alpha: f32, accumulate: bool) -> Result<()> {
    let err = unsafe { ffi::squash_gemv(fmt as i32, m as i32, p(x), ldx as i32, p(w), opt_f(scale), y as *mut f32, ldy as i32, partial as *mut f32, n as i32, k as i32, splits as i32, alpha, accumulate as i32, s.raw()) };
    check(err, "gemv")
}

/// BF16 [N, K] -> INT8 [N, K] with per-row scales absmax / 127.
pub fn quantize_int8_rows(s: &Stream, w: u64, n: usize, k: usize, q: u64, scale: u64) -> Result<()> {
    check(unsafe { ffi::squash_quantize_int8_rows(p(w), n as i32, k as i32, pm(q), scale as *mut f32, s.raw()) }, "quantize_int8_rows")
}

/// BF16 [N, K] -> FP8 E4M3 [N, K] with per-row scales absmax / 448.
pub fn quantize_fp8_rows(s: &Stream, w: u64, n: usize, k: usize, q: u64, scale: u64) -> Result<()> {
    check(unsafe { ffi::squash_quantize_fp8_rows(p(w), n as i32, k as i32, pm(q), scale as *mut f32, s.raw()) }, "quantize_fp8_rows")
}

fn f(addr: u64) -> *const f32 {
    addr as *const f32
}
fn fm(addr: u64) -> *mut f32 {
    addr as *mut f32
}
fn i(addr: u64) -> *const i32 {
    addr as *const i32
}
fn im(addr: u64) -> *mut i32 {
    addr as *mut i32
}
fn n32(v: usize) -> i32 {
    v as i32
}

macro_rules! launch {
    ($name:literal, $call:expr) => {
        check(unsafe { $call }, $name)
    };
}

/// Row-wise top-k (k ≤ 512) of `rows` × `n` FP32 values with row stride `ld`; sorted by value
/// (descending) or by index (ascending). `val` may be 0.
#[allow(clippy::too_many_arguments)]
pub fn topk(s: &Stream, x: u64, rows: usize, n: usize, ld: usize, k: usize, sort_by_index: bool, idx: u64, val: u64) -> Result<()> {
    let val = if val == 0 { std::ptr::null_mut() } else { fm(val) };
    launch!("topk", ffi::squash_topk(f(x), n32(rows), n32(n), ld, n32(k), sort_by_index as i32, im(idx), val, s.raw()))
}

/// Per-row log-sum-exp, and the logit at `targets[r]` when `targets` is non-zero.
pub fn row_lse(s: &Stream, x: u64, rows: usize, n: usize, ld: usize, targets: u64, lse: u64, target_logit: u64) -> Result<()> {
    let targets = if targets == 0 { std::ptr::null() } else { i(targets) };
    launch!("row_lse", ffi::squash_row_lse(f(x), n32(rows), n32(n), ld, targets, fm(lse), fm(target_logit), s.raw()))
}

pub fn embed(s: &Stream, tokens: u64, t: usize, table: u64, h: u64) -> Result<()> {
    launch!("embed", ffi::squash_embed(i(tokens), n32(t), p(table), fm(h), s.raw()))
}

pub fn grouped_rmsnorm(s: &Stream, h: u64, w: u64, n: u64, t: usize) -> Result<()> {
    launch!("grouped_rmsnorm", ffi::squash_grouped_rmsnorm(f(h), p(w), pm(n), n32(t), s.raw()))
}

pub fn hc_inject(s: &Stream, n: u64, w: u64, inj: u64, t: usize) -> Result<()> {
    launch!("hc_inject", ffi::squash_hc_inject(p(n), p(w), fm(inj), n32(t), s.raw()))
}

pub fn hc_low_rank_act(s: &Stream, x: u64, z: u64, count: usize) -> Result<()> {
    launch!("hc_low_rank_act", ffi::squash_hc_low_rank_act(f(x), pm(z), n32(count), s.raw()))
}

pub fn hc_mix(s: &Stream, u: u64, n: u64, m: u64, t: usize) -> Result<()> {
    launch!("hc_mix", ffi::squash_hc_mix(f(u), p(n), pm(m), n32(t), s.raw()))
}

pub fn hc_combine(s: &Stream, h: u64, y: u64, inj: u64, t: usize) -> Result<()> {
    launch!("hc_combine", ffi::squash_hc_combine(fm(h), f(y), f(inj), n32(t), s.raw()))
}

pub fn f32_to_bf16(s: &Stream, x: u64, y: u64, count: usize) -> Result<()> {
    launch!("f32_to_bf16", ffi::squash_f32_to_bf16(f(x), pm(y), count, s.raw()))
}

pub fn add_f32(s: &Stream, y: u64, x: u64, count: usize) -> Result<()> {
    launch!("add_f32", ffi::squash_add_f32(fm(y), f(x), count, s.raw()))
}

pub fn gdn_conv(s: &Stream, x: u64, out: u64, state: u64, w: u64, t: usize) -> Result<()> {
    launch!("gdn_conv", ffi::squash_gdn_conv(f(x), fm(out), fm(state), p(w), n32(t), s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_prep(s: &Stream, qkv: u64, a: u64, b: u64, a_log: u64, dt_bias: u64, beta: u64, g: u64, t: usize) -> Result<()> {
    launch!("gdn_prep", ffi::squash_gdn_prep(fm(qkv), f(a), f(b), p(a_log), p(dt_bias), fm(beta), fm(g), n32(t), s.raw()))
}

pub fn gdn_recurrent(s: &Stream, qkv: u64, beta: u64, g: u64, state: u64, out: u64, t: usize) -> Result<()> {
    launch!("gdn_recurrent", ffi::squash_gdn_recurrent(f(qkv), f(beta), f(g), fm(state), fm(out), n32(t), s.raw()))
}

pub fn gdn_gated_norm(s: &Stream, o: u64, z: u64, w: u64, y: u64, t: usize) -> Result<()> {
    launch!("gdn_gated_norm", ffi::squash_gdn_gated_norm(f(o), f(z), p(w), pm(y), n32(t), s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_qkv_prep(s: &Stream, qg: u64, kv: u64, q_norm: u64, k_norm: u64, q: u64, kcache: u64, vcache: u64, pos3: u64, start: u64, t: usize) -> Result<()> {
    launch!("qsa_qkv_prep", ffi::squash_qsa_qkv_prep(f(qg), f(kv), p(q_norm), p(k_norm), fm(q), pm(kcache), pm(vcache), i(pos3), i(start), n32(t), s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_index_prep(s: &Stream, idx: u64, q_ln: u64, iq: u64, raw: u64, pos3: u64, start: u64, t: usize) -> Result<()> {
    launch!("qsa_index_prep", ffi::squash_qsa_index_prep(f(idx), p(q_ln), fm(iq), pm(raw), i(pos3), i(start), n32(t), s.raw()))
}

pub fn qsa_pool(s: &Stream, raw: u64, k_ln: u64, pooled: u64, pos3: u64, start: u64, t: usize) -> Result<()> {
    launch!("qsa_pool", ffi::squash_qsa_pool(p(raw), p(k_ln), pm(pooled), i(pos3), i(start), n32(t), s.raw()))
}

pub fn qsa_index_scores(s: &Stream, iq: u64, pooled: u64, scores: u64, ld: usize, start: u64, t: usize) -> Result<()> {
    launch!("qsa_index_scores", ffi::squash_qsa_index_scores(f(iq), p(pooled), fm(scores), n32(ld), i(start), n32(t), s.raw()))
}

/// Top-512 group selection for the rows whose queries see more than 512 complete groups.
pub fn qsa_select(s: &Stream, scores: u64, t: usize, ld: usize, start: u64, sel: u64) -> Result<()> {
    launch!("qsa_select", ffi::squash_qsa_select(f(scores), n32(t), ld, i(start), im(sel), s.raw()))
}

pub fn store_positions(s: &Stream, src: u64, pos3: u64, start: u64, t: usize) -> Result<()> {
    launch!("store_positions", ffi::squash_store_positions(i(src), im(pos3), i(start), n32(t), s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_attention(s: &Stream, q: u64, qg: u64, kcache: u64, vcache: u64, sel: u64, start: u64, t: usize, out: u64) -> Result<()> {
    launch!("qsa_attention", ffi::squash_qsa_attention(f(q), f(qg), p(kcache), p(vcache), i(sel), i(start), n32(t), pm(out), s.raw()))
}

pub fn moe_tile_rows() -> usize {
    unsafe { ffi::squash_moe_tile_rows() as usize }
}

pub fn moe_route(s: &Stream, logits: u64, t: usize, top_i: u64, top_w: u64) -> Result<()> {
    launch!("moe_route", ffi::squash_moe_route(f(logits), n32(t), im(top_i), fm(top_w), s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn moe_plan(s: &Stream, top_i: u64, t: usize, counts: u64, cursor: u64, offsets: u64, tile_expert: u64, tile_row: u64, num_tiles: u64, pair_row: u64) -> Result<()> {
    launch!("moe_plan", ffi::squash_moe_plan(i(top_i), n32(t), im(counts), im(cursor), im(offsets), im(tile_expert), im(tile_row), im(num_tiles), im(pair_row), s.raw()))
}

pub fn moe_gather(s: &Stream, m: u64, pair_row: u64, t: usize, xs: u64) -> Result<()> {
    launch!("moe_gather", ffi::squash_moe_gather(p(m), i(pair_row), n32(t), pm(xs), s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn moe_nvfp4_gemm(s: &Stream, xs: u64, w_ptr: u64, s_ptr: u64, ws2: u64, offsets: u64, tile_expert: u64, tile_row: u64, num_tiles: u64, max_tiles: usize, y: u64, n: usize, k: usize) -> Result<()> {
    launch!(
        "moe_nvfp4_gemm",
        ffi::squash_moe_nvfp4_gemm(p(xs), w_ptr as *const u64, s_ptr as *const u64, f(ws2), i(offsets), i(tile_expert), i(tile_row), i(num_tiles), n32(max_tiles), fm(y), n32(n), n32(k), s.raw())
    )
}

pub fn swiglu(s: &Stream, gate: u64, up: u64, act: u64, count: usize) -> Result<()> {
    launch!("swiglu", ffi::squash_swiglu(f(gate), f(up), pm(act), count, s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn moe_combine(s: &Stream, yd: u64, pair_row: u64, top_w: u64, sg: u64, sh: u64, y: u64, t: usize) -> Result<()> {
    launch!("moe_combine", ffi::squash_moe_combine(f(yd), i(pair_row), f(top_w), f(sg), f(sh), fm(y), n32(t), s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn ple_gate(s: &Stream, kp: u64, h: u64, v: u64, wk: u64, wq: u64, wc: u64, gv: u64, xn: u64, t: usize) -> Result<()> {
    launch!("ple_gate", ffi::squash_ple_gate(f(kp), f(h), f(v), p(wk), p(wq), p(wc), fm(gv), fm(xn), n32(t), s.raw()))
}

pub fn ple_conv(s: &Stream, gv: u64, xn: u64, w: u64, state: u64, h: u64, t: usize) -> Result<()> {
    launch!("ple_conv", ffi::squash_ple_conv(f(gv), f(xn), p(w), fm(state), fm(h), n32(t), s.raw()))
}

pub fn moe_active(s: &Stream, counts: u64, active: u64, num_active: u64) -> Result<()> {
    launch!("moe_active", ffi::squash_moe_active(i(counts), im(active), im(num_active), s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn moe_nvfp4_gemv(s: &Stream, xs: u64, w_ptr: u64, s_ptr: u64, ws2: u64, offsets: u64, active: u64, num_active: u64, max_active: usize, y: u64, n: usize, k: usize) -> Result<()> {
    launch!("moe_nvfp4_gemv", ffi::squash_moe_nvfp4_gemv(p(xs), w_ptr as *const u64, s_ptr as *const u64, f(ws2), i(offsets), i(active), i(num_active), n32(max_active), fm(y), n32(n), n32(k), s.raw()))
}

pub fn moe_small_tokens() -> usize {
    unsafe { ffi::squash_moe_small_tokens() as usize }
}

/// Routing and expert grouping for at most [`moe_small_tokens`] tokens, in one kernel.
#[allow(clippy::too_many_arguments)]
pub fn moe_route_small(s: &Stream, logits: u64, t: usize, top_i: u64, top_w: u64, offsets: u64, pair_row: u64, row_token: u64, active: u64, num_active: u64) -> Result<()> {
    launch!("moe_route_small", ffi::squash_moe_route_small(f(logits), n32(t), im(top_i), fm(top_w), im(offsets), im(pair_row), im(row_token), im(active), im(num_active), s.raw()))
}

/// Expert gate and up GEMVs with the SwiGLU epilogue (W4A16, small batches).
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_up_small(s: &Stream, m: u64, row_token: u64, gate: (u64, u64, u64), up: (u64, u64, u64), offsets: u64, active: u64, num_active: u64, max_active: usize, act: u64) -> Result<()> {
    launch!(
        "moe_gate_up_small",
        ffi::squash_moe_gate_up_small(p(m), i(row_token), gate.0 as *const u64, gate.1 as *const u64, f(gate.2), up.0 as *const u64, up.1 as *const u64, f(up.2), i(offsets), i(active), i(num_active), n32(max_active), pm(act), s.raw())
    )
}

#[allow(clippy::too_many_arguments)]
pub fn moe_down_small(s: &Stream, act: u64, down: (u64, u64, u64), offsets: u64, active: u64, num_active: u64, max_active: usize, yd: u64) -> Result<()> {
    launch!("moe_down_small", ffi::squash_moe_down_small(p(act), down.0 as *const u64, down.1 as *const u64, f(down.2), i(offsets), i(active), i(num_active), n32(max_active), fm(yd), s.raw()))
}

pub fn qsa_split_floats(t: usize) -> usize {
    unsafe { ffi::squash_qsa_split_floats(t as i32) }
}

/// Key-split attention for small batches; `part` holds [`qsa_split_floats`]`(t)` floats.
#[allow(clippy::too_many_arguments)]
pub fn qsa_attention_split(s: &Stream, q: u64, qg: u64, kcache: u64, vcache: u64, sel: u64, start: u64, t: usize, part: u64, out: u64) -> Result<()> {
    launch!("qsa_attention_split", ffi::squash_qsa_attention_split(f(q), f(qg), p(kcache), p(vcache), i(sel), i(start), n32(t), fm(part), pm(out), s.raw()))
}
