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

/// Y[M, N] = alpha · X[M, K] · W[N, K]ᵀ (or `+=` with `accumulate`), BF16 inputs, FP32 output.
#[allow(clippy::too_many_arguments)]
pub fn gemm_f32(s: &Stream, x: u64, ldx: usize, w: u64, ldw: usize, y: u64, ldy: usize, m: usize, n: usize, k: usize, alpha: f32, accumulate: bool) -> Result<()> {
    let err = unsafe { ffi::squash_gemm_bf16_f32(p(x), ldx as i32, p(w), ldw as i32, y as *mut f32, ldy as i32, m as i32, n as i32, k as i32, alpha, accumulate as i32, s.raw()) };
    check(err, "gemm_bf16_f32")
}

/// As [`gemm_f32`] with a BF16 output.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16(s: &Stream, x: u64, ldx: usize, w: u64, ldw: usize, y: u64, ldy: usize, m: usize, n: usize, k: usize, alpha: f32, accumulate: bool) -> Result<()> {
    let err = unsafe { ffi::squash_gemm_bf16_bf16(p(x), ldx as i32, p(w), ldw as i32, pm(y), ldy as i32, m as i32, n as i32, k as i32, alpha, accumulate as i32, s.raw()) };
    check(err, "gemm_bf16_bf16")
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
pub fn qsa_qkv_prep(s: &Stream, qg: u64, kv: u64, q_norm: u64, k_norm: u64, q: u64, kcache: u64, vcache: u64, pos3: u64, start: usize, t: usize) -> Result<()> {
    launch!("qsa_qkv_prep", ffi::squash_qsa_qkv_prep(f(qg), f(kv), p(q_norm), p(k_norm), fm(q), pm(kcache), pm(vcache), i(pos3), n32(start), n32(t), s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_index_prep(s: &Stream, idx: u64, q_ln: u64, iq: u64, raw: u64, pos3: u64, start: usize, t: usize) -> Result<()> {
    launch!("qsa_index_prep", ffi::squash_qsa_index_prep(f(idx), p(q_ln), fm(iq), pm(raw), i(pos3), n32(start), n32(t), s.raw()))
}

pub fn qsa_pool(s: &Stream, raw: u64, k_ln: u64, pooled: u64, pos3: u64, start: usize, t: usize) -> Result<()> {
    launch!("qsa_pool", ffi::squash_qsa_pool(p(raw), p(k_ln), pm(pooled), i(pos3), n32(start), n32(t), s.raw()))
}

pub fn qsa_index_scores(s: &Stream, iq: u64, pooled: u64, scores: u64, ld: usize, start: usize, t: usize) -> Result<()> {
    launch!("qsa_index_scores", ffi::squash_qsa_index_scores(f(iq), p(pooled), fm(scores), n32(ld), n32(start), n32(t), s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_attention(s: &Stream, q: u64, qg: u64, kcache: u64, vcache: u64, sel: u64, start: usize, t: usize, out: u64) -> Result<()> {
    launch!("qsa_attention", ffi::squash_qsa_attention(f(q), f(qg), p(kcache), p(vcache), i(sel), n32(start), n32(t), pm(out), s.raw()))
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
