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

/// Skinny GEMM for M <= 16 rows of BF16 activations, FP32 output; formats as in [`gemm_f32`]. With
/// `splits` > 1 the K range is split across CTAs and `partial` must hold `splits * M * N` floats.
#[allow(clippy::too_many_arguments)]
pub fn gemv(s: &Stream, m: usize, x: u64, ldx: usize, fmt: WeightFormat, w: u64, scale: u64, y: u64, ldy: usize, partial: u64, n: usize, k: usize, splits: usize, alpha: f32, accumulate: bool) -> Result<()> {
    let err = unsafe { ffi::squash_gemv(fmt as i32, m as i32, p(x), ldx as i32, p(w), opt_f(scale), y as *mut f32, ldy as i32, partial as *mut f32, n as i32, k as i32, splits as i32, alpha, accumulate as i32, s.raw()) };
    check(err, "gemv")
}

/// W[n, k] *= 1 + g[k] in place (BF16).
pub fn fold_norm(s: &Stream, w: u64, g: u64, n: usize, k: usize) -> Result<()> {
    check(unsafe { ffi::squash_fold_norm(pm(w), p(g), n as i32, k as i32, s.raw()) }, "fold_norm")
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

/// n = rms(h per stream) · (1 + w), or without the weight when `w` is 0.
pub fn grouped_rmsnorm(s: &Stream, h: u64, w: u64, n: u64, t: usize) -> Result<()> {
    launch!("grouped_rmsnorm", ffi::squash_grouped_rmsnorm(f(h), p(w), pm(n), n32(t), s.raw()))
}

pub fn hc_inject(s: &Stream, n: u64, w: u64, inj: u64, t: usize) -> Result<()> {
    launch!("hc_inject", ffi::squash_hc_inject(p(n), p(w), fm(inj), n32(t), s.raw()))
}

/// From the fused HyperConnection projection x [T, 324] (row stride `ld`): the BF16 low-rank activation
/// z [T, 320], and the injection weights [T, 4] when `inj` is non-zero.
pub fn hc_act(s: &Stream, x: u64, ld: usize, z: u64, inj: u64, t: usize) -> Result<()> {
    let inj = if inj == 0 { std::ptr::null_mut() } else { fm(inj) };
    launch!("hc_act", ffi::squash_hc_act(f(x), n32(ld), pm(z), inj, n32(t), s.raw()))
}

/// m = mean over streams of sigmoid(u) · n · (1 + g); `g` 0 omits the norm weight.
pub fn hc_mix(s: &Stream, u: u64, n: u64, g: u64, m: u64, t: usize) -> Result<()> {
    launch!("hc_mix", ffi::squash_hc_mix(f(u), p(n), p(g), pm(m), n32(t), s.raw()))
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

pub fn gdn_conv(s: &Stream, x: u64, ldx: usize, out: u64, state: u64, w: u64, t: usize) -> Result<()> {
    launch!("gdn_conv", ffi::squash_gdn_conv(f(x), n32(ldx), fm(out), fm(state), p(w), n32(t), s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_prep(s: &Stream, qkv: u64, a: u64, b: u64, ldab: usize, a_log: u64, dt_bias: u64, beta: u64, g: u64, t: usize) -> Result<()> {
    launch!("gdn_prep", ffi::squash_gdn_prep(fm(qkv), f(a), f(b), n32(ldab), p(a_log), p(dt_bias), fm(beta), fm(g), n32(t), s.raw()))
}

pub fn gdn_recurrent(s: &Stream, qkv: u64, beta: u64, g: u64, state: u64, out: u64, t: usize) -> Result<()> {
    launch!("gdn_recurrent", ffi::squash_gdn_recurrent(f(qkv), f(beta), f(g), fm(state), fm(out), n32(t), s.raw()))
}

pub fn gdn_gated_norm(s: &Stream, o: u64, z: u64, ldz: usize, w: u64, y: u64, t: usize) -> Result<()> {
    launch!("gdn_gated_norm", ffi::squash_gdn_gated_norm(f(o), f(z), n32(ldz), p(w), pm(y), n32(t), s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_qkv_prep(s: &Stream, qg: u64, kv: u64, ld: usize, q_norm: u64, k_norm: u64, q: u64, kcache: u64, vcache: u64, pos3: u64, start: u64, t: usize) -> Result<()> {
    launch!("qsa_qkv_prep", ffi::squash_qsa_qkv_prep(f(qg), f(kv), n32(ld), p(q_norm), p(k_norm), fm(q), pm(kcache), pm(vcache), i(pos3), i(start), n32(t), s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_index_prep(s: &Stream, idx: u64, ld: usize, q_ln: u64, iq: u64, raw: u64, pos3: u64, start: u64, t: usize) -> Result<()> {
    launch!("qsa_index_prep", ffi::squash_qsa_index_prep(f(idx), n32(ld), p(q_ln), fm(iq), pm(raw), i(pos3), i(start), n32(t), s.raw()))
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
pub fn qsa_attention(s: &Stream, q: u64, qg: u64, ld: usize, kcache: u64, vcache: u64, sel: u64, start: u64, t: usize, out: u64) -> Result<()> {
    launch!("qsa_attention", ffi::squash_qsa_attention(f(q), f(qg), n32(ld), p(kcache), p(vcache), i(sel), i(start), n32(t), pm(out), s.raw()))
}

pub fn moe_tile_rows() -> usize {
    unsafe { ffi::squash_moe_tile_rows() as usize }
}

pub fn moe_route(s: &Stream, logits: u64, ld: usize, t: usize, top_i: u64, top_w: u64) -> Result<()> {
    launch!("moe_route", ffi::squash_moe_route(f(logits), n32(ld), n32(t), im(top_i), fm(top_w), s.raw()))
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

/// act[r, i] = bf16(silu(gate[r·ld + i]) · up[r·ld + i]) for `rows` rows of `width`.
pub fn swiglu(s: &Stream, gate: u64, up: u64, ld: usize, width: usize, act: u64, rows: usize) -> Result<()> {
    launch!("swiglu", ffi::squash_swiglu(f(gate), f(up), n32(ld), n32(width), pm(act), rows * width, s.raw()))
}

#[allow(clippy::too_many_arguments)]
pub fn moe_combine(s: &Stream, yd: u64, pair_row: u64, top_w: u64, sg: u64, ld_sg: usize, sh: u64, y: u64, t: usize) -> Result<()> {
    launch!("moe_combine", ffi::squash_moe_combine(f(yd), i(pair_row), f(top_w), f(sg), n32(ld_sg), f(sh), fm(y), n32(t), s.raw()))
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
pub fn moe_route_small(s: &Stream, logits: u64, ld: usize, t: usize, top_i: u64, top_w: u64, offsets: u64, pair_row: u64, row_token: u64, active: u64, num_active: u64) -> Result<()> {
    launch!("moe_route_small", ffi::squash_moe_route_small(f(logits), n32(ld), n32(t), im(top_i), fm(top_w), im(offsets), im(pair_row), im(row_token), im(active), im(num_active), s.raw()))
}

/// Expert gate and up GEMVs with the SwiGLU epilogue (W4A16, small batches).
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_up_small(s: &Stream, m: u64, row_token: u64, gate: (u64, u64, u64), up: (u64, u64, u64), offsets: u64, active: u64, num_active: u64, max_active: usize, t: usize, act: u64) -> Result<()> {
    launch!(
        "moe_gate_up_small",
        ffi::squash_moe_gate_up_small(p(m), i(row_token), gate.0 as *const u64, gate.1 as *const u64, f(gate.2), up.0 as *const u64, up.1 as *const u64, f(up.2), i(offsets), i(active), i(num_active), n32(max_active), n32(t), pm(act), s.raw())
    )
}

#[allow(clippy::too_many_arguments)]
pub fn moe_down_small(s: &Stream, act: u64, down: (u64, u64, u64), offsets: u64, active: u64, num_active: u64, max_active: usize, t: usize, yd: u64) -> Result<()> {
    launch!("moe_down_small", ffi::squash_moe_down_small(p(act), down.0 as *const u64, down.1 as *const u64, f(down.2), i(offsets), i(active), i(num_active), n32(max_active), n32(t), fm(yd), s.raw()))
}

pub fn qsa_split_floats(t: usize) -> usize {
    unsafe { ffi::squash_qsa_split_floats(t as i32) }
}

/// Key-split attention for small batches; `part` holds [`qsa_split_floats`]`(t)` floats.
#[allow(clippy::too_many_arguments)]
pub fn qsa_attention_split(s: &Stream, q: u64, qg: u64, ld: usize, kcache: u64, vcache: u64, sel: u64, start: u64, t: usize, part: u64, out: u64) -> Result<()> {
    launch!("qsa_attention_split", ffi::squash_qsa_attention_split(f(q), f(qg), n32(ld), p(kcache), p(vcache), i(sel), i(start), n32(t), fm(part), pm(out), s.raw()))
}

/// Fused small-batch GDN: conv, gating, delta rule and gated norm in one kernel (one block per value head).
#[allow(clippy::too_many_arguments)]
pub fn gdn_fused_small(s: &Stream, proj: u64, ld: usize, t: usize, conv_w: u64, conv_state: u64, state: u64, a_log: u64, dt_bias: u64, norm_w: u64, z_col: usize, a_col: usize, b_col: usize, y: u64) -> Result<()> {
    launch!(
        "gdn_fused_small",
        ffi::squash_gdn_fused_small(f(proj), n32(ld), n32(t), p(conv_w), fm(conv_state), fm(state), p(a_log), p(dt_bias), p(norm_w), n32(z_col), n32(a_col), n32(b_col), pm(y), s.raw())
    )
}

pub fn argmax_scratch(rows: usize, n: usize) -> usize {
    unsafe { ffi::squash_argmax_scratch(rows as i32, n as i32) }
}

/// Row-wise argmax (ties: lower index); `scratch` holds [`argmax_scratch`] bytes.
pub fn argmax(s: &Stream, x: u64, rows: usize, n: usize, ld: usize, scratch: u64, out: u64) -> Result<()> {
    launch!("argmax", ffi::squash_argmax(f(x), n32(rows), n32(n), ld, pm(scratch), im(out), s.raw()))
}

pub fn hc_small_scratch_floats(t: usize) -> usize {
    unsafe { ffi::squash_hc_small_scratch_floats(t as i32) }
}

/// Weight of one projection for kernels that take raw pointers: (format, codes, scales).
pub type Weight = (WeightFormat, u64, u64);

/// One cooperative kernel for a whole HyperConnection on <= 16 tokens: applies the pending residual
/// update `h += y · inj_prev` (when `y` is non-zero), projects through the norm-folded down/injection
/// weights, and writes the block input `m` and, when `inject` is given, the new injection weights.
#[allow(clippy::too_many_arguments)]
pub fn hc_small(s: &Stream, h: u64, t: usize, y: u64, inj_prev: u64, down: Weight, inject: Option<Weight>, up: Weight, g: u64, scratch: u64, z: u64, r: u64, inj: u64, m: u64) -> Result<()> {
    anyhow::ensure!(down.0 == up.0 && inject.is_none_or(|w| w.0 == down.0), "hc_small needs one weight format");
    let (wi, si, inj) = match inject {
        Some(w) => (p(w.1), f(w.2), fm(inj)),
        None => (std::ptr::null(), std::ptr::null(), std::ptr::null_mut()),
    };
    let y = if y == 0 { std::ptr::null() } else { f(y) };
    launch!("hc_small", ffi::squash_hc_small(down.0 as i32, fm(h), n32(t), y, f(inj_prev), p(down.1), f(down.2), wi, si, p(up.1), f(up.2), p(g), fm(scratch), pm(z), fm(r), inj, pm(m), s.raw()))
}

/// Small batches: shared-expert down projection over swiglu(gate, up) taken from the fused MoE
/// projection rows `proj` (row stride `ld`), plus the weighted routed outputs: the MoE block output y.
#[allow(clippy::too_many_arguments)]
pub fn moe_shared_combine(s: &Stream, proj: u64, ld: usize, t: usize, down: Weight, yd: u64, pair_row: u64, top_w: u64, y: u64) -> Result<()> {
    launch!("moe_shared_combine", ffi::squash_moe_shared_combine(down.0 as i32, f(proj), n32(ld), n32(t), p(down.1), opt_f(down.2), f(yd), i(pair_row), f(top_w), fm(y), s.raw()))
}

/// Single-token MoE experts with routing fused into the gate/up kernel (see the kernel). Tables are
/// (codes, scales, weight_scale_2) device arrays over the 512 experts.
#[allow(clippy::too_many_arguments)]
pub fn moe_experts_t1(s: &Stream, m: u64, logits: u64, gate: (u64, u64, u64), up: (u64, u64, u64), down: (u64, u64, u64), top_i: u64, top_w: u64, pair_row: u64, act: u64, yd: u64) -> Result<()> {
    let t = |a: u64| a as *const u64;
    launch!(
        "moe_experts_t1",
        ffi::squash_moe_experts_t1(p(m), f(logits), t(gate.0), t(gate.1), f(gate.2), t(up.0), t(up.1), f(up.2), t(down.0), t(down.1), f(down.2), im(top_i), fm(top_w), im(pair_row), pm(act), fm(yd), s.raw())
    )
}

/// NVFP4-quantizes `count` consecutive BF16 matrices of `n` x `k` (checkpoint layout: codes, FP8 block
/// scales per 16, FP32 global scale per matrix). `amax` is scratch for `count` u32.
#[allow(clippy::too_many_arguments)]
pub fn quantize_nvfp4(s: &Stream, w: u64, count: usize, n: usize, k: usize, codes: u64, scales: u64, scale2: u64, amax: u64) -> Result<()> {
    launch!("quantize_nvfp4", ffi::squash_quantize_nvfp4(p(w), n32(count), n32(n), n32(k), pm(codes), pm(scales), fm(scale2), amax as *mut u32, s.raw()))
}

/// out[r] = bf16(rms(x[r, :2560]) · (1 + w)) for `rows` rows of FP32 at stride `ldx`.
pub fn rmsnorm_rows(s: &Stream, x: u64, ldx: usize, w: u64, out: u64, rows: usize) -> Result<()> {
    launch!("rmsnorm_rows", ffi::squash_rmsnorm_rows(f(x), n32(ldx), p(w), pm(out), n32(rows), s.raw()))
}
