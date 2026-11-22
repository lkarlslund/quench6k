//! Launchers for the Hy4-flash kernels (`kernels/hy4.cu`). Pointers are raw device addresses; every
//! launch is asynchronous on the given stream.

use std::ffi::{c_int, c_void};

use anyhow::Result;

use crate::{Stream, check};

mod ffi {
    use super::*;
    use crate::ffi::{Error, Stream};

    unsafe extern "C" {
        pub fn quench6k_hy4_hc_pre(
            lanes: *const c_void,
            fn_: *const f32,
            base: *const f32,
            scale: *const f32,
            norm_w: *const c_void,
            hc_eps: f32,
            mag: f32,
            eps: f32,
            head: c_int,
            t: c_int,
            partial: *mut f32,
            out: *mut c_void,
            post: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_hc_partial_floats(t: c_int) -> usize;
        pub fn quench6k_hy4_hc_post(
            lanes: *mut c_void,
            post: *const f32,
            y: *const c_void,
            ldy: c_int,
            t: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_hc_expand(emb: *const c_void, lanes: *mut c_void, t: c_int, st: Stream) -> Error;
        pub fn quench6k_hy4_rms_rows(
            x: *const f32,
            ldi: c_int,
            w: *const c_void,
            n: c_int,
            eps: f32,
            out: *mut c_void,
            ldo: c_int,
            rows: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_swiglu(
            g: *const f32,
            u: *const f32,
            ld: c_int,
            width: c_int,
            limit: f32,
            act: *mut c_void,
            lda: c_int,
            rows: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_add_rows(
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            n: c_int,
            rows: c_int,
            out: *mut f32,
            ldo: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_mla_prep(
            q: *const f32,
            ldq: c_int,
            ckv: *const f32,
            ldc: c_int,
            kv_norm: *const c_void,
            eps: f32,
            theta: f32,
            h: c_int,
            pos: *const i32,
            seq: *const i32,
            cache: SeqRows,
            t: c_int,
            qn: *mut c_void,
            qa: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_heads_gemv_max_t() -> c_int;
        pub fn quench6k_hy4_heads_gemv(
            x: *const c_void,
            ldx: c_int,
            xh: c_int,
            w: *const c_void,
            wh: usize,
            n: c_int,
            k: c_int,
            h: c_int,
            t: c_int,
            gate: *const f32,
            ldg: c_int,
            y: *mut c_void,
            ldy: c_int,
            yh: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_gate_mul(v: *mut c_void, gate: *const f32, count: usize, st: Stream) -> Error;
        pub fn quench6k_hy4_attn_part_floats(t: c_int, h: c_int, splits: c_int) -> usize;
        pub fn quench6k_hy4_mla_attention(
            mode: c_int,
            qa: *const c_void,
            h: c_int,
            cache: SeqRows,
            sel: *const i32,
            sel_n: *const i32,
            sinks: *const f32,
            pos: *const i32,
            seq: *const i32,
            t: c_int,
            splits: c_int,
            heads_per_block: c_int,
            part: *mut f32,
            out: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_idx_prep(
            tk: *const f32,
            ldk: c_int,
            tk_norm: *const c_void,
            eps: f32,
            iq: *const f32,
            ldq: c_int,
            hw: *const f32,
            ldw: c_int,
            theta: f32,
            pos: *const i32,
            seq: *const i32,
            t: c_int,
            tk_cache: SeqRows,
            iq_out: *mut c_void,
            hw_out: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_idx_pool(
            tk_cache: SeqRows,
            wv: *const f32,
            wg: *const f32,
            ape: *const f32,
            key_norm: *const f32,
            eps: f32,
            theta: f32,
            pos: *const i32,
            seq: *const i32,
            t: c_int,
            pooled: SeqRows,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_idx_scores(
            iq: *const c_void,
            hw: *const f32,
            pooled: SeqRows,
            pos: *const i32,
            seq: *const i32,
            t: c_int,
            g: c_int,
            scores: *mut f32,
            ld: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_sel_width() -> c_int;
        pub fn quench6k_hy4_idx_expand(
            picked: *const i32,
            npick: c_int,
            pos: *const i32,
            t: c_int,
            sel: *mut i32,
            sel_n: *mut i32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_route(
            logits: *const f32,
            ld: c_int,
            bias: *const f32,
            scaling: f32,
            t: c_int,
            sel_e: *mut i32,
            sel_w: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_experts_small(
            x: *const c_void,
            sel_e: *const i32,
            gc: *const u8,
            gs: *const u8,
            g2: *const f32,
            uc: *const u8,
            us: *const u8,
            u2: *const f32,
            dc: *const u8,
            ds: *const u8,
            d2: *const f32,
            limit: f32,
            t: c_int,
            order: *mut i32,
            act: *mut c_void,
            y: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_quantize_fp8b(
            w: *const c_void,
            n: c_int,
            k: c_int,
            q: *mut u8,
            scale: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_to_bf16(x: *const f32, y: *mut c_void, count: usize, st: Stream) -> Error;
        pub fn quench6k_hy4_dfly_norm(
            src: *const c_void,
            src_tab: *const u64,
            lds: usize,
            idx: *const i32,
            w: *const c_void,
            eps: f32,
            out: *mut c_void,
            ldo: c_int,
            rows: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_dfly_embed(table: *const c_void, idx: *const i32, rows: c_int, x: *mut f32, st: Stream) -> Error;
        pub fn quench6k_hy4_dfly_add(
            x: *mut f32,
            xs: c_int,
            yb: *const c_void,
            yf: *const f32,
            ys: c_int,
            gate: *const c_void,
            out: *mut c_void,
            rows: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_dfly_sel(
            keep_tab: *const u64,
            keep_off: i64,
            keep_n_off: i64,
            anchors: *const i32,
            krows: *const i32,
            seqs: c_int,
            w: c_int,
            sel: *mut i32,
            sel_n: *mut i32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_seq_scatter(
            src: *const c_void,
            lds: usize,
            bytes: c_int,
            count: *const i32,
            slot: *const i32,
            seq: *const i32,
            t: c_int,
            seq_tab: *const u64,
            off: i64,
            ld: i64,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_nvfp4_to_bf16(
            codes: *const u8,
            scales: *const u8,
            s2: f32,
            n: c_int,
            k: c_int,
            out: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_experts_combine(
            y: *const f32,
            pair_row: *const i32,
            sel_w: *const f32,
            t: c_int,
            out: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_moe_plan(
            sel_e: *const i32,
            t: c_int,
            bm: c_int,
            counts: *mut i32,
            offsets: *mut i32,
            cursor: *mut i32,
            tile_expert: *mut i32,
            tile_row: *mut i32,
            num_tiles: *mut i32,
            pair_row: *mut i32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_moe_gather(
            x: *const c_void,
            pair_row: *const i32,
            t: c_int,
            xs: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_engram_embed(
            q8: *const u8,
            sc: *const f32,
            head_norm: *const c_void,
            t: c_int,
            e: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_hy4_engram_apply(
            kv: *const f32,
            ldkv: c_int,
            bias: *const c_void,
            norm_key: *const c_void,
            norm_query: *const c_void,
            gamma: *const c_void,
            hc_scale: *const c_void,
            mask: *const u8,
            t: c_int,
            lanes: *mut c_void,
            st: Stream,
        ) -> Error;
    }
}

/// One kind of per-sequence cache of a layer (kernels/hy4.cu SeqRows): the row of position p of sequence s
/// starts at element `base[s] + off + (ring > 0 ? p % ring : p) * ld`, `base` a device table of addresses.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SeqRows {
    pub base: *const u64,
    pub off: i64,
    pub ld: i32,
    pub ring: i32,
}

impl SeqRows {
    /// Rows at element offset `off` of the buffers listed (device addresses) in the table at `tab`.
    pub fn new(tab: u64, off: usize, ld: usize, ring: usize) -> Self {
        Self {
            base: tab as *const u64,
            off: off as i64,
            ld: ld as i32,
            ring: ring as i32,
        }
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
fn u8p(a: u64) -> *const u8 {
    a as *const u8
}
fn n(v: usize) -> c_int {
    v as c_int
}

macro_rules! go {
    ($name:literal, $call:expr) => {
        check(unsafe { $call }, $name)
    };
}

/// Hyper-connection pre-mix of `t` tokens: `fn_` [8 or 4, 4 * 3072] (`head` uses 4 rows and no post
/// weights), followed by the RMSNorm `norm_w`. `out` BF16 [t, 3072]; `post` F32 [t, 4].
#[allow(clippy::too_many_arguments)]
pub fn hc_pre(
    s: &Stream,
    lanes: u64,
    fn_: u64,
    base: u64,
    scale: u64,
    norm_w: u64,
    hc_eps: f32,
    mag: f32,
    eps: f32,
    head: bool,
    t: usize,
    partial: u64,
    out: u64,
    post: u64,
) -> Result<()> {
    go!(
        "hy4_hc_pre",
        ffi::quench6k_hy4_hc_pre(
            p(lanes),
            f(fn_),
            f(base),
            f(scale),
            p(norm_w),
            hc_eps,
            mag,
            eps,
            head as c_int,
            n(t),
            fm(partial),
            pm(out),
            fm(post),
            s.raw()
        )
    )
}

pub fn hc_partial_floats(t: usize) -> usize {
    unsafe { ffi::quench6k_hy4_hc_partial_floats(n(t)) }
}

pub fn hc_post(s: &Stream, lanes: u64, post: u64, y: u64, ldy: usize, t: usize) -> Result<()> {
    go!(
        "hy4_hc_post",
        ffi::quench6k_hy4_hc_post(pm(lanes), f(post), p(y), n(ldy), n(t), s.raw())
    )
}

pub fn hc_expand(s: &Stream, emb: u64, lanes: u64, t: usize) -> Result<()> {
    go!(
        "hy4_hc_expand",
        ffi::quench6k_hy4_hc_expand(p(emb), pm(lanes), n(t), s.raw())
    )
}

/// out[r] = bf16(w * bf16(bf16(x[r]) * rsqrt(mean + eps))) for `rows` rows of `width` (≤ 4096).
#[allow(clippy::too_many_arguments)]
pub fn rms_rows(
    s: &Stream,
    x: u64,
    ldx: usize,
    w: u64,
    width: usize,
    eps: f32,
    out: u64,
    ldo: usize,
    rows: usize,
) -> Result<()> {
    go!(
        "hy4_rms_rows",
        ffi::quench6k_hy4_rms_rows(f(x), n(ldx), p(w), n(width), eps, pm(out), n(ldo), n(rows), s.raw())
    )
}

/// act = bf16(bf16(silu(bf16 g)) * bf16 u) per element; `limit` > 0 applies the experts' clamps.
#[allow(clippy::too_many_arguments)]
pub fn swiglu(
    s: &Stream,
    g: u64,
    u: u64,
    ld: usize,
    width: usize,
    limit: f32,
    act: u64,
    lda: usize,
    rows: usize,
) -> Result<()> {
    go!(
        "hy4_swiglu",
        ffi::quench6k_hy4_swiglu(f(g), f(u), n(ld), n(width), limit, pm(act), n(lda), n(rows), s.raw())
    )
}

/// out = bf16(bf16(a) + bf16(b)) (stored as F32) over `rows` x `width`.
#[allow(clippy::too_many_arguments)]
pub fn add_rows(
    s: &Stream,
    a: u64,
    lda: usize,
    b: u64,
    ldb: usize,
    width: usize,
    rows: usize,
    out: u64,
    ldo: usize,
) -> Result<()> {
    go!(
        "hy4_add_rows",
        ffi::quench6k_hy4_add_rows(f(a), n(lda), f(b), n(ldb), n(width), n(rows), fm(out), n(ldo), s.raw())
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mla_prep(
    s: &Stream,
    q: u64,
    ldq: usize,
    ckv: u64,
    ldc: usize,
    kv_norm: u64,
    eps: f32,
    theta: f32,
    heads: usize,
    pos: u64,
    seq: u64,
    cache: SeqRows,
    t: usize,
    qn: u64,
    qa: u64,
) -> Result<()> {
    go!(
        "hy4_mla_prep",
        ffi::quench6k_hy4_mla_prep(
            f(q),
            n(ldq),
            f(ckv),
            n(ldc),
            p(kv_norm),
            eps,
            theta,
            n(heads),
            i(pos),
            i(seq),
            cache,
            n(t),
            pm(qn),
            pm(qa),
            s.raw()
        )
    )
}

pub fn heads_gemv_max_t() -> usize {
    unsafe { ffi::quench6k_hy4_heads_gemv_max_t() as usize }
}

/// y[t, h, :N] = W[h] . x[t, h, :K] for t < 16 (BF16 in and out); with `gate` the output is multiplied
/// by bf16(sigmoid(gate[t, h * N + n])).
#[allow(clippy::too_many_arguments)]
pub fn heads_gemv(
    s: &Stream,
    x: u64,
    ldx: usize,
    xh: usize,
    w: u64,
    wh: usize,
    nn: usize,
    k: usize,
    heads: usize,
    t: usize,
    gate: u64,
    ldg: usize,
    y: u64,
    ldy: usize,
    yh: usize,
) -> Result<()> {
    go!(
        "hy4_heads_gemv",
        ffi::quench6k_hy4_heads_gemv(
            p(x),
            n(ldx),
            n(xh),
            p(w),
            wh,
            n(nn),
            n(k),
            n(heads),
            n(t),
            f(gate),
            n(ldg),
            pm(y),
            n(ldy),
            n(yh),
            s.raw()
        )
    )
}

pub fn gate_mul(s: &Stream, v: u64, gate: u64, count: usize) -> Result<()> {
    go!(
        "hy4_gate_mul",
        ffi::quench6k_hy4_gate_mul(pm(v), f(gate), count, s.raw())
    )
}

pub fn attn_part_floats(t: usize, heads: usize, splits: usize) -> usize {
    unsafe { ffi::quench6k_hy4_attn_part_floats(n(t), n(heads), n(splits)) }
}

/// Which keys a query sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnMode {
    Sliding = 0,
    Dense = 1,
    Selected = 2,
}

#[allow(clippy::too_many_arguments)]
pub fn mla_attention(
    s: &Stream,
    mode: AttnMode,
    qa: u64,
    heads: usize,
    cache: SeqRows,
    sel: u64,
    sel_n: u64,
    sinks: u64,
    pos: u64,
    seq: u64,
    t: usize,
    splits: usize,
    heads_per_block: usize,
    part: u64,
    out: u64,
) -> Result<()> {
    go!(
        "hy4_mla_attention",
        ffi::quench6k_hy4_mla_attention(
            mode as c_int,
            p(qa),
            n(heads),
            cache,
            i(sel),
            i(sel_n),
            f(sinks),
            i(pos),
            i(seq),
            n(t),
            n(splits),
            n(heads_per_block),
            fm(part),
            pm(out),
            s.raw()
        )
    )
}

#[allow(clippy::too_many_arguments)]
pub fn idx_prep(
    s: &Stream,
    tk: u64,
    ldk: usize,
    tk_norm: u64,
    eps: f32,
    iq: u64,
    ldq: usize,
    hw: u64,
    ldw: usize,
    theta: f32,
    pos: u64,
    seq: u64,
    t: usize,
    tk_cache: SeqRows,
    iq_out: u64,
    hw_out: u64,
) -> Result<()> {
    go!(
        "hy4_idx_prep",
        ffi::quench6k_hy4_idx_prep(
            f(tk),
            n(ldk),
            p(tk_norm),
            eps,
            f(iq),
            n(ldq),
            f(hw),
            n(ldw),
            theta,
            i(pos),
            i(seq),
            n(t),
            tk_cache,
            pm(iq_out),
            fm(hw_out),
            s.raw()
        )
    )
}

#[allow(clippy::too_many_arguments)]
/// Pools the group each of the `t` rows completes (rows at positions 4g + 3) from the token-key cache.
pub fn idx_pool(
    s: &Stream,
    tk_cache: SeqRows,
    wv: u64,
    wg: u64,
    ape: u64,
    key_norm: u64,
    eps: f32,
    theta: f32,
    pos: u64,
    seq: u64,
    t: usize,
    pooled: SeqRows,
) -> Result<()> {
    go!(
        "hy4_idx_pool",
        ffi::quench6k_hy4_idx_pool(tk_cache, f(wv), f(wg), f(ape), f(key_norm), eps, theta, i(pos), i(seq), n(t), pooled, s.raw())
    )
}

/// Indexer scores of the `t` rows over `groups` columns (groups the row's sequence has not completed
/// before its own: -inf); rows within the first 2048 positions are left unwritten.
#[allow(clippy::too_many_arguments)]
pub fn idx_scores(
    s: &Stream,
    iq: u64,
    hw: u64,
    pooled: SeqRows,
    pos: u64,
    seq: u64,
    t: usize,
    groups: usize,
    scores: u64,
    ld: usize,
) -> Result<()> {
    go!(
        "hy4_idx_scores",
        ffi::quench6k_hy4_idx_scores(p(iq), f(hw), pooled, i(pos), i(seq), n(t), n(groups), fm(scores), n(ld), s.raw())
    )
}

/// Positions per query in a selected-attention list (2048).
pub fn sel_width() -> usize {
    unsafe { ffi::quench6k_hy4_sel_width() as usize }
}

pub fn idx_expand(s: &Stream, picked: u64, npick: usize, pos: u64, t: usize, sel: u64, sel_n: u64) -> Result<()> {
    go!(
        "hy4_idx_expand",
        ffi::quench6k_hy4_idx_expand(i(picked), n(npick), i(pos), n(t), im(sel), im(sel_n), s.raw())
    )
}

#[allow(clippy::too_many_arguments)]
pub fn route(
    s: &Stream,
    logits: u64,
    ld: usize,
    bias: u64,
    scaling: f32,
    t: usize,
    sel_e: u64,
    sel_w: u64,
) -> Result<()> {
    go!(
        "hy4_route",
        ffi::quench6k_hy4_route(f(logits), n(ld), f(bias), scaling, n(t), im(sel_e), fm(sel_w), s.raw())
    )
}

/// One projection kind of a layer's routed experts: codes, block scales and per-expert FP32 scales.
#[derive(Clone, Copy, Debug)]
pub struct Experts {
    pub codes: u64,
    pub scales: u64,
    pub s2: u64,
}

#[allow(clippy::too_many_arguments)]
pub fn experts_small(
    s: &Stream,
    x: u64,
    sel_e: u64,
    gate: Experts,
    up: Experts,
    down: Experts,
    limit: f32,
    t: usize,
    order: u64,
    act: u64,
    y: u64,
) -> Result<()> {
    go!(
        "hy4_experts_small",
        ffi::quench6k_hy4_experts_small(
            p(x),
            i(sel_e),
            u8p(gate.codes),
            u8p(gate.scales),
            f(gate.s2),
            u8p(up.codes),
            u8p(up.scales),
            f(up.s2),
            u8p(down.codes),
            u8p(down.scales),
            f(down.s2),
            limit,
            n(t),
            im(order),
            pm(act),
            fm(y),
            s.raw()
        )
    )
}

/// BF16 [n, k] -> FP8 E4M3 with an FP32 scale per 128 x 128 block (WeightFormat::Fp8Block).
pub fn quantize_fp8b(s: &Stream, w: u64, nn: usize, k: usize, q: u64, scale: u64) -> Result<()> {
    go!(
        "hy4_quantize_fp8b",
        ffi::quench6k_hy4_quantize_fp8b(p(w), n(nn), n(k), q as *mut u8, fm(scale), s.raw())
    )
}

/// y = bf16(x); its dependents may start only after this kernel's own dependency wait, so they may read
/// data written before its predecessor started (the routing, for the experts).
pub fn to_bf16(s: &Stream, x: u64, y: u64, count: usize) -> Result<()> {
    go!("hy4_to_bf16", ffi::quench6k_hy4_to_bf16(f(x), pm(y), count, s.raw()))
}

pub fn experts_combine(s: &Stream, y: u64, pair_row: u64, sel_w: u64, t: usize, out: u64) -> Result<()> {
    go!(
        "hy4_experts_combine",
        ffi::quench6k_hy4_experts_combine(f(y), i(pair_row), f(sel_w), n(t), pm(out), s.raw())
    )
}

#[allow(clippy::too_many_arguments)]
pub fn moe_plan(
    s: &Stream,
    sel_e: u64,
    t: usize,
    bm: usize,
    counts: u64,
    offsets: u64,
    cursor: u64,
    tile_expert: u64,
    tile_row: u64,
    num_tiles: u64,
    pair_row: u64,
) -> Result<()> {
    go!(
        "hy4_moe_plan",
        ffi::quench6k_hy4_moe_plan(
            i(sel_e),
            n(t),
            n(bm),
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

pub fn moe_gather(s: &Stream, x: u64, pair_row: u64, t: usize, xs: u64) -> Result<()> {
    go!(
        "hy4_moe_gather",
        ffi::quench6k_hy4_moe_gather(p(x), i(pair_row), n(t), pm(xs), s.raw())
    )
}

pub fn engram_embed(s: &Stream, q8: u64, sc: u64, head_norm: u64, t: usize, e: u64) -> Result<()> {
    go!(
        "hy4_engram_embed",
        ffi::quench6k_hy4_engram_embed(u8p(q8), f(sc), p(head_norm), n(t), pm(e), s.raw())
    )
}

#[allow(clippy::too_many_arguments)]
pub fn engram_apply(
    s: &Stream,
    kv: u64,
    ldkv: usize,
    bias: u64,
    norm_key: u64,
    norm_query: u64,
    gamma: u64,
    hc_scale: u64,
    mask: u64,
    t: usize,
    lanes: u64,
) -> Result<()> {
    go!(
        "hy4_engram_apply",
        ffi::quench6k_hy4_engram_apply(
            f(kv),
            n(ldkv),
            p(bias),
            p(norm_key),
            p(norm_query),
            p(gamma),
            p(hc_scale),
            u8p(mask),
            n(t),
            pm(lanes),
            s.raw()
        )
    )
}

/// out[r, :3072] = bf16(w * rmsnorm(x)) for BF16 rows x = src + (idx ? idx[r] : r) * lds elements (idx: device
/// I32 indices, 0 for none); with `src_tab` (device u64 [rows], 0 for none) row r's src is src_tab[r].
#[allow(clippy::too_many_arguments)]
pub fn dfly_norm(s: &Stream, src: u64, src_tab: u64, lds: usize, idx: u64, w: u64, eps: f32, out: u64, ldo: usize, rows: usize) -> Result<()> {
    go!(
        "hy4_dfly_norm",
        ffi::quench6k_hy4_dfly_norm(p(src), src_tab as *const u64, lds, i(idx), p(w), eps, pm(out), n(ldo), n(rows), s.raw())
    )
}

/// x[r] (F32 [rows, 3072]) = table[idx[r]] (BF16 embedding rows; idx device I32).
pub fn dfly_embed(s: &Stream, table: u64, idx: u64, rows: usize, x: u64) -> Result<()> {
    go!("hy4_dfly_embed", ffi::quench6k_hy4_dfly_embed(p(table), i(idx), n(rows), fm(x), s.raw()))
}

/// x row r * xs (F32 rows of 3072) += y row r * ys, y BF16 (`yb`) or F32 (`yf`; 0 for the other), times
/// bf16 `gate` [3072] when given; `out` (when not 0) row r gets bf16(x).
#[allow(clippy::too_many_arguments)]
pub fn dfly_add(s: &Stream, x: u64, xs: usize, yb: u64, yf: u64, ys: usize, gate: u64, out: u64, rows: usize) -> Result<()> {
    go!(
        "hy4_dfly_add",
        ffi::quench6k_hy4_dfly_add(fm(x), n(xs), p(yb), f(yf), n(ys), p(gate), pm(out), n(rows), s.raw())
    )
}

/// Key lists (sel [seqs * w, sel_width()], sel_n [seqs * w]) of the drafter's query blocks from the target's
/// kept lists: sequence s's lists at I32 offset `keep_off` (lengths `keep_n_off`) of the buffer at
/// keep_tab[s], its block anchored at anchors[s] with source row krows[s] (device I32).
#[allow(clippy::too_many_arguments)]
pub fn dfly_sel(
    s: &Stream,
    keep_tab: u64,
    keep_off: usize,
    keep_n_off: usize,
    anchors: u64,
    krows: u64,
    seqs: usize,
    w: usize,
    sel: u64,
    sel_n: u64,
) -> Result<()> {
    go!(
        "hy4_dfly_sel",
        ffi::quench6k_hy4_dfly_sel(
            keep_tab as *const u64,
            keep_off as i64,
            keep_n_off as i64,
            i(anchors),
            i(krows),
            n(seqs),
            n(w),
            im(sel),
            im(sel_n),
            s.raw()
        )
    )
}

/// Copies row t (`bytes` bytes at src + t * lds, or 4 * count[t] bytes with `count`) of the `t` rows with
/// slot[t] >= 0 to byte offset off + slot[t] * ld of the buffer at seq_tab[seq[t]].
#[allow(clippy::too_many_arguments)]
pub fn seq_scatter(
    s: &Stream,
    src: u64,
    lds: usize,
    bytes: usize,
    count: u64,
    slot: u64,
    seq: u64,
    t: usize,
    seq_tab: u64,
    off: usize,
    ld: usize,
) -> Result<()> {
    go!(
        "hy4_seq_scatter",
        ffi::quench6k_hy4_seq_scatter(
            p(src),
            lds,
            n(bytes),
            i(count),
            i(slot),
            i(seq),
            n(t),
            seq_tab as *const u64,
            off as i64,
            ld as i64,
            s.raw()
        )
    )
}

/// NVFP4 [nn, k] (codes, E4M3 block scales, global scale s2) -> BF16 [nn, k].
pub fn nvfp4_to_bf16(s: &Stream, codes: u64, scales: u64, s2: f32, nn: usize, k: usize, out: u64) -> Result<()> {
    go!(
        "hy4_nvfp4_to_bf16",
        ffi::quench6k_hy4_nvfp4_to_bf16(codes as *const u8, scales as *const u8, s2, n(nn), n(k), pm(out), s.raw())
    )
}
