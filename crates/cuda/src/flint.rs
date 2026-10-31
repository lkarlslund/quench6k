//! Launchers for the Flint kernels (`kernels/flint.cu`). Pointers are raw device addresses.

use std::ffi::{c_int, c_void};

use anyhow::Result;

use crate::{Stream, check};

mod ffi {
    use super::*;
    use crate::ffi::{Error, Stream};

    unsafe extern "C" {
        pub fn quench6k_flint_repack(
            raw: *const u8,
            blocks: usize,
            first: usize,
            kblocks: c_int,
            rows: c_int,
            pb: c_int,
            codes: *mut u8,
            scales: *mut u8,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_embed(
            tokens: *const i32,
            t: c_int,
            table: *const c_void,
            h: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_rmsnorm(
            x: *mut f32,
            add: *const f32,
            w: *const f32,
            eps: f32,
            out: *mut c_void,
            t: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_qk_prep(
            qi: *const f32,
            ldq: c_int,
            ki: *const f32,
            ldk: c_int,
            vi: *const f32,
            ldv: c_int,
            q_norm: *const f32,
            k_norm: *const f32,
            rope: c_int,
            start: *const i32,
            mask: c_int,
            f8_rows: c_int,
            t: c_int,
            q: *mut c_void,
            kc: *mut c_void,
            vc: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_attn_splits(t: c_int, start: c_int, window: c_int) -> c_int;
        pub fn quench6k_flint_attention(
            q: *const c_void,
            kc: *const c_void,
            vc: *const c_void,
            t: c_int,
            start: *const i32,
            window: c_int,
            mask: c_int,
            splits: c_int,
            sinks: *const f32,
            part_o: *mut f32,
            part_ml: *mut f32,
            tickets: *mut i32,
            out: *mut c_void,
            f8_rows: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_gemv_max_m() -> c_int;
        pub fn quench6k_flint_gemv(
            x: *const c_void,
            norm_w: *const f32,
            m: c_int,
            k: c_int,
            codes: *const *const u8,
            scales: *const *const u8,
            s2: *const *const f32,
            n: *const c_int,
            y: *mut f32,
            ldy: c_int,
            accumulate: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_router(
            h: *const f32,
            norm_w: *const f32,
            w: *const c_void,
            t: c_int,
            logits: *mut f32,
            xn: *mut c_void,
            ticket: *mut i32,
            sel_e: *mut i32,
            sel_w: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_route(
            logits: *const f32,
            t: c_int,
            sel_e: *mut i32,
            sel_w: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_dense_slots(
            t: c_int,
            s: c_int,
            sel_e: *mut i32,
            sel_w: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_moe_small(
            xn: *const c_void,
            t: c_int,
            s: c_int,
            sel_e: *const i32,
            sel_w: *const f32,
            gw: *const u64,
            gs: *const u64,
            g2: *const f32,
            uw: *const u64,
            us: *const u64,
            u2: *const f32,
            dw: *const u64,
            ds: *const u64,
            d2: *const f32,
            act: *mut f32,
            h: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_moe_plan(
            sel_e: *const i32,
            pairs: c_int,
            ne: c_int,
            bm: c_int,
            counts: *mut i32,
            cursor: *mut i32,
            offsets: *mut i32,
            tile_expert: *mut i32,
            tile_row: *mut i32,
            num_tiles: *mut i32,
            pair_row: *mut i32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_gather(
            xn: *const c_void,
            pair_row: *const i32,
            pairs: c_int,
            s: c_int,
            xs: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_swiglu(
            g: *const f32,
            u: *const f32,
            count: usize,
            act: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_combine(
            yd: *const f32,
            pair_row: *const i32,
            sel_w: *const f32,
            t: c_int,
            s: c_int,
            h: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_flint_dense_plan(
            t: c_int,
            bm: c_int,
            offsets: *mut i32,
            tile_expert: *mut i32,
            tile_row: *mut i32,
            num_tiles: *mut i32,
            st: Stream,
        ) -> Error;
    }
}

fn p<T>(a: u64) -> *const T {
    a as *const T
}
fn pm<T>(a: u64) -> *mut T {
    a as *mut T
}

/// NVFP4 matrix: codes [n, k/2], E4M3 scales [n, k/16], device FP32 global scale at `s2`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp4 {
    pub codes: u64,
    pub scales: u64,
    pub s2: u64,
    pub n: usize,
    pub k: usize,
}

/// llama.cpp NVFP4 blocks `[first, first + blocks)` of a matrix with `kblocks` 64-value blocks per row
/// (`raw` holds just those blocks) -> codes/scales; `pb` blocks per destination piece row (= kblocks: none).
#[allow(clippy::too_many_arguments)]
pub fn repack(
    s: &Stream,
    raw: u64,
    blocks: usize,
    first: usize,
    kblocks: usize,
    rows: usize,
    pb: usize,
    codes: u64,
    scales: u64,
) -> Result<()> {
    check(
        unsafe {
            ffi::quench6k_flint_repack(
                p(raw),
                blocks,
                first,
                kblocks as c_int,
                rows as c_int,
                pb as c_int,
                pm(codes),
                pm(scales),
                s.raw(),
            )
        },
        "flint_repack",
    )
}

pub fn embed(s: &Stream, tokens: u64, t: usize, table: u64, h: u64) -> Result<()> {
    check(
        unsafe { ffi::quench6k_flint_embed(p(tokens), t as c_int, p(table), pm(h), s.raw()) },
        "flint_embed",
    )
}

pub fn rmsnorm(s: &Stream, x: u64, w: u64, eps: f32, out: u64, t: usize) -> Result<()> {
    rmsnorm_add(s, x, 0, w, eps, out, t)
}

/// x += add (when non-zero), then RMSNorm into out.
pub fn rmsnorm_add(
    s: &Stream,
    x: u64,
    add: u64,
    w: u64,
    eps: f32,
    out: u64,
    t: usize,
) -> Result<()> {
    check(
        unsafe {
            ffi::quench6k_flint_rmsnorm(
                pm(x),
                if add == 0 { std::ptr::null() } else { p(add) },
                p(w),
                eps,
                pm(out),
                t as c_int,
                s.raw(),
            )
        },
        "flint_rmsnorm",
    )
}

/// q/k/v inputs (FP32 rows with strides) -> normed (and roped) q BF16 [t, 32, 128], k/v into the cache.
#[allow(clippy::too_many_arguments)]
/// `f8_rows`: 0 for BF16 caches, else the cache rows of an FP8 cache (codes, then per-row-head scales).
pub fn qk_prep(
    s: &Stream,
    qi: (u64, usize),
    ki: (u64, usize),
    vi: (u64, usize),
    q_norm: u64,
    k_norm: u64,
    rope: bool,
    start: u64,
    mask: i32,
    f8_rows: usize,
    t: usize,
    q: u64,
    kc: u64,
    vc: u64,
) -> Result<()> {
    check(
        unsafe {
            ffi::quench6k_flint_qk_prep(
                p(qi.0),
                qi.1 as c_int,
                p(ki.0),
                ki.1 as c_int,
                p(vi.0),
                vi.1 as c_int,
                p(q_norm),
                p(k_norm),
                rope as c_int,
                p(start),
                mask,
                f8_rows as c_int,
                t as c_int,
                pm(q),
                pm(kc),
                pm(vc),
                s.raw(),
            )
        },
        "flint_qk_prep",
    )
}

pub fn attn_splits(t: usize, start: usize, window: usize) -> usize {
    unsafe { ffi::quench6k_flint_attn_splits(t as c_int, start as c_int, window as c_int) as usize }
}

#[allow(clippy::too_many_arguments)]
pub fn attention(
    s: &Stream,
    q: u64,
    kc: u64,
    vc: u64,
    t: usize,
    start: u64,
    window: usize,
    mask: i32,
    splits: usize,
    sinks: u64,
    part_o: u64,
    part_ml: u64,
    tickets: u64,
    out: u64,
    f8_rows: usize,
) -> Result<()> {
    check(
        unsafe {
            ffi::quench6k_flint_attention(
                p(q),
                p(kc),
                p(vc),
                t as c_int,
                p(start),
                window as c_int,
                mask,
                splits as c_int,
                p(sinks),
                pm(part_o),
                pm(part_ml),
                pm(tickets),
                pm(out),
                f8_rows as c_int,
                s.raw(),
            )
        },
        "flint_attention",
    )
}

pub fn gemv_max_m() -> usize {
    unsafe { ffi::quench6k_flint_gemv_max_m() as usize }
}

/// y[m, c] (+)= x[m] . W_i[n] for up to three matrices with concatenated output columns. With `norm`
/// (an RMSNorm weight), x is FP32 [m, k] and is normalized first; otherwise BF16.
#[allow(clippy::too_many_arguments)]
pub fn gemv(
    s: &Stream,
    x: u64,
    norm: Option<u64>,
    m: usize,
    mats: &[Fp4],
    y: u64,
    ldy: usize,
    accumulate: bool,
) -> Result<()> {
    assert!(!mats.is_empty() && mats.len() <= 3);
    let k = mats[0].k;
    let mut codes = [std::ptr::null::<u8>(); 3];
    let mut scales = [std::ptr::null::<u8>(); 3];
    let mut s2 = [std::ptr::null::<f32>(); 3];
    let mut n = [0 as c_int; 3];
    for (i, w) in mats.iter().enumerate() {
        assert_eq!(w.k, k);
        codes[i] = p(w.codes);
        scales[i] = p(w.scales);
        s2[i] = p(w.s2);
        n[i] = w.n as c_int;
    }
    check(
        unsafe {
            ffi::quench6k_flint_gemv(
                p(x),
                norm.map_or(std::ptr::null(), p),
                m as c_int,
                k as c_int,
                codes.as_ptr(),
                scales.as_ptr(),
                s2.as_ptr(),
                n.as_ptr(),
                pm(y),
                ldy as c_int,
                accumulate as c_int,
                s.raw(),
            )
        },
        "flint_gemv",
    )
}

/// Router of a few tokens: RMSNorm(h) -> xn (BF16), logits, top-6 routing into sel_e/sel_w (8 slots).
#[allow(clippy::too_many_arguments)]
pub fn router(
    s: &Stream,
    h: u64,
    norm: u64,
    w: u64,
    t: usize,
    logits: u64,
    xn: u64,
    ticket: u64,
    sel_e: u64,
    sel_w: u64,
) -> Result<()> {
    check(
        unsafe {
            ffi::quench6k_flint_router(
                p(h),
                p(norm),
                p(w),
                t as c_int,
                pm(logits),
                pm(xn),
                pm(ticket),
                pm(sel_e),
                pm(sel_w),
                s.raw(),
            )
        },
        "flint_router",
    )
}

pub fn route(s: &Stream, logits: u64, t: usize, sel_e: u64, sel_w: u64) -> Result<()> {
    check(
        unsafe { ffi::quench6k_flint_route(p(logits), t as c_int, pm(sel_e), pm(sel_w), s.raw()) },
        "flint_route",
    )
}

pub fn dense_slots(s: &Stream, t: usize, slots: usize, sel_e: u64, sel_w: u64) -> Result<()> {
    check(
        unsafe {
            ffi::quench6k_flint_dense_slots(
                t as c_int,
                slots as c_int,
                pm(sel_e),
                pm(sel_w),
                s.raw(),
            )
        },
        "flint_dense_slots",
    )
}

/// Expert-table addresses of one projection: (codes ptrs, scale ptrs, global scales).
pub type Table = (u64, u64, u64);

#[allow(clippy::too_many_arguments)]
pub fn moe_small(
    s: &Stream,
    xn: u64,
    t: usize,
    slots: usize,
    sel_e: u64,
    sel_w: u64,
    gate: Table,
    up: Table,
    down: Table,
    act: u64,
    h: u64,
) -> Result<()> {
    check(
        unsafe {
            ffi::quench6k_flint_moe_small(
                p(xn),
                t as c_int,
                slots as c_int,
                p(sel_e),
                p(sel_w),
                p(gate.0),
                p(gate.1),
                p(gate.2),
                p(up.0),
                p(up.1),
                p(up.2),
                p(down.0),
                p(down.1),
                p(down.2),
                pm(act),
                pm(h),
                s.raw(),
            )
        },
        "flint_moe_small",
    )
}

/// Grouped-GEMM tables (as `kernels::moe_plan`) for `pairs` = t * slots entries of `sel_e` over `ne` experts.
#[allow(clippy::too_many_arguments)]
pub fn moe_plan(
    s: &Stream,
    sel_e: u64,
    pairs: usize,
    ne: usize,
    bm: usize,
    counts: u64,
    cursor: u64,
    offsets: u64,
    tile_expert: u64,
    tile_row: u64,
    num_tiles: u64,
    pair_row: u64,
) -> Result<()> {
    check(
        unsafe {
            ffi::quench6k_flint_moe_plan(
                p(sel_e),
                pairs as c_int,
                ne as c_int,
                bm as c_int,
                pm(counts),
                pm(cursor),
                pm(offsets),
                pm(tile_expert),
                pm(tile_row),
                pm(num_tiles),
                pm(pair_row),
                s.raw(),
            )
        },
        "flint_moe_plan",
    )
}

pub fn gather(
    s: &Stream,
    xn: u64,
    pair_row: u64,
    pairs: usize,
    slots: usize,
    xs: u64,
) -> Result<()> {
    check(
        unsafe {
            ffi::quench6k_flint_gather(
                p(xn),
                p(pair_row),
                pairs as c_int,
                slots as c_int,
                pm(xs),
                s.raw(),
            )
        },
        "flint_gather",
    )
}

pub fn swiglu(s: &Stream, g: u64, u: u64, count: usize, act: u64) -> Result<()> {
    check(
        unsafe { ffi::quench6k_flint_swiglu(p(g), p(u), count, pm(act), s.raw()) },
        "flint_swiglu",
    )
}

pub fn combine(
    s: &Stream,
    yd: u64,
    pair_row: u64,
    sel_w: u64,
    t: usize,
    slots: usize,
    h: u64,
) -> Result<()> {
    check(
        unsafe {
            ffi::quench6k_flint_combine(
                p(yd),
                p(pair_row),
                p(sel_w),
                t as c_int,
                slots as c_int,
                pm(h),
                s.raw(),
            )
        },
        "flint_combine",
    )
}

pub fn dense_plan(
    s: &Stream,
    t: usize,
    bm: usize,
    offsets: u64,
    tile_expert: u64,
    tile_row: u64,
    num_tiles: u64,
) -> Result<()> {
    check(
        unsafe {
            ffi::quench6k_flint_dense_plan(
                t as c_int,
                bm as c_int,
                pm(offsets),
                pm(tile_expert),
                pm(tile_row),
                pm(num_tiles),
                s.raw(),
            )
        },
        "flint_dense_plan",
    )
}
