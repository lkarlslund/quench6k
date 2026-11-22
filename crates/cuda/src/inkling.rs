//! Launchers for the Inkling 2 kernels (`kernels/inkling.cu`). Pointers are raw device addresses (0 for
//! an absent optional input); every launch is asynchronous on the given stream.

use std::ffi::{c_int, c_void};

use anyhow::Result;

use crate::{Stream, check};

mod ffi {
    use super::*;
    use crate::ffi::{Error, Stream};

    unsafe extern "C" {
        pub fn quench6k_ink_embed(tokens: *const i32, t: c_int, table: *const c_void, w: *const c_void, out: *mut f32, st: Stream) -> Error;
        pub fn quench6k_ink_rms(
            sets: c_int,
            x: *const *const f32,
            w: *const *const c_void,
            out: *const *mut c_void,
            ldo: *const c_int,
            t: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_attn_res(
            src: *const *const f32,
            ns: c_int,
            queries: *const f32,
            nq: c_int,
            t: c_int,
            out: *mut f32,
            norm_w: *const *const c_void,
            norm_o: *const *mut c_void,
            ldo: c_int,
            cv_src: *const f32,
            cv_cy: *const f32,
            cv_cw: *const f32,
            cv_ring: *mut c_void,
            cv_ring_rows: c_int,
            cv_conv: *const c_void,
            cv_base: *const f32,
            cv_start: *const i32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_msg_gates(
            n: c_int,
            hs: *const *const f32,
            m: *const *const f32,
            down: *const *const c_void,
            fg: *const *const c_void,
            out: *const *mut f32,
            t: c_int,
            cat: *mut c_void,
            z: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_qkv_prep(
            lin_q: *const f32,
            ldq: c_int,
            qscale: *const c_void,
            rscale: *const c_void,
            qclip: f32,
            rclip: f32,
            full: c_int,
            q: *mut c_void,
            rel: *mut f32,
            sink: *mut f32,
            lin_k: *const f32,
            lin_v: *const f32,
            ldk: c_int,
            kring: *mut c_void,
            vring: *mut c_void,
            kconv: *const c_void,
            vconv: *const c_void,
            kstatic: *const c_void,
            vstatic: *const c_void,
            kcache: *mut c_void,
            vcache: *mut c_void,
            ring_rows: c_int,
            cache_rows: c_int,
            start: *const i32,
            t: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_hidden_conv(
            src: *const f32,
            cy: *const f32,
            cw: *const f32,
            ring: *mut c_void,
            ring_rows: c_int,
            conv: *const c_void,
            base: *const f32,
            start: *const i32,
            t: c_int,
            out: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_attn_part_floats(t: c_int, splits: c_int) -> usize;
        pub fn quench6k_ink_attention(
            q: *const c_void,
            rel: *const f32,
            sink: *const f32,
            kcache: *const c_void,
            vcache: *const c_void,
            cache_rows: c_int,
            proj: *const c_void,
            window: c_int,
            block: c_int,
            start: *const i32,
            t: c_int,
            splits: c_int,
            min_keys: c_int,
            lin: *const f32,
            ld: c_int,
            gate_off: c_int,
            part: *mut f32,
            out: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_route(
            logits: *const f32,
            bias: *const f32,
            scale: f32,
            t: c_int,
            sel_e: *mut i32,
            sel_w: *mut f32,
            order: *mut i32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_experts(
            x: *const c_void,
            sel_e: *const i32,
            sel_w: *const f32,
            fp4: c_int,
            gu_w: *const u8,
            gu_sc: *const u8,
            gu_s2: *const f32,
            d_w: *const u8,
            d_sc: *const u8,
            d_s2: *const f32,
            gcap: f32,
            ucap: f32,
            t: c_int,
            order: *mut i32,
            ordered: c_int,
            act: *mut c_void,
            y: *mut f32,
            out: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_softcap(x: *mut f32, rows: c_int, n: c_int, ld: c_int, cap: f32, st: Stream) -> Error;
        pub fn quench6k_ink_moe_plan(
            sel_e: *const i32,
            t: c_int,
            bm: c_int,
            offsets: *mut i32,
            tile_expert: *mut i32,
            tile_row: *mut i32,
            num_tiles: *mut i32,
            pair_row: *mut i32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_moe_gather(x: *const c_void, pair_row: *const i32, t: c_int, xs: *mut c_void, st: Stream) -> Error;
        pub fn quench6k_ink_moe_act(y: *const f32, rows: c_int, gcap: f32, ucap: f32, act: *mut c_void, st: Stream) -> Error;
        pub fn quench6k_ink_moe_combine_rows(y: *const f32, pair_row: *const i32, sel_w: *const f32, t: c_int, out: *mut f32, st: Stream) -> Error;
        pub fn quench6k_ink_mx8_grouped_gemm(
            xs: *const c_void,
            w: *const u8,
            sc: *const u8,
            offsets: *const i32,
            tile_expert: *const i32,
            tile_row: *const i32,
            num_tiles: *const i32,
            max_tiles: c_int,
            y: *mut f32,
            n: c_int,
            k: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_msg_apply_rows(hs: *const f32, m: *const f32, fg: *const f32, t: c_int, out: *mut f32, st: Stream) -> Error;
        pub fn quench6k_ink_prefetch(p: *const *const c_void, bytes: *const usize, n: c_int, st: Stream) -> Error;
        pub fn quench6k_ink_markov_cand_blocks(v: c_int) -> usize;
        pub fn quench6k_ink_markov_cand(base: *const f32, fold_b: *const f32, v: c_int, n: c_int, c: c_int, cand: *mut i32, st: Stream) -> Error;
        pub fn quench6k_ink_nv4_gemm(
            xs: *const c_void,
            w_ptr: *const u64,
            s_ptr: *const u64,
            ws2: *const f32,
            offsets: *const i32,
            tile_expert: *const i32,
            tile_row: *const i32,
            num_tiles: *const i32,
            max_tiles: c_int,
            bm: c_int,
            y: *mut f32,
            n: c_int,
            k: c_int,
            act: *mut c_void,
            gcap: f32,
            ucap: f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_to_bf16_rows(src: *const f32, t: c_int, dst: *mut c_void, ldd: c_int, st: Stream) -> Error;
        pub fn quench6k_ink_mix(
            feats: *const c_void,
            rows_from: *const i32,
            alpha: *const f32,
            w: *const c_void,
            out: *mut c_void,
            ldo: c_int,
            rows: c_int,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_ctx_kv(
            kv: *const f32,
            knorm: *const c_void,
            theta: f32,
            start: *const i32,
            t: c_int,
            kcache: *mut c_void,
            vcache: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_draft_qkv(
            qkv: *const f32,
            qnorm: *const c_void,
            knorm: *const c_void,
            theta: f32,
            start: *const i32,
            t: c_int,
            q: *mut c_void,
            kcache: *mut c_void,
            vcache: *mut c_void,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_silu_mul(gu: *const f32, n: c_int, t: c_int, out: *mut c_void, st: Stream) -> Error;
        pub fn quench6k_ink_block_embed(
            anchor: *const i32,
            table: *const c_void,
            w: *const c_void,
            mask: *const c_void,
            rows: c_int,
            x: *mut f32,
            st: Stream,
        ) -> Error;
        pub fn quench6k_ink_markov_blocks(v: c_int) -> usize;
        pub fn quench6k_ink_markov(
            w: *const c_void,
            ctx: *const f32,
            bias: *const c_void,
            state_in: *const c_void,
            w1: *const c_void,
            prev: *mut i32,
            state_out: *mut c_void,
            latent: *mut c_void,
            fold_w: *const c_void,
            fold_b: *const f32,
            base: *const f32,
            v: c_int,
            cap: f32,
            best_v: *mut f32,
            best_i: *mut i32,
            out: *mut i32,
            conf: *mut f32,
            cand: *const i32,
            ncand: c_int,
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

/// out F32 [t, 2304] = bf16(w * rmsnorm(table[tokens])).
pub fn embed(s: &Stream, tokens: u64, t: usize, table: u64, w: u64, out: u64) -> Result<()> {
    go!("ink_embed", ffi::quench6k_ink_embed(i(tokens), n(t), p(table), p(w), fm(out), s.raw()))
}

/// One RMSNorm row set: x F32 [t, 2304] -> out BF16 rows at stride `ldo`, weight `w` (0: unweighted).
#[derive(Clone, Copy)]
pub struct RmsSet {
    pub x: u64,
    pub w: u64,
    pub out: u64,
    pub ldo: usize,
}

/// RMSNorm of up to 8 row sets in one launch.
pub fn rms(s: &Stream, sets: &[RmsSet], t: usize) -> Result<()> {
    let x: Vec<*const f32> = sets.iter().map(|r| f(r.x)).collect();
    let w: Vec<*const c_void> = sets.iter().map(|r| p(r.w)).collect();
    let out: Vec<*mut c_void> = sets.iter().map(|r| pm(r.out)).collect();
    let ldo: Vec<c_int> = sets.iter().map(|r| n(r.ldo)).collect();
    go!(
        "ink_rms",
        ffi::quench6k_ink_rms(n(sets.len()), x.as_ptr(), w.as_ptr(), out.as_ptr(), ldo.as_ptr(), n(t), s.raw())
    )
}

/// out BF16 [t, 2304] = w * rmsnorm(x) (w = 0: unweighted).
pub fn rms_bf16(s: &Stream, x: u64, w: u64, out: u64, t: usize) -> Result<()> {
    rms(s, &[RmsSet { x, w, out, ldo: 2304 }], t)
}

/// A sublayer output's convolution (see [`hidden_conv`]) computed inside [`attn_res`] as its last source.
#[derive(Clone, Copy, Default)]
pub struct ConvSrc {
    pub src: u64,
    pub cy: u64,
    pub cw: u64,
    pub ring: u64,
    pub ring_rows: usize,
    pub conv: u64,
    pub base: u64,
    pub start: u64,
}

/// Block attention residual of `t` tokens: `nq` queries (F32 [nq, 2304]) over the sources (F32 [t, 2304]
/// each, the partial sum last); out F32 [nq][t, 2304] (0: none); `norms`: per query (weight, BF16 output
/// rows at stride `ldo`) of its RMSNorm, or empty. With `conv`, the last source is that convolution's
/// output, computed here and stored at the last source's address.
#[allow(clippy::too_many_arguments)]
pub fn attn_res(
    s: &Stream,
    src: &[u64],
    queries: u64,
    nq: usize,
    t: usize,
    out: u64,
    norms: &[(u64, u64)],
    ldo: usize,
    conv: Option<ConvSrc>,
) -> Result<()> {
    let ptrs: Vec<*const f32> = src.iter().map(|&a| f(a)).collect();
    let w: Vec<*const c_void> = norms.iter().map(|x| p(x.0)).collect();
    let o: Vec<*mut c_void> = norms.iter().map(|x| pm(x.1)).collect();
    let c = conv.unwrap_or_default();
    go!(
        "ink_attn_res",
        ffi::quench6k_ink_attn_res(
            ptrs.as_ptr(),
            n(src.len()),
            f(queries),
            n(nq),
            n(t),
            fm(out),
            if norms.is_empty() { std::ptr::null() } else { w.as_ptr() },
            if norms.is_empty() { std::ptr::null() } else { o.as_ptr() },
            n(ldo),
            f(c.src),
            f(c.cy),
            f(c.cw),
            pm(c.ring),
            n(c.ring_rows),
            p(c.conv),
            f(c.base),
            i(c.start),
            s.raw()
        )
    )
}

/// One message gate: input `hs`, message `m`, weights down [64, 4608] and forget | gain [4608, 64], output.
#[derive(Clone, Copy)]
pub struct MsgGate {
    pub hs: u64,
    pub m: u64,
    pub down: u64,
    pub fg: u64,
    pub out: u64,
}

/// Up to 4 message gates in one group (3 launches): out = (1 - tanh(F z)) hs + tanh(G z) m,
/// z = bf16(down [rms(hs), rms(m)]); cat BF16 [n, t, 4608] and z BF16 [n, t, 64] are scratch.
pub fn msg_gates(s: &Stream, gates: &[MsgGate], t: usize, cat: u64, z: u64) -> Result<()> {
    let hs: Vec<*const f32> = gates.iter().map(|g| f(g.hs)).collect();
    let m: Vec<*const f32> = gates.iter().map(|g| f(g.m)).collect();
    let down: Vec<*const c_void> = gates.iter().map(|g| p(g.down)).collect();
    let fg: Vec<*const c_void> = gates.iter().map(|g| p(g.fg)).collect();
    let out: Vec<*mut f32> = gates.iter().map(|g| fm(g.out)).collect();
    go!(
        "ink_msg_gates",
        ffi::quench6k_ink_msg_gates(
            n(gates.len()),
            hs.as_ptr(),
            m.as_ptr(),
            down.as_ptr(),
            fg.as_ptr(),
            out.as_ptr(),
            n(t),
            pm(cat),
            pm(z),
            s.raw()
        )
    )
}

/// Keys or values of [`qkv_prep`]: projection ring, static taps (sliding: `conv`; full: `static_w`), cache.
#[derive(Clone, Copy)]
pub struct KvSide {
    pub ring: u64,
    pub conv: u64,
    pub static_w: u64,
    pub cache: u64,
}

/// The query side of an attention layer (q | gate | relative states | sinks rows) and the key and value
/// convolutions into the caches (`KvSide.conv` [1024, 4] for sliding layers, else `static_w` [4, 1024] with
/// the per-token taps after the projections in `lin_k` / `lin_v`).
#[allow(clippy::too_many_arguments)]
pub fn qkv_prep(
    s: &Stream,
    lin_q: u64,
    ldq: usize,
    qscale: u64,
    rscale: u64,
    qclip: f32,
    rclip: f32,
    full: bool,
    q: u64,
    rel: u64,
    sink: u64,
    lin_k: u64,
    lin_v: u64,
    ldk: usize,
    k: KvSide,
    v: KvSide,
    ring_rows: usize,
    cache_rows: usize,
    start: u64,
    t: usize,
) -> Result<()> {
    go!(
        "ink_qkv_prep",
        ffi::quench6k_ink_qkv_prep(
            f(lin_q),
            n(ldq),
            p(qscale),
            p(rscale),
            qclip,
            rclip,
            full as c_int,
            pm(q),
            fm(rel),
            fm(sink),
            f(lin_k),
            f(lin_v),
            n(ldk),
            pm(k.ring),
            pm(v.ring),
            p(k.conv),
            p(v.conv),
            p(k.static_w),
            p(v.static_w),
            pm(k.cache),
            pm(v.cache),
            n(ring_rows),
            n(cache_rows),
            i(start),
            n(t),
            s.raw()
        )
    )
}

/// out F32 [t, 2304] = base + (conv(x) + x) / sqrt(5) (base 0: none), x from `src` or, with `cy` (F32
/// [t * 8, 2304]) and `cw` (F32 [t * 8]), the experts' combine; the step's rows join the ring.
#[allow(clippy::too_many_arguments)]
pub fn hidden_conv(
    s: &Stream,
    src: u64,
    cy: u64,
    cw: u64,
    ring: u64,
    ring_rows: usize,
    conv: u64,
    base: u64,
    start: u64,
    t: usize,
    out: u64,
) -> Result<()> {
    go!(
        "ink_hidden_conv",
        ffi::quench6k_ink_hidden_conv(f(src), f(cy), f(cw), pm(ring), n(ring_rows), p(conv), f(base), i(start), n(t), fm(out), s.raw())
    )
}

pub fn attn_part_floats(t: usize, splits: usize) -> usize {
    unsafe { ffi::quench6k_ink_attn_part_floats(n(t), n(splits)) }
}

/// Attention of `t` queries over the caches, merged with the sinks and gated: out BF16 [t, 3072]. Keys of
/// query t end at its own position (or with `block` > 0 at the block's last, start + block - 1) and span
/// `window` positions (0: all); `rel`, `sink` and `lin` (the gate) are optional (0).
#[allow(clippy::too_many_arguments)]
pub fn attention(
    s: &Stream,
    q: u64,
    rel: u64,
    sink: u64,
    kcache: u64,
    vcache: u64,
    cache_rows: usize,
    proj: u64,
    window: usize,
    block: usize,
    start: u64,
    t: usize,
    splits: usize,
    min_keys: usize,
    lin: u64,
    ld: usize,
    gate_off: usize,
    part: u64,
    out: u64,
) -> Result<()> {
    go!(
        "ink_attention",
        ffi::quench6k_ink_attention(
            p(q),
            f(rel),
            f(sink),
            p(kcache),
            p(vcache),
            n(cache_rows),
            p(proj),
            n(window),
            n(block),
            i(start),
            n(t),
            n(splits),
            n(min_keys),
            f(lin),
            n(ld),
            n(gate_off),
            fm(part),
            pm(out),
            s.raw()
        )
    )
}

/// Routing; with `order` != 0 (t <= 16) also the pairs' order by expert for [`experts`].
#[allow(clippy::too_many_arguments)]
pub fn route(s: &Stream, logits: u64, bias: u64, scale: f32, t: usize, sel_e: u64, sel_w: u64, order: u64) -> Result<()> {
    go!("ink_route", ffi::quench6k_ink_route(f(logits), f(bias), scale, n(t), im(sel_e), fm(sel_w), im(order), s.raw()))
}

/// One expert stack: weights, block scales and (NVFP4) per-expert scales.
#[derive(Clone, Copy)]
pub struct ExpertStack {
    pub w: u64,
    pub sc: u64,
    pub s2: u64,
}

/// The routed experts of `t` tokens (x BF16 [t, 2304]) into y (per pair) and out F32 [t, 2304] (0: no
/// combine); `ordered`: `order` already holds the pairs' order ([`route`]).
#[allow(clippy::too_many_arguments)]
pub fn experts(
    s: &Stream,
    x: u64,
    sel_e: u64,
    sel_w: u64,
    fp4: bool,
    gu: ExpertStack,
    down: ExpertStack,
    gcap: f32,
    ucap: f32,
    t: usize,
    order: u64,
    ordered: bool,
    act: u64,
    y: u64,
    out: u64,
) -> Result<()> {
    go!(
        "ink_experts",
        ffi::quench6k_ink_experts(
            p(x),
            i(sel_e),
            f(sel_w),
            fp4 as c_int,
            u8p(gu.w),
            u8p(gu.sc),
            f(gu.s2),
            u8p(down.w),
            u8p(down.sc),
            f(down.s2),
            gcap,
            ucap,
            n(t),
            im(order),
            ordered as c_int,
            pm(act),
            fm(y),
            fm(out),
            s.raw()
        )
    )
}

pub fn softcap(s: &Stream, x: u64, rows: usize, nn: usize, ld: usize, cap: f32) -> Result<()> {
    go!("ink_softcap", ffi::quench6k_ink_softcap(fm(x), n(rows), n(nn), n(ld), cap, s.raw()))
}

/// F32 rows [t, 2304] to BF16 rows at stride `ldd`.
pub fn to_bf16_rows(s: &Stream, src: u64, t: usize, dst: u64, ldd: usize) -> Result<()> {
    go!("ink_to_bf16_rows", ffi::quench6k_ink_to_bf16_rows(f(src), n(t), pm(dst), n(ldd), s.raw()))
}

/// `rows` rows of bf16(w * rmsnorm(sum_k alpha[k] feats[row, k])) at stride `ldo`; row = r, or the device
/// int at `rows_from` for every r.
#[allow(clippy::too_many_arguments)]
pub fn mix(s: &Stream, feats: u64, rows_from: u64, alpha: u64, w: u64, out: u64, ldo: usize, rows: usize) -> Result<()> {
    go!(
        "ink_mix",
        ffi::quench6k_ink_mix(p(feats), i(rows_from), f(alpha), p(w), pm(out), n(ldo), n(rows), s.raw())
    )
}

#[allow(clippy::too_many_arguments)]
pub fn ctx_kv(s: &Stream, kv: u64, knorm: u64, theta: f32, start: u64, t: usize, kcache: u64, vcache: u64) -> Result<()> {
    go!(
        "ink_ctx_kv",
        ffi::quench6k_ink_ctx_kv(f(kv), p(knorm), theta, i(start), n(t), pm(kcache), pm(vcache), s.raw())
    )
}

#[allow(clippy::too_many_arguments)]
pub fn draft_qkv(
    s: &Stream,
    qkv: u64,
    qnorm: u64,
    knorm: u64,
    theta: f32,
    start: u64,
    t: usize,
    q: u64,
    kcache: u64,
    vcache: u64,
) -> Result<()> {
    go!(
        "ink_draft_qkv",
        ffi::quench6k_ink_draft_qkv(f(qkv), p(qnorm), p(knorm), theta, i(start), n(t), pm(q), pm(kcache), pm(vcache), s.raw())
    )
}

pub fn silu_mul(s: &Stream, gu: u64, nn: usize, t: usize, out: u64) -> Result<()> {
    go!("ink_silu_mul", ffi::quench6k_ink_silu_mul(f(gu), n(nn), n(t), pm(out), s.raw()))
}

pub fn block_embed(s: &Stream, anchor: u64, table: u64, w: u64, mask: u64, rows: usize, x: u64) -> Result<()> {
    go!(
        "ink_block_embed",
        ffi::quench6k_ink_block_embed(i(anchor), p(table), p(w), p(mask), n(rows), fm(x), s.raw())
    )
}

pub fn markov_blocks(v: usize) -> usize {
    unsafe { ffi::quench6k_ink_markov_blocks(n(v)) }
}

/// The Markov head's weights: W [768, 512] (state | previous token), bias, w1 [vocab, 256], folded LM head.
#[derive(Clone, Copy)]
pub struct MarkovW {
    pub w: u64,
    pub bias: u64,
    pub w1: u64,
    pub fold_w: u64,
    pub fold_b: u64,
}

/// One Markov step: state_in -> state_out, the pick (from `base` logits F32 [vocab]) to `out` and `prev`,
/// its probability to `conf` (F32). `best_v` holds 2 * markov_blocks(vocab) values.
#[allow(clippy::too_many_arguments)]
pub fn markov(
    s: &Stream,
    m: MarkovW,
    ctx: u64,
    state_in: u64,
    prev: u64,
    state_out: u64,
    latent: u64,
    base: u64,
    vocab: usize,
    cap: f32,
    best_v: u64,
    best_i: u64,
    out: u64,
    conf: u64,
    cand: Option<(u64, usize)>,
) -> Result<()> {
    go!(
        "ink_markov",
        ffi::quench6k_ink_markov(
            p(m.w),
            f(ctx),
            p(m.bias),
            p(state_in),
            p(m.w1),
            im(prev),
            pm(state_out),
            pm(latent),
            p(m.fold_w),
            f(m.fold_b),
            f(base),
            n(vocab),
            cap,
            fm(best_v),
            im(best_i),
            im(out),
            fm(conf),
            cand.map_or(std::ptr::null(), |c| c.0 as *const i32),
            cand.map_or(0, |c| n(c.1)),
            s.raw()
        )
    )
}

/// Groups `t` tokens' routed pairs by expert into tiles of `bm` rows (see moe.cu's NVFP4 GEMM).
#[allow(clippy::too_many_arguments)]
pub fn moe_plan(
    s: &Stream,
    sel_e: u64,
    t: usize,
    bm: usize,
    offsets: u64,
    tile_expert: u64,
    tile_row: u64,
    num_tiles: u64,
    pair_row: u64,
) -> Result<()> {
    go!(
        "ink_moe_plan",
        ffi::quench6k_ink_moe_plan(i(sel_e), n(t), n(bm), im(offsets), im(tile_expert), im(tile_row), im(num_tiles), im(pair_row), s.raw())
    )
}

pub fn moe_gather(s: &Stream, x: u64, pair_row: u64, t: usize, xs: u64) -> Result<()> {
    go!("ink_moe_gather", ffi::quench6k_ink_moe_gather(p(x), i(pair_row), n(t), pm(xs), s.raw()))
}

pub fn moe_act(s: &Stream, y: u64, rows: usize, gcap: f32, ucap: f32, act: u64) -> Result<()> {
    go!("ink_moe_act", ffi::quench6k_ink_moe_act(f(y), n(rows), gcap, ucap, pm(act), s.raw()))
}

pub fn moe_combine_rows(s: &Stream, y: u64, pair_row: u64, sel_w: u64, t: usize, out: u64) -> Result<()> {
    go!("ink_moe_combine_rows", ffi::quench6k_ink_moe_combine_rows(f(y), i(pair_row), f(sel_w), n(t), fm(out), s.raw()))
}

/// Grouped MXFP8 expert GEMM over a [`moe_plan`] (tiles of 64 rows): y [rows, n] = xs [rows, k] . W_e^T.
#[allow(clippy::too_many_arguments)]
pub fn mx8_grouped_gemm(
    s: &Stream,
    xs: u64,
    w: ExpertStack,
    offsets: u64,
    tile_expert: u64,
    tile_row: u64,
    num_tiles: u64,
    max_tiles: usize,
    y: u64,
    nn: usize,
    k: usize,
) -> Result<()> {
    go!(
        "ink_mx8_grouped_gemm",
        ffi::quench6k_ink_mx8_grouped_gemm(
            p(xs),
            u8p(w.w),
            u8p(w.sc),
            i(offsets),
            i(tile_expert),
            i(tile_row),
            i(num_tiles),
            n(max_tiles),
            fm(y),
            n(nn),
            n(k),
            s.raw()
        )
    )
}

/// out = (1 - tanh(f)) * hs + tanh(g) * m from fg F32 [t, 2 * 2304] (forget | gain).
pub fn msg_apply_rows(s: &Stream, hs: u64, m: u64, fg: u64, t: usize, out: u64) -> Result<()> {
    go!("ink_msg_apply_rows", ffi::quench6k_ink_msg_apply_rows(f(hs), f(m), f(fg), n(t), fm(out), s.raw()))
}

/// Requests up to 16 (address, bytes) ranges of constants into L2 (asynchronous, no result).
pub fn prefetch(s: &Stream, ranges: &[(u64, usize)]) -> Result<()> {
    let ptrs: Vec<*const c_void> = ranges.iter().map(|r| p(r.0)).collect();
    let bytes: Vec<usize> = ranges.iter().map(|r| r.1).collect();
    go!("ink_prefetch", ffi::quench6k_ink_prefetch(ptrs.as_ptr(), bytes.as_ptr(), n(ranges.len()), s.raw()))
}

/// Blocks of the Markov candidates' vocabulary split (candidates per base row = blocks * c).
pub fn markov_cand_blocks(v: usize) -> usize {
    unsafe { ffi::quench6k_ink_markov_cand_blocks(n(v)) }
}

/// The Markov head's candidates: per 256-row vocabulary block and base row, the `c` best by base + fold_b.
pub fn markov_cand(s: &Stream, base: u64, fold_b: u64, vocab: usize, rows: usize, c: usize, cand: u64) -> Result<()> {
    go!("ink_markov_cand", ffi::quench6k_ink_markov_cand(f(base), f(fold_b), n(vocab), n(rows), n(c), im(cand), s.raw()))
}

/// W4A16 NVFP4 grouped GEMM with tiles of `bm` rows (64 or 128, as planned), F32 out; with `act`
/// (gate/up) the activation BF16 [rows, nn / 2] (caps `caps`) instead.
#[allow(clippy::too_many_arguments)]
pub fn nv4_gemm(
    s: &Stream,
    xs: u64,
    w_ptr: u64,
    s_ptr: u64,
    ws2: u64,
    offsets: u64,
    tile_expert: u64,
    tile_row: u64,
    num_tiles: u64,
    max_tiles: usize,
    bm: usize,
    y: u64,
    nn: usize,
    k: usize,
    act: Option<(u64, f32, f32)>,
) -> Result<()> {
    let (a, gcap, ucap) = act.unwrap_or((0, 0.0, 0.0));
    go!(
        "ink_nv4_gemm",
        ffi::quench6k_ink_nv4_gemm(
            p(xs),
            w_ptr as *const u64,
            s_ptr as *const u64,
            f(ws2),
            im(offsets),
            im(tile_expert),
            im(tile_row),
            im(num_tiles),
            n(max_tiles),
            n(bm),
            fm(y),
            n(nn),
            n(k),
            pm(a),
            gcap,
            ucap,
            s.raw()
        )
    )
}
