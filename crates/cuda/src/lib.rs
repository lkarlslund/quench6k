//! Minimal bindings to the CUDA runtime and to squash's kernels.
//!
//! squash links its own nvcc-built kernels against cudart, so the host side uses the same runtime
//! API and the device's primary context throughout.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::ptr::null_mut;

use anyhow::{Result, bail};

pub mod ffi {
    use super::*;

    pub type Error = c_int;
    pub type Stream = *mut c_void;
    pub type Event = *mut c_void;

    pub const SUCCESS: Error = 0;
    pub const MEMCPY_HOST_TO_DEVICE: c_int = 1;
    pub const MEMCPY_DEVICE_TO_HOST: c_int = 2;
    pub const MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
    pub const MEMCPY_DEFAULT: c_int = 4;
    pub const STREAM_NON_BLOCKING: c_uint = 1;
    pub const EVENT_DISABLE_TIMING: c_uint = 2;
    pub const HOST_ALLOC_PORTABLE: c_uint = 1;

    unsafe extern "C" {
        pub fn cudaGetErrorString(err: Error) -> *const c_char;
        pub fn squash_set_variant(key: *const c_char, value: c_int);
        pub fn squash_clear_variants();
        pub fn cudaSetDevice(device: c_int) -> Error;
        pub fn cudaDeviceSynchronize() -> Error;
        pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> Error;
        pub fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> Error;
        pub fn cudaFree(ptr: *mut c_void) -> Error;
        pub fn cudaHostAlloc(ptr: *mut *mut c_void, bytes: usize, flags: c_uint) -> Error;
        pub fn cudaFreeHost(ptr: *mut c_void) -> Error;
        pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int, stream: Stream) -> Error;
        pub fn cudaMemsetAsync(dst: *mut c_void, value: c_int, bytes: usize, stream: Stream) -> Error;
        pub fn cudaStreamCreateWithFlags(stream: *mut Stream, flags: c_uint) -> Error;
        pub fn cudaStreamDestroy(stream: Stream) -> Error;
        pub fn cudaStreamSynchronize(stream: Stream) -> Error;
        pub fn cudaEventCreateWithFlags(event: *mut Event, flags: c_uint) -> Error;
        pub fn cudaEventDestroy(event: Event) -> Error;
        pub fn cudaEventRecord(event: Event, stream: Stream) -> Error;
        pub fn cudaEventSynchronize(event: Event) -> Error;
        pub fn cudaStreamWaitEvent(stream: Stream, event: Event, flags: c_uint) -> Error;
        pub fn squash_variant_get(key: *const c_char, fallback: c_int) -> c_int;
        pub fn squash_l2_prefetch(ranges: *const u64, n: c_int, stream: Stream) -> Error;
        pub fn cudaEventElapsedTime(ms: *mut f32, start: Event, end: Event) -> Error;

        pub fn cudaStreamBeginCapture(stream: Stream, mode: c_int) -> Error;
        pub fn cudaStreamEndCapture(stream: Stream, graph: *mut *mut c_void) -> Error;
        pub fn cudaGraphInstantiateWithFlags(exec: *mut *mut c_void, graph: *mut c_void, flags: u64) -> Error;
        pub fn cudaProfilerStart() -> Error;
        pub fn cudaProfilerStop() -> Error;
        pub fn cudaGraphLaunch(exec: *mut c_void, stream: Stream) -> Error;
        pub fn cudaGraphExecDestroy(exec: *mut c_void) -> Error;
        pub fn cudaGraphDestroy(graph: *mut c_void) -> Error;

        pub fn squash_checksum(data: *const c_void, bytes: usize, out: *mut u64, stream: Stream) -> Error;
        pub fn squash_gemm_bf16_f32(fmt: c_int, x: *const c_void, ldx: c_int, w: *const c_void, ldw: c_int, scale: *const f32, y: *mut f32, ldy: c_int, m: c_int, n: c_int, k: c_int, alpha: f32, accumulate: c_int, stream: Stream) -> Error;
        pub fn squash_gemm_bf16_bf16(fmt: c_int, x: *const c_void, ldx: c_int, w: *const c_void, ldw: c_int, scale: *const f32, y: *mut c_void, ldy: c_int, m: c_int, n: c_int, k: c_int, alpha: f32, accumulate: c_int, stream: Stream) -> Error;
        pub fn squash_vit_layernorm(x: *const f32, t: c_int, d: c_int, w: *const c_void, b: *const c_void, out: *mut c_void, stream: Stream) -> Error;
        pub fn squash_vit_bias_act(y: *const f32, t: c_int, n: c_int, bias: *const c_void, act: c_int, out: *mut c_void, ldo: c_int, stream: Stream) -> Error;
        pub fn squash_vit_residual(h: *mut f32, y: *const f32, bias: *const c_void, extra: *const f32, t: c_int, n: c_int, assign: c_int, stream: Stream) -> Error;
        pub fn squash_vit_qkv_rope(qkv: *const f32, bias: *const c_void, cos: *const f32, sin: *const f32, t: c_int, q: *mut f32, k: *mut f32, v: *mut f32, stream: Stream) -> Error;
        pub fn squash_vit_attention(q: *const f32, k: *const f32, v: *const f32, t: c_int, seg: c_int, out: *mut c_void, stream: Stream) -> Error;
        pub fn squash_to_bf16(x: *const f32, n: usize, out: *mut c_void, stream: Stream) -> Error;
        pub fn squash_embed_rows(emb: *const f32, rows: c_int, h: *mut f32, stream: Stream) -> Error;
        pub fn squash_pad_cols(src: *const c_void, n: c_int, k: c_int, dst: *mut c_void, ldd: c_int, stream: Stream) -> Error;
        pub fn squash_gemv_max_m() -> c_int;
        pub fn squash_gemv(fmt: c_int, m: c_int, x: *const c_void, ldx: c_int, w: *const c_void, scale: *const f32, y: *mut f32, ldy: c_int, partial: *mut f32, n: c_int, k: c_int, splits: c_int, alpha: f32, accumulate: c_int, stream: Stream) -> Error;
        pub fn squash_fold_norm(w: *mut c_void, g: *const c_void, n: c_int, k: c_int, stream: Stream) -> Error;
        pub fn squash_hc_small_scratch_floats(t: c_int) -> usize;
        pub fn squash_hc_small(fmt: c_int, h: *mut f32, t: c_int, y: *const f32, inj_prev: *const f32, wd: *const c_void, sd: *const f32, wi: *const c_void, si: *const f32, wu: *const c_void, su: *const f32, g: *const c_void, scratch: *mut f32, z: *mut c_void, r: *mut f32, inj: *mut f32, m: *mut c_void, stream: Stream) -> Error;
        pub fn squash_int8_group() -> c_int;
        pub fn squash_dequant_int8_groups(q: *const c_void, scale: *const f32, n: c_int, k: c_int, out: *mut c_void, stream: Stream) -> Error;
        pub fn squash_quantize_int8_groups(w: *const c_void, n: c_int, k: c_int, q: *mut c_void, scale: *mut f32, stream: Stream) -> Error;
        pub fn squash_moe_active(counts: *const i32, active: *mut i32, num_active: *mut i32, stream: Stream) -> Error;
        pub fn squash_topk(x: *const f32, rows: c_int, n: c_int, ld: usize, k: c_int, sort_by_index: c_int, out_idx: *mut i32, out_val: *mut f32, stream: Stream) -> Error;
        pub fn squash_argmax_scratch(rows: c_int, n: c_int) -> usize;
        pub fn squash_argmax(x: *const f32, rows: c_int, n: c_int, ld: usize, scratch: *mut c_void, out: *mut i32, stream: Stream) -> Error;
        pub fn squash_row_lse(x: *const f32, rows: c_int, n: c_int, ld: usize, targets: *const i32, lse: *mut f32, target_logit: *mut f32, stream: Stream) -> Error;
        pub fn squash_embed(tokens: *const i32, t: c_int, table: *const c_void, h: *mut f32, stream: Stream) -> Error;
        pub fn squash_grouped_rmsnorm(h: *const f32, w: *const c_void, n: *mut c_void, t: c_int, stream: Stream) -> Error;
        pub fn squash_rmsnorm_rows(x: *const f32, ldx: c_int, w: *const c_void, out: *mut c_void, rows: c_int, stream: Stream) -> Error;
        pub fn squash_hc_inject(n: *const c_void, w: *const c_void, inj: *mut f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_hc_act(x: *const f32, ld: c_int, z: *mut c_void, inj: *mut f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_hc_mix(u: *const f32, n: *const c_void, g: *const c_void, m: *mut c_void, t: c_int, stream: Stream) -> Error;
        pub fn squash_hc_combine(h: *mut f32, y: *const f32, inj: *const f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_f32_to_bf16(x: *const f32, y: *mut c_void, count: usize, stream: Stream) -> Error;
        pub fn squash_add_f32(y: *mut f32, x: *const f32, count: usize, stream: Stream) -> Error;
        pub fn squash_gdn_conv(x: *const f32, ldx: c_int, out: *mut f32, state: *mut f32, w: *const c_void, t: c_int, stream: Stream) -> Error;
        pub fn squash_gdn_prep(qkv: *mut f32, a: *const f32, b: *const f32, ldab: c_int, a_log: *const c_void, dt_bias: *const c_void, beta: *mut f32, g: *mut f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_gdn_recurrent(qkv: *const f32, beta: *const f32, g: *const f32, state: *mut f32, out: *mut f32, t: c_int, scratch: *mut f32, stream: Stream) -> Error;
        pub fn squash_gdn_chunk_scratch_floats(t: c_int) -> usize;
        pub fn squash_gdn_conv_tickets() -> usize;
        pub fn squash_gdn_fused_small(proj: *const f32, ld: c_int, t: c_int, conv_w: *const c_void, conv_state: *mut f32, state: *mut f32, a_log: *const c_void, dt_bias: *const c_void, norm_w: *const c_void, z_col: c_int, a_col: c_int, b_col: c_int, y: *mut c_void, rec: *mut f32, ctl: *const i32, defer: c_int, stream: Stream) -> Error;
        pub fn squash_gdn_record_floats() -> usize;
        pub fn squash_ple_record_floats() -> usize;
        pub fn squash_gdn_gated_norm(o: *const f32, z: *const f32, ldz: c_int, w: *const c_void, y: *mut c_void, t: c_int, stream: Stream) -> Error;
        pub fn squash_qsa_qkv_prep(qg: *const f32, kv: *const f32, ld: c_int, q_norm: *const c_void, k_norm: *const c_void, q: *mut f32, kcache: *mut c_void, vcache: *mut c_void, pos3: *const i32, start: *const i32, t: c_int, fp8: c_int, stream: Stream) -> Error;
        pub fn squash_qsa_index_prep(idx: *const f32, ld: c_int, q_ln: *const c_void, iq: *mut f32, raw: *mut c_void, pos3: *const i32, start: *const i32, t: c_int, stream: Stream) -> Error;
        pub fn squash_qsa_pool(raw: *const c_void, k_ln: *const c_void, pooled: *mut c_void, pos3: *const i32, start: *const i32, t: c_int, stream: Stream) -> Error;
        pub fn squash_qsa_index_scores(iq: *const f32, pooled: *const c_void, scores: *mut f32, ld: c_int, start: *const i32, t: c_int, stream: Stream) -> Error;
        pub fn squash_qsa_select(scores: *const f32, t: c_int, ld: usize, start: *const i32, sel: *mut i32, stream: Stream) -> Error;
        pub fn squash_store_positions(src: *const i32, pos3: *mut i32, start: *const i32, t: c_int, stream: Stream) -> Error;
        pub fn squash_qsa_attention(q: *const f32, qg: *const f32, ld: c_int, kcache: *const c_void, vcache: *const c_void, sel: *const i32, start: *const i32, t: c_int, fp8: c_int, out: *mut c_void, stream: Stream) -> Error;
        pub fn squash_moe_tile_rows() -> c_int;
        pub fn squash_moe_small_tokens() -> c_int;
        pub fn squash_moe_route_small(logits: *const f32, ld: c_int, t: c_int, top_i: *mut i32, top_w: *mut f32, offsets: *mut i32, pair_row: *mut i32, row_token: *mut i32, active: *mut i32, num_active: *mut i32, stream: Stream) -> Error;
        pub fn squash_moe_gate_up_small(m: *const c_void, row_token: *const i32, gw: *const u64, gs: *const u64, g2: *const f32, uw: *const u64, us: *const u64, u2: *const f32, offsets: *const i32, active: *const i32, num_active: *const i32, max_active: c_int, t: c_int, act: *mut c_void, stream: Stream) -> Error;
        pub fn squash_moe_down_small(act: *const c_void, dw: *const u64, ds: *const u64, d2: *const f32, offsets: *const i32, active: *const i32, num_active: *const i32, max_active: c_int, t: c_int, yd: *mut f32, stream: Stream) -> Error;
        pub fn squash_moe_shared_combine(fmt: c_int, proj: *const f32, ld: c_int, t: c_int, w: *const c_void, scale: *const f32, yd: *const f32, pair_row: *const i32, top_w: *const f32, y: *mut f32, stream: Stream) -> Error;
        pub fn squash_moe_experts_t1(m: *const c_void, logits: *const f32, gw: *const u64, gs: *const u64, g2: *const f32, uw: *const u64, us: *const u64, u2: *const f32, dw: *const u64, ds: *const u64, d2: *const f32, top_i: *mut i32, top_w: *mut f32, pair_row: *mut i32, act: *mut c_void, yd: *mut f32, stream: Stream) -> Error;
        pub fn squash_quantize_nvfp4(w: *const c_void, count: c_int, n: c_int, k: c_int, codes: *mut c_void, scales: *mut c_void, scale2: *mut f32, amax: *mut u32, stream: Stream) -> Error;
        pub fn squash_spec_layout(field: c_int) -> c_int;
        pub fn squash_spec_mtp_header(rc: *const i32, hdr: *mut i32, tokens_off: c_int, pos_off: c_int, j: c_int, c: c_int, stream: Stream) -> Error;
        pub fn squash_fake_mxfp8(x: *mut c_void, rows: c_int, cols: c_int, ld: c_int, stream: Stream) -> Error;
        pub fn squash_fake_fp4_act(x: *mut c_void, rows: c_int, cols: c_int, stream: Stream) -> Error;
        pub fn squash_spec_accept(rc: *mut i32, ctl: *mut i32, n: c_int, stream: Stream) -> Error;
        pub fn squash_spec_sample_rows(ids: *const i32, vals: *const f32, k_list: c_int, n: c_int, rc: *mut i32, params: *mut c_void, qs: *const c_void, stream: Stream) -> Error;
        pub fn squash_steer_max_rank() -> c_int;
        pub fn squash_steer(h: *mut f32, t: c_int, directions: *const f32, ranks: *const i32, masks: *const i32, norm_preserve: c_int, strength: *const f32, layer: c_int, stream: Stream) -> Error;
        pub fn squash_draft_sketch_bytes(n: c_int) -> usize;
        pub fn squash_draft_slices(n: c_int) -> c_int;
        pub fn squash_build_draft_sketch(w: *const c_void, sketch: *mut c_void, n: c_int, stream: Stream) -> Error;
        pub fn squash_draft_head_argmax(x: *const c_void, sketch: *const c_void, codes: *const c_void, scales: *const c_void, scale2: *const f32, ys: *mut f32, pick: *mut c_void, n: c_int, out: *mut i32, params: *mut c_void, q: *mut c_void, stream: Stream) -> Error;
        pub fn squash_draft_picks() -> c_int;
        pub fn squash_nvfp4_dense_gemv(x: *const c_void, codes: *const c_void, scales: *const c_void, scale2: *const f32, y: *mut f32, n: c_int, stream: Stream) -> Error;
        pub fn squash_qsa_split_floats(t: c_int) -> usize;
        pub fn squash_qsa_attention_split(q: *const f32, qg: *const f32, ld: c_int, kcache: *const c_void, vcache: *const c_void, sel: *const i32, start: *const i32, t: c_int, fp8: c_int, part: *mut f32, out: *mut c_void, stream: Stream) -> Error;
        pub fn squash_moe_route(logits: *const f32, ld: c_int, t: c_int, top_i: *mut i32, top_w: *mut f32, stream: Stream) -> Error;
        pub fn squash_moe_plan(top_i: *const i32, t: c_int, counts: *mut i32, cursor: *mut i32, offsets: *mut i32, tile_expert: *mut i32, tile_row: *mut i32, num_tiles: *mut i32, pair_row: *mut i32, tile_rows: c_int, stream: Stream) -> Error;
        pub fn squash_moe_gather(m: *const c_void, pair_row: *const i32, t: c_int, xs: *mut c_void, stream: Stream) -> Error;
        pub fn squash_fp4_tile_rows() -> c_int;
        pub fn squash_hc_moe_combine_norm(h: *mut f32, yd: *const c_void, pair_row: *const i32, top_w: *const f32, sg: *const f32, ld_sg: c_int, sh: *const f32, inj: *const f32, n: *mut c_void, t: c_int, stream: Stream) -> Error;
        pub fn squash_gdn_conv_prep(x: *const f32, ldx: c_int, out: *mut f32, state: *mut f32, w: *const c_void, a: *const f32, b: *const f32, ldab: c_int, a_log: *const c_void, dt_bias: *const c_void, beta: *mut f32, g: *mut f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_gemm_wide(x: *const c_void, ldx: c_int, w: *const c_void, ldw: c_int, scale: *const f32, y: *mut f32, ldy: c_int, m: c_int, n: c_int, k: c_int, stream: Stream) -> Error;
        pub fn squash_gemm_bf16_f32_split(fmt: c_int, x: *const c_void, ldx: c_int, w: *const c_void, ldw: c_int, scale: *const f32, y: *mut f32, ldy: c_int, m: c_int, n: c_int, k: c_int, alpha: f32, accumulate: c_int, partial: *mut f32, splits: c_int, stream: Stream) -> Error;
        pub fn squash_mxfp8_rows(x: *const c_void, rows: c_int, cols: c_int, ld: c_int, codes: *mut u8, scales: *mut u8, stream: Stream) -> Error;
        pub fn squash_gemm_mx8(a: *const u8, a_scales: *const u8, w: *const u8, w_scales: *const u8, y: *mut f32, ldy: c_int, m: c_int, n: c_int, k: c_int, stream: Stream) -> Error;
        pub fn squash_moe_fp4_gate_up(a: *const u8, a_scales: *const u8, a_rscale: *const f32, w_ptr: *const u64, s_ptr: *const u64, ws2: *const f32, u_ptr: *const u64, us_ptr: *const u64, us2: *const f32, offsets: *const i32, tile_expert: *const i32, tile_row: *const i32, num_tiles: *const i32, max_tiles: c_int, act: *mut c_void, n: c_int, k: c_int, stream: Stream) -> Error;
        pub fn squash_gdn_tc_scratch_floats(t: c_int) -> usize;
        pub fn squash_gdn_chunked_tc(qkv: *const f32, beta: *const f32, g: *const f32, state: *mut f32, out: *mut f32, t: c_int, scratch: *mut f32, stream: Stream) -> Error;
        pub fn squash_hc_combine_norm(h: *mut f32, y: *const f32, inj: *const f32, n: *mut c_void, t: c_int, stream: Stream) -> Error;
        pub fn squash_gemm_hc_up_mix(fmt: c_int, x: *const c_void, ldx: c_int, w: *const c_void, ldw: c_int, scale: *const f32, n: *const c_void, g: *const c_void, m: *mut c_void, rows: c_int, k: c_int, stream: Stream) -> Error;
        pub fn squash_fp4_quant_rows(x: *const c_void, rows: c_int, k: c_int, codes: *mut u8, scales: *mut u8, rscale: *mut f32, stream: Stream) -> Error;
        pub fn squash_fp4_gather(codes: *const u8, scales: *const u8, rscale: *const f32, pair_row: *const i32, t: c_int, k: c_int, gcodes: *mut u8, gscales: *mut u8, grscale: *mut f32, stream: Stream) -> Error;
        pub fn squash_moe_fp4_gemm(a: *const u8, a_scales: *const u8, a_rscale: *const f32, w_ptr: *const u64, s_ptr: *const u64, ws2: *const f32, offsets: *const i32, tile_expert: *const i32, tile_row: *const i32, num_tiles: *const i32, max_tiles: c_int, y: *mut c_void, n: c_int, k: c_int, out_bf16: c_int, stream: Stream) -> Error;
        pub fn squash_moe_combine_bf16(yd: *const c_void, pair_row: *const i32, top_w: *const f32, sg: *const f32, ld_sg: c_int, sh: *const f32, y: *mut f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_moe_nvfp4_gemm(xs: *const c_void, w_ptr: *const u64, s_ptr: *const u64, ws2: *const f32, offsets: *const i32, tile_expert: *const i32, tile_row: *const i32, num_tiles: *const i32, max_tiles: c_int, y: *mut f32, n: c_int, k: c_int, stream: Stream) -> Error;
        pub fn squash_swiglu(gate: *const f32, up: *const f32, ld: c_int, width: c_int, act: *mut c_void, count: usize, stream: Stream) -> Error;
        pub fn squash_moe_combine(yd: *const f32, pair_row: *const i32, top_w: *const f32, sg: *const f32, ld_sg: c_int, sh: *const f32, y: *mut f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_ple_gate(kp: *const f32, h: *const f32, v: *const f32, wk: *const c_void, wq: *const c_void, wc: *const c_void, gv: *mut f32, xn: *mut f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_ple_conv(gv: *const f32, xn: *const f32, w: *const c_void, state: *mut f32, h: *mut f32, t: c_int, rec: *mut f32, ctl: *const i32, defer: c_int, stream: Stream) -> Error;
    }
}

pub mod bf16;
pub mod kernels;

/// Runtime kernel variants (docs/KBENCH.md): tuning keys that kernel launchers read when they launch
/// (for CUDA graphs: at capture). Unset keys keep the engine's normal behaviour.
pub mod variants {
    /// Sets `key` (e.g. "gemv.warps") to `value` for launches from now on.
    pub fn set(key: &str, value: i32) {
        let k = std::ffi::CString::new(key).expect("variant key without NUL");
        unsafe { super::ffi::squash_set_variant(k.as_ptr(), value) }
    }

    /// Removes every key (including those from SQUASH_VARIANTS).
    pub fn clear() {
        unsafe { super::ffi::squash_clear_variants() }
    }

    /// The value of `key`, or `fallback` when it is not set.
    pub fn get(key: &str, fallback: i32) -> i32 {
        let k = std::ffi::CString::new(key).expect("variant key without NUL");
        unsafe { super::ffi::squash_variant_get(k.as_ptr(), fallback) }
    }
}

/// Converts a CUDA error code into a `Result`.
pub fn check(err: ffi::Error, what: &str) -> Result<()> {
    if err == ffi::SUCCESS {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(ffi::cudaGetErrorString(err)) }.to_string_lossy();
    bail!("{what}: CUDA error {err}: {message}")
}

pub fn set_device(device: i32) -> Result<()> {
    check(unsafe { ffi::cudaSetDevice(device) }, "cudaSetDevice")
}

pub fn synchronize() -> Result<()> {
    check(unsafe { ffi::cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
}

/// Brackets the region an external profiler (`ncu --profile-from-start off`) collects.
pub fn profiler_start() -> Result<()> {
    check(unsafe { ffi::cudaProfilerStart() }, "cudaProfilerStart")
}

pub fn profiler_stop() -> Result<()> {
    check(unsafe { ffi::cudaProfilerStop() }, "cudaProfilerStop")
}

/// Free and total device memory in bytes.
pub fn mem_info() -> Result<(usize, usize)> {
    let (mut free, mut total) = (0, 0);
    check(unsafe { ffi::cudaMemGetInfo(&mut free, &mut total) }, "cudaMemGetInfo")?;
    Ok((free, total))
}

/// An owned device allocation.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut ptr = null_mut();
        check(unsafe { ffi::cudaMalloc(&mut ptr, bytes.max(1)) }, "cudaMalloc")
            .map_err(|e| e.context(format!("allocating {bytes} device bytes")))?;
        Ok(Self { ptr, bytes })
    }

    /// Allocates and zero-fills.
    pub fn zeroed(bytes: usize) -> Result<Self> {
        let b = Self::new(bytes)?;
        check(unsafe { ffi::cudaMemsetAsync(b.ptr, 0, bytes, null_mut()) }, "cudaMemset")?;
        synchronize()?;
        Ok(b)
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn addr(&self) -> u64 {
        self.ptr as u64
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe { ffi::cudaFree(self.ptr) };
    }
}

/// Page-locked host memory.
pub struct PinnedBuffer {
    ptr: *mut u8,
    bytes: usize,
}

unsafe impl Send for PinnedBuffer {}

impl PinnedBuffer {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut ptr = null_mut();
        check(unsafe { ffi::cudaHostAlloc(&mut ptr, bytes, ffi::HOST_ALLOC_PORTABLE) }, "cudaHostAlloc")?;
        Ok(Self { ptr: ptr.cast(), bytes })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.bytes) }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe { ffi::cudaFreeHost(self.ptr.cast()) };
    }
}

pub struct Stream(ffi::Stream);

unsafe impl Send for Stream {}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut stream = null_mut();
        check(unsafe { ffi::cudaStreamCreateWithFlags(&mut stream, ffi::STREAM_NON_BLOCKING) }, "cudaStreamCreate")?;
        Ok(Self(stream))
    }

    pub fn raw(&self) -> ffi::Stream {
        self.0
    }

    /// Makes later work on this stream wait for `event` (recorded on another stream).
    pub fn wait(&self, event: &Event) -> Result<()> {
        check(unsafe { ffi::cudaStreamWaitEvent(self.0, event.0, 0) }, "cudaStreamWaitEvent")
    }

    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { ffi::cudaStreamSynchronize(self.0) }, "cudaStreamSynchronize")
    }

    pub fn memset(&self, dst: u64, value: u8, bytes: usize) -> Result<()> {
        check(unsafe { ffi::cudaMemsetAsync(dst as *mut c_void, value as c_int, bytes, self.0) }, "cudaMemsetAsync")
    }

    /// Enqueues a host-to-device copy from pageable memory (synchronous with respect to `src`).
    pub fn upload_to<T: Copy>(&self, dst: u64, src: &[T]) -> Result<()> {
        unsafe { self.copy_h2d(dst, src.as_ptr().cast(), std::mem::size_of_val(src))? };
        self.synchronize()
    }

    /// Enqueues a host-to-device copy. The caller keeps `src` alive and unmodified until the copy completes.
    pub unsafe fn copy_h2d(&self, dst: u64, src: *const u8, bytes: usize) -> Result<()> {
        check(
            unsafe { ffi::cudaMemcpyAsync(dst as *mut c_void, src.cast(), bytes, ffi::MEMCPY_HOST_TO_DEVICE, self.0) },
            "cudaMemcpyAsync H2D",
        )
    }

    pub fn upload<T: Copy>(&self, dst: &DeviceBuffer, src: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(src);
        anyhow::ensure!(bytes <= dst.bytes(), "upload of {bytes} bytes into a {}-byte buffer", dst.bytes());
        unsafe { self.copy_h2d(dst.addr(), src.as_ptr().cast(), bytes)? };
        self.synchronize()
    }

    pub fn download<T: Copy + Default>(&self, src: u64, count: usize) -> Result<Vec<T>> {
        let mut out = vec![T::default(); count];
        unsafe { self.copy_d2h(out.as_mut_ptr().cast(), src, count * std::mem::size_of::<T>())? };
        self.synchronize()?;
        Ok(out)
    }

    /// Enqueues a device-to-device copy.
    pub fn copy_d2d(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        check(unsafe { ffi::cudaMemcpyAsync(dst as *mut c_void, src as *const c_void, bytes, ffi::MEMCPY_DEVICE_TO_DEVICE, self.0) }, "cudaMemcpyAsync D2D")
    }

    /// Enqueues a device-to-host copy. The caller keeps `dst` alive until the copy completes.
    pub unsafe fn copy_d2h(&self, dst: *mut u8, src: u64, bytes: usize) -> Result<()> {
        check(
            unsafe { ffi::cudaMemcpyAsync(dst.cast(), src as *const c_void, bytes, ffi::MEMCPY_DEVICE_TO_HOST, self.0) },
            "cudaMemcpyAsync D2H",
        )
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { ffi::cudaStreamDestroy(self.0) };
    }
}

pub struct Event(ffi::Event);

unsafe impl Send for Event {}

impl Event {
    pub fn new() -> Result<Self> {
        let mut event = null_mut();
        check(unsafe { ffi::cudaEventCreateWithFlags(&mut event, ffi::EVENT_DISABLE_TIMING) }, "cudaEventCreate")?;
        Ok(Self(event))
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        check(unsafe { ffi::cudaEventRecord(self.0, stream.raw()) }, "cudaEventRecord")
    }

    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { ffi::cudaEventSynchronize(self.0) }, "cudaEventSynchronize")
    }
}

impl Event {
    /// An event that records timestamps.
    pub fn timing() -> Result<Self> {
        let mut event = null_mut();
        check(unsafe { ffi::cudaEventCreateWithFlags(&mut event, 0) }, "cudaEventCreate")?;
        Ok(Self(event))
    }

    /// Milliseconds between two recorded timing events.
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32> {
        let mut ms = 0f32;
        check(unsafe { ffi::cudaEventElapsedTime(&mut ms, start.0, end.0) }, "cudaEventElapsedTime")?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { ffi::cudaEventDestroy(self.0) };
    }
}

/// An instantiated CUDA graph captured from work enqueued on a stream.
pub struct Graph {
    graph: *mut c_void,
    exec: *mut c_void,
}

unsafe impl Send for Graph {}

impl Graph {
    /// Captures everything `body` enqueues on `stream` (which must not synchronize) into a graph.
    pub fn capture(stream: &Stream, body: impl FnOnce(&Stream) -> Result<()>) -> Result<Self> {
        const THREAD_LOCAL: c_int = 1;
        check(unsafe { ffi::cudaStreamBeginCapture(stream.raw(), THREAD_LOCAL) }, "cudaStreamBeginCapture")?;
        let result = body(stream);
        let mut graph = null_mut();
        let end = check(unsafe { ffi::cudaStreamEndCapture(stream.raw(), &mut graph) }, "cudaStreamEndCapture");
        result?;
        end?;
        let mut exec = null_mut();
        if let Err(e) = check(unsafe { ffi::cudaGraphInstantiateWithFlags(&mut exec, graph, 0) }, "cudaGraphInstantiate") {
            unsafe { ffi::cudaGraphDestroy(graph) };
            return Err(e);
        }
        Ok(Self { graph, exec })
    }

    pub fn launch(&self, stream: &Stream) -> Result<()> {
        check(unsafe { ffi::cudaGraphLaunch(self.exec, stream.raw()) }, "cudaGraphLaunch")
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            ffi::cudaGraphExecDestroy(self.exec);
            ffi::cudaGraphDestroy(self.graph);
        }
    }
}

/// Wrapping sum of the little-endian u32 words of `bytes` device bytes at `addr` (tail zero-padded).
pub fn device_checksum(addr: u64, bytes: usize, stream: &Stream) -> Result<u64> {
    let out = DeviceBuffer::new(8)?;
    check(
        unsafe { ffi::squash_checksum(addr as *const c_void, bytes, out.ptr().cast(), stream.raw()) },
        "squash_checksum",
    )?;
    let mut sum = 0u64;
    unsafe { stream.copy_d2h((&mut sum as *mut u64).cast(), out.addr(), 8)? };
    stream.synchronize()?;
    Ok(sum)
}

/// Host twin of [`device_checksum`].
pub fn host_checksum(bytes: &[u8]) -> u64 {
    let mut chunks = bytes.chunks_exact(4);
    let mut sum = 0u64;
    for word in &mut chunks {
        sum = sum.wrapping_add(u32::from_le_bytes(word.try_into().unwrap()) as u64);
    }
    let mut tail = [0u8; 4];
    tail[..chunks.remainder().len()].copy_from_slice(chunks.remainder());
    sum.wrapping_add(u32::from_le_bytes(tail) as u64)
}
