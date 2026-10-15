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

        pub fn squash_checksum(data: *const c_void, bytes: usize, out: *mut u64, stream: Stream) -> Error;
        pub fn squash_gemm_bf16_f32(x: *const c_void, ldx: c_int, w: *const c_void, ldw: c_int, y: *mut f32, ldy: c_int, m: c_int, n: c_int, k: c_int, alpha: f32, accumulate: c_int, stream: Stream) -> Error;
        pub fn squash_gemm_bf16_bf16(x: *const c_void, ldx: c_int, w: *const c_void, ldw: c_int, y: *mut c_void, ldy: c_int, m: c_int, n: c_int, k: c_int, alpha: f32, accumulate: c_int, stream: Stream) -> Error;
        pub fn squash_topk(x: *const f32, rows: c_int, n: c_int, ld: usize, k: c_int, sort_by_index: c_int, out_idx: *mut i32, out_val: *mut f32, stream: Stream) -> Error;
        pub fn squash_row_lse(x: *const f32, rows: c_int, n: c_int, ld: usize, targets: *const i32, lse: *mut f32, target_logit: *mut f32, stream: Stream) -> Error;
        pub fn squash_embed(tokens: *const i32, t: c_int, table: *const c_void, h: *mut f32, stream: Stream) -> Error;
        pub fn squash_grouped_rmsnorm(h: *const f32, w: *const c_void, n: *mut c_void, t: c_int, stream: Stream) -> Error;
        pub fn squash_hc_inject(n: *const c_void, w: *const c_void, inj: *mut f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_hc_low_rank_act(x: *const f32, z: *mut c_void, count: c_int, stream: Stream) -> Error;
        pub fn squash_hc_mix(u: *const f32, n: *const c_void, m: *mut c_void, t: c_int, stream: Stream) -> Error;
        pub fn squash_hc_combine(h: *mut f32, y: *const f32, inj: *const f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_f32_to_bf16(x: *const f32, y: *mut c_void, count: usize, stream: Stream) -> Error;
        pub fn squash_add_f32(y: *mut f32, x: *const f32, count: usize, stream: Stream) -> Error;
        pub fn squash_gdn_conv(x: *const f32, out: *mut f32, state: *mut f32, w: *const c_void, t: c_int, stream: Stream) -> Error;
        pub fn squash_gdn_prep(qkv: *mut f32, a: *const f32, b: *const f32, a_log: *const c_void, dt_bias: *const c_void, beta: *mut f32, g: *mut f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_gdn_recurrent(qkv: *const f32, beta: *const f32, g: *const f32, state: *mut f32, out: *mut f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_gdn_gated_norm(o: *const f32, z: *const f32, w: *const c_void, y: *mut c_void, t: c_int, stream: Stream) -> Error;
        pub fn squash_qsa_qkv_prep(qg: *const f32, kv: *const f32, q_norm: *const c_void, k_norm: *const c_void, q: *mut f32, kcache: *mut c_void, vcache: *mut c_void, pos3: *const i32, start: c_int, t: c_int, stream: Stream) -> Error;
        pub fn squash_qsa_index_prep(idx: *const f32, q_ln: *const c_void, iq: *mut f32, raw: *mut c_void, pos3: *const i32, start: c_int, t: c_int, stream: Stream) -> Error;
        pub fn squash_qsa_pool(raw: *const c_void, k_ln: *const c_void, pooled: *mut c_void, pos3: *const i32, start: c_int, t: c_int, stream: Stream) -> Error;
        pub fn squash_qsa_index_scores(iq: *const f32, pooled: *const c_void, scores: *mut f32, ld: c_int, start: c_int, t: c_int, stream: Stream) -> Error;
        pub fn squash_qsa_attention(q: *const f32, qg: *const f32, kcache: *const c_void, vcache: *const c_void, sel: *const i32, start: c_int, t: c_int, out: *mut c_void, stream: Stream) -> Error;
        pub fn squash_moe_tile_rows() -> c_int;
        pub fn squash_moe_route(logits: *const f32, t: c_int, top_i: *mut i32, top_w: *mut f32, stream: Stream) -> Error;
        pub fn squash_moe_plan(top_i: *const i32, t: c_int, counts: *mut i32, cursor: *mut i32, offsets: *mut i32, tile_expert: *mut i32, tile_row: *mut i32, num_tiles: *mut i32, pair_row: *mut i32, stream: Stream) -> Error;
        pub fn squash_moe_gather(m: *const c_void, pair_row: *const i32, t: c_int, xs: *mut c_void, stream: Stream) -> Error;
        pub fn squash_moe_nvfp4_gemm(xs: *const c_void, w_ptr: *const u64, s_ptr: *const u64, ws2: *const f32, offsets: *const i32, tile_expert: *const i32, tile_row: *const i32, num_tiles: *const i32, max_tiles: c_int, y: *mut f32, n: c_int, k: c_int, stream: Stream) -> Error;
        pub fn squash_swiglu(gate: *const f32, up: *const f32, act: *mut c_void, count: usize, stream: Stream) -> Error;
        pub fn squash_moe_combine(yd: *const f32, pair_row: *const i32, top_w: *const f32, sg: *const f32, sh: *const f32, y: *mut f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_ple_gate(kp: *const f32, h: *const f32, v: *const f32, wk: *const c_void, wq: *const c_void, wc: *const c_void, gv: *mut f32, xn: *mut f32, t: c_int, stream: Stream) -> Error;
        pub fn squash_ple_conv(gv: *const f32, xn: *const f32, w: *const c_void, state: *mut f32, h: *mut f32, t: c_int, stream: Stream) -> Error;
    }
}

pub mod bf16;
pub mod kernels;

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

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { ffi::cudaEventDestroy(self.0) };
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
