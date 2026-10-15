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

    /// Enqueues a host-to-device copy. The caller keeps `src` alive and unmodified until the copy completes.
    pub unsafe fn copy_h2d(&self, dst: u64, src: *const u8, bytes: usize) -> Result<()> {
        check(
            unsafe { ffi::cudaMemcpyAsync(dst as *mut c_void, src.cast(), bytes, ffi::MEMCPY_HOST_TO_DEVICE, self.0) },
            "cudaMemcpyAsync H2D",
        )
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
