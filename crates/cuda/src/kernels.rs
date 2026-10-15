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
