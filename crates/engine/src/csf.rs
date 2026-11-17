//! NVFP4 block scales stored in ModelOpt's lossless "CSF" form (`weight_scale_encoding: csf`), decoded
//! back to the plain E4M3 scale matrix `[rows, cols]` of the uncompressed checkpoint.
//!
//! - `<scale>.nvfp4_csf_fixed`: U8 `[rows / 16, 16 * (1 + cols / 2)]`. Each 16-row slab holds one base
//!   byte per row, then the rows' 4-bit codes (row-major, two per byte, low nibble first); a scale byte
//!   is `base + code`.
//! - `<scale>.nvfp4_csf_exceptions`: U32, sorted words whose low 24 bits are a flat `row * cols + col`
//!   position and whose high 8 bits replace the byte there.

use anyhow::{Result, ensure};

pub const FIXED: &str = "nvfp4_csf_fixed";
pub const EXCEPTIONS: &str = "nvfp4_csf_exceptions";

/// Rows and columns of the scale matrix described by a fixed stream of shape `shape`.
pub fn dims(shape: &[usize]) -> Result<(usize, usize)> {
    ensure!(
        shape.len() == 2 && shape[1] % 16 == 0 && shape[1] >= 16,
        "CSF fixed stream of shape {shape:?}"
    );
    Ok((shape[0] * 16, (shape[1] / 16 - 1) * 2))
}

/// Decodes into `out` (`rows * cols` bytes). `exceptions` holds the raw little-endian U32 words.
pub fn decode(fixed: &[u8], shape: &[usize], exceptions: &[u8], out: &mut [u8]) -> Result<()> {
    let (rows, cols) = dims(shape)?;
    ensure!(
        fixed.len() == shape[0] * shape[1] && out.len() == rows * cols,
        "CSF fixed stream of {} bytes for shape {shape:?}, output of {} bytes",
        fixed.len(),
        out.len()
    );
    ensure!(rows * cols <= 1 << 24, "CSF scale exceeds 24-bit exception positions");
    ensure!(exceptions.len() % 4 == 0, "CSF exceptions are not U32 words");
    let slab = shape[1];
    for (s, src) in fixed.chunks_exact(slab).enumerate() {
        let (bases, codes) = src.split_at(16);
        for r in 0..16 {
            let base = bases[r];
            ensure!(base <= 240, "CSF row base {base} above 240");
            let codes = &codes[r * cols / 2..(r + 1) * cols / 2];
            let dst = &mut out[(s * 16 + r) * cols..(s * 16 + r + 1) * cols];
            for (pair, &c) in dst.chunks_exact_mut(2).zip(codes) {
                pair[0] = base + (c & 15);
                pair[1] = base + (c >> 4);
            }
        }
    }
    let mut last = None;
    for w in exceptions.chunks_exact(4) {
        let w = u32::from_le_bytes(w.try_into().unwrap());
        let pos = (w & 0xff_ffff) as usize;
        ensure!(
            pos < rows * cols && last.is_none_or(|l| pos > l),
            "CSF exception positions must be increasing and in range"
        );
        out[pos] = (w >> 24) as u8;
        last = Some(pos);
    }
    Ok(())
}

/// Encodes `scales` (`[rows, cols]`, rows a multiple of 16, cols even) with each row's minimum as its
/// base; bytes more than 15 above it become exceptions. For tests.
pub fn encode(scales: &[u8], rows: usize, cols: usize) -> (Vec<u8>, Vec<usize>, Vec<u8>) {
    let mut fixed = Vec::new();
    let mut exceptions = Vec::new();
    for s in 0..rows / 16 {
        let rows16 = &scales[s * 16 * cols..(s + 1) * 16 * cols];
        let bases: Vec<u8> = rows16
            .chunks_exact(cols)
            .map(|r| *r.iter().min().unwrap())
            .collect();
        fixed.extend(&bases);
        for (r, row) in rows16.chunks_exact(cols).enumerate() {
            for (c, pair) in row.chunks_exact(2).enumerate() {
                let mut code = |i: usize| {
                    let d = pair[i] - bases[r];
                    if d > 15 {
                        let pos = (s * 16 + r) * cols + 2 * c + i;
                        exceptions.push((pos as u32) | ((pair[i] as u32) << 24));
                        0
                    } else {
                        d
                    }
                };
                let (lo, hi) = (code(0), code(1));
                fixed.push(lo | (hi << 4));
            }
        }
    }
    let ex = exceptions.iter().flat_map(|w| w.to_le_bytes()).collect();
    (fixed, vec![rows / 16, 16 * (1 + cols / 2)], ex)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let (rows, cols) = (48, 20);
        let mut x = 0x9e37_79b9_u32;
        let scales: Vec<u8> = (0..rows * cols)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                // Mostly within 15 of a per-row level, sometimes far above it.
                let level = 40 + (i / cols) as u32 % 50;
                (if x % 17 == 0 { level + 16 + x % 60 } else { level + x % 16 }) as u8
            })
            .collect();
        let (fixed, shape, ex) = encode(&scales, rows, cols);
        assert!(!ex.is_empty());
        assert_eq!(dims(&shape).unwrap(), (rows, cols));
        let mut out = vec![0; rows * cols];
        decode(&fixed, &shape, &ex, &mut out).unwrap();
        assert_eq!(out, scales);
    }

    #[test]
    fn layout() {
        // One slab, two columns: row r has base r and codes (1, 2); one exception at row 3, column 1.
        let mut fixed: Vec<u8> = (0..16).collect();
        fixed.extend(std::iter::repeat_n(0x21, 16));
        let ex = ((3 * 2 + 1) as u32 | (200 << 24)).to_le_bytes();
        let mut out = vec![0; 32];
        decode(&fixed, &[1, 32], &ex, &mut out).unwrap();
        for r in 0..16 {
            let want1 = if r == 3 { 200 } else { r as u8 + 2 };
            assert_eq!((out[2 * r], out[2 * r + 1]), (r as u8 + 1, want1));
        }
        // Unsorted or out-of-range exceptions and bases over 240 are errors.
        let two: Vec<u8> = [5u32, 4].iter().flat_map(|w| w.to_le_bytes()).collect();
        assert!(decode(&fixed, &[1, 32], &two, &mut out).is_err());
        assert!(decode(&fixed, &[1, 32], &32u32.to_le_bytes(), &mut out).is_err());
        fixed[0] = 241;
        assert!(decode(&fixed, &[1, 32], &[], &mut out).is_err());
    }
}
