//! Uploads checkpoint tensors to the GPU without repacking.
//!
//! Selected tensors of each file are coalesced into a few contiguous byte ranges ("segments"). Each
//! segment becomes one device allocation that receives a byte-exact copy of that file range, so a
//! tensor's device address is its segment base plus its offset within the range. Worker threads
//! read fixed-size chunks into double-buffered pinned memory and stream them to the device.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Result, anyhow};
use squash_cuda::{DeviceBuffer, Event, PinnedBuffer, Stream};

use crate::checkpoint::{Checkpoint, DType};

/// Unselected bytes tolerated inside a segment instead of splitting it.
const MAX_GAP: u64 = 1 << 20;

#[derive(Clone, Debug)]
pub struct DeviceTensor {
    pub addr: u64,
    pub bytes: u64,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// For FP8 tensors converted at load: per-row FP32 scales.
    pub scale: u64,
}

pub struct DeviceWeights {
    pub segments: Vec<DeviceBuffer>,
    pub tensors: HashMap<String, DeviceTensor>,
}

impl DeviceWeights {
    pub fn get(&self, name: &str) -> Result<&DeviceTensor> {
        self.tensors.get(name).ok_or_else(|| anyhow!("weights have no device tensor {name}"))
    }

    pub fn device_bytes(&self) -> usize {
        self.segments.iter().map(|s| s.bytes()).sum()
    }

    pub fn merge(&mut self, other: DeviceWeights) {
        self.segments.extend(other.segments);
        self.tensors.extend(other.tensors);
    }
}

/// Uploads the selected 2-D BF16 tensors and converts each to an 8-bit format (FP8 E4M3, INT8 or
/// group-scaled INT8) with per-row or per-group scales on the GPU, one file at a time, so only one file's
/// BF16 copy is ever resident.
///
/// `prep(name, addr, n, k)` may transform each staged BF16 tensor in place before it is quantized.
///
/// Each `groups` entry `(name, members)` quantizes its members (same K, same file) into consecutive rows
/// of one weight, registered under `name` with shape `[sum of rows, K]`, so a single GEMV computes all
/// of them; the members stay addressable under their own names.
pub fn upload_quantized(
    ckpt: &Checkpoint,
    select: impl Fn(&str) -> bool,
    format: squash_cuda::kernels::WeightFormat,
    groups: &[(String, Vec<String>)],
    prep: &dyn Fn(&str, u64, usize, usize) -> Result<()>,
    opts: LoadOptions,
    stream: &squash_cuda::Stream,
) -> Result<(DeviceWeights, LoadStats)> {
    use squash_cuda::kernels::{self as k, WeightFormat};
    anyhow::ensure!(format != WeightFormat::Bf16, "upload_quantized needs an 8-bit format");
    let start = Instant::now();
    let scales_per_row = |kk: usize| if format == WeightFormat::Int8Group { kk / k::int8_group() } else { 1 };

    // Units of contiguous placement: a group's members, or a single tensor.
    let mut grouped = std::collections::HashSet::new();
    let mut units: Vec<Vec<usize>> = vec![Vec::new(); ckpt.files.len()];
    let mut unit_list: Vec<(Option<&str>, Vec<&str>)> = Vec::new();
    for (name, members) in groups {
        anyhow::ensure!(!members.is_empty(), "empty fusion group {name}");
        let file = ckpt.get(&members[0])?.file;
        let kk = ckpt.get(&members[0])?.shape[1];
        for m in members {
            anyhow::ensure!(select(m), "fusion group {name}: member {m} is not selected for quantization");
            let info = ckpt.get(m)?;
            anyhow::ensure!(info.file == file && info.shape[1] == kk, "fusion group {name}: members differ in file or K");
            grouped.insert(m.as_str());
        }
        units[file].push(unit_list.len());
        unit_list.push((Some(name.as_str()), members.iter().map(|m| m.as_str()).collect()));
    }
    let mut singles: Vec<&str> = ckpt.tensors.keys().map(|n| n.as_str()).filter(|n| select(n) && !grouped.contains(n)).collect();
    singles.sort();
    for name in singles {
        let info = &ckpt.tensors[name];
        anyhow::ensure!(info.dtype == DType::BF16 && info.shape.len() == 2, "{name}: only 2-D BF16 tensors quantize");
        units[info.file].push(unit_list.len());
        unit_list.push((None, vec![name]));
    }

    let mut out = DeviceWeights { segments: Vec::new(), tensors: HashMap::new() };
    let mut stats = LoadStats::default();
    for file_units in units.iter().filter(|u| !u.is_empty()) {
        let mut names: Vec<&str> = file_units.iter().flat_map(|&u| unit_list[u].1.iter().copied()).collect();
        names.sort();
        let (staged, s) = upload(ckpt, |n| names.binary_search(&n).is_ok(), opts)?;
        stats.bytes += s.bytes;
        // Layout: per unit, the members' codes back to back, then their scales back to back.
        let mut placements = Vec::new();
        let mut total = 0usize;
        for &u in file_units {
            let members = &unit_list[u].1;
            let kk = ckpt.tensors[members[0]].shape[1];
            let rows: usize = members.iter().map(|m| ckpt.tensors[*m].shape[0]).sum();
            let (q, sc) = (total, total + (rows * kk).div_ceil(256) * 256);
            total = sc + (rows * scales_per_row(kk) * 4).div_ceil(256) * 256;
            placements.push((u, q, sc, rows, kk));
        }
        let arena = DeviceBuffer::new(total)?;
        for (u, q, sc, rows, kk) in placements {
            let (group, members) = &unit_list[u];
            let (mut qa, mut sa) = (arena.addr() + q as u64, arena.addr() + sc as u64);
            let dtype = match format {
                WeightFormat::Fp8 => DType::F8E4M3,
                WeightFormat::Int8 => DType::I8,
                _ => DType::I8G,
            };
            if let Some(g) = group {
                out.tensors.insert(g.to_string(), DeviceTensor { addr: qa, bytes: (rows * kk) as u64, dtype, shape: vec![rows, kk], scale: sa });
            }
            for name in members {
                let src = staged.get(name)?;
                let n = src.shape[0];
                prep(name, src.addr, n, kk)?;
                match format {
                    WeightFormat::Fp8 => k::quantize_fp8_rows(stream, src.addr, n, kk, qa, sa)?,
                    WeightFormat::Int8 => k::quantize_int8_rows(stream, src.addr, n, kk, qa, sa)?,
                    _ => {
                        anyhow::ensure!(kk % k::int8_group() == 0, "{name}: K {kk} is not a multiple of the INT8 group");
                        k::quantize_int8_groups(stream, src.addr, n, kk, qa, sa)?
                    }
                }
                out.tensors.insert(name.to_string(), DeviceTensor { addr: qa, bytes: (n * kk) as u64, dtype, shape: src.shape.clone(), scale: sa });
                qa += (n * kk) as u64;
                sa += (n * scales_per_row(kk) * 4) as u64;
            }
        }
        stream.synchronize()?;
        out.segments.push(arena);
    }
    stats.segments = out.segments.len();
    stats.seconds = start.elapsed().as_secs_f64();
    Ok((out, stats))
}

#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    pub threads: usize,
    pub chunk_bytes: usize,
    /// Read with `O_DIRECT`, bypassing the page cache (falls back to buffered reads if unsupported).
    pub direct: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self { threads: 8, chunk_bytes: 32 << 20, direct: true }
    }
}

#[derive(Debug, Default)]
pub struct LoadStats {
    pub bytes: u64,
    pub segments: usize,
    pub seconds: f64,
}

struct Chunk {
    file: usize,
    offset: u64,
    len: usize,
    dst: u64,
}

fn required_alignment(ckpt: &Checkpoint, name: &str) -> u64 {
    let info = &ckpt.tensors[name];
    if info.bytes >= 16 { 16 } else { info.dtype.size() as u64 }
}

/// Uploads every tensor for which `select` returns true.
pub fn upload(ckpt: &Checkpoint, select: impl Fn(&str) -> bool, opts: LoadOptions) -> Result<(DeviceWeights, LoadStats)> {
    let start = Instant::now();

    let mut by_file: Vec<Vec<(&str, u64, u64)>> = vec![Vec::new(); ckpt.files.len()];
    for (name, info) in &ckpt.tensors {
        if select(name) {
            by_file[info.file].push((name.as_str(), info.offset, info.offset + info.bytes));
        }
    }

    let mut segments = Vec::new();
    let mut tensors = HashMap::new();
    let mut chunks = Vec::new();
    let mut total = 0u64;
    for (file, mut list) in by_file.into_iter().enumerate() {
        list.sort_by_key(|&(_, begin, _)| begin);
        let mut i = 0;
        while i < list.len() {
            // The segment starts at its first tensor, which lands on the (256-byte aligned) allocation
            // base; a later tensor joins only if it stays aligned (16 bytes, or its dtype size if smaller).
            let seg_begin = list[i].1;
            let mut seg_end = list[i].2;
            let mut j = i + 1;
            while j < list.len() && list[j].1 <= seg_end + MAX_GAP && (list[j].1 - seg_begin) % required_alignment(ckpt, list[j].0) == 0 {
                seg_end = seg_end.max(list[j].2);
                j += 1;
            }
            let len = (seg_end - seg_begin) as usize;
            let buffer = DeviceBuffer::new(len)?;
            for &(name, begin, end) in &list[i..j] {
                let info = &ckpt.tensors[name];
                tensors.insert(
                    name.to_string(),
                    DeviceTensor { addr: buffer.addr() + (begin - seg_begin), bytes: end - begin, dtype: info.dtype, shape: info.shape.clone(), scale: 0 },
                );
            }
            let mut offset = 0usize;
            while offset < len {
                let n = opts.chunk_bytes.min(len - offset);
                chunks.push(Chunk { file, offset: seg_begin + offset as u64, len: n, dst: buffer.addr() + offset as u64 });
                offset += n;
            }
            total += len as u64;
            segments.push(buffer);
            i = j;
        }
    }

    // Largest chunks first keeps the threads evenly loaded at the end.
    chunks.sort_by_key(|c| std::cmp::Reverse(c.len));
    let direct = if opts.direct { open_direct(&ckpt.files) } else { None };
    let next = AtomicUsize::new(0);
    let error = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..opts.threads.max(1) {
            scope.spawn(|| {
                if let Err(e) = upload_worker(ckpt, direct.as_deref(), &chunks, &next, opts.chunk_bytes) {
                    error.lock().unwrap().get_or_insert(e);
                    next.store(chunks.len(), Ordering::Relaxed);
                }
            });
        }
    });
    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }

    let stats = LoadStats { bytes: total, segments: segments.len(), seconds: start.elapsed().as_secs_f64() };
    Ok((DeviceWeights { segments, tensors }, stats))
}

/// Direct I/O block size: offsets, lengths and buffer addresses of `O_DIRECT` reads are multiples of it.
const DIRECT_ALIGN: u64 = 4096;
const O_DIRECT: i32 = 0o40000;

fn open_direct(files: &[crate::checkpoint::SafetensorsFile]) -> Option<Vec<File>> {
    files.iter().map(|f| OpenOptions::new().read(true).custom_flags(O_DIRECT).open(&f.path).ok()).collect()
}

fn upload_worker(ckpt: &Checkpoint, direct: Option<&[File]>, chunks: &[Chunk], next: &AtomicUsize, chunk_bytes: usize) -> Result<()> {
    let stream = Stream::new()?;
    let buffer_bytes = chunk_bytes + 2 * DIRECT_ALIGN as usize;
    let mut buffers = [PinnedBuffer::new(buffer_bytes)?, PinnedBuffer::new(buffer_bytes)?];
    let events = [Event::new()?, Event::new()?];
    let mut used = [false; 2];
    let mut slot = 0;
    loop {
        let index = next.fetch_add(1, Ordering::Relaxed);
        let Some(chunk) = chunks.get(index) else { break };
        if used[slot] {
            events[slot].synchronize()?;
        }
        let buffer = buffers[slot].as_mut_slice();
        let data = match direct {
            Some(files) => {
                // Read the enclosing aligned window; the final block of a file may come back short.
                let begin = chunk.offset / DIRECT_ALIGN * DIRECT_ALIGN;
                let end = (chunk.offset + chunk.len as u64).div_ceil(DIRECT_ALIGN) * DIRECT_ALIGN;
                let window = &mut buffer[..(end - begin) as usize];
                let skip = (chunk.offset - begin) as usize;
                let mut filled = 0;
                while filled < skip + chunk.len {
                    let n = files[chunk.file].read_at(&mut window[filled..], begin + filled as u64)?;
                    anyhow::ensure!(n > 0, "unexpected end of {}", ckpt.files[chunk.file].path.display());
                    filled += n;
                }
                &buffer[skip..skip + chunk.len]
            }
            None => {
                ckpt.files[chunk.file].file.read_exact_at(&mut buffer[..chunk.len], chunk.offset)?;
                &buffer[..chunk.len]
            }
        };
        unsafe { stream.copy_h2d(chunk.dst, data.as_ptr(), chunk.len)? };
        events[slot].record(&stream)?;
        used[slot] = true;
        slot ^= 1;
    }
    stream.synchronize()
}
