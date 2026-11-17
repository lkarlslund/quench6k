//! Uploads checkpoint tensors to the GPU without repacking.
//!
//! Selected tensors of each file are coalesced into a few contiguous byte ranges ("segments"). Each
//! segment becomes one device allocation that receives a byte-exact copy of that file range, so a
//! tensor's device address is its segment base plus its offset within the range. Worker threads
//! read fixed-size chunks into double-buffered pinned memory and stream them to the device.

use std::collections::HashMap;
use std::fs::File;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Result, anyhow};
use quench6k_cuda::{DeviceBuffer, Event, PinnedBuffer, Stream};

use crate::checkpoint::{Checkpoint, DType};

/// Unselected bytes between tensors that are still read as part of one segment (one device allocation,
/// one sequence of large reads).
const MAX_GAP: u64 = 1 << 20;

/// Unselected bytes between tensors that are also copied to the device, keeping both in one run; larger
/// gaps are read but not copied (the runs are packed in the segment's allocation).
const PACK_GAP: u64 = 4096;

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
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("weights have no device tensor {name}"))
    }

    pub fn device_bytes(&self) -> usize {
        self.segments.iter().map(|s| s.bytes()).sum()
    }

    pub fn merge(&mut self, other: DeviceWeights) {
        self.segments.extend(other.segments);
        self.tensors.extend(other.tensors);
    }
}

/// Uploads the selected 2-D BF16 tensors and converts each to INT8 with one scale per 64 weights on the GPU, one file at a time, so only one file's
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
    format: quench6k_cuda::kernels::WeightFormat,
    groups: &[(String, Vec<String>)],
    prep: &dyn Fn(&str, u64, usize, usize) -> Result<()>,
    opts: LoadOptions,
    stream: &quench6k_cuda::Stream,
) -> Result<(DeviceWeights, LoadStats)> {
    use quench6k_cuda::kernels::{self as k, WeightFormat};
    anyhow::ensure!(
        format == WeightFormat::Int8Group,
        "upload_quantized produces group-scaled INT8 only"
    );
    let start = Instant::now();
    // GDN input projections become MXFP8 (gate PASS; FP8 tensor cores for prefill), the rest INT8 with
    // group scales. QUENCH6K_GDN_MXFP8=0 keeps them INT8.
    let mx8_on = std::env::var("QUENCH6K_GDN_MXFP8").map_or(true, |v| v != "0");
    let is_mx8 = |group: Option<&str>| {
        mx8_on && group.is_some_and(|g| g.ends_with(".linear_attn.quench6k.in_proj"))
    };
    let scale_bytes = |mx8: bool, rows: usize, kk: usize| {
        if mx8 {
            rows * kk / 32
        } else {
            rows * (kk / k::int8_group()) * 4
        }
    };

    // Units of contiguous placement: a group's members, or a single tensor. Units are uploaded in
    // batches per checkpoint file (a group with the file of its first member; the others are read
    // from wherever they are).
    let mut grouped = std::collections::HashSet::new();
    let mut units: Vec<Vec<usize>> = vec![Vec::new(); ckpt.files.len()];
    let mut unit_list: Vec<(Option<&str>, Vec<&str>)> = Vec::new();
    for (name, members) in groups {
        anyhow::ensure!(!members.is_empty(), "empty fusion group {name}");
        let file = ckpt.get(&members[0])?.file;
        let kk = ckpt.get(&members[0])?.shape[1];
        for m in members {
            anyhow::ensure!(
                select(m),
                "fusion group {name}: member {m} is not selected for quantization"
            );
            let info = ckpt.get(m)?;
            anyhow::ensure!(
                info.shape[1] == kk,
                "fusion group {name}: members differ in K"
            );
            grouped.insert(m.as_str());
        }
        let native = members.iter().filter(|m| native_mxfp8(ckpt, m)).count();
        anyhow::ensure!(
            native == 0 || native == members.len(),
            "fusion group {name}: members mix checkpoint MXFP8 and other formats"
        );
        units[file].push(unit_list.len());
        unit_list.push((
            Some(name.as_str()),
            members.iter().map(|m| m.as_str()).collect(),
        ));
    }
    let mut singles: Vec<&str> = ckpt
        .tensors
        .keys()
        .map(|n| n.as_str())
        .filter(|n| select(n) && !grouped.contains(n))
        .collect();
    singles.sort();
    for name in singles {
        let info = &ckpt.tensors[name];
        anyhow::ensure!(
            (info.dtype == DType::BF16 || native_mxfp8(ckpt, name)) && info.shape.len() == 2,
            "{name}: only 2-D BF16 tensors quantize"
        );
        units[info.file].push(unit_list.len());
        unit_list.push((None, vec![name]));
    }

    let mut out = DeviceWeights {
        segments: Vec::new(),
        tensors: HashMap::new(),
    };
    let mut stats = LoadStats::default();
    for file_units in units.iter().filter(|u| !u.is_empty()) {
        let scale_names: Vec<String> = file_units
            .iter()
            .flat_map(|&u| unit_list[u].1.iter().copied())
            .filter(|m| native_mxfp8(ckpt, m))
            .map(|m| format!("{m}_scale"))
            .collect();
        let mut names: Vec<&str> = file_units
            .iter()
            .flat_map(|&u| unit_list[u].1.iter().copied())
            .chain(scale_names.iter().map(|n| n.as_str()))
            .collect();
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
            let mx8 = is_mx8(unit_list[u].0) || native_mxfp8(ckpt, members[0]);
            let (q, sc) = (total, total + (rows * kk).div_ceil(256) * 256);
            total = sc + scale_bytes(mx8, rows, kk).div_ceil(256) * 256;
            placements.push((u, q, sc, rows, kk));
        }
        let arena = DeviceBuffer::new(total)?;
        for (u, q, sc, rows, kk) in placements {
            let (group, members) = &unit_list[u];
            let (mut qa, mut sa) = (arena.addr() + q as u64, arena.addr() + sc as u64);
            let native = native_mxfp8(ckpt, members[0]);
            let mx8 = is_mx8(*group) || native;
            let dtype = if mx8 { DType::MX8 } else { DType::I8G };
            if let Some(g) = group {
                out.tensors.insert(
                    g.to_string(),
                    DeviceTensor {
                        addr: qa,
                        bytes: (rows * kk) as u64,
                        dtype,
                        shape: vec![rows, kk],
                        scale: sa,
                    },
                );
            }
            for name in members {
                let src = staged.get(name)?;
                let n = src.shape[0];
                if native {
                    // Stored as MXFP8 in the checkpoint: codes and scales are copied as they are.
                    let sc = staged.get(&format!("{name}_scale"))?;
                    anyhow::ensure!(
                        sc.bytes as usize == n * kk / 32,
                        "{name}_scale: {} bytes, {} expected",
                        sc.bytes,
                        n * kk / 32
                    );
                    stream.copy_d2d(qa, src.addr, n * kk)?;
                    stream.copy_d2d(sa, sc.addr, n * kk / 32)?;
                } else if mx8 {
                    prep(name, src.addr, n, kk)?;
                    anyhow::ensure!(kk % 128 == 0, "{name}: K {kk} is not a multiple of 128");
                    k::mxfp8_rows(stream, src.addr, n, kk, kk, qa, sa)?;
                } else {
                    prep(name, src.addr, n, kk)?;
                    anyhow::ensure!(
                        kk % k::int8_group() == 0,
                        "{name}: K {kk} is not a multiple of the INT8 group"
                    );
                    k::quantize_int8_groups(stream, src.addr, n, kk, qa, sa)?;
                }
                out.tensors.insert(
                    name.to_string(),
                    DeviceTensor {
                        addr: qa,
                        bytes: (n * kk) as u64,
                        dtype,
                        shape: src.shape.clone(),
                        scale: sa,
                    },
                );
                qa += (n * kk) as u64;
                sa += scale_bytes(mx8, n, kk) as u64;
            }
        }
        stream.synchronize()?;
        out.segments.push(arena);
    }
    stats.segments = out.segments.len();
    stats.seconds = start.elapsed().as_secs_f64();
    Ok((out, stats))
}

/// Whether projection `name` is stored as MXFP8 in the checkpoint: E4M3 codes `[n, k]` with a sibling
/// `<name>_scale` of UE8M0 bytes `[n, k / 32]`.
pub fn native_mxfp8(ckpt: &Checkpoint, name: &str) -> bool {
    let (Some(w), Some(s)) = (
        ckpt.tensors.get(name),
        ckpt.tensors.get(&format!("{name}_scale")),
    ) else {
        return false;
    };
    w.dtype == DType::F8E4M3
        && s.dtype == DType::U8
        && w.shape.len() == 2
        && w.shape[1] % 32 == 0
        && s.shape == [w.shape[0], w.shape[1] / 32]
}

/// Decodes every NVFP4 block-scale matrix stored in CSF form (see [`crate::csf`]) among the selected
/// scales, `<scale>.nvfp4_csf_fixed` and `<scale>.nvfp4_csf_exceptions`, and uploads it under `<scale>` as
/// F8 E4M3 `[rows, cols]`. Scales are batched into device allocations of about `batch_bytes`.
pub fn upload_csf_scales(
    ckpt: &Checkpoint,
    select: impl Fn(&str) -> bool,
    batch_bytes: usize,
    stream: &Stream,
) -> Result<(DeviceWeights, LoadStats)> {
    use crate::csf;
    let start = Instant::now();
    let suffix = format!(".{}", csf::FIXED);
    let mut scales: Vec<(String, usize, usize)> = ckpt
        .tensors
        .iter()
        .filter_map(|(n, info)| {
            let base = n.strip_suffix(&suffix)?;
            select(base).then(|| (base.to_string(), info.shape.clone()))
        })
        .map(|(base, shape)| {
            let (rows, cols) = csf::dims(&shape)?;
            Ok((base, rows, cols))
        })
        .collect::<Result<_>>()?;
    scales.sort();
    let mut out = DeviceWeights {
        segments: Vec::new(),
        tensors: HashMap::new(),
    };
    let mut stats = LoadStats::default();
    let mut i = 0;
    while i < scales.len() {
        let mut j = i;
        let mut bytes = 0;
        while j < scales.len() && (j == i || bytes + scales[j].1 * scales[j].2 <= batch_bytes) {
            bytes += (scales[j].1 * scales[j].2).div_ceil(256) * 256;
            j += 1;
        }
        let batch = &scales[i..j];
        let mut offsets = Vec::with_capacity(batch.len());
        let mut at = 0;
        for (_, rows, cols) in batch {
            offsets.push(at);
            at += (rows * cols).div_ceil(256) * 256;
        }
        let mut host = vec![0u8; at];
        // Split the host buffer per scale and decode in parallel.
        let mut parts: Vec<(&(String, usize, usize), &mut [u8])> = Vec::with_capacity(batch.len());
        let mut rest = host.as_mut_slice();
        for (k, s) in batch.iter().enumerate() {
            let len = offsets.get(k + 1).copied().unwrap_or(at) - offsets[k];
            let (head, tail) = rest.split_at_mut(len);
            parts.push((s, &mut head[..s.1 * s.2]));
            rest = tail;
        }
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).min(16);
        let per = parts.len().div_ceil(threads);
        let errors = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for chunk in parts.chunks_mut(per.max(1)) {
                let errors = &errors;
                scope.spawn(move || {
                    for ((base, _, _), dst) in chunk.iter_mut() {
                        let r = (|| -> Result<()> {
                            let fixed_name = format!("{base}.{}", csf::FIXED);
                            let fixed = ckpt.read(&fixed_name)?;
                            let ex = ckpt.read(&format!("{base}.{}", csf::EXCEPTIONS))?;
                            csf::decode(&fixed, &ckpt.tensors[&fixed_name].shape, &ex, dst)
                        })();
                        if let Err(e) = r {
                            errors.lock().unwrap().push(e.context(format!("decoding {base}")));
                        }
                    }
                });
            }
        });
        if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
            return Err(e);
        }
        let buffer = DeviceBuffer::new(at)?;
        stream.upload_to(buffer.addr(), &host)?;
        stream.synchronize()?;
        for ((base, rows, cols), off) in batch.iter().zip(offsets) {
            out.tensors.insert(
                base.clone(),
                DeviceTensor {
                    addr: buffer.addr() + off as u64,
                    bytes: (rows * cols) as u64,
                    dtype: DType::F8E4M3,
                    shape: vec![*rows, *cols],
                    scale: 0,
                },
            );
        }
        stats.bytes += at as u64;
        out.segments.push(buffer);
        i = j;
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
    /// Text layers to load (the first `layers` of 48; fewer give a truncated model for kernel work).
    pub layers: usize,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            threads: 8,
            chunk_bytes: 32 << 20,
            direct: true,
            layers: crate::model::LAYERS,
        }
    }
}

#[derive(Debug, Default)]
pub struct LoadStats {
    pub bytes: u64,
    pub segments: usize,
    pub seconds: f64,
}

/// A read of `len` bytes at `offset` of `file`, and the pieces of it that are copied to the device:
/// (offset in the read, bytes, device address).
struct Chunk {
    file: usize,
    offset: u64,
    len: usize,
    pieces: Vec<(usize, usize, u64)>,
}

fn required_alignment(ckpt: &Checkpoint, name: &str) -> u64 {
    let info = &ckpt.tensors[name];
    if info.bytes >= 16 {
        16
    } else {
        info.dtype.size() as u64
    }
}

/// Uploads every tensor for which `select` returns true.
pub fn upload(
    ckpt: &Checkpoint,
    select: impl Fn(&str) -> bool,
    opts: LoadOptions,
) -> Result<(DeviceWeights, LoadStats)> {
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
            // A segment: tensors whose gaps are at most MAX_GAP, read together. Within it, runs: tensors
            // whose gaps are at most PACK_GAP and that stay aligned relative to the run's first tensor
            // (16 bytes, or their dtype size if smaller), copied as one range; each run starts on a
            // 256-byte boundary of the segment's allocation.
            let seg_begin = list[i].1;
            let mut seg_end = list[i].2;
            let mut j = i + 1;
            while j < list.len() && list[j].1 <= seg_end + MAX_GAP {
                seg_end = seg_end.max(list[j].2);
                j += 1;
            }
            // (file begin, file end, offset in the allocation)
            let mut runs: Vec<(u64, u64, u64)> = Vec::new();
            let mut run_of = Vec::with_capacity(j - i);
            let mut alloc = 0u64;
            for &(name, begin, end) in &list[i..j] {
                let joins = runs.last().is_some_and(|r| {
                    begin <= r.1 + PACK_GAP
                        && (begin - r.0) % required_alignment(ckpt, name) == 0
                });
                if joins {
                    let r = runs.last_mut().unwrap();
                    r.1 = r.1.max(end);
                } else {
                    if let Some(r) = runs.last() {
                        alloc += (r.1 - r.0).div_ceil(256) * 256;
                    }
                    runs.push((begin, end, alloc));
                }
                run_of.push(runs.len() - 1);
            }
            let last = runs.last().unwrap();
            let len = (alloc + (last.1 - last.0)) as usize;
            let buffer = DeviceBuffer::new(len)?;
            for (&(name, begin, end), &r) in list[i..j].iter().zip(&run_of) {
                let info = &ckpt.tensors[name];
                tensors.insert(
                    name.to_string(),
                    DeviceTensor {
                        addr: buffer.addr() + runs[r].2 + (begin - runs[r].0),
                        bytes: end - begin,
                        dtype: info.dtype,
                        shape: info.shape.clone(),
                        scale: 0,
                    },
                );
            }
            // Reads of up to chunk_bytes over the segment's file range, each copying the runs it covers.
            let mut at = seg_begin;
            let mut r = 0;
            while at < seg_end {
                let n = (opts.chunk_bytes as u64).min(seg_end - at);
                let mut pieces = Vec::new();
                while r < runs.len() && runs[r].0 < at + n {
                    let (b, e, dst) = runs[r];
                    let (pb, pe) = (b.max(at), e.min(at + n));
                    if pe > pb {
                        pieces.push((
                            (pb - at) as usize,
                            (pe - pb) as usize,
                            buffer.addr() + dst + (pb - b),
                        ));
                    }
                    if e <= at + n {
                        r += 1;
                    } else {
                        break;
                    }
                }
                if !pieces.is_empty() {
                    chunks.push(Chunk {
                        file,
                        offset: at,
                        len: n as usize,
                        pieces,
                    });
                }
                at += n;
            }
            total += len as u64;
            segments.push(buffer);
            i = j;
        }
    }

    // Largest chunks first keeps the threads evenly loaded at the end.
    chunks.sort_by_key(|c| std::cmp::Reverse(c.len));
    let direct = if opts.direct {
        open_direct(&ckpt.files)
    } else {
        None
    };
    let next = AtomicUsize::new(0);
    let error = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..opts.threads.max(1) {
            scope.spawn(|| {
                if let Err(e) =
                    upload_worker(ckpt, direct.as_deref(), &chunks, &next, opts.chunk_bytes)
                {
                    error.lock().unwrap().get_or_insert(e);
                    next.store(chunks.len(), Ordering::Relaxed);
                }
            });
        }
    });
    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }

    let stats = LoadStats {
        bytes: total,
        segments: segments.len(),
        seconds: start.elapsed().as_secs_f64(),
    };
    Ok((DeviceWeights { segments, tensors }, stats))
}

/// Direct I/O block size: offsets, lengths and buffer addresses of `O_DIRECT` reads are multiples of it.
const DIRECT_ALIGN: u64 = 4096;

fn open_direct(files: &[crate::checkpoint::SafetensorsFile]) -> Option<Vec<File>> {
    files
        .iter()
        .map(|f| crate::fileio::open_direct(&f.path))
        .collect()
}

fn upload_worker(
    ckpt: &Checkpoint,
    direct: Option<&[File]>,
    chunks: &[Chunk],
    next: &AtomicUsize,
    chunk_bytes: usize,
) -> Result<()> {
    let stream = Stream::new()?;
    let buffer_bytes = chunk_bytes + 2 * DIRECT_ALIGN as usize;
    let mut buffers = [
        PinnedBuffer::new(buffer_bytes)?,
        PinnedBuffer::new(buffer_bytes)?,
    ];
    let events = [Event::new()?, Event::new()?];
    let mut used = [false; 2];
    let mut slot = 0;
    loop {
        let index = next.fetch_add(1, Ordering::Relaxed);
        let Some(chunk) = chunks.get(index) else {
            break;
        };
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
                    let n = crate::fileio::read_at(
                        &files[chunk.file],
                        &mut window[filled..],
                        begin + filled as u64,
                    )?;
                    anyhow::ensure!(
                        n > 0,
                        "unexpected end of {}",
                        ckpt.files[chunk.file].path.display()
                    );
                    filled += n;
                }
                &buffer[skip..skip + chunk.len]
            }
            None => {
                crate::fileio::read_exact_at(
                    &ckpt.files[chunk.file].file,
                    &mut buffer[..chunk.len],
                    chunk.offset,
                )?;
                &buffer[..chunk.len]
            }
        };
        for &(off, len, dst) in &chunk.pieces {
            unsafe { stream.copy_h2d(dst, data.as_ptr().add(off), len)? };
        }
        events[slot].record(&stream)?;
        used[slot] = true;
        slot ^= 1;
    }
    stream.synchronize()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes a safetensors file of U8 tensors laid out back to back (in the given order).
    fn write(path: &std::path::Path, tensors: &[(String, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut at = 0;
        for (name, data) in tensors {
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "U8", "shape": [data.len()], "data_offsets": [at, at + data.len()]}),
            );
            at += data.len();
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        tensors.iter().for_each(|t| bytes.extend_from_slice(&t.1));
        std::fs::write(path, bytes).unwrap();
    }

    /// Selected tensors separated by unselected gaps of every size class (within a run, packed out of a
    /// segment, splitting segments), odd sizes and chunk boundaries inside tensors: every selected
    /// tensor reads back exactly, and gaps larger than PACK_GAP take no device memory.
    #[test]
    fn upload_packs_runs() {
        quench6k_cuda::set_device(0).unwrap();
        let dir = std::env::temp_dir().join(format!("quench6k-loader-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut x = 0x2545_f491_u32;
        let mut bytes = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    x as u8
                })
                .collect()
        };
        let sizes = [
            (1000, 10),
            (3, 100),
            (50_000, 60_000),
            (16, 2),
            (200_000, 3_000_000),
            (77_777, 4096),
            (5, 0),
            (40_000, 1),
        ];
        let mut tensors = Vec::new();
        for (k, &(keep, gap)) in sizes.iter().enumerate() {
            tensors.push((format!("keep{k}"), bytes(keep)));
            if gap > 0 {
                tensors.push((format!("skip{k}"), bytes(gap)));
            }
        }
        write(&dir.join("a.safetensors"), &tensors);
        let ckpt = Checkpoint::open(&dir).unwrap();
        let opts = LoadOptions {
            chunk_bytes: 32 << 10,
            threads: 3,
            direct: false,
            ..LoadOptions::default()
        };
        let (w, stats) = upload(&ckpt, |n| n.starts_with("keep"), opts).unwrap();
        let s = Stream::new().unwrap();
        for (name, data) in &tensors {
            if name.starts_with("keep") {
                let t = w.get(name).unwrap();
                let got: Vec<u8> = s.download(t.addr, data.len()).unwrap();
                assert!(got == *data, "{name} differs");
            }
        }
        let kept: usize = tensors.iter().filter(|t| t.0.starts_with("keep")).map(|t| t.1.len()).sum();
        let small_gaps: usize = sizes.iter().map(|s| s.1).filter(|&g| g <= PACK_GAP as usize).sum();
        assert!(
            stats.bytes as usize <= kept + small_gaps + 256 * tensors.len(),
            "{} device bytes for {kept} selected",
            stats.bytes
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
