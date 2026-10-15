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

/// Segment base alignment relative to the file's data section; tensors keep their relative alignment.
const SEGMENT_ALIGN: u64 = 256;
/// Unselected bytes tolerated inside a segment instead of splitting it.
const MAX_GAP: u64 = 1 << 20;

#[derive(Clone, Debug)]
pub struct DeviceTensor {
    pub addr: u64,
    pub bytes: u64,
    pub dtype: DType,
    pub shape: Vec<usize>,
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
        let data_start = ckpt.files[file].data_start;
        let mut i = 0;
        while i < list.len() {
            let seg_begin = data_start + (list[i].1 - data_start) / SEGMENT_ALIGN * SEGMENT_ALIGN;
            let mut seg_end = list[i].2;
            let mut j = i + 1;
            while j < list.len() && list[j].1 <= seg_end + MAX_GAP {
                seg_end = seg_end.max(list[j].2);
                j += 1;
            }
            let len = (seg_end - seg_begin) as usize;
            let buffer = DeviceBuffer::new(len)?;
            for &(name, begin, end) in &list[i..j] {
                let info = &ckpt.tensors[name];
                tensors.insert(
                    name.to_string(),
                    DeviceTensor { addr: buffer.addr() + (begin - seg_begin), bytes: end - begin, dtype: info.dtype, shape: info.shape.clone() },
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
