//! Stored contexts for prefix reuse beyond the GPU-resident conversation: each entry is the complete
//! sequence state at a prompt position (recurrent state, position table and attention caches up to
//! it) keyed by its tokens. Entries live in pinned host memory up to a byte budget and are persisted
//! eagerly by a bounded background writer; disk entries survive restarts and are indexed at startup. The
//! disk tier is kept within a size and an age limit by deleting the least recently used files; a file's
//! modification time is its entry's last use, so the order survives restarts. Entries of a model
//! variant (a steering strength, an adapter) carry its tag and only match prompts of that variant.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, ensure};
use quench6k_cuda::PinnedBuffer;

#[derive(Clone, Debug, Default)]
pub struct StoreOptions {
    /// Pinned host memory for stored contexts (0: no host tier).
    pub host_bytes: usize,
    /// Directory for the disk tier and persistence (None: no disk tier).
    pub dir: Option<PathBuf>,
    /// Most bytes of context files in `dir`, including those of other models (0: no limit).
    pub disk_bytes: usize,
    /// Context files unused for longer are deleted (None: no limit).
    pub max_age: Option<Duration>,
    /// Identity of the model and numerics; entries written under another identity are ignored.
    pub model_id: String,
}

struct Entry {
    id: usize,
    tokens: Vec<i32>,
    /// The model variant the state was computed with ("" for the base model).
    variant: String,
    bytes: usize,
    /// The data in pinned memory, and its file in the disk tier; at least one of them.
    host: Option<Arc<PinnedBuffer>>,
    pending: bool,
    disk_evicted: bool,
    write_error: Option<String>,
    file: Option<PathBuf>,
    /// Last written or loaded.
    used: SystemTime,
}

impl Entry {
    fn file_bytes(&self) -> usize {
        self.bytes + 4 * self.tokens.len()
    }
}

pub struct ContextStore {
    opts: StoreOptions,
    entries: Vec<Entry>,
    /// Context files of other models or numerics (path, bytes, modified), oldest first: deleted before
    /// any entry.
    foreign: Vec<(PathBuf, usize, SystemTime)>,
    writer: Option<DiskWriter>,
    next_id: usize,
}

/// One bounded disk task; its buffer remains owned and accounted for by its host-tier entry.
type WriteTask = Box<dyn FnOnce() -> WriteResult + Send>;

struct WriteResult {
    id: usize,
    path: PathBuf,
    result: Result<()>,
}

struct DiskWriter {
    tx: Option<mpsc::SyncSender<WriteTask>>,
    rx: mpsc::Receiver<WriteResult>,
    thread: Option<std::thread::JoinHandle<()>>,
    busy: bool,
}

impl DiskWriter {
    fn new() -> Result<Self> {
        Self::with_executor(|job| job())
    }

    fn with_executor(
        mut run: impl FnMut(WriteTask) -> WriteResult + Send + 'static,
    ) -> Result<Self> {
        let (tx, jobs) = mpsc::sync_channel::<WriteTask>(1);
        let (done, rx) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("context-writer".into())
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    if done.send(run(job)).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self {
            tx: Some(tx),
            rx,
            thread: Some(thread),
            busy: false,
        })
    }

    fn submit(&mut self, task: WriteTask) -> Result<()> {
        anyhow::ensure!(!self.busy, "context writer already busy");
        self.tx
            .as_ref()
            .unwrap()
            .try_send(task)
            .map_err(|_| anyhow::anyhow!("context writer queue unavailable"))?;
        self.busy = true;
        Ok(())
    }

    fn completion(&mut self, wait: bool) -> Result<Option<WriteResult>> {
        if !self.busy {
            return Ok(None);
        }
        let result = if wait {
            Some(self.rx.recv().context("context writer stopped")?)
        } else {
            match self.rx.try_recv() {
                Ok(result) => Some(result),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => anyhow::bail!("context writer stopped"),
            }
        };
        if result.is_some() {
            self.busy = false;
        }
        Ok(result)
    }
}

impl Drop for DiskWriter {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn slow_io(start: Instant) -> bool {
    start.elapsed() >= Duration::from_millis(100) || std::env::var_os("QUENCH6K_TIMING").is_some()
}

const MAGIC: &[u8; 8] = b"SQCTX01\0";

impl ContextStore {
    pub fn open(opts: StoreOptions) -> Result<Self> {
        let mut entries = Vec::new();
        let mut foreign = Vec::new();
        if let Some(dir) = &opts.dir {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("context directory {}", dir.display()))?;
            for e in std::fs::read_dir(dir)? {
                let path = e?.path();
                match path.extension().and_then(|x| x.to_str()) {
                    Some("ctx") => {}
                    // Left by a write that did not finish.
                    Some("tmp") => {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                    _ => continue,
                }
                let meta = std::fs::metadata(&path)?;
                let used = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                match read_header(&path) {
                    Ok((model_id, tokens, bytes)) => match variant_of(&model_id, &opts.model_id) {
                        Some(variant) => entries.push(Entry {
                            id: entries.len(),
                            pending: false,
                            disk_evicted: false,
                            write_error: None,
                            tokens,
                            variant: variant.to_string(),
                            bytes,
                            host: None,
                            file: Some(path),
                            used,
                        }),
                        None => foreign.push((path, meta.len() as usize, used)),
                    },
                    Err(e) => eprintln!("ignoring context file {}: {e:#}", path.display()),
                }
            }
        }
        foreign.sort_by_key(|f| f.2);
        let next_id = entries.len();
        let writer = if opts.dir.is_some() {
            Some(DiskWriter::new()?)
        } else {
            None
        };
        let mut store = Self {
            next_id,
            writer,
            opts,
            entries,
            foreign,
        };
        store.trim_disk();
        Ok(store)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn enabled(&self) -> bool {
        self.opts.host_bytes > 0 || self.opts.dir.is_some()
    }

    /// The entry of `variant` with the longest token prefix of `prompt` (shorter than the prompt):
    /// (stable entry ID, length).
    pub fn best(&self, prompt: &[i32], variant: &str) -> Option<(usize, usize)> {
        self.entries
            .iter()
            .filter(|e| {
                e.variant == variant
                    && e.tokens.len() < prompt.len()
                    && prompt.starts_with(&e.tokens)
            })
            .max_by_key(|e| e.tokens.len())
            .map(|e| (e.id, e.tokens.len()))
    }

    pub fn contains(&self, tokens: &[i32], variant: &str) -> bool {
        self.entries
            .iter()
            .any(|e| e.variant == variant && e.tokens == tokens)
    }

    /// Adds an immutable host entry and starts eager background persistence. The writer shares this
    /// allocation; queued/in-flight bytes still count against the host budget until writing finishes.
    pub fn insert(&mut self, tokens: Vec<i32>, variant: &str, data: PinnedBuffer) -> Result<()> {
        self.poll_io(false)?;
        let bytes = data.len();
        // Make room before committing the new allocation. A failed spill must not leave another
        // entry behind on every subsequent insertion attempt.
        self.evict_to(self.opts.host_bytes.saturating_sub(bytes), None)?;
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(Entry {
            id,
            tokens,
            variant: variant.to_string(),
            bytes,
            host: Some(Arc::new(data)),
            file: None,
            used: SystemTime::now(),
            pending: false,
            disk_evicted: false,
            write_error: None,
        });
        let result = self.schedule_write().and_then(|()| self.evict(None));
        if result.is_err() {
            if let Some(i) = self.entries.iter().position(|e| e.id == id && !e.pending) {
                self.entries.remove(i);
            }
        }
        result
    }

    /// Returns an entry by its stable ID. Loading a disk entry also trims other host entries, so
    /// repeatedly reading disk contexts cannot silently grow the pinned tier past its budget.
    pub fn get(&mut self, id: usize) -> Result<(Vec<i32>, &PinnedBuffer)> {
        self.poll_io(false)?;
        let i = self.index(id)?;
        self.entries[i].used = SystemTime::now();
        self.entries[i].disk_evicted = false;
        let loaded = self.entries[i].host.is_none();
        if self.entries[i].host.is_none() {
            let e = &self.entries[i];
            let path = e
                .file
                .as_ref()
                .context("context has neither host data nor a disk file")?;
            let data = read_data(path, e.bytes)?;
            self.entries[i].host = Some(Arc::new(data));
        }
        if let Some(path) = &self.entries[i].file {
            if let Err(err) = std::fs::File::options()
                .write(true)
                .open(path)
                .and_then(|f| f.set_modified(self.entries[i].used))
            {
                eprintln!("marking context file {} as used: {err}", path.display());
            }
        }
        if let Err(err) = self.evict(Some(id)) {
            if loaded {
                let i = self.index(id)?;
                self.entries[i].host = None;
            }
            return Err(err);
        }
        let e = &self.entries[self.index(id)?];
        Ok((e.tokens.clone(), e.host.as_deref().unwrap()))
    }

    fn index(&self, id: usize) -> Result<usize> {
        self.entries
            .iter()
            .position(|e| e.id == id)
            .context("context entry is no longer available")
    }

    /// Collect completed writes and keep the eager writer busy, without waiting on disk I/O.
    pub fn poll(&mut self) -> Result<()> {
        self.poll_io(false)?;
        self.schedule_write()?;
        self.trim_disk();
        Ok(())
    }

    /// Release a get() call's protected entry after its host-to-device transfer has synchronized.
    pub fn trim_host(&mut self) -> Result<()> {
        self.evict(None)
    }

    fn poll_io(&mut self, wait: bool) -> Result<()> {
        let result = match &mut self.writer {
            Some(writer) => writer.completion(wait)?,
            None => None,
        };
        if let Some(result) = result {
            let i = self.index(result.id)?;
            self.entries[i].pending = false;
            match result.result {
                Ok(()) => {
                    let e = &mut self.entries[i];
                    // A request may have reused the host entry while the writer was working.
                    if let Err(err) = std::fs::File::options()
                        .write(true)
                        .open(&result.path)
                        .and_then(|f| f.set_modified(e.used))
                    {
                        eprintln!(
                            "marking context file {} as used: {err}",
                            result.path.display()
                        );
                    }
                    e.file = Some(result.path);
                }
                Err(err) => {
                    let message = format!("persisting context {}: {err:#}", result.path.display());
                    self.entries[i].write_error = Some(message.clone());
                    anyhow::bail!("{message}");
                }
            }
        }
        Ok(())
    }

    fn fits_disk(&self, e: &Entry) -> bool {
        self.opts.disk_bytes == 0 || e.file_bytes() <= self.opts.disk_bytes
    }

    fn start_write(&mut self, id: usize) -> Result<()> {
        let i = self.index(id)?;
        let e = &self.entries[i];
        let dir = self.opts.dir.as_ref().context("disk cache disabled")?;
        let path = dir.join(file_name(&e.tokens, &e.variant));
        let tokens = e.tokens.clone();
        let model_id = file_model_id(&self.opts.model_id, &e.variant);
        let data = e.host.as_ref().context("context has no host data")?.clone();
        let used = e.used;
        let bytes = e.bytes;
        self.writer
            .as_mut()
            .context("disk writer unavailable")?
            .submit(Box::new(move || {
                let start = Instant::now();
                let result = write_file(&path, &model_id, &tokens, data.as_slice(), used);
                if slow_io(start) {
                    eprintln!(
                        "cache disk write: {} MiB in {:.3}s{}",
                        bytes >> 20,
                        start.elapsed().as_secs_f64(),
                        if result.is_err() { " (failed)" } else { "" }
                    );
                }
                WriteResult { id, path, result }
            }))?;
        self.entries[i].pending = true;
        // Reserve the in-flight file's bytes against the disk budget as well.
        self.trim_disk();
        Ok(())
    }

    fn schedule_write(&mut self) -> Result<()> {
        if self.writer.as_ref().is_none_or(|w| w.busy) {
            return Ok(());
        }
        let id = self
            .entries
            .iter()
            .filter(|e| {
                e.host.is_some()
                    && e.file.is_none()
                    && !e.pending
                    && !e.disk_evicted
                    && e.write_error.is_none()
                    && self.fits_disk(e)
            })
            .min_by_key(|e| e.used)
            .map(|e| e.id);
        if let Some(id) = id {
            self.start_write(id)?;
        }
        Ok(())
    }

    fn ensure_file(&mut self, id: usize) -> Result<()> {
        loop {
            self.poll_io(false)?;
            let e = &self.entries[self.index(id)?];
            if e.file.is_some() {
                return Ok(());
            }
            if let Some(err) = &e.write_error {
                anyhow::bail!("{err}");
            }
            if self.writer.as_ref().is_some_and(|w| w.busy) {
                let start = Instant::now();
                self.poll_io(true)?;
                if slow_io(start) {
                    eprintln!(
                        "cache writer backpressure: {:.3}s",
                        start.elapsed().as_secs_f64()
                    );
                }
            } else {
                self.start_write(id)?;
            }
        }
    }

    fn host_bytes(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.host.is_some())
            .map(|e| e.bytes)
            .sum()
    }

    fn disk_bytes(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.file.is_some() || e.pending)
            .map(Entry::file_bytes)
            .sum::<usize>()
            + self.foreign.iter().map(|f| f.1).sum::<usize>()
    }

    /// Keep pinned memory bounded. Pending writes retain their host allocation; if disk cannot keep
    /// up, explicit backpressure waits for completion rather than growing an unbounded queue.
    fn evict(&mut self, protect: Option<usize>) -> Result<()> {
        self.evict_to(self.opts.host_bytes, protect)
    }

    fn evict_to(&mut self, budget: usize, protect: Option<usize>) -> Result<()> {
        while self.host_bytes() > budget {
            self.poll_io(false)?;
            let Some(id) = self
                .entries
                .iter()
                .filter(|e| e.host.is_some() && Some(e.id) != protect)
                .min_by_key(|e| e.used)
                .map(|e| e.id)
            else {
                break;
            };
            let i = self.index(id)?;
            if self.opts.dir.is_none()
                || self.entries[i].disk_evicted
                || !self.fits_disk(&self.entries[i])
            {
                if self.entries[i].pending {
                    self.poll_io(true)?;
                    continue;
                }
                self.entries.remove(i);
            } else {
                self.ensure_file(id)?;
                let i = self.index(id)?;
                self.entries[i].host = None;
                self.trim_disk();
            }
        }
        self.schedule_write()?;
        Ok(())
    }

    /// Deletes context files until the disk tier fits its size and age limits: other models' files first
    /// (oldest first), then the least recently used entries' (an entry whose data is also in host memory
    /// stays, without its file).
    fn trim_disk(&mut self) {
        let cutoff = self
            .opts
            .max_age
            .and_then(|a| SystemTime::now().checked_sub(a));
        let expired = |used: SystemTime| cutoff.is_some_and(|c| used < c);
        let mut total = self.disk_bytes();
        loop {
            let over = self.opts.disk_bytes > 0 && total > self.opts.disk_bytes;
            let (path, bytes) = if self.foreign.first().is_some_and(|f| over || expired(f.2)) {
                let (path, bytes, _) = self.foreign.remove(0);
                (path, bytes)
            } else {
                let Some(i) = self
                    .entries
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.file.is_some())
                    .min_by_key(|(_, e)| e.used)
                    .map(|(i, _)| i)
                else {
                    break;
                };
                if !over && !expired(self.entries[i].used) {
                    break;
                }
                let bytes = self.entries[i].file_bytes();
                let path = self.entries[i].file.take().unwrap();
                self.entries[i].disk_evicted = true;
                if self.entries[i].host.is_none() {
                    self.entries.remove(i);
                }
                (path, bytes)
            };
            if let Err(e) = std::fs::remove_file(&path)
                && e.kind() != std::io::ErrorKind::NotFound
            {
                eprintln!("deleting context file {}: {e}", path.display());
            }
            total -= bytes;
        }
    }

    /// Drain background writes and persist each eligible host entry once before shutdown.
    pub fn persist(&mut self) -> Result<()> {
        if self.opts.dir.is_none() {
            return Ok(());
        }
        self.poll_io(false)?;
        let ids: Vec<_> = self
            .entries
            .iter()
            .filter(|e| {
                e.host.is_some() && e.file.is_none() && !e.disk_evicted && self.fits_disk(e)
            })
            .map(|e| e.id)
            .collect();
        for id in ids {
            self.ensure_file(id)?;
        }
        if self.writer.as_ref().is_some_and(|w| w.busy) {
            self.poll_io(true)?;
        }
        self.trim_disk();
        Ok(())
    }
}

/// File name of an entry: FNV-1a over the token bytes, then the variant's (none for the base model).
fn file_name(tokens: &[i32], variant: &str) -> String {
    let h = tokens
        .iter()
        .flat_map(|t| t.to_le_bytes())
        .chain(variant.bytes())
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
        });
    format!("{h:016x}.ctx")
}

/// The identity a variant's files are written under: the model's, plus "+variant".
fn file_model_id(model_id: &str, variant: &str) -> String {
    if variant.is_empty() {
        model_id.to_string()
    } else {
        format!("{model_id}+{variant}")
    }
}

/// The variant of a file written under `file_id`, when it belongs to `model_id`.
fn variant_of<'a>(file_id: &'a str, model_id: &str) -> Option<&'a str> {
    if file_id == model_id {
        return Some("");
    }
    file_id
        .strip_prefix(model_id)
        .and_then(|rest| rest.strip_prefix('+'))
}

/// File: magic, u64 header length, JSON header {model_id, tokens, bytes}, token ids (i32 LE), data.
/// Its modification time is `used`, the entry's last use.
fn write_file(
    path: &Path,
    model_id: &str,
    tokens: &[i32],
    data: &[u8],
    used: SystemTime,
) -> Result<()> {
    let header =
        serde_json::json!({"model_id": model_id, "tokens": tokens.len(), "bytes": data.len()})
            .to_string();
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(MAGIC)?;
        f.write_all(&(header.len() as u64).to_le_bytes())?;
        f.write_all(header.as_bytes())?;
        for t in tokens {
            f.write_all(&t.to_le_bytes())?;
        }
        f.write_all(data)?;
        f.into_inner()
            .map_err(|e| e.into_error())?
            .set_modified(used)?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// The token IDs a context file holds (the conversation, as the model saw it).
pub fn file_tokens(path: &Path) -> Result<Vec<i32>> {
    Ok(read_header(path)?.1)
}

fn read_header(path: &Path) -> Result<(String, Vec<i32>, usize)> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    ensure!(&magic == MAGIC, "not a quench6k context file");
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut header)?;
    let h: serde_json::Value = serde_json::from_slice(&header)?;
    let n = h["tokens"].as_u64().context("tokens")? as usize;
    let mut raw = vec![0u8; n * 4];
    f.read_exact(&mut raw)?;
    let tokens = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok((
        h["model_id"].as_str().unwrap_or("").to_string(),
        tokens,
        h["bytes"].as_u64().context("bytes")? as usize,
    ))
}

fn read_data(path: &Path, bytes: usize) -> Result<PinnedBuffer> {
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut header)?;
    let h: serde_json::Value = serde_json::from_slice(&header)?;
    let n = h["tokens"].as_u64().context("tokens")? as usize;
    std::io::copy(&mut (&mut f).take((n * 4) as u64), &mut std::io::sink())?;
    let start = Instant::now();
    let mut buf = PinnedBuffer::new(bytes)?;
    let read_start = Instant::now();
    f.read_exact(buf.as_mut_slice())?;
    if slow_io(start) {
        eprintln!(
            "cache disk read: {} MiB, alloc {:.3}s read {:.3}s",
            bytes >> 20,
            (read_start - start).as_secs_f64(),
            read_start.elapsed().as_secs_f64()
        );
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(bytes: usize, fill: u8) -> PinnedBuffer {
        let mut b = PinnedBuffer::new(bytes).unwrap();
        b.as_mut_slice().fill(fill);
        b
    }

    fn files(dir: &Path) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        v.sort();
        v
    }

    fn age(path: &Path, days: u64) {
        let t = SystemTime::now() - Duration::from_secs(days * 86400);
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(t)
            .unwrap();
    }

    // Needs the CUDA driver (pinned host memory).
    #[test]
    fn disk_tier_limits() {
        let dir = std::env::temp_dir().join(format!("quench6k-store-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("left.tmp"), b"partial").unwrap();
        let opts = |disk_bytes, max_age| StoreOptions {
            host_bytes: 1000,
            dir: Some(dir.clone()),
            disk_bytes,
            max_age,
            model_id: "m".into(),
        };
        // A file of another model, written by a store with another identity.
        let mut other = ContextStore::open(StoreOptions {
            model_id: "other".into(),
            ..opts(0, None)
        })
        .unwrap();
        other.insert(vec![9], "", data(1000, 9)).unwrap();
        other.persist().unwrap();
        let other_file = files(&dir)
            .into_iter()
            .find(|p| p.extension().unwrap() == "ctx")
            .unwrap();
        age(&other_file, 1);

        // Each entry is 1000 bytes + 4 per token; the host tier holds one.
        let mut st = ContextStore::open(opts(3000, None)).unwrap();
        assert!(
            !dir.join("left.tmp").exists(),
            "a partial write is cleaned up"
        );
        assert_eq!(st.len(), 0);
        st.insert(vec![1], "", data(1000, 1)).unwrap();
        st.persist().unwrap();
        std::thread::sleep(Duration::from_millis(20));
        st.insert(vec![1, 2], "", data(1000, 2)).unwrap();
        st.persist().unwrap();
        assert!(
            !other_file.exists(),
            "foreign files leave first as disk space is reserved"
        );
        assert_eq!(files(&dir).len(), 2);
        // Renew [1] before the next eager write, making [1, 2] the disk LRU.
        let (id, _) = st.best(&[1, 7], "").unwrap();
        assert_eq!(st.get(id).unwrap().1.as_slice()[0], 1);
        std::thread::sleep(Duration::from_millis(20));
        st.insert(vec![4], "", data(1000, 4)).unwrap();
        st.persist().unwrap();
        assert!(st.best(&[4, 5], "").is_some_and(|(_, n)| n == 1));
        assert!(
            st.best(&[1, 2, 5], "").is_some_and(|(_, n)| n == 1),
            "[1, 2] was deleted"
        );
        assert_eq!(files(&dir).len(), 2);
        drop(st);

        // At startup the order comes from the files' modification times; an age limit deletes old ones.
        let mut names = files(&dir);
        names.retain(|p| p.extension().unwrap() == "ctx");
        let st = ContextStore::open(opts(0, None)).unwrap();
        assert_eq!(st.len(), 2);
        drop(st);
        age(&names[0], 10);
        let st = ContextStore::open(opts(0, Some(Duration::from_secs(5 * 86400)))).unwrap();
        assert_eq!(st.len(), 1);
        assert!(!names[0].exists() && names[1].exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn test_store(name: &str, host_bytes: usize, disk_bytes: usize) -> (ContextStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!("quench6k-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = ContextStore::open(StoreOptions {
            host_bytes,
            disk_bytes,
            dir: Some(dir.clone()),
            model_id: "test".into(),
            ..Default::default()
        })
        .unwrap();
        (store, dir)
    }

    fn blocked_writer() -> (DiskWriter, mpsc::Receiver<()>, mpsc::Sender<()>) {
        let (started, ready) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let mut first = true;
        let writer = DiskWriter::with_executor(move |job| {
            if first {
                first = false;
                started.send(()).unwrap();
                gate.recv().unwrap();
            }
            job()
        })
        .unwrap();
        (writer, ready, release)
    }

    #[test]
    fn pending_write_is_nonblocking_readable_and_memory_accounted() {
        let (mut store, dir) = test_store("pending-cache", 2000, 0);
        let (writer, ready, release) = blocked_writer();
        store.writer = Some(writer);
        store.insert(vec![1], "", data(1000, 7)).unwrap();
        ready.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(store.entries[0].pending);
        assert!(store.entries[0].file.is_none());
        assert_eq!(store.host_bytes(), 1000);
        let id = store.best(&[1, 2], "").unwrap().0;
        assert_eq!(store.get(id).unwrap().1.as_slice()[999], 7);
        store.insert(vec![2], "", data(500, 9)).unwrap();
        assert_eq!(store.host_bytes(), 1500);
        assert_eq!(store.entries.iter().filter(|e| e.pending).count(), 1);
        release.send(()).unwrap();
        store.persist().unwrap();
        assert!(store.entries.iter().all(|e| e.file.is_some() && !e.pending));
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn disk_backpressure_bounds_host_memory_and_shutdown_drains() {
        let (mut store, dir) = test_store("bounded-cache", 1000, 0);
        let (writer, ready, release) = blocked_writer();
        store.writer = Some(writer);
        store.insert(vec![1], "", data(1000, 1)).unwrap();
        ready.recv_timeout(Duration::from_secs(2)).unwrap();
        let (started, waiting) = mpsc::channel();
        let (finished, done) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            started.send(()).unwrap();
            store.insert(vec![2], "", data(1000, 2)).unwrap();
            finished.send(()).unwrap();
            store
        });
        waiting.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(
            done.recv_timeout(Duration::from_millis(30)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        release.send(()).unwrap();
        done.recv_timeout(Duration::from_secs(2)).unwrap();
        let mut store = thread.join().unwrap();
        assert!(store.host_bytes() <= 1000);
        store.persist().unwrap();
        assert!(!store.writer.as_ref().unwrap().busy);
        drop(store);
        let mut restored = ContextStore::open(StoreOptions {
            host_bytes: 1000,
            dir: Some(dir.clone()),
            model_id: "test".into(),
            ..Default::default()
        })
        .unwrap();
        for (tokens, value) in [(vec![1, 9], 1), (vec![2, 9], 2)] {
            let id = restored.best(&tokens, "").unwrap().0;
            assert_eq!(restored.get(id).unwrap().1.as_slice()[0], value);
        }
        drop(restored);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn repeated_disk_reads_enforce_host_budget() {
        let (mut store, dir) = test_store("read-budget-cache", 0, 0);
        for value in 1..=3 {
            store
                .insert(vec![value], "", data(1000, value as u8))
                .unwrap();
        }
        assert_eq!(store.host_bytes(), 0);
        store.opts.host_bytes = 1000;
        for value in [1, 2, 3, 1] {
            let id = store.best(&[value, 9], "").unwrap().0;
            assert_eq!(store.get(id).unwrap().1.as_slice()[0], value as u8);
            assert_eq!(store.host_bytes(), 1000);
        }
        store.opts.host_bytes = 0;
        store.trim_host().unwrap();
        assert_eq!(store.host_bytes(), 0);
        let id = store.best(&[1, 9], "").unwrap().0;
        assert_eq!(store.get(id).unwrap().1.as_slice()[0], 1);
        store.trim_host().unwrap();
        assert_eq!(store.host_bytes(), 0);
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn disk_eviction_does_not_rewrite_cold_host_entries() {
        let (mut store, dir) = test_store("disk-budget-cache", 4000, 1500);
        for value in 1..=3 {
            store
                .insert(vec![value], "", data(1000, value as u8))
                .unwrap();
        }
        store.persist().unwrap();
        assert!(store.disk_bytes() <= 1500);
        assert_eq!(store.entries.iter().filter(|e| e.disk_evicted).count(), 2);
        store.poll().unwrap();
        assert!(!store.writer.as_ref().unwrap().busy);
        // A host hit makes an entry hot again and eligible for disk persistence.
        let id = store.best(&[1, 9], "").unwrap().0;
        assert_eq!(store.get(id).unwrap().1.as_slice()[0], 1);
        store.persist().unwrap();
        assert!(store.entries[store.index(id).unwrap()].file.is_some());
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn background_write_failure_keeps_host_data_and_surfaces_error() {
        let (mut store, dir) = test_store("failed-cache", 1000, 0);
        let failed_path = dir.join("failed.ctx");
        store.writer = Some(
            DiskWriter::with_executor(move |job| {
                drop(job);
                WriteResult {
                    id: 0,
                    path: failed_path.clone(),
                    result: Err(anyhow::anyhow!("injected disk failure")),
                }
            })
            .unwrap(),
        );
        store.insert(vec![1], "", data(1000, 4)).unwrap();
        assert!(format!("{:#}", store.persist().unwrap_err()).contains("injected disk failure"));
        assert!(store.entries[0].file.is_none());
        assert!(store.entries[0].write_error.is_some());
        let id = store.best(&[1, 9], "").unwrap().0;
        assert_eq!(store.get(id).unwrap().1.as_slice()[0], 4);
        assert!(store.persist().is_err());
        assert!(store.insert(vec![2], "", data(1000, 5)).is_err());
        assert_eq!(store.host_bytes(), 1000);
        assert_eq!(
            store.len(),
            1,
            "failed spills must not accumulate new host allocations"
        );
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn oversized_disk_entries_do_not_loop_or_leave_partial_files() {
        let (mut store, dir) = test_store("oversize-cache", 0, 1000);
        store.insert(vec![1], "", data(1000, 2)).unwrap();
        store.persist().unwrap();
        assert!(store.is_empty());
        assert!(files(&dir).is_empty());
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn variant_ids() {
        assert_eq!(variant_of("m", "m"), Some(""));
        assert_eq!(variant_of("m+adapter=1", "m"), Some("adapter=1"));
        assert_eq!(variant_of("mx", "m"), None);
        assert_ne!(file_name(&[1, 2], ""), file_name(&[1, 2], "adapter=1"));
    }

    // Needs the CUDA driver (pinned host memory).
    #[test]
    fn variants_are_separate() {
        let dir = std::env::temp_dir().join(format!(
            "quench6k-store-variant-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let opts = StoreOptions {
            host_bytes: 0,
            dir: Some(dir.clone()),
            disk_bytes: 0,
            max_age: None,
            model_id: "m".into(),
        };
        let mut st = ContextStore::open(opts.clone()).unwrap();
        st.insert(vec![1, 2], "", data(10, 1)).unwrap();
        st.insert(vec![1, 2], "adapter=1", data(10, 2)).unwrap();
        assert_eq!(
            files(&dir).len(),
            2,
            "the same tokens, one file per variant"
        );
        drop(st);
        let mut st = ContextStore::open(opts).unwrap();
        let (i, _) = st.best(&[1, 2, 3], "adapter=1").unwrap();
        assert_eq!(st.get(i).unwrap().1.as_slice()[0], 2);
        let (i, _) = st.best(&[1, 2, 3], "").unwrap();
        assert_eq!(st.get(i).unwrap().1.as_slice()[0], 1);
        assert!(st.best(&[1, 2, 3], "steer=0.5").is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
