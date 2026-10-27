//! Stored contexts for prefix reuse beyond the GPU-resident conversation: each entry is the complete
//! sequence state at a prompt position (recurrent state, position table and attention caches up to
//! it) keyed by its tokens. Entries live in pinned host memory up to a byte budget and spill to disk
//! (least recently used first); disk entries persist across restarts and are indexed at startup. The
//! disk tier is kept within a size and an age limit by deleting the least recently used files; a file's
//! modification time is its entry's last use, so the order survives restarts.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

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
    tokens: Vec<i32>,
    bytes: usize,
    /// The data in pinned memory, and its file in the disk tier; at least one of them.
    host: Option<PinnedBuffer>,
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
}

const MAGIC: &[u8; 8] = b"SQCTX01\0";

impl ContextStore {
    pub fn open(opts: StoreOptions) -> Result<Self> {
        let mut entries = Vec::new();
        let mut foreign = Vec::new();
        if let Some(dir) = &opts.dir {
            std::fs::create_dir_all(dir).with_context(|| format!("context directory {}", dir.display()))?;
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
                    Ok((model_id, tokens, bytes)) if model_id == opts.model_id => entries.push(Entry { tokens, bytes, host: None, file: Some(path), used }),
                    Ok(_) => foreign.push((path, meta.len() as usize, used)),
                    Err(e) => eprintln!("ignoring context file {}: {e:#}", path.display()),
                }
            }
        }
        foreign.sort_by_key(|f| f.2);
        let mut store = Self { opts, entries, foreign };
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

    /// The entry with the longest token prefix of `prompt` (shorter than the prompt): (index, length).
    pub fn best(&self, prompt: &[i32]) -> Option<(usize, usize)> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.tokens.len() < prompt.len() && prompt.starts_with(&e.tokens))
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, e)| (i, e.tokens.len()))
    }

    pub fn contains(&self, tokens: &[i32]) -> bool {
        self.entries.iter().any(|e| e.tokens == tokens)
    }

    /// Adds an entry (data in pinned memory), then keeps the host tier within budget.
    pub fn insert(&mut self, tokens: Vec<i32>, data: PinnedBuffer) -> Result<()> {
        let bytes = data.len();
        self.entries.push(Entry { tokens, bytes, host: Some(data), file: None, used: SystemTime::now() });
        self.evict()
    }

    /// The tokens and data of entry `i`, read from disk into pinned memory when needed. Marks the entry
    /// (and its file) as just used.
    pub fn get(&mut self, i: usize) -> Result<(Vec<i32>, &PinnedBuffer)> {
        let e = &mut self.entries[i];
        e.used = SystemTime::now();
        if let Some(path) = &e.file {
            if e.host.is_none() {
                e.host = Some(read_data(path, e.bytes)?);
            }
            if let Err(err) = std::fs::File::options().write(true).open(path).and_then(|f| f.set_modified(e.used)) {
                eprintln!("marking context file {} as used: {err}", path.display());
            }
        }
        Ok((e.tokens.clone(), e.host.as_ref().unwrap()))
    }

    fn host_bytes(&self) -> usize {
        self.entries.iter().filter(|e| e.host.is_some()).map(|e| e.bytes).sum()
    }

    fn disk_bytes(&self) -> usize {
        self.entries.iter().filter(|e| e.file.is_some()).map(Entry::file_bytes).sum::<usize>() + self.foreign.iter().map(|f| f.1).sum::<usize>()
    }

    /// Moves least recently used host entries to disk (or drops them without a disk tier) until the
    /// host tier fits its budget. An entry read back from disk still has its file and is not rewritten.
    fn evict(&mut self) -> Result<()> {
        while self.host_bytes() > self.opts.host_bytes {
            let Some(i) = self.entries.iter().enumerate().filter(|(_, e)| e.host.is_some()).min_by_key(|(_, e)| e.used).map(|(i, _)| i) else { break };
            match &self.opts.dir {
                Some(dir) => {
                    let e = &mut self.entries[i];
                    if e.file.is_none() {
                        let path = dir.join(format!("{:016x}.ctx", hash_tokens(&e.tokens)));
                        write_file(&path, &self.opts.model_id, &e.tokens, e.host.as_ref().unwrap().as_slice(), e.used)?;
                        e.file = Some(path);
                    }
                    e.host = None;
                    self.trim_disk();
                }
                None => {
                    self.entries.remove(i);
                }
            }
        }
        Ok(())
    }

    /// Deletes context files until the disk tier fits its size and age limits: other models' files first
    /// (oldest first), then the least recently used entries' (an entry whose data is also in host memory
    /// stays, without its file).
    fn trim_disk(&mut self) {
        let cutoff = self.opts.max_age.and_then(|a| SystemTime::now().checked_sub(a));
        let expired = |used: SystemTime| cutoff.is_some_and(|c| used < c);
        let mut total = self.disk_bytes();
        loop {
            let over = self.opts.disk_bytes > 0 && total > self.opts.disk_bytes;
            let (path, bytes) = if self.foreign.first().is_some_and(|f| over || expired(f.2)) {
                let (path, bytes, _) = self.foreign.remove(0);
                (path, bytes)
            } else {
                let Some(i) = self.entries.iter().enumerate().filter(|(_, e)| e.file.is_some()).min_by_key(|(_, e)| e.used).map(|(i, _)| i) else { break };
                if !over && !expired(self.entries[i].used) {
                    break;
                }
                let bytes = self.entries[i].file_bytes();
                let path = self.entries[i].file.take().unwrap();
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

    /// Writes every host entry to the disk tier (for persistence at shutdown).
    pub fn persist(&mut self) -> Result<()> {
        let Some(dir) = self.opts.dir.clone() else { return Ok(()) };
        for e in &mut self.entries {
            if let (Some(buf), None) = (&e.host, &e.file) {
                let path = dir.join(format!("{:016x}.ctx", hash_tokens(&e.tokens)));
                write_file(&path, &self.opts.model_id, &e.tokens, buf.as_slice(), e.used)?;
                e.file = Some(path);
            }
        }
        self.trim_disk();
        Ok(())
    }
}

fn hash_tokens(tokens: &[i32]) -> u64 {
    // FNV-1a over the token bytes.
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// File: magic, u64 header length, JSON header {model_id, tokens, bytes}, token ids (i32 LE), data.
/// Its modification time is `used`, the entry's last use.
fn write_file(path: &Path, model_id: &str, tokens: &[i32], data: &[u8], used: SystemTime) -> Result<()> {
    let header = serde_json::json!({"model_id": model_id, "tokens": tokens.len(), "bytes": data.len()}).to_string();
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
        f.into_inner().map_err(|e| e.into_error())?.set_modified(used)?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
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
    let tokens = raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    Ok((h["model_id"].as_str().unwrap_or("").to_string(), tokens, h["bytes"].as_u64().context("bytes")? as usize))
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
    let mut buf = PinnedBuffer::new(bytes)?;
    f.read_exact(buf.as_mut_slice())?;
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
        let mut v: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
        v.sort();
        v
    }

    fn age(path: &Path, days: u64) {
        let t = SystemTime::now() - Duration::from_secs(days * 86400);
        std::fs::File::options().write(true).open(path).unwrap().set_modified(t).unwrap();
    }

    // Needs the CUDA driver (pinned host memory).
    #[test]
    fn disk_tier_limits() {
        let dir = std::env::temp_dir().join(format!("quench6k-store-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("left.tmp"), b"partial").unwrap();
        let opts = |disk_bytes, max_age| StoreOptions { host_bytes: 1000, dir: Some(dir.clone()), disk_bytes, max_age, model_id: "m".into() };
        // A file of another model, written by a store with another identity.
        let mut other = ContextStore::open(StoreOptions { model_id: "other".into(), ..opts(0, None) }).unwrap();
        other.insert(vec![9], data(1000, 9)).unwrap();
        other.persist().unwrap();
        let other_file = files(&dir).into_iter().find(|p| p.extension().unwrap() == "ctx").unwrap();
        age(&other_file, 1);

        // Each entry is 1000 bytes + 4 per token; the host tier holds one.
        let mut st = ContextStore::open(opts(3000, None)).unwrap();
        assert!(!dir.join("left.tmp").exists(), "a partial write is cleaned up");
        assert_eq!(st.len(), 0);
        st.insert(vec![1], data(1000, 1)).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        st.insert(vec![1, 2], data(1000, 2)).unwrap(); // [1] to disk
        std::thread::sleep(Duration::from_millis(20));
        st.insert(vec![1, 2, 3], data(1000, 3)).unwrap(); // [1, 2] to disk: 3 files, over budget, the other model's goes
        assert!(!other_file.exists());
        assert_eq!(files(&dir).len(), 2);
        // Reusing [1] renews it, so [1, 2] is the least recently used when the next one spills.
        let (i, _) = st.best(&[1, 7]).unwrap();
        assert_eq!(st.get(i).unwrap().1.as_slice()[0], 1);
        std::thread::sleep(Duration::from_millis(20));
        st.insert(vec![4], data(1000, 4)).unwrap(); // [1] (host, has a file) and [1, 2, 3] leave the host tier
        assert!(st.best(&[1, 2, 3, 4]).is_some_and(|(_, n)| n == 3));
        assert!(st.best(&[1, 2, 5]).is_some_and(|(_, n)| n == 1), "[1, 2] was deleted");
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
}
