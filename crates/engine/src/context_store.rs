//! Stored contexts for prefix reuse beyond the GPU-resident conversation: each entry is the complete
//! sequence state at a prompt position (recurrent state, position table and attention caches up to
//! it) keyed by its tokens. Entries live in pinned host memory up to a byte budget and spill to disk
//! (least recently used first); disk entries persist across restarts and are indexed at startup.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use quench6k_cuda::PinnedBuffer;

#[derive(Clone, Debug, Default)]
pub struct StoreOptions {
    /// Pinned host memory for stored contexts (0: no host tier).
    pub host_bytes: usize,
    /// Directory for the disk tier and persistence (None: no disk tier).
    pub dir: Option<PathBuf>,
    /// Identity of the model and numerics; entries written under another identity are ignored.
    pub model_id: String,
}

enum Loc {
    Host(PinnedBuffer),
    Disk(PathBuf),
}

struct Entry {
    tokens: Vec<i32>,
    bytes: usize,
    loc: Loc,
    used: u64,
}

pub struct ContextStore {
    opts: StoreOptions,
    entries: Vec<Entry>,
    clock: u64,
}

const MAGIC: &[u8; 8] = b"SQCTX01\0";

impl ContextStore {
    pub fn open(opts: StoreOptions) -> Result<Self> {
        let mut entries = Vec::new();
        if let Some(dir) = &opts.dir {
            std::fs::create_dir_all(dir).with_context(|| format!("context directory {}", dir.display()))?;
            for e in std::fs::read_dir(dir)? {
                let path = e?.path();
                if path.extension().and_then(|x| x.to_str()) != Some("ctx") {
                    continue;
                }
                match read_header(&path) {
                    Ok((model_id, tokens, bytes)) if model_id == opts.model_id => entries.push(Entry { tokens, bytes, loc: Loc::Disk(path), used: 0 }),
                    Ok(_) => {}
                    Err(e) => eprintln!("ignoring context file {}: {e:#}", path.display()),
                }
            }
        }
        Ok(Self { opts, entries, clock: 0 })
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
        self.clock += 1;
        let bytes = data.len();
        self.entries.push(Entry { tokens, bytes, loc: Loc::Host(data), used: self.clock });
        self.evict()
    }

    /// The tokens and data of entry `i`, read from disk into pinned memory when needed.
    pub fn get(&mut self, i: usize) -> Result<(Vec<i32>, &PinnedBuffer)> {
        self.clock += 1;
        self.entries[i].used = self.clock;
        if let Loc::Disk(path) = &self.entries[i].loc {
            let data = read_data(path, self.entries[i].bytes)?;
            self.entries[i].loc = Loc::Host(data);
        }
        let e = &self.entries[i];
        match &e.loc {
            Loc::Host(buf) => Ok((e.tokens.clone(), buf)),
            Loc::Disk(_) => unreachable!(),
        }
    }

    fn host_bytes(&self) -> usize {
        self.entries.iter().filter(|e| matches!(e.loc, Loc::Host(_))).map(|e| e.bytes).sum()
    }

    /// Moves least recently used host entries to disk (or drops them without a disk tier) until the
    /// host tier fits its budget.
    fn evict(&mut self) -> Result<()> {
        while self.host_bytes() > self.opts.host_bytes {
            let Some(i) = self.entries.iter().enumerate().filter(|(_, e)| matches!(e.loc, Loc::Host(_))).min_by_key(|(_, e)| e.used).map(|(i, _)| i) else { break };
            match &self.opts.dir {
                Some(dir) => {
                    let path = dir.join(format!("{:016x}.ctx", hash_tokens(&self.entries[i].tokens)));
                    if let Loc::Host(buf) = &self.entries[i].loc {
                        write_file(&path, &self.opts.model_id, &self.entries[i].tokens, buf.as_slice())?;
                    }
                    self.entries[i].loc = Loc::Disk(path);
                }
                None => {
                    self.entries.remove(i);
                }
            }
        }
        Ok(())
    }

    /// Writes every host entry to the disk tier (for persistence at shutdown).
    pub fn persist(&mut self) -> Result<()> {
        let Some(dir) = self.opts.dir.clone() else { return Ok(()) };
        for e in &mut self.entries {
            if let Loc::Host(buf) = &e.loc {
                let path = dir.join(format!("{:016x}.ctx", hash_tokens(&e.tokens)));
                if !path.exists() {
                    write_file(&path, &self.opts.model_id, &e.tokens, buf.as_slice())?;
                }
            }
        }
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
fn write_file(path: &Path, model_id: &str, tokens: &[i32], data: &[u8]) -> Result<()> {
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
        f.flush()?;
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
