//! Oracle distributions (`oracle.distributions/1`) and the correctness gate of GOAL.md §1.
//!
//! A distribution file pair `<stem>.json` + `<stem>.bin` describes, for each of `positions` P
//! teacher-forced positions, the log-sum-exp of the logits, the logit of the sequence's next token,
//! and the top-K token IDs and logits. Position `p` predicts token `p + 1` of the fixture sequence.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DistributionMeta {
    pub engine: String,
    pub format: String,
    pub positions: usize,
    pub prompt_len: usize,
    pub top_k: usize,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

pub struct Distributions {
    pub meta: DistributionMeta,
    pub lse: Vec<f32>,
    pub target: Vec<f32>,
    pub top_ids: Vec<i32>,
    pub top_logits: Vec<f32>,
}

impl Distributions {
    pub fn load(stem: &Path) -> Result<Self> {
        let meta: DistributionMeta = serde_json::from_slice(&std::fs::read(stem.with_extension("json")).with_context(|| format!("reading {}.json", stem.display()))?)?;
        ensure!(meta.format == "oracle.distributions/1", "{}: unsupported format {}", stem.display(), meta.format);
        let raw = std::fs::read(stem.with_extension("bin")).with_context(|| format!("reading {}.bin", stem.display()))?;
        let (p, k) = (meta.positions, meta.top_k);
        ensure!(raw.len() == 4 * (2 * p + 2 * p * k), "{}.bin has {} bytes, expected {}", stem.display(), raw.len(), 4 * (2 * p + 2 * p * k));
        let words: Vec<[u8; 4]> = raw.chunks_exact(4).map(|c| c.try_into().unwrap()).collect();
        let f = |r: std::ops::Range<usize>| words[r].iter().map(|w| f32::from_le_bytes(*w)).collect::<Vec<_>>();
        let top_ids = words[2 * p..2 * p + p * k].iter().map(|w| i32::from_le_bytes(*w)).collect();
        Ok(Self { lse: f(0..p), target: f(p..2 * p), top_ids, top_logits: f(2 * p + p * k..2 * p + 2 * p * k), meta })
    }

    pub fn save(&self, stem: &Path) -> Result<()> {
        let mut bytes = Vec::with_capacity(4 * (self.lse.len() * 2 + self.top_ids.len() * 2));
        for v in self.lse.iter().chain(&self.target) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.top_ids {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.top_logits {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(stem.with_extension("bin"), bytes)?;
        std::fs::write(stem.with_extension("json"), serde_json::to_string_pretty(&self.meta)? + "\n")?;
        Ok(())
    }

    /// Log-probability of the sequence token at each position.
    pub fn logprob(&self) -> Vec<f64> {
        self.target.iter().zip(&self.lse).map(|(t, l)| (*t as f64) - (*l as f64)).collect()
    }

    pub fn top1(&self, position: usize) -> i32 {
        self.top_ids[position * self.meta.top_k]
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Spread {
    pub mean: f64,
    pub p99: f64,
    pub max: f64,
}

/// Distribution of |a - b| over paired values.
pub fn spread(a: &[f64], b: &[f64]) -> Spread {
    let mut d: Vec<f64> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
    if d.is_empty() {
        return Spread::default();
    }
    d.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mean = d.iter().sum::<f64>() / d.len() as f64;
    // Linear interpolation between order statistics, as numpy's default percentile.
    let rank = 0.99 * (d.len() - 1) as f64;
    let (lo, hi) = (rank.floor() as usize, rank.ceil() as usize);
    let p99 = d[lo] + (d[hi] - d[lo]) * (rank - lo as f64);
    Spread { mean, p99, max: *d.last().unwrap() }
}

#[derive(Debug, Serialize)]
pub struct RegionVerdict {
    pub region: &'static str,
    pub positions: usize,
    pub floor: Spread,
    pub vs_a: Spread,
    pub vs_b: Spread,
    pub bound_mean: f64,
    pub bound_p99: f64,
    pub top1_agree_a: f64,
    pub pass: bool,
}

#[derive(Debug, Serialize)]
pub struct FixtureVerdict {
    pub fixture: String,
    pub regions: Vec<RegionVerdict>,
    pub pass: bool,
}

/// Applies the GOAL.md gate to a teacher-forced candidate against fixture references A and B.
pub fn gate(fixture: &Path, candidate: &Distributions) -> Result<FixtureVerdict> {
    let a = Distributions::load(&fixture.join("llama-w4a4"))?;
    let b = Distributions::load(&fixture.join("llama-q8"))?;
    let p = a.meta.positions;
    ensure!(b.meta.positions == p && candidate.meta.positions == p, "position counts differ: A {p}, B {}, candidate {}", b.meta.positions, candidate.meta.positions);
    let (la, lb, lc) = (a.logprob(), b.logprob(), candidate.logprob());
    let split = a.meta.prompt_len.saturating_sub(1).min(p);
    let mut regions = Vec::new();
    for (region, range) in [("prompt", 0..split), ("completion", split..p)] {
        if range.is_empty() {
            continue;
        }
        let r = range.clone();
        let floor = spread(&la[r.clone()], &lb[r.clone()]);
        let vs_a = spread(&lc[r.clone()], &la[r.clone()]);
        let vs_b = spread(&lc[r.clone()], &lb[r.clone()]);
        let bound_mean = (2.0 * floor.mean).max(0.01);
        let bound_p99 = (2.0 * floor.p99).max(0.05);
        let pass = vs_a.mean.min(vs_b.mean) <= bound_mean && vs_a.p99.min(vs_b.p99) <= bound_p99;
        let agree = r.clone().filter(|&i| candidate.top1(i) == a.top1(i)).count() as f64 / r.len() as f64;
        regions.push(RegionVerdict { region, positions: r.len(), floor, vs_a, vs_b, bound_mean, bound_p99, top1_agree_a: agree, pass });
    }
    let pass = regions.iter().all(|r| r.pass);
    Ok(FixtureVerdict { fixture: fixture.file_name().unwrap().to_string_lossy().into_owned(), regions, pass })
}

/// The fixture directories under `root` (each with a `sequence.json`).
pub fn fixtures(root: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.join("sequence.json").exists()).collect();
    dirs.sort();
    Ok(dirs)
}
