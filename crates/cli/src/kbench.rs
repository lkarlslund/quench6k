//! Fast kernel benchmarking on a truncated model (docs/KBENCH.md): the first K layers with real weights,
//! a real prefilled context, and a decode step of R rows captured as a CUDA graph per kernel variant.
//! Each variant is checked bit for bit against the baseline (hashes of the final hidden state and the
//! residual stream after every replay) and timed against it in alternating pairs.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use quench6k_cuda::{DeviceBuffer, Event, Graph, Stream, variants};
use quench6k_engine::checkpoint::Checkpoint;
use quench6k_engine::forward::{self, Part, SequenceState, Workspace};
use quench6k_engine::loader::LoadOptions;
use quench6k_engine::model::{D, H, Model, Precision};

pub struct Options {
    pub layers: usize,
    pub ctx: usize,
    pub rows: usize,
    pub steps: usize,
    pub pairs: usize,
    pub chunk: usize,
    /// Keys set for every configuration (baseline included).
    pub fixed: Vec<(String, i32)>,
    /// key -> values; the cartesian product is swept.
    pub sweep: Vec<(String, Vec<i32>)>,
}

/// Parses "key=value".
pub fn parse_fixed(s: &str) -> Result<(String, i32)> {
    let (k, v) = s.split_once('=').context("expected key=value")?;
    Ok((k.to_string(), v.parse()?))
}

/// Parses "key=v1,v2,...".
pub fn parse_sweep(s: &str) -> Result<(String, Vec<i32>)> {
    let (k, v) = s.split_once('=').context("expected key=v1,v2,...")?;
    Ok((k.to_string(), v.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?))
}

/// GDN and PLE state of the truncated model, restorable before every replay sequence.
struct Snapshot {
    pieces: Vec<(u64, DeviceBuffer)>,
}

impl Snapshot {
    fn take(state: &SequenceState, layers: usize, s: &Stream) -> Result<Self> {
        let mut src: Vec<(u64, usize)> = Vec::new();
        for g in state.gdn.iter().take(layers).flatten() {
            src.push((g.conv.addr(), g.conv.bytes()));
            src.push((g.recurrent.addr(), g.recurrent.bytes()));
        }
        src.push((state.ple_conv.addr(), state.ple_conv.bytes()));
        let mut pieces = Vec::new();
        for (addr, bytes) in src {
            let b = DeviceBuffer::new(bytes)?;
            s.copy_d2d(b.addr(), addr, bytes)?;
            pieces.push((addr, b));
        }
        s.synchronize()?;
        Ok(Self { pieces })
    }

    fn restore(&self, s: &Stream) -> Result<()> {
        for (addr, b) in &self.pieces {
            s.copy_d2d(*addr, b.addr(), b.bytes())?;
        }
        Ok(())
    }
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

struct Bench<'a> {
    model: &'a Model,
    state: &'a SequenceState,
    ws: &'a Workspace,
    s: &'a Stream,
    snap: Snapshot,
    rows: usize,
    bytes: usize,
    steps: usize,
}

impl Bench<'_> {
    fn capture(&self) -> Result<Graph> {
        Graph::capture(self.s, |s| forward::forward_part(self.model, self.state, self.ws, s, self.rows, self.bytes, Part::Whole, false, None))
    }

    /// Hashes of (final hidden state, residual) after each of `steps` replays from the snapshot.
    fn hashes(&self, g: &Graph) -> Result<Vec<(u64, u64)>> {
        self.snap.restore(self.s)?;
        let mut out = Vec::with_capacity(self.steps);
        for _ in 0..self.steps {
            g.launch(self.s)?;
            let o: Vec<u8> = self.s.download(self.ws.out, self.rows * D * 2)?;
            let h: Vec<u8> = self.s.download(self.ws.h, self.rows * H * 4)?;
            out.push((fnv(&o), fnv(&h)));
        }
        Ok(out)
    }

    /// Microseconds per replay over `steps` replays from the snapshot.
    fn time(&self, g: &Graph) -> Result<f64> {
        self.snap.restore(self.s)?;
        let (a, b) = (Event::timing()?, Event::timing()?);
        a.record(self.s)?;
        for _ in 0..self.steps {
            g.launch(self.s)?;
        }
        b.record(self.s)?;
        b.synchronize()?;
        Ok(Event::elapsed_ms(&a, &b)? as f64 * 1000.0 / self.steps as f64)
    }
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn set_all(keys: &[(String, i32)]) {
    variants::clear();
    for (k, v) in keys {
        variants::set(k, *v);
    }
}

pub fn run(model_dir: &Path, o: &Options) -> Result<()> {
    ensure!(o.rows >= 1 && o.rows <= 16, "rows in 1..=16");
    quench6k_cuda::set_device(0)?;
    let ckpt = Checkpoint::open(model_dir)?;
    let s = Stream::new()?;
    let t0 = std::time::Instant::now();
    let opts = LoadOptions { layers: o.layers, direct: false, ..Default::default() };
    let (model, stats) = quench6k_engine::model::load_text(&ckpt, Precision::Int8Group, false, opts, &s)?;
    println!("loaded {} layers: {:.2} GiB in {:.1}s", o.layers, stats.bytes as f64 / 1073741824.0, t0.elapsed().as_secs_f64());

    // Prefill the context, then prepare one decode step of `rows` tokens (the corpus continuation).
    let ids = crate::bench::prompt(o.ctx + o.rows)?;
    let mut state = SequenceState::new(o.ctx + o.rows + 16)?;
    let mut ws = Workspace::for_model(&model, o.chunk.max(o.rows), o.ctx + o.rows + 16)?;
    let t1 = std::time::Instant::now();
    for c in ids[..o.ctx].chunks(o.chunk) {
        forward::forward(&model, &mut state, &mut ws, &s, c)?;
    }
    println!("prefilled {} tokens in {:.1}s", o.ctx, t1.elapsed().as_secs_f64());
    forward::prepare_tokens(&mut state, &mut ws, &ids[o.ctx..])?;
    let bytes = forward::prepare_ple(&model, &state, &mut ws, o.rows)?;
    let snap = Snapshot::take(&state, o.layers, &s)?;
    let b = Bench { model: &model, state: &state, ws: &ws, s: &s, snap, rows: o.rows, bytes, steps: o.steps };

    set_all(&o.fixed);
    let base = b.capture()?;
    let base_hashes = b.hashes(&base)?;
    ensure!(base_hashes == b.hashes(&base)?, "the baseline is not deterministic");
    let base_us = {
        let mut v: Vec<f64> = (0..o.pairs).map(|_| b.time(&base)).collect::<Result<_>>()?;
        quantile(&mut v, 0.5)
    };
    println!("baseline: {} layers, {} row(s), context {}: {:.1} us per step ({:.1} us per layer)", o.layers, o.rows, o.ctx, base_us, base_us / o.layers as f64);
    // Bandwidth floor of the step: static weights plus 10 routed experts per layer and row (2.76 MB
    // each in NVFP4, an upper bound when rows share experts), at the measured 1.647 TB/s read rate.
    let static_bytes: u64 = ws.layer_weight_bytes.iter().sum();
    let expert_bytes = (o.layers * o.rows.min(4) * 10) as f64 * 3.0 * 640.0 * 2560.0 * 0.5625;
    let floor = (static_bytes as f64 + expert_bytes) / 1.647e6;
    println!("weights per step: {:.1} MB static ({:?} MB per layer) + {:.1} MB experts; floor {:.1} us ({:.0}% of it reached)",
             static_bytes as f64 / 1e6, ws.layer_weight_bytes.iter().map(|b| b / 1_000_000).collect::<Vec<_>>(), expert_bytes / 1e6, floor, 100.0 * floor / base_us);

    // Cartesian product of the sweep values.
    let mut configs: Vec<Vec<(String, i32)>> = vec![Vec::new()];
    for (k, vals) in &o.sweep {
        configs = configs.into_iter().flat_map(|c| vals.iter().map(move |&v| [c.clone(), vec![(k.clone(), v)]].concat())).collect();
    }
    if o.sweep.is_empty() {
        return Ok(());
    }
    println!("{:<44} {:>9} {:>17} {:>8}", "variant", "ratio", "25%..75%", "exact");
    let mut best: Option<(f64, String)> = None;
    for c in &configs {
        let name = c.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ");
        let mut keys = o.fixed.clone();
        keys.extend(c.iter().cloned());
        set_all(&keys);
        let g = match b.capture() {
            Ok(g) => g,
            Err(e) => {
                println!("{name:<44} {:>9}", format!("rejected: {e:#}"));
                continue;
            }
        };
        set_all(&o.fixed);
        let exact = b.hashes(&g)? == base_hashes;
        // Alternating pairs (A B, B A, ...): drift in clocks and temperature cancels in the ratio.
        let mut ratios = Vec::with_capacity(o.pairs);
        for p in 0..o.pairs {
            let (x, y) = if p % 2 == 0 { (b.time(&base)?, b.time(&g)?) } else {
                let y = b.time(&g)?;
                (b.time(&base)?, y)
            };
            ratios.push(y / x);
        }
        let (lo, mid, hi) = (quantile(&mut ratios, 0.25), quantile(&mut ratios, 0.5), quantile(&mut ratios, 0.75));
        println!("{name:<44} {mid:>9.4} {:>17} {:>8}", format!("{lo:.4}..{hi:.4}"), if exact { "yes" } else { "NO" });
        if exact && best.as_ref().is_none_or(|(r, _)| mid < *r) {
            best = Some((mid, name));
        }
    }
    variants::clear();
    if let Some((r, name)) = best {
        println!("best exact: {name} ({:+.2}% time)", (r - 1.0) * 100.0);
    }
    Ok(())
}
