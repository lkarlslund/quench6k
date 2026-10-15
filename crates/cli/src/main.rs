use std::path::PathBuf;

use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use squash_engine::checkpoint::Checkpoint;
use squash_engine::loader::{self, LoadOptions};
use squash_engine::oracle;
use squash_engine::selection;

#[derive(Parser)]
#[command(name = "squash", about = "Qwen3.8 Flash-Next inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Upload the checkpoint to the GPU, report throughput, and optionally verify tensors.
    Load {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        mtp: bool,
        #[arg(long)]
        vision: bool,
        #[arg(long, default_value_t = 8)]
        threads: usize,
        #[arg(long, default_value_t = 32)]
        chunk_mib: usize,
        /// Read through the page cache instead of O_DIRECT.
        #[arg(long)]
        buffered: bool,
        /// Verify this many tensors (spread across the checkpoint) against the file bytes.
        #[arg(long, default_value_t = 0)]
        verify: usize,
    },
    /// Score fixture sequences teacher-forced and write oracle distributions.
    Score {
        #[arg(long)]
        model: PathBuf,
        /// Fixture directories (each with sequence.json).
        #[arg(long, num_args = 1.., required = true)]
        fixture: Vec<PathBuf>,
        /// Output directory; writes `<fixture><suffix>.{json,bin}`.
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "")]
        suffix: String,
        /// Tokens per forward step.
        #[arg(long, default_value_t = 1024)]
        chunk: usize,
    },
    /// Apply the correctness gate (GOAL.md §1) to teacher-forced candidate distributions.
    Gate {
        /// Fixture root containing one directory per fixture.
        #[arg(long, default_value = "oracle/fixtures")]
        fixtures: PathBuf,
        /// Directory holding `<fixture>.json`/`<fixture>.bin` candidate files, or `<dir>/<fixture>-<suffix>`.
        #[arg(long)]
        candidates: PathBuf,
        /// Candidate file suffix: files are named `<fixture><suffix>.{json,bin}`.
        #[arg(long, default_value = "")]
        suffix: String,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Load { model, mtp, vision, threads, chunk_mib, buffered, verify } => {
            squash_cuda::set_device(0)?;
            let t = std::time::Instant::now();
            let ckpt = Checkpoint::open(&model)?;
            println!("indexed {} tensors in {} files in {:.2}s", ckpt.tensors.len(), ckpt.files.len(), t.elapsed().as_secs_f64());

            let select = |name: &str| selection::text(name) || (mtp && selection::mtp(name)) || (vision && selection::vision(name));
            let opts = LoadOptions { threads, chunk_bytes: chunk_mib << 20, direct: !buffered };
            let (weights, stats) = loader::upload(&ckpt, select, opts)?;
            let gib = stats.bytes as f64 / (1u64 << 30) as f64;
            println!(
                "uploaded {} tensors, {:.2} GiB in {} segments, {:.2}s, {:.2} GiB/s",
                weights.tensors.len(),
                gib,
                stats.segments,
                stats.seconds,
                gib / stats.seconds
            );
            let misaligned = weights.tensors.values().filter(|t| t.bytes >= 16 && t.addr % 16 != 0).count();
            println!("tensors >= 16 bytes not 16-byte aligned: {misaligned}");
            let (free, total) = squash_cuda::mem_info()?;
            println!("device memory: {:.2} GiB free of {:.2} GiB", free as f64 / 1073741824.0, total as f64 / 1073741824.0);

            if verify > 0 {
                let mut names: Vec<&String> = weights.tensors.keys().collect();
                names.sort();
                let stream = squash_cuda::Stream::new()?;
                let step = (names.len() / verify).max(1);
                let mut checked = 0;
                for name in names.iter().step_by(step) {
                    let tensor = weights.get(name)?;
                    let host = squash_cuda::host_checksum(&ckpt.read(name)?);
                    let device = squash_cuda::device_checksum(tensor.addr, tensor.bytes as usize, &stream)?;
                    ensure!(host == device, "checksum mismatch for {name}: host {host:#x}, device {device:#x}");
                    checked += 1;
                }
                println!("verified {checked} tensors byte-exact");
            }
        }
        Command::Score { model, fixture, out, suffix, chunk } => score(&model, &fixture, &out, &suffix, chunk)?,
        Command::Gate { fixtures, candidates, suffix } => {
            let mut all = true;
            let mut any = false;
            for dir in oracle::fixtures(&fixtures)? {
                let name = dir.file_name().unwrap().to_string_lossy().into_owned();
                let stem = candidates.join(format!("{name}{suffix}"));
                if !stem.with_extension("json").exists() {
                    println!("{name:14} (no candidate)");
                    continue;
                }
                any = true;
                let verdict = oracle::gate(&dir, &oracle::Distributions::load(&stem)?)?;
                for r in &verdict.regions {
                    println!(
                        "{name:14} {:10} n={:4} {}  |dlogp| mean {:.4} p99 {:.4} (bound {:.4} / {:.4}; floor {:.4} / {:.4}; vs B {:.4} / {:.4})  top1 agree {:.3}",
                        r.region,
                        r.positions,
                        if r.pass { "PASS" } else { "FAIL" },
                        r.vs_a.mean,
                        r.vs_a.p99,
                        r.bound_mean,
                        r.bound_p99,
                        r.floor.mean,
                        r.floor.p99,
                        r.vs_b.mean,
                        r.vs_b.p99,
                        r.top1_agree_a
                    );
                }
                all &= verdict.pass;
            }
            ensure!(any, "no candidate files found in {}", candidates.display());
            println!("gate: {}", if all { "PASS" } else { "FAIL" });
            if !all {
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

fn score(model_dir: &std::path::Path, fixtures: &[PathBuf], out: &std::path::Path, suffix: &str, chunk: usize) -> Result<()> {
    use squash_engine::forward::{self, HeadWorkspace, SequenceState, Workspace};
    use squash_engine::model::Model;

    #[derive(serde::Deserialize)]
    struct Sequence {
        tokens: Vec<i32>,
        prompt_len: usize,
    }
    const TOP_K: usize = 32;

    squash_cuda::set_device(0)?;
    let ckpt = Checkpoint::open(model_dir)?;
    let (weights, stats) = loader::upload(&ckpt, selection::text, LoadOptions::default())?;
    println!("loaded {:.2} GiB in {:.1}s", stats.bytes as f64 / 1073741824.0, stats.seconds);
    let stream = squash_cuda::Stream::new()?;
    let model = Model::bind(&ckpt, weights, &stream)?;
    std::fs::create_dir_all(out)?;

    let seqs: Vec<(PathBuf, Sequence)> = fixtures
        .iter()
        .map(|f| Ok((f.clone(), serde_json::from_slice(&std::fs::read(f.join("sequence.json"))?)?)))
        .collect::<Result<_>>()?;
    let capacity = seqs.iter().map(|(_, s)| s.tokens.len()).max().unwrap_or(0);
    let ws = Workspace::new(chunk, capacity)?;
    let hw = HeadWorkspace::new(256, TOP_K)?;
    println!("workspace {:.2} GiB", ws.bytes() as f64 / 1073741824.0);

    for (dir, seq) in &seqs {
        let t0 = std::time::Instant::now();
        let mut state = SequenceState::new(seq.tokens.len())?;
        let positions = seq.tokens.len() - 1;
        let mut d = oracle::Distributions {
            meta: oracle::DistributionMeta {
                engine: "squash".into(),
                format: "oracle.distributions/1".into(),
                positions,
                prompt_len: seq.prompt_len,
                top_k: TOP_K,
                extra: Default::default(),
            },
            lse: vec![],
            target: vec![],
            top_ids: vec![],
            top_logits: vec![],
        };
        for c0 in (0..seq.tokens.len()).step_by(chunk) {
            let c1 = (c0 + chunk).min(seq.tokens.len());
            forward::forward(&model, &mut state, &ws, &stream, &seq.tokens[c0..c1])?;
            let rows = c1.min(positions) - c0.min(positions);
            if rows > 0 {
                let h = forward::head_stats(&model, &hw, &stream, ws.out, &seq.tokens[c0 + 1..c0 + 1 + rows], TOP_K)?;
                d.lse.extend(h.lse);
                d.target.extend(h.target);
                d.top_ids.extend(h.top_ids);
                d.top_logits.extend(h.top_logits);
            }
        }
        stream.synchronize()?;
        let secs = t0.elapsed().as_secs_f64();
        d.meta.extra.insert("seconds".into(), secs.into());
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        d.save(&out.join(format!("{name}{suffix}")))?;
        println!("{name}: {} tokens in {secs:.2}s ({:.0} tok/s)", seq.tokens.len(), seq.tokens.len() as f64 / secs);
    }
    Ok(())
}
