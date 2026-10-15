mod bench;
mod mtp;

use std::path::PathBuf;

use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use squash_engine::checkpoint::Checkpoint;
use squash_engine::loader::{self, LoadOptions};
use squash_engine::oracle;
use squash_engine::model::{Model, Precision};
use squash_engine::selection;

#[derive(Clone, Copy, clap::ValueEnum)]
enum PrecisionArg {
    Bf16,
    Int8g,
}

impl From<PrecisionArg> for Precision {
    fn from(p: PrecisionArg) -> Self {
        match p {
            PrecisionArg::Bf16 => Precision::Bf16,
            PrecisionArg::Int8g => Precision::Int8Group,
        }
    }
}

/// Opens the checkpoint and loads the text model onto GPU 0.
pub fn load(model_dir: &std::path::Path, precision: Precision, mtp: bool) -> Result<(Model, squash_cuda::Stream)> {
    squash_cuda::set_device(0)?;
    let ckpt = Checkpoint::open(model_dir)?;
    let stream = squash_cuda::Stream::new()?;
    let (model, stats) = squash_engine::model::load_text(&ckpt, precision, mtp, LoadOptions::default(), &stream)?;
    let (free, _) = squash_cuda::mem_info()?;
    println!("loaded {precision:?}: {:.2} GiB read in {:.1}s; {:.1} GiB device memory free", stats.bytes as f64 / 1073741824.0, stats.seconds, free as f64 / 1073741824.0);
    Ok((model, stream))
}

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
        /// Non-expert projection precision: bf16 (checkpoint) or int8g (INT8, a scale per 64 weights).
        #[arg(long, value_enum, default_value_t = PrecisionArg::Int8g)]
        precision: PrecisionArg,
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
        /// Run the completion one token per step (decode) after prefilling the prompt.
        #[arg(long)]
        decode: bool,
    },
    /// Greedy generation from each fixture prompt, checked against the fixture sequence (GOAL.md §1).
    Generate {
        #[arg(long)]
        model: PathBuf,
        /// Non-expert projection precision: bf16 (checkpoint) or int8g (INT8, a scale per 64 weights).
        #[arg(long, value_enum, default_value_t = PrecisionArg::Int8g)]
        precision: PrecisionArg,
        #[arg(long, num_args = 1.., required = true)]
        fixture: Vec<PathBuf>,
        /// Reference logit margin at or below which a divergence counts as a near-tie.
        #[arg(long, default_value_t = 0.5)]
        tie: f32,
    },
    /// Measure prefill and greedy decode throughput at the given prompt lengths.
    Bench {
        #[arg(long)]
        model: PathBuf,
        /// Non-expert projection precision: bf16 (checkpoint) or int8g (INT8, a scale per 64 weights).
        #[arg(long, value_enum, default_value_t = PrecisionArg::Int8g)]
        precision: PrecisionArg,
        #[arg(long, num_args = 1.., default_values_t = [512usize])]
        ctx: Vec<usize>,
        #[arg(long, default_value_t = 256)]
        generate: usize,
        #[arg(long, default_value_t = 1024)]
        chunk: usize,
        /// Also print per-op GPU time for a few eager decode steps.
        #[arg(long)]
        profile: bool,
        /// After the timed run, decode this many graph steps inside a cudaProfilerStart/Stop region
        /// (for `ncu --profile-from-start off`).
        #[arg(long, default_value_t = 0)]
        ncu_steps: usize,
    },
    /// Measure MTP draft acceptance on a fixture (drafts are compared with the target's greedy tokens).
    MtpProbe {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_enum, default_value_t = PrecisionArg::Int8g)]
        precision: PrecisionArg,
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long, default_value_t = 200)]
        steps: usize,
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
        Command::Score { model, precision, fixture, out, suffix, chunk, decode } => score(&model, precision.into(), &fixture, &out, &suffix, chunk, decode)?,
        Command::Generate { model, precision, fixture, tie } => generate(&model, precision.into(), &fixture, tie)?,
        Command::MtpProbe { model, precision, fixture, steps, chunk } => mtp::probe(&model, precision.into(), &fixture, steps, chunk)?,
        Command::Bench { model, precision, ctx, generate, chunk, profile, ncu_steps } => bench::run(&model, precision.into(), &ctx, generate, chunk, profile, ncu_steps)?,
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

fn score(model_dir: &std::path::Path, precision: Precision, fixtures: &[PathBuf], out: &std::path::Path, suffix: &str, chunk: usize, decode: bool) -> Result<()> {
    use squash_engine::forward::{self, HeadWorkspace, SequenceState, Workspace};

    #[derive(serde::Deserialize)]
    struct Sequence {
        tokens: Vec<i32>,
        prompt_len: usize,
    }
    const TOP_K: usize = 32;

    let (model, stream) = load(model_dir, precision, false)?;
    std::fs::create_dir_all(out)?;

    let seqs: Vec<(PathBuf, Sequence)> = fixtures
        .iter()
        .map(|f| Ok((f.clone(), serde_json::from_slice(&std::fs::read(f.join("sequence.json"))?)?)))
        .collect::<Result<_>>()?;
    let capacity = seqs.iter().map(|(_, s)| s.tokens.len()).max().unwrap_or(0);
    let mut ws = Workspace::new(chunk, capacity)?;
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
        let mut c0 = 0;
        while c0 < seq.tokens.len() {
            let step = if decode && c0 >= seq.prompt_len { 1 } else { chunk };
            let limit = if decode && c0 < seq.prompt_len { seq.prompt_len } else { seq.tokens.len() };
            let c1 = (c0 + step).min(limit);
            forward::forward(&model, &mut state, &mut ws, &stream, &seq.tokens[c0..c1])?;
            let rows = c1.min(positions) - c0.min(positions);
            if rows > 0 {
                let h = forward::head_stats(&model, &hw, &stream, ws.out, &seq.tokens[c0 + 1..c0 + 1 + rows], TOP_K)?;
                d.lse.extend(h.lse);
                d.target.extend(h.target);
                d.top_ids.extend(h.top_ids);
                d.top_logits.extend(h.top_logits);
            }
            c0 = c1;
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

fn generate(model_dir: &std::path::Path, precision: Precision, fixtures: &[PathBuf], tie: f32) -> Result<()> {
    use squash_engine::forward::{self, HeadWorkspace, SequenceState, Workspace};

    #[derive(serde::Deserialize)]
    struct Sequence {
        tokens: Vec<i32>,
        prompt_len: usize,
    }

    let (model, stream) = load(model_dir, precision, false)?;
    let seqs: Vec<(PathBuf, Sequence)> = fixtures
        .iter()
        .map(|f| Ok((f.clone(), serde_json::from_slice(&std::fs::read(f.join("sequence.json"))?)?)))
        .collect::<Result<_>>()?;
    let capacity = seqs.iter().map(|(_, s)| s.tokens.len()).max().unwrap_or(0);
    let mut ws = Workspace::new(1024, capacity)?;
    let hw = HeadWorkspace::new(1, 1)?;
    let mut all_ok = true;
    for (dir, seq) in &seqs {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let refs = [oracle::Distributions::load(&dir.join("llama-w4a4"))?, oracle::Distributions::load(&dir.join("llama-q8"))?];
        let mut state = SequenceState::new(seq.tokens.len())?;
        let prompt = &seq.tokens[..seq.prompt_len];
        for c in prompt.chunks(1024) {
            forward::forward(&model, &mut state, &mut ws, &stream, c)?;
        }
        let last = ws.out + ((prompt.len() - 1) % 1024 * squash_engine::model::D * 2) as u64;
        let mut next = forward::head_stats(&model, &hw, &stream, last, &[0], 1)?.top_ids[0];
        let t0 = std::time::Instant::now();
        let mut steps = 0;
        let mut verdict = String::from("identical");
        for p in seq.prompt_len..seq.tokens.len() {
            let want = seq.tokens[p];
            if next != want {
                // Margin each reference gives its own choice over ours at position p - 1.
                let margin = refs
                    .iter()
                    .map(|r| {
                        let pos = p - 1;
                        let k = r.meta.top_k;
                        let ids = &r.top_ids[pos * k..pos * k + k];
                        let logits = &r.top_logits[pos * k..pos * k + k];
                        let ours = ids.iter().position(|&i| i == next).map(|j| logits[j]).unwrap_or(f32::NEG_INFINITY);
                        r.target[pos] - ours
                    })
                    .fold(f32::INFINITY, f32::min);
                let near = margin <= tie;
                all_ok &= near;
                verdict = format!("{} at completion token {} (margin {margin:.3})", if near { "near-tie" } else { "DIVERGED" }, p - seq.prompt_len);
                break;
            }
            if p + 1 == seq.tokens.len() {
                break;
            }
            forward::forward(&model, &mut state, &mut ws, &stream, &[want])?;
            steps += 1;
            next = forward::head_stats(&model, &hw, &stream, ws.out, &[0], 1)?.top_ids[0];
        }
        let secs = t0.elapsed().as_secs_f64();
        println!("{name:14} {verdict}; {steps} decode steps, {:.1} tok/s", steps as f64 / secs.max(1e-9));
    }
    println!("generation: {}", if all_ok { "PASS" } else { "FAIL" });
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
