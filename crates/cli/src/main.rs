mod gpumon;
mod bench;
mod kbench;
mod chat;
mod mtp;
mod serve;

use std::path::PathBuf;

use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use quench6k_engine::checkpoint::Checkpoint;
use quench6k_engine::loader::{self, LoadOptions};
use quench6k_engine::oracle;
use quench6k_engine::model::{Model, Precision};
use quench6k_engine::selection;

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
/// Direction-removal steering (a `ninfer.steering/1` pack) for scoring and generation.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct SteerArgs {
    /// Steering pack to load.
    #[arg(long)]
    steering: Option<PathBuf>,
    /// Steering strength (default: the pack's default_steering_strength).
    #[arg(long)]
    strength: Option<f32>,
}

impl SteerArgs {
    /// Loads the pack into the model; returns the strength for new sequences.
    pub fn attach(&self, model: &mut Model, stream: &quench6k_cuda::Stream) -> Result<f32> {
        let Some(path) = &self.steering else { return Ok(0.0) };
        let pack = quench6k_engine::steering::Steering::load(path, stream)?;
        let strength = self.strength.unwrap_or(pack.default_strength);
        println!(
            "steering pack {}: layers {:?}, norm_preserve {}, strength {strength} (pack model_sha {})",
            path.display(),
            pack.layers.iter().map(|l| l.0).collect::<Vec<_>>(),
            pack.norm_preserve,
            pack.model_sha.as_deref().unwrap_or("-")
        );
        model.steering = Some(pack);
        Ok(strength)
    }
}

pub fn load(model_dir: &std::path::Path, precision: Precision, mtp: bool) -> Result<(Model, quench6k_cuda::Stream)> {
    quench6k_cuda::set_device(0)?;
    let ckpt = Checkpoint::open(model_dir)?;
    let stream = quench6k_cuda::Stream::new()?;
    let (model, stats) = quench6k_engine::model::load_text(&ckpt, precision, mtp, LoadOptions::default(), &stream)?;
    let (free, _) = quench6k_cuda::mem_info()?;
    println!("loaded {precision:?}: {:.2} GiB read in {:.1}s; {:.1} GiB device memory free", stats.bytes as f64 / 1073741824.0, stats.seconds, free as f64 / 1073741824.0);
    Ok((model, stream))
}

#[derive(Parser)]
#[command(name = "quench6k", about = "Qwen3.8 Flash-Next inference engine")]
struct Cli {
    /// KV cache format for attention layers: bf16, or fp8 (E4M3 with a per-position, per-head scale;
    /// about half the memory).
    #[arg(long, global = true, default_value = "bf16", value_parser = ["bf16", "fp8"])]
    kv: String,
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
        /// With --decode: decode each fixture in a batch of two, next to the following fixture
        /// (teacher-forced with its own tokens), through the batched forward.
        #[arg(long, requires = "decode")]
        batch2: bool,
        #[command(flatten)]
        steer: SteerArgs,
    },
    /// Greedy generation from each fixture prompt, checked against the fixture sequence (dev/GOAL.md §1).
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
        #[command(flatten)]
        steer: SteerArgs,
    },
    /// One chat turn from the command line through the serving session (tokenizer, chat template,
    /// greedy MTP or sampled decoding), streamed to stdout.
    Chat {
        #[arg(long)]
        model: PathBuf,
        /// The user message.
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        /// 0: greedy (speculative).
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        #[arg(long, default_value_t = 20)]
        top_k: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Let the model think (the template's enable_thinking).
        #[arg(long)]
        think: bool,
        #[arg(long, default_value_t = 32768)]
        context: usize,
        /// Images to show the model (placed before the prompt text).
        #[arg(long)]
        image: Vec<PathBuf>,
        /// Videos to show the model (after the images; decoded with ffmpeg).
        #[arg(long)]
        video: Vec<PathBuf>,
        #[command(flatten)]
        steer: SteerArgs,
    },
    /// HTTP server: OpenAI (/v1/chat/completions, /v1/models) and Anthropic (/v1/messages) APIs.
    Serve {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: std::net::SocketAddr,
        /// Longest prompt + completion in tokens.
        #[arg(long, default_value_t = 65536)]
        context: usize,
        /// Longest speculative draft (adaptive 1..=N).
        #[arg(long, default_value_t = 6)]
        drafts: usize,
        /// Model name reported by /v1/models.
        #[arg(long, default_value = "quench6k")]
        name: String,
        #[command(flatten)]
        steer: SteerArgs,
        /// With --steering: name of the model that applies the pack (default "<name>-steered"); the base
        /// name is served unsteered. Its strength is --strength (default 1).
        #[arg(long)]
        steered_name: Option<String>,
        /// Pinned host memory for stored conversation contexts, in GiB (0: none).
        #[arg(long, default_value_t = 16.0)]
        cache_host_gib: f64,
        /// Directory for the disk tier of stored contexts, kept across restarts.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Chat template file overriding the model's chat_template.jinja.
        #[arg(long)]
        chat_template: Option<PathBuf>,
        /// Regenerations of a tool call that fails validation (parse, declared function, required
        /// parameters, schema types) before the output is returned as is.
        #[arg(long, default_value_t = 2)]
        tool_retries: usize,
        /// Completion limit for requests that do not set max_tokens (capped by the context).
        #[arg(long, default_value_t = 65536)]
        default_max_tokens: usize,
        /// Concurrent requests decoded together (1 or 2).
        #[arg(long, default_value_t = 2)]
        slots: usize,
        /// Directory for activation captures requested per request ("capture": {"name": ...}).
        #[arg(long)]
        capture_dir: Option<PathBuf>,
        /// Do not load the vision tower (requests with images are rejected).
        #[arg(long)]
        no_vision: bool,
        /// Largest image area after resizing, in pixels (the checkpoint's processor allows 16777216;
        /// each 32x32 block becomes one token).
        #[arg(long, default_value_t = 4194304)]
        image_max_pixels: usize,
        /// Most frames sampled from a video (2 per second; the processor allows 768).
        #[arg(long, default_value_t = 128)]
        video_max_frames: usize,
    },
    /// Batched decode check: two fixture prompts decoded together must give each one's single-sequence
    /// greedy tokens, bit for bit; also times both.
    BatchCheck {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, num_args = 2, required = true)]
        fixture: Vec<PathBuf>,
        #[arg(long, default_value_t = 64)]
        n: usize,
    },
    /// Vision tower check: encodes reference pixels (tools/ref/vision_ref.py output) and compares the
    /// merged embeddings with the HF reference.
    VisionCheck {
        #[arg(long)]
        model: PathBuf,
        /// Directory with pixels.f32, embeds.f32 and meta.json.
        #[arg(long)]
        reference: PathBuf,
        /// max_pixels the reference was preprocessed with (to compare quench6k's own preprocessing).
        #[arg(long, default_value_t = 16777216)]
        max_pixels: usize,
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
        /// Prefill chunk (tokens per forward pass).
        #[arg(long, default_value_t = 8192)]
        chunk: usize,
        /// Also print per-op GPU time for a few eager decode steps.
        #[arg(long)]
        profile: bool,
        /// After the timed run, decode this many graph steps inside a cudaProfilerStart/Stop region
        /// (for `ncu --profile-from-start off`).
        #[arg(long, default_value_t = 0)]
        ncu_steps: usize,
        /// MTP speculative decoding with this many drafts per round (0: plain decoding).
        #[arg(long, default_value_t = 0)]
        mtp: usize,
        /// With --mtp N: choose each round's draft length in 1..=N from the observed acceptance.
        #[arg(long)]
        adaptive: bool,
        /// With --mtp N: sample with the server's defaults (temperature 1.0, top_p 0.95, top_k 20)
        /// instead of greedy decoding.
        #[arg(long)]
        sampled: bool,
        /// Append one JSON record per context to this file (JSON lines; see bench/README.md).
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Kernel variant benchmark on a truncated model (dev/KBENCH.md): bitwise check and paired timing
    /// of each configuration against the baseline.
    Kbench {
        #[arg(long)]
        model: PathBuf,
        /// Text layers to load (one period of the layer pattern is 4).
        #[arg(long, default_value_t = 4)]
        layers: usize,
        #[arg(long, default_value_t = 8192)]
        ctx: usize,
        /// Rows per decode step (1: plain decoding; 4: an MTP verify round).
        #[arg(long, default_value_t = 1)]
        rows: usize,
        /// Replays per measurement (and per bitwise check).
        #[arg(long, default_value_t = 16)]
        steps: usize,
        /// Timed baseline / variant pairs per configuration.
        #[arg(long, default_value_t = 9)]
        pairs: usize,
        #[arg(long, default_value_t = 8192)]
        chunk: usize,
        /// key=value set for every configuration, baseline included.
        #[arg(long)]
        variant: Vec<String>,
        /// key=v1,v2,...: swept (cartesian product over all --sweep keys).
        #[arg(long)]
        sweep: Vec<String>,
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
    /// Check that MTP speculative decoding produces exactly the plain greedy tokens.
    SpecCheck {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_enum, default_value_t = PrecisionArg::Int8g)]
        precision: PrecisionArg,
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long, default_value_t = 128)]
        n: usize,
        #[arg(long, default_value_t = 3)]
        drafts: usize,
        #[arg(long, default_value_t = 1024)]
        chunk: usize,
        /// Run the speculative rounds as CUDA graphs.
        #[arg(long)]
        graphs: bool,
        /// With --graphs: adaptive draft length in 1..=drafts.
        #[arg(long)]
        adaptive: bool,
        #[command(flatten)]
        steer: SteerArgs,
        /// With --graphs, run this many more rounds inside a cudaProfilerStart/Stop region.
        #[arg(long, default_value_t = 0)]
        profile_rounds: usize,
    },
    /// Apply the correctness gate (dev/GOAL.md §1) to teacher-forced candidate distributions.
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
    let cli = Cli::parse();
    quench6k_engine::forward::KV_FP8.store(cli.kv == "fp8", std::sync::atomic::Ordering::Relaxed);
    match cli.command {
        Command::Load { model, mtp, vision, threads, chunk_mib, buffered, verify } => {
            quench6k_cuda::set_device(0)?;
            let t = std::time::Instant::now();
            let ckpt = Checkpoint::open(&model)?;
            println!("indexed {} tensors in {} files in {:.2}s", ckpt.tensors.len(), ckpt.files.len(), t.elapsed().as_secs_f64());

            let select = |name: &str| selection::text(name) || (mtp && selection::mtp(name)) || (vision && selection::vision(name));
            let opts = LoadOptions { threads, chunk_bytes: chunk_mib << 20, direct: !buffered, ..Default::default() };
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
            let (free, total) = quench6k_cuda::mem_info()?;
            println!("device memory: {:.2} GiB free of {:.2} GiB", free as f64 / 1073741824.0, total as f64 / 1073741824.0);

            if verify > 0 {
                let mut names: Vec<&String> = weights.tensors.keys().collect();
                names.sort();
                let stream = quench6k_cuda::Stream::new()?;
                let step = (names.len() / verify).max(1);
                let mut checked = 0;
                for name in names.iter().step_by(step) {
                    let tensor = weights.get(name)?;
                    let host = quench6k_cuda::host_checksum(&ckpt.read(name)?);
                    let device = quench6k_cuda::device_checksum(tensor.addr, tensor.bytes as usize, &stream)?;
                    ensure!(host == device, "checksum mismatch for {name}: host {host:#x}, device {device:#x}");
                    checked += 1;
                }
                println!("verified {checked} tensors byte-exact");
            }
        }
        Command::VisionCheck { model, reference, max_pixels } => vision_check(&model, &reference, max_pixels)?,
        Command::Score { model, precision, fixture, out, suffix, chunk, decode, batch2, steer } => score(&model, precision.into(), &fixture, &out, &suffix, chunk, decode, batch2, &steer)?,
        Command::Serve { model, listen, context, drafts, name, steer, steered_name, cache_host_gib, cache_dir, chat_template, tool_retries, default_max_tokens, slots, capture_dir, no_vision, image_max_pixels, video_max_frames } => {
            serve::run(serve::ServeOptions { model, listen, context, drafts, name, steer, steered_name, cache_host_gib, cache_dir, chat_template, tool_retries, default_max_tokens, slots, capture_dir, vision: !no_vision, image_max_pixels, video_max_frames })?
        }
        Command::BatchCheck { model, fixture, n } => batch_check(&model, &fixture, n)?,
        Command::Chat { model, prompt, max_tokens, temperature, top_p, top_k, seed, think, context, image, video, steer } => {
            chat_once(&model, &prompt, max_tokens, temperature, top_p, top_k, seed, think, context, &image, &video, &steer)?
        }
        Command::Generate { model, precision, fixture, tie, steer } => generate(&model, precision.into(), &fixture, tie, &steer)?,
        Command::MtpProbe { model, precision, fixture, steps, chunk } => mtp::probe(&model, precision.into(), &fixture, steps, chunk)?,
        Command::SpecCheck { model, precision, fixture, n, drafts, chunk, graphs, adaptive, profile_rounds, steer } => mtp::check(&model, precision.into(), &fixture, n, drafts, chunk, graphs, adaptive, profile_rounds, &steer)?,
        Command::Bench { model, precision, ctx, generate, chunk, profile, ncu_steps, mtp, adaptive, sampled, json } => {
            let out = json.map(|path| bench::JsonOut { path, kv: cli.kv.clone(), sampled });
            bench::run(&model, precision.into(), &ctx, generate, chunk, profile, ncu_steps, mtp, adaptive, sampled, out.as_ref())?
        }
        Command::Kbench { model, layers, ctx, rows, steps, pairs, chunk, variant, sweep } => {
            let fixed = variant.iter().map(|v| kbench::parse_fixed(v)).collect::<Result<_>>()?;
            let sweep = sweep.iter().map(|v| kbench::parse_sweep(v)).collect::<Result<_>>()?;
            kbench::run(&model, &kbench::Options { layers, ctx, rows, steps, pairs, chunk, fixed, sweep })?
        }
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

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn score(model_dir: &std::path::Path, precision: Precision, fixtures: &[PathBuf], out: &std::path::Path, suffix: &str, chunk: usize, decode: bool, batch2: bool, steer: &SteerArgs) -> Result<()> {
    use quench6k_engine::forward::{self, HeadWorkspace, SequenceState, Workspace};

    #[derive(serde::Deserialize)]
    struct Sequence {
        tokens: Vec<i32>,
        prompt_len: usize,
    }
    const TOP_K: usize = 32;

    let (mut model, stream) = load(model_dir, precision, false)?;
    let strength = steer.attach(&mut model, &stream)?;
    std::fs::create_dir_all(out)?;

    let seqs: Vec<(PathBuf, Sequence)> = fixtures
        .iter()
        .map(|f| Ok((f.clone(), serde_json::from_slice(&std::fs::read(f.join("sequence.json"))?)?)))
        .collect::<Result<_>>()?;
    let longest = seqs.iter().map(|(_, s)| s.tokens.len()).max().unwrap_or(0);
    let capacity = if batch2 { 2 * longest } else { longest };
    let mut ws = Workspace::for_model(&model, chunk, capacity)?;
    let hw = HeadWorkspace::new(256, TOP_K)?;
    println!("workspace {:.2} GiB", ws.bytes() as f64 / 1073741824.0);

    for (i, (dir, seq)) in seqs.iter().enumerate() {
        let t0 = std::time::Instant::now();
        // Batch-2 partner: the next fixture's prompt, then its own tokens (cyclically) as inputs.
        let mut partner = None;
        if batch2 {
            let other = &seqs[(i + 1) % seqs.len()].1;
            let mut st = SequenceState::new(other.prompt_len + seq.tokens.len())?;
            st.set_steering_strength(&stream, strength)?;
            for c in other.tokens[..other.prompt_len].chunks(chunk) {
                forward::forward(&model, &mut st, &mut ws, &stream, c)?;
            }
            partner = Some((st, other));
        }
        let mut state = SequenceState::new(seq.tokens.len())?;
        state.set_steering_strength(&stream, strength)?;
        let positions = seq.tokens.len() - 1;
        let mut d = oracle::Distributions {
            meta: oracle::DistributionMeta {
                engine: "quench6k".into(),
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
            match &mut partner {
                Some((st, other)) if step == 1 => {
                    let k = c0 - seq.prompt_len;
                    let toks = [seq.tokens[c0], other.tokens[(other.prompt_len + k) % other.tokens.len()]];
                    let bytes = forward::prepare_batch(&model, &mut [&mut state, st], &mut ws, &toks)?;
                    forward::forward_batch(&model, &[&state, st], &ws, &stream, bytes)?;
                }
                _ => forward::forward(&model, &mut state, &mut ws, &stream, &seq.tokens[c0..c1])?,
            }
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

fn generate(model_dir: &std::path::Path, precision: Precision, fixtures: &[PathBuf], tie: f32, steer: &SteerArgs) -> Result<()> {
    use quench6k_engine::forward::{self, HeadWorkspace, SequenceState, Workspace};

    #[derive(serde::Deserialize)]
    struct Sequence {
        tokens: Vec<i32>,
        prompt_len: usize,
    }

    let (mut model, stream) = load(model_dir, precision, false)?;
    let strength = steer.attach(&mut model, &stream)?;
    let seqs: Vec<(PathBuf, Sequence)> = fixtures
        .iter()
        .map(|f| Ok((f.clone(), serde_json::from_slice(&std::fs::read(f.join("sequence.json"))?)?)))
        .collect::<Result<_>>()?;
    let capacity = seqs.iter().map(|(_, s)| s.tokens.len()).max().unwrap_or(0);
    let mut ws = Workspace::for_model(&model, 1024, capacity)?;
    let hw = HeadWorkspace::new(1, 1)?;
    let mut all_ok = true;
    for (dir, seq) in &seqs {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let refs = [oracle::Distributions::load(&dir.join("llama-w4a4"))?, oracle::Distributions::load(&dir.join("llama-q8"))?];
        let mut state = SequenceState::new(seq.tokens.len())?;
        state.set_steering_strength(&stream, strength)?;
        let prompt = &seq.tokens[..seq.prompt_len];
        for c in prompt.chunks(1024) {
            forward::forward(&model, &mut state, &mut ws, &stream, c)?;
        }
        let last = ws.out + ((prompt.len() - 1) % 1024 * quench6k_engine::model::D * 2) as u64;
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

#[allow(clippy::too_many_arguments)]
fn chat_once(model_dir: &std::path::Path, prompt: &str, max_tokens: usize, temperature: f32, top_p: f32, top_k: usize, seed: u64, think: bool, context: usize, images: &[PathBuf], videos: &[PathBuf], steer: &SteerArgs) -> Result<()> {
    use std::io::Write;
    let chat = chat::Chat::load(model_dir, None)?;
    let (mut model, stream) = load(model_dir, Precision::Int8Group, true)?;
    let strength = steer.attach(&mut model, &stream)?;
    let t0 = std::time::Instant::now();
    let mut session = quench6k_engine::session::Session::new(&model, &stream, context, 8192, 6, Default::default())?;
    let mut content = vec![];
    let mut inputs = vec![];
    let mut clips = vec![];
    if !images.is_empty() || !videos.is_empty() {
        session.set_vision(quench6k_engine::vision::VisionModel::load(&Checkpoint::open(model_dir)?, LoadOptions::default(), &stream)?);
        for path in images {
            inputs.push(quench6k_engine::vision::preprocess_image(&std::fs::read(path)?, 65536, 16777216)?);
            content.push(serde_json::json!({"type": "image"}));
        }
        for path in videos {
            clips.push(quench6k_engine::vision::preprocess_video(path, 4096, 25165824, 768)?);
            content.push(serde_json::json!({"type": "video"}));
        }
    }
    content.push(serde_json::json!({"type": "text", "text": prompt}));
    println!("session ready in {:.1}s", t0.elapsed().as_secs_f64());
    let messages = serde_json::json!([{"role": "user", "content": content}]);
    let text = chat.render(&messages, None, Some(think), None)?;
    let (ids, images) = chat.expand_media(&chat.encode(&text)?, inputs, clips)?;
    let params = quench6k_engine::session::GenParams { max_tokens, temperature, top_p, top_k, min_p: 0.0, presence_penalty: 0.0, repetition_penalty: 1.0, seed, stop_tokens: chat.stop_tokens.clone(), steering_strength: strength, boundaries: chat.boundaries(&ids), capture: false, images, progress: None };
    let mut out = chat::TextStream::default();
    let t1 = std::time::Instant::now();
    let mut first = None;
    let mut err = None;
    let (finish, n) = session.generate(&model, &stream, &ids, &params, |tok| {
        first.get_or_insert_with(|| t1.elapsed());
        match out.push(&chat, tok) {
            Ok(piece) => {
                print!("{piece}");
                let _ = std::io::stdout().flush();
                true
            }
            Err(e) => {
                err = Some(e);
                false
            }
        }
    })?;
    if let Some(e) = err {
        return Err(e);
    }
    let total = t1.elapsed().as_secs_f64();
    let ttft = first.map(|d| d.as_secs_f64()).unwrap_or(total);
    println!("\n[{n} tokens, {finish:?}; prompt {} tokens; first token {:.0} ms; {:.1} tok/s after it]", ids.len(), 1e3 * ttft, (n.saturating_sub(1)) as f64 / (total - ttft).max(1e-9));
    Ok(())
}

fn batch_check(model_dir: &std::path::Path, fixtures: &[PathBuf], n: usize) -> Result<()> {
    use quench6k_engine::forward::{self, HeadWorkspace, SequenceState, Workspace};
    #[derive(serde::Deserialize)]
    struct Sequence {
        tokens: Vec<i32>,
        prompt_len: usize,
    }
    let (model, s) = load(model_dir, Precision::Int8Group, false)?;
    let prompts: Vec<Vec<i32>> = fixtures
        .iter()
        .map(|f| -> Result<Vec<i32>> {
            let seq: Sequence = serde_json::from_slice(&std::fs::read(f.join("sequence.json"))?)?;
            Ok(seq.tokens[..seq.prompt_len].to_vec())
        })
        .collect::<Result<_>>()?;
    let cap = prompts.iter().map(|p| p.len()).max().unwrap() + 2 * n + 16;
    let mut ws = Workspace::for_model(&model, 2048, cap)?;
    let hw = HeadWorkspace::new(2, 1)?;
    let out = quench6k_cuda::DeviceBuffer::new(8)?;
    // Prefill each sequence; returns (state, first token).
    let prefill = |ws: &mut Workspace, p: &[i32]| -> Result<(SequenceState, i32)> {
        let mut st = SequenceState::new(cap)?;
        for c in p.chunks(2048) {
            forward::forward(&model, &mut st, ws, &s, c)?;
        }
        let last = ws.out + (((p.len() - 1) % 2048) * 2560 * 2) as u64;
        forward::head_argmax_into(&model, &hw, &s, last, 1, out.addr())?;
        Ok((st, s.download::<i32>(out.addr(), 1)?[0]))
    };
    // Single-sequence reference.
    let mut single = Vec::new();
    let t0 = std::time::Instant::now();
    for p in &prompts {
        let (mut st, mut tok) = prefill(&mut ws, p)?;
        let mut got = vec![tok];
        let t1 = std::time::Instant::now();
        for _ in 1..n {
            forward::forward(&model, &mut st, &mut ws, &s, &[tok])?;
            forward::head_argmax_into(&model, &hw, &s, ws.out, 1, out.addr())?;
            tok = s.download::<i32>(out.addr(), 1)?[0];
            got.push(tok);
        }
        println!("single: {:.1} tok/s eager", (n - 1) as f64 / t1.elapsed().as_secs_f64());
        single.push(got);
    }
    let _ = t0;
    // Batched.
    let (mut a, ta) = prefill(&mut ws, &prompts[0])?;
    let (mut b, tb) = prefill(&mut ws, &prompts[1])?;
    let mut toks = [ta, tb];
    let mut batched = vec![vec![ta], vec![tb]];
    let t1 = std::time::Instant::now();
    for _ in 1..n {
        let bytes = forward::prepare_batch(&model, &mut [&mut a, &mut b], &mut ws, &toks)?;
        forward::forward_batch(&model, &[&a, &b], &ws, &s, bytes)?;
        forward::head_argmax_into(&model, &hw, &s, ws.out, 2, out.addr())?;
        let next: Vec<i32> = s.download(out.addr(), 2)?;
        toks = [next[0], next[1]];
        batched[0].push(next[0]);
        batched[1].push(next[1]);
    }
    println!("batched: {:.1} tok/s eager (2 sequences)", 2.0 * (n - 1) as f64 / t1.elapsed().as_secs_f64());
    // The same with a captured graph, continuing both sequences.
    let g = forward::BatchGraph::capture(&model, &[&a, &b], &ws, &hw, &s, out.addr())?;
    let t2 = std::time::Instant::now();
    for _ in 0..n {
        g.step(&model, &mut [&mut a, &mut b], &mut ws, &s, &toks)?;
        let next: Vec<i32> = s.download(out.addr(), 2)?;
        toks = [next[0], next[1]];
    }
    println!("batched graph: {:.1} tok/s (2 sequences)", 2.0 * n as f64 / t2.elapsed().as_secs_f64());
    for i in 0..2 {
        let same = batched[i].iter().zip(&single[i]).take_while(|(x, y)| x == y).count();
        println!("sequence {i}: {same}/{n} tokens identical to single-sequence decoding");
    }
    Ok(())
}

fn vision_check(model_dir: &std::path::Path, reference: &std::path::Path, max_pixels: usize) -> Result<()> {
    use quench6k_engine::vision::{VisionInput, VisionModel};
    quench6k_cuda::set_device(0)?;
    let ckpt = Checkpoint::open(model_dir)?;
    let s = quench6k_cuda::Stream::new()?;
    let t0 = std::time::Instant::now();
    let vm = VisionModel::load(&ckpt, LoadOptions::default(), &s)?;
    println!("vision tower loaded in {:.1}s", t0.elapsed().as_secs_f64());
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(reference.join("meta.json"))?)?;
    let grid: Vec<usize> = meta["grid_thw"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let read_f32 = |name: &str| -> Result<Vec<f32>> { Ok(std::fs::read(reference.join(name))?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()) };
    let input = VisionInput { pixels: read_f32("pixels.f32")?, grid: [grid[0], grid[1], grid[2]] };
    let want = read_f32("embeds.f32")?;
    if let Some(path) = meta["video"].as_str() {
        let clip = quench6k_engine::vision::preprocess_video(std::path::Path::new(path), 4096, 25165824, 768)?;
        println!("video timestamps {:?}", clip.timestamps);
        if clip.input.grid == input.grid {
            let d: Vec<f32> = clip.input.pixels.iter().zip(&input.pixels).map(|(a, b)| (a - b).abs()).collect();
            println!("video preprocessing: grid {:?}; pixel |diff| max {:.4} mean {:.5}", clip.input.grid, d.iter().cloned().fold(0f32, f32::max), d.iter().sum::<f32>() / d.len() as f32);
            let own = vm.encode(&s, &clip.input)?;
            let got: Vec<f32> = s.download(own.addr(), want.len())?;
            let cos: f64 = (0..input.tokens()).map(|t| {
                let (a, b) = (&got[t * 2560..(t + 1) * 2560], &want[t * 2560..(t + 1) * 2560]);
                let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
                dot / (a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt() * b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt())
            }).sum::<f64>() / input.tokens() as f64;
            println!("embeddings from quench6k's own preprocessing vs the reference: mean cosine {cos:.6}");
        } else {
            println!("video preprocessing: grid {:?} differs from the reference {:?}", clip.input.grid, input.grid);
        }
    }
    if let Some(path) = meta["image"].as_str() {
        let own = quench6k_engine::vision::preprocess_image(&std::fs::read(path)?, 65536, max_pixels)?;
        if own.grid == input.grid {
            let d: Vec<f32> = own.pixels.iter().zip(&input.pixels).map(|(a, b)| (a - b).abs()).collect();
            let off = d.iter().filter(|&&x| x > 1e-6).count();
            println!("preprocessing: grid {:?}; pixel |diff| max {:.4} mean {:.5}; {off} of {} values differ", own.grid, d.iter().cloned().fold(0f32, f32::max), d.iter().sum::<f32>() / d.len() as f32, d.len());
        } else {
            println!("preprocessing: grid {:?} differs from the reference {:?}", own.grid, input.grid);
        }
    }
    vm.encode(&s, &input)?; // warm-up
    let t1 = std::time::Instant::now();
    let out = vm.encode(&s, &input)?;
    let ms = t1.elapsed().as_secs_f64() * 1e3;
    let got: Vec<f32> = s.download(out.addr(), want.len())?;
    let (mut max_err, mut sum_err, mut min_cos, mut sum_cos) = (0f64, 0f64, 1f64, 0f64);
    let tokens = input.tokens();
    for t in 0..tokens {
        let (a, b) = (&got[t * 2560..(t + 1) * 2560], &want[t * 2560..(t + 1) * 2560]);
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (*x as f64, *y as f64);
            max_err = max_err.max((x - y).abs());
            sum_err += (x - y).abs();
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        let c = dot / (na.sqrt() * nb.sqrt());
        min_cos = min_cos.min(c);
        sum_cos += c;
    }
    let scale = want.iter().map(|v| v.abs() as f64).sum::<f64>() / want.len() as f64;
    println!("{} patches -> {tokens} tokens in {ms:.1} ms", input.patches());
    println!("|err| max {max_err:.5} mean {:.6} (mean |ref| {scale:.5}); cosine min {min_cos:.6} mean {:.6}", sum_err / want.len() as f64, sum_cos / tokens as f64);
    Ok(())
}
