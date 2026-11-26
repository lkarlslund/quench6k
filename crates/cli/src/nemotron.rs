//! Commands for Nemotron-H checkpoints (`nemotron_h`: Nemotron 3 Super, Munin 2.0).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::Subcommand;
use quench6k_cuda::kernels as k;
use quench6k_engine::nemotron::forward::{CANON_CHUNK, Kv, Logits, MAX_SEQS, State, TOP_K, Workspace};
use quench6k_engine::nemotron::{Config, NemotronModel, WeightPlan};

use crate::chat::Chat;

#[derive(Subcommand)]
pub enum NemotronCmd {
    /// Device memory of the weights (from config.json and hf_quant_config.json; no GPU, no weights read)
    /// and of the per-sequence caches and states: how many sequences of a given context fit.
    Plan {
        #[arg(long)]
        model: PathBuf,
        /// Device memory in GiB.
        #[arg(long, default_value_t = 95.0)]
        device_gib: f64,
        /// Context per sequence.
        #[arg(long, default_value_t = 262144)]
        ctx: usize,
        /// Also check the safetensors (complete shards, every tensor's dtype and shape, the scales) and
        /// count what a load allocates, without a GPU.
        #[arg(long)]
        headers: bool,
        /// Attention cache: bf16, or fp8 (a scale per position and KV head).
        #[arg(long, default_value = "bf16")]
        kv: String,
    },
    /// Greedy generation from a chat prompt, raw text or token ids.
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "")]
        prompt: String,
        /// Read the prompt from a file (long prompts).
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// The prompt as token ids (comma-separated), instead of text.
        #[arg(long, value_delimiter = ',')]
        ids: Vec<i32>,
        /// The prompt text as is, without the chat template.
        #[arg(long)]
        raw: bool,
        /// Thinking: on (default), off, or low (the template's low-effort mode).
        #[arg(long, default_value = "on")]
        think: String,
        #[arg(long, default_value_t = 128)]
        n: usize,
        #[arg(long, default_value_t = 32768)]
        context: usize,
        /// Attention cache: bf16, or fp8 (a scale per position and KV head).
        #[arg(long, default_value = "bf16")]
        kv: String,
        /// Hidden states after each layer (lanes-NN.bin, F32 [T, 4096]) and after the final norm
        /// (final.bin) appended to this directory, for comparison with tools/ref dumps.
        #[arg(long)]
        dump_dir: Option<PathBuf>,
    },
    /// Teacher-forced distributions of a fixture's sequence (oracle.distributions/1).
    Score {
        #[arg(long)]
        model: PathBuf,
        /// Fixture directories with sequence.json.
        #[arg(long, num_args = 1..)]
        fixture: Vec<PathBuf>,
        /// Output directory: writes <fixture><suffix>.{json,bin}.
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "", allow_hyphen_values = true)]
        suffix: String,
        #[arg(long, default_value_t = 512)]
        chunk: usize,
        /// Score only the first N tokens.
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Attention cache: bf16, or fp8 (a scale per position and KV head).
        #[arg(long, default_value = "bf16")]
        kv: String,
        #[arg(long)]
        dump_dir: Option<PathBuf>,
    },
    /// Batch invariance: N sequences (corpus windows of different lengths) decoded alone, then the same
    /// steps in batched steps (forward_batch); every step's logits compared bitwise.
    BatchCheck {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: PathBuf,
        /// Prompt length of the first sequence (the others are shorter by 37 tokens each).
        #[arg(long, default_value_t = 1000)]
        ctx: usize,
        #[arg(long, default_value_t = 4)]
        seqs: usize,
        #[arg(long, default_value_t = 8)]
        steps: usize,
        #[arg(long, default_value = "bf16")]
        kv: String,
    },
    /// Prefill and greedy decode throughput at each context length, on a text corpus (its first `ctx`
    /// tokens); with --seqs N, N sequences (corpus windows) decoded in lockstep batched steps.
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: PathBuf,
        #[arg(long, num_args = 1.., default_values_t = [512usize, 8192])]
        ctx: Vec<usize>,
        #[arg(long, default_value_t = 64)]
        generate: usize,
        #[arg(long, default_value_t = 1)]
        seqs: usize,
        /// Attention cache: bf16, or fp8 (a scale per position and KV head).
        #[arg(long, default_value = "bf16")]
        kv: String,
        #[arg(long)]
        json: Option<PathBuf>,
    },
}

pub fn run(cmd: NemotronCmd) -> Result<()> {
    match cmd {
        NemotronCmd::Plan {
            model,
            device_gib,
            ctx,
            headers,
            kv,
        } => plan(&model, device_gib, ctx, headers, Kv::parse(&kv)?),
        NemotronCmd::Generate {
            model,
            prompt,
            prompt_file,
            ids,
            raw,
            think,
            n,
            context,
            kv,
            dump_dir,
        } => {
            let prompt = match prompt_file {
                Some(f) => std::fs::read_to_string(f)?,
                None => prompt,
            };
            generate(&model, &prompt, &ids, raw, &think, n, context, Kv::parse(&kv)?, dump_dir)
        }
        NemotronCmd::Score {
            model,
            fixture,
            out,
            suffix,
            chunk,
            limit,
            kv,
            dump_dir,
        } => {
            let kv = Kv::parse(&kv)?;
            let (m, s) = load(&model)?;
            for f in &fixture {
                let name = f.file_name().context("fixture name")?.to_string_lossy().into_owned();
                score(&m, &s, f, &out.join(format!("{name}{suffix}")), chunk, limit, kv, dump_dir.clone())?;
            }
            Ok(())
        }
        NemotronCmd::BatchCheck {
            model,
            text,
            ctx,
            seqs,
            steps,
            kv,
        } => batch_check(&model, &text, ctx, seqs, steps, Kv::parse(&kv)?),
        NemotronCmd::Bench {
            model,
            text,
            ctx,
            generate,
            seqs,
            kv,
            json,
        } => bench(&model, &text, &ctx, generate, seqs, Kv::parse(&kv)?, json.as_deref()),
    }
}

fn gib(b: usize) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn plan(dir: &Path, device_gib: f64, ctx: usize, headers: bool, kv: Kv) -> Result<()> {
    let cfg = Config::read(dir)?;
    let p = WeightPlan::read(dir)?;
    println!("weights on the device: {:.2} GiB", gib(p.device()));
    for (name, b) in [
        ("routed experts (NVFP4)", p.experts),
        ("Mamba projections and parameters", p.mamba),
        ("attention projections", p.attention),
        ("MoE latent and shared projections", p.moe_dense),
        ("routers", p.router),
        ("norms", p.norms_small),
        ("LM head", p.lm_head),
        ("MTP layer", p.mtp),
    ] {
        println!("  {name:36} {:7.2} GiB", gib(b));
    }
    println!("embedding (host, memory-mapped): {:.2} GiB", gib(p.embed_host));
    let ws = Workspace::bytes(&cfg, CANON_CHUNK, 1);
    let per_pos = State::position_bytes(&cfg, kv);
    let fixed = State::state_bytes(&cfg);
    println!("workspace (chunks of {CANON_CHUNK}): {:.2} GiB", gib(ws));
    println!("per sequence: {:.1} MiB of Mamba states + {per_pos} bytes per position of attention cache ({kv:?})", fixed as f64 / (1 << 20) as f64);
    let free = device_gib * (1u64 << 30) as f64 - (p.device() + ws) as f64;
    let seq = (fixed + ctx * per_pos) as f64;
    println!(
        "{device_gib:.1} GiB device: {:.2} GiB left for sequences; a sequence of {ctx} positions takes {:.2} GiB; {} fit",
        gib(free.max(0.0) as usize),
        seq / (1u64 << 30) as f64,
        (free / seq).floor().max(0.0)
    );
    if headers {
        let t0 = Instant::now();
        let d = NemotronModel::dry_run(dir)?;
        println!("checkpoint headers ({:.1}s): a load allocates {:.2} GiB (MTP layer {})", t0.elapsed().as_secs_f64(), gib(d.device_bytes), if d.mtp { "loaded" } else { "not loaded" });
        for ((stored, kept), (count, bytes)) in &d.kept {
            println!("  {count:4} projections stored {stored:6} kept {kept:12} {:7.2} GiB", gib(*bytes));
        }
        for (layer, ks, vs) in &d.cache_scales {
            println!("  layer {layer:2}: checkpoint k_scale {ks:.6}, v_scale {vs:.6} (not used)");
        }
    }
    Ok(())
}

/// Removes an earlier dump's files from `dir` (dumps are appended chunk by chunk).
fn fresh_dump(dir: &Path) -> Result<()> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if (name.starts_with("lanes-") || name == "final.bin") && name.ends_with(".bin") {
                std::fs::remove_file(e.path())?;
            }
        }
    }
    Ok(())
}

pub fn load(dir: &Path) -> Result<(NemotronModel, quench6k_cuda::Stream)> {
    quench6k_cuda::set_device(0)?;
    let stream = quench6k_cuda::Stream::new()?;
    let model = NemotronModel::load(dir, &stream)?;
    Ok((model, stream))
}

pub(crate) fn argmax(ws: &Workspace, model: &NemotronModel, s: &quench6k_cuda::Stream, rows: usize) -> Result<Vec<i32>> {
    let v = model.cfg.vocab;
    k::argmax(s, ws.out_logits, rows, v, v, ws.argmax_scratch, ws.argmax)?;
    s.download::<i32>(ws.argmax, rows)
}

#[allow(clippy::too_many_arguments)]
fn generate(dir: &Path, prompt: &str, ids: &[i32], raw: bool, think: &str, n: usize, context: usize, kv: Kv, dump: Option<PathBuf>) -> Result<()> {
    let chat = Chat::load(dir, None)?;
    let ids = if !ids.is_empty() {
        ids.to_vec()
    } else {
        let text = if raw {
            prompt.to_string()
        } else {
            let (on, effort) = match think {
                "on" => (None, None),
                "off" => (Some(false), None),
                "low" => (None, Some("low")),
                other => anyhow::bail!("--think {other}: on, off or low"),
            };
            chat.render(&serde_json::json!([{"role": "user", "content": prompt}]), None, on, effort)?
        };
        chat.encode(&text)?
    };
    ensure!(!ids.is_empty(), "empty prompt");
    let (model, s) = load(dir)?;
    let mut ws = Workspace::new(&model, CANON_CHUNK, 1)?;
    ws.dump = dump.or(ws.dump.take());
    if let Some(d) = &ws.dump {
        fresh_dump(d)?;
    }
    let mut st = State::new(&model, context, kv, &s)?;
    eprintln!("nemotron: workspace {:.2} GiB, sequence {:.2} GiB fixed", gib(ws.device_bytes), gib(st.device_bytes()));
    let t0 = Instant::now();
    model.prefill(&mut st, &mut ws, &s, &ids)?;
    let mut next = argmax(&ws, &model, &s, 1)?[0];
    let pf = t0.elapsed().as_secs_f64();
    eprintln!(
        "nemotron: sequence {:.2} GiB after the prompt; device {:.2} GiB used",
        gib(st.device_bytes()),
        gib(quench6k_cuda::mem_info().map_or(0, |(free, total)| total - free))
    );
    let stop = |t: i32| model.cfg.eos.contains(&t) || chat.stop_tokens.contains(&t);
    let mut out = vec![next];
    let t1 = Instant::now();
    while out.len() < n && !stop(next) {
        model.forward(&mut st, &mut ws, &s, &[next], Logits::Last)?;
        next = argmax(&ws, &model, &s, 1)?[0];
        out.push(next);
    }
    let dec = t1.elapsed().as_secs_f64();
    println!("{}", chat.decode(&out, false)?);
    eprintln!("ids: {out:?}");
    eprintln!(
        "prompt {} tokens: prefill {:.2}s ({:.0} tok/s); {} tokens decoded at {:.1} tok/s",
        ids.len(),
        pf,
        ids.len() as f64 / pf,
        out.len(),
        (out.len() - 1) as f64 / dec.max(1e-9)
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn score(model: &NemotronModel, s: &quench6k_cuda::Stream, fixture: &Path, out: &Path, chunk: usize, limit: usize, kv: Kv, dump: Option<PathBuf>) -> Result<()> {
    let seq: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixture.join("sequence.json")).with_context(|| format!("{}/sequence.json", fixture.display()))?,
    )?;
    let mut tokens: Vec<i32> = seq["tokens"]
        .as_array()
        .context("sequence.json: tokens")?
        .iter()
        .map(|v| v.as_i64().unwrap_or(0) as i32)
        .collect();
    if limit > 0 {
        tokens.truncate(limit);
    }
    ensure!(tokens.len() >= 2, "{}: fewer than two tokens", fixture.display());
    let prompt_len = (seq["prompt_len"].as_u64().unwrap_or(0) as usize).min(tokens.len());
    let vocab = model.cfg.vocab;
    let mut ws = Workspace::new(model, chunk, chunk)?;
    if let Some(d) = dump {
        ws.dump = Some(d.join(fixture.file_name().unwrap_or_default()));
    }
    if let Some(d) = &ws.dump {
        fresh_dump(d)?;
    }
    let mut st = State::new(model, tokens.len() + 8, kv, s)?;
    let p = tokens.len() - 1;
    let (mut lse, mut target, mut top_ids, mut top_logits) = (vec![], vec![], vec![], vec![]);
    let t0 = Instant::now();
    let mut at = 0;
    while at < p {
        let n = chunk.min(tokens.len() - at);
        model.forward(&mut st, &mut ws, s, &tokens[at..at + n], Logits::All)?;
        let rows = n.min(p - at);
        s.upload_to(ws.targets, &tokens[at + 1..at + 1 + rows])?;
        k::row_lse(s, ws.out_logits, rows, vocab, vocab, ws.targets, ws.lse, ws.target)?;
        k::topk(s, ws.out_logits, rows, vocab, vocab, TOP_K, false, ws.top_idx, ws.top_val)?;
        lse.extend(s.download::<f32>(ws.lse, rows)?);
        target.extend(s.download::<f32>(ws.target, rows)?);
        top_ids.extend(s.download::<i32>(ws.top_idx, rows * TOP_K)?);
        top_logits.extend(s.download::<f32>(ws.top_val, rows * TOP_K)?);
        at += n;
        eprint!("\rscored {at}/{}", tokens.len());
    }
    eprintln!();
    let secs = t0.elapsed().as_secs_f64();
    let mut extra = serde_json::Map::new();
    extra.insert("mode".into(), "teacher_forced".into());
    extra.insert("chunk".into(), chunk.into());
    extra.insert("seconds".into(), ((secs * 10.0).round() / 10.0).into());
    let d = quench6k_engine::oracle::Distributions {
        meta: quench6k_engine::oracle::DistributionMeta {
            engine: "quench6k-nemotron".into(),
            format: "oracle.distributions/1".into(),
            positions: p,
            prompt_len,
            top_k: TOP_K,
            extra,
        },
        lse,
        target,
        top_ids,
        top_logits,
    };
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    d.save(out)?;
    let nll: f64 = d.logprob().iter().map(|v| -v).sum::<f64>() / p as f64;
    println!("{}: {p} positions in {secs:.1}s ({:.0} tok/s), mean NLL {nll:.4}", fixture.display(), tokens.len() as f64 / secs);
    Ok(())
}

fn bench(dir: &Path, text: &Path, ctxs: &[usize], generate: usize, seqs: usize, kv: Kv, json: Option<&Path>) -> Result<()> {
    ensure!((1..=MAX_SEQS).contains(&seqs), "--seqs 1 to {MAX_SEQS}");
    ensure!(generate >= 2, "--generate at least 2");
    let chat = Chat::load(dir, None)?;
    let ids = chat.encode(&std::fs::read_to_string(text)?)?;
    let (model, s) = load(dir)?;
    let mut ws = Workspace::new(&model, CANON_CHUNK, seqs)?;
    for &ctx in ctxs {
        // Sequence j reads the corpus from token 37 j on (different prompts of the same length).
        ensure!(ctx + generate + 37 * seqs <= ids.len(), "corpus has {} tokens, {} needed", ids.len(), ctx + generate + 37 * seqs);
        let mut states = Vec::with_capacity(seqs);
        let mut firsts = Vec::with_capacity(seqs);
        // QUENCH6K_NEM_PROFILE=1: the prefills inside a cudaProfilerStart/Stop range (nsys
        // --capture-range=cudaProfilerApi).
        let profile = std::env::var("QUENCH6K_NEM_PROFILE").is_ok_and(|v| v == "1");
        s.synchronize()?;
        if profile {
            quench6k_cuda::profiler_start()?;
        }
        let t0 = Instant::now();
        for j in 0..seqs {
            let mut st = State::new(&model, ctx + generate + 16, kv, &s)?;
            model.prefill(&mut st, &mut ws, &s, &ids[37 * j..37 * j + ctx])?;
            firsts.push(argmax(&ws, &model, &s, 1)?[0]);
            states.push(st);
        }
        let prefill_s = t0.elapsed().as_secs_f64();
        if profile {
            quench6k_cuda::profiler_stop()?;
        }
        let mut outs: Vec<Vec<i32>> = firsts.iter().map(|&t| vec![t]).collect();
        let t1 = Instant::now();
        for _ in 1..generate {
            let toks: Vec<[i32; 1]> = outs.iter().map(|o| [*o.last().unwrap()]).collect();
            let tok_refs: Vec<&[i32]> = toks.iter().map(|t| t.as_slice()).collect();
            let mut refs: Vec<&mut State> = states.iter_mut().collect();
            if seqs == 1 {
                model.forward(refs[0], &mut ws, &s, tok_refs[0], Logits::Last)?;
            } else {
                model.forward_batch(&mut refs, &mut ws, &s, &tok_refs)?;
            }
            let picks = argmax(&ws, &model, &s, seqs)?;
            for (o, p) in outs.iter_mut().zip(picks) {
                o.push(p);
            }
        }
        let decode_s = t1.elapsed().as_secs_f64();
        let copied = outs[0].iter().zip(&ids[ctx..]).filter(|(a, b)| a == b).count();
        let pf = (ctx * seqs) as f64 / prefill_s;
        let tg = ((generate - 1) * seqs) as f64 / decode_s;
        println!(
            "ctx {ctx:>6} x {seqs}: prefill {pf:>7.0} tok/s ({prefill_s:.2}s), decode {tg:>6.1} tok/s; corpus continuation {copied}/{generate}; {:?}",
            chat.decode(&outs[0][..outs[0].len().min(24)], false)?
        );
        if let Some(path) = json {
            use std::io::Write;
            let rec = serde_json::json!({"engine": "quench6k", "model": "nemotron-h", "kv": format!("{kv:?}").to_lowercase(), "seqs": seqs, "ctx": ctx,
                "generate": generate, "prefill_tok_s": (pf * 10.0).round() / 10.0, "decode_tok_s": (tg * 10.0).round() / 10.0,
                "copied": copied});
            writeln!(std::fs::OpenOptions::new().create(true).append(true).open(path)?, "{rec}")?;
        }
    }
    Ok(())
}

fn batch_check(dir: &Path, text: &Path, ctx: usize, seqs: usize, steps: usize, kv: Kv) -> Result<()> {
    ensure!((2..=MAX_SEQS).contains(&seqs), "--seqs 2 to {MAX_SEQS}");
    ensure!(ctx > 37 * seqs, "--ctx must exceed 37 x seqs");
    let chat = Chat::load(dir, None)?;
    let ids = chat.encode(&std::fs::read_to_string(text)?)?;
    ensure!(ids.len() >= ctx + 64 * seqs, "corpus too short");
    let (model, s) = load(dir)?;
    let v = model.cfg.vocab;
    let mut ws = Workspace::new(&model, CANON_CHUNK, seqs)?;
    // Sequence j: corpus from token 64 j on, ctx - 37 j tokens.
    let prompt = |j: usize| &ids[64 * j..64 * j + ctx - 37 * j];
    // Alone: each sequence's greedy tokens and every step's logits.
    let mut toks: Vec<Vec<i32>> = Vec::new();
    let mut alone: Vec<Vec<Vec<f32>>> = Vec::new();
    for j in 0..seqs {
        let mut st = State::new(&model, ctx + steps + 16, kv, &s)?;
        model.prefill(&mut st, &mut ws, &s, prompt(j))?;
        let mut t = vec![argmax(&ws, &model, &s, 1)?[0]];
        let mut l = Vec::new();
        for _ in 0..steps {
            model.forward(&mut st, &mut ws, &s, &[*t.last().unwrap()], Logits::All)?;
            l.push(s.download::<f32>(ws.out_logits, v)?);
            t.push(argmax(&ws, &model, &s, 1)?[0]);
        }
        toks.push(t);
        alone.push(l);
    }
    // Batched: the same tokens, all sequences in each step.
    let mut states = Vec::new();
    for j in 0..seqs {
        let mut st = State::new(&model, ctx + steps + 16, kv, &s)?;
        model.prefill(&mut st, &mut ws, &s, prompt(j))?;
        states.push(st);
    }
    let mut mismatches = 0;
    for step in 0..steps {
        let cur: Vec<[i32; 1]> = toks.iter().map(|t| [t[step]]).collect();
        let refs: Vec<&[i32]> = cur.iter().map(|c| c.as_slice()).collect();
        let mut sts: Vec<&mut State> = states.iter_mut().collect();
        model.forward_batch(&mut sts, &mut ws, &s, &refs)?;
        let all = s.download::<f32>(ws.out_logits, seqs * v)?;
        for j in 0..seqs {
            let row = &all[j * v..(j + 1) * v];
            let same = row.iter().zip(&alone[j][step]).all(|(a, b)| a.to_bits() == b.to_bits());
            if !same {
                mismatches += 1;
                eprintln!("step {step} sequence {j}: logits differ from the sequence alone");
            }
        }
    }
    for (j, st) in states.iter().enumerate() {
        println!("sequence {j}: {} positions, tokens {:?}", st.pos, &toks[j][..toks[j].len().min(8)]);
    }
    ensure!(mismatches == 0, "{mismatches} of {} step rows differ", seqs * steps);
    println!("batch-check: {seqs} sequences x {steps} steps bit-identical alone and batched");
    Ok(())
}
