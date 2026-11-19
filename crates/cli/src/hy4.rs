//! Commands for Hy4-flash checkpoints (`hy_v4_flash`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::Subcommand;
use quench6k_cuda::kernels as k;
use quench6k_engine::hy4::Hy4Model;
use quench6k_engine::hy4::forward::{Logits, State, TOP_K, Workspace};

use crate::chat::Chat;

#[derive(Subcommand)]
pub enum Hy4Cmd {
    /// Teacher-forced distributions of a fixture's sequence (oracle.distributions/1), as
    /// tools/ref/hy4_ref.py writes them.
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
    },
    /// Greedy generation from a chat prompt (or raw text).
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        raw: bool,
        /// Reasoning effort of the chat template (no_think, high, max).
        #[arg(long, default_value = "no_think")]
        effort: String,
        #[arg(long, default_value_t = 128)]
        n: usize,
        #[arg(long, default_value_t = 32768)]
        context: usize,
        #[arg(long, default_value_t = 2048)]
        chunk: usize,
    },
    /// Prefill and greedy decode throughput at each context length, on a text corpus (its first
    /// `ctx` tokens).
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: PathBuf,
        #[arg(long, num_args = 1.., default_values_t = [512usize, 8192])]
        ctx: Vec<usize>,
        #[arg(long, default_value_t = 64)]
        generate: usize,
        #[arg(long, default_value_t = 2048)]
        chunk: usize,
        #[arg(long)]
        json: Option<PathBuf>,
    },
}

pub fn run(cmd: Hy4Cmd) -> Result<()> {
    match cmd {
        Hy4Cmd::Score {
            model,
            fixture,
            out,
            suffix,
            chunk,
            limit,
        } => {
            let (m, s) = load(&model)?;
            for f in &fixture {
                let name = f.file_name().context("fixture name")?.to_string_lossy().into_owned();
                score(&m, &s, f, &out.join(format!("{name}{suffix}")), chunk, limit)?;
            }
            Ok(())
        }
        Hy4Cmd::Generate {
            model,
            prompt,
            raw,
            effort,
            n,
            context,
            chunk,
        } => generate(&model, &prompt, raw, &effort, n, context, chunk),
        Hy4Cmd::Bench {
            model,
            text,
            ctx,
            generate,
            chunk,
            json,
        } => bench(&model, &text, &ctx, generate, chunk, json.as_deref()),
    }
}

pub fn load(dir: &Path) -> Result<(Hy4Model, quench6k_cuda::Stream)> {
    quench6k_cuda::set_device(0)?;
    let stream = quench6k_cuda::Stream::new()?;
    let model = Hy4Model::load(dir, &stream)?;
    Ok((model, stream))
}

/// Runs `ids` through the model in chunks; the last chunk produces the logits of its last token.
fn prefill(model: &Hy4Model, st: &mut State, ws: &mut Workspace, s: &quench6k_cuda::Stream, ids: &[i32]) -> Result<()> {
    let mut at = 0;
    while at < ids.len() {
        let n = ws.max_t.min(ids.len() - at);
        let want = if at + n == ids.len() { Logits::Last } else { Logits::None };
        model.forward(st, ws, s, &ids[at..at + n], want)?;
        at += n;
    }
    Ok(())
}

fn argmax(ws: &Workspace, model: &Hy4Model, s: &quench6k_cuda::Stream) -> Result<i32> {
    k::argmax(s, ws.out_logits, 1, model.cfg.vocab, model.cfg.vocab, ws.argmax_scratch, ws.argmax)?;
    Ok(s.download::<i32>(ws.argmax, 1)?[0])
}

fn score(model: &Hy4Model, s: &quench6k_cuda::Stream, fixture: &Path, out: &Path, chunk: usize, limit: usize) -> Result<()> {
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
    let prompt_len = (seq["prompt_len"].as_u64().unwrap_or(0) as usize).min(tokens.len());
    let vocab = model.cfg.vocab;
    let mut ws = Workspace::new(model, chunk, tokens.len() + 8, chunk)?;
    let mut st = State::new(model, tokens.len() + 8, s)?;
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
            engine: "quench6k-hy4".into(),
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
    println!(
        "{}: {} positions in {secs:.1}s ({:.0} tok/s), mean NLL {nll:.4}",
        fixture.display(),
        p,
        tokens.len() as f64 / secs
    );
    Ok(())
}

fn generate(dir: &Path, prompt: &str, raw: bool, effort: &str, n: usize, context: usize, chunk: usize) -> Result<()> {
    let chat = Chat::load(dir, None)?;
    let text = if raw {
        prompt.to_string()
    } else {
        chat.render(&serde_json::json!([{"role": "user", "content": prompt}]), None, None, Some(effort))?
    };
    let ids = chat.encode(&text)?;
    let (model, s) = load(dir)?;
    let mut ws = Workspace::new(&model, chunk, context, 1)?;
    let mut st = State::new(&model, context, &s)?;
    eprintln!(
        "hy4: workspace {:.2} GiB, cache {:.2} GiB",
        ws.device_bytes as f64 / (1u64 << 30) as f64,
        st.device_bytes as f64 / (1u64 << 30) as f64
    );
    let t0 = Instant::now();
    prefill(&model, &mut st, &mut ws, &s, &ids)?;
    let mut next = argmax(&ws, &model, &s)?;
    let pf = t0.elapsed().as_secs_f64();
    let mut out = vec![next];
    let t1 = Instant::now();
    while out.len() < n && !model.cfg.eos.contains(&next) && !chat.stop_tokens.contains(&next) {
        model.forward(&mut st, &mut ws, &s, &[next], Logits::Last)?;
        next = argmax(&ws, &model, &s)?;
        out.push(next);
    }
    let dec = t1.elapsed().as_secs_f64();
    println!("{}", chat.decode(&out, false)?);
    eprintln!(
        "prompt {} tokens: prefill {:.2}s ({:.0} tok/s); {} tokens decoded at {:.1} tok/s",
        ids.len(),
        pf,
        ids.len() as f64 / pf,
        out.len(),
        (out.len() - 1) as f64 / dec
    );
    Ok(())
}

fn bench(dir: &Path, text: &Path, ctxs: &[usize], generate: usize, chunk: usize, json: Option<&Path>) -> Result<()> {
    let chat = Chat::load(dir, None)?;
    let ids = chat.encode(&std::fs::read_to_string(text)?)?;
    let (model, s) = load(dir)?;
    let max_ctx = ctxs.iter().max().copied().unwrap_or(0) + generate + 8 + 64;
    let mut ws = Workspace::new(&model, chunk, max_ctx, 1)?;
    for &ctx in ctxs {
        ensure!(ctx + generate <= ids.len(), "corpus has {} tokens, {} needed", ids.len(), ctx + generate);
        let mut st = State::new(&model, ctx + generate + 8 + 64, &s)?;
        s.synchronize()?;
        let t0 = Instant::now();
        prefill(&model, &mut st, &mut ws, &s, &ids[..ctx])?;
        let mut next = argmax(&ws, &model, &s)?;
        let prefill_s = t0.elapsed().as_secs_f64();
        let mut out = vec![next];
        let t1 = Instant::now();
        for _ in 1..generate {
            model.forward(&mut st, &mut ws, &s, &[next], Logits::Last)?;
            next = argmax(&ws, &model, &s)?;
            out.push(next);
        }
        let decode_s = t1.elapsed().as_secs_f64();
        // QUENCH6K_HY4_PROFILE_STEPS=N: N more decode steps inside a cudaProfilerStart/Stop region (nsys
        // --capture-range=cudaProfilerApi; tools/census.py --rounds N).
        if let Some(n) = std::env::var("QUENCH6K_HY4_PROFILE_STEPS").ok().and_then(|v| v.parse::<usize>().ok()) {
            s.synchronize()?;
            quench6k_cuda::profiler_start()?;
            for _ in 0..n {
                model.forward(&mut st, &mut ws, &s, &[next], Logits::Last)?;
                next = argmax(&ws, &model, &s)?;
            }
            s.synchronize()?;
            quench6k_cuda::profiler_stop()?;
        }
        let copied = out.iter().zip(&ids[ctx..]).filter(|(a, b)| a == b).count();
        let pf = ctx as f64 / prefill_s;
        let tg = (generate - 1) as f64 / decode_s;
        println!(
            "ctx {ctx:>6}: prefill {pf:>7.0} tok/s ({prefill_s:.2}s), decode {tg:>6.1} tok/s; corpus continuation {copied}/{generate}; {:?}",
            chat.decode(&out[..out.len().min(24)], false)?
        );
        if let Some(path) = json {
            use std::io::Write;
            let rec = serde_json::json!({"engine": "quench6k", "model": "hy4-flash-nvfp4", "mode": "plain", "ctx": ctx,
                "generate": generate, "prefill_tok_s": (pf * 10.0).round() / 10.0, "decode_tok_s": (tg * 10.0).round() / 10.0,
                "copied": copied});
            writeln!(std::fs::OpenOptions::new().create(true).append(true).open(path)?, "{rec}")?;
        }
    }
    Ok(())
}
