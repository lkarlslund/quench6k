//! Commands for Inkling 2 checkpoints (`inkling2_loop`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::Subcommand;
use quench6k_cuda::kernels as k;
use quench6k_engine::inkling::InklingModel;
use quench6k_engine::inkling::forward::{Logits, State, TOP_K, Workspace};

use crate::chat::Chat;

#[derive(Subcommand)]
pub enum InklingCmd {
    /// Teacher-forced distributions of a fixture's sequence (oracle.distributions/1), as
    /// tools/ref/inkling_ref.py writes them.
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
        #[arg(long, default_value = "")]
        prompt: String,
        /// Read the prompt from a file (long prompts).
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        #[arg(long)]
        raw: bool,
        /// Thinking effort of the chat template (none, minimal, low, medium, high, xhigh, max).
        #[arg(long, default_value = "none")]
        effort: String,
        #[arg(long, default_value_t = 128)]
        n: usize,
        #[arg(long, default_value_t = 32768)]
        context: usize,
        #[arg(long, default_value_t = 2048)]
        chunk: usize,
        /// Greedy decoding with the checkpoint's DSpark drafter.
        #[arg(long)]
        draft: bool,
    },
    /// Prefill and greedy decode throughput at each context length, on a text corpus (its first `ctx`
    /// tokens).
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
        /// Greedy decoding with the checkpoint's DSpark drafter.
        #[arg(long)]
        draft: bool,
    },
    /// Batched small steps: N sequences (different prompts and lengths from the corpus) decoded alone and
    /// in lockstep batches (forward_batch / draft_batch), every step's logits and drafts compared bitwise;
    /// then step time and aggregate tokens per second for each sequence count at each context.
    BatchCheck {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: PathBuf,
        /// Contexts (prompt lengths of the first sequence; the others are shorter by 37 tokens each).
        #[arg(long, num_args = 1.., default_values_t = [512usize])]
        ctx: Vec<usize>,
        /// Sequence counts.
        #[arg(long, num_args = 1.., default_values_t = [1usize, 2, 4, 8])]
        seqs: Vec<usize>,
        /// Steps of the bitwise check (each sequence).
        #[arg(long, default_value_t = 12)]
        steps: usize,
        /// Timed steps per sequence count (0: no timing).
        #[arg(long, default_value_t = 32)]
        time_steps: usize,
        /// Skip the bitwise check.
        #[arg(long)]
        no_check: bool,
    },
}

pub fn run(cmd: InklingCmd) -> Result<()> {
    match cmd {
        InklingCmd::Score {
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
        InklingCmd::Generate {
            model,
            prompt,
            prompt_file,
            raw,
            effort,
            n,
            context,
            chunk,
            draft,
        } => {
            let prompt = match prompt_file {
                Some(f) => std::fs::read_to_string(f)?,
                None => prompt,
            };
            generate(&model, &prompt, raw, &effort, n, context, chunk, draft)
        }
        InklingCmd::BatchCheck {
            model,
            text,
            ctx,
            seqs,
            steps,
            time_steps,
            no_check,
        } => crate::inkling_batch::run(&model, &text, &ctx, &seqs, steps, time_steps, !no_check),
        InklingCmd::Bench {
            model,
            text,
            ctx,
            generate,
            chunk,
            json,
            draft,
        } => bench(&model, &text, &ctx, generate, chunk, json.as_deref(), draft),
    }
}

pub fn load(dir: &Path) -> Result<(InklingModel, quench6k_cuda::Stream)> {
    quench6k_cuda::set_device(0)?;
    let stream = quench6k_cuda::Stream::new()?;
    let model = InklingModel::load(dir, &stream)?;
    Ok((model, stream))
}

/// Runs `ids` through the model in chunks; the last chunk produces the logits of its last token.
pub(crate) fn prefill(model: &InklingModel, st: &mut State, ws: &mut Workspace, s: &quench6k_cuda::Stream, ids: &[i32]) -> Result<()> {
    let mut at = 0;
    while at < ids.len() {
        let n = ws.max_t.min(ids.len() - at);
        let want = if at + n == ids.len() { Logits::Last } else { Logits::None };
        model.forward(st, ws, s, &ids[at..at + n], want)?;
        at += n;
    }
    Ok(())
}

pub(crate) fn argmax(ws: &Workspace, model: &InklingModel, s: &quench6k_cuda::Stream) -> Result<i32> {
    k::argmax(s, ws.out_logits, 1, model.cfg.vocab, model.cfg.vocab, ws.argmax_scratch, ws.argmax)?;
    Ok(s.download::<i32>(ws.argmax, 1)?[0])
}

/// Drafted-and-verified greedy decoding after a prefill whose last step chose `first`: up to `n` tokens
/// (stopping after any of `stop`). Returns the tokens and the number of verification steps.
fn decode_draft(
    model: &InklingModel,
    st: &mut State,
    ws: &mut Workspace,
    s: &quench6k_cuda::Stream,
    first: i32,
    n: usize,
    stop: &dyn Fn(i32) -> bool,
) -> Result<(Vec<i32>, usize)> {
    let vocab = model.cfg.vocab;
    let mut out = vec![first];
    let mut pending = first;
    let mut row = st.last_t - 1;
    let mut steps = 0;
    let timing = std::env::var_os("QUENCH6K_INK_TIMING").is_some();
    let (mut t_draft, mut t_verify) = (0.0f64, 0.0f64);
    let mut hist = [0usize; 9];
    let fixed = std::env::var("QUENCH6K_INK_DRAFTS").ok().and_then(|v| v.parse::<usize>().ok());
    let mut policy = quench6k_engine::hy4::policy::DraftPolicy::new(
        model.drafter.as_ref().map_or(0, |d| d.block - 1),
        0.0045,
        0.0008,
        0.0006,
    );
    // QUENCH6K_INK_CONF_LOG=file: the drafts' confidences and outcomes (calibration data).
    let mut conf_log = match std::env::var_os("QUENCH6K_INK_CONF_LOG") {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => None,
    };
    while out.len() < n && !stop(pending) {
        let t0 = Instant::now();
        // Fixed count (QUENCH6K_INK_DRAFTS), or the policy: run the drafter while it pays, then verify
        // the count its stated probabilities favour.
        let run = fixed.map_or_else(|| policy.want_draft(), |f| f > 0);
        let mut drafts = if run { model.draft(st, ws, s, pending, row)? } else { Vec::new() };
        let conf = model.draft_conf(ws);
        let want_drafts = fixed.unwrap_or_else(|| if drafts.is_empty() { 0 } else { policy.drafts_conf(&conf[..drafts.len()]) });
        let t1 = Instant::now();
        t_draft += (t1 - t0).as_secs_f64();
        drafts.truncate(want_drafts.min(n - out.len()));
        let mut toks = vec![pending];
        toks.extend_from_slice(&drafts);
        let before = st.pos;
        let want = if toks.len() == 1 { Logits::Last } else { Logits::All };
        model.forward(st, ws, s, &toks, want)?;
        steps += 1;
        k::argmax(s, ws.out_logits, toks.len(), vocab, vocab, ws.argmax_scratch, ws.argmax)?;
        let picks = s.download::<i32>(ws.argmax, toks.len())?;
        let verify = t1.elapsed().as_secs_f64();
        t_verify += verify;
        let mut j = 0;
        loop {
            let tok = picks[j];
            out.push(tok);
            if j < drafts.len() && tok == drafts[j] && !stop(tok) && out.len() < n {
                j += 1;
                continue;
            }
            pending = tok;
            break;
        }
        if let Some(log) = conf_log.as_mut() {
            use std::io::Write;
            // Each judged draft (the accepted ones and the first rejected): position, confidence, kept.
            for k in 0..drafts.len().min(j + 1) {
                writeln!(log, "{k} {} {}", conf[k], (k < j) as u8)?;
            }
        }
        st.truncate(before + 1 + j);
        row = j;
        hist[j.min(8)] += 1;
        policy.update_conf(&conf, drafts.len(), j, verify, run.then(|| (t1 - t0).as_secs_f64()));
    }
    if timing {
        eprintln!("draft: accepted drafts per step (0, 1, ...): {:?}", &hist);
        eprintln!(
            "draft: {steps} steps, draft {:.2} ms, verify {:.2} ms per step",
            t_draft * 1e3 / steps.max(1) as f64,
            t_verify * 1e3 / steps.max(1) as f64
        );
    }
    Ok((out, steps))
}

fn score(model: &InklingModel, s: &quench6k_cuda::Stream, fixture: &Path, out: &Path, chunk: usize, limit: usize) -> Result<()> {
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
    let mut ws = Workspace::new(model, chunk, chunk)?;
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
            engine: "quench6k-inkling".into(),
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

#[allow(clippy::too_many_arguments)]
fn generate(dir: &Path, prompt: &str, raw: bool, effort: &str, n: usize, context: usize, chunk: usize, draft: bool) -> Result<()> {
    let chat = Chat::load(dir, None)?;
    let text = if raw {
        prompt.to_string()
    } else {
        chat.render(&serde_json::json!([{"role": "user", "content": prompt}]), None, None, Some(effort))?
    };
    let ids = chat.encode(&text)?;
    let (model, s) = load(dir)?;
    let mut ws = Workspace::new(&model, chunk, if draft { 8 } else { 1 })?;
    let mut st = State::new(&model, context, &s)?;
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    eprintln!("inkling: workspace {:.2} GiB, sequence {:.2} GiB fixed", gib(ws.device_bytes), gib(st.device_bytes()));
    let t0 = Instant::now();
    prefill(&model, &mut st, &mut ws, &s, &ids)?;
    let mut next = argmax(&ws, &model, &s)?;
    let pf = t0.elapsed().as_secs_f64();
    eprintln!(
        "inkling: sequence {:.2} GiB after the prompt; device {:.2} GiB used",
        gib(st.device_bytes()),
        gib(quench6k_cuda::mem_info().map_or(0, |(free, total)| total - free))
    );
    let mut out = vec![next];
    let t1 = Instant::now();
    let mut steps = 0;
    if draft {
        let stop = |t: i32| model.cfg.eos.contains(&t) || chat.stop_tokens.contains(&t);
        (out, steps) = decode_draft(&model, &mut st, &mut ws, &s, next, n, &stop)?;
    } else {
        while out.len() < n && !model.cfg.eos.contains(&next) && !chat.stop_tokens.contains(&next) {
            model.forward(&mut st, &mut ws, &s, &[next], Logits::Last)?;
            next = argmax(&ws, &model, &s)?;
            out.push(next);
        }
    }
    let dec = t1.elapsed().as_secs_f64();
    if draft {
        eprintln!("draft: {:.2} tokens per step", (out.len() - 1) as f64 / steps.max(1) as f64);
    }
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

fn bench(dir: &Path, text: &Path, ctxs: &[usize], generate: usize, chunk: usize, json: Option<&Path>, draft: bool) -> Result<()> {
    let chat = Chat::load(dir, None)?;
    let ids = chat.encode(&std::fs::read_to_string(text)?)?;
    let (model, s) = load(dir)?;
    let mut ws = Workspace::new(&model, chunk, if draft { 8 } else { 1 })?;
    for &ctx in ctxs {
        ensure!(ctx + generate <= ids.len(), "corpus has {} tokens, {} needed", ids.len(), ctx + generate);
        // Room for the profile steps (QUENCH6K_INK_PROFILE_STEPS: each keeps one token).
        let profile = std::env::var("QUENCH6K_INK_PROFILE_STEPS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let mut st = State::new(&model, ctx + generate + 16 + profile, &s)?;
        s.synchronize()?;
        let t0 = Instant::now();
        prefill(&model, &mut st, &mut ws, &s, &ids[..ctx])?;
        let mut next = argmax(&ws, &model, &s)?;
        let prefill_s = t0.elapsed().as_secs_f64();
        let mut out = vec![next];
        let t1 = Instant::now();
        let mut steps = generate - 1;
        if draft {
            (out, steps) = decode_draft(&model, &mut st, &mut ws, &s, next, generate, &|_| false)?;
            next = *out.last().unwrap();
        } else {
            for _ in 1..generate {
                model.forward(&mut st, &mut ws, &s, &[next], Logits::Last)?;
                next = argmax(&ws, &model, &s)?;
                out.push(next);
            }
        }
        let decode_s = t1.elapsed().as_secs_f64();
        // QUENCH6K_INK_PROFILE_STEPS=N: N more decode steps inside a cudaProfilerStart/Stop region.
        if let Some(n) = std::env::var("QUENCH6K_INK_PROFILE_STEPS").ok().and_then(|v| v.parse::<usize>().ok()) {
            s.synchronize()?;
            quench6k_cuda::profiler_start()?;
            // QUENCH6K_INK_PROFILE_WIDTH=w: steps of w tokens (a verification step's shape; the cache
            // rolls back after each).
            ensure!(ctx + generate + n + 16 <= ids.len(), "corpus too short for the profile steps");
            let w = std::env::var("QUENCH6K_INK_PROFILE_WIDTH").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                if w == 1 {
                    model.forward(&mut st, &mut ws, &s, &[next], Logits::Last)?;
                    next = argmax(&ws, &model, &s)?;
                } else {
                    let before = st.pos;
                    // The corpus's own next tokens (as accepted drafts would be).
                    let mut toks = vec![next];
                    toks.extend_from_slice(&ids[ctx + generate..ctx + generate + w - 1]);
                    model.forward(&mut st, &mut ws, &s, &toks, Logits::All)?;
                    st.truncate(before + 1);
                }
            }
            s.synchronize()?;
            quench6k_cuda::profiler_stop()?;
            let step = t0.elapsed().as_secs_f64() / n.max(1) as f64;
            println!("ctx {ctx:>6}: {n} steps of {w} tokens, {:.3} ms a step, verify{w} {:.1} tok/s", step * 1e3, w as f64 / step);
        }
        let copied = out.iter().zip(&ids[ctx..]).filter(|(a, b)| a == b).count();
        let pf = ctx as f64 / prefill_s;
        let tg = (generate - 1) as f64 / decode_s;
        println!(
            "ctx {ctx:>6}: prefill {pf:>7.0} tok/s ({prefill_s:.2}s), decode {tg:>6.1} tok/s ({:.2} tokens/step); corpus continuation {copied}/{generate}; {:?}",
            (out.len() - 1) as f64 / steps.max(1) as f64,
            chat.decode(&out[..out.len().min(24)], false)?
        );
        if let Some(path) = json {
            use std::io::Write;
            let rec = serde_json::json!({"engine": "quench6k", "model": "inkling-2-mini", "mode": if draft { "dspark" } else { "plain" }, "ctx": ctx,
                "generate": generate, "prefill_tok_s": (pf * 10.0).round() / 10.0, "decode_tok_s": (tg * 10.0).round() / 10.0,
                "copied": copied});
            writeln!(std::fs::OpenOptions::new().create(true).append(true).open(path)?, "{rec}")?;
        }
    }
    Ok(())
}
