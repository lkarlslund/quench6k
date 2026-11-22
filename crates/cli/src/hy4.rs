//! Commands for Hy4-flash checkpoints (`hy_v4_flash`).

mod batch;

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
        /// Greedy decoding with the checkpoint's DFly drafter.
        #[arg(long)]
        dfly: bool,
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
        /// Greedy decoding with the checkpoint's DFly drafter.
        #[arg(long)]
        dfly: bool,
    },
    /// Width invariance: a fixture's tokens teacher-forced in steps of each width after a common prefill;
    /// every position's logits row compared bitwise with the one-token steps' row.
    WidthCheck {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        fixture: PathBuf,
        /// Positions prefilled before the steps.
        #[arg(long, default_value_t = 2040)]
        prefill: usize,
        /// Positions run in steps after the prefill.
        #[arg(long, default_value_t = 240)]
        steps: usize,
        #[arg(long, num_args = 1.., default_values_t = [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16])]
        widths: Vec<usize>,
    },
    /// Several sequences (fixtures at different prompt lengths) stepped alone and batched: teacher-forced
    /// logits rows and caches, and greedy decoding with drafts (tokens, drafts, logits) compared bitwise;
    /// then (with --bench-ctx) the aggregate decode throughput of 1, 2, 4 and 8 batched sequences.
    BatchCheck {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, num_args = 1..)]
        fixture: Vec<PathBuf>,
        #[arg(long, default_value_t = 8)]
        seqs: usize,
        /// Teacher-forced positions per sequence (0: skip).
        #[arg(long, default_value_t = 40)]
        steps: usize,
        /// Greedy tokens per sequence (0: skip).
        #[arg(long, default_value_t = 48)]
        decode: usize,
        /// Text for the throughput prompts (default: the longest fixture's tokens).
        #[arg(long)]
        text: Option<PathBuf>,
        #[arg(long, num_args = 0..)]
        bench_ctx: Vec<usize>,
        #[arg(long, default_value_t = 32)]
        bench_steps: usize,
        /// Drafts per sequence of the drafted throughput runs.
        #[arg(long, num_args = 0.., default_values_t = [3usize])]
        bench_drafts: Vec<usize>,
    },
    /// Canonical prefill: multi-turn conversations sharing a system prompt computed cold (a fresh state
    /// per turn), warm (reusing the conversation's previous turn) and cross (reusing prefixes cached by
    /// other conversations, interleaved and batched with another sequence); every turn's logits rows and
    /// greedy tokens must be bit-identical. Prints reuse points and extra prefill per turn.
    CanonCheck {
        #[arg(long)]
        model: PathBuf,
        /// Text files (concatenated) for the prompts.
        #[arg(long, num_args = 1..)]
        text: Vec<PathBuf>,
        /// Shared system prompt tokens.
        #[arg(long, default_value_t = 3109)]
        system: usize,
        #[arg(long, default_value_t = 3)]
        convs: usize,
        #[arg(long, default_value_t = 3)]
        turns: usize,
        /// Reply tokens per turn.
        #[arg(long, default_value_t = 12)]
        reply: usize,
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
            dfly,
        } => generate(&model, &prompt, raw, &effort, n, context, chunk, dfly),
        Hy4Cmd::Bench {
            model,
            text,
            ctx,
            generate,
            chunk,
            json,
            dfly,
        } => bench(&model, &text, &ctx, generate, chunk, json.as_deref(), dfly),
        Hy4Cmd::WidthCheck {
            model,
            fixture,
            prefill,
            steps,
            widths,
        } => width_check(&model, &fixture, prefill, steps, &widths),
        Hy4Cmd::BatchCheck {
            model,
            fixture,
            seqs,
            steps,
            decode,
            text,
            bench_ctx,
            bench_steps,
            bench_drafts,
        } => batch::run(&model, &fixture, seqs, steps, decode, text.as_deref(), &bench_ctx, bench_steps, &bench_drafts),
        Hy4Cmd::CanonCheck {
            model,
            text,
            system,
            convs,
            turns,
            reply,
        } => canon_check(&model, &text, system, convs, turns, reply),
    }
}

fn fixture_tokens(fixture: &Path) -> Result<Vec<i32>> {
    let seq: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixture.join("sequence.json")).with_context(|| format!("{}/sequence.json", fixture.display()))?,
    )?;
    Ok(seq["tokens"]
        .as_array()
        .context("sequence.json: tokens")?
        .iter()
        .map(|v| v.as_i64().unwrap_or(0) as i32)
        .collect())
}

/// Hashes of `rows` logits rows at `addr` (F32 [rows, vocab]).
fn row_hashes(s: &quench6k_cuda::Stream, addr: u64, rows: usize, vocab: usize) -> Result<Vec<u64>> {
    use std::hash::{Hash, Hasher};
    let v: Vec<u32> = s.download(addr, rows * vocab)?;
    Ok(v.chunks_exact(vocab)
        .map(|r| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            r.hash(&mut h);
            h.finish()
        })
        .collect())
}

fn width_check(dir: &Path, fixture: &Path, prefill_n: usize, steps: usize, widths: &[usize]) -> Result<()> {
    let tokens = fixture_tokens(fixture)?;
    ensure!(prefill_n + steps <= tokens.len(), "fixture has {} tokens", tokens.len());
    let (model, s) = load(dir)?;
    let (free, total) = quench6k_cuda::mem_info()?;
    eprintln!(
        "device memory after load: {:.2} GiB free of {:.2} GiB; weights {:.2} GiB; mapping granularity {} KiB",
        free as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64,
        model.device_bytes as f64 / (1u64 << 30) as f64,
        quench6k_cuda::vm_granularity()? >> 10
    );
    let vocab = model.cfg.vocab;
    let ctx = prefill_n + steps + 64;
    let mut ws = Workspace::new(&model, 2048, ctx, 16)?;
    let mut base: Option<Vec<u64>> = None;
    for &w in widths {
        let mut st = State::new(&model, ctx, &s)?;
        prefill(&model, &mut st, &mut ws, &s, &tokens[..prefill_n])?;
        let mut hashes = Vec::with_capacity(steps);
        let mut at = prefill_n;
        let t0 = Instant::now();
        while at < prefill_n + steps {
            let n = w.min(prefill_n + steps - at);
            model.forward(&mut st, &mut ws, &s, &tokens[at..at + n], Logits::All)?;
            hashes.extend(row_hashes(&s, ws.out_logits, n, vocab)?);
            at += n;
        }
        let secs = t0.elapsed().as_secs_f64();
        match &base {
            None => {
                println!("width {w:>2}: reference ({steps} positions, {:.1} ms per step)", secs * 1e3 / steps.div_ceil(w) as f64);
                base = Some(hashes);
            }
            Some(b) => {
                let diff: Vec<usize> = (0..steps).filter(|&i| b[i] != hashes[i]).collect();
                println!(
                    "width {w:>2}: {} of {steps} positions differ{}",
                    diff.len(),
                    diff.first().map_or(String::new(), |f| format!(" (first at position {})", prefill_n + f))
                );
            }
        }
    }
    Ok(())
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

/// Drafted-and-verified greedy decoding after a prefill whose last step chose `first`: up to `n` tokens
/// (stopping after any of `stop`). Returns the tokens and the number of verification steps.
fn decode_dfly(
    model: &Hy4Model,
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
    let timing = std::env::var_os("QUENCH6K_HY4_TIMING").is_some();
    let (mut t_draft, mut t_verify) = (0.0f64, 0.0f64);
    let mut hist = [0usize; 17];
    let fixed = std::env::var("QUENCH6K_HY4_DRAFTS").ok().and_then(|v| v.parse::<usize>().ok());
    let mut policy = quench6k_engine::hy4::policy::DraftPolicy::new(
        model.dfly.as_ref().map_or(0, |f| f.width - 1),
        0.010,
        0.0018,
        0.0024,
    );
    while out.len() < n && !stop(pending) {
        let t0 = Instant::now();
        let want_drafts = fixed.unwrap_or_else(|| policy.drafts());
        let mut drafts = if want_drafts > 0 { model.draft(st, ws, s, pending, row)? } else { Vec::new() };
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
        st.truncate(before + 1 + j);
        row = j;
        hist[j.min(16)] += 1;
        policy.update(drafts.len(), j, verify, (!drafts.is_empty()).then(|| (t1 - t0).as_secs_f64()));
    }
    if timing {
        eprintln!("dfly: accepted drafts per step (0, 1, ...): {:?}", &hist[..8]);
        eprintln!(
            "dfly: {steps} steps, draft {:.2} ms, verify {:.2} ms per step",
            t_draft * 1e3 / steps.max(1) as f64,
            t_verify * 1e3 / steps.max(1) as f64
        );
    }
    Ok((out, steps))
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

#[allow(clippy::too_many_arguments)]
fn generate(dir: &Path, prompt: &str, raw: bool, effort: &str, n: usize, context: usize, chunk: usize, dfly: bool) -> Result<()> {
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
    let mut steps = 0;
    if dfly {
        let stop = |t: i32| model.cfg.eos.contains(&t) || chat.stop_tokens.contains(&t);
        (out, steps) = decode_dfly(&model, &mut st, &mut ws, &s, next, n, &stop)?;
    } else {
        while out.len() < n && !model.cfg.eos.contains(&next) && !chat.stop_tokens.contains(&next) {
            model.forward(&mut st, &mut ws, &s, &[next], Logits::Last)?;
            next = argmax(&ws, &model, &s)?;
            out.push(next);
        }
    }
    let dec = t1.elapsed().as_secs_f64();
    if dfly {
        eprintln!("dfly: {:.2} tokens per step", (out.len() - 1) as f64 / steps.max(1) as f64);
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

fn bench(dir: &Path, text: &Path, ctxs: &[usize], generate: usize, chunk: usize, json: Option<&Path>, dfly: bool) -> Result<()> {
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
        let mut steps = generate - 1;
        if dfly {
            (out, steps) = decode_dfly(&model, &mut st, &mut ws, &s, next, generate, &|_| false)?;
            next = *out.last().unwrap();
        } else {
            for _ in 1..generate {
                model.forward(&mut st, &mut ws, &s, &[next], Logits::Last)?;
                next = argmax(&ws, &model, &s)?;
                out.push(next);
            }
        }
        let decode_s = t1.elapsed().as_secs_f64();
        // QUENCH6K_HY4_PROFILE_STEPS=N: N more decode steps inside a cudaProfilerStart/Stop region (nsys
        // --capture-range=cudaProfilerApi; tools/census.py --rounds N).
        if let Some(n) = std::env::var("QUENCH6K_HY4_PROFILE_STEPS").ok().and_then(|v| v.parse::<usize>().ok())
            && dfly
        {
            // Drafted steps: N tokens' worth; the step count is printed for census --rounds.
            s.synchronize()?;
            quench6k_cuda::profiler_start()?;
            let (_, steps) = decode_dfly(&model, &mut st, &mut ws, &s, next, n, &|_| false)?;
            s.synchronize()?;
            quench6k_cuda::profiler_stop()?;
            eprintln!("profiled {steps} drafted steps");
        } else if let Some(n) = std::env::var("QUENCH6K_HY4_PROFILE_STEPS").ok().and_then(|v| v.parse::<usize>().ok()) {
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
            "ctx {ctx:>6}: prefill {pf:>7.0} tok/s ({prefill_s:.2}s), decode {tg:>6.1} tok/s ({:.2} tokens/step); corpus continuation {copied}/{generate}; {:?}",
            (out.len() - 1) as f64 / steps.max(1) as f64,
            chat.decode(&out[..out.len().min(24)], false)?
        );
        if let Some(path) = json {
            use std::io::Write;
            let rec = serde_json::json!({"engine": "quench6k", "model": "hy4-flash-nvfp4", "mode": if dfly { "dfly" } else { "plain" }, "ctx": ctx,
                "generate": generate, "prefill_tok_s": (pf * 10.0).round() / 10.0, "decode_tok_s": (tg * 10.0).round() / 10.0,
                "copied": copied});
            writeln!(std::fs::OpenOptions::new().create(true).append(true).open(path)?, "{rec}")?;
        }
    }
    Ok(())
}

/// [`crate::canon`]'s view of the Hy4 engine.
struct Canon<'m> {
    model: &'m Hy4Model,
    s: &'m quench6k_cuda::Stream,
    ws: Workspace,
}

impl Canon<'_> {
    fn rows(&self, rows: usize) -> Result<Vec<crate::canon::Row>> {
        let vocab = self.model.cfg.vocab;
        let h = row_hashes(self.s, self.ws.out_logits, rows, vocab)?;
        k::argmax(self.s, self.ws.out_logits, rows, vocab, vocab, self.ws.argmax_scratch, self.ws.argmax)?;
        let t = self.s.download::<i32>(self.ws.argmax, rows)?;
        Ok(h.into_iter().zip(t).collect())
    }
}

impl crate::canon::Engine for Canon<'_> {
    type St = State;
    fn name(&self) -> &'static str {
        "hy4"
    }
    fn chunk(&self) -> usize {
        quench6k_engine::hy4::forward::CANON_CHUNK
    }
    fn ring(&self) -> usize {
        quench6k_engine::hy4::forward::RING
    }
    fn new_state(&mut self, capacity: usize) -> Result<State> {
        State::new(self.model, capacity, self.s)
    }
    fn history<'a>(&self, st: &'a State) -> &'a [i32] {
        &st.history
    }
    fn frontier(&self, st: &State) -> usize {
        st.frontier()
    }
    fn snapshots(&self, st: &State) -> Vec<usize> {
        st.snapshot_boundaries()
    }
    fn rewind(&mut self, st: &mut State, prompt: &[i32]) -> Result<usize> {
        st.rewind(prompt, self.s)
    }
    fn prefill(&mut self, st: &mut State, tokens: &[i32]) -> Result<crate::canon::Row> {
        self.model.prefill(st, &mut self.ws, self.s, tokens)?;
        Ok(self.rows(1)?[0])
    }
    fn feed(&mut self, st: &mut State, tokens: &[i32]) -> Result<()> {
        self.model.forward(st, &mut self.ws, self.s, tokens, Logits::None)
    }
    fn step(&mut self, states: &mut [&mut State], tokens: &[&[i32]]) -> Result<Vec<crate::canon::Row>> {
        self.model.forward_batch(states, &mut self.ws, self.s, tokens)?;
        self.rows(tokens.iter().map(|t| t.len()).sum())
    }
    fn truncate(&mut self, st: &mut State, pos: usize) {
        st.truncate(pos);
    }    fn sync(&self) -> Result<()> {
        self.s.synchronize()
    }
}

fn canon_check(dir: &Path, text: &[PathBuf], system: usize, convs: usize, turns: usize, reply: usize) -> Result<()> {
    let chat = Chat::load(dir, None)?;
    let mut all = String::new();
    for t in text {
        all.push_str(&std::fs::read_to_string(t).with_context(|| t.display().to_string())?);
    }
    let ids = chat.encode(&all)?;
    let (model, s) = load(dir)?;
    let ws = Workspace::new(&model, quench6k_engine::hy4::forward::CANON_CHUNK, 0, 16)?;
    let mut e = Canon { model: &model, s: &s, ws };
    crate::canon::check(&mut e, &ids, system, convs, turns, reply)
}
