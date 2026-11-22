//! Commands for Recursive Flint checkpoints (the vendor's NVFP4 GGUF).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use clap::Subcommand;
use quench6k_engine::flint::{self as fl, FlintModel, FlintState, Workspace};

use crate::chat::Chat;

#[derive(Subcommand)]
pub enum FlintCmd {
    /// Greedy generation from a chat prompt (or raw text), printing token ids and timings.
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        /// Tokenize the prompt as is instead of applying the chat template.
        #[arg(long)]
        raw: bool,
        #[arg(long, default_value_t = 64)]
        n: usize,
        #[arg(long, default_value_t = 32768)]
        context: usize,
    },
    /// Perplexity of a text file as llama-perplexity computes it: chunks of `ctx` tokens, each scored
    /// on its second half.
    Ppl {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value_t = 2048)]
        ctx: usize,
        #[arg(long, default_value_t = 8)]
        chunks: usize,
    },
    /// Prefill and greedy decode throughput at each context length, on a text corpus tokenized with
    /// Flint's tokenizer (the first `ctx` tokens). Checks graph decoding against eager decoding at the
    /// first context and reports how often greedy decoding reproduces the corpus continuation.
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: PathBuf,
        #[arg(long, num_args = 1.., default_values_t = [512usize, 8192, 65536, 131072, 260000])]
        ctx: Vec<usize>,
        #[arg(long, default_value_t = 128)]
        generate: usize,
        /// Append one JSON record per context to this file.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Write the prefill's and every decode step's next token and logits hash (FNV-1a of the
        /// FP32 row) to this file, one line per step: comparing two builds bitwise.
        #[arg(long)]
        dump: Option<PathBuf>,
        /// Also time steps of 1..=8 tokens of the sequence (speculative verifications) at each context.
        #[arg(long)]
        widths: bool,
    },
    /// Batched decoding check and throughput: `seqs` sequences (different prompts and lengths from the
    /// text) decode `steps` greedy tokens alone (one token per step), then all together in batched
    /// steps, then with speculative steps of mixed widths (prompt-lookup and continuation drafts, some
    /// corrupted); every row's logits must be bitwise those of decoding alone, and the speculative
    /// tokens those of plain greedy decoding. Then times batched decode steps of 1, 2, 4, .. sequences.
    BatchCheck {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: PathBuf,
        #[arg(long, default_value_t = 8)]
        seqs: usize,
        /// Prompt length of the first sequence (the others are shorter by 97 tokens each), per run.
        #[arg(long, num_args = 1.., default_values_t = [2048usize])]
        ctx: Vec<usize>,
        #[arg(long, default_value_t = 48)]
        steps: usize,
        /// Timed batched decode steps per batch size (0: no timing).
        #[arg(long, default_value_t = 64)]
        time_steps: usize,
        /// Batch sizes to time.
        #[arg(long, num_args = 1.., default_values_t = [1usize, 2, 4, 8])]
        batches: Vec<usize>,
        /// Run everything (from fresh states and prefills) this many times; every run's logits and
        /// tokens must be the same (determinism).
        #[arg(long, default_value_t = 1)]
        repeat: usize,
    },
}

pub fn run(cmd: FlintCmd) -> Result<()> {
    match cmd {
        FlintCmd::Generate {
            model,
            prompt,
            raw,
            n,
            context,
        } => generate(&model, &prompt, raw, n, context),
        FlintCmd::Ppl {
            model,
            file,
            ctx,
            chunks,
        } => ppl(&model, &file, ctx, chunks),
        FlintCmd::Bench {
            model,
            text,
            ctx,
            generate,
            json,
            dump,
            widths,
        } => bench(
            &model,
            &text,
            &ctx,
            generate,
            json.as_deref(),
            dump.as_deref(),
            widths,
        ),
        FlintCmd::BatchCheck {
            model,
            text,
            seqs,
            ctx,
            steps,
            time_steps,
            batches,
            repeat,
        } => batch_check(&model, &text, seqs, &ctx, steps, time_steps, &batches, repeat),
    }
}

fn bench(
    dir: &Path,
    text: &Path,
    ctxs: &[usize],
    generate: usize,
    json: Option<&Path>,
    dump: Option<&Path>,
    widths: bool,
) -> Result<()> {
    let chat = Chat::load(dir, None)?;
    let ids = chat.encode(&std::fs::read_to_string(text)?)?;
    let (model, s) = load(dir)?;
    let mut ws = Workspace::new(if widths { fl::MAX_STEP } else { 1 })?;
    let mut dump = match dump {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => None,
    };
    for (i, &ctx) in ctxs.iter().enumerate() {
        anyhow::ensure!(
            ctx + generate <= ids.len(),
            "corpus has {} tokens, {} needed",
            ids.len(),
            ctx + generate
        );
        let mut seq = FlintState::new(&model, ctx + generate + 8, &s)?;
        s.synchronize()?;
        let t0 = Instant::now();
        fl::prefill(&model, &ws, &mut seq, &ids[..ctx], &s)?;
        let mut next = fl::argmax(&ws, 1, &s)?[0];
        let prefill = t0.elapsed().as_secs_f64();
        let mut out = vec![next];
        let mut hashes = Vec::new();
        if dump.is_some() {
            hashes.push(logits_hash(&s, ws.logits, 1)?[0]);
        }
        let t1 = Instant::now();
        for _ in 1..generate {
            fl::decode(&model, &mut ws, &mut seq, next, &s)?;
            next = fl::argmax(&ws, 1, &s)?[0];
            out.push(next);
            if dump.is_some() {
                hashes.push(logits_hash(&s, ws.logits, 1)?[0]);
            }
        }
        let decode = t1.elapsed().as_secs_f64();
        if let Some(f) = dump.as_mut() {
            use std::io::Write;
            for (i, (t, h)) in out.iter().zip(&hashes).enumerate() {
                writeln!(f, "ctx {ctx} step {i} token {t} logits {h:016x}")?;
            }
            f.flush()?;
        }
        let copied = out.iter().zip(&ids[ctx..]).filter(|(a, b)| a == b).count();
        if i == 0 {
            // Eager decoding of the same prompt must give the same tokens.
            let mut eager = FlintState::new(&model, ctx + generate + 8, &s)?;
            fl::prefill(&model, &ws, &mut eager, &ids[..ctx], &s)?;
            let mut e = vec![fl::argmax(&ws, 1, &s)?[0]];
            while e.len() < out.len().min(64) {
                fl::forward(&model, &ws, &mut eager, &[*e.last().unwrap()], false, &s)?;
                e.push(fl::argmax(&ws, 1, &s)?[0]);
            }
            anyhow::ensure!(
                e[..] == out[..e.len()],
                "graph decoding differs from eager decoding: {e:?} vs {:?}",
                &out[..e.len()]
            );
            println!("graph decoding matches eager decoding ({} tokens)", e.len());
        }
        if widths {
            // Steps of w tokens (the greedy continuation), each undone again.
            let mut line = String::new();
            for w in 1..=fl::MAX_STEP.min(out.len()) {
                let reps = 16;
                let mut tt = 0.0;
                for r in 0..reps + 1 {
                    let t2 = Instant::now();
                    let before = seq.tokens.len();
                    fl::step(&model, &mut ws, &mut seq, &out[..w], &s)?;
                    fl::argmax(&ws, w, &s)?;
                    seq.tokens.truncate(before);
                    if r > 0 {
                        tt += t2.elapsed().as_secs_f64();
                    }
                }
                let ms = tt * 1e3 / reps as f64;
                line.push_str(&format!(" {w}:{ms:.2}"));
            }
            println!("ctx {ctx:>6}: ms per step by tokens{line}");
        }
        let pf = ctx as f64 / prefill;
        let tg = (generate - 1) as f64 / decode;
        println!(
            "ctx {ctx:>6}: prefill {pf:>7.0} tok/s ({prefill:.2}s), decode {tg:>6.1} tok/s; corpus continuation {copied}/{generate}; {:?}",
            chat.decode(&out[..out.len().min(24)], false)?
        );
        if let Some(path) = json {
            use std::io::Write;
            let rec = serde_json::json!({"engine": "quench6k", "model": "flint-nvfp4", "mode": "plain", "kv": if seq.fp8 { "fp8" } else { "bf16" }, "ctx": ctx, "generate": generate,
                "prefill_tok_s": (pf * 10.0).round() / 10.0, "decode_tok_s": (tg * 10.0).round() / 10.0, "copied": copied});
            writeln!(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)?,
                "{rec}"
            )?;
        }
    }
    Ok(())
}

fn batch_check(
    dir: &Path,
    text: &Path,
    seqs: usize,
    ctxs: &[usize],
    steps: usize,
    time_steps: usize,
    batches: &[usize],
    repeat: usize,
) -> Result<()> {
    anyhow::ensure!(
        (1..=fl::MAX_SEQS).contains(&seqs),
        "1..={} sequences",
        fl::MAX_SEQS
    );
    let chat = Chat::load(dir, None)?;
    let ids = chat.encode(&std::fs::read_to_string(text)?)?;
    let (model, s) = load(dir)?;
    let mut ws = Workspace::new(fl::MAX_ROWS)?;
    let mut failures = 0usize;
    let mut digests: Vec<(usize, u64)> = Vec::new();
    for rep_i in 0..repeat.max(1) {
    for &ctx in ctxs {
        let time_steps = if rep_i == 0 { time_steps } else { 0 };
        let lens: Vec<usize> = (0..seqs)
            .map(|i| ctx.saturating_sub(97 * i).max(ctx / 2).max(1))
            .collect();
        anyhow::ensure!(ids.len() > ctx + 1, "the text has {} tokens", ids.len());
        let prompts: Vec<&[i32]> = lens
            .iter()
            .enumerate()
            .map(|(i, &len)| {
                let off = (i * 104_729) % (ids.len() - len);
                &ids[off..off + len]
            })
            .collect();
        let mut states = Vec::new();
        let mut first = Vec::new();
        let t0 = Instant::now();
        for p in &prompts {
            let mut st = FlintState::new(&model, p.len() + steps.max(time_steps + 2) + 2 * fl::MAX_STEP, &s)?;
            fl::prefill(&model, &ws, &mut st, p, &s)?;
            first.push(fl::argmax(&ws, 1, &s)?[0]);
            states.push(st);
        }
        println!(
            "ctx {ctx}: {seqs} sequences of {}..{} prompt tokens prefilled in {:.1}s",
            lens.iter().min().unwrap(),
            lens.iter().max().unwrap(),
            t0.elapsed().as_secs_f64()
        );
        // A: each sequence alone, one token per step. alone[i][j]: logits hash after feeding token j
        // (toks[i][j]); toks[i][j + 1] is its greedy successor.
        let mut alone: Vec<Vec<u64>> = Vec::new();
        let mut toks: Vec<Vec<i32>> = Vec::new();
        for (st, &f) in states.iter_mut().zip(&first) {
            let (mut h, mut t) = (Vec::new(), vec![f]);
            for j in 0..steps {
                fl::step(&model, &mut ws, st, &[t[j]], &s)?;
                // The greedy pick right after the step (as a server samples), then the logits.
                t.push(fl::argmax(&ws, 1, &s)?[0]);
                h.push(logits_hash(&s, ws.logits, 1)?[0]);
            }
            alone.push(h);
            toks.push(t);
        }
        // B: all sequences together, one token each per step.
        for (st, p) in states.iter_mut().zip(&prompts) {
            st.trim(p.len());
        }
        let (mut equal, mut total) = (0usize, 0usize);
        for j in 0..steps {
            let feed: Vec<[i32; 1]> = toks.iter().map(|t| [t[j]]).collect();
            let feed: Vec<&[i32]> = feed.iter().map(|f| &f[..]).collect();
            let mut refs: Vec<&mut FlintState> = states.iter_mut().collect();
            model.forward_batch(&mut refs, &mut ws, &s, &feed)?;
            let next = fl::argmax(&ws, seqs, &s)?;
            let h = logits_hash(&s, ws.logits, seqs)?;
            for i in 0..seqs {
                total += 1;
                equal += (h[i] == alone[i][j] && next[i] == toks[i][j + 1]) as usize;
            }
        }
        failures += total - equal;
        let digest = alone
            .iter()
            .flatten()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
                (h ^ x).wrapping_mul(0x100_0000_01b3)
            });
        println!("  decoding alone: logits digest {digest:016x}");
        println!("  batched plain decode: {equal}/{total} rows bitwise equal to decoding alone");
        // C: speculative steps of mixed widths; sequences leave the batch as they finish.
        for (st, p) in states.iter_mut().zip(&prompts) {
            st.trim(p.len());
        }
        let mut at = vec![0usize; seqs];
        let mut out: Vec<Vec<i32>> = first.iter().map(|&f| vec![f]).collect();
        let (mut equal, mut total, mut rounds, mut fed_rows) = (0usize, 0usize, 0usize, 0usize);
        let mut spec_hashes: Vec<u64> = Vec::new();
        let mut widths = [0usize; fl::MAX_STEP + 1];
        while at.iter().any(|&a| a < steps) {
            let live: Vec<usize> = (0..seqs).filter(|&i| at[i] < steps).collect();
            let mut feeds = Vec::new();
            for &i in &live {
                let w = (1 + (rounds * 3 + i * 5) % fl::MAX_STEP).min(steps - at[i]);
                let pending = *out[i].last().unwrap();
                let mut f = vec![pending];
                if rounds % 2 == 1 {
                    let mut hist = prompts[i].to_vec();
                    hist.extend_from_slice(&out[i]);
                    f.extend(crate::flint::lookup_draft(&hist, w - 1));
                } else {
                    f.extend_from_slice(&toks[i][at[i] + 1..at[i] + w]);
                    if (rounds + i) % 3 == 0 && w > 1 {
                        let k = 1 + (rounds + i) % (w - 1);
                        f[k] = f[k].wrapping_add(1) % fl::VOCAB as i32;
                    }
                }
                widths[f.len()] += 1;
                feeds.push(f);
            }
            let before: Vec<usize> = live.iter().map(|&i| states[i].pos()).collect();
            {
                let feed: Vec<&[i32]> = feeds.iter().map(|f| &f[..]).collect();
                let mut refs: Vec<&mut FlintState> = states
                    .iter_mut()
                    .enumerate()
                    .filter(|(i, _)| live.contains(i))
                    .map(|(_, st)| st)
                    .collect();
                model.forward_batch(&mut refs, &mut ws, &s, &feed)?;
            }
            let rows: usize = feeds.iter().map(|f| f.len()).sum();
            fed_rows += rows;
            let next = fl::argmax(&ws, rows, &s)?;
            let h = logits_hash(&s, ws.logits, rows)?;
            spec_hashes.extend_from_slice(&h);
            let mut r = 0;
            for (k, &i) in live.iter().enumerate() {
                let f = &feeds[k];
                // Rows whose fed tokens all follow the plain greedy sequence compare bitwise.
                let valid = f
                    .iter()
                    .enumerate()
                    .take_while(|&(j, &t)| toks[i][at[i] + j] == t)
                    .count();
                for j in 0..valid {
                    total += 1;
                    let same = h[r + j] == alone[i][at[i] + j];
                    equal += same as usize;
                    if !same && total - equal <= 12 {
                        println!(
                            "    differs: sequence {i} position {} (row {j} of {}, prompt {})",
                            before[k] + j,
                            f.len(),
                            prompts[i].len()
                        );
                    }
                }
                // Accept as the server does: each row's token while it equals the next draft.
                let mut j = 0;
                loop {
                    let t = next[r + j];
                    out[i].push(t);
                    if j + 1 < f.len() && t == f[j + 1] && at[i] + j + 1 < steps {
                        j += 1;
                        continue;
                    }
                    break;
                }
                at[i] += j + 1;
                states[i].truncate(before[k] + j + 1);
                r += f.len();
            }
            rounds += 1;
        }
        failures += total - equal;
        let same = (0..seqs)
            .filter(|&i| out[i][..=steps] == toks[i][..=steps])
            .count();
        // This run's digest: decoding alone, then the speculative rows' logits and tokens.
        let run_digest = std::iter::once(digest)
            .chain(spec_hashes.iter().copied())
            .chain(out.iter().flatten().map(|&t| t as u64))
            .fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ x).wrapping_mul(0x100_0000_01b3));
        if repeat > 1 {
            println!("  run {} digest {run_digest:016x}", rep_i + 1);
        }
        match digests.iter().find(|(c, _)| *c == ctx) {
            Some(&(_, d)) if d != run_digest => {
                failures += 1;
                println!("  run {}: differs from run 1", rep_i + 1);
            }
            Some(_) => {}
            None => digests.push((ctx, run_digest)),
        }
        failures += seqs - same;
        println!(
            "  batched speculative steps: {equal}/{total} rows bitwise equal to decoding alone; tokens equal to plain greedy decoding in {same}/{seqs} sequences ({rounds} steps, {fed_rows} rows, widths 1..8: {:?})",
            &widths[1..]
        );
        // D: throughput of batched decode steps.
        if time_steps > 0 {
            for &n in batches.iter().filter(|&&n| n >= 1 && n <= seqs) {
                for (st, p) in states.iter_mut().zip(&prompts) {
                    st.trim(p.len());
                }
                let mut last: Vec<i32> = first[..n].to_vec();
                let mut timed = 0.0;
                for j in 0..time_steps + 2 {
                    let feed: Vec<[i32; 1]> = last.iter().map(|&t| [t]).collect();
                    let feed: Vec<&[i32]> = feed.iter().map(|f| &f[..]).collect();
                    let t1 = Instant::now();
                    let mut refs: Vec<&mut FlintState> = states[..n].iter_mut().collect();
                    model.forward_batch(&mut refs, &mut ws, &s, &feed)?;
                    last = fl::argmax(&ws, n, &s)?;
                    if j >= 2 {
                        timed += t1.elapsed().as_secs_f64();
                    }
                }
                let per = timed / time_steps as f64;
                println!(
                    "  {n} sequence(s): {:.2} ms per step, {:.1} tok/s per sequence, {:.1} tok/s together",
                    per * 1e3,
                    1.0 / per,
                    n as f64 / per
                );
            }
        }
    }
    }
    anyhow::ensure!(failures == 0, "{failures} mismatches");
    println!("batch check passed");
    Ok(())
}

/// FNV-1a hash of each of `rows` FP32 logits rows at `logits` (bitwise comparisons between builds).
pub fn logits_hash(s: &quench6k_cuda::Stream, logits: u64, rows: usize) -> Result<Vec<u64>> {
    let v = s.download::<u32>(logits, rows * fl::VOCAB)?;
    Ok(v.chunks(fl::VOCAB)
        .map(|row| {
            row.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
                x.to_le_bytes()
                    .iter()
                    .fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
            })
        })
        .collect())
}

fn ppl(dir: &Path, file: &Path, ctx: usize, chunks: usize) -> Result<()> {
    let chat = Chat::load(dir, None)?;
    let ids = chat.encode(&std::fs::read_to_string(file)?)?;
    let n_chunks = (ids.len() / ctx).min(chunks);
    anyhow::ensure!(
        n_chunks > 0,
        "{} tokens: fewer than one chunk of {ctx}",
        ids.len()
    );
    let (model, s) = load(dir)?;
    let half = ctx / 2;
    let piece = fl::MAX_CHUNK.min(half);
    let ws = Workspace::new(piece)?;
    let lse = quench6k_cuda::DeviceBuffer::new(piece * 4)?;
    let tl = quench6k_cuda::DeviceBuffer::new(piece * 4)?;
    let targets = quench6k_cuda::DeviceBuffer::new(piece * 4)?;
    let (mut nll, mut count) = (0f64, 0usize);
    let t0 = Instant::now();
    for c in 0..n_chunks {
        let toks = &ids[c * ctx..(c + 1) * ctx];
        let mut seq = FlintState::new(&model, ctx, &s)?;
        // Logits of positions half .. ctx-2 predict the next token (llama-perplexity's first = n_ctx / 2).
        fl::prefill(&model, &ws, &mut seq, &toks[..half], &s)?;
        let mut at = half;
        while at < ctx - 1 {
            let rows = piece.min(ctx - 1 - at);
            fl::forward(&model, &ws, &mut seq, &toks[at..at + rows], true, &s)?;
            s.upload(&targets, &toks[at + 1..at + 1 + rows])?;
            quench6k_cuda::kernels::row_lse(
                &s,
                ws.logits,
                rows,
                fl::VOCAB,
                fl::VOCAB,
                targets.addr(),
                lse.addr(),
                tl.addr(),
            )?;
            let a = s.download::<f32>(lse.addr(), rows)?;
            let b = s.download::<f32>(tl.addr(), rows)?;
            for (x, y) in a.iter().zip(&b) {
                nll += (x - y) as f64;
                count += 1;
            }
            at += rows;
        }
        println!("[{}] {:.4}", c + 1, (nll / count as f64).exp());
    }
    println!(
        "PPL over {n_chunks} chunks of {ctx}: {:.4} ({:.1}s)",
        (nll / count as f64).exp(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

pub fn load(dir: &Path) -> Result<(FlintModel, quench6k_cuda::Stream)> {
    quench6k_cuda::set_device(0)?;
    let stream = quench6k_cuda::Stream::new()?;
    let model = FlintModel::load(dir, &stream)?;
    Ok((model, stream))
}

fn generate(dir: &Path, prompt: &str, raw: bool, n: usize, context: usize) -> Result<()> {
    let chat = Chat::load(dir, None)?;
    let text = if raw {
        prompt.to_string()
    } else {
        chat.render(
            &serde_json::json!([{"role": "user", "content": prompt}]),
            None,
            None,
            None,
        )?
    };
    let ids = chat.encode(&text)?;
    println!("prompt: {} tokens", ids.len());
    let (model, s) = load(dir)?;
    let ws = Workspace::new(1)?;
    let mut seq = FlintState::new(&model, context, &s)?;
    let t0 = Instant::now();
    fl::prefill(&model, &ws, &mut seq, &ids, &s)?;
    let mut next = fl::argmax(&ws, 1, &s)?[0];
    let prefill = t0.elapsed().as_secs_f64();
    {
        let (idx, val) = (
            quench6k_cuda::DeviceBuffer::new(32)?,
            quench6k_cuda::DeviceBuffer::new(32)?,
        );
        quench6k_cuda::kernels::topk(
            &s,
            ws.logits,
            1,
            fl::VOCAB,
            fl::VOCAB,
            5,
            false,
            idx.addr(),
            val.addr(),
        )?;
        let (i, v) = (
            s.download::<i32>(idx.addr(), 5)?,
            s.download::<f32>(val.addr(), 5)?,
        );
        let top: Vec<String> = i
            .iter()
            .zip(&v)
            .map(|(t, l)| format!("{:?} {l:.3}", chat.decode(&[*t], false).unwrap_or_default()))
            .collect();
        println!("first-token top 5: {}", top.join(", "));
    }
    let mut out = vec![next];
    let t1 = Instant::now();
    for _ in 1..n {
        if model.stop_tokens.contains(&next) {
            break;
        }
        fl::forward(&model, &ws, &mut seq, &[next], false, &s)?;
        next = fl::argmax(&ws, 1, &s)?[0];
        out.push(next);
    }
    let decode = t1.elapsed().as_secs_f64();
    println!("ids: {out:?}");
    println!("text: {:?}", chat.decode(&out, false)?);
    println!(
        "prefill {} tokens in {:.3}s ({:.0} tok/s); decode {} tokens in {:.3}s ({:.1} tok/s)",
        ids.len(),
        prefill,
        ids.len() as f64 / prefill,
        out.len() - 1,
        decode,
        (out.len() - 1) as f64 / decode.max(1e-9)
    );
    Ok(())
}

/// Host-side sampling: the top candidates of up to MAX_ROWS logits rows come from the device in one
/// kernel and one download ([`Sampler::fetch`]); penalties, temperature, top-k, min-p and top-p apply to
/// them per row ([`Sampler::pick`]).
pub struct Sampler {
    /// Candidate ids [rows, pool] (i32) followed by their logits [rows, pool] (f32).
    buf: quench6k_cuda::DeviceBuffer,
    ids: Vec<i32>,
    vals: Vec<f32>,
    pool: usize,
    state: u64,
    seed: Option<u64>,
}

const POOL: usize = 256;
const MAX_ROWS: usize = 8;

impl Sampler {
    pub fn new() -> Result<Self> {
        Ok(Self {
            buf: quench6k_cuda::DeviceBuffer::new(MAX_ROWS * POOL * 8)?,
            ids: Vec::new(),
            vals: Vec::new(),
            pool: 1,
            state: 0,
            seed: None,
        })
    }

    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The candidates of `rows` consecutive logits rows at `logits` (n columns each).
    pub fn fetch(
        &mut self,
        s: &quench6k_cuda::Stream,
        logits: u64,
        rows: usize,
        n: usize,
        p: &quench6k_engine::session::GenParams,
    ) -> Result<()> {
        anyhow::ensure!(rows <= MAX_ROWS, "{rows} rows to sample");
        let penalized = p.presence_penalty != 0.0 || p.repetition_penalty != 1.0;
        let greedy = p.temperature <= 0.0 || p.top_k == 1;
        let pool = if greedy && !penalized {
            1
        } else if p.top_k == 0 {
            POOL
        } else {
            p.top_k.max(if penalized { 64 } else { 1 }).min(POOL)
        };
        let (idx, val) = (self.buf.addr(), self.buf.addr() + (rows * pool * 4) as u64);
        quench6k_cuda::kernels::topk(s, logits, rows, n, n, pool, false, idx, val)?;
        let raw = s.download::<u32>(idx, 2 * rows * pool)?;
        self.ids = raw[..rows * pool].iter().map(|&v| v as i32).collect();
        self.vals = raw[rows * pool..]
            .iter()
            .map(|&v| f32::from_bits(v))
            .collect();
        self.pool = pool;
        Ok(())
    }

    /// The next token from fetched row `row`; `generated`: this request's output so far.
    pub fn pick(
        &mut self,
        row: usize,
        p: &quench6k_engine::session::GenParams,
        generated: &[i32],
    ) -> i32 {
        if generated.is_empty() || self.seed != Some(p.seed) {
            self.seed = Some(p.seed);
            self.state = p.seed;
        }
        let pool = self.pool;
        let ids = &self.ids[row * pool..(row + 1) * pool];
        let mut vals = self.vals[row * pool..(row + 1) * pool].to_vec();
        let penalized = p.presence_penalty != 0.0 || p.repetition_penalty != 1.0;
        if penalized {
            for (v, id) in vals.iter_mut().zip(ids) {
                if generated.contains(id) {
                    *v -= p.presence_penalty;
                    *v = if *v > 0.0 {
                        *v / p.repetition_penalty
                    } else {
                        *v * p.repetition_penalty
                    };
                }
            }
        }
        let mut order: Vec<usize> = (0..pool).collect();
        order.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]).then(ids[a].cmp(&ids[b])));
        if p.temperature <= 0.0 || p.top_k == 1 {
            return ids[order[0]];
        }
        let keep = if p.top_k == 0 {
            pool
        } else {
            p.top_k.min(pool)
        };
        let order = &order[..keep];
        let top = vals[order[0]];
        let mut probs: Vec<f64> = order
            .iter()
            .map(|&i| (((vals[i] - top) / p.temperature) as f64).exp())
            .collect();
        if p.min_p > 0.0 {
            for q in probs.iter_mut() {
                if *q < p.min_p as f64 {
                    *q = 0.0;
                }
            }
        }
        let total: f64 = probs.iter().sum();
        let mut cut = probs.len();
        if p.top_p < 1.0 {
            let mut acc = 0.0;
            for (i, q) in probs.iter().enumerate() {
                acc += q / total;
                if acc >= p.top_p as f64 {
                    cut = i + 1;
                    break;
                }
            }
        }
        let total: f64 = probs[..cut].iter().sum();
        let ids: Vec<i32> = order.iter().map(|&i| ids[i]).collect();
        let mut r = self.uniform() * total;
        for (i, q) in probs[..cut].iter().enumerate() {
            r -= q;
            if r <= 0.0 {
                return ids[i];
            }
        }
        ids[cut - 1]
    }
}

/// A prompt-lookup draft: up to `k` tokens that followed the latest earlier occurrence of the last 4,
/// 3 or 2 tokens of `hist` (searched within the last 128K positions).
pub fn lookup_draft(hist: &[i32], k: usize) -> Vec<i32> {
    if k == 0 {
        return Vec::new();
    }
    let len = hist.len();
    for n in [4usize, 3] {
        if len <= n {
            continue;
        }
        let pat = &hist[len - n..];
        let lo = len.saturating_sub(1 << 17);
        let mut i = len - n;
        while i > lo {
            i -= 1;
            if hist[i] == pat[0] && &hist[i..i + n] == pat {
                let start = i + n;
                let end = (start + k).min(len);
                if end > start {
                    return hist[start..end].to_vec();
                }
            }
        }
    }
    Vec::new()
}
