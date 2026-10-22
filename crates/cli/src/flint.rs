//! Commands for Recursive Flint checkpoints (the vendor's NVFP4 GGUF).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use clap::Subcommand;
use quench6k_engine::flint::{self as fl, FlintModel, FlintSeq, Workspace};

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
    },
}

pub fn run(cmd: FlintCmd) -> Result<()> {
    match cmd {
        FlintCmd::Generate { model, prompt, raw, n, context } => generate(&model, &prompt, raw, n, context),
        FlintCmd::Ppl { model, file, ctx, chunks } => ppl(&model, &file, ctx, chunks),
        FlintCmd::Bench { model, text, ctx, generate, json } => bench(&model, &text, &ctx, generate, json.as_deref()),
    }
}

fn bench(dir: &Path, text: &Path, ctxs: &[usize], generate: usize, json: Option<&Path>) -> Result<()> {
    let chat = Chat::load(dir, None)?;
    let ids = chat.encode(&std::fs::read_to_string(text)?)?;
    let (model, s) = load(dir)?;
    let mut ws = Workspace::new(1)?;
    for (i, &ctx) in ctxs.iter().enumerate() {
        anyhow::ensure!(ctx + generate <= ids.len(), "corpus has {} tokens, {} needed", ids.len(), ctx + generate);
        let mut seq = FlintSeq::new(&model, ctx + generate + 8)?;
        s.synchronize()?;
        let t0 = Instant::now();
        fl::prefill(&model, &ws, &mut seq, &ids[..ctx], &s)?;
        let mut next = fl::argmax(&ws, 1, &s)?[0];
        let prefill = t0.elapsed().as_secs_f64();
        let mut out = vec![next];
        let t1 = Instant::now();
        for _ in 1..generate {
            fl::decode(&model, &mut ws, &mut seq, next, &s)?;
            next = fl::argmax(&ws, 1, &s)?[0];
            out.push(next);
        }
        let decode = t1.elapsed().as_secs_f64();
        let copied = out.iter().zip(&ids[ctx..]).filter(|(a, b)| a == b).count();
        if i == 0 {
            // Eager decoding of the same prompt must give the same tokens.
            let mut eager = FlintSeq::new(&model, ctx + generate + 8)?;
            fl::prefill(&model, &ws, &mut eager, &ids[..ctx], &s)?;
            let mut e = vec![fl::argmax(&ws, 1, &s)?[0]];
            while e.len() < out.len().min(64) {
                fl::forward(&model, &ws, &mut eager, &[*e.last().unwrap()], false, &s)?;
                e.push(fl::argmax(&ws, 1, &s)?[0]);
            }
            anyhow::ensure!(e[..] == out[..e.len()], "graph decoding differs from eager decoding: {e:?} vs {:?}", &out[..e.len()]);
            println!("graph decoding matches eager decoding ({} tokens)", e.len());
        }
        let pf = ctx as f64 / prefill;
        let tg = (generate - 1) as f64 / decode;
        println!(
            "ctx {ctx:>6}: prefill {pf:>7.0} tok/s ({prefill:.2}s), decode {tg:>6.1} tok/s; corpus continuation {copied}/{generate}; {:?}",
            chat.decode(&out[..out.len().min(24)], false)?
        );
        if let Some(path) = json {
            use std::io::Write;
            let rec = serde_json::json!({"engine": "quench6k", "model": "flint-nvfp4", "mode": "plain", "kv": "bf16", "ctx": ctx, "generate": generate,
                "prefill_tok_s": (pf * 10.0).round() / 10.0, "decode_tok_s": (tg * 10.0).round() / 10.0, "copied": copied});
            writeln!(std::fs::OpenOptions::new().create(true).append(true).open(path)?, "{rec}")?;
        }
    }
    Ok(())
}

fn ppl(dir: &Path, file: &Path, ctx: usize, chunks: usize) -> Result<()> {
    let chat = Chat::load(dir, None)?;
    let ids = chat.encode(&std::fs::read_to_string(file)?)?;
    let n_chunks = (ids.len() / ctx).min(chunks);
    anyhow::ensure!(n_chunks > 0, "{} tokens: fewer than one chunk of {ctx}", ids.len());
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
        let mut seq = FlintSeq::new(&model, ctx)?;
        // Logits of positions half .. ctx-2 predict the next token (llama-perplexity's first = n_ctx / 2).
        fl::prefill(&model, &ws, &mut seq, &toks[..half], &s)?;
        let mut at = half;
        while at < ctx - 1 {
            let rows = piece.min(ctx - 1 - at);
            fl::forward(&model, &ws, &mut seq, &toks[at..at + rows], true, &s)?;
            s.upload(&targets, &toks[at + 1..at + 1 + rows])?;
            quench6k_cuda::kernels::row_lse(&s, ws.logits, rows, fl::VOCAB, fl::VOCAB, targets.addr(), lse.addr(), tl.addr())?;
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
    println!("PPL over {n_chunks} chunks of {ctx}: {:.4} ({:.1}s)", (nll / count as f64).exp(), t0.elapsed().as_secs_f64());
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
    let text = if raw { prompt.to_string() } else { chat.render(&serde_json::json!([{"role": "user", "content": prompt}]), None, None, None)? };
    let ids = chat.encode(&text)?;
    println!("prompt: {} tokens", ids.len());
    let (model, s) = load(dir)?;
    let ws = Workspace::new(1)?;
    let mut seq = FlintSeq::new(&model, context)?;
    let t0 = Instant::now();
    fl::prefill(&model, &ws, &mut seq, &ids, &s)?;
    let mut next = fl::argmax(&ws, 1, &s)?[0];
    let prefill = t0.elapsed().as_secs_f64();
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
