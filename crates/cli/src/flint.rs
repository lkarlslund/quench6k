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
}

pub fn run(cmd: FlintCmd) -> Result<()> {
    match cmd {
        FlintCmd::Generate { model, prompt, raw, n, context } => generate(&model, &prompt, raw, n, context),
        FlintCmd::Ppl { model, file, ctx, chunks } => ppl(&model, &file, ctx, chunks),
    }
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
