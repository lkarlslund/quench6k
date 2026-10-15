//! Throughput benchmark with NInfer 6000's methodology: one request, a prompt of the given length,
//! then greedy decoding of a fixed number of tokens.

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use squash_engine::forward::{self, HeadWorkspace, SequenceState, Workspace};
use squash_engine::model::{D, Precision};

/// A prompt of `len` tokens made by repeating the long-context fixture's text.
fn prompt(len: usize) -> Result<Vec<i32>> {
    #[derive(serde::Deserialize)]
    struct Sequence {
        tokens: Vec<i32>,
    }
    let seq: Sequence = serde_json::from_slice(&std::fs::read("oracle/fixtures/long-context/sequence.json")?)?;
    Ok(seq.tokens.iter().cycle().take(len).copied().collect())
}

pub fn run(model_dir: &Path, precision: Precision, contexts: &[usize], generate: usize, chunk: usize, profile: bool, ncu_steps: usize) -> Result<()> {
    let (model, stream) = crate::load(model_dir, precision)?;
    let capacity = contexts.iter().max().copied().unwrap_or(0) + generate + 16 + ncu_steps;
    let mut ws = Workspace::new(chunk, capacity)?;
    let hw = HeadWorkspace::new(1, 1)?;

    sanity(&model, &stream, chunk)?;
    println!("{:>8} {:>14} {:>14}", "context", "prefill tok/s", "decode tok/s");
    for &ctx in contexts {
        let tokens = prompt(ctx)?;
        let mut state = SequenceState::new(ctx + generate + 16 + ncu_steps)?;
        let t0 = Instant::now();
        for c in tokens.chunks(chunk) {
            forward::forward(&model, &mut state, &mut ws, &stream, c)?;
        }
        let last = ws.out + (((ctx - 1) % chunk) * D * 2) as u64;
        let mut next = forward::head_stats(&model, &hw, &stream, last, &[0], 1)?.top_ids[0];
        let prefill = ctx as f64 / t0.elapsed().as_secs_f64();
        let graph = forward::DecodeGraph::capture(&model, &state, &ws, &hw, &stream)?;
        let t1 = Instant::now();
        let mut times = [0.0; 2];
        for _ in 0..generate {
            next = graph.step_timed(&model, &mut state, &mut ws, &hw, &stream, next, &mut times)?;
        }
        let decode = generate as f64 / t1.elapsed().as_secs_f64();
        if ncu_steps > 0 {
            squash_cuda::profiler_start()?;
            for _ in 0..ncu_steps {
                next = graph.step(&model, &mut state, &mut ws, &hw, &stream, next)?;
            }
            squash_cuda::profiler_stop()?;
        }
        println!(
            "{ctx:>8} {prefill:>14.1} {decode:>14.1}   per token: host prepare {:.3} ms, graph {:.3} ms",
            1e3 * times[0] / generate as f64,
            1e3 * times[1] / generate as f64
        );
        if profile {
            const STEPS: usize = 16;
            let mut prof = squash_engine::profile::Profiler::default();
            for _ in 0..STEPS {
                forward::forward_profiled(&model, &mut state, &mut ws, &stream, &[next], &mut prof)?;
                prof.time("head_argmax", &stream, || forward::head_argmax(&model, &hw, &stream, ws.out))?;
                stream.synchronize()?;
                prof.collect()?;
            }
            println!("per-op GPU time over {STEPS} eager decode steps at context {ctx}:\n{}", prof.report());
        }
    }
    Ok(())
}

/// Before timing anything, decodes the `code` fixture with the same prefill + graph path that is
/// benchmarked and checks the greedy tokens against the reference sequence (which squash reproduces
/// exactly under the correctness gate). Fails the benchmark on a mismatch.
fn sanity(model: &squash_engine::model::Model, stream: &squash_cuda::Stream, chunk: usize) -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Sequence {
        tokens: Vec<i32>,
        prompt_len: usize,
    }
    const CHECK: usize = 64;
    let seq: Sequence = serde_json::from_slice(&std::fs::read("oracle/fixtures/code/sequence.json")?)?;
    let prompt = &seq.tokens[..seq.prompt_len];
    let want = &seq.tokens[seq.prompt_len..seq.prompt_len + CHECK];
    let mut ws = Workspace::new(chunk, seq.tokens.len())?;
    let hw = HeadWorkspace::new(1, 1)?;
    let mut state = SequenceState::new(seq.tokens.len())?;
    for c in prompt.chunks(chunk) {
        forward::forward(model, &mut state, &mut ws, stream, c)?;
    }
    let last = ws.out + (((prompt.len() - 1) % chunk) * D * 2) as u64;
    let mut next = forward::head_stats(model, &hw, stream, last, &[0], 1)?.top_ids[0];
    let graph = forward::DecodeGraph::capture(model, &state, &ws, &hw, stream)?;
    let mut got = Vec::with_capacity(CHECK);
    for _ in 0..CHECK {
        got.push(next);
        next = graph.step(model, &mut state, &mut ws, &hw, stream, next)?;
    }
    match got.iter().zip(want).position(|(a, b)| a != b) {
        None => println!("sanity: {CHECK}/{CHECK} greedy tokens of the code fixture match the reference"),
        Some(i) => anyhow::bail!("sanity check failed: greedy token {i} is {} where the reference has {}", got[i], want[i]),
    }
    Ok(())
}
