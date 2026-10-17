//! Throughput benchmark with NInfer 6000's methodology: one request, a prompt of the given length,
//! then greedy decoding of a fixed number of tokens.

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use quench6k_engine::forward::{self, HeadWorkspace, SequenceState, Workspace};
use quench6k_engine::model::{D, Precision};

/// The first `len` tokens of the benchmark corpus (bench/context.ids, shared with NInfer 6000), or of
/// the whitespace-separated token file named by QUENCH6K_BENCH_IDS.
pub(crate) fn prompt(len: usize) -> Result<Vec<i32>> {
    let path = std::env::var("QUENCH6K_BENCH_IDS").unwrap_or_else(|_| "bench/context.ids".into());
    let text = std::fs::read_to_string(&path)?;
    let ids: Vec<i32> = text.split_ascii_whitespace().take(len).map(|t| t.parse()).collect::<Result<_, _>>()?;
    anyhow::ensure!(ids.len() == len, "{path} holds only {} tokens", ids.len());
    Ok(ids)
}

#[allow(clippy::too_many_arguments)]
pub fn run(model_dir: &Path, precision: Precision, contexts: &[usize], generate: usize, chunk: usize, profile: bool, ncu_steps: usize, mtp: usize, adaptive: bool) -> Result<()> {
    if mtp > 0 {
        return run_mtp(model_dir, precision, contexts, generate, chunk, mtp, adaptive, ncu_steps);
    }
    let (model, stream) = crate::load(model_dir, precision, false)?;
    let capacity = contexts.iter().max().copied().unwrap_or(0) + generate + 16 + ncu_steps;
    let mut ws = Workspace::for_model(&model, chunk, capacity)?;
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
        // Three decode runs of `generate / 3` tokens each; the median run is reported.
        let per_run = (generate / 3).max(1);
        let mut runs = Vec::new();
        let mut times = [0.0; 2];
        for _ in 0..3 {
            let t1 = Instant::now();
            for _ in 0..per_run {
                next = graph.step_timed(&model, &mut state, &mut ws, &hw, &stream, next, &mut times)?;
            }
            runs.push(per_run as f64 / t1.elapsed().as_secs_f64());
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let decode = runs[1];
        let generate = 3 * per_run;
        if ncu_steps > 0 {
            quench6k_cuda::profiler_start()?;
            for _ in 0..ncu_steps {
                next = graph.step(&model, &mut state, &mut ws, &hw, &stream, next)?;
            }
            quench6k_cuda::profiler_stop()?;
        }
        println!(
            "{ctx:>8} {prefill:>14.1} {decode:>14.1}   per token: host prepare {:.3} ms, graph {:.3} ms",
            1e3 * times[0] / generate as f64,
            1e3 * times[1] / generate as f64
        );
        if profile {
            const STEPS: usize = 16;
            let mut prof = quench6k_engine::profile::Profiler::default();
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
/// benchmarked and checks the greedy tokens against the reference sequence (which quench6k reproduces
/// exactly under the correctness gate). Fails the benchmark on a mismatch.
fn sanity(model: &quench6k_engine::model::Model, stream: &quench6k_cuda::Stream, chunk: usize) -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Sequence {
        tokens: Vec<i32>,
        prompt_len: usize,
    }
    const CHECK: usize = 64;
    let seq: Sequence = serde_json::from_slice(&std::fs::read("oracle/fixtures/code/sequence.json")?)?;
    let prompt = &seq.tokens[..seq.prompt_len];
    let want = &seq.tokens[seq.prompt_len..seq.prompt_len + CHECK];
    let mut ws = Workspace::for_model(&model, chunk, seq.tokens.len())?;
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

/// MTP speculative decoding benchmark: prefill, then greedy draft-and-verify rounds as CUDA graphs.
/// The sanity check runs the same path on the code fixture against its reference tokens first.
#[allow(clippy::too_many_arguments)]
fn run_mtp(model_dir: &Path, precision: Precision, contexts: &[usize], generate: usize, chunk: usize, drafts: usize, adaptive: bool, ncu_rounds: usize) -> Result<()> {
    use quench6k_engine::spec::{SpecDecoder, SpecGraphs};
    let (model, stream) = crate::load(model_dir, precision, true)?;
    let max_ctx = contexts.iter().max().copied().unwrap_or(0).max(512);
    let slack = (ncu_rounds + 4) * (drafts + 1) + 16;
    let mut ws = Workspace::for_model(&model, chunk, max_ctx + generate + slack)?;

    // Sanity: speculative greedy tokens of the code fixture equal its reference.
    {
        #[derive(serde::Deserialize)]
        struct Sequence {
            tokens: Vec<i32>,
            prompt_len: usize,
        }
        const CHECK: usize = 64;
        let seq: Sequence = serde_json::from_slice(&std::fs::read("oracle/fixtures/code/sequence.json")?)?;
        let mut state = SequenceState::with_mtp(seq.tokens.len() + slack, true)?;
        let mut dec = SpecDecoder::prefill(&model, &mut state, &mut ws, &stream, &seq.tokens[..seq.prompt_len], drafts, chunk)?;
        let mut g = SpecGraphs::capture(&model, &state, &mut ws, &dec, &stream, adaptive, None)?;
        let mut got = Vec::new();
        while got.len() < CHECK {
            got.extend(g.round(&model, &mut state, &mut ws, &mut dec, &stream)?);
        }
        let want = &seq.tokens[seq.prompt_len..seq.prompt_len + CHECK];
        match got.iter().zip(want).position(|(a, b)| a != b) {
            None => println!("sanity: {CHECK}/{CHECK} speculative greedy tokens of the code fixture match the reference"),
            Some(i) => anyhow::bail!("sanity check failed: speculative token {i} is {} where the reference has {}", got[i], want[i]),
        }
    }

    println!("{:>8} {:>14} {:>14} {:>16}", "context", "prefill tok/s", "decode tok/s", "tokens/round");
    for &ctx in contexts {
        let tokens = prompt(ctx)?;
        let mut state = SequenceState::with_mtp(ctx + generate + slack, true)?;
        let t0 = Instant::now();
        let mut dec = SpecDecoder::prefill(&model, &mut state, &mut ws, &stream, &tokens, drafts, chunk)?;
        let prefill = ctx as f64 / t0.elapsed().as_secs_f64();
        let mut g = SpecGraphs::capture(&model, &state, &mut ws, &dec, &stream, adaptive, None)?;
        let t1 = Instant::now();
        let mut produced = 0;
        while produced < generate {
            produced += g.round(&model, &mut state, &mut ws, &mut dec, &stream)?.len();
        }
        let decode = produced as f64 / t1.elapsed().as_secs_f64();
        let per_round = produced as f64 / dec.rounds as f64;
        println!(
            "{ctx:>8} {prefill:>14.1} {decode:>14.1} {per_round:>16.2}   per round: draft {:.3} ms, verify {:.3} ms",
            1e3 * dec.draft_secs / dec.rounds as f64,
            1e3 * dec.verify_secs / dec.rounds as f64
        );
        if ncu_rounds > 0 {
            quench6k_cuda::profiler_start()?;
            for _ in 0..ncu_rounds {
                g.round(&model, &mut state, &mut ws, &mut dec, &stream)?;
            }
            quench6k_cuda::profiler_stop()?;
        }
    }
    Ok(())
}
