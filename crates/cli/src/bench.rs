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

pub fn run(model_dir: &Path, precision: Precision, contexts: &[usize], generate: usize, chunk: usize, profile: bool) -> Result<()> {
    let (model, stream) = crate::load(model_dir, precision)?;
    let capacity = contexts.iter().max().copied().unwrap_or(0) + generate + 16;
    let mut ws = Workspace::new(chunk, capacity)?;
    let hw = HeadWorkspace::new(1, 1)?;

    println!("{:>8} {:>14} {:>14}", "context", "prefill tok/s", "decode tok/s");
    for &ctx in contexts {
        let tokens = prompt(ctx)?;
        let mut state = SequenceState::new(ctx + generate + 16)?;
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
