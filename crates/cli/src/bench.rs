//! Throughput benchmark with NInfer 6000's methodology: one request, a prompt of the given length,
//! then greedy decoding of a fixed number of tokens (or, with `--sampled`, MTP decoding sampled with the
//! server's default parameters).

use std::path::Path;
use std::time::Instant;

use anyhow::Result;

use crate::gpumon;
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

/// Where `bench --json` appends its records, and the run's KV format.
pub struct JsonOut {
    pub path: std::path::PathBuf,
    pub kv: String,
    pub sampled: bool,
}

impl JsonOut {
    /// Appends one record: a measurement at one context.
    #[allow(clippy::too_many_arguments)]
    fn record(&self, mode: &str, drafts: usize, adaptive: bool, ctx: usize, generate: usize, prefill: f64, decode: f64, per_round: Option<f64>, gpu: Option<&gpumon::Window>) -> Result<()> {
        use std::io::Write;
        let mut rec = serde_json::json!({"engine": "quench6k", "date": utc_date(), "mode": mode, "drafts": drafts, "adaptive": adaptive, "kv": self.kv,
                                     "ctx": ctx, "generate": generate, "prefill_tok_s": (prefill * 10.0).round() / 10.0, "decode_tok_s": (decode * 10.0).round() / 10.0,
                                     "tokens_per_round": per_round.map(|r| (r * 100.0).round() / 100.0), "sampled": self.sampled});
        if let Some(w) = gpu {
            rec["gpu"] = w.json();
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        writeln!(f, "{rec}")?;
        Ok(())
    }
}

/// Seeds per context in sampled benchmarks.
const SAMPLED_SEEDS: usize = 4;

/// The contexts to measure, each after an unreported warm-up run at the same context: the first run
/// pays for lazy initialization, and the tokens a run generates have their embedding rows read from
/// the checkpoint (the page cache keeps them for the measured run, as a running server does).
fn with_warmup(contexts: &[usize]) -> Vec<(usize, bool)> {
    contexts.iter().flat_map(|&c| [(c, false), (c, true)]).collect()
}

/// Ends a GPU monitor window: prints it, warns when the GPU slowed down during the measurement.
fn gpu_window(mon: &Option<gpumon::Monitor>) -> Option<gpumon::Window> {
    let w = mon.as_ref()?.end();
    println!("         {}", w.summary());
    if w.slowed() {
        println!("         WARNING: the GPU slowed down during this measurement ({})", w.reason_names().join(", "));
    }
    Some(w)
}

/// Today's UTC date, YYYY-MM-DD.
fn utc_date() -> String {
    let days = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() / 86_400) as i64;
    // Days since 1970-01-01 to a civil date (H. Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02}")
}

#[allow(clippy::too_many_arguments)]
pub fn run(model_dir: &Path, precision: Precision, contexts: &[usize], generate: usize, chunk: usize, profile: bool, ncu_steps: usize, mtp: usize, adaptive: bool, sampled: bool, json: Option<&JsonOut>) -> Result<()> {
    if mtp > 0 {
        return run_mtp(model_dir, precision, contexts, generate, chunk, mtp, adaptive, sampled, ncu_steps, json);
    }
    anyhow::ensure!(!sampled, "--sampled needs --mtp");
    let (model, stream) = crate::load(model_dir, precision, false)?;
    let capacity = contexts.iter().max().copied().unwrap_or(0) + generate + 16 + ncu_steps;
    let mut ws = Workspace::for_model(&model, chunk, capacity)?;
    let hw = HeadWorkspace::new(1, 1)?;

    sanity(model_dir, &model, &stream, chunk)?;
    let mon = gpumon::Monitor::start();
    println!("{:>8} {:>14} {:>14}", "context", "prefill tok/s", "decode tok/s");
    for (ctx, report) in with_warmup(contexts) {
        let tokens = prompt(ctx)?;
        if let Some(m) = &mon {
            m.begin();
        }
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
        if ncu_steps > 0 && report {
            quench6k_cuda::profiler_start()?;
            for _ in 0..ncu_steps {
                next = graph.step(&model, &mut state, &mut ws, &hw, &stream, next)?;
            }
            quench6k_cuda::profiler_stop()?;
        }
        if !report {
            continue;
        }
        println!(
            "{ctx:>8} {prefill:>14.1} {decode:>14.1}   per token: host prepare {:.3} ms, graph {:.3} ms",
            1e3 * times[0] / generate as f64,
            1e3 * times[1] / generate as f64
        );
        let gpu = gpu_window(&mon);
        if let Some(j) = json {
            j.record("plain", 0, false, ctx, generate, prefill, decode, None, gpu.as_ref())?;
        }
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

/// Greedy tokens of the sanity check.
const CHECK: usize = 64;

/// The `code` fixture's prompt and the CHECK greedy tokens the benchmarked paths must reproduce, and
/// what they are. For the oracle's checkpoint (identified by its config.json) they are the fixture's
/// reference sequence, which quench6k reproduces exactly under the correctness gate; for any other
/// checkpoint they come from eager decoding (one forward pass per token, no CUDA graphs, no MTP).
fn reference(model_dir: &Path, model: &quench6k_engine::model::Model, stream: &quench6k_cuda::Stream, chunk: usize) -> Result<(Vec<i32>, Vec<i32>, &'static str)> {
    #[derive(serde::Deserialize)]
    struct Sequence {
        tokens: Vec<i32>,
        prompt_len: usize,
    }
    let seq: Sequence = serde_json::from_slice(&std::fs::read("oracle/fixtures/code/sequence.json")?)?;
    let prompt = seq.tokens[..seq.prompt_len].to_vec();
    let config = std::fs::read(model_dir.join("config.json"))?;
    if fnv(&config) == ORACLE_CONFIG_FNV {
        return Ok((prompt, seq.tokens[seq.prompt_len..seq.prompt_len + CHECK].to_vec(), "the reference"));
    }
    let mut ws = Workspace::for_model(model, chunk, prompt.len() + CHECK + 1)?;
    let hw = HeadWorkspace::new(1, 1)?;
    let mut state = SequenceState::new(prompt.len() + CHECK + 1)?;
    for c in prompt.chunks(chunk) {
        forward::forward(model, &mut state, &mut ws, stream, c)?;
    }
    let last = ws.out + (((prompt.len() - 1) % chunk) * D * 2) as u64;
    let mut want = vec![forward::head_stats(model, &hw, stream, last, &[0], 1)?.top_ids[0]];
    while want.len() < CHECK {
        forward::forward(model, &mut state, &mut ws, stream, &[*want.last().unwrap()])?;
        want.push(forward::head_stats(model, &hw, stream, ws.out, &[0], 1)?.top_ids[0]);
    }
    Ok((prompt, want, "eager decoding (this checkpoint is not the oracle's)"))
}

/// FNV-1a of the oracle checkpoint's config.json (ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4).
const ORACLE_CONFIG_FNV: u64 = 0x312a_2326_caf4_35b3;

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// Before timing anything, decodes the `code` fixture with the same prefill + graph path that is
/// benchmarked and checks the greedy tokens against [`reference`]. Fails the benchmark on a mismatch.
fn sanity(model_dir: &Path, model: &quench6k_engine::model::Model, stream: &quench6k_cuda::Stream, chunk: usize) -> Result<()> {
    let (prompt, want, what) = reference(model_dir, model, stream, chunk)?;
    let prompt = &prompt[..];
    let mut ws = Workspace::for_model(model, chunk, prompt.len() + CHECK + 1)?;
    let hw = HeadWorkspace::new(1, 1)?;
    let mut state = SequenceState::new(prompt.len() + CHECK + 1)?;
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
    match got.iter().zip(&want).position(|(a, b)| a != b) {
        None => println!("sanity: {CHECK}/{CHECK} greedy tokens of the code fixture match {what}"),
        Some(i) => anyhow::bail!("sanity check failed: greedy token {i} is {} where {what} has {}", got[i], want[i]),
    }
    Ok(())
}

/// MTP speculative decoding benchmark: prefill, then draft-and-verify rounds as CUDA graphs, greedy
/// or sampled with the server's defaults (temperature 1.0, top_p 0.95, top_k 20).
/// The sanity check runs the same path on the code fixture against its reference tokens first; sampled
/// runs check it through the sampling kernels with top_k 1.
#[allow(clippy::too_many_arguments)]
fn run_mtp(model_dir: &Path, precision: Precision, contexts: &[usize], generate: usize, chunk: usize, drafts: usize, adaptive: bool, sampled: bool, ncu_rounds: usize, json: Option<&JsonOut>) -> Result<()> {
    use quench6k_engine::spec::{Sampling, SpecDecoder, SpecGraphs};
    let defaults = |seed| Sampling { temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, seed };
    let (model, stream) = crate::load(model_dir, precision, true)?;
    let max_ctx = contexts.iter().max().copied().unwrap_or(0).max(512);
    let slack = (ncu_rounds + 4) * (drafts + 1) + 16;
    let mut ws = Workspace::for_model(&model, chunk, max_ctx + generate + slack)?;

    // Sanity: speculative greedy tokens of the code fixture equal the reference (sampled runs: through
    // the sampling kernels with top_k 1).
    {
        let (prompt, want, what) = reference(model_dir, &model, &stream, chunk)?;
        let mut state = SequenceState::with_mtp(prompt.len() + CHECK + slack, true)?;
        let mut dec = SpecDecoder::prefill(&model, &mut state, &mut ws, &stream, &prompt, drafts, chunk)?;
        let mut g = SpecGraphs::capture(&model, &state, &mut ws, &dec, &stream, adaptive, None, None)?;
        if sampled {
            g.set_sampling(&stream, Some(Sampling { top_k: 1, ..defaults(1) }))?;
        }
        let mut got = Vec::new();
        while got.len() < CHECK {
            got.extend(g.round(&model, &mut state, &mut ws, &mut dec, &stream)?);
        }
        match got.iter().zip(&want).position(|(a, b)| a != b) {
            None => println!("sanity: {CHECK}/{CHECK} speculative greedy tokens of the code fixture match {what}"),
            Some(i) => anyhow::bail!("sanity check failed: speculative token {i} is {} where {what} has {}", got[i], want[i]),
        }
    }

    println!("{:>8} {:>14} {:>14} {:>16}", "context", "prefill tok/s", "decode tok/s", "tokens/round");
    // Sampled runs average SAMPLED_SEEDS seeds: one 256-token sample's acceptance depends on the
    // text it happens to produce.
    let seeds = if sampled { SAMPLED_SEEDS } else { 1 };
    let chat = if sampled { Some(crate::chat::Chat::load(model_dir, None)?) } else { None };
    let mon = gpumon::Monitor::start();
    for (ctx, report) in with_warmup(contexts) {
        let tokens = prompt(ctx)?;
        if let Some(m) = &mon {
            m.begin();
        }
        let (mut prefill, mut produced, mut secs, mut rounds, mut draft, mut verify) = (0.0, 0, 0.0, 0, 0.0, 0.0);
        let mut first = Vec::new();
        for seed in 0..seeds {
            let mut state = SequenceState::with_mtp(ctx + generate + slack, true)?;
            let t0 = Instant::now();
            let mut dec = SpecDecoder::prefill(&model, &mut state, &mut ws, &stream, &tokens, drafts, chunk)?;
            if seed == 0 {
                prefill = ctx as f64 / t0.elapsed().as_secs_f64();
            }
            let mut g = SpecGraphs::capture(&model, &state, &mut ws, &dec, &stream, adaptive, None, None)?;
            if sampled {
                g.set_sampling(&stream, Some(defaults(1 + seed as u64)))?;
            }
            let t1 = Instant::now();
            let mut out = Vec::new();
            while out.len() < generate {
                out.extend(g.round(&model, &mut state, &mut ws, &mut dec, &stream)?);
            }
            secs += t1.elapsed().as_secs_f64();
            produced += out.len();
            (rounds, draft, verify) = (rounds + dec.rounds, draft + dec.draft_secs, verify + dec.verify_secs);
            if seed == 0 {
                first = out;
            }
            if ncu_rounds > 0 && report && seed + 1 == seeds {
                quench6k_cuda::profiler_start()?;
                for _ in 0..ncu_rounds {
                    g.round(&model, &mut state, &mut ws, &mut dec, &stream)?;
                }
                quench6k_cuda::profiler_stop()?;
            }
        }
        if !report {
            continue;
        }
        let decode = produced as f64 / secs;
        let per_round = produced as f64 / rounds as f64;
        println!(
            "{ctx:>8} {prefill:>14.1} {decode:>14.1} {per_round:>16.2}   per round: draft {:.3} ms, verify {:.3} ms",
            1e3 * draft / rounds as f64,
            1e3 * verify / rounds as f64
        );
        if let Some(chat) = &chat {
            // Sampled output has no reference; print its start to show it is text, not noise.
            println!("         output: {:?}", chat.decode(&first[..first.len().min(40)], false)?);
        }
        let gpu = gpu_window(&mon);
        if let Some(j) = json {
            j.record(if model.dflash.is_some() { "dflash" } else { "mtp" }, drafts, adaptive, ctx, produced, prefill, decode, Some(per_round), gpu.as_ref())?;
        }
    }
    Ok(())
}
