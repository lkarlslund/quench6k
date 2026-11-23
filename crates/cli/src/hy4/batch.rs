//! `quench6k hy4 batch-check`: several sequences stepped alone and batched, compared bitwise, and the
//! aggregate decode throughput of batched steps.

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, ensure};
use quench6k_cuda::{Stream, kernels as k};
use quench6k_engine::hy4::Hy4Model;
use quench6k_engine::hy4::forward::{Logits, MAX_SEQS, MAX_STEP, State, Workspace};

use super::{fixture_tokens, load, prefill, row_hashes};

const CHUNK: usize = 2048;
const ROWS: usize = MAX_SEQS * MAX_STEP;

/// A sequence of the check: its tokens (prompt, then the teacher-forced continuation) and prompt length.
struct Seq {
    tokens: Vec<i32>,
    prompt: usize,
}

/// Prompt lengths spread over each fixture (sequence i of n), leaving `after` tokens to run.
fn sequences(fixtures: &[Vec<i32>], n: usize, after: usize) -> Result<Vec<Seq>> {
    let mut out = Vec::new();
    for i in 0..n {
        let tokens = fixtures[i % fixtures.len()].clone();
        ensure!(tokens.len() > after + 8, "a fixture of {} tokens is too short", tokens.len());
        let room = tokens.len() - after;
        // A spread of prompt lengths (golden-ratio fractions of what the fixture allows).
        let frac = 0.35 + 0.65 * ((i as f64 * 0.618_034).fract());
        let prompt = ((room as f64 * frac) as usize).clamp(8, room);
        out.push(Seq { tokens, prompt });
    }
    Ok(out)
}

fn argmax_rows(model: &Hy4Model, ws: &Workspace, s: &Stream, rows: usize) -> Result<Vec<i32>> {
    let vocab = model.cfg.vocab;
    if std::env::var("QUENCH6K_HY4_PICK").is_ok_and(|v| v == "topk") {
        // As the server's greedy sampler picks: a top-1 (top-k kernel).
        k::topk(s, ws.out_logits, rows, vocab, vocab, 1, false, ws.top_idx, ws.top_val)?;
        return s.download::<i32>(ws.top_idx, rows);
    }
    k::argmax(s, ws.out_logits, rows, vocab, vocab, ws.argmax_scratch, ws.argmax)?;
    s.download::<i32>(ws.argmax, rows)
}

/// Widths of sequence i's teacher-forced steps: a pattern over 1..=MAX_STEP.
fn width(i: usize, step: usize) -> usize {
    1 + (i * 3 + step * 5) % MAX_STEP
}

/// Teacher-forced: every sequence alone in one-token steps, then all batched in steps of varying widths;
/// logits rows and caches compared bitwise. Returns the number of differences.
fn teacher_forced(model: &Hy4Model, s: &Stream, ws: &mut Workspace, seqs: &[Seq], steps: usize) -> Result<usize> {
    let vocab = model.cfg.vocab;
    let mut alone = Vec::new();
    let mut alone_sums = Vec::new();
    for q in seqs {
        let mut st = State::new(model, q.prompt + steps + 64, s)?;
        prefill(model, &mut st, ws, s, &q.tokens[..q.prompt])?;
        let mut h = Vec::new();
        for p in q.prompt..q.prompt + steps {
            model.forward(&mut st, ws, s, &q.tokens[p..p + 1], Logits::All)?;
            h.extend(row_hashes(s, ws.out_logits, 1, vocab)?);
        }
        alone_sums.push(st.checksums(s)?);
        alone.push(h);
    }
    let mut states = Vec::new();
    for q in seqs {
        let mut st = State::new(model, q.prompt + steps + 64, s)?;
        prefill(model, &mut st, ws, s, &q.tokens[..q.prompt])?;
        states.push(st);
    }
    let mut hashes: Vec<Vec<u64>> = vec![Vec::new(); seqs.len()];
    let mut done = vec![0usize; seqs.len()];
    let mut step = 0;
    let mut batched_steps = 0;
    loop {
        let live: Vec<usize> = (0..seqs.len()).filter(|&i| done[i] < steps).collect();
        if live.is_empty() {
            break;
        }
        let toks: Vec<&[i32]> = live
            .iter()
            .map(|&i| {
                let n = width(i, step).min(steps - done[i]);
                let p = seqs[i].prompt + done[i];
                &seqs[i].tokens[p..p + n]
            })
            .collect();
        let mut refs: Vec<&mut State> = Vec::new();
        for (i, st) in states.iter_mut().enumerate() {
            if live.contains(&i) {
                refs.push(st);
            }
        }
        model.forward_batch(&mut refs, ws, s, &toks)?;
        let rows: usize = toks.iter().map(|t| t.len()).sum();
        let h = row_hashes(s, ws.out_logits, rows, vocab)?;
        let mut at = 0;
        for (j, &i) in live.iter().enumerate() {
            hashes[i].extend_from_slice(&h[at..at + toks[j].len()]);
            at += toks[j].len();
            done[i] += toks[j].len();
        }
        step += 1;
        batched_steps += 1;
    }
    let mut diffs = 0;
    for (i, q) in seqs.iter().enumerate() {
        let d: Vec<usize> = (0..steps).filter(|&p| alone[i][p] != hashes[i][p]).collect();
        let sums = states[i].checksums(s)?;
        let cache_same = sums == alone_sums[i];
        println!(
            "  seq {i}: prompt {:>5}, {steps} positions: {} logits rows differ{}; caches {}",
            q.prompt,
            d.len(),
            d.first().map_or(String::new(), |f| format!(" (first at {})", q.prompt + f)),
            if cache_same { "identical" } else { "DIFFER" }
        );
        diffs += d.len() + (!cache_same) as usize;
    }
    println!("  ({batched_steps} batched steps of 1 to {MAX_STEP} tokens per sequence against one-token steps alone)");
    Ok(diffs)
}

/// Greedy decoding record of one sequence: tokens, the drafts of each step, and per-step logits hashes.
#[derive(Default, PartialEq)]
struct Trace {
    tokens: Vec<i32>,
    drafts: Vec<Vec<i32>>,
    logits: Vec<u64>,
}

/// Drafts verified per step by sequence i (fixed, 0 to the drafter's 5).
fn drafts_of(i: usize, max: usize) -> usize {
    [3, 5, 1, 4, 2, 0, 5, 3][i % 8].min(max)
}

/// Greedy decoding of `n` tokens for every sequence after its prompt: alone (drafted, and plain), then
/// batched with drafts; traces compared. Returns the number of differing sequences.
fn greedy(model: &Hy4Model, s: &Stream, ws: &mut Workspace, seqs: &[Seq], n: usize) -> Result<usize> {
    let vocab = model.cfg.vocab;
    let max_d = model.dfly.as_ref().map_or(0, |f| f.width - 1);
    // Alone, drafted: the sequence's own loop.
    let run_alone = |ws: &mut Workspace, q: &Seq, d: usize| -> Result<Trace> {
        let mut st = State::new(model, q.prompt + n + 64, s)?;
        prefill(model, &mut st, ws, s, &q.tokens[..q.prompt])?;
        let mut tr = Trace::default();
        let mut pending = argmax_rows(model, ws, s, 1)?[0];
        tr.tokens.push(pending);
        let mut row = st.last_t - 1;
        while tr.tokens.len() < n {
            let mut drafts = if d > 0 { model.draft(&mut st, ws, s, pending, row)? } else { Vec::new() };
            drafts.truncate(d.min(n - tr.tokens.len()));
            tr.drafts.push(drafts.clone());
            let mut toks = vec![pending];
            toks.extend_from_slice(&drafts);
            let before = st.pos;
            model.forward(&mut st, ws, s, &toks, Logits::All)?;
            tr.logits.extend(row_hashes(s, ws.out_logits, toks.len(), vocab)?);
            let picks = argmax_rows(model, ws, s, toks.len())?;
            let mut j = 0;
            loop {
                tr.tokens.push(picks[j]);
                if j < drafts.len() && picks[j] == drafts[j] && tr.tokens.len() < n {
                    j += 1;
                    continue;
                }
                pending = picks[j];
                break;
            }
            st.truncate(before + 1 + j);
            row = j;
        }
        Ok(tr)
    };
    let mut alone = Vec::new();
    let mut plain = Vec::new();
    for (i, q) in seqs.iter().enumerate() {
        alone.push(run_alone(ws, q, drafts_of(i, max_d))?);
        plain.push(run_alone(ws, q, 0)?.tokens);
    }
    // Batched.
    let mut states = Vec::new();
    let mut traces: Vec<Trace> = Vec::new();
    let mut pending = Vec::new();
    let mut row = Vec::new();
    for q in seqs {
        let mut st = State::new(model, q.prompt + n + 64, s)?;
        prefill(model, &mut st, ws, s, &q.tokens[..q.prompt])?;
        let first = argmax_rows(model, ws, s, 1)?[0];
        traces.push(Trace {
            tokens: vec![first],
            ..Default::default()
        });
        pending.push(first);
        row.push(st.last_t - 1);
        states.push(st);
    }
    let mut steps = 0;
    loop {
        let live: Vec<usize> = (0..seqs.len()).filter(|&i| traces[i].tokens.len() < n).collect();
        if live.is_empty() {
            break;
        }
        let mut refs: Vec<&mut State> = Vec::new();
        for (i, st) in states.iter_mut().enumerate() {
            if live.contains(&i) {
                refs.push(st);
            }
        }
        let p: Vec<i32> = live.iter().map(|&i| pending[i]).collect();
        let rws: Vec<usize> = live.iter().map(|&i| row[i]).collect();
        let mut drafts = model.draft_batch(&mut refs, ws, s, &p, &rws)?;
        let mut toks: Vec<Vec<i32>> = Vec::new();
        for (j, &i) in live.iter().enumerate() {
            let d = drafts_of(i, max_d);
            if d == 0 {
                drafts[j].clear();
            }
            drafts[j].truncate(d.min(n - traces[i].tokens.len()));
            traces[i].drafts.push(drafts[j].clone());
            let mut t = vec![pending[i]];
            t.extend_from_slice(&drafts[j]);
            toks.push(t);
        }
        let before: Vec<usize> = refs.iter().map(|st| st.pos).collect();
        let tok_refs: Vec<&[i32]> = toks.iter().map(|t| t.as_slice()).collect();
        model.forward_batch(&mut refs, ws, s, &tok_refs)?;
        let rows: usize = toks.iter().map(|t| t.len()).sum();
        let h = row_hashes(s, ws.out_logits, rows, vocab)?;
        let picks = argmax_rows(model, ws, s, rows)?;
        let mut at = 0;
        for (j, &i) in live.iter().enumerate() {
            let tr = &mut traces[i];
            tr.logits.extend_from_slice(&h[at..at + toks[j].len()]);
            let mut a = 0;
            loop {
                tr.tokens.push(picks[at + a]);
                if a < drafts[j].len() && picks[at + a] == drafts[j][a] && tr.tokens.len() < n {
                    a += 1;
                    continue;
                }
                pending[i] = picks[at + a];
                break;
            }
            refs[j].truncate(before[j] + 1 + a);
            row[i] = a;
            at += toks[j].len();
        }
        steps += 1;
    }
    let mut bad = 0;
    for (i, q) in seqs.iter().enumerate() {
        let (a, b) = (&alone[i], &traces[i]);
        let same_tokens = a.tokens == b.tokens;
        let same_drafts = a.drafts == b.drafts;
        let same_logits = a.logits == b.logits;
        let plain_same = plain[i] == a.tokens;
        let accepted: usize = a.tokens.len() - 1;
        println!(
            "  seq {i}: prompt {:>5}, {} drafts per step: tokens {}, drafts {}, logits {} ({} steps alone, {:.2} tokens per step); plain greedy {}",
            q.prompt,
            drafts_of(i, max_d),
            if same_tokens { "same" } else { "DIFFER" },
            if same_drafts { "same" } else { "DIFFER" },
            if same_logits { "same" } else { "DIFFER" },
            a.drafts.len(),
            accepted as f64 / a.drafts.len().max(1) as f64,
            if plain_same { "same tokens" } else { "DIFFERENT tokens" }
        );
        bad += (!same_tokens || !same_drafts || !same_logits || !plain_same) as usize;
    }
    // A digest of every sequence's tokens, drafts and logits rows (to compare builds).
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    for tr in &traces {
        for v in tr.tokens.iter().map(|&t| t as u64).chain(tr.drafts.iter().flatten().map(|&t| t as u64 | 1 << 40)).chain(tr.logits.iter().copied()) {
            digest = (digest ^ v).wrapping_mul(0x100_0000_01b3);
        }
    }
    println!("  ({steps} batched steps; digest {digest:016x})");
    Ok(bad)
}

/// One timed throughput run of `n` sequences (truncated back to their prompts) with `d` drafts per step:
/// (seconds, tokens produced, every sequence's tokens).
#[allow(clippy::too_many_arguments)]
fn timed(
    model: &Hy4Model,
    s: &Stream,
    ws: &mut Workspace,
    states: &mut [State],
    firsts: &[i32],
    n: usize,
    d: usize,
    steps: usize,
) -> Result<(f64, usize, Vec<i32>)> {
    let vocab = model.cfg.vocab;
    let ctx: Vec<usize> = states.iter().map(|st| st.pos).collect();
    let mut pending: Vec<i32> = firsts[..n].to_vec();
    // The prefill's last row; after a truncation the drafter has no source until a step ran.
    let mut row = vec![0usize; n];
    let mut produced = 0usize;
    let mut text = Vec::new();
    // Two warm-up steps (the first plain: the drafter needs a step's hidden states; graph captures),
    // then the timed ones.
    let mut t0 = Instant::now();
    for it in 0..steps + 2 {
        if it == 2 {
            s.synchronize()?;
            t0 = Instant::now();
            produced = 0;
        }
        let mut refs: Vec<&mut State> = states[..n].iter_mut().collect();
        let dr = if d > 0 && it > 0 { model.draft_batch(&mut refs, ws, s, &pending, &row)? } else { vec![Vec::new(); n] };
        let toks: Vec<Vec<i32>> = (0..n)
            .map(|j| {
                let mut t = vec![pending[j]];
                t.extend(dr[j].iter().take(d));
                t
            })
            .collect();
        let before: Vec<usize> = refs.iter().map(|st| st.pos).collect();
        let tok_refs: Vec<&[i32]> = toks.iter().map(|t| t.as_slice()).collect();
        model.forward_batch(&mut refs, ws, s, &tok_refs)?;
        let rows: usize = toks.iter().map(|t| t.len()).sum();
        k::argmax(s, ws.out_logits, rows, vocab, vocab, ws.argmax_scratch, ws.argmax)?;
        let picks = s.download::<i32>(ws.argmax, rows)?;
        let mut at = 0;
        for j in 0..n {
            let mut a = 0;
            while a < toks[j].len() - 1 && picks[at + a] == toks[j][a + 1] {
                a += 1;
            }
            produced += a + 1;
            text.extend_from_slice(&toks[j][..a + 1]);
            pending[j] = picks[at + a];
            refs[j].truncate(before[j] + 1 + a);
            row[j] = a;
            at += toks[j].len();
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    for (st, &c) in states.iter_mut().zip(&ctx) {
        st.truncate(c);
    }
    Ok((secs, produced, text))
}

/// Aggregate decode throughput of batched steps for 1, 2, 4, 8 sequences at each context: plain (one token
/// per sequence per step) and with a fixed number of drafts per sequence.
fn throughput(model: &Hy4Model, s: &Stream, ws: &mut Workspace, text: &[i32], ctxs: &[usize], steps: usize, drafts: &[usize]) -> Result<()> {
    for &ctx in ctxs {
        let mut states = Vec::new();
        let mut firsts = Vec::new();
        for i in 0..MAX_SEQS {
            let off = i * 397;
            ensure!(off + ctx <= text.len(), "text of {} tokens is too short for {ctx} positions", text.len());
            let mut st = State::new(model, ctx + steps * MAX_STEP + 64, s)?;
            prefill(model, &mut st, ws, s, &text[off..off + ctx])?;
            firsts.push(argmax_rows(model, ws, s, 1)?[0]);
            states.push(st);
        }
        let mem: usize = states.iter().map(|st| st.device_bytes()).sum();
        println!(
            "ctx {ctx}: {MAX_SEQS} sequences hold {:.2} GiB ({:.1} MiB each)",
            mem as f64 / (1u64 << 30) as f64,
            mem as f64 / MAX_SEQS as f64 / (1u64 << 20) as f64
        );
        // QUENCH6K_HY4_BENCH_SEQS: the sequence counts to time (comma-separated; default 1,2,4,8).
        let counts: Vec<usize> = std::env::var("QUENCH6K_HY4_BENCH_SEQS")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).filter(|&n| (1..=MAX_SEQS).contains(&n)).collect())
            .unwrap_or_else(|| vec![1, 2, 4, 8]);
        // QUENCH6K_HY4_AB=1: each case four times, batched GEMV forms on, off, off, on (the same bits).
        let ab = std::env::var("QUENCH6K_HY4_AB").is_ok_and(|v| v == "1");
        for &n in &counts {
            for &d in std::iter::once(&0).chain(drafts) {
                let arms: &[Option<bool>] = if ab { &[Some(true), Some(false), Some(false), Some(true)] } else { &[None] };
                let mut texts: Vec<Vec<i32>> = Vec::new();
                for &arm in arms {
                    if let Some(on) = arm {
                        quench6k_engine::hy4::forward::set_batched_gemv(on);
                        ws.clear_graphs();
                    }
                    let (secs, produced, text) = timed(model, s, ws, &mut states, &firsts, n, d, steps)?;
                    texts.push(text);
                    println!(
                        "  {n} seq, {}{}: {:>6.1} ms per step, {:>6.1} tok/s aggregate ({:.2} tokens per sequence per step)",
                        if d == 0 { "plain    ".to_string() } else { format!("{d} drafts ") },
                        match arm {
                            Some(true) => " [new]",
                            Some(false) => " [old]",
                            None => "",
                        },
                        secs * 1e3 / steps as f64,
                        produced as f64 / secs,
                        produced as f64 / (steps * n) as f64
                    );
                }
                ensure!(texts.iter().all(|t| *t == texts[0]), "{n} seq, {d} drafts: the arms produced different tokens");
            }
        }
    }
    Ok(())
}

/// Steps of one sequence interleaved with another's: sequence B's prefix, then sequence A's prefill and
/// one-token steps, then B's next `t` tokens as one step (a short prompt's graph step); B's logits row
/// compared with B run alone. Also B as a fresh sequence whose whole prompt is that one step.
fn interleave(model: &Hy4Model, s: &Stream, ws: &mut Workspace, a_tokens: &[i32], b_tokens: &[i32]) -> Result<usize> {
    let vocab = model.cfg.vocab;
    let mut bad = 0;
    for (prefix, t) in [(40usize, 3usize), (40, 7), (40, 12), (0, 5), (0, 10), (0, 16), (100, 1)] {
        let b_step = &b_tokens[prefix..prefix + t];
        let run_b = |ws: &mut Workspace, st: &mut State| -> Result<u64> {
            model.forward(st, ws, s, b_step, Logits::Last)?;
            Ok(row_hashes(s, ws.out_logits, 1, vocab)?[0])
        };
        // Alone.
        let mut b = State::new(model, 512, s)?;
        if prefix > 0 {
            prefill(model, &mut b, ws, s, &b_tokens[..prefix])?;
        }
        let alone = run_b(ws, &mut b)?;
        // Interleaved.
        let mut b = State::new(model, 512, s)?;
        if prefix > 0 {
            prefill(model, &mut b, ws, s, &b_tokens[..prefix])?;
        }
        let mut a = State::new(model, 1024, s)?;
        prefill(model, &mut a, ws, s, &a_tokens[..300])?;
        let mut next = argmax_rows(model, ws, s, 1)?[0];
        for _ in 0..6 {
            model.forward(&mut a, ws, s, &[next], Logits::Last)?;
            next = argmax_rows(model, ws, s, 1)?[0];
        }
        let mixed = run_b(ws, &mut b)?;
        // Batched with A.
        let mut b2 = State::new(model, 512, s)?;
        if prefix > 0 {
            prefill(model, &mut b2, ws, s, &b_tokens[..prefix])?;
        }
        let tb: Vec<i32> = b_step.iter().copied().take(MAX_STEP).collect();
        model.forward_batch(&mut [&mut a, &mut b2], ws, s, &[&[next], &tb])?;
        let h = row_hashes(s, ws.out_logits, 1 + tb.len(), vocab)?;
        let batched_ok = if tb.len() == t { h[tb.len()] == alone } else { true };
        println!(
            "  B prefix {prefix:>3} + {t:>2} tokens: after A's steps {}, batched with A {}",
            if mixed == alone { "same" } else { "DIFFERENT" },
            if tb.len() < t { "n/a" } else if batched_ok { "same" } else { "DIFFERENT" }
        );
        bad += (mixed != alone) as usize + (!batched_ok) as usize;
    }
    Ok(bad)
}

/// Greedy decoding of short raw prompts (each prefilled in one step: a CUDA graph when at most 16 tokens)
/// alone, then staggered as a server runs them: a prompt's prefill between the running sequences' one-token
/// steps (`batched`: the running sequences step together). Returns the number of differing sequences.
fn staggered(model: &Hy4Model, s: &Stream, ws: &mut Workspace, prompts: &[Vec<i32>], n: usize, batched: bool) -> Result<usize> {
    let alone: Vec<Vec<i32>> = prompts
        .iter()
        .map(|p| {
            let mut st = State::new(model, 1024, s)?;
            model.forward(&mut st, ws, s, p, Logits::Last)?;
            let mut out = vec![argmax_rows(model, ws, s, 1)?[0]];
            while out.len() < n {
                model.forward(&mut st, ws, s, &[*out.last().unwrap()], Logits::Last)?;
                out.push(argmax_rows(model, ws, s, 1)?[0]);
            }
            Ok(out)
        })
        .collect::<Result<_>>()?;
    let mut states: Vec<State> = Vec::new();
    let mut outs: Vec<Vec<i32>> = Vec::new();
    let mut started = 0;
    let mut tick = 0;
    while outs.len() < prompts.len() || outs.iter().any(|o| o.len() < n) {
        // A new prompt every third tick (four running at most, like four slots).
        let running = outs.iter().filter(|o| o.len() < n).count();
        if started < prompts.len() && tick % 3 == 0 && running < 4 {
            let mut st = State::new(model, 1024, s)?;
            model.forward(&mut st, ws, s, &prompts[started], Logits::Last)?;
            outs.push(vec![argmax_rows(model, ws, s, 1)?[0]]);
            states.push(st);
            started += 1;
        }
        let live: Vec<usize> = (0..outs.len()).filter(|&i| outs[i].len() < n).collect();
        if batched && !live.is_empty() {
            let toks: Vec<[i32; 1]> = live.iter().map(|&i| [*outs[i].last().unwrap()]).collect();
            let mut refs: Vec<&mut State> = Vec::new();
            for (i, st) in states.iter_mut().enumerate() {
                if live.contains(&i) {
                    refs.push(st);
                }
            }
            let tr: Vec<&[i32]> = toks.iter().map(|t| t.as_slice()).collect();
            model.forward_batch(&mut refs, ws, s, &tr)?;
            let picks = argmax_rows(model, ws, s, live.len())?;
            for (j, &i) in live.iter().enumerate() {
                outs[i].push(picks[j]);
            }
        } else {
            for &i in &live {
                let t = *outs[i].last().unwrap();
                model.forward(&mut states[i], ws, s, &[t], Logits::Last)?;
                outs[i].push(argmax_rows(model, ws, s, 1)?[0]);
            }
        }
        tick += 1;
    }
    let mut bad = 0;
    for (i, p) in prompts.iter().enumerate() {
        let first = (0..n).find(|&k| alone[i][k] != outs[i][k]);
        println!(
            "  prompt {i} ({} tokens): {}",
            p.len(),
            first.map_or("same tokens".to_string(), |k| format!("DIFFERENT from token {k}"))
        );
        bad += first.is_some() as usize;
    }
    Ok(bad)
}

/// The server's order with drafts: running sequences verify their drafts in one batched step, then draft
/// together (draft_batch); a new prompt is prefilled right after that (one step, Logits::Last) and drafts
/// alone. Every sequence's tokens compared with the plain greedy tokens of its prompt alone.
fn staggered_drafted(model: &Hy4Model, s: &Stream, ws: &mut Workspace, prompts: &[Vec<i32>], n: usize, slots: usize) -> Result<usize> {
    let alone: Vec<Vec<i32>> = prompts
        .iter()
        .map(|p| {
            let mut st = State::new(model, 1024, s)?;
            model.forward(&mut st, ws, s, p, Logits::Last)?;
            let mut out = vec![argmax_rows(model, ws, s, 1)?[0]];
            while out.len() < n {
                model.forward(&mut st, ws, s, &[*out.last().unwrap()], Logits::Last)?;
                out.push(argmax_rows(model, ws, s, 1)?[0]);
            }
            Ok(out)
        })
        .collect::<Result<_>>()?;
    let d = 3;
    let mut states: Vec<State> = Vec::new();
    let mut outs: Vec<Vec<i32>> = Vec::new();
    let mut drafts: Vec<Vec<i32>> = Vec::new();
    let mut started = 0;
    loop {
        // New prompts while slots are free, each right after the previous GPU work (a batched step and
        // its drafter pass, or another prefill).
        while started < prompts.len() && outs.iter().filter(|o| o.len() < n).count() < slots {
            let mut st = State::new(model, 1024, s)?;
            model.forward(&mut st, ws, s, &prompts[started], Logits::Last)?;
            let first = argmax_rows(model, ws, s, 1)?[0];
            let row = st.last_t - 1;
            let dr = model.draft(&mut st, ws, s, first, row)?;
            outs.push(vec![first]);
            drafts.push(dr);
            states.push(st);
            started += 1;
        }
        let live: Vec<usize> = (0..outs.len()).filter(|&i| outs[i].len() < n).collect();
        if live.is_empty() {
            break;
        }
        let toks: Vec<Vec<i32>> = live
            .iter()
            .map(|&i| {
                let mut t = vec![*outs[i].last().unwrap()];
                t.extend(drafts[i].iter().take(d.min(n - outs[i].len())));
                t
            })
            .collect();
        let mut refs: Vec<&mut State> = Vec::new();
        for (i, st) in states.iter_mut().enumerate() {
            if live.contains(&i) {
                refs.push(st);
            }
        }
        let before: Vec<usize> = refs.iter().map(|st| st.pos).collect();
        let tr: Vec<&[i32]> = toks.iter().map(|t| t.as_slice()).collect();
        model.forward_batch(&mut refs, ws, s, &tr)?;
        let rows: usize = toks.iter().map(|t| t.len()).sum();
        let picks = argmax_rows(model, ws, s, rows)?;
        let (mut at, mut pend, mut srcs) = (0, Vec::new(), Vec::new());
        for (j, &i) in live.iter().enumerate() {
            let mut a = 0;
            loop {
                outs[i].push(picks[at + a]);
                if a + 1 < toks[j].len() && picks[at + a] == toks[j][a + 1] && outs[i].len() < n {
                    a += 1;
                    continue;
                }
                break;
            }
            refs[j].truncate(before[j] + 1 + a);
            pend.push(*outs[i].last().unwrap());
            srcs.push(a);
            at += toks[j].len();
        }
        let dr = model.draft_batch(&mut refs, ws, s, &pend, &srcs)?;
        for (j, &i) in live.iter().enumerate() {
            drafts[i] = dr[j].clone();
        }
    }
    let mut bad = 0;
    for (i, p) in prompts.iter().enumerate() {
        let first = (0..n).find(|&k| alone[i][k] != outs[i][k]);
        if first.is_some() || i < 2 {
            println!(
                "  prompt {i} ({} tokens): {}",
                p.len(),
                first.map_or("same tokens".to_string(), |k| format!("DIFFERENT from token {k}"))
            );
        }
        bad += first.is_some() as usize;
    }
    println!("  {} of {} prompts differ", bad, prompts.len());
    Ok(bad)
}

const NAMES: [&str; 16] = [
    "Amber", "Basil", "Cedar", "Delta", "Ember", "Fjord", "Garnet", "Harbor", "Indigo", "Jasper", "Kelp", "Lumen", "Marble",
    "Nectar", "Onyx", "Pepper",
];
const QUESTIONS: [&str; 8] = [
    "explain how a binary heap works",
    "list five uses of a hash set",
    "write a haiku about rain",
    "describe the water cycle briefly",
    "give three tips for code review",
    "explain TCP slow start",
    "compare arrays and linked lists",
    "summarize what a mutex is for",
];

const PROMPTS: [&str; 8] = [
    "Rust ownership explained for a C programmer, in five short paragraphs:\n",
    "Write a Python function that merges two sorted lists, then explain its complexity.\n",
    "List ten facts about the Moon, one per line:\n",
    "Translate into French and German: 'The weather is lovely today, let us walk by the river.'\n",
    "Describe how a hash map handles collisions, with an example in C.\n",
    "Compose a short poem about autumn leaves falling on a quiet street.\n",
    "Summarize the causes of the First World War in a few sentences.\n",
    "Quick sort versus merge sort: compare them in a table.\n",
];

#[allow(clippy::too_many_arguments)]
pub fn run(
    dir: &Path,
    fixtures: &[std::path::PathBuf],
    seqs: usize,
    steps: usize,
    decode: usize,
    text: Option<&Path>,
    bench_ctx: &[usize],
    bench_steps: usize,
    bench_drafts: &[usize],
) -> Result<()> {
    ensure!((1..=MAX_SEQS).contains(&seqs), "1 to {MAX_SEQS} sequences");
    let fx: Vec<Vec<i32>> = fixtures.iter().map(|f| fixture_tokens(f)).collect::<Result<_>>()?;
    let (model, s) = load(dir)?;
    let mut ws = Workspace::new(&model, CHUNK, 0, ROWS)?;
    let (free, _) = quench6k_cuda::mem_info()?;
    eprintln!(
        "workspace {:.2} GiB; {:.2} GiB free; {} bytes per position",
        ws.device_bytes as f64 / (1u64 << 30) as f64,
        free as f64 / (1u64 << 30) as f64,
        State::bytes_per_position(&model)
    );
    let mut bad = 0;
    if std::env::var_os("QUENCH6K_HY4_INTERLEAVE").is_some() && fx.len() >= 2 {
        println!("interleaved steps:");
        bad += interleave(&model, &s, &mut ws, &fx[0], &fx[1])?;
    }
    if let Some(slots) = std::env::var("QUENCH6K_HY4_STAGGER_DRAFTS").ok().and_then(|v| v.parse::<usize>().ok()) {
        let chat = crate::chat::Chat::load(dir, None)?;
        let prompts: Vec<Vec<i32>> = (0..NAMES.len())
            .map(|i| chat.encode(&format!("{} asks: {}.\n", NAMES[i], QUESTIONS[i % QUESTIONS.len()])))
            .collect::<Result<_>>()?;
        println!("staggered short prompts with drafts, {slots} slots:");
        bad += staggered_drafted(&model, &s, &mut ws, &prompts, 24, slots)?;
    }
    if std::env::var_os("QUENCH6K_HY4_STAGGER").is_some() {
        let chat = crate::chat::Chat::load(dir, None)?;
        let prompts: Vec<Vec<i32>> = PROMPTS.iter().map(|p| chat.encode(p)).collect::<Result<_>>()?;
        for batched in [false, true] {
            println!("staggered short prompts, {}:", if batched { "running sequences batched" } else { "one sequence per step" });
            bad += staggered(&model, &s, &mut ws, &prompts, 40, batched)?;
        }
    }
    if steps > 0 && !fx.is_empty() {
        println!("teacher-forced, {seqs} sequences:");
        bad += teacher_forced(&model, &s, &mut ws, &sequences(&fx, seqs, steps)?, steps)?;
    }
    if decode > 0 && !fx.is_empty() {
        println!("greedy with drafts, {seqs} sequences, {decode} tokens each:");
        bad += greedy(&model, &s, &mut ws, &sequences(&fx, seqs, 0)?, decode)?;
    }
    if !bench_ctx.is_empty() {
        let tokens = match text {
            Some(t) => {
                let chat = crate::chat::Chat::load(dir, None)?;
                chat.encode(&std::fs::read_to_string(t)?)?
            }
            None => fx.iter().max_by_key(|f| f.len()).cloned().unwrap_or_default(),
        };
        throughput(&model, &s, &mut ws, &tokens, bench_ctx, bench_steps, bench_drafts)?;
    }
    if bad > 0 {
        anyhow::bail!("{bad} differences");
    }
    println!("all bit-identical");
    Ok(())
}
