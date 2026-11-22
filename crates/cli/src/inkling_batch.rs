//! `quench6k inkling batch-check`: several Inkling sequences decoded in batched small steps give each
//! sequence the bits it gets alone, and how fast such steps run.

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, ensure};
use quench6k_cuda::Stream;
use quench6k_cuda::kernels as k;
use quench6k_engine::inkling::InklingModel;
use quench6k_engine::inkling::forward::{BATCH_ROWS, Logits, State, Workspace};

use crate::chat::Chat;
use crate::inkling::{argmax, load, prefill};

/// One step of a sequence decoded alone: what it fed, drafted and produced.
struct StepRec {
    toks: Vec<i32>,
    drafts: Vec<i32>,
    conf: Vec<u32>,
    /// FNV-1a of each fed token's logits row.
    rows: Vec<u64>,
    /// Drafts accepted (greedy).
    accepted: usize,
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Hashes of `rows` logits rows of the workspace, and their greedy picks.
fn row_hashes(model: &InklingModel, ws: &Workspace, s: &Stream, rows: usize) -> Result<(Vec<u64>, Vec<i32>)> {
    let v = model.cfg.vocab;
    let logits: Vec<f32> = s.download(ws.out_logits, rows * v)?;
    let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
    let hashes = bytes.chunks(v * 4).map(fnv).collect();
    k::argmax(s, ws.out_logits, rows, v, v, ws.argmax_scratch, ws.argmax)?;
    Ok((hashes, s.download::<i32>(ws.argmax, rows)?))
}

/// The drafts a step of sequence `i` verifies: a width that varies with the sequence and the step.
fn width(i: usize, step: usize, block: usize, drafted: bool) -> usize {
    if drafted { 1 + (i * 3 + step * 5) % block } else { 1 }
}

/// Greedy acceptance: how many drafts the picks confirm.
fn accept(drafts: &[i32], picks: &[i32]) -> usize {
    drafts.iter().zip(picks).take_while(|(d, p)| d == p).count()
}

struct Seq {
    st: State,
    pending: i32,
    row: usize,
}

fn start(model: &InklingModel, ws: &mut Workspace, s: &Stream, prompt: &[i32], cap: usize) -> Result<Seq> {
    let mut st = State::new(model, cap, s)?;
    prefill(model, &mut st, ws, s, prompt)?;
    let pending = argmax(ws, model, s)?;
    let row = st.last_t - 1;
    Ok(Seq { st, pending, row })
}

/// Sequence `i` decoded alone for `steps` steps.
#[allow(clippy::too_many_arguments)]
fn alone(model: &InklingModel, ws: &mut Workspace, s: &Stream, prompt: &[i32], cap: usize, i: usize, steps: usize, drafted: bool) -> Result<Vec<StepRec>> {
    let block = model.drafter.as_ref().map_or(1, |d| d.block);
    let mut q = start(model, ws, s, prompt, cap)?;
    let mut out = Vec::new();
    for step in 0..steps {
        let w = width(i, step, block, drafted);
        let (drafts, conf) = if drafted {
            let d = model.draft(&mut q.st, ws, s, q.pending, q.row)?;
            (d, model.draft_conf(ws).iter().map(|c| c.to_bits()).collect())
        } else {
            (Vec::new(), Vec::new())
        };
        let mut toks = vec![q.pending];
        toks.extend(drafts.iter().take(w - 1));
        let before = q.st.pos;
        model.forward(&mut q.st, ws, s, &toks, Logits::All)?;
        let (rows, picks) = row_hashes(model, ws, s, toks.len())?;
        let j = accept(&toks[1..], &picks);
        q.st.truncate(before + 1 + j);
        q.row = j;
        q.pending = picks[j];
        out.push(StepRec { toks, drafts, conf, rows, accepted: j });
    }
    Ok(out)
}

/// The first `n` sequences in lockstep batches; returns the number of mismatches (printed).
fn batched(model: &InklingModel, ws: &mut Workspace, s: &Stream, prompts: &[&[i32]], cap: usize, recs: &[Vec<StepRec>], drafted: bool) -> Result<usize> {
    let n = prompts.len();
    let mut seqs: Vec<Seq> = prompts.iter().map(|p| start(model, ws, s, p, cap)).collect::<Result<_>>()?;
    let mut bad = 0;
    for step in 0..recs[0].len() {
        let mut states: Vec<&mut State> = seqs.iter_mut().map(|q| &mut q.st).collect();
        let pend: Vec<i32> = (0..n).map(|i| recs[i][step].toks[0]).collect();
        let rows: Vec<usize> = (0..n).map(|i| if step == 0 { usize::MAX } else { recs[i][step - 1].accepted }).collect();
        if drafted {
            // The source rows: the last prompt row first, then each sequence's last accepted row.
            let rows: Vec<usize> = rows.iter().zip(&states).map(|(&r, st)| if r == usize::MAX { st.last_t - 1 } else { r }).collect();
            let d = model.draft_batch(&mut states, ws, s, &pend, &rows)?;
            let conf = model.draft_conf_batch(ws);
            for i in 0..n {
                let c: Vec<u32> = conf[i].iter().map(|x| x.to_bits()).collect();
                if d[i] != recs[i][step].drafts || c != recs[i][step].conf {
                    bad += 1;
                    println!("  step {step} sequence {i}: drafts {:?} / {:?} alone", d[i], recs[i][step].drafts);
                }
            }
        }
        let toks: Vec<&[i32]> = (0..n).map(|i| recs[i][step].toks.as_slice()).collect();
        let before: Vec<usize> = states.iter().map(|st| st.pos).collect();
        model.forward_batch(&mut states, ws, s, &toks)?;
        let total: usize = toks.iter().map(|t| t.len()).sum();
        let (hashes, _) = row_hashes(model, ws, s, total)?;
        let mut at = 0;
        for i in 0..n {
            let w = toks[i].len();
            if hashes[at..at + w] != recs[i][step].rows[..] {
                bad += 1;
                println!("  step {step} sequence {i}: logits differ ({w} rows)");
            }
            at += w;
            states[i].truncate(before[i] + 1 + recs[i][step].accepted);
        }
    }
    Ok(bad)
}

fn gib(b: usize) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn used() -> f64 {
    gib(quench6k_cuda::mem_info().map_or(0, |(free, total)| total - free))
}

pub fn run(dir: &Path, text: &Path, ctxs: &[usize], counts: &[usize], steps: usize, time_steps: usize, check: bool) -> Result<()> {
    let chat = Chat::load(dir, None)?;
    let ids = chat.encode(&std::fs::read_to_string(text)?)?;
    let (model, s) = load(dir)?;
    let nmax = counts.iter().copied().max().unwrap_or(1);
    ensure!(nmax <= quench6k_engine::inkling::forward::MAX_SEQS, "at most {} sequences", quench6k_engine::inkling::forward::MAX_SEQS);
    let block = model.drafter.as_ref().map_or(1, |d| d.block);
    let mut ws = Workspace::new(&model, 2048, BATCH_ROWS)?;
    println!("inkling: workspace {:.2} GiB; {} bytes per position of a sequence", gib(ws.device_bytes), State::bytes_per_position(&model));
    for &ctx in ctxs {
        // Sequence i: its own stretch of the corpus, 37 tokens shorter per i.
        let room = ids.len().saturating_sub(ctx + 64 * time_steps + 64);
        ensure!(room > 0, "corpus too short for {ctx} tokens");
        let prompts: Vec<&[i32]> = (0..nmax).map(|i| {
            let at = (i * 7919) % room;
            &ids[at..at + ctx.saturating_sub(37 * i).max(16)]
        }).collect();
        let cap = ctx + 16 * steps.max(time_steps) + 256;
        if check {
            for drafted in [false, true] {
                if drafted && model.drafter.is_none() {
                    continue;
                }
                let mode = if drafted { "drafted" } else { "plain" };
                let recs: Vec<Vec<StepRec>> =
                    (0..nmax).map(|i| alone(&model, &mut ws, &s, prompts[i], cap, i, steps, drafted)).collect::<Result<_>>()?;
                let rows: usize = recs.iter().map(|r| r.iter().map(|x| x.toks.len()).sum::<usize>()).sum();
                for &n in counts {
                    let bad = batched(&model, &mut ws, &s, &prompts[..n], cap, &recs[..n], drafted)?;
                    println!(
                        "check ctx {ctx} {mode}: {n} sequences x {steps} steps ({} rows in all): {}",
                        if n == nmax { rows } else { recs[..n].iter().map(|r| r.iter().map(|x| x.toks.len()).sum::<usize>()).sum() },
                        if bad == 0 { "bit-identical to alone".to_string() } else { format!("{bad} MISMATCHES") }
                    );
                }
            }
        }
        if time_steps == 0 {
            continue;
        }
        // Speed: nmax sequences prefilled once; each count decodes from their prompts' ends.
        let t0 = Instant::now();
        let mut seqs: Vec<Seq> = prompts.iter().map(|p| start(&model, &mut ws, &s, p, cap)).collect::<Result<_>>()?;
        let pf = t0.elapsed().as_secs_f64();
        let fed: usize = prompts.iter().map(|p| p.len()).sum();
        println!(
            "ctx {ctx}: {nmax} prompts ({fed} tokens) prefilled in {pf:.1}s; sequences {:.2} GiB each, device {:.1} GiB used",
            gib(seqs[0].st.device_bytes()),
            used()
        );
        let base: Vec<usize> = seqs.iter().map(|q| q.st.pos).collect();
        for &n in counts {
            let mut row = vec![0usize; n];
            // Plain decode: one greedy token per sequence per step.
            let mut pend: Vec<i32> = seqs[..n].iter().map(|q| q.pending).collect();
            s.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..time_steps {
                let mut states: Vec<&mut State> = seqs[..n].iter_mut().map(|q| &mut q.st).collect();
                let toks: Vec<[i32; 1]> = pend.iter().map(|&p| [p]).collect();
                let refs: Vec<&[i32]> = toks.iter().map(|t| t.as_slice()).collect();
                model.forward_batch(&mut states, &mut ws, &s, &refs)?;
                k::argmax(&s, ws.out_logits, n, model.cfg.vocab, model.cfg.vocab, ws.argmax_scratch, ws.argmax)?;
                pend = s.download::<i32>(ws.argmax, n)?;
            }
            let plain = t0.elapsed().as_secs_f64() / time_steps as f64;
            for (q, &b) in seqs.iter_mut().zip(&base) {
                q.st.truncate(b);
            }
            // Verify steps of `block / 2` tokens (the corpus's next tokens as drafts) and drafter passes.
            let w = (block / 2).max(1);
            let mut t_verify = 0.0;
            let mut t_draft = 0.0;
            let mut pend: Vec<i32> = seqs[..n].iter().map(|q| q.pending).collect();
            for step in 0..time_steps {
                let mut states: Vec<&mut State> = seqs[..n].iter_mut().map(|q| &mut q.st).collect();
                if model.drafter.is_some() && step > 0 {
                    let t0 = Instant::now();
                    model.draft_batch(&mut states, &mut ws, &s, &pend, &row)?;
                    t_draft += t0.elapsed().as_secs_f64();
                }
                let toks: Vec<Vec<i32>> = (0..n)
                    .map(|i| {
                        let mut v = vec![pend[i]];
                        v.extend_from_slice(&ids[(i * 7919) % room + ctx + step..][..w - 1]);
                        v
                    })
                    .collect();
                let refs: Vec<&[i32]> = toks.iter().map(|t| t.as_slice()).collect();
                let before: Vec<usize> = states.iter().map(|st| st.pos).collect();
                s.synchronize()?;
                let t0 = Instant::now();
                model.forward_batch(&mut states, &mut ws, &s, &refs)?;
                k::argmax(&s, ws.out_logits, n * w, model.cfg.vocab, model.cfg.vocab, ws.argmax_scratch, ws.argmax)?;
                let picks = s.download::<i32>(ws.argmax, n * w)?;
                t_verify += t0.elapsed().as_secs_f64();
                for i in 0..n {
                    // One token kept per step (as a rejected draft would).
                    states[i].truncate(before[i] + 1);
                    pend[i] = picks[i * w];
                    row[i] = 0;
                }
            }
            for (q, &b) in seqs.iter_mut().zip(&base) {
                q.st.truncate(b);
            }
            let verify = t_verify / time_steps as f64;
            let draft = t_draft / (time_steps - 1).max(1) as f64;
            println!(
                "ctx {ctx} x{n}: plain step {:.2} ms, {:.0} tok/s in all ({:.1} per sequence); verify-{w} step {:.2} ms ({:.0} rows/s); draft pass {:.2} ms",
                plain * 1e3,
                n as f64 / plain,
                1.0 / plain,
                verify * 1e3,
                (n * w) as f64 / verify,
                draft * 1e3
            );
        }
        drop(seqs);
    }
    Ok(())
}

/// One turn's results: the prompt's last logits row and each decoded token's logits (FNV), and the tokens.
#[derive(PartialEq, Debug)]
struct Turn {
    hashes: Vec<u64>,
    reply: Vec<i32>,
}

/// FNV of row `row` of the workspace's logits, and its greedy token.
fn hash_row(model: &InklingModel, ws: &Workspace, s: &Stream, row: usize) -> Result<(u64, i32)> {
    let v = model.cfg.vocab;
    let logits: Vec<f32> = s.download(ws.out_logits + (row * v * 4) as u64, v)?;
    let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut best = 0;
    for (i, &x) in logits.iter().enumerate() {
        if x > logits[best] {
            best = i;
        }
    }
    Ok((fnv(&bytes), best as i32))
}

/// A turn on `st`: rewind to the canonical boundary, prefill the rest, then `n` greedy tokens, each step
/// batched with `other` (one token of its own) when given. Returns the turn and the reused length.
#[allow(clippy::too_many_arguments)]
fn turn(model: &InklingModel, ws: &mut Workspace, s: &Stream, st: &mut State, prompt: &[i32], n: usize, other: Option<&mut Seq>) -> Result<(Turn, usize)> {
    let keep = st.rewind(prompt, s)?;
    model.prefill(st, ws, s, &prompt[keep..])?;
    let (h, mut tok) = hash_row(model, ws, s, 0)?;
    let mut t = Turn { hashes: vec![h], reply: vec![tok] };
    let mut other = other;
    for _ in 1..n {
        match other.as_deref_mut() {
            Some(o) => {
                let mut states: Vec<&mut State> = vec![&mut o.st, st];
                let a = [o.pending];
                let b = [tok];
                model.forward_batch(&mut states, ws, s, &[&a, &b])?;
                o.pending = hash_row(model, ws, s, 0)?.1;
                let (h, x) = hash_row(model, ws, s, 1)?;
                t.hashes.push(h);
                tok = x;
            }
            None => {
                model.forward(st, ws, s, &[tok], Logits::Last)?;
                let (h, x) = hash_row(model, ws, s, 0)?;
                t.hashes.push(h);
                tok = x;
            }
        }
        t.reply.push(tok);
    }
    Ok((t, keep))
}

/// Canonical prefill: a multi-turn conversation's logits and tokens are the same cold (a fresh sequence
/// each turn), warm (its previous turn cached) and cross (the sequence held another conversation with the
/// same system prompt, decodes batched with another sequence, and branched off in between).
pub fn canon_check(dir: &Path, text: &Path, system: usize, turns: usize, n: usize) -> Result<()> {
    let chat = Chat::load(dir, None)?;
    let ids = chat.encode(&std::fs::read_to_string(text)?)?;
    let (model, s) = load(dir)?;
    let mut ws = Workspace::new(&model, quench6k_engine::inkling::forward::CANON_CHUNK, BATCH_ROWS)?;
    let sys = &ids[..system];
    let user = |c: usize, k: usize| -> &[i32] {
        let at = system + 20000 * (c + 1) + 1500 * k;
        &ids[at..at + 600 + 97 * k + 31 * c]
    };
    let cap = system + turns * (2000 + n) + 8192;
    // Cold: every turn on a fresh sequence; its replies build the later prompts.
    let mut prompts: Vec<Vec<i32>> = Vec::new();
    let mut cold = Vec::new();
    let mut hist = sys.to_vec();
    for k in 0..turns {
        hist.extend_from_slice(user(0, k));
        let mut st = State::new(&model, cap, &s)?;
        let (t, _) = turn(&model, &mut ws, &s, &mut st, &hist, n, None)?;
        prompts.push(hist.clone());
        hist.extend_from_slice(&t.reply);
        cold.push(t);
    }
    let report = |mode: &str, k: usize, t: &Turn, keep: usize, prev: usize| {
        let same = *t == cold[k];
        println!(
            "canon {mode} turn {k}: prompt {} tokens, reused {keep}, recomputed {} of the cached {prev} ({} more than new tokens): {}",
            prompts[k].len(),
            prompts[k].len() - keep,
            prev.saturating_sub(keep),
            if same { "bit-identical to cold".to_string() } else { format!("DIFFERS ({} of {} rows)", t.hashes.iter().zip(&cold[k].hashes).filter(|(a, b)| a != b).count(), t.hashes.len()) }
        );
        same
    };
    let mut ok = true;
    // Warm: one sequence through the turns.
    let mut st = State::new(&model, cap, &s)?;
    for k in 0..turns {
        let prev = st.pos;
        let (t, keep) = turn(&model, &mut ws, &s, &mut st, &prompts[k], n, None)?;
        ok &= report("warm", k, &t, keep, prev);
    }
    drop(st);
    // Cross: the sequence first serves conversation 1 (same system prompt), decoding batched with another
    // sequence; then conversation 0, with a long generation (3104 tokens in steps of 16) before the last
    // turn, so that its rings are beyond reach and the boundary snapshot is restored.
    let mut st = State::new(&model, cap + 4096, &s)?;
    let mut other = start(&model, &mut ws, &s, &ids[100000..100700], cap)?;
    let mut h1 = sys.to_vec();
    for k in 0..2 {
        h1.extend_from_slice(user(1, k));
        let (t, _) = turn(&model, &mut ws, &s, &mut st, &h1, n, Some(&mut other))?;
        h1.extend_from_slice(&t.reply);
    }
    for k in 0..turns {
        if k + 1 == turns && k > 0 {
            for c in ids[150000..153104].chunks(16) {
                model.forward(&mut st, &mut ws, &s, c, Logits::Last)?;
            }
            println!("canon cross: generated 3104 more tokens; at {}, frontier {}", st.pos, st.frontier());
        }
        let prev = st.pos;
        let (t, keep) = turn(&model, &mut ws, &s, &mut st, &prompts[k], n, Some(&mut other))?;
        ok &= report("cross", k, &t, keep, prev);
    }
    println!("canon: {}", if ok { "all turns bit-identical" } else { "MISMATCHES" });
    Ok(())
}
