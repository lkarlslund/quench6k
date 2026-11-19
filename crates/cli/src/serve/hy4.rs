//! The server's worker loop for Hy4-flash checkpoints: one sequence, its cached prefix reused across
//! requests (the sliding rings are snapshotted at the end of each prompt), greedy or sampled decoding
//! with prompt-lookup drafts verified in one step.

use std::sync::atomic::Ordering;
use std::sync::mpsc;

use anyhow::Result;
use quench6k_engine::hy4::Hy4Model;
use quench6k_engine::hy4::forward::{Logits, State, Workspace};

use super::{Exit, Finish, Metrics, Msg, Work, fail_work};

/// Prefill chunk (tokens per forward step).
const CHUNK: usize = 2048;
/// Most tokens of a verification step (the pending token and its drafts); the sampler takes 8 rows.
const MAX_STEP: usize = 8;

pub(super) fn schedule(
    model: &Hy4Model,
    s: &quench6k_cuda::Stream,
    context: usize,
    jobs: &mpsc::Receiver<Work>,
    metrics: &Metrics,
    waiting: Vec<Work>,
    unload_idle: Option<std::time::Duration>,
) -> Exit {
    let vocab = model.cfg.vocab;
    let setup = || -> Result<_> {
        let ws = Workspace::new(model, CHUNK, context, MAX_STEP)?;
        let st = State::new(model, context, s)?;
        eprintln!(
            "hy4 worker: workspace {:.2} GiB, cache {:.2} GiB for {context} tokens",
            ws.device_bytes as f64 / (1u64 << 30) as f64,
            st.device_bytes as f64 / (1u64 << 30) as f64
        );
        Ok((ws, st, crate::flint::Sampler::new()?))
    };
    let (mut ws, mut st, mut sampler) = match setup() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("hy4 worker: {e:#}");
            for w in waiting {
                fail_work(w, metrics, &format!("{e:#}"));
            }
            return Exit::Unload(Vec::new());
        }
    };
    let mut waiting = waiting.into_iter();
    loop {
        let w = match (waiting.next(), unload_idle) {
            (Some(w), _) => w,
            (None, Some(idle)) => match jobs.recv_timeout(idle) {
                Ok(w) => w,
                Err(mpsc::RecvTimeoutError::Timeout) => return Exit::Unload(Vec::new()),
                Err(mpsc::RecvTimeoutError::Disconnected) => return Exit::Closed,
            },
            (None, None) => match jobs.recv() {
                Ok(w) => w,
                Err(_) => return Exit::Closed,
            },
        };
        let job = match w {
            Work::Persist(ack) => {
                let _ = ack.send(());
                continue;
            }
            Work::Unload(ack) => return Exit::Unload(vec![ack]),
            Work::Generate(queued) => queued.job,
            Work::Inspect(job) => {
                let _ = job.tx.send(Err("inspection is not supported for this model".into()));
                continue;
            }
            Work::Scan(job) => {
                let _ = job.tx.send(Err("atlases are not supported for this model".into()));
                continue;
            }
            Work::Lens(job) => {
                let _ = job.tx.send(Err("not supported for this model".into()));
                continue;
            }
        };
        metrics.queued.fetch_sub(1, Ordering::Relaxed);
        if job.ids.len() >= context {
            let _ = job.tx.send(Msg::Error(format!(
                "prompt of {} tokens exceeds the context of {context}",
                job.ids.len()
            )));
            continue;
        }
        metrics.live[0].begin(job.ids.len());
        let t0 = std::time::Instant::now();
        metrics.running.store(1, Ordering::Relaxed);
        let result = (|| -> Result<(Finish, usize, usize, f64, f64)> {
            let p = &job.params;
            let reused = st.rewind(&job.ids, s)?;
            metrics.live[0].prefill(reused, reused);
            let mut at = reused;
            while at < job.ids.len() {
                let end = (at + CHUNK).min(job.ids.len());
                let want = if end == job.ids.len() { Logits::Last } else { Logits::None };
                model.forward(&mut st, &mut ws, s, &job.ids[at..end], want)?;
                at = end;
                metrics.live[0].prefill(at, reused);
                if let Some(cb) = &p.progress {
                    s.synchronize()?;
                    (cb.0)(at, reused);
                }
            }
            st.snapshot(s)?;
            let prefill = t0.elapsed().as_secs_f64();
            metrics.live[0].decoding();
            let budget = p.max_tokens.min(context - job.ids.len()).max(1);
            // Prompt-lookup drafts (request field "mtp", default on): the tokens that followed the latest
            // earlier occurrence of the last few tokens are verified in one step.
            let lookup = job.mtp && quench6k_cuda::variants::get("hy4.lookup", 1) != 0;
            let mut hist = job.ids.clone();
            let mut generated = Vec::new();
            let (mut drafted, mut accepted, mut steps) = (0usize, 0usize, 0usize);
            let t1 = std::time::Instant::now();
            let emit = |tok: i32, generated: &mut Vec<i32>, hist: &mut Vec<i32>| -> Option<Finish> {
                generated.push(tok);
                metrics.live[0].output();
                hist.push(tok);
                if job.tx.send(Msg::Token(tok)).is_err() {
                    return Some(Finish::Cancelled);
                }
                if p.stop_tokens.contains(&tok) || model.cfg.eos.contains(&tok) {
                    return Some(Finish::Stop);
                }
                (generated.len() >= budget).then_some(Finish::Length)
            };
            sampler.fetch(s, ws.out_logits, 1, vocab, p)?;
            let first = sampler.pick(0, p, &generated);
            let mut done = emit(first, &mut generated, &mut hist);
            let mut pending = first;
            let mut rate = 0.5f64;
            while done.is_none() {
                let cap = if rate < 0.15 {
                    1
                } else if rate < 0.35 {
                    2
                } else if rate < 0.6 {
                    4
                } else {
                    MAX_STEP - 1
                };
                let room = cap.min(budget - generated.len()).min(context - st.pos - 1);
                let draft = if lookup {
                    crate::flint::lookup_draft(&hist, room)
                } else {
                    Vec::new()
                };
                let mut toks = Vec::with_capacity(draft.len() + 1);
                toks.push(pending);
                toks.extend_from_slice(&draft);
                let before = st.pos;
                let want = if toks.len() == 1 { Logits::Last } else { Logits::All };
                model.forward(&mut st, &mut ws, s, &toks, want)?;
                steps += 1;
                drafted += draft.len();
                sampler.fetch(s, ws.out_logits, toks.len(), vocab, p)?;
                let mut j = 0;
                loop {
                    let tok = sampler.pick(j, p, &generated);
                    done = emit(tok, &mut generated, &mut hist);
                    if done.is_none() && j < draft.len() && tok == draft[j] {
                        j += 1;
                        continue;
                    }
                    pending = tok;
                    break;
                }
                accepted += j;
                if !draft.is_empty() {
                    rate = 0.8 * rate + 0.2 * (j as f64 / draft.len() as f64);
                }
                // Keep the fed tokens that were accepted; the rest are overwritten later.
                st.truncate(before + 1 + j);
            }
            let finish = done.unwrap();
            let spec_note = if lookup && steps > 0 {
                format!(
                    ", lookup {accepted}/{drafted} drafts accepted, {:.2} tokens/step",
                    generated.len().saturating_sub(1) as f64 / steps as f64
                )
            } else {
                String::new()
            };
            let secs = t1.elapsed().as_secs_f64();
            eprintln!(
                "done: prompt {} ({} reused) prefill {:.2}s ({:.0} tok/s), {} generated in {:.2}s ({:.1} tok/s){spec_note}, {:?}",
                job.ids.len(),
                reused,
                prefill,
                (job.ids.len() - reused) as f64 / prefill.max(1e-9),
                generated.len(),
                secs,
                generated.len().saturating_sub(1) as f64 / secs.max(1e-9),
                finish
            );
            Ok((finish, reused, generated.len(), prefill, secs))
        })();
        metrics.running.store(0, Ordering::Relaxed);
        metrics.live[0].end(result.is_err());
        match result {
            Ok((finish, reused, generated, prefill, decode)) => {
                metrics.done(0, job.ids.len(), reused, generated, prefill, decode);
                let _ = job.tx.send(Msg::Done(finish, reused));
            }
            Err(e) => {
                // The sequence state may be partial: start over next time.
                st.reset();
                let _ = job.tx.send(Msg::Error(format!("{e:#}")));
            }
        }
    }
}
