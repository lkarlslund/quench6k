//! The server's worker loop for Inkling 2 checkpoints: one sequence, its cached prefix reused across
//! requests while the rings still hold it, greedy or sampled decoding with drafts verified in one step:
//! the checkpoint's DSpark drafter (as many of its drafts as [`DraftPolicy`] expects to pay off), or
//! prompt-lookup drafts without one.

use std::sync::atomic::Ordering;
use std::sync::mpsc;

use anyhow::Result;
use quench6k_engine::hy4::policy::DraftPolicy;
use quench6k_engine::inkling::InklingModel;
use quench6k_engine::inkling::forward::{Logits, State, Workspace};

use super::{Exit, Finish, Metrics, Msg, Work, fail_work};

/// Prefill chunk (tokens per forward step).
const CHUNK: usize = 2048;
/// Most tokens of a verification step (the pending token and its drafts); the sampler takes 8 rows.
const MAX_STEP: usize = 8;

pub(super) fn schedule(
    model: &InklingModel,
    s: &quench6k_cuda::Stream,
    context: usize,
    jobs: &mpsc::Receiver<Work>,
    metrics: &Metrics,
    waiting: Vec<Work>,
    unload_idle: Option<std::time::Duration>,
) -> Exit {
    let vocab = model.cfg.vocab;
    let setup = || -> Result<_> {
        let ws = Workspace::new(model, CHUNK, MAX_STEP)?;
        let st = State::new(model, context, s)?;
        eprintln!(
            "inkling worker: workspace {:.2} GiB, cache {:.2} GiB for {context} tokens",
            ws.device_bytes as f64 / (1u64 << 30) as f64,
            st.device_bytes as f64 / (1u64 << 30) as f64
        );
        Ok((ws, st, crate::flint::Sampler::new()?))
    };
    let (mut ws, mut st, mut sampler) = match setup() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("inkling worker: {e:#}");
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
            let reused = st.rewind(&job.ids);
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
            let prefill = t0.elapsed().as_secs_f64();
            metrics.live[0].decoding();
            let budget = p.max_tokens.min(context - job.ids.len()).max(1);
            // Drafts (request field "mtp", default on): the DSpark drafter's, or else the tokens that followed the
            // latest earlier occurrence of the last few tokens.
            let dspark = job.mtp && model.drafter.is_some();
            let lookup = job.mtp && !dspark && quench6k_cuda::variants::get("inkling.lookup", 1) != 0;
            let mut policy = DraftPolicy::new(model.drafter.as_ref().map_or(0, |d| d.block - 1), 0.008, 0.0015, 0.0015);
            let mut row = st.last_t - 1;
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
                let room = (budget - generated.len()).min(context - st.pos - 1);
                let td = std::time::Instant::now();
                let draft = if dspark {
                    let n = policy.drafts().min(room);
                    let mut d = if n > 0 { model.draft(&mut st, &mut ws, s, pending, row)? } else { Vec::new() };
                    d.truncate(n);
                    d
                } else if lookup {
                    crate::flint::lookup_draft(&hist, cap.min(room))
                } else {
                    Vec::new()
                };
                let draft_secs = (!draft.is_empty()).then(|| td.elapsed().as_secs_f64());
                let tv = std::time::Instant::now();
                let mut toks = Vec::with_capacity(draft.len() + 1);
                toks.push(pending);
                toks.extend_from_slice(&draft);
                let before = st.pos;
                let want = if toks.len() == 1 { Logits::Last } else { Logits::All };
                model.forward(&mut st, &mut ws, s, &toks, want)?;
                steps += 1;
                drafted += draft.len();
                sampler.fetch(s, ws.out_logits, toks.len(), vocab, p)?;
                let verify_secs = tv.elapsed().as_secs_f64();
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
                if dspark {
                    policy.update(draft.len(), j, verify_secs, draft_secs);
                }
                // Keep the fed tokens that were accepted; the rest are overwritten later.
                st.truncate(before + 1 + j);
                row = j;
            }
            let finish = done.unwrap();
            let spec_note = if (lookup || dspark) && steps > 0 {
                format!(
                    ", {} {accepted}/{drafted} drafts accepted, {:.2} tokens/step",
                    if dspark { "dspark" } else { "lookup" },
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
