//! The server's worker loop for the models with engines of their own (Inkling 2, Hy4-flash, Flint): up
//! to --slots requests at once, each slot with its own sequence state and cached prefix, over any
//! engine that implements [`SlotEngine`].
//!
//! - A request goes to the free slot whose state holds the longest prefix of its prompt, else to the
//!   least recently used one.
//! - One prompt is prefilled at a time, in the engine's chunks; while other requests decode, in slices
//!   that take turns with their steps by time (--prefill-share).
//! - The decoding requests step together: one [`SlotEngine::step`] for all of them, each with its own
//!   drafts (the engine's drafter as its [`DraftPolicy`] decides, or prompt lookup) and sampling.
//! - Memory: a request starts when its plan (prompt + expected completion + a margin, in
//!   [`SlotEngine::plan_bytes`]) fits next to the running requests' plans in free device memory and
//!   what idle slots hold; idle slots give their memory up when the device is full (least recently used
//!   first); a request that does not fit waits for running ones to finish, and one that does not fit
//!   with nothing else on the device fails.
//!
//! A sequence's results never depend on which others share its steps: the engine's batched step is
//! bit-identical to its steps alone, and every per-request decision (drafts, sampling, stopping) is the
//! request's own.

use std::cmp::Reverse;
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::Result;
use quench6k_engine::hy4::policy::DraftPolicy;
use quench6k_engine::session::{Finish, GenParams};

use super::{Exit, Job, Metrics, Msg, Work};

/// Prompt tokens per prefill slice while other requests decode (whole chunks, at least one).
const PREFILL_SLICE: usize = 4096;
/// Device memory the admission plan leaves free (snapshots, CUDA graphs, allocator slack).
const HEADROOM: usize = 1 << 30;
/// Positions a request's plan holds beyond its prompt and expected completion.
const PLAN_MARGIN: usize = 4096;

/// Where a sampler finds a step's logits.
pub(super) trait SlotSampler {
    /// A step's logits for one sequence (a device address for the GPU engines).
    type Src: Copy;
    /// Loads `rows` rows of logits from `src`.
    fn fetch(&mut self, src: Self::Src, rows: usize, p: &GenParams) -> Result<()>;
    /// The token of fetched row `row`; `generated`: the request's output so far.
    fn pick(&mut self, row: usize, p: &GenParams, generated: &[i32]) -> i32;
}

/// A drafter pass's result.
pub(super) struct Draft {
    pub tokens: Vec<i32>,
    /// The drafter's stated probability of each draft (drafters that state them; else empty).
    pub conf: Vec<f32>,
    /// Seconds of the pass.
    pub secs: f64,
}

/// A sequence fed in a step: its position before the step, and when its step began.
#[derive(Clone, Copy)]
pub(super) struct Fed {
    pub at: usize,
    pub started: Instant,
}

/// What a step's sequence keeps: its first `keep` fed tokens (the rest are dropped), and, when the
/// request goes on with drafts, the token to draft after and the step row it was chosen from.
pub(super) struct Verdict {
    pub keep: usize,
    pub draft: Option<(i32, usize)>,
}

/// Called per sequence of a step once its logits are ready.
pub(super) type After<'a, S> = dyn FnMut(usize, S, Fed) -> Result<Verdict> + 'a;

/// How a model drafts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Drafter {
    /// No drafter: prompt-lookup drafts when `lookup`.
    None,
    /// A drafter that states each draft's probability ([`DraftPolicy::drafts_conf`]).
    Conf,
    /// A drafter whose draft count [`DraftPolicy::drafts`] picks.
    Count,
}

pub(super) struct Drafting {
    pub kind: Drafter,
    /// Drafts per drafter pass.
    pub drafts: usize,
    /// [`DraftPolicy::new`]'s initial costs: a one-token step, each further token, a drafter pass.
    pub costs: (f64, f64, f64),
    /// Prompt-lookup drafts without a drafter.
    pub lookup: bool,
}

/// One engine serving several sequences. Prefill runs one sequence at a time ([`SlotEngine::prefill`]),
/// decoding all of them in one [`SlotEngine::step`].
pub(super) trait SlotEngine {
    type State;
    type Sampler: SlotSampler;

    /// Log name ("inkling", ...).
    fn name(&self) -> &'static str;
    /// Tokens per prefill chunk.
    fn chunk(&self) -> usize;
    /// Most tokens of one sequence in a step (the pending token and its drafts).
    fn max_step(&self) -> usize;
    /// Tokens that end a generation besides the request's stop tokens.
    fn eos(&self) -> &[i32];
    fn drafting(&self) -> Drafting;
    fn new_sampler(&self) -> Result<Self::Sampler>;

    /// A sequence state for up to `capacity` positions. Out of device memory: an error whose message
    /// contains "out of memory".
    fn new_state(&mut self, capacity: usize) -> Result<Self::State>;
    /// Whether states map memory as they grow ([`SlotEngine::reserve`], [`SlotEngine::trim`]); states
    /// that do not are dropped to free memory.
    fn growable(&self) -> bool;
    /// Maps memory for positions < `positions`. Out of device memory: an error whose message contains
    /// "out of memory", nothing changed.
    fn reserve(&mut self, st: &mut Self::State, positions: usize) -> Result<()>;
    /// Unmaps memory beyond `positions` and forgets the cached tokens there.
    fn trim(&mut self, st: &mut Self::State, positions: usize);
    fn device_bytes(&self, st: &Self::State) -> usize;
    /// Device bytes a state needs to hold `positions` positions (0 while unknown).
    fn plan_bytes(&self, positions: usize) -> usize;
    /// Free device memory.
    fn free_bytes(&self) -> usize;

    /// The tokens a state holds.
    fn history<'a>(&self, st: &'a Self::State) -> &'a [i32];
    fn pos(&self, st: &Self::State) -> usize;
    /// Tokens of the state's last step.
    fn last_t(&self, st: &Self::State) -> usize;
    /// Keeps the longest reusable cached prefix of `prompt` (at most all but its last token).
    fn rewind(&mut self, st: &mut Self::State, prompt: &[i32]) -> Result<usize>;
    /// Remembers the state at the end of a prompt (for [`SlotEngine::rewind`]).
    fn snapshot(&mut self, st: &mut Self::State) -> Result<()>;
    fn reset(&mut self, st: &mut Self::State);
    /// Waits for the queued work (before progress callbacks).
    fn sync(&self) -> Result<()>;

    /// A prefill chunk of at most [`SlotEngine::chunk`] tokens; with `last`, the logits of its last token
    /// (one row) are at the returned source.
    fn prefill(&mut self, st: &mut Self::State, tokens: &[i32], last: bool) -> Result<<Self::Sampler as SlotSampler>::Src>;
    /// Drafts after `token`, chosen after row `row` of the state's last step. Called right after that step.
    fn draft(&mut self, st: &mut Self::State, token: i32, row: usize) -> Result<Draft>;
    /// One decode step: `tokens[i]` (1..=max_step) through `states[i]`, logits for every fed token. Per
    /// sequence, once its logits are ready, `after(i, logits, fed)` says what it keeps; the step then
    /// truncates the state to `fed.at + keep` and runs the requested draft. Results per sequence (an
    /// engine that cannot tell whose fault an error is returns it for each). Every sequence's results
    /// are bit-identical to its steps alone.
    fn step(
        &mut self,
        states: &mut [&mut Self::State],
        tokens: &[&[i32]],
        after: &mut After<'_, <Self::Sampler as SlotSampler>::Src>,
    ) -> Vec<Result<Option<Draft>>>;
}

pub(super) struct Config {
    pub slots: usize,
    /// Each slot's capacity (prompt + completion).
    pub context: usize,
    pub plan_tokens: usize,
    pub prefill_share: f64,
    pub unload_idle: Option<Duration>,
}

/// How a request drafts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Plain,
    Lookup,
    Conf,
    Count,
}

struct Req {
    job: Job,
    slot: usize,
    reused: usize,
    /// Prompt tokens computed or reused so far.
    at: usize,
    /// Positions the memory plan holds.
    planned: usize,
    budget: usize,
    /// Seconds computing the prompt.
    prefill: f64,
    decode_start: Instant,
    decoding: bool,
    generated: Vec<i32>,
    hist: Vec<i32>,
    /// The next token to feed and the row of the last step it was chosen from.
    pending: i32,
    row: usize,
    mode: Mode,
    policy: DraftPolicy,
    /// Recent acceptance rate (prompt-lookup draft lengths).
    rate: f64,
    /// Drafts to verify in the next step, with their stated probabilities and drafter time.
    draft: Vec<i32>,
    conf: Vec<f32>,
    draft_secs: Option<f64>,
    /// The planned draft's room and (counting drafters) count.
    room: usize,
    count: usize,
    drafted: usize,
    accepted: usize,
    steps: usize,
    done: Option<Finish>,
}

impl Req {
    /// Sends a token; Some when the run is over.
    fn emit(&mut self, tok: i32, eos: &[i32], metrics: &Metrics) -> Option<Finish> {
        self.generated.push(tok);
        metrics.live[self.slot].output();
        self.hist.push(tok);
        if self.job.tx.send(Msg::Token(tok)).is_err() {
            return Some(Finish::Cancelled);
        }
        if self.job.params.stop_tokens.contains(&tok) || eos.contains(&tok) {
            return Some(Finish::Stop);
        }
        (self.generated.len() >= self.budget).then_some(Finish::Length)
    }

    /// Plans the next step's drafts at position `pos` (the pending token's): prompt-lookup drafts now,
    /// or the drafter pass to run (Some).
    fn plan_draft(&mut self, pos: usize, context: usize, max_step: usize) -> Option<(i32, usize)> {
        self.draft.clear();
        self.conf.clear();
        self.draft_secs = None;
        let cap = if self.rate < 0.15 {
            1
        } else if self.rate < 0.35 {
            2
        } else if self.rate < 0.6 {
            4
        } else {
            max_step - 1
        };
        self.room = (self.budget - self.generated.len()).min(context.saturating_sub(pos + 1));
        match self.mode {
            Mode::Plain => None,
            Mode::Lookup => {
                self.draft = crate::flint::lookup_draft(&self.hist, cap.min(self.room));
                None
            }
            Mode::Conf => (self.policy.want_draft() && self.room > 0).then_some((self.pending, self.row)),
            Mode::Count => {
                self.count = self.policy.drafts().min(self.room);
                (self.count > 0).then_some((self.pending, self.row))
            }
        }
    }

    /// Takes the drafter's pass planned by [`Req::plan_draft`].
    fn take_draft(&mut self, d: Draft) {
        let mut tokens = d.tokens;
        match self.mode {
            Mode::Conf => {
                if !tokens.is_empty() {
                    self.conf = d.conf;
                    tokens.truncate(self.policy.drafts_conf(&self.conf[..tokens.len().min(self.conf.len())]).min(self.room));
                    self.draft_secs = Some(d.secs);
                }
            }
            _ => {
                tokens.truncate(self.count);
                self.draft_secs = (!tokens.is_empty()).then_some(d.secs);
            }
        }
        self.draft = tokens;
    }

    /// Takes a step's logits: picks and emits tokens while they match the drafts, updates the draft
    /// policy, and plans the next drafts.
    #[allow(clippy::too_many_arguments)]
    fn verify<S: SlotSampler>(
        &mut self,
        sampler: &mut S,
        src: S::Src,
        fed: Fed,
        eos: &[i32],
        context: usize,
        max_step: usize,
        metrics: &Metrics,
    ) -> Result<Verdict> {
        let draft = std::mem::take(&mut self.draft);
        sampler.fetch(src, draft.len() + 1, &self.job.params)?;
        let verify_secs = fed.started.elapsed().as_secs_f64();
        self.steps += 1;
        self.drafted += draft.len();
        let mut j = 0;
        let done = loop {
            let tok = sampler.pick(j, &self.job.params, &self.generated);
            let done = self.emit(tok, eos, metrics);
            if done.is_none() && j < draft.len() && tok == draft[j] {
                j += 1;
                continue;
            }
            self.pending = tok;
            break done;
        };
        self.accepted += j;
        if !draft.is_empty() {
            self.rate = 0.8 * self.rate + 0.2 * (j as f64 / draft.len() as f64);
        }
        match self.mode {
            Mode::Conf => self.policy.update_conf(&self.conf, draft.len(), j, verify_secs, self.draft_secs),
            Mode::Count => self.policy.update(draft.len(), j, verify_secs, self.draft_secs),
            _ => {}
        }
        self.row = j;
        self.done = done;
        let next = if done.is_none() {
            self.plan_draft(fed.at + 1 + j, context, max_step)
        } else {
            None
        };
        Ok(Verdict { keep: 1 + j, draft: next })
    }
}

struct Sched<'a, E: SlotEngine> {
    e: E,
    cfg: &'a Config,
    metrics: &'a Metrics,
    eos: Vec<i32>,
    drafting: Drafting,
    states: Vec<Option<E::State>>,
    samplers: Vec<E::Sampler>,
    reqs: Vec<Option<Req>>,
    /// When each slot last served a request (the least recently used idle slot gives way first).
    used: Vec<Instant>,
    /// Slots whose prompts are being computed, oldest first.
    prefilling: VecDeque<usize>,
    waiting_for_memory: bool,
}

fn is_oom(e: &anyhow::Error) -> bool {
    format!("{e:#}").contains("out of memory")
}

impl<E: SlotEngine> Sched<'_, E> {
    fn busy(&self) -> bool {
        self.reqs.iter().any(Option::is_some)
    }

    fn device_bytes(&self, i: usize) -> usize {
        self.states[i].as_ref().map_or(0, |st| self.e.device_bytes(st))
    }

    /// Positions a request's plan holds: its prompt and expected completion, plus a margin.
    fn plan_positions(&self, prompt: usize, max_tokens: usize) -> usize {
        let budget = max_tokens.min(self.cfg.context.saturating_sub(prompt));
        (prompt + budget.min(self.cfg.plan_tokens) + PLAN_MARGIN).min(self.cfg.context)
    }

    /// Whether a request's plan fits on `slot` next to the running requests' plans, from free device
    /// memory and what idle slots hold. With nothing running, any request starts.
    fn admits(&self, slot: usize, planned: usize) -> bool {
        if !self.busy() {
            return true;
        }
        let need = self.e.plan_bytes(planned).saturating_sub(self.device_bytes(slot));
        let outstanding: usize = (0..self.cfg.slots)
            .filter_map(|i| self.reqs[i].as_ref().map(|r| self.e.plan_bytes(r.planned).saturating_sub(self.device_bytes(i))))
            .sum();
        let reclaimable: usize = (0..self.cfg.slots).filter(|&i| i != slot && self.reqs[i].is_none()).map(|i| self.device_bytes(i)).sum();
        let free = self.e.free_bytes().saturating_sub(HEADROOM);
        need + outstanding <= free + reclaimable
    }

    /// Frees device memory for `keep`: mapped memory past the positions other slots hold (growable
    /// states), else the state of the least recently used idle slot. False when nothing is left.
    fn free_memory(&mut self, keep: usize) -> bool {
        if self.e.growable() {
            let before: usize = (0..self.cfg.slots).map(|i| self.device_bytes(i)).sum();
            for i in (0..self.cfg.slots).filter(|&i| i != keep) {
                if let Some(st) = self.states[i].as_mut() {
                    let pos = self.e.pos(st);
                    self.e.trim(st, pos);
                }
            }
            if (0..self.cfg.slots).map(|i| self.device_bytes(i)).sum::<usize>() < before {
                return true;
            }
        }
        let lru = (0..self.cfg.slots)
            .filter(|&i| i != keep && self.reqs[i].is_none() && self.device_bytes(i) > 0)
            .min_by_key(|&i| self.used[i]);
        let Some(i) = lru else {
            return false;
        };
        let held = self.states[i].as_ref().map_or(0, |st| self.e.history(st).len());
        if self.e.growable() {
            let st = self.states[i].as_mut().unwrap();
            self.e.trim(st, 0);
        } else {
            self.states[i] = None;
        }
        eprintln!("{} worker: slot {i}: context of {held} tokens dropped from device memory", self.e.name());
        self.metrics.kv_bytes[i].store(self.device_bytes(i) as u64, Ordering::Relaxed);
        true
    }

    /// Runs `f` on `slot`'s state, freeing memory elsewhere while it fails for lack of it.
    fn with_memory<T>(&mut self, slot: usize, mut f: impl FnMut(&mut E, &mut E::State) -> Result<T>) -> Result<T> {
        loop {
            let st = self.states[slot].as_mut().expect("slot state");
            match f(&mut self.e, st) {
                Err(e) if is_oom(&e) => {
                    if !self.free_memory(slot) {
                        return Err(e.context(format!("slot {slot}: device memory is held by running requests")));
                    }
                }
                r => return r,
            }
        }
    }

    /// Gives `slot` a state, freeing memory elsewhere while the device is full.
    fn ensure_state(&mut self, slot: usize) -> Result<()> {
        if self.states[slot].is_some() {
            return Ok(());
        }
        loop {
            match self.e.new_state(self.cfg.context) {
                Ok(st) => {
                    let bytes = self.e.device_bytes(&st);
                    eprintln!(
                        "{} worker: slot {slot}: state {:.2} GiB for {} tokens",
                        self.e.name(),
                        bytes as f64 / (1u64 << 30) as f64,
                        self.cfg.context
                    );
                    self.states[slot] = Some(st);
                    return Ok(());
                }
                Err(e) if is_oom(&e) => {
                    if !self.free_memory(slot) {
                        return Err(e);
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Ends `slot`'s request with an error; its state starts over.
    fn fail(&mut self, slot: usize, e: &anyhow::Error) {
        let Some(r) = self.reqs[slot].take() else {
            return;
        };
        eprintln!(
            "{} worker: slot {slot}, prompt {}, {} generated: {e:#}",
            self.e.name(),
            r.job.ids.len(),
            r.generated.len()
        );
        self.prefilling.retain(|&s| s != slot);
        if let Some(st) = self.states[slot].as_mut() {
            self.e.reset(st);
        }
        self.used[slot] = Instant::now();
        self.metrics.live[slot].end(true);
        let _ = r.job.tx.send(Msg::Error(format!("{e:#}")));
    }

    /// Ends `slot`'s request.
    fn complete(&mut self, slot: usize) {
        let Some(r) = self.reqs[slot].take() else {
            return;
        };
        let finish = r.done.unwrap_or(Finish::Cancelled);
        let secs = r.decode_start.elapsed().as_secs_f64();
        let n = r.generated.len();
        let spec_note = if r.mode != Mode::Plain && r.steps > 0 {
            let kind = match r.mode {
                Mode::Lookup => "lookup",
                _ => "drafter",
            };
            format!(
                ", {kind} {}/{} drafts accepted, {:.2} tokens/step",
                r.accepted,
                r.drafted,
                n.saturating_sub(1) as f64 / r.steps as f64
            )
        } else {
            String::new()
        };
        let prompt = r.job.ids.len();
        eprintln!(
            "done: slot {slot}, prompt {prompt} ({} reused) prefill {:.2}s ({:.0} tok/s), {n} generated in {secs:.2}s ({:.1} tok/s){spec_note}, {finish:?}",
            r.reused,
            r.prefill,
            (prompt - r.reused) as f64 / r.prefill.max(1e-9),
            n.saturating_sub(1) as f64 / secs.max(1e-9),
        );
        self.metrics.done(slot, prompt, r.reused, n, r.prefill, secs);
        self.metrics.live[slot].end(false);
        self.used[slot] = Instant::now();
        let _ = r.job.tx.send(Msg::Done(finish, r.reused));
    }

    /// Starts queued requests on free slots while their plans fit. False when the front one waits.
    fn admit(&mut self, queue: &mut VecDeque<Job>) {
        while let Some(job) = queue.front() {
            if job.tx.is_closed() {
                queue.pop_front();
                self.metrics.queued.fetch_sub(1, Ordering::Relaxed);
                continue;
            }
            if job.ids.is_empty() || job.ids.len() >= self.cfg.context {
                let job = queue.pop_front().unwrap();
                self.metrics.queued.fetch_sub(1, Ordering::Relaxed);
                let msg = if job.ids.is_empty() {
                    "empty prompt".to_string()
                } else {
                    format!("prompt of {} tokens exceeds the context of {}", job.ids.len(), self.cfg.context)
                };
                let _ = job.tx.send(Msg::Error(msg));
                continue;
            }
            // The free slot holding the most of the prompt, else the least recently used one.
            let free = (0..self.cfg.slots)
                .filter(|&i| self.reqs[i].is_none())
                .max_by_key(|&i| {
                    let common = self.states[i].as_ref().map_or(0, |st| {
                        let h = self.e.history(st);
                        h.iter().zip(&job.ids).take_while(|(a, b)| a == b).count()
                    });
                    (common.min(job.ids.len() - 1), Reverse(self.used[i]), Reverse(i))
                });
            let Some(slot) = free else {
                break;
            };
            let planned = self.plan_positions(job.ids.len(), job.params.max_tokens);
            if !self.admits(slot, planned) {
                if !self.waiting_for_memory {
                    eprintln!("{} worker: request of {} prompt tokens waits for device memory", self.e.name(), job.ids.len());
                    self.waiting_for_memory = true;
                }
                break;
            }
            if let Err(e) = self.ensure_state(slot) {
                if is_oom(&e) && self.busy() {
                    if !self.waiting_for_memory {
                        eprintln!("{} worker: request of {} prompt tokens waits for device memory", self.e.name(), job.ids.len());
                        self.waiting_for_memory = true;
                    }
                    break;
                }
                let job = queue.pop_front().unwrap();
                self.metrics.queued.fetch_sub(1, Ordering::Relaxed);
                eprintln!("{} worker: slot {slot}, prompt {}: {e:#}", self.e.name(), job.ids.len());
                let _ = job.tx.send(Msg::Error(format!("{e:#}")));
                continue;
            }
            self.waiting_for_memory = false;
            let job = queue.pop_front().unwrap();
            self.metrics.queued.fetch_sub(1, Ordering::Relaxed);
            self.used[slot] = Instant::now();
            self.metrics.live[slot].begin(job.ids.len());
            let t0 = Instant::now();
            let reused = {
                let st = self.states[slot].as_mut().unwrap();
                self.e.rewind(st, &job.ids)
            };
            let d = &self.drafting;
            let mode = if !job.mtp {
                Mode::Plain
            } else {
                match d.kind {
                    Drafter::Conf => Mode::Conf,
                    Drafter::Count => Mode::Count,
                    Drafter::None if d.lookup => Mode::Lookup,
                    Drafter::None => Mode::Plain,
                }
            };
            let budget = job.params.max_tokens.min(self.cfg.context - job.ids.len()).max(1);
            let r = Req {
                hist: job.ids.clone(),
                job,
                slot,
                reused: 0,
                at: 0,
                planned,
                budget,
                prefill: 0.0,
                decode_start: Instant::now(),
                decoding: false,
                generated: Vec::new(),
                pending: 0,
                row: 0,
                mode,
                policy: DraftPolicy::new(d.drafts, d.costs.0, d.costs.1, d.costs.2),
                rate: 0.5,
                draft: Vec::new(),
                conf: Vec::new(),
                draft_secs: None,
                room: 0,
                count: 0,
                drafted: 0,
                accepted: 0,
                steps: 0,
                done: None,
            };
            self.reqs[slot] = Some(r);
            match reused {
                Ok(reused) => {
                    let r = self.reqs[slot].as_mut().unwrap();
                    r.reused = reused;
                    r.at = reused;
                    r.prefill = t0.elapsed().as_secs_f64();
                    self.metrics.live[slot].prefill(reused, reused);
                    // Prefill begins: the client can tell it from waiting for a slot.
                    if let Some(cb) = &r.job.params.progress {
                        (cb.0)(reused, reused);
                    }
                    self.prefilling.push_back(slot);
                }
                Err(e) => self.fail(slot, &e),
            }
        }
    }

    /// Computes up to `limit` prompt tokens (whole chunks, at least one) of the oldest prefilling
    /// request; at the prompt's end, emits its first token and plans its drafts.
    fn advance(&mut self, limit: usize) {
        let Some(&slot) = self.prefilling.front() else {
            return;
        };
        if self.reqs[slot].as_ref().unwrap().job.tx.is_closed() {
            // The client went away; the chunks computed so far stay cached.
            self.prefilling.pop_front();
            self.reqs[slot] = None;
            self.used[slot] = Instant::now();
            self.metrics.live[slot].end(false);
            return;
        }
        let t = Instant::now();
        let r = self.prefill_slice(slot, limit);
        if let Some(r) = self.reqs[slot].as_mut() {
            r.prefill += t.elapsed().as_secs_f64();
        }
        match r {
            Ok(None) => {}
            Ok(Some(src)) => {
                self.prefilling.pop_front();
                if let Err(e) = self.first_token(slot, src) {
                    self.fail(slot, &e);
                } else if self.reqs[slot].as_ref().is_some_and(|r| r.done.is_some()) {
                    self.complete(slot);
                }
            }
            Err(e) => self.fail(slot, &e),
        }
    }

    /// Some(logits of the prompt's last token) once the prompt is done.
    #[allow(clippy::type_complexity)]
    fn prefill_slice(&mut self, slot: usize, limit: usize) -> Result<Option<<E::Sampler as SlotSampler>::Src>> {
        let chunk = self.e.chunk();
        let mut computed = 0;
        loop {
            let r = self.reqs[slot].as_ref().unwrap();
            let (at, len) = (r.at, r.job.ids.len());
            if at == len {
                // A prompt is never fully cached: rewind keeps at most all but its last token.
                anyhow::bail!("nothing to prefill");
            }
            if computed >= limit {
                return Ok(None);
            }
            let end = (at + chunk).min(len);
            let last = end == len;
            let ids = r.job.ids[at..end].to_vec();
            self.with_memory(slot, |e, st| e.reserve(st, end))?;
            let src = self.with_memory(slot, |e, st| e.prefill(st, &ids, last))?;
            computed += end - at;
            let r = self.reqs[slot].as_mut().unwrap();
            r.at = end;
            self.metrics.live[slot].prefill(end, r.reused);
            if let Some(cb) = &r.job.params.progress {
                self.e.sync()?;
                (cb.0)(end, r.reused);
            }
            if last {
                return Ok(Some(src));
            }
        }
    }

    /// The prompt's end: snapshot, first token, first drafts.
    fn first_token(&mut self, slot: usize, src: <E::Sampler as SlotSampler>::Src) -> Result<()> {
        self.with_memory(slot, |e, st| e.snapshot(st))?;
        self.metrics.live[slot].decoding();
        let r = self.reqs[slot].as_mut().unwrap();
        r.decode_start = Instant::now();
        r.decoding = true;
        let sampler = &mut self.samplers[slot];
        sampler.fetch(src, 1, &r.job.params)?;
        let first = sampler.pick(0, &r.job.params, &r.generated);
        r.done = r.emit(first, &self.eos, self.metrics);
        r.pending = first;
        if r.done.is_some() {
            return Ok(());
        }
        let st = self.states[slot].as_ref().unwrap();
        r.row = self.e.last_t(st) - 1;
        let pos = self.e.pos(st);
        if let Some((tok, row)) = r.plan_draft(pos, self.cfg.context, self.e.max_step()) {
            // The drafter writes its block past the prompt.
            let want = (pos + 2 * self.e.max_step()).min(self.cfg.context);
            self.with_memory(slot, |e, st| e.reserve(st, want))?;
            let st = self.states[slot].as_mut().unwrap();
            let d = self.e.draft(st, tok, row)?;
            self.reqs[slot].as_mut().unwrap().take_draft(d);
        }
        Ok(())
    }

    /// One decode step of every decoding request.
    fn decode(&mut self) {
        let max_step = self.e.max_step();
        // Memory for each request's next step (and its drafter's block); a request that cannot get it
        // ends with an error.
        for slot in 0..self.cfg.slots {
            if !self.reqs[slot].as_ref().is_some_and(|r| r.decoding) {
                continue;
            }
            let pos = self.e.pos(self.states[slot].as_ref().unwrap());
            let want = (pos + 2 * max_step).min(self.cfg.context);
            if let Err(e) = self.with_memory(slot, |e, st| e.reserve(st, want)) {
                self.fail(slot, &e);
            }
        }
        let order: Vec<usize> = (0..self.cfg.slots).filter(|&i| self.reqs[i].as_ref().is_some_and(|r| r.decoding)).collect();
        if order.is_empty() {
            return;
        }
        let tokens: Vec<Vec<i32>> = order
            .iter()
            .map(|&i| {
                let r = self.reqs[i].as_ref().unwrap();
                std::iter::once(r.pending).chain(r.draft.iter().copied()).collect()
            })
            .collect();
        let token_refs: Vec<&[i32]> = tokens.iter().map(Vec::as_slice).collect();
        let mut states: Vec<&mut E::State> = self
            .states
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| order.contains(i))
            .map(|(_, st)| st.as_mut().unwrap())
            .collect();
        let (reqs, samplers, eos, metrics, context) = (&mut self.reqs, &mut self.samplers, &self.eos, self.metrics, self.cfg.context);
        let results = self.e.step(&mut states, &token_refs, &mut |i, src, fed| {
            let slot = order[i];
            reqs[slot].as_mut().unwrap().verify(&mut samplers[slot], src, fed, eos, context, max_step, metrics)
        });
        drop(states);
        for (i, result) in results.into_iter().enumerate() {
            let slot = order[i];
            match result {
                Ok(d) => {
                    let r = self.reqs[slot].as_mut().unwrap();
                    if let Some(d) = d {
                        r.take_draft(d);
                    }
                    if r.done.is_some() {
                        self.complete(slot);
                    }
                }
                Err(e) => self.fail(slot, &e),
            }
        }
    }

    fn publish(&self) {
        let running = self.reqs.iter().flatten().count();
        self.metrics.running.store(running as u64, Ordering::Relaxed);
        for i in 0..self.cfg.slots {
            self.metrics.kv_bytes[i].store(self.device_bytes(i) as u64, Ordering::Relaxed);
        }
    }
}

/// Serves requests until the channel closes or the model is to be unloaded (an unload request, or
/// --unload-idle without requests).
pub(super) fn schedule<E: SlotEngine>(e: E, cfg: &Config, jobs: &mpsc::Receiver<Work>, metrics: &Metrics, waiting: Vec<Work>) -> Exit {
    let setup = || -> Result<Vec<E::Sampler>> { (0..cfg.slots).map(|_| e.new_sampler()).collect() };
    let samplers = match setup() {
        Ok(s) => s,
        Err(err) => {
            eprintln!("{} worker: {err:#}", e.name());
            for w in waiting {
                super::fail_work(w, metrics, &format!("{err:#}"));
            }
            return Exit::Unload(Vec::new());
        }
    };
    let mut s = Sched {
        eos: e.eos().to_vec(),
        drafting: e.drafting(),
        e,
        cfg,
        metrics,
        states: (0..cfg.slots).map(|_| None).collect(),
        samplers,
        reqs: (0..cfg.slots).map(|_| None).collect(),
        used: vec![Instant::now(); cfg.slots],
        prefilling: VecDeque::new(),
        waiting_for_memory: false,
    };
    let mut queue: VecDeque<Job> = VecDeque::new();
    // Unload requests wait until nothing runs or is queued.
    let mut unload: Vec<mpsc::Sender<()>> = Vec::new();
    let take = |w: Work, queue: &mut VecDeque<Job>, unload: &mut Vec<mpsc::Sender<()>>| match w {
        Work::Generate(q) => queue.push_back(q.job),
        Work::Inspect(job) => {
            let _ = job.tx.send(Err("inspection is not supported for this model".into()));
        }
        Work::Scan(job) => {
            let _ = job.tx.send(Err("atlases are not supported for this model".into()));
        }
        Work::Lens(job) => {
            let _ = job.tx.send(Err("not supported for this model".into()));
        }
        Work::Persist(ack) => {
            let _ = ack.send(());
        }
        Work::Unload(ack) => unload.push(ack),
    };
    for w in waiting {
        take(w, &mut queue, &mut unload);
    }
    let mut idle_since = Instant::now();
    // Decode time owed to the running requests for the prefill time spent (see --prefill-share).
    let mut decode_credit = 0.0f64;
    let share = cfg.prefill_share.clamp(0.05, 1.0);
    loop {
        s.publish();
        if !s.busy() && queue.is_empty() {
            if !unload.is_empty() || cfg.unload_idle.is_some_and(|d| idle_since.elapsed() >= d) {
                return Exit::Unload(unload);
            }
            match jobs.recv_timeout(Duration::from_millis(100)) {
                Ok(w) => take(w, &mut queue, &mut unload),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => return Exit::Closed,
            }
        }
        while let Ok(w) = jobs.try_recv() {
            take(w, &mut queue, &mut unload);
        }
        idle_since = Instant::now();
        s.admit(&mut queue);
        s.publish();
        let running = s.reqs.iter().flatten().any(|r| r.decoding);
        if !running || s.prefilling.is_empty() {
            decode_credit = 0.0;
        }
        if !s.prefilling.is_empty() && (!running || decode_credit <= 0.0) {
            let t = Instant::now();
            s.advance(if running { PREFILL_SLICE } else { usize::MAX });
            if running {
                decode_credit += t.elapsed().as_secs_f64() * (1.0 - share) / share;
            }
            continue;
        }
        if !running {
            // Waiting for memory with nothing to run cannot happen (admission then always starts one),
            // but never spin.
            if !queue.is_empty() && s.prefilling.is_empty() {
                std::thread::sleep(Duration::from_millis(1));
            }
            continue;
        }
        let t = Instant::now();
        s.decode();
        decode_credit -= t.elapsed().as_secs_f64();
    }
}

/// The GPU engines' sampler: top candidates of a step's logits rows on the device, sampling on the host
/// ([`crate::flint::Sampler`]); one per slot (its random state is the request's).
pub(super) struct GpuSampler<'s> {
    s: &'s quench6k_cuda::Stream,
    vocab: usize,
    inner: crate::flint::Sampler,
}

impl<'s> GpuSampler<'s> {
    pub(super) fn new(s: &'s quench6k_cuda::Stream, vocab: usize) -> Result<Self> {
        Ok(Self { s, vocab, inner: crate::flint::Sampler::new()? })
    }
}

impl SlotSampler for GpuSampler<'_> {
    /// F32 logits [rows, vocab] on the device.
    type Src = u64;
    fn fetch(&mut self, src: u64, rows: usize, p: &GenParams) -> Result<()> {
        self.inner.fetch(self.s, src, rows, self.vocab, p)
    }
    fn pick(&mut self, row: usize, p: &GenParams, generated: &[i32]) -> i32 {
        self.inner.pick(row, p, generated)
    }
}

/// [`SlotEngine::step`] for engines without a batched step: each sequence's step alone, in turn. Its
/// drafter pass runs right after its own step, while the workspace still holds that step's features.
/// `one` runs a step of one sequence and returns where its logits are; `truncate` drops positions.
pub(super) fn step_each<E: SlotEngine>(
    e: &mut E,
    states: &mut [&mut E::State],
    tokens: &[&[i32]],
    after: &mut After<'_, <E::Sampler as SlotSampler>::Src>,
    mut one: impl FnMut(&mut E, &mut E::State, &[i32]) -> Result<<E::Sampler as SlotSampler>::Src>,
    mut truncate: impl FnMut(&mut E::State, usize),
) -> Vec<Result<Option<Draft>>> {
    let mut out = Vec::with_capacity(states.len());
    for (i, (st, toks)) in states.iter_mut().zip(tokens).enumerate() {
        let mut run = || -> Result<Option<Draft>> {
            let fed = Fed { at: e.pos(st), started: Instant::now() };
            let src = one(e, st, toks)?;
            let v = after(i, src, fed)?;
            truncate(st, fed.at + v.keep);
            v.draft.map(|(tok, row)| e.draft(st, tok, row)).transpose()
        };
        out.push(run());
    }
    out
}

#[cfg(test)]
mod tests;
