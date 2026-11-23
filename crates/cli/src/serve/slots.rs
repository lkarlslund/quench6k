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
//!   first, the whole state); a request that does not fit waits for running ones to finish, and one that
//!   does not fit with nothing else on the device fails. Running requests that together outgrow the
//!   device: the newest one beyond its plan ends (else the newest one), with an error that says so;
//!   requests within their plans go on.
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

/// A batched engine's step time against the rows it feeds, T = a + b R, fitted over recent steps (with
/// a prior from the model's draft costs). A sequence's share of a step is the fixed part split evenly
/// plus its own rows: what the sequence costs the batch. The draft policy sees that share as the step's
/// time, so with many sequences running (where a row costs about as much as the fixed part is worth to
/// each) drafts are verified only when they pay for their rows.
/// A batched step about to run several sequences, with its fitted cost T = a + b R.
#[derive(Clone, Copy)]
struct Batch {
    seqs: usize,
    a: f64,
    b: f64,
}

impl Batch {
    /// Most drafts worth verifying for a sequence whose drafts are each accepted with chance `q` (given
    /// the ones before it) and whose drafter pass costs `pass` seconds: the count k with the largest
    /// net gain q + q^2 + .. + q^k - rate (k b + pass), where rate is what the batch's time is worth in
    /// plain steps (seqs tokens per a + b seqs seconds). Alone this is left to the draft policy.
    fn cap(&self, q: f64, pass: f64, most: usize) -> usize {
        let rate = self.seqs as f64 / (self.a + self.b * self.seqs as f64);
        let (mut best, mut best_net) = (0, 0.0);
        let (mut gain, mut p) = (0.0, 1.0);
        for k in 1..=most {
            p *= q;
            gain += p;
            let net = gain - rate * (k as f64 * self.b + pass);
            if net > best_net {
                best = k;
                best_net = net;
            }
        }
        best
    }
}

struct StepCost {
    prior: (f64, f64),
    /// Decayed sums of 1, R, R^2, T, R T.
    sums: [f64; 5],
}

impl StepCost {
    fn new(one_token: f64, per_token: f64) -> Self {
        Self { prior: ((one_token - per_token).max(per_token), per_token), sums: [0.0; 5] }
    }

    fn add(&mut self, rows: f64, secs: f64) {
        for v in &mut self.sums {
            *v *= 0.98;
        }
        for (v, x) in self.sums.iter_mut().zip([1.0, rows, rows * rows, secs, rows * secs]) {
            *v += x;
        }
    }

    /// (a, b): least squares over the recent steps plus prior steps of 1 and 16 rows (weight 2 each).
    fn fit(&self) -> (f64, f64) {
        let (a0, b0) = self.prior;
        let mut m = self.sums;
        for r in [1.0, 16.0] {
            let t = a0 + b0 * r;
            for (v, x) in m.iter_mut().zip([1.0, r, r * r, t, r * t]) {
                *v += 2.0 * x;
            }
        }
        let det = m[0] * m[2] - m[1] * m[1];
        let b = ((m[0] * m[4] - m[1] * m[3]) / det).max(b0 * 0.1);
        let a = ((m[3] - b * m[1]) / m[0]).max(0.0);
        (a, b)
    }

    /// What the next step of `seqs` sequences looks like to a sequence planning its drafts (None alone).
    fn batch(&self, seqs: usize) -> Option<Batch> {
        (seqs > 1).then(|| {
            let (a, b) = self.fit();
            Batch { seqs, a, b }
        })
    }

    /// The share of a step of `seqs` sequences and `rows` rows that a sequence feeding `own` rows costs.
    fn share(&self, seqs: usize, own: usize, rows: usize) -> f64 {
        if seqs <= 1 {
            return 1.0;
        }
        let (a, b) = self.fit();
        (a / seqs as f64 + b * own as f64) / (a + b * rows as f64)
    }
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
    /// Whether [`SlotEngine::step`] runs its sequences in one batched pass (their step time is shared)
    /// rather than one after another.
    fn batched(&self) -> bool;
    /// Device bytes per position of a state's full-length caches.
    fn bytes_per_position(&self) -> usize;
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
    /// Admission order (the newest request gives way first when running requests outgrow the device).
    order: u64,
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
    /// Chance a draft is accepted given the ones before it, and the drafter pass's seconds (this
    /// request's share), for the cap on drafts while several sequences step together (Batch::cap).
    q: f64,
    pass: f64,
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
    fn plan_draft(&mut self, pos: usize, context: usize, max_step: usize, batch: Option<Batch>) -> Option<(i32, usize)> {
        self.draft.clear();
        self.conf.clear();
        self.draft_secs = None;
        // Several sequences stepping together: no more drafts than pay for their rows in the batch
        // (every 8th step one at least, to keep the acceptance estimate current).
        let limit = batch.map_or(max_step - 1, |b| {
            let k = b.cap(self.q, self.pass, max_step - 1);
            if k == 0 && self.steps % 8 == 7 { 1 } else { k }
        });
        let cap = if self.rate < 0.15 {
            1
        } else if self.rate < 0.35 {
            2
        } else if self.rate < 0.6 {
            4
        } else {
            max_step - 1
        };
        self.room = (self.budget - self.generated.len()).min(context.saturating_sub(pos + 1)).min(limit);
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
        share: f64,
        batch: Option<Batch>,
        eos: &[i32],
        context: usize,
        max_step: usize,
        metrics: &Metrics,
    ) -> Result<Verdict> {
        let draft = std::mem::take(&mut self.draft);
        sampler.fetch(src, draft.len() + 1, &self.job.params)?;
        // The step's time this sequence accounts for (all of it when it ran alone).
        let verify_secs = fed.started.elapsed().as_secs_f64() * share;
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
            for k in 0..draft.len().min(j + 1) {
                self.q += 0.1 * ((k < j) as u8 as f64 - self.q);
            }
        }
        if let Some(d) = self.draft_secs {
            self.pass += 0.2 * (d - self.pass);
        }
        match self.mode {
            Mode::Conf => self.policy.update_conf(&self.conf, draft.len(), j, verify_secs, self.draft_secs),
            Mode::Count => self.policy.update(draft.len(), j, verify_secs, self.draft_secs),
            _ => {}
        }
        self.row = j;
        self.done = done;
        let next = if done.is_none() {
            self.plan_draft(fed.at + 1 + j, context, max_step, batch)
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
    cost: StepCost,
    /// Requests admitted so far.
    admitted: u64,
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

    /// Frees device memory for `keep`: mapped memory past the positions other slots hold or (decoding
    /// requests) their next step needs (growable states), else the whole state of the least recently
    /// used idle slot (its caches and its fixed buffers: an emptied state keeps nothing worth its rings).
    /// False when nothing is left.
    fn free_memory(&mut self, keep: usize) -> bool {
        if self.e.growable() {
            let before: usize = (0..self.cfg.slots).map(|i| self.device_bytes(i)).sum();
            let next_step = 2 * self.e.max_step();
            for i in (0..self.cfg.slots).filter(|&i| i != keep) {
                let decoding = self.reqs[i].as_ref().is_some_and(|r| r.decoding);
                if let Some(st) = self.states[i].as_mut() {
                    // A decoding request's next step was mapped already (decode maps every request's
                    // before the step): keep it.
                    let pos = self.e.pos(st);
                    let hold = if decoding { (pos + next_step).min(self.cfg.context) } else { pos };
                    self.e.trim(st, hold);
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
        // The device may still use the state's buffers.
        if let Err(e) = self.e.sync() {
            eprintln!("{} worker: {e:#}", self.e.name());
        }
        self.states[i] = None;
        eprintln!("{} worker: slot {i}: context of {held} tokens dropped from device memory", self.e.name());
        self.metrics.kv_bytes[i].store(self.device_bytes(i) as u64, Ordering::Relaxed);
        true
    }

    /// Positions one sequence could hold with every other slot's memory given up: free device memory
    /// (less the headroom) and what all states hold now.
    fn positions_alone(&self) -> usize {
        let held: usize = (0..self.cfg.slots).map(|i| self.device_bytes(i)).sum();
        let avail = (self.e.free_bytes() + held).saturating_sub(HEADROOM);
        let fixed = self.e.plan_bytes(0);
        let per = self.e.bytes_per_position().max(1);
        (avail.saturating_sub(fixed) / per).min(self.cfg.context)
    }

    /// Why memory ran out on `slot` at `positions`.
    fn memory_note(&self, slot: usize, positions: usize) -> String {
        let others = (0..self.cfg.slots).filter(|&i| i != slot && self.reqs[i].is_some()).count();
        if others > 0 {
            format!(
                "slot {slot}: device memory for {positions} positions is held by {others} running request(s) (this request is the newest{})",
                if self.beyond_plan(slot) { " beyond its plan" } else { "" }
            )
        } else {
            format!(
                "slot {slot}: device memory is full at {positions} positions: one sequence fits about {} positions on this device (--context {})",
                self.positions_alone(),
                self.cfg.context
            )
        }
    }

    /// Whether running request `slot` has grown beyond the positions its admission planned for.
    fn beyond_plan(&self, slot: usize) -> bool {
        match (&self.reqs[slot], &self.states[slot]) {
            (Some(r), Some(st)) => self.e.pos(st) + 2 * self.e.max_step() > r.planned,
            _ => false,
        }
    }

    /// The running request that gives way when running requests outgrow the device: the newest one
    /// beyond its plan, else the newest one (`slot` among them).
    fn gives_way(&self) -> Option<usize> {
        let running = || (0..self.cfg.slots).filter(|&i| self.reqs[i].is_some());
        let order = |i: &usize| self.reqs[*i].as_ref().map_or(0, |r| r.order);
        running().filter(|&i| self.beyond_plan(i)).max_by_key(order).or_else(|| running().max_by_key(order))
    }

    /// Runs `f` on `slot`'s state, freeing memory elsewhere while it fails for lack of it: idle slots'
    /// memory first, then the running request that gives way ([`Sched::gives_way`]; when that is `slot`'s
    /// own, the error).
    fn with_memory<T>(&mut self, slot: usize, mut f: impl FnMut(&mut E, &mut E::State) -> Result<T>) -> Result<T> {
        loop {
            let st = self.states[slot].as_mut().expect("slot state");
            match f(&mut self.e, st) {
                Err(e) if is_oom(&e) => {
                    if self.free_memory(slot) {
                        continue;
                    }
                    let pos = self.states[slot].as_ref().map_or(0, |st| self.e.pos(st));
                    match self.gives_way() {
                        Some(v) if v != slot => {
                            let note = self.give_way_note(v, slot, pos);
                            self.fail(v, &anyhow::anyhow!("{note}"));
                            // Its memory goes now (it is idle, but the most recently used).
                            let _ = self.e.sync();
                            self.states[v] = None;
                            self.metrics.kv_bytes[v].store(0, Ordering::Relaxed);
                        }
                        _ => return Err(e.context(self.memory_note(slot, pos))),
                    }
                }
                r => return r,
            }
        }
    }

    /// Why running request `v` ended for `slot`'s memory at `pos` positions.
    fn give_way_note(&self, v: usize, slot: usize, pos: usize) -> String {
        let (r, at) = (self.reqs[v].as_ref().unwrap(), self.states[v].as_ref().map_or(0, |st| self.e.pos(st)));
        format!(
            "device memory is full: slot {v} stopped at {at} positions ({}) so that slot {slot} can go on at {pos} positions; {} requests were running",
            if self.beyond_plan(v) {
                format!("the newest request beyond its plan of {} positions", r.planned)
            } else {
                "the newest request".to_string()
            },
            self.reqs.iter().flatten().count()
        )
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
            // The free slot whose conversation the request continues (it holds the most of the prompt,
            // at least half of what it holds), else the least recently used one: a slot is not taken
            // from another conversation for a shared beginning (a system prompt, a template header).
            let free = (0..self.cfg.slots)
                .filter(|&i| self.reqs[i].is_none())
                .max_by_key(|&i| {
                    let (common, held) = self.states[i].as_ref().map_or((0, 0), |st| {
                        let h = self.e.history(st);
                        (h.iter().zip(&job.ids).take_while(|(a, b)| a == b).count(), h.len())
                    });
                    let continues = if 2 * common >= held { common.min(job.ids.len() - 1) } else { 0 };
                    (continues, Reverse(self.used[i]), Reverse(i))
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
            // A prompt whose caches cannot fit on the device even alone fails now (growable states:
            // the per-request limit is memory, not only --context). After the slot has its state, so
            // that the engine knows a state's fixed size.
            if self.e.growable() && !self.busy() && job.ids.len() + 1 > self.positions_alone() {
                let job = queue.pop_front().unwrap();
                self.metrics.queued.fetch_sub(1, Ordering::Relaxed);
                let msg = format!(
                    "prompt of {} tokens does not fit in device memory: one sequence fits about {} positions on this device (--context {})",
                    job.ids.len(),
                    self.positions_alone(),
                    self.cfg.context
                );
                eprintln!("{} worker: {msg}", self.e.name());
                let _ = job.tx.send(Msg::Error(msg));
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
            self.admitted += 1;
            let r = Req {
                hist: job.ids.clone(),
                job,
                slot,
                order: self.admitted,
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
                q: 0.6,
                pass: d.costs.2,
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
            // Chunks end at multiples of the chunk size (or the prompt's end) whatever was reused, so a
            // prompt's prefill is the same computation however much of it was cached.
            let end = ((at / chunk + 1) * chunk).min(len);
            let last = end == len;
            let ids = r.job.ids[at..end].to_vec();
            self.with_memory(slot, |e, st| e.reserve(st, end))?;
            // A chunk that ran out of memory after it advanced the sequence (an engine allocating at a
            // boundary) is not fed again.
            let src = self.with_memory(slot, |e, st| {
                anyhow::ensure!(e.pos(st) == at, "prefill chunk at {at} failed after advancing the sequence to {}", e.pos(st));
                e.prefill(st, &ids, last)
            })?;
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
        let decoding = self.reqs.iter().flatten().filter(|r| r.decoding).count();
        let batch = if self.e.batched() { self.cost.batch(decoding) } else { None };
        let r = self.reqs[slot].as_mut().unwrap();
        if let Some((tok, row)) = r.plan_draft(pos, self.cfg.context, self.e.max_step(), batch) {
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
        // A batched step's time is shared out by what each sequence costs it (see StepCost); steps run
        // one after another time each sequence alone.
        let rows: usize = tokens.iter().map(Vec::len).sum();
        let batched = self.e.batched();
        let shares: Vec<f64> = tokens
            .iter()
            .map(|t| if batched { self.cost.share(tokens.len(), t.len(), rows) } else { 1.0 })
            .collect();
        let mut step_secs = 0.0f64;
        let batch = if batched { self.cost.batch(tokens.len()) } else { None };
        let (reqs, samplers, eos, metrics, context) = (&mut self.reqs, &mut self.samplers, &self.eos, self.metrics, self.cfg.context);
        let results = self.e.step(&mut states, &token_refs, &mut |i, src, fed| {
            let slot = order[i];
            let v = reqs[slot].as_mut().unwrap().verify(&mut samplers[slot], src, fed, shares[i], batch, eos, context, max_step, metrics);
            step_secs = step_secs.max(fed.started.elapsed().as_secs_f64());
            v
        });
        drop(states);
        if batched && step_secs > 0.0 {
            self.cost.add(rows as f64, step_secs);
        }
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
    let drafting_costs = e.drafting().costs;
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
        cost: StepCost::new(drafting_costs.0, drafting_costs.1),
        admitted: 0,
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
#[cfg(test)]
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
