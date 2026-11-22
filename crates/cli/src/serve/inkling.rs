//! Inkling 2 checkpoints in the server's slot scheduler ([`super::slots`]): each slot's sequence its own
//! [`State`], one shared [`Workspace`]; drafts from the checkpoint's DSpark drafter (as many as
//! [`DraftPolicy`](quench6k_engine::hy4::policy::DraftPolicy) expects to pay off, from the drafter's
//! stated probabilities), or prompt-lookup drafts without one.
//!
//! Interim (until the engine has a batched step and growable states): a step runs the sequences' steps
//! one after another, and every state holds its whole capacity from the start, so `--slots` states of
//! `--context` positions must fit next to each other for all slots to run at once (idle slots' states
//! are dropped to make room).

use anyhow::Result;
use quench6k_cuda::Stream;
use quench6k_engine::inkling::InklingModel;
use quench6k_engine::inkling::forward::{Logits, State, Workspace};

use super::slots::{After, Draft, Drafter, Drafting, GpuSampler, SlotEngine, SlotSampler, step_each};

/// Prefill chunk (tokens per forward step).
const CHUNK: usize = 2048;
/// Most tokens of a verification step (the pending token and its drafts); the sampler takes 8 rows.
const MAX_STEP: usize = 8;

pub(super) struct Engine<'m> {
    model: &'m InklingModel,
    s: &'m Stream,
    ws: Workspace,
    /// Bytes of a state (all the same size: --context positions).
    state_bytes: usize,
}

impl<'m> Engine<'m> {
    pub(super) fn new(model: &'m InklingModel, s: &'m Stream) -> Result<Self> {
        let ws = Workspace::new(model, CHUNK, MAX_STEP)?;
        eprintln!("inkling worker: workspace {:.2} GiB", ws.device_bytes as f64 / (1u64 << 30) as f64);
        Ok(Self { model, s, ws, state_bytes: 0 })
    }
}

impl<'m> SlotEngine for Engine<'m> {
    type State = State;
    type Sampler = GpuSampler<'m>;

    fn name(&self) -> &'static str {
        "inkling"
    }
    fn chunk(&self) -> usize {
        CHUNK
    }
    fn max_step(&self) -> usize {
        MAX_STEP
    }
    fn eos(&self) -> &[i32] {
        &self.model.cfg.eos
    }
    fn drafting(&self) -> Drafting {
        let d = self.model.drafter.as_ref();
        Drafting {
            kind: if d.is_some() { Drafter::Conf } else { Drafter::None },
            drafts: d.map_or(0, |d| d.block - 1),
            costs: (0.0045, 0.0008, 0.0006),
            lookup: quench6k_cuda::variants::get("inkling.lookup", 1) != 0,
        }
    }
    fn new_sampler(&self) -> Result<Self::Sampler> {
        GpuSampler::new(self.s, self.model.cfg.vocab)
    }
    fn new_state(&mut self, capacity: usize) -> Result<State> {
        let st = State::new(self.model, capacity, self.s)?;
        self.state_bytes = st.device_bytes;
        Ok(st)
    }
    fn growable(&self) -> bool {
        false
    }
    fn reserve(&mut self, _st: &mut State, _positions: usize) -> Result<()> {
        Ok(())
    }
    fn trim(&mut self, st: &mut State, positions: usize) {
        if positions == 0 {
            st.reset();
        }
    }
    fn device_bytes(&self, st: &State) -> usize {
        st.device_bytes
    }
    fn plan_bytes(&self, _positions: usize) -> usize {
        self.state_bytes
    }
    fn free_bytes(&self) -> usize {
        quench6k_cuda::mem_info().map_or(0, |(f, _)| f)
    }
    fn history<'a>(&self, st: &'a State) -> &'a [i32] {
        &st.history
    }
    fn pos(&self, st: &State) -> usize {
        st.pos
    }
    fn last_t(&self, st: &State) -> usize {
        st.last_t
    }
    fn rewind(&mut self, st: &mut State, prompt: &[i32]) -> Result<usize> {
        st.rewind(prompt, self.s)
    }
    fn snapshot(&mut self, st: &mut State) -> Result<()> {
        st.snapshot(self.s)
    }
    fn reset(&mut self, st: &mut State) {
        st.reset();
    }
    fn sync(&self) -> Result<()> {
        self.s.synchronize()
    }
    fn prefill(&mut self, st: &mut State, tokens: &[i32], last: bool) -> Result<u64> {
        let want = if last { Logits::Last } else { Logits::None };
        self.model.forward(st, &mut self.ws, self.s, tokens, want)?;
        Ok(self.ws.out_logits)
    }
    fn draft(&mut self, st: &mut State, token: i32, row: usize) -> Result<Draft> {
        let t = std::time::Instant::now();
        let tokens = self.model.draft(st, &mut self.ws, self.s, token, row)?;
        let conf = if tokens.is_empty() { Vec::new() } else { self.model.draft_conf(&self.ws) };
        Ok(Draft { tokens, conf, secs: t.elapsed().as_secs_f64() })
    }
    fn step(
        &mut self,
        states: &mut [&mut State],
        tokens: &[&[i32]],
        after: &mut After<'_, <Self::Sampler as SlotSampler>::Src>,
    ) -> Vec<Result<Option<Draft>>> {
        step_each(
            self,
            states,
            tokens,
            after,
            |e, st, toks| {
                let want = if toks.len() == 1 { Logits::Last } else { Logits::All };
                e.model.forward(st, &mut e.ws, e.s, toks, want)?;
                Ok(e.ws.out_logits)
            },
            |st, pos| st.truncate(pos),
        )
    }
}
