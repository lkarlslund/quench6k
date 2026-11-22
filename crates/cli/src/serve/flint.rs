//! Flint checkpoints in the server's slot scheduler ([`super::slots`]): each slot's sequence its own
//! [`fl::FlintSeq`], one shared [`fl::Workspace`]; prompt-lookup drafts (Flint has no drafter).
//!
//! Interim (until the engine has a batched step and growable sequences): a step runs the sequences'
//! steps one after another, and every sequence holds its whole capacity from the start, so `--slots`
//! sequences of `--context` positions must fit next to each other for all slots to run at once (idle
//! slots' sequences are dropped to make room).

use anyhow::Result;
use quench6k_cuda::Stream;
use quench6k_engine::flint as fl;

use super::slots::{After, Draft, Drafter, Drafting, GpuSampler, SlotEngine, SlotSampler, step_each};

pub(super) struct Engine<'m> {
    model: &'m fl::FlintModel,
    s: &'m Stream,
    ws: fl::Workspace,
    /// Bytes of a sequence (all the same size: --context positions).
    state_bytes: usize,
    capacity: usize,
}

impl<'m> Engine<'m> {
    pub(super) fn new(model: &'m fl::FlintModel, s: &'m Stream) -> Result<Self> {
        Ok(Self { model, s, ws: fl::Workspace::new(fl::MAX_STEP)?, state_bytes: 0, capacity: 0 })
    }
}

fn seq_bytes(seq: &fl::FlintSeq) -> usize {
    seq.k.iter().chain(&seq.v).map(|b| b.bytes()).sum()
}

impl<'m> SlotEngine for Engine<'m> {
    type State = fl::FlintSeq;
    type Sampler = GpuSampler<'m>;

    fn name(&self) -> &'static str {
        "flint"
    }
    fn chunk(&self) -> usize {
        fl::MAX_CHUNK
    }
    fn max_step(&self) -> usize {
        fl::MAX_STEP
    }
    fn eos(&self) -> &[i32] {
        // The chat's stop tokens arrive with each request.
        &[]
    }
    fn drafting(&self) -> Drafting {
        Drafting {
            kind: Drafter::None,
            drafts: 0,
            costs: (0.01, 0.001, 0.001),
            lookup: quench6k_cuda::variants::get("flint.lookup", 1) != 0,
        }
    }
    fn new_sampler(&self) -> Result<Self::Sampler> {
        GpuSampler::new(self.s, fl::VOCAB)
    }
    fn new_state(&mut self, capacity: usize) -> Result<fl::FlintSeq> {
        let seq = fl::FlintSeq::new(self.model, capacity)?;
        // FlintSeq::new leaves its caches uninitialized; start from zeros as the other engines' states do.
        for b in seq.k.iter().chain(&seq.v) {
            self.s.memset(b.addr(), 0, b.bytes())?;
        }
        self.state_bytes = seq_bytes(&seq);
        self.capacity = capacity;
        Ok(seq)
    }
    fn growable(&self) -> bool {
        false
    }
    fn batched(&self) -> bool {
        false
    }
    fn bytes_per_position(&self) -> usize {
        // Interim: a state's whole size spread over its positions.
        self.state_bytes / self.capacity.max(1)
    }
    fn reserve(&mut self, _st: &mut fl::FlintSeq, _positions: usize) -> Result<()> {
        Ok(())
    }
    fn trim(&mut self, st: &mut fl::FlintSeq, positions: usize) {
        if positions == 0 {
            st.tokens.clear();
        }
    }
    fn device_bytes(&self, st: &fl::FlintSeq) -> usize {
        seq_bytes(st)
    }
    fn plan_bytes(&self, _positions: usize) -> usize {
        self.state_bytes
    }
    fn free_bytes(&self) -> usize {
        quench6k_cuda::mem_info().map_or(0, |(f, _)| f)
    }
    fn history<'a>(&self, st: &'a fl::FlintSeq) -> &'a [i32] {
        &st.tokens
    }
    fn pos(&self, st: &fl::FlintSeq) -> usize {
        st.tokens.len()
    }
    fn last_t(&self, _st: &fl::FlintSeq) -> usize {
        1
    }
    fn rewind(&mut self, st: &mut fl::FlintSeq, prompt: &[i32]) -> Result<usize> {
        st.rewind(prompt, self.s)
    }
    fn snapshot(&mut self, st: &mut fl::FlintSeq) -> Result<()> {
        st.snapshot(self.model, self.s)
    }
    fn reset(&mut self, st: &mut fl::FlintSeq) {
        st.tokens.clear();
    }
    fn sync(&self) -> Result<()> {
        self.s.synchronize()
    }
    fn prefill(&mut self, st: &mut fl::FlintSeq, tokens: &[i32], _last: bool) -> Result<u64> {
        fl::forward(self.model, &self.ws, st, tokens, false, self.s)?;
        Ok(self.ws.logits)
    }
    fn draft(&mut self, _st: &mut fl::FlintSeq, _token: i32, _row: usize) -> Result<Draft> {
        Ok(Draft { tokens: Vec::new(), conf: Vec::new(), secs: 0.0 })
    }
    fn step(
        &mut self,
        states: &mut [&mut fl::FlintSeq],
        tokens: &[&[i32]],
        after: &mut After<'_, <Self::Sampler as SlotSampler>::Src>,
    ) -> Vec<Result<Option<Draft>>> {
        step_each(
            self,
            states,
            tokens,
            after,
            |e, st, toks| {
                fl::step(e.model, &mut e.ws, st, toks, e.s)?;
                Ok(e.ws.logits)
            },
            |st, pos| st.tokens.truncate(pos),
        )
    }
}
