//! Hy4-flash checkpoints in the server's slot scheduler ([`super::slots`]): each slot's sequence its own
//! growable [`State`] (its full-length caches mapped as it grows), one shared [`Workspace`]; the decoding
//! sequences step together through [`Hy4Model::forward_batch`] and draft together through
//! [`Hy4Model::draft_batch`] (each sequence's bits as alone). Drafts come from the checkpoint's DFly
//! drafter (as many as [`DraftPolicy`](quench6k_engine::hy4::policy::DraftPolicy) expects to pay off), or
//! from prompt lookup without one.

use anyhow::{Result, anyhow};
use quench6k_cuda::Stream;
use quench6k_engine::hy4::Hy4Model;
use quench6k_engine::hy4::forward::{Logits, MAX_SEQS, MAX_STEP, State, Workspace};

use super::slots::{After, Draft, Drafter, Drafting, Fed, GpuSampler, SlotEngine, SlotSampler};

/// Prefill chunk (tokens per forward step).
const CHUNK: usize = 2048;

pub(super) struct Engine<'m> {
    model: &'m Hy4Model,
    s: &'m Stream,
    ws: Workspace,
    /// Device bytes of a new state (its rings, drafter buffers and snapshot), before any position.
    fixed: usize,
    per_position: usize,
}

impl<'m> Engine<'m> {
    pub(super) fn new(model: &'m Hy4Model, s: &'m Stream) -> Result<Self> {
        // Logits of a whole batch: MAX_SEQS sequences of MAX_STEP tokens.
        let ws = Workspace::new(model, CHUNK, 0, MAX_SEQS * MAX_STEP)?;
        let fixed = State::new(model, 4, s)?.device_bytes();
        let per_position = State::bytes_per_position(model);
        eprintln!(
            "hy4 worker: workspace {:.2} GiB; per sequence {:.0} MiB and {per_position} bytes per position",
            ws.device_bytes as f64 / (1u64 << 30) as f64,
            fixed as f64 / (1u64 << 20) as f64
        );
        Ok(Self { model, s, ws, fixed, per_position })
    }
}

impl<'m> SlotEngine for Engine<'m> {
    type State = State;
    type Sampler = GpuSampler<'m>;

    fn name(&self) -> &'static str {
        "hy4"
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
        let f = self.model.dfly.as_ref();
        Drafting {
            kind: if f.is_some() { Drafter::Count } else { Drafter::None },
            drafts: f.map_or(0, |f| f.width - 1),
            costs: (0.010, 0.0018, 0.0024),
            lookup: quench6k_cuda::variants::get("hy4.lookup", 1) != 0,
        }
    }
    fn new_sampler(&self) -> Result<Self::Sampler> {
        GpuSampler::new(self.s, self.model.cfg.vocab)
    }
    fn new_state(&mut self, capacity: usize) -> Result<State> {
        State::new(self.model, capacity, self.s)
    }
    fn growable(&self) -> bool {
        true
    }
    fn batched(&self) -> bool {
        true
    }
    fn reserve(&mut self, st: &mut State, positions: usize) -> Result<()> {
        st.reserve(positions)
    }
    fn trim(&mut self, st: &mut State, positions: usize) {
        st.trim(positions);
    }
    fn device_bytes(&self, st: &State) -> usize {
        st.device_bytes()
    }
    fn bytes_per_position(&self) -> usize {
        self.per_position
    }
    fn plan_bytes(&self, positions: usize) -> usize {
        self.fixed + positions * self.per_position
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
        Ok(Draft { tokens, conf: Vec::new(), secs: t.elapsed().as_secs_f64() })
    }
    fn step(
        &mut self,
        states: &mut [&mut State],
        tokens: &[&[i32]],
        after: &mut After<'_, <Self::Sampler as SlotSampler>::Src>,
    ) -> Vec<Result<Option<Draft>>> {
        let started = std::time::Instant::now();
        let at: Vec<usize> = states.iter().map(|st| st.pos).collect();
        if let Err(e) = self.model.forward_batch(states, &mut self.ws, self.s, tokens) {
            let msg = format!("{e:#}");
            return states.iter().map(|_| Err(anyhow!("{msg}"))).collect();
        }
        // Logits sequence by sequence, row by row.
        let row_bytes = (self.model.cfg.vocab * 4) as u64;
        let mut row = 0;
        let mut out: Vec<Result<Option<Draft>>> = Vec::with_capacity(states.len());
        let mut wanted: Vec<(usize, i32, usize)> = Vec::new();
        for (i, st) in states.iter_mut().enumerate() {
            let src = self.ws.out_logits + row as u64 * row_bytes;
            row += tokens[i].len();
            match after(i, src, Fed { at: at[i], started }) {
                Ok(v) => {
                    st.truncate(at[i] + v.keep);
                    if let Some((tok, r)) = v.draft {
                        wanted.push((i, tok, r));
                    }
                    out.push(Ok(None));
                }
                Err(e) => out.push(Err(e)),
            }
        }
        if wanted.is_empty() {
            return out;
        }
        // One drafter pass for every sequence that asked; its time is shared out evenly.
        let t = std::time::Instant::now();
        let pending: Vec<i32> = wanted.iter().map(|w| w.1).collect();
        let rows: Vec<usize> = wanted.iter().map(|w| w.2).collect();
        let mut sub: Vec<&mut State> = states
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| wanted.iter().any(|w| w.0 == *i))
            .map(|(_, st)| &mut **st)
            .collect();
        match self.model.draft_batch(&mut sub, &mut self.ws, self.s, &pending, &rows) {
            Ok(drafts) => {
                let secs = t.elapsed().as_secs_f64() / wanted.len() as f64;
                for (w, tokens) in wanted.iter().zip(drafts) {
                    out[w.0] = Ok(Some(Draft { tokens, conf: Vec::new(), secs }));
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                for w in &wanted {
                    out[w.0] = Err(anyhow!("drafting: {msg}"));
                }
            }
        }
        out
    }
}
