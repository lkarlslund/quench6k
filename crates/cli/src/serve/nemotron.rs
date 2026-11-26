//! Nemotron-H checkpoints in the server's slot scheduler ([`super::slots`]): each slot's sequence its own
//! growable [`State`] (attention caches mapped as it grows, its Mamba block and the block's snapshots at
//! canonical boundaries), one shared [`Workspace`]; prompts are prefilled in the scheduler's chunks, which
//! end at multiples of [`CANON_CHUNK`], and a cached prefix is reused only up to such a boundary
//! ([`State::rewind`]), so a sequence's bits depend only on its tokens. The decoding sequences step
//! together through [`NemotronModel::forward_batch`] (each sequence's bits as alone). No drafts yet: the
//! Mamba states keep only what a step feeds (MTP drafting will defer them).

use anyhow::{Result, anyhow};
use quench6k_cuda::Stream;
use quench6k_engine::nemotron::NemotronModel;
use quench6k_engine::nemotron::forward::{CANON_CHUNK, Kv, Logits, MAX_SEQS, State, Workspace};

use super::slots::{After, Draft, Drafter, Drafting, Fed, GpuSampler, Shared, SlotEngine, SlotSampler, StepCandidates};

pub(super) struct Engine<'m> {
    model: &'m NemotronModel,
    s: &'m Stream,
    ws: Workspace,
    kv: Kv,
    /// Sampling candidates of the last batched step (variant nemotron.batch_topk, default on).
    cand: Option<Shared>,
}

impl<'m> Engine<'m> {
    pub(super) fn new(model: &'m NemotronModel, s: &'m Stream) -> Result<Self> {
        let kv = if quench6k_engine::forward::KV_FP8.load(std::sync::atomic::Ordering::Relaxed) { Kv::Fp8 } else { Kv::Bf16 };
        let ws = Workspace::new(model, CANON_CHUNK, MAX_SEQS)?;
        eprintln!(
            "nemotron worker: workspace {:.2} GiB; {} bytes per position ({kv:?} cache), {:.0} MiB of Mamba state per sequence (and per snapshot)",
            ws.device_bytes as f64 / (1u64 << 30) as f64,
            State::position_bytes(&model.cfg, kv),
            State::state_bytes(&model.cfg) as f64 / (1u64 << 20) as f64
        );
        let cand = StepCandidates::shared("nemotron", model.cfg.vocab, MAX_SEQS)?;
        Ok(Self { model, s, ws, kv, cand })
    }
}

impl<'m> SlotEngine for Engine<'m> {
    type State = State;
    type Sampler = GpuSampler<'m>;

    fn name(&self) -> &'static str {
        "nemotron"
    }
    fn chunk(&self) -> usize {
        CANON_CHUNK
    }
    fn max_step(&self) -> usize {
        1
    }
    fn eos(&self) -> &[i32] {
        &self.model.cfg.eos
    }
    fn drafting(&self) -> Drafting {
        Drafting {
            kind: Drafter::None,
            drafts: 0,
            costs: (0.0125, 0.0, 0.0),
            lookup: false,
        }
    }
    fn new_sampler(&self) -> Result<Self::Sampler> {
        GpuSampler::new(self.s, self.model.cfg.vocab, self.cand.clone())
    }
    fn new_state(&mut self, capacity: usize) -> Result<State> {
        State::new(self.model, capacity, self.kv, self.s)
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
        if let Err(e) = st.trim(positions, self.s) {
            eprintln!("nemotron worker: trim: {e:#}");
        }
    }
    fn device_bytes(&self, st: &State) -> usize {
        st.device_bytes()
    }
    fn bytes_per_position(&self) -> usize {
        State::position_bytes(&self.model.cfg, self.kv)
    }
    fn plan_bytes(&self, positions: usize) -> usize {
        State::plan_bytes(&self.model.cfg, self.kv, positions)
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
    fn snapshot(&mut self, _st: &mut State) -> Result<()> {
        // Snapshots are taken as whole canonical chunks complete (State::rewind).
        Ok(())
    }
    fn reset(&mut self, st: &mut State) {
        if let Err(e) = st.reset(self.s) {
            eprintln!("nemotron worker: reset: {e:#}");
        }
    }
    fn sync(&self) -> Result<()> {
        self.s.synchronize()
    }
    fn prefill(&mut self, st: &mut State, tokens: &[i32], last: bool) -> Result<u64> {
        if let Some(c) = &self.cand {
            // The logits are about to be overwritten.
            c.borrow_mut().invalidate();
        }
        // The scheduler's chunks end at multiples of CANON_CHUNK: one canonical chunk per call.
        let want = if last { Logits::Last } else { Logits::None };
        self.model.forward(st, &mut self.ws, self.s, tokens, want)?;
        Ok(self.ws.out_logits)
    }
    fn draft(&mut self, _st: &mut State, _token: i32, _row: usize, _room: usize) -> Result<Draft> {
        Ok(Draft { tokens: Vec::new(), conf: Vec::new(), secs: 0.0 })
    }
    fn step(
        &mut self,
        states: &mut [&mut State],
        tokens: &[&[i32]],
        after: &mut After<'_, <Self::Sampler as SlotSampler>::Src>,
    ) -> Vec<Result<Option<Draft>>> {
        let started = std::time::Instant::now();
        let at: Vec<usize> = states.iter().map(|st| st.pos).collect();
        let rows: usize = tokens.iter().map(|t| t.len()).sum();
        if let Some(c) = &self.cand {
            c.borrow_mut().invalidate();
        }
        let run = self.model.forward_batch(states, &mut self.ws, self.s, tokens).and_then(|()| match &self.cand {
            Some(c) => c.borrow_mut().launch(self.s, self.ws.out_logits, rows),
            None => Ok(()),
        });
        if let Err(e) = run {
            let msg = format!("{e:#}");
            return states.iter().map(|_| Err(anyhow!("{msg}"))).collect();
        }
        // Logits sequence by sequence, row by row.
        let row_bytes = (self.model.cfg.vocab * 4) as u64;
        let mut row = 0;
        let mut out = Vec::with_capacity(states.len());
        for (i, st) in states.iter_mut().enumerate() {
            let src = self.ws.out_logits + row as u64 * row_bytes;
            row += tokens[i].len();
            out.push(after(i, src, Fed { at: at[i], started }).and_then(|v| {
                st.truncate(at[i] + v.keep, self.s)?;
                Ok(None)
            }));
        }
        out
    }
}
