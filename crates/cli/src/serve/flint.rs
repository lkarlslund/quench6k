//! Flint checkpoints in the server's slot scheduler ([`super::slots`]): each slot's sequence its own
//! growable [`fl::FlintState`] (full-attention caches mapped as it grows), one shared [`fl::Workspace`];
//! the decoding sequences step together through [`fl::FlintModel::forward_batch`] (each sequence's bits
//! as alone); prompt-lookup drafts (Flint has no drafter). A batched step's sampling candidates come
//! from the device together ([`StepCandidates`]) rather than in a top-k launch and download per sequence.

use anyhow::{Result, anyhow};
use quench6k_cuda::Stream;
use quench6k_engine::flint as fl;

use super::slots::{After, Draft, Drafter, Drafting, Fed, GpuSampler, Shared, SlotEngine, SlotSampler, StepCandidates};

pub(super) struct Engine<'m> {
    model: &'m fl::FlintModel,
    s: &'m Stream,
    ws: fl::Workspace,
    per_position: usize,
    /// The sliding rings once a sequence reaches SWA_RING positions (they grow with it below that).
    rings: usize,
    /// Sampling candidates of the last batched step (variant flint.batch_topk, default on).
    cand: Option<Shared>,
}

impl<'m> Engine<'m> {
    pub(super) fn new(model: &'m fl::FlintModel, s: &'m Stream) -> Result<Self> {
        let per_position = fl::FlintState::bytes_per_position(model);
        let rings = fl::FlintState::ring_bytes(model);
        eprintln!(
            "flint worker: {per_position} bytes per position, rings up to {:.0} MiB per sequence",
            rings as f64 / (1u64 << 20) as f64
        );
        let cand = StepCandidates::shared("flint", fl::VOCAB, fl::MAX_ROWS)?;
        Ok(Self { model, s, ws: fl::Workspace::new(fl::MAX_ROWS)?, per_position, rings, cand })
    }
}

impl<'m> SlotEngine for Engine<'m> {
    type State = fl::FlintState;
    type Sampler = GpuSampler<'m>;

    fn name(&self) -> &'static str {
        "flint"
    }
    fn chunk(&self) -> usize {
        // The engine's canonical chunk: the scheduler's aligned chunks advance the canonical frontier.
        fl::CANON_CHUNK
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
            costs: (0.0021, 0.0006, 0.0),
            lookup: quench6k_cuda::variants::get("flint.lookup", 1) != 0,
        }
    }
    fn new_sampler(&self) -> Result<Self::Sampler> {
        GpuSampler::new(self.s, fl::VOCAB, self.cand.clone())
    }
    fn new_state(&mut self, capacity: usize) -> Result<fl::FlintState> {
        fl::FlintState::new(self.model, capacity, self.s)
    }
    fn growable(&self) -> bool {
        true
    }
    fn batched(&self) -> bool {
        true
    }
    fn reserve(&mut self, st: &mut fl::FlintState, positions: usize) -> Result<()> {
        st.reserve(positions)
    }
    fn trim(&mut self, st: &mut fl::FlintState, positions: usize) {
        // The device must be done with the sequence's steps.
        if let Err(e) = self.s.synchronize() {
            eprintln!("flint worker: trim: {e:#}");
            return;
        }
        st.trim(positions);
    }
    fn device_bytes(&self, st: &fl::FlintState) -> usize {
        st.device_bytes()
    }
    fn bytes_per_position(&self) -> usize {
        self.per_position
    }
    fn plan_bytes(&self, positions: usize) -> usize {
        positions * self.per_position + self.rings * positions.min(fl::SWA_RING) / fl::SWA_RING
    }
    fn free_bytes(&self) -> usize {
        quench6k_cuda::mem_info().map_or(0, |(f, _)| f)
    }
    fn history<'a>(&self, st: &'a fl::FlintState) -> &'a [i32] {
        &st.tokens
    }
    fn pos(&self, st: &fl::FlintState) -> usize {
        st.pos()
    }
    fn last_t(&self, _st: &fl::FlintState) -> usize {
        1
    }
    fn rewind(&mut self, st: &mut fl::FlintState, prompt: &[i32]) -> Result<usize> {
        st.rewind(prompt, self.s)
    }
    fn snapshot(&mut self, st: &mut fl::FlintState) -> Result<()> {
        st.snapshot(self.model, self.s)
    }
    fn reset(&mut self, st: &mut fl::FlintState) {
        if let Err(e) = self.s.synchronize() {
            eprintln!("flint worker: reset: {e:#}");
        }
        st.reset();
    }
    fn sync(&self) -> Result<()> {
        self.s.synchronize()
    }
    fn prefill(&mut self, st: &mut fl::FlintState, tokens: &[i32], _last: bool) -> Result<u64> {
        if let Some(c) = &self.cand {
            // The logits are about to be overwritten.
            c.borrow_mut().invalidate();
        }
        fl::forward(self.model, &self.ws, st, tokens, false, self.s)?;
        Ok(self.ws.logits)
    }
    fn draft(&mut self, _st: &mut fl::FlintState, _token: i32, _row: usize, _room: usize) -> Result<Draft> {
        Ok(Draft { tokens: Vec::new(), conf: Vec::new(), secs: 0.0 })
    }
    fn step(
        &mut self,
        states: &mut [&mut fl::FlintState],
        tokens: &[&[i32]],
        after: &mut After<'_, <Self::Sampler as SlotSampler>::Src>,
    ) -> Vec<Result<Option<Draft>>> {
        let started = std::time::Instant::now();
        let at: Vec<usize> = states.iter().map(|st| st.pos()).collect();
        let rows: usize = tokens.iter().map(|t| t.len()).sum();
        if let Some(c) = &self.cand {
            c.borrow_mut().invalidate();
        }
        let run = self.model.forward_batch(states, &mut self.ws, self.s, tokens).and_then(|()| match &self.cand {
            Some(c) => c.borrow_mut().launch(self.s, self.ws.logits, rows),
            None => Ok(()),
        });
        if let Err(e) = run {
            let msg = format!("{e:#}");
            return states.iter().map(|_| Err(anyhow!("{msg}"))).collect();
        }
        // Logits sequence by sequence, row by row.
        let row_bytes = (fl::VOCAB * 4) as u64;
        let mut row = 0;
        let mut out = Vec::with_capacity(states.len());
        for (i, st) in states.iter_mut().enumerate() {
            let src = self.ws.logits + row as u64 * row_bytes;
            row += tokens[i].len();
            out.push(after(i, src, Fed { at: at[i], started }).map(|v| {
                st.truncate(at[i] + v.keep);
                None
            }));
        }
        out
    }
}
