//! Inkling 2 checkpoints in the server's slot scheduler ([`super::slots`]): each slot's sequence its own
//! growable [`State`] (full-length caches mapped as it grows), one shared [`Workspace`]; prompts are
//! prefilled canonically ([`InklingModel::prefill`], chunks aligned to [`CANON_CHUNK`]), the decoding
//! sequences step together through [`InklingModel::forward_batch`] and draft together through
//! [`InklingModel::draft_batch_some`] (each sequence's bits as alone; the drafter's head computes only
//! the drafts the requests can use). Drafts come from the checkpoint's DSpark drafter (as many as pay
//! off, from the drafter's stated probabilities: see [`super::slots`]), or from prompt lookup without
//! one. A batched step's sampling candidates come from the device together
//! ([`StepCandidates`]).

use anyhow::{Result, anyhow};
use quench6k_cuda::Stream;
use quench6k_engine::inkling::InklingModel;
use quench6k_engine::inkling::forward::{BATCH_ROWS, CANON_CHUNK, State, Workspace};

use super::slots::{After, Draft, Drafter, Drafting, Fed, GpuSampler, Shared, SlotEngine, SlotSampler, StepCandidates};

/// Most tokens of a verification step (the pending token and its drafts); the sampler takes 8 rows.
const MAX_STEP: usize = 8;

pub(super) struct Engine<'m> {
    model: &'m InklingModel,
    s: &'m Stream,
    ws: Workspace,
    /// Device bytes of a new state (rings and small buffers), measured on the first one (0 before).
    fixed: usize,
    per_position: usize,
    /// Sampling candidates of the last batched step (variant inkling.batch_topk, default on).
    cand: Option<Shared>,
}

impl<'m> Engine<'m> {
    pub(super) fn new(model: &'m InklingModel, s: &'m Stream) -> Result<Self> {
        let ws = Workspace::new(model, CANON_CHUNK, BATCH_ROWS)?;
        let per_position = State::bytes_per_position(model);
        eprintln!(
            "inkling worker: workspace {:.2} GiB; {per_position} bytes per position",
            ws.device_bytes as f64 / (1u64 << 30) as f64
        );
        let cand = StepCandidates::shared("inkling", model.cfg.vocab, BATCH_ROWS)?;
        Ok(Self { model, s, ws, fixed: 0, per_position, cand })
    }
}

impl<'m> SlotEngine for Engine<'m> {
    type State = State;
    type Sampler = GpuSampler<'m>;

    fn name(&self) -> &'static str {
        "inkling"
    }
    fn chunk(&self) -> usize {
        CANON_CHUNK
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
        GpuSampler::new(self.s, self.model.cfg.vocab, self.cand.clone())
    }
    fn new_state(&mut self, capacity: usize) -> Result<State> {
        let st = State::new(self.model, capacity, self.s)?;
        if self.fixed == 0 {
            self.fixed = st.device_bytes();
            eprintln!("inkling worker: per sequence {:.2} GiB fixed", self.fixed as f64 / (1u64 << 30) as f64);
        }
        Ok(st)
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
    fn prefill(&mut self, st: &mut State, tokens: &[i32], _last: bool) -> Result<u64> {
        if let Some(c) = &self.cand {
            // The logits are about to be overwritten.
            c.borrow_mut().invalidate();
        }
        // The scheduler's chunks end at multiples of CANON_CHUNK: one canonical chunk per call.
        self.model.prefill(st, &mut self.ws, self.s, tokens)?;
        Ok(self.ws.out_logits)
    }
    fn draft(&mut self, st: &mut State, token: i32, row: usize, room: usize) -> Result<Draft> {
        let t = std::time::Instant::now();
        let tokens = self.model.draft_batch_some(&mut [st], &mut self.ws, self.s, &[token], &[row], &[room])?.pop().unwrap_or_default();
        let conf = if tokens.is_empty() { Vec::new() } else { self.model.draft_conf(&self.ws) };
        Ok(Draft { tokens, conf, secs: t.elapsed().as_secs_f64() })
    }
    fn step(
        &mut self,
        states: &mut [&mut State],
        tokens: &[&[i32]],
        after: &mut After<'_, <Self::Sampler as SlotSampler>::Src>,
    ) -> Vec<Result<Option<Draft>>> {
        let started = std::time::Instant::now();
        let at: Vec<usize> = states.iter().map(|st| st.pos).collect();
        if let Some(c) = &self.cand {
            c.borrow_mut().invalidate();
        }
        let rows: usize = tokens.iter().map(|t| t.len()).sum();
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
        let mut out: Vec<Result<Option<Draft>>> = Vec::with_capacity(states.len());
        let mut wanted: Vec<(usize, i32, usize, usize)> = Vec::new();
        for (i, st) in states.iter_mut().enumerate() {
            let src = self.ws.out_logits + row as u64 * row_bytes;
            row += tokens[i].len();
            match after(i, src, Fed { at: at[i], started }) {
                Ok(v) => {
                    st.truncate(at[i] + v.keep);
                    if let Some((tok, r, room)) = v.draft {
                        wanted.push((i, tok, r, room));
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
        // The drafter computes only the drafts each request can use.
        let most: Vec<usize> = wanted.iter().map(|w| w.3).collect();
        let mut sub: Vec<&mut State> = states
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| wanted.iter().any(|w| w.0 == *i))
            .map(|(_, st)| &mut **st)
            .collect();
        match self.model.draft_batch_some(&mut sub, &mut self.ws, self.s, &pending, &rows, &most) {
            Ok(drafts) => {
                let secs = t.elapsed().as_secs_f64() / wanted.len() as f64;
                let confs = self.model.draft_conf_batch(&self.ws);
                for (j, (w, tokens)) in wanted.iter().zip(drafts).enumerate() {
                    let conf = if tokens.is_empty() { Vec::new() } else { confs.get(j).cloned().unwrap_or_default() };
                    out[w.0] = Ok(Some(Draft { tokens, conf, secs }));
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
