//! Nemotron-H checkpoints in the server's slot scheduler ([`super::slots`]): each slot's sequence its own
//! growable [`State`] (attention caches mapped as it grows, its Mamba block and the block's snapshots at
//! canonical boundaries), one shared [`Workspace`]; prompts are prefilled in the scheduler's chunks, which
//! end at multiples of [`CANON_CHUNK`], and a cached prefix is reused only up to such a boundary
//! ([`State::rewind`]), so a sequence's bits depend only on its tokens. The decoding sequences step
//! together through [`NemotronModel::verify`] (each sequence's bits as alone) with the drafts of the
//! checkpoint's MTP layer ([`NemotronModel::commit`]: the Mamba states keep the accepted rows; the layer
//! drafts greedily and states each draft's probability, so the scheduler's policy verifies as many as
//! pay off). Without the MTP layer, plain batched steps.

use anyhow::{Result, anyhow};
use quench6k_cuda::Stream;
use quench6k_engine::nemotron::NemotronModel;
use quench6k_engine::nemotron::forward::{CANON_CHUNK, Kv, Logits, MAX_DRAFTS, MAX_SEQS, MAX_VERIFY, State, Workspace};

use super::slots::{After, Draft, Drafter, Drafting, Fed, GpuSampler, Shared, SlotEngine, SlotSampler, StepCandidates};

/// Most tokens of one prefill call (whole canonical chunks while no other request decodes): the
/// workspace's step size. Larger steps pass over the routed experts' weights less often (8K prompts
/// +10%, 32K +15% against single chunks).
const PREFILL_MAX: usize = 4 * CANON_CHUNK;

pub(super) struct Engine<'m> {
    model: &'m NemotronModel,
    s: &'m Stream,
    ws: Workspace,
    kv: Kv,
    /// MTP drafts per pass (0: no MTP layer; QUENCH6K_NEM_DRAFTS, default 4).
    drafts: usize,
    /// Sampling candidates of the last batched step (variant nemotron.batch_topk, default on).
    cand: Option<Shared>,
}

impl<'m> Engine<'m> {
    pub(super) fn new(model: &'m NemotronModel, s: &'m Stream) -> Result<Self> {
        let kv = if quench6k_engine::forward::KV_FP8.load(std::sync::atomic::Ordering::Relaxed) { Kv::Fp8 } else { Kv::Bf16 };
        let ws = Workspace::new(model, PREFILL_MAX, MAX_SEQS * MAX_VERIFY)?;
        let drafts = if model.mtp.is_some() {
            std::env::var("QUENCH6K_NEM_DRAFTS").ok().and_then(|v| v.parse().ok()).unwrap_or(4usize).min(MAX_DRAFTS)
        } else {
            0
        };
        eprintln!(
            "nemotron worker: workspace {:.2} GiB; {} bytes per position ({kv:?} cache), {:.0} MiB of Mamba state per sequence (and per snapshot)",
            ws.device_bytes as f64 / (1u64 << 30) as f64,
            State::position_bytes(&model.cfg, kv),
            State::state_bytes(&model.cfg) as f64 / (1u64 << 20) as f64
        );
        let cand = StepCandidates::shared("nemotron", model.cfg.vocab, MAX_SEQS * MAX_VERIFY)?;
        Ok(Self { model, s, ws, kv, drafts, cand })
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
    fn max_prefill(&self) -> usize {
        // QUENCH6K_NEM_PREFILL_MAX: fewer tokens per call (whole chunks; for comparisons).
        let cap = std::env::var("QUENCH6K_NEM_PREFILL_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(PREFILL_MAX);
        cap.clamp(CANON_CHUNK, PREFILL_MAX)
    }
    fn max_step(&self) -> usize {
        1 + self.drafts
    }
    fn eos(&self) -> &[i32] {
        &self.model.cfg.eos
    }
    fn drafting(&self) -> Drafting {
        Drafting {
            kind: if self.drafts > 0 { Drafter::Conf } else { Drafter::None },
            drafts: self.drafts,
            costs: (0.0095, 0.0025, 0.0012),
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
        // The scheduler's chunks end at multiples of CANON_CHUNK: one canonical chunk per call, or (no
        // other request decoding) up to PREFILL_MAX tokens of whole chunks: the same bits per row.
        let want = if last { Logits::Last } else { Logits::None };
        if std::env::var_os("QUENCH6K_NEM_LOG_PREFILL").is_some() {
            eprintln!("nemotron prefill: {} tokens at {}", tokens.len(), st.pos);
        }
        self.model.forward(st, &mut self.ws, self.s, tokens, want)?;
        Ok(self.ws.out_logits)
    }
    fn draft(&mut self, st: &mut State, token: i32, _row: usize, room: usize) -> Result<Draft> {
        // After a prefill (a plain step): the MTP layer at the prompt's last position, then the drafts.
        let t = std::time::Instant::now();
        let want = room.min(self.drafts);
        let d = self.model.commit(&mut [st], &mut self.ws, self.s, &[0], &[Some(token)], &[want])?.pop().unwrap_or_default();
        Ok(Draft { tokens: d.tokens, conf: d.conf, secs: t.elapsed().as_secs_f64() })
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
        let verify = self.drafts > 0;
        let fwd = if verify {
            self.model.verify(states, &mut self.ws, self.s, tokens)
        } else {
            self.model.forward_batch(states, &mut self.ws, self.s, tokens)
        };
        let run = fwd.and_then(|()| match &self.cand {
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
        let (mut keep, mut next, mut want) = (Vec::new(), Vec::new(), Vec::new());
        for (i, st) in states.iter_mut().enumerate() {
            let src = self.ws.out_logits + row as u64 * row_bytes;
            row += tokens[i].len();
            match after(i, src, Fed { at: at[i], started }) {
                Ok(v) => {
                    keep.push(v.keep);
                    match v.draft {
                        Some((tok, _, room)) if verify => {
                            next.push(Some(tok));
                            want.push(room.min(self.drafts));
                        }
                        _ => {
                            next.push(None);
                            want.push(0);
                        }
                    }
                    if !verify && let Err(e) = st.truncate(at[i] + v.keep, self.s) {
                        out.push(Err(e));
                        continue;
                    }
                    out.push(Ok(None));
                }
                Err(e) => {
                    // The request ends; its sequence keeps the fed token.
                    keep.push(1);
                    next.push(None);
                    want.push(0);
                    out.push(Err(e));
                }
            }
        }
        if !verify {
            return out;
        }
        // The verdicts into the Mamba states, then one MTP pass for every sequence that asked for drafts;
        // its time is shared out evenly.
        let t = std::time::Instant::now();
        match self.model.commit(states, &mut self.ws, self.s, &keep, &next, &want) {
            Ok(drafts) => {
                let asked = want.iter().filter(|&&w| w > 0).count().max(1);
                let secs = t.elapsed().as_secs_f64() / asked as f64;
                for (i, d) in drafts.into_iter().enumerate() {
                    if want[i] > 0 && out[i].is_ok() {
                        out[i] = Ok(Some(Draft { tokens: d.tokens, conf: d.conf, secs }));
                    }
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                for o in out.iter_mut() {
                    *o = Err(anyhow!("commit: {msg}"));
                }
            }
        }
        out
    }
}
