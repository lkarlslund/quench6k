//! Flint checkpoints in the server's slot scheduler ([`super::slots`]): each slot's sequence its own
//! growable [`fl::FlintState`] (full-attention caches mapped as it grows), one shared [`fl::Workspace`];
//! the decoding sequences step together through [`fl::FlintModel::forward_batch`] (each sequence's bits
//! as alone); prompt-lookup drafts (Flint has no drafter). A batched step's sampling candidates come
//! from the device together ([`Candidates`]) rather than in a top-k launch and download per sequence.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::{Result, anyhow};
use quench6k_cuda::{DeviceBuffer, Event, PinnedBuffer, Stream};
use quench6k_engine::flint as fl;
use quench6k_engine::session::GenParams;

use super::slots::{After, Draft, Drafter, Drafting, Fed, SlotEngine, SlotSampler};
use crate::flint::{POOL, SAMPLE_ROWS, Sampler};

/// A batched step's top candidates of every logits row, fetched right after the step is enqueued: one
/// top-k launch per SAMPLE_ROWS rows (each row's candidates are those of a launch for its sequence's rows
/// alone: top-k works row by row) and one download. A sequence whose parameters want another number of
/// candidates fetches its own ([`Sampler::fetch`]); the next step then uses the pool most sequences
/// asked for.
struct Candidates {
    /// Candidate ids [rows, pool] (i32), then values [rows, pool] (f32), per launch of SAMPLE_ROWS rows.
    dev: DeviceBuffer,
    host: PinnedBuffer,
    done: Event,
    /// The step's logits (address of row 0), rows and pool; None when the logits are not the step's.
    step: Option<(u64, usize, usize)>,
    /// Whether `done` has been waited on, and the candidates unpacked into `ids`/`vals` [rows, pool].
    ready: bool,
    ids: Vec<i32>,
    vals: Vec<f32>,
    /// Pools the sequences of the current step fetched with.
    asked: Vec<usize>,
}

impl Candidates {
    fn new() -> Result<Self> {
        let bytes = fl::MAX_ROWS * POOL * 8;
        Ok(Self {
            dev: DeviceBuffer::new(bytes)?,
            host: PinnedBuffer::new(bytes)?,
            done: Event::new()?,
            step: None,
            ready: false,
            ids: Vec::new(),
            vals: Vec::new(),
            asked: Vec::new(),
        })
    }

    /// The pool most of the last step's fetches asked for (1 when none did).
    fn pool(&self) -> usize {
        let mut best = (0, 1);
        for &p in &self.asked {
            let n = self.asked.iter().filter(|&&q| q == p).count();
            if n > best.0 || (n == best.0 && p < best.1) {
                best = (n, p);
            }
        }
        best.1
    }

    /// Enqueues the candidates of `rows` logits rows at `logits` (after the step that writes them).
    fn launch(&mut self, s: &Stream, logits: u64, rows: usize) -> Result<()> {
        let pool = self.pool();
        self.asked.clear();
        self.step = None;
        self.ready = false;
        let row_bytes = (fl::VOCAB * 4) as u64;
        let mut r = 0;
        while r < rows {
            let n = (rows - r).min(SAMPLE_ROWS);
            let at = self.dev.addr() + (r * pool * 8) as u64;
            quench6k_cuda::kernels::topk(s, logits + r as u64 * row_bytes, n, fl::VOCAB, fl::VOCAB, pool, false, at, at + (n * pool * 4) as u64)?;
            r += n;
        }
        unsafe { s.copy_d2h(self.host.as_ptr(), self.dev.addr(), rows * pool * 8)? };
        self.done.record(s)?;
        self.step = Some((logits, rows, pool));
        Ok(())
    }

    /// The candidates of `rows` rows at `src` when this step's batch has them with `pool` per row.
    fn take(&mut self, src: u64, rows: usize, pool: usize) -> Result<Option<(&[i32], &[f32])>> {
        self.asked.push(pool);
        let Some((base, total, have)) = self.step else { return Ok(None) };
        let row_bytes = (fl::VOCAB * 4) as u64;
        if have != pool || src < base || (src - base) % row_bytes != 0 || ((src - base) / row_bytes) as usize + rows > total {
            return Ok(None);
        }
        if !self.ready {
            self.done.synchronize()?;
            let raw = self.host.as_slice();
            let word = |i: usize| u32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
            self.ids.clear();
            self.vals.clear();
            let mut r = 0;
            while r < total {
                let n = (total - r).min(SAMPLE_ROWS);
                let at = r * pool * 2;
                self.ids.extend((0..n * pool).map(|i| word(at + i) as i32));
                self.vals.extend((0..n * pool).map(|i| f32::from_bits(word(at + n * pool + i))));
                r += n;
            }
            self.ready = true;
        }
        let first = ((src - base) / row_bytes) as usize;
        Ok(Some((&self.ids[first * pool..(first + rows) * pool], &self.vals[first * pool..(first + rows) * pool])))
    }
}

/// A slot's sampler: the batched step's candidates when they fit, else its own fetch.
pub(super) struct FlintSampler<'s> {
    s: &'s Stream,
    shared: Rc<RefCell<Candidates>>,
    inner: Sampler,
}

impl SlotSampler for FlintSampler<'_> {
    /// F32 logits [rows, VOCAB] on the device.
    type Src = u64;
    fn fetch(&mut self, src: u64, rows: usize, p: &GenParams) -> Result<()> {
        let pool = Sampler::pool(p);
        let mut c = self.shared.borrow_mut();
        if let Some((ids, vals)) = c.take(src, rows, pool)? {
            self.inner.load(ids, vals, pool);
            return Ok(());
        }
        drop(c);
        self.inner.fetch(self.s, src, rows, fl::VOCAB, p)
    }
    fn pick(&mut self, row: usize, p: &GenParams, generated: &[i32]) -> i32 {
        self.inner.pick(row, p, generated)
    }
}

pub(super) struct Engine<'m> {
    model: &'m fl::FlintModel,
    s: &'m Stream,
    ws: fl::Workspace,
    per_position: usize,
    /// The sliding rings once a sequence reaches SWA_RING positions (they grow with it below that).
    rings: usize,
    /// Sampling candidates of the last batched step (variant flint.batch_topk, default on).
    cand: Option<Rc<RefCell<Candidates>>>,
}

impl<'m> Engine<'m> {
    pub(super) fn new(model: &'m fl::FlintModel, s: &'m Stream) -> Result<Self> {
        let per_position = fl::FlintState::bytes_per_position(model);
        let rings = fl::FlintState::ring_bytes(model);
        eprintln!(
            "flint worker: {per_position} bytes per position, rings up to {:.0} MiB per sequence",
            rings as f64 / (1u64 << 20) as f64
        );
        let cand = if quench6k_cuda::variants::get("flint.batch_topk", 1) != 0 {
            Some(Rc::new(RefCell::new(Candidates::new()?)))
        } else {
            None
        };
        Ok(Self { model, s, ws: fl::Workspace::new(fl::MAX_ROWS)?, per_position, rings, cand })
    }
}

impl<'m> SlotEngine for Engine<'m> {
    type State = fl::FlintState;
    type Sampler = FlintSampler<'m>;

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
        // Without batched candidates every sampler gets a store of its own that is never filled.
        let shared = match &self.cand {
            Some(c) => c.clone(),
            None => Rc::new(RefCell::new(Candidates::new()?)),
        };
        Ok(FlintSampler { s: self.s, shared, inner: Sampler::new()? })
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
            c.borrow_mut().step = None;
        }
        fl::forward(self.model, &self.ws, st, tokens, false, self.s)?;
        Ok(self.ws.logits)
    }
    fn draft(&mut self, _st: &mut fl::FlintState, _token: i32, _row: usize) -> Result<Draft> {
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
            c.borrow_mut().step = None;
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
