//! A reusable generation session for serving: one sequence's state, workspace and captured graphs,
//! reset between requests. Greedy requests decode speculatively (MTP, adaptive draft length); sampled
//! requests decode one token per step and sample on the host from the LM head's top-k logits.

use anyhow::{Result, ensure};
use squash_cuda::{DeviceBuffer, Stream};

use crate::forward::{DecodeGraph, HeadWorkspace, SequenceState, Workspace};
use crate::model::{Model, VOCAB};
use crate::spec::{SpecDecoder, SpecGraphs};

/// Sampling and stopping for one request. `temperature == 0` is greedy.
#[derive(Clone, Debug)]
pub struct GenParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    /// Generation ends after any of these tokens (which is emitted).
    pub stop_tokens: Vec<i32>,
    pub steering_strength: f32,
    /// Prompt positions worth a recurrent-state snapshot for prefix reuse (e.g. the start of the final
    /// assistant turn, which a follow-up turn shares).
    pub boundaries: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// A stop token was generated.
    Stop,
    /// `max_tokens` reached, or the context is full.
    Length,
    /// The token callback asked to stop.
    Cancelled,
}

pub struct Session {
    ws: Workspace,
    state: SequenceState,
    hw: HeadWorkspace,
    graph: DecodeGraph,
    dec: SpecDecoder,
    spec: SpecGraphs,
    chunk: usize,
    capacity: usize,
    top_idx: DeviceBuffer,
    top_val: DeviceBuffer,
    /// Tokens whose attention K/V are cached at their positions (the committed sequence).
    kv_tokens: Vec<i32>,
    /// Recurrent-state snapshots at prompt positions, oldest first.
    snapshots: Vec<Snapshot>,
    /// Prompt tokens served from a snapshot in the last request.
    pub reused: usize,
}

/// The recurrent state at a position: every GDN layer's conv history and delta-rule state, the PLE conv
/// history and the speculative decoder's pending row (attention caches are position-indexed and stay).
struct Snapshot {
    pos: usize,
    buf: DeviceBuffer,
}

const MAX_SNAPSHOTS: usize = 4;

/// The largest top-k the session samples from.
pub const MAX_TOP_K: usize = 64;

impl Session {
    /// `capacity`: longest prompt + completion; `chunk`: prefill tokens per forward pass; `max_drafts`:
    /// longest speculative draft (adaptive 1..=max_drafts).
    pub fn new(model: &Model, s: &Stream, capacity: usize, chunk: usize, max_drafts: usize) -> Result<Self> {
        let slack = 2 * (max_drafts + 2) + 16;
        let mut ws = Workspace::new(chunk, capacity + slack)?;
        let mut state = SequenceState::with_mtp(capacity + slack, true)?;
        let hw = HeadWorkspace::new(1, MAX_TOP_K)?;
        // A one-token prompt to bring the sequence into a valid state for capture.
        let dec = SpecDecoder::prefill(model, &mut state, &mut ws, s, &[0], max_drafts, chunk)?;
        let graph = DecodeGraph::capture(model, &state, &ws, &hw, s)?;
        let spec = SpecGraphs::capture(model, &state, &mut ws, &dec, s, true)?;
        Ok(Self {
            ws,
            state,
            hw,
            graph,
            dec,
            spec,
            chunk,
            capacity,
            top_idx: DeviceBuffer::new(MAX_TOP_K * 4)?,
            top_val: DeviceBuffer::new(MAX_TOP_K * 4)?,
            kv_tokens: Vec::new(),
            snapshots: Vec::new(),
            reused: 0,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The pieces of the recurrent state, as (device address, bytes); `pending` is the speculative
    /// decoder's pending row.
    fn pieces(state: &SequenceState, pending: u64) -> Vec<(u64, usize)> {
        let mut v: Vec<(u64, usize)> = state.gdn.iter().flatten().flat_map(|g| [(g.conv.addr(), g.conv.bytes()), (g.recurrent.addr(), g.recurrent.bytes())]).collect();
        v.push((state.ple_conv.addr(), state.ple_conv.bytes()));
        v.push((pending, crate::model::H * 4));
        v
    }

    fn save_snapshot(state: &SequenceState, pending: u64, s: &Stream, pos: usize, recycled: Option<DeviceBuffer>) -> Result<Snapshot> {
        let pieces = Self::pieces(state, pending);
        let total: usize = pieces.iter().map(|p| p.1).sum();
        let buf = match recycled {
            Some(b) if b.bytes() >= total => b,
            _ => DeviceBuffer::new(total)?,
        };
        let mut off = 0u64;
        for (addr, bytes) in pieces {
            s.copy_d2d(buf.addr() + off, addr, bytes)?;
            off += bytes as u64;
        }
        Ok(Snapshot { pos, buf })
    }

    fn restore_snapshot(&self, s: &Stream, snap: &Snapshot) -> Result<()> {
        let mut off = 0u64;
        for (addr, bytes) in Self::pieces(&self.state, self.dec.pending_addr()) {
            s.copy_d2d(addr, snap.buf.addr() + off, bytes)?;
            off += bytes as u64;
        }
        Ok(())
    }

    /// Positions the state, target and MTP caches on the longest snapshot whose prefix the caches still
    /// hold, prefills the rest of `prompt`, and takes new snapshots at `boundaries`.
    fn prefill(&mut self, model: &Model, s: &Stream, prompt: &[i32], boundaries: &[usize]) -> Result<()> {
        let lcp = self.kv_tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        // Snapshots past the common prefix describe tokens the new prompt overwrites.
        self.snapshots.retain(|sn| sn.pos <= lcp);
        self.state.reset(s)?;
        let best = self.snapshots.iter().enumerate().filter(|(_, sn)| sn.pos < prompt.len()).max_by_key(|(_, sn)| sn.pos).map(|(i, _)| i);
        let start = match best {
            Some(i) => {
                let snap = self.snapshots.remove(i);
                self.restore_snapshot(s, &snap)?;
                let pos = snap.pos;
                self.snapshots.push(snap); // most recently used last
                self.state.tokens = prompt[..pos].to_vec();
                pos
            }
            None => {
                s.memset(self.dec.pending_addr(), 0, crate::model::H * 4)?;
                0
            }
        };
        self.reused = start;
        let mut pauses: Vec<usize> = boundaries.iter().copied().filter(|&b| b > start && b < prompt.len() && !self.snapshots.iter().any(|sn| sn.pos == b)).collect();
        pauses.sort_unstable();
        pauses.dedup();
        let snapshots = &mut self.snapshots;
        self.dec.prefill_from(model, &mut self.state, &mut self.ws, s, prompt, start, self.chunk, &pauses, |state, pos, pending| {
            let recycled = if snapshots.len() >= MAX_SNAPSHOTS { Some(snapshots.remove(0).buf) } else { None };
            snapshots.push(Self::save_snapshot(state, pending, s, pos, recycled)?);
            Ok(())
        })?;
        self.kv_tokens = prompt.to_vec();
        Ok(())
    }

    /// Generates from `prompt`, calling `on_token` for every generated token (return false to stop).
    pub fn generate(&mut self, model: &Model, s: &Stream, prompt: &[i32], p: &GenParams, mut on_token: impl FnMut(i32) -> bool) -> Result<(Finish, usize)> {
        ensure!(!prompt.is_empty(), "empty prompt");
        ensure!(prompt.len() < self.capacity, "prompt of {} tokens exceeds the context of {}", prompt.len(), self.capacity);
        self.state.reset(s)?;
        self.state.set_steering_strength(s, p.steering_strength)?;
        let budget = p.max_tokens.min(self.capacity - prompt.len());
        let mut produced = 0usize;
        let mut emit = |tok: i32, produced: &mut usize| -> Option<Finish> {
            *produced += 1;
            if !on_token(tok) {
                return Some(Finish::Cancelled);
            }
            if p.stop_tokens.contains(&tok) {
                return Some(Finish::Stop);
            }
            if *produced >= budget {
                return Some(Finish::Length);
            }
            None
        };
        if budget == 0 {
            return Ok((Finish::Length, 0));
        }
        self.prefill(model, s, prompt, &p.boundaries)?;
        if p.temperature <= 0.0 {
            self.spec.restart(&self.state, &self.dec, s)?;
            // The target's first token, then speculative rounds (each commits the previous `next`
            // and the accepted drafts; the round's own `next` is emitted by the following round).
            loop {
                let committed = self.spec.round(model, &mut self.state, &mut self.ws, &mut self.dec, s)?;
                for &tok in &committed {
                    if let Some(f) = emit(tok, &mut produced) {
                        self.kv_tokens = self.state.tokens.clone();
                        return Ok((f, produced));
                    }
                }
                if self.state.len() + 2 * (self.dec.drafts + 2) >= self.state.capacity {
                    self.kv_tokens = self.state.tokens.clone();
                    return Ok((Finish::Length, produced));
                }
            }
        }
        // Sampled: the first token from the last prompt row's logits, then one token per step.
        let mut rng = Rng(p.seed ^ 0x9e37_79b9_7f4a_7c15);
        let mut tok = self.sample(s, self.dec.last_logits_addr(), p, &mut rng)?;
        loop {
            if let Some(f) = emit(tok, &mut produced) {
                self.kv_tokens = self.state.tokens.clone();
                return Ok((f, produced));
            }
            self.graph.step(model, &mut self.state, &mut self.ws, &self.hw, s, tok)?;
            tok = self.sample(s, self.hw.logits_addr(), p, &mut rng)?;
        }
    }

    /// Samples from a logits row (VOCAB f32 at `logits`): top-k, temperature, top-p.
    fn sample(&self, s: &Stream, logits: u64, p: &GenParams, rng: &mut Rng) -> Result<i32> {
        let k = if p.top_k == 0 { MAX_TOP_K } else { p.top_k.min(MAX_TOP_K) };
        squash_cuda::kernels::topk(s, logits, 1, VOCAB, VOCAB, k, false, self.top_idx.addr(), self.top_val.addr())?;
        let ids: Vec<i32> = s.download(self.top_idx.addr(), k)?;
        let vals: Vec<f32> = s.download(self.top_val.addr(), k)?;
        let t = p.temperature.max(1e-4);
        let m = vals[0];
        let mut probs: Vec<f64> = vals.iter().map(|v| (((v - m) / t) as f64).exp()).collect();
        let total: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|x| *x /= total);
        let mut keep = k;
        if p.top_p < 1.0 {
            let mut acc = 0.0;
            for (i, x) in probs.iter().enumerate() {
                acc += x;
                if acc >= p.top_p as f64 {
                    keep = i + 1;
                    break;
                }
            }
        }
        let mass: f64 = probs[..keep].iter().sum();
        let mut r = rng.next() * mass;
        for i in 0..keep {
            r -= probs[i];
            if r <= 0.0 {
                return Ok(ids[i]);
            }
        }
        Ok(ids[keep - 1])
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}
