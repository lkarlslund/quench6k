//! A reusable generation session for serving: one sequence's state, workspace and captured graphs,
//! reset between requests. Greedy requests decode speculatively (MTP, adaptive draft length); sampled
//! requests decode one token per step and sample on the host from the LM head's top-k logits.

use anyhow::{Result, ensure};
use squash_cuda::{DeviceBuffer, Stream};

use crate::forward::{self, DecodeGraph, HeadWorkspace, SequenceState, Workspace};
use crate::model::{D, Model, VOCAB};
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
}

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
        Ok(Self { ws, state, hw, graph, dec, spec, chunk, capacity, top_idx: DeviceBuffer::new(MAX_TOP_K * 4)?, top_val: DeviceBuffer::new(MAX_TOP_K * 4)? })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
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
        if p.temperature <= 0.0 {
            self.dec.restart(model, &mut self.state, &mut self.ws, s, prompt, self.chunk)?;
            self.spec.restart(&self.state, &self.dec, s)?;
            // The target's first token, then speculative rounds (each commits the previous `next`
            // and the accepted drafts; the round's own `next` is emitted by the following round).
            loop {
                let committed = self.spec.round(model, &mut self.state, &mut self.ws, &mut self.dec, s)?;
                for &tok in &committed {
                    if let Some(f) = emit(tok, &mut produced) {
                        return Ok((f, produced));
                    }
                }
                if self.state.len() + 2 * (self.dec.drafts + 2) >= self.state.capacity {
                    return Ok((Finish::Length, produced));
                }
            }
        }
        // Sampled: prefill, sample the first token from the last prompt row, then one token per step.
        let mut rng = Rng(p.seed ^ 0x9e37_79b9_7f4a_7c15);
        for c in prompt.chunks(self.chunk) {
            forward::forward(model, &mut self.state, &mut self.ws, s, c)?;
        }
        let last = self.ws.out + (((prompt.len() - 1) % self.chunk) * D * 2) as u64;
        forward::head_logits(model, &self.hw, s, last)?;
        let mut tok = self.sample(s, p, &mut rng)?;
        loop {
            if let Some(f) = emit(tok, &mut produced) {
                return Ok((f, produced));
            }
            self.graph.step(model, &mut self.state, &mut self.ws, &self.hw, s, tok)?;
            tok = self.sample(s, p, &mut rng)?;
        }
    }

    /// Samples from the logits in the head workspace (row 0): top-k, temperature, top-p.
    fn sample(&self, s: &Stream, p: &GenParams, rng: &mut Rng) -> Result<i32> {
        let k = if p.top_k == 0 { MAX_TOP_K } else { p.top_k.min(MAX_TOP_K) };
        squash_cuda::kernels::topk(s, self.hw.logits_addr(), 1, VOCAB, VOCAB, k, false, self.top_idx.addr(), self.top_val.addr())?;
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
