//! Greedy speculative decoding with the MTP drafter (SPEC §10).
//!
//! Each round drafts `drafts` tokens with the MTP layer, verifies them with one target step over
//! `[x, d1 .. dn]`, and keeps the longest prefix the target agrees with plus the target's own next
//! token. Recurrent state is committed one step behind (see `forward::Spec`), so rejected drafts never
//! reach it; attention caches are position-indexed and simply get overwritten.

use anyhow::{Context, Result, ensure};
use squash_cuda::{DeviceBuffer, Stream};

use crate::forward::{self, HeadWorkspace, Part, SequenceState, Workspace};
use crate::model::{D, H, Model};

pub struct SpecDecoder {
    pub drafts: usize,
    /// Target pre-mixer residual of the last committed position (the MTP input for the next token).
    pending: DeviceBuffer,
    /// Target residual rows of the last verification's accepted tokens.
    verified: DeviceBuffer,
    /// Accepted tokens of the last verification, which the MTP cache has seen only as drafts.
    catch_up: Vec<i32>,
    head: HeadWorkspace,
    /// The next token: produced by the target, not yet run through it.
    pub next: i32,
    pub rounds: usize,
    pub accepted: usize,
    /// Seconds spent drafting (MTP) and verifying, over all rounds.
    pub draft_secs: f64,
    pub verify_secs: f64,
}

const ROW: u64 = (H * 4) as u64;

impl SpecDecoder {
    /// Prefills `prompt` into `state` (target and MTP caches) and returns the decoder positioned on the
    /// target's greedy next token.
    pub fn prefill(model: &Model, state: &mut SequenceState, ws: &mut Workspace, s: &Stream, prompt: &[i32], drafts: usize, chunk: usize) -> Result<Self> {
        ensure!(state.spec.is_some() && model.mtp.is_some(), "speculative decoding needs the MTP drafter and a sequence with MTP state");
        ensure!((1..=8).contains(&drafts), "draft length {drafts} out of range");
        let mut dec = Self {
            drafts,
            pending: DeviceBuffer::zeroed(H * 4)?,
            verified: DeviceBuffer::new((drafts + 1) * H * 4)?,
            catch_up: Vec::new(),
            head: HeadWorkspace::new(drafts + 1, 1)?,
            next: 0,
            rounds: 0,
            accepted: 0,
            draft_secs: 0.0,
            verify_secs: 0.0,
        };
        dec.restart(model, state, ws, s, prompt, chunk)?;
        Ok(dec)
    }

    /// Prefills a new `prompt` into the (reset) `state`, reusing this decoder's buffers so graphs captured
    /// for it stay valid.
    pub fn restart(&mut self, model: &Model, state: &mut SequenceState, ws: &mut Workspace, s: &Stream, prompt: &[i32], chunk: usize) -> Result<()> {
        s.memset(self.pending.addr(), 0, H * 4)?;
        self.prefill_from(model, state, ws, s, prompt, 0, chunk, &[], |_, _, _| Ok(()), None)
    }

    /// Prefills `prompt[start..]` into `state`, which already holds `prompt[..start]` (target and MTP
    /// caches, recurrent state, and the pending row: see `pending_addr`). At each position in
    /// `pauses` (ascending, in start..len) the chunking breaks and `at_pause(state, pos, pending)` runs.
    /// `progress(pos)` is called when the prompt up to `pos` has been computed (after each chunk).
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_from(&mut self, model: &Model, state: &mut SequenceState, ws: &mut Workspace, s: &Stream, prompt: &[i32], start: usize, chunk: usize, pauses: &[usize], mut at_pause: impl FnMut(&SequenceState, usize, u64) -> Result<()>, progress: Option<&(dyn Fn(usize) + Send + Sync)>) -> Result<()> {
        ensure!(start < prompt.len(), "nothing to prefill");
        ensure!(state.len() == start, "state holds {} tokens, expected {start}", state.len());
        let mut next = 0;
        let mut c0 = start;
        let mut pause = pauses.iter().copied().filter(|&p| p > start && p < prompt.len()).peekable();
        while c0 < prompt.len() {
            let mut end = (c0 + chunk).min(prompt.len());
            if let Some(&p) = pause.peek() {
                end = end.min(p);
            }
            let c = &prompt[c0..end];
            let t = c.len();
            forward::forward(model, state, ws, s, c)?;
            if end == prompt.len() {
                next = argmax_rows(model, &self.head, s, ws.out + ((t - 1) * D * 2) as u64, 1)?[0];
            }
            s.copy_d2d(ws.mtp_hin, self.pending.addr(), H * 4)?;
            s.copy_d2d(ws.mtp_hin + ROW, ws.h, (t - 1) * H * 4)?;
            s.copy_d2d(self.pending.addr(), ws.h + (t - 1) as u64 * ROW, H * 4)?;
            forward::prepare_mtp(ws, c0, c)?;
            forward::forward_mtp(model, state, ws, s, t, ws.mtp_hin, None)?;
            if pause.peek() == Some(&end) {
                pause.next();
                at_pause(state, end, self.pending.addr())?;
            }
            if let Some(f) = progress {
                s.synchronize()?;
                f(end);
            }
            c0 = end;
        }
        s.synchronize()?;
        self.next = next;
        self.catch_up.clear();
        self.rounds = 0;
        self.accepted = 0;
        self.draft_secs = 0.0;
        self.verify_secs = 0.0;
        Ok(())
    }

    /// Device address of the pending row (the target residual of the last committed position, H f32).
    pub fn pending_addr(&self) -> u64 {
        self.pending.addr()
    }

    /// Device address of the LM-head logits of the last prefilled prompt row (VOCAB f32).
    pub fn last_logits_addr(&self) -> u64 {
        self.head.logits_addr()
    }

    /// One draft-and-verify round; returns the tokens committed to the sequence (the previous `next`
    /// and the accepted drafts). `self.next` becomes the target's following token.
    pub fn round(&mut self, model: &Model, state: &mut SequenceState, ws: &mut Workspace, s: &Stream) -> Result<Vec<i32>> {
        let p = state.len();
        let n = self.drafts;
        // Draft 1: the MTP cache catches up on the last accepted tokens, paired with the target rows
        // before them, and the pending token is paired with the newest target row.
        let mut mtp_tokens = self.catch_up.clone();
        mtp_tokens.push(self.next);
        let t = mtp_tokens.len();
        s.copy_d2d(ws.mtp_hin, self.pending.addr(), H * 4)?;
        if t > 1 {
            s.copy_d2d(ws.mtp_hin + ROW, self.verified.addr(), (t - 1) * H * 4)?;
            // The newest committed target row becomes the pending row.
            s.copy_d2d(self.pending.addr(), self.verified.addr() + ((t - 2) as u64) * ROW, H * 4)?;
        }
        forward::prepare_mtp(ws, p - (t - 1), &mtp_tokens)?;
        forward::forward_mtp(model, state, ws, s, t, ws.mtp_hin, None)?;
        let mut d = Vec::with_capacity(n);
        d.push(argmax_rows(model, &self.head, s, ws.out + ((t - 1) * D * 2) as u64, 1)?[0]);
        s.copy_d2d(ws.mtp_hin, ws.h + ((t - 1) as u64) * ROW, H * 4)?;
        for j in 1..n {
            forward::prepare_mtp(ws, p + j, &[d[j - 1]])?;
            forward::forward_mtp(model, state, ws, s, 1, ws.mtp_hin, None)?;
            d.push(argmax_rows(model, &self.head, s, ws.out, 1)?[0]);
            s.copy_d2d(ws.mtp_hin, ws.h, H * 4)?;
        }

        // Verify [next, d1 .. dn]; the recurrent layers record instead of committing.
        let mut verify = vec![self.next];
        verify.extend_from_slice(&d);
        let bytes = forward::prepare(model, state, ws, &verify)?;
        forward::forward_part(model, state, ws, s, n + 1, bytes, Part::Whole, true, None)?;
        let g = argmax_rows(model, &self.head, s, ws.out, n + 1)?;
        let mut a = 0;
        while a < n && g[a] == d[a] {
            a += 1;
        }

        // Commit [next, d1 .. da]: replay a + 1 recorded tokens in the next step.
        let spec = state.spec.as_mut().context("sequence without speculative state")?;
        spec.parity ^= 1;
        spec.replay = (a + 1) as i32;
        s.upload_to(spec.ctl.addr(), &[spec.replay, spec.parity])?;
        state.tokens.truncate(p + a + 1);
        // The next round's catch-up pairs [next, d1 .. da] with [pending, rows 0 .. a-1] and its new token
        // with row a.
        s.copy_d2d(self.verified.addr(), ws.h, (a + 1) * H * 4)?;
        let committed = verify[..=a].to_vec();
        self.catch_up = committed.clone();
        self.next = g[a];
        self.rounds += 1;
        self.accepted += a;
        s.synchronize()?;
        Ok(committed)
    }
}

/// Greedy tokens for `rows` consecutive final hidden states (BF16 [rows, 2560]) at `hidden`.
fn argmax_rows(model: &Model, head: &HeadWorkspace, s: &Stream, hidden: u64, rows: usize) -> Result<Vec<i32>> {
    Ok(forward::head_stats(model, head, s, hidden, &vec![0; rows], 1)?.top_ids)
}

/// The speculative round as CUDA graphs: per draft length a set of MTP graphs (one per catch-up length:
/// catch-up, first draft and chained drafts, argmaxes on the device) and the verification split around
/// PLE like plain decode. Two host synchronizations per round: to read the drafts (the host gathers
/// their PLE rows) and to read the acceptance. With several draft lengths, [`DraftPolicy`] picks the
/// length of each round.
pub struct SpecGraphs {
    sets: Vec<GraphSet>,
    _rc: DeviceBuffer,
    rc_host: squash_cuda::PinnedBuffer,
    pub policy: DraftPolicy,
    /// Sampled rounds: the parameters and generator state, and the verified rows' top candidates.
    sample_params: DeviceBuffer,
    _top_ids: DeviceBuffer,
    _top_vals: DeviceBuffer,
    /// Distributions of the round's sampled drafts (one record per draft).
    _draft_q: DeviceBuffer,
    sampled: bool,
}

/// Candidates per verified row in sampled rounds (the request's top_k is a prefix of them).
pub const SAMPLE_TOP: usize = 64;

/// Sampled rounds draw their drafts at this percentage of the request's temperature (a sharper proposal
/// than the target's distribution is accepted more often; any proposal keeps the output exact).
const DRAFT_TEMPERATURE_PERCENT: u32 = 100;

struct GraphSet {
    n: usize,
    /// Indexed by catch-up length (0 ..= the longest draft length + 1).
    mtp: Vec<squash_cuda::Graph>,
    verify_before_ple: squash_cuda::Graph,
    verify_from_ple: squash_cuda::Graph,
    verify_from_ple_sampled: squash_cuda::Graph,
    bytes: usize,
}

/// Chooses the draft length that maximizes expected tokens per unit of round time. Acceptance is
/// estimated per draft position (draft i given drafts 1 .. i - 1 were accepted, decayed counts), so
/// expected tokens per round are 1 + sum_i prod_{j <= i} q_j. Round times are measured per draft
/// length (moving average); lengths not yet measured are extrapolated with round time a (1 + r n).
pub struct DraftPolicy {
    lengths: Vec<usize>,
    /// Per draft position: decayed (accepted, tried) counts.
    counts: Vec<(f64, f64)>,
    /// Per draft length: moving average of the round time in seconds.
    times: Vec<Option<f64>>,
    /// Relative cost of one more draft for unmeasured lengths.
    pub r: f64,
}

impl DraftPolicy {
    const DECAY: f64 = 0.9;

    pub fn new(lengths: Vec<usize>) -> Self {
        let max = lengths.iter().copied().max().unwrap_or(1);
        Self { lengths, counts: vec![(0.0, 0.0); max], times: vec![None; max + 1], r: 0.19 }
    }

    /// Estimated acceptance of draft position i (0-based), with a weak prior of 0.8.
    fn q(&self, i: usize) -> f64 {
        let (a, t) = self.counts[i];
        (a + 0.8) / (t + 1.0)
    }

    /// Round time of draft length n (relative units when nothing is measured yet).
    fn time(&self, n: usize) -> f64 {
        if let Some(t) = self.times[n] {
            return t;
        }
        let model = |m: usize| 1.0 + self.r * m as f64;
        match (1..self.times.len()).filter_map(|m| self.times[m].map(|t| (m, t))).min_by_key(|&(m, _)| m.abs_diff(n)) {
            Some((m, t)) => t * model(n) / model(m),
            None => model(n),
        }
    }

    pub fn choose(&self) -> usize {
        let value = |n: usize| {
            let (mut tokens, mut reach) = (1.0, 1.0);
            for i in 0..n {
                reach *= self.q(i);
                tokens += reach;
            }
            tokens / self.time(n)
        };
        *self.lengths.iter().max_by(|&&a, &&b| value(a).partial_cmp(&value(b)).unwrap()).unwrap()
    }

    /// A round with `n` drafts of which the first `a` were accepted, taking `secs`.
    pub fn record(&mut self, n: usize, a: usize, secs: f64) {
        self.times[n] = Some(match self.times[n] {
            Some(t) => 0.8 * t + 0.2 * secs,
            None => secs,
        });
        for (i, c) in self.counts.iter_mut().enumerate() {
            c.0 *= Self::DECAY;
            c.1 *= Self::DECAY;
            if i < n && i <= a {
                c.1 += 1.0;
                if i < a {
                    c.0 += 1.0;
                }
            }
        }
    }
}

impl SpecGraphs {
    /// Captures the graphs for `dec` (after its prefill) with the fixed draft length `dec.drafts`, or with
    /// every length 1 ..= `dec.drafts` when `adaptive`, and loads the round control buffer.
    pub fn capture(model: &Model, state: &SequenceState, ws: &mut Workspace, dec: &SpecDecoder, s: &Stream, adaptive: bool) -> Result<Self> {
        use squash_cuda::kernels as k;
        let nmax = dec.drafts;
        let (layout_max, draft_at, greedy_at, ints) = k::spec_layout();
        ensure!(nmax <= layout_max, "at most {layout_max} drafts");
        let rc = DeviceBuffer::zeroed(ints * 4)?;
        let sample_params = DeviceBuffer::zeroed(k::SAMPLE_PARAMS_BYTES)?;
        let top_ids = DeviceBuffer::new((nmax + 1) * SAMPLE_TOP * 4)?;
        let top_vals = DeviceBuffer::new((nmax + 1) * SAMPLE_TOP * 4)?;
        let draft_q = DeviceBuffer::zeroed(nmax * k::draft_q_bytes())?;
        let rc_host = squash_cuda::PinnedBuffer::new(ints * 4)?;
        let ctl = state.spec.as_ref().context("sequence without speculative state")?.ctl.addr();
        let (tok_off, pos_off) = forward::step_offsets(ws);
        let lengths: Vec<usize> = if adaptive { (1..=nmax).collect() } else { vec![nmax] };
        let mut sets = Vec::with_capacity(lengths.len());
        for &n in &lengths {
            let mut mtp = Vec::with_capacity(nmax + 2);
            for c in 0..=nmax + 1 {
                let g = squash_cuda::Graph::capture(s, |s| {
                    s.copy_d2d(ws.mtp_hin, dec.pending.addr(), H * 4)?;
                    if c > 0 {
                        s.copy_d2d(ws.mtp_hin + ROW, dec.verified.addr(), c * H * 4)?;
                        s.copy_d2d(dec.pending.addr(), dec.verified.addr() + ((c - 1) as u64) * ROW, H * 4)?;
                    }
                    for j in 0..n {
                        let t = if j == 0 { c + 1 } else { 1 };
                        k::spec_mtp_header(s, rc.addr(), ws.mtp_step, tok_off, pos_off, j, c)?;
                        forward::forward_mtp_staged(model, state, ws, s, t, ws.mtp_hin, None)?;
                        let q = draft_q.addr() + (j * k::draft_q_bytes()) as u64;
                        forward::draft_head_argmax_into(model, &dec.head, s, ws.out + ((t - 1) * D * 2) as u64, rc.addr() + ((draft_at + j) * 4) as u64, sample_params.addr(), q)?;
                        s.copy_d2d(ws.mtp_hin, ws.h + ((t - 1) as u64) * ROW, H * 4)?;
                    }
                    unsafe { s.copy_d2h(rc_host.as_ptr(), rc.addr(), ints * 4) }
                })?;
                mtp.push(g);
            }
            let bytes = forward::step_bytes(ws, n + 1);
            let verify_before_ple = squash_cuda::Graph::capture(s, |s| forward::forward_part(model, state, ws, s, n + 1, bytes, Part::BeforePle, true, None))?;
            let verify_from_ple = squash_cuda::Graph::capture(s, |s| {
                forward::forward_part(model, state, ws, s, n + 1, bytes, Part::FromPle, true, None)?;
                forward::head_argmax_into(model, &dec.head, s, ws.out, n + 1, rc.addr() + (greedy_at * 4) as u64)?;
                k::spec_accept(s, rc.addr(), ctl, n)?;
                s.copy_d2d(dec.verified.addr(), ws.h, (n + 1) * H * 4)?;
                unsafe { s.copy_d2h(rc_host.as_ptr(), rc.addr(), ints * 4) }
            })?;
            let verify_from_ple_sampled = squash_cuda::Graph::capture(s, |s| {
                forward::forward_part(model, state, ws, s, n + 1, bytes, Part::FromPle, true, None)?;
                forward::head_logits_rows(model, &dec.head, s, ws.out, n + 1)?;
                k::topk(s, dec.head.logits_addr(), n + 1, crate::model::VOCAB, crate::model::VOCAB, SAMPLE_TOP, false, top_ids.addr(), top_vals.addr())?;
                k::spec_sample_rows(s, top_ids.addr(), top_vals.addr(), SAMPLE_TOP, n, rc.addr(), sample_params.addr(), draft_q.addr())?;
                k::spec_accept(s, rc.addr(), ctl, n)?;
                s.copy_d2d(dec.verified.addr(), ws.h, (n + 1) * H * 4)?;
                unsafe { s.copy_d2h(rc_host.as_ptr(), rc.addr(), ints * 4) }
            })?;
            sets.push(GraphSet { n, mtp, verify_before_ple, verify_from_ple, verify_from_ple_sampled, bytes });
        }
        // Round state: next token at position len(state), nothing to catch up yet.
        s.upload_to(rc.addr(), &[state.len() as i32, dec.next, 0, 0])?;
        Ok(Self { sets, _rc: rc, rc_host, policy: DraftPolicy::new(lengths), sample_params, _top_ids: top_ids, _top_vals: top_vals, _draft_q: draft_q, sampled: false })
    }

    /// Greedy rounds (`None`), or sampled rounds with (temperature, top_p, top_k, seed); top_k is capped
    /// at [`SAMPLE_TOP`] (0: all of them). Sampled rounds draw their drafts from the drafter's
    /// distribution (`SQUASH_SAMPLED_DRAFTS=0`: its most likely token, as greedy rounds do).
    pub fn set_sampling(&mut self, s: &Stream, sampling: Option<(f32, f32, usize, u64)>) -> Result<()> {
        self.sampled = sampling.is_some();
        let mut b = vec![0u8; squash_cuda::kernels::SAMPLE_PARAMS_BYTES];
        if let Some((t, top_p, k, seed)) = sampling {
            let k = if k == 0 { SAMPLE_TOP } else { k.min(SAMPLE_TOP) } as u32;
            // Drafts are drawn at this percentage of the temperature (0: greedy drafts).
            let draft_sampled: u32 = std::env::var("SQUASH_SAMPLED_DRAFTS").ok().and_then(|v| v.parse().ok()).unwrap_or(DRAFT_TEMPERATURE_PERCENT);
            b.clear();
            b.extend_from_slice(&t.to_le_bytes());
            b.extend_from_slice(&top_p.to_le_bytes());
            b.extend_from_slice(&k.to_le_bytes());
            b.extend_from_slice(&draft_sampled.to_le_bytes());
            b.extend_from_slice(&seed.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
        }
        // Greedy rounds read the parameters too (the draft flag), so they are always rewritten.
        s.upload_to(self.sample_params.addr(), &b)
    }

    /// Points the round control at a decoder restarted on a new prompt.
    pub fn restart(&self, state: &SequenceState, dec: &SpecDecoder, s: &Stream) -> Result<()> {
        s.upload_to(self._rc.addr(), &[state.len() as i32, dec.next, 0, 0])
    }

    fn rc_word(&self, i: usize) -> i32 {
        i32::from_le_bytes(self.rc_host.as_slice()[4 * i..4 * i + 4].try_into().unwrap())
    }

    /// One round with the policy's draft length; returns the committed tokens.
    pub fn round(&mut self, model: &Model, state: &mut SequenceState, ws: &mut Workspace, dec: &mut SpecDecoder, s: &Stream) -> Result<Vec<i32>> {
        let (_, draft_at, _, _) = squash_cuda::kernels::spec_layout();
        let n = self.policy.choose();
        let set = self.sets.iter().find(|g| g.n == n).expect("a graph set per draft length");
        let c = dec.catch_up.len();
        let t0 = std::time::Instant::now();
        set.mtp[c].launch(s)?;
        s.synchronize()?;
        let t1 = std::time::Instant::now();
        dec.draft_secs += (t1 - t0).as_secs_f64();
        let mut verify = vec![dec.next];
        verify.extend((0..n).map(|j| self.rc_word(draft_at + j)));
        let p = state.len();
        forward::prepare_tokens(state, ws, &verify)?;
        set.verify_before_ple.launch(s)?;
        let bytes = forward::prepare_ple(model, state, ws, n + 1)?;
        debug_assert_eq!(bytes, set.bytes);
        if self.sampled { set.verify_from_ple_sampled.launch(s)? } else { set.verify_from_ple.launch(s)? }
        s.synchronize()?;
        dec.verify_secs += t1.elapsed().as_secs_f64();
        let a = self.rc_word(3) as usize;
        state.tokens.truncate(p + a + 1);
        let committed = verify[..=a].to_vec();
        dec.catch_up = committed.clone();
        dec.next = self.rc_word(1);
        dec.rounds += 1;
        dec.accepted += a;
        self.policy.record(n, a, t0.elapsed().as_secs_f64());
        Ok(committed)
    }
}
