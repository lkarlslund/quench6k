//! Greedy speculative decoding with the MTP drafter (SPEC §10).
//!
//! Each round drafts `drafts` tokens with the MTP layer, verifies them with one target step over
//! `[x, d1 .. dn]`, and keeps the longest prefix the target agrees with plus the target's own next
//! token. Recurrent state is committed one step behind (see `forward::Spec`), so rejected drafts never
//! reach it; attention caches are position-indexed and simply get overwritten.

use anyhow::{Context, Result, ensure};
use quench6k_cuda::{DeviceBuffer, Stream};

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
    /// The DFlash drafter's state, when the model drafts with it instead of the MTP layer.
    pub dflash: Option<crate::dflash::DState>,
}

const ROW: u64 = (H * 4) as u64;

impl SpecDecoder {
    /// Prefills `prompt` into `state` (target and MTP caches) and returns the decoder positioned on the
    /// target's greedy next token.
    pub fn prefill(model: &Model, state: &mut SequenceState, ws: &mut Workspace, s: &Stream, prompt: &[i32], drafts: usize, chunk: usize) -> Result<Self> {
        ensure!(state.spec.is_some() && (model.mtp.is_some() || model.dflash.is_some()), "speculative decoding needs a drafter and a sequence with speculative state");
        let max = if model.dflash.is_some() { crate::dflash::DRAFTS } else { 8 };
        ensure!((1..=max).contains(&drafts), "draft length {drafts} out of range (1 ..= {max})");
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
            dflash: if model.dflash.is_some() { Some(crate::dflash::DState::new(crate::dflash::BLOCK)?) } else { None },
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

    /// As [`restart`](Self::restart), breaking the chunking at `pauses` (as a fresh prompt is chunked at
    /// turn boundaries) and reporting `progress`.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_new(&mut self, model: &Model, state: &mut SequenceState, ws: &mut Workspace, s: &Stream, prompt: &[i32], chunk: usize, pauses: &[usize], progress: Option<&(dyn Fn(usize) + Send + Sync)>) -> Result<()> {
        s.memset(self.pending.addr(), 0, H * 4)?;
        self.prefill_from(model, state, ws, s, prompt, 0, chunk, pauses, |_, _, _| Ok(()), progress)
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
        // DFlash: the drafter's context ring gets the features of the prompt's last RING positions.
        let lo = prompt.len().saturating_sub(crate::dflash::RING).max(start);
        if let Some(st) = &self.dflash {
            ensure!(ws.dflash.is_some(), "workspace without DFlash scratch");
            st.set_lo(s, lo)?;
        }
        while c0 < prompt.len() {
            let mut end = (c0 + chunk).min(prompt.len());
            if let Some(&p) = pause.peek() {
                end = end.min(p);
            }
            let c = &prompt[c0..end];
            let t = c.len();
            let first = lo.max(c0).min(end) - c0;
            if self.dflash.is_some() && first < t {
                let buf = ws.dflash.as_ref().map(|sc| sc.feat).context("workspace without DFlash scratch")?;
                ws.taps = Some(crate::dflash::Taps { buf, first, rows: t - first });
            }
            let r = forward::forward(model, state, ws, s, c);
            ws.taps = None;
            r?;
            if end == prompt.len() {
                next = argmax_rows(model, &self.head, s, ws.out + ((t - 1) * D * 2) as u64, 1)?[0];
            }
            if let (Some(st), Some(d)) = (&self.dflash, &model.dflash) {
                let sc = ws.dflash.as_ref().context("workspace without DFlash scratch")?;
                crate::dflash::context(d, sc, st, s, sc.feat, t - first, crate::dflash::Pos(0, c0 + first))?;
            } else {
                s.copy_d2d(ws.mtp_hin, self.pending.addr(), H * 4)?;
                s.copy_d2d(ws.mtp_hin + ROW, ws.h, (t - 1) * H * 4)?;
                s.copy_d2d(self.pending.addr(), ws.h + (t - 1) as u64 * ROW, H * 4)?;
                forward::prepare_mtp(ws, c0, c)?;
                forward::forward_mtp(model, state, ws, s, t, ws.mtp_hin, None)?;
            }
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
        ensure!(self.dflash.is_none(), "DFlash rounds run as graphs (SpecGraphs)");
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
    /// DFlash drafting graphs, indexed by catch-up length (the same for every draft length); empty
    /// when the MTP layer drafts.
    draft: Vec<quench6k_cuda::Graph>,
    _rc: DeviceBuffer,
    rc_host: quench6k_cuda::PinnedBuffer,
    pub policy: DraftPolicy,
    /// Sampled rounds: the parameters and generator state, and the verified rows' top candidates.
    sample_params: DeviceBuffer,
    _top_ids: DeviceBuffer,
    _top_vals: DeviceBuffer,
    /// Distributions of the round's sampled drafts (one record per draft).
    _draft_q: DeviceBuffer,
    sampled: bool,
    /// Sampled rounds penalize the verified rows (the session's slot-0 penalty history).
    pub penalized: bool,
}

/// Candidates per verified row in sampled rounds (the request's top_k is a prefix of them).
pub const SAMPLE_TOP: usize = 64;

/// Sampling parameters of a request (temperature > 0).
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub seed: u64,
}

/// Sampled rounds draw their drafts at this percentage of the request's temperature (a sharper proposal
/// than the target's distribution is accepted more often; any proposal keeps the output exact).
const DRAFT_TEMPERATURE_PERCENT: u32 = 100;

struct GraphSet {
    n: usize,
    /// Indexed by catch-up length (0 ..= the longest draft length + 1); empty with DFlash.
    mtp: Vec<quench6k_cuda::Graph>,
    verify_before_ple: quench6k_cuda::Graph,
    verify_from_ple: quench6k_cuda::Graph,
    verify_from_ple_sampled: quench6k_cuda::Graph,
    /// The sampled verification with repetition/presence penalties: the verified rows are penalized
    /// before top-k, and the committed tokens join the history.
    verify_from_ple_penalized: Option<quench6k_cuda::Graph>,
    /// Pair rounds ([`PairGraphs`]): the head, acceptance and bookkeeping of this sequence's rows of a
    /// batched verification step, per mode (greedy, sampled, penalized).
    pair_tail: [Option<quench6k_cuda::Graph>; 3],
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
    decay: f64,
    prior: f64,
    /// Always this draft length (`QUENCH6K_POLICY=fixed=N`, for comparisons).
    fixed: Option<usize>,
}

impl DraftPolicy {
    pub fn new(lengths: Vec<usize>) -> Self {
        let max = lengths.iter().copied().max().unwrap_or(1);
        let mut p = Self { lengths, counts: vec![(0.0, 0.0); max], times: vec![None; max + 1], r: 0.19, decay: 0.9, prior: 0.8, fixed: None };
        // Experiments: QUENCH6K_POLICY=decay=0.95,prior=0.6,fixed=3
        for kv in std::env::var("QUENCH6K_POLICY").unwrap_or_default().split(',').filter(|kv| !kv.is_empty()) {
            match kv.split_once('=') {
                Some(("decay", v)) => p.decay = v.parse().expect("decay"),
                Some(("prior", v)) => p.prior = v.parse().expect("prior"),
                Some(("fixed", v)) => p.fixed = Some(v.parse::<usize>().expect("fixed")).filter(|n| p.lengths.contains(n)),
                _ => panic!("QUENCH6K_POLICY: unknown setting {kv}"),
            }
        }
        p
    }

    /// Estimated acceptance of draft position i (0-based), with a weak prior (0.8).
    fn q(&self, i: usize) -> f64 {
        let (a, t) = self.counts[i];
        (a + self.prior) / (t + 1.0)
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
        if let Some(n) = self.fixed {
            return n;
        }
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
            c.0 *= self.decay;
            c.1 *= self.decay;
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
    /// `pen`: the penalty history (state, flags, list; see `kernels::penalty_apply`), for penalized rounds.
    /// `pair_slot`: also capture the tails of pair rounds, for this sequence as segment 0 or 1 of the
    /// batched verification.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(model: &Model, state: &SequenceState, ws: &mut Workspace, dec: &SpecDecoder, s: &Stream, adaptive: bool, pen: Option<(u64, u64, u64)>, pair_slot: Option<usize>) -> Result<Self> {
        use quench6k_cuda::kernels as k;
        let nmax = dec.drafts;
        let (layout_max, draft_at, greedy_at, ints) = k::spec_layout();
        let (commit_at, commit_count) = k::spec_commit_layout();
        ensure!(nmax <= layout_max, "at most {layout_max} drafts");
        let rc = DeviceBuffer::zeroed(ints * 4)?;
        let sample_params = DeviceBuffer::zeroed(k::SAMPLE_PARAMS_BYTES)?;
        let top_ids = DeviceBuffer::new((nmax + 1) * SAMPLE_TOP * 4)?;
        let top_vals = DeviceBuffer::new((nmax + 1) * SAMPLE_TOP * 4)?;
        let dflash = dec.dflash.as_ref().zip(model.dflash.as_ref());
        ensure!(dflash.is_none() || nmax <= crate::dflash::DRAFTS, "DFlash drafts at most {} tokens", crate::dflash::DRAFTS);
        // DFlash writes the distributions of all 7 drafts.
        let draft_q = DeviceBuffer::zeroed(nmax.max(if dflash.is_some() { crate::dflash::DRAFTS } else { 0 }) * k::draft_q_bytes())?;
        let rc_host = quench6k_cuda::PinnedBuffer::new(ints * 4)?;
        let ctl = state.spec.as_ref().context("sequence without speculative state")?.ctl.addr();
        let (tok_off, pos_off) = forward::step_offsets(ws);
        let lengths: Vec<usize> = if adaptive { (1..=nmax).collect() } else { vec![nmax] };
        let mut draft = Vec::new();
        let vfeat = match dflash {
            Some((st, d)) => {
                let sc = ws.dflash.as_ref().context("workspace without DFlash scratch")?;
                for c in 0..=nmax + 1 {
                    draft.push(quench6k_cuda::Graph::capture(s, |s| {
                        crate::dflash::draft(model, d, sc, st, s, c, rc.addr(), rc.addr() + 4, rc.addr() + (draft_at * 4) as u64, draft_q.addr(), sample_params.addr(), None)?;
                        unsafe { s.copy_d2h(rc_host.as_ptr(), rc.addr(), ints * 4) }
                    })?);
                }
                Some(sc.vfeat)
            }
            None => None,
        };
        let mut sets = Vec::with_capacity(lengths.len());
        for &n in &lengths {
            let mut mtp = Vec::with_capacity(nmax + 2);
            for c in (0..=nmax + 1).filter(|_| dflash.is_none()) {
                let g = quench6k_cuda::Graph::capture(s, |s| {
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
            let bytes = forward::step_bytes(n + 1);
            // DFlash: the verification captures the drafter's features of its rows.
            ws.taps = vfeat.map(|buf| crate::dflash::Taps { buf, first: 0, rows: n + 1 });
            let ws: &Workspace = ws;
            let verify_before_ple = quench6k_cuda::Graph::capture(s, |s| forward::forward_part(model, state, ws, s, n + 1, bytes, Part::BeforePle, true, None))?;
            // The verification's head, sampling, acceptance and bookkeeping for rows `off ..= off + n` of
            // the step output. Modes: 0 greedy, 1 sampled, 2 sampled with penalties.
            let tail = |s: &Stream, mode: usize, off: usize| -> Result<()> {
                let (out, h) = (ws.out + (off * D * 2) as u64, ws.h + off as u64 * ROW);
                if mode == 0 {
                    forward::head_argmax_into(model, &dec.head, s, out, n + 1, rc.addr() + (greedy_at * 4) as u64)?;
                } else {
                    forward::head_logits_rows(model, &dec.head, s, out, n + 1)?;
                    if mode == 2 {
                        let pen = pen.context("graphs captured without penalties")?;
                        k::penalty_apply(s, dec.head.logits_addr(), n + 1, crate::model::VOCAB, pen.0, pen.1, pen.2, rc.addr())?;
                    }
                    k::topk(s, dec.head.logits_addr(), n + 1, crate::model::VOCAB, crate::model::VOCAB, SAMPLE_TOP, false, top_ids.addr(), top_vals.addr())?;
                    k::spec_sample_rows(s, top_ids.addr(), top_vals.addr(), SAMPLE_TOP, n, rc.addr(), sample_params.addr(), draft_q.addr())?;
                }
                k::spec_accept(s, rc.addr(), ctl, n)?;
                if mode == 2 {
                    let pen = pen.context("graphs captured without penalties")?;
                    k::penalty_mark(s, pen.0, pen.1, pen.2, rc.addr() + (commit_at * 4) as u64, 0, rc.addr() + (commit_count * 4) as u64, 3)?;
                }
                s.copy_d2d(dec.verified.addr(), h, (n + 1) * H * 4)?;
                if let (Some(st), Some(vf)) = (&dec.dflash, vfeat) {
                    s.copy_d2d(st.feat.addr(), vf + (off * crate::dflash::FEAT_BYTES) as u64, (n + 1) * crate::dflash::FEAT_BYTES)?;
                }
                unsafe { s.copy_d2h(rc_host.as_ptr(), rc.addr(), ints * 4) }
            };
            let verify = |mode: usize| quench6k_cuda::Graph::capture(s, |s| {
                forward::forward_part(model, state, ws, s, n + 1, bytes, Part::FromPle, true, None)?;
                tail(s, mode, 0)
            });
            let verify_from_ple = verify(0)?;
            let verify_from_ple_sampled = verify(1)?;
            let verify_from_ple_penalized = pen.map(|_| verify(2)).transpose()?;
            let mut pair_tail = [None, None, None];
            if let Some(slot) = pair_slot {
                let off = slot * (n + 1);
                for (mode, g) in pair_tail.iter_mut().enumerate() {
                    if mode < 2 || pen.is_some() {
                        *g = Some(quench6k_cuda::Graph::capture(s, |s| tail(s, mode, off))?);
                    }
                }
            }
            sets.push(GraphSet { n, mtp, verify_before_ple, verify_from_ple, verify_from_ple_sampled, verify_from_ple_penalized, pair_tail, bytes });
        }
        ws.taps = None;
        // Round state: next token at position len(state), nothing to catch up yet.
        s.upload_to(rc.addr(), &[state.len() as i32, dec.next, 0, 0])?;
        let mut policy = DraftPolicy::new(lengths);
        if dflash.is_some() {
            // Drafting costs the same for every length; only the verification grows.
            policy.r = 0.05;
        }
        Ok(Self { sets, draft, _rc: rc, rc_host, policy, sample_params, _top_ids: top_ids, _top_vals: top_vals, _draft_q: draft_q, sampled: false, penalized: false })
    }

    /// Greedy rounds (`None`), or sampled rounds; top_k is capped at [`SAMPLE_TOP`] (0: all of them).
    /// Sampled rounds draw their drafts from the drafter's distribution (`QUENCH6K_SAMPLED_DRAFTS=0`: its
    /// most likely token, as greedy rounds do).
    pub fn set_sampling(&mut self, s: &Stream, sampling: Option<Sampling>) -> Result<()> {
        self.sampled = sampling.is_some();
        let mut b = vec![0u8; quench6k_cuda::kernels::SAMPLE_PARAMS_BYTES];
        if let Some(Sampling { temperature: t, top_p, top_k: k, min_p, seed }) = sampling {
            let k = if k == 0 { SAMPLE_TOP } else { k.min(SAMPLE_TOP) } as u32;
            // Drafts are drawn at this percentage of the temperature (0: greedy drafts).
            // Top-1 rounds (penalized greedy) take the drafter's most likely token.
            let draft_sampled: u32 = if k == 1 { 0 } else { std::env::var("QUENCH6K_SAMPLED_DRAFTS").ok().and_then(|v| v.parse().ok()).unwrap_or(DRAFT_TEMPERATURE_PERCENT) };
            b.clear();
            b.extend_from_slice(&t.to_le_bytes());
            b.extend_from_slice(&top_p.to_le_bytes());
            b.extend_from_slice(&k.to_le_bytes());
            b.extend_from_slice(&draft_sampled.to_le_bytes());
            b.extend_from_slice(&seed.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
            b.extend_from_slice(&min_p.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
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
        let (_, draft_at, _, _) = quench6k_cuda::kernels::spec_layout();
        let n = self.policy.choose();
        let set = self.sets.iter().find(|g| g.n == n).expect("a graph set per draft length");
        let c = dec.catch_up.len();
        // Debugging: QUENCH6K_DFLASH_PROFILE=N times round N's drafting eagerly, per op, before the graph
        // repeats it (context rows are rewritten in place; greedy drafts are the same).
        if let (Some(at), Some(st), Some(d), Some(sc)) = (std::env::var("QUENCH6K_DFLASH_PROFILE").ok().and_then(|v| v.parse::<usize>().ok()), &dec.dflash, &model.dflash, &ws.dflash)
            && at == dec.rounds
        {
            let mut prof = crate::profile::Profiler::default();
            for _ in 0..10 {
                crate::dflash::draft(model, d, sc, st, s, c, self._rc.addr(), self._rc.addr() + 4, self._rc.addr() + (draft_at * 4) as u64, self._draft_q.addr(), self.sample_params.addr(), Some(&mut prof))?;
            }
            prof.collect()?;
            eprintln!("DFlash drafting (c = {c}), 10 eager steps:\n{}", prof.report());
        }
        let t0 = std::time::Instant::now();
        self.draft_graph(set, c).launch(s)?;
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
        if self.penalized {
            set.verify_from_ple_penalized.as_ref().context("graphs captured without penalties")?.launch(s)?
        } else if self.sampled {
            set.verify_from_ple_sampled.launch(s)?
        } else {
            set.verify_from_ple.launch(s)?
        }
        s.synchronize()?;
        dec.verify_secs += t1.elapsed().as_secs_f64();
        let (a, committed) = self.commit(state, dec, &verify, p);
        self.policy.record(n, a, t0.elapsed().as_secs_f64());
        Ok(committed)
    }

    /// After a verification (read back into `rc_host`): truncates the sequence to the accepted tokens
    /// and positions the decoder on the target's next token. Returns the accepted draft count and the
    /// committed tokens.
    fn commit(&self, state: &mut SequenceState, dec: &mut SpecDecoder, verify: &[i32], p: usize) -> (usize, Vec<i32>) {
        let a = self.rc_word(3) as usize;
        state.tokens.truncate(p + a + 1);
        let committed = verify[..=a].to_vec();
        dec.catch_up = committed.clone();
        dec.next = self.rc_word(1);
        dec.rounds += 1;
        dec.accepted += a;
        (a, committed)
    }

    /// The drafting graph of a round with draft set `set` after `c` committed tokens.
    fn draft_graph<'a>(&'a self, set: &'a GraphSet, c: usize) -> &'a quench6k_cuda::Graph {
        if self.draft.is_empty() { &set.mtp[c] } else { &self.draft[c] }
    }

    /// The tail mode of this sequence's rounds (greedy, sampled, penalized).
    fn mode(&self) -> usize {
        if self.penalized { 2 } else if self.sampled { 1 } else { 0 }
    }
}

/// Speculative rounds of two sequences together: both drafters run, then one batched verification
/// step over `[x, d1 .. dn]` of each (segments of [`forward::forward_segments`], recurrent commits
/// deferred per sequence), then each sequence's own tail (head, sampling, acceptance; captured in its
/// [`SpecGraphs`] with `pair_slot`). Both sequences draft the same length, chosen by a policy fed
/// with both sequences' acceptance.
pub struct PairGraphs {
    sets: Vec<(usize, quench6k_cuda::Graph, quench6k_cuda::Graph, usize)>,
    pub policy: DraftPolicy,
}

impl PairGraphs {
    /// Captures the batched verification for every draft length both decoders support (1 ..= the
    /// shorter longest draft) over `states` (slot 0, slot 1).
    pub fn capture(model: &Model, states: [&SequenceState; 2], ws: &mut Workspace, s: &Stream, max_drafts: usize) -> Result<Self> {
        let mut sets = Vec::with_capacity(max_drafts);
        let vfeat = ws.dflash.as_ref().filter(|_| model.dflash.is_some()).map(|sc| sc.vfeat);
        for n in 1..=max_drafts {
            ws.taps = vfeat.map(|buf| crate::dflash::Taps { buf, first: 0, rows: 2 * (n + 1) });
            let ws: &Workspace = ws;
            let segs = [forward::Seg { state: states[0], rows: n + 1, defer: Some(true) }, forward::Seg { state: states[1], rows: n + 1, defer: Some(true) }];
            let bytes = forward::segments_bytes(2 * (n + 1));
            let before = quench6k_cuda::Graph::capture(s, |s| forward::forward_segments(model, &segs, ws, s, bytes, Part::BeforePle))?;
            let from = quench6k_cuda::Graph::capture(s, |s| forward::forward_segments(model, &segs, ws, s, bytes, Part::FromPle))?;
            sets.push((n, before, from, bytes));
        }
        ws.taps = None;
        Ok(Self { sets, policy: DraftPolicy::new((1..=max_drafts).collect()) })
    }

    /// One round of both sequences; returns each one's committed tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn round(&mut self, model: &Model, ws: &mut Workspace, s: &Stream, a: (&mut SpecGraphs, &mut SequenceState, &mut SpecDecoder), b: (&mut SpecGraphs, &mut SequenceState, &mut SpecDecoder)) -> Result<[Vec<i32>; 2]> {
        let (_, draft_at, _, _) = quench6k_cuda::kernels::spec_layout();
        let (ga, sa, da) = a;
        let (gb, sb, db) = b;
        let n = self.policy.choose();
        let (_, before, from, bytes) = self.sets.iter().find(|x| x.0 == n).context("no pair graphs for this draft length")?;
        let seta = ga.sets.iter().find(|g| g.n == n).context("slot 0 lacks this draft length")?;
        let setb = gb.sets.iter().find(|g| g.n == n).context("slot 1 lacks this draft length")?;
        let t0 = std::time::Instant::now();
        ga.draft_graph(seta, da.catch_up.len()).launch(s)?;
        gb.draft_graph(setb, db.catch_up.len()).launch(s)?;
        s.synchronize()?;
        let t1 = std::time::Instant::now();
        let mut va = vec![da.next];
        va.extend((0..n).map(|j| ga.rc_word(draft_at + j)));
        let mut vb = vec![db.next];
        vb.extend((0..n).map(|j| gb.rc_word(draft_at + j)));
        let (pa, pb) = (sa.len(), sb.len());
        forward::prepare_segments_tokens(&mut [&mut *sa, &mut *sb], ws, &[&va, &vb])?;
        before.launch(s)?;
        let used = forward::prepare_segments_ple(model, &[&*sa, &*sb], ws, &[n + 1, n + 1])?;
        debug_assert_eq!(used, *bytes);
        from.launch(s)?;
        seta.pair_tail[ga.mode()].as_ref().context("slot 0 graphs captured without pair tails")?.launch(s)?;
        setb.pair_tail[gb.mode()].as_ref().context("slot 1 graphs captured without pair tails")?.launch(s)?;
        s.synchronize()?;
        let (draft, verify) = ((t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
        for d in [&mut *da, &mut *db] {
            d.draft_secs += draft / 2.0;
            d.verify_secs += verify / 2.0;
        }
        let (acc_a, ca) = ga.commit(sa, da, &va, pa);
        let (acc_b, cb) = gb.commit(sb, db, &vb, pb);
        let secs = t0.elapsed().as_secs_f64();
        self.policy.record(n, acc_a, secs);
        self.policy.record(n, acc_b, secs);
        Ok([ca, cb])
    }
}
