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
        let pending = DeviceBuffer::zeroed(H * 4)?;
        let head = HeadWorkspace::new(drafts + 1, 1)?;
        let mut next = 0;
        for c0 in (0..prompt.len()).step_by(chunk) {
            let c = &prompt[c0..(c0 + chunk).min(prompt.len())];
            let t = c.len();
            forward::forward(model, state, ws, s, c)?;
            if c0 + t == prompt.len() {
                next = argmax_rows(model, &head, s, ws.out + ((t - 1) * D * 2) as u64, 1)?[0];
            }
            s.copy_d2d(ws.mtp_hin, pending.addr(), H * 4)?;
            s.copy_d2d(ws.mtp_hin + ROW, ws.h, (t - 1) * H * 4)?;
            s.copy_d2d(pending.addr(), ws.h + (t - 1) as u64 * ROW, H * 4)?;
            forward::prepare_mtp(ws, c0, c)?;
            forward::forward_mtp(model, state, ws, s, t, ws.mtp_hin, None)?;
        }
        s.synchronize()?;
        Ok(Self { drafts, pending, verified: DeviceBuffer::new((drafts + 1) * H * 4)?, catch_up: Vec::new(), head, next, rounds: 0, accepted: 0, draft_secs: 0.0, verify_secs: 0.0 })
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

/// The speculative round as CUDA graphs: per catch-up length an MTP graph (catch-up, first draft and
/// chained drafts, argmaxes on the device), and the verification split around PLE like plain decode.
/// Two host synchronizations per round: to read the drafts (the host gathers their PLE rows) and to
/// read the acceptance.
pub struct SpecGraphs {
    mtp: Vec<squash_cuda::Graph>,
    verify_before_ple: squash_cuda::Graph,
    verify_from_ple: squash_cuda::Graph,
    _rc: DeviceBuffer,
    rc_host: squash_cuda::PinnedBuffer,
    bytes: usize,
    n: usize,
}

impl SpecGraphs {
    /// Captures the graphs for `dec` (after its prefill) and loads the round control buffer.
    pub fn capture(model: &Model, state: &SequenceState, ws: &mut Workspace, dec: &SpecDecoder, s: &Stream) -> Result<Self> {
        use squash_cuda::kernels as k;
        let n = dec.drafts;
        let (nmax, draft_at, greedy_at, ints) = k::spec_layout();
        ensure!(n <= nmax, "at most {nmax} drafts");
        let rc = DeviceBuffer::zeroed(ints * 4)?;
        let rc_host = squash_cuda::PinnedBuffer::new(ints * 4)?;
        let ctl = state.spec.as_ref().context("sequence without speculative state")?.ctl.addr();
        let (tok_off, pos_off) = forward::step_offsets(ws);
        let mut mtp = Vec::with_capacity(n + 2);
        for c in 0..=n + 1 {
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
                    forward::draft_head_argmax_into(model, &dec.head, s, ws.out + ((t - 1) * D * 2) as u64, rc.addr() + ((draft_at + j) * 4) as u64)?;
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
        // Round state: next token at position len(state), nothing to catch up yet.
        s.upload_to(rc.addr(), &[state.len() as i32, dec.next, 0, 0])?;
        Ok(Self { mtp, verify_before_ple, verify_from_ple, _rc: rc, rc_host, bytes, n })
    }

    fn rc_word(&self, i: usize) -> i32 {
        i32::from_le_bytes(self.rc_host.as_slice()[4 * i..4 * i + 4].try_into().unwrap())
    }

    /// One round; returns the committed tokens.
    pub fn round(&self, model: &Model, state: &mut SequenceState, ws: &mut Workspace, dec: &mut SpecDecoder, s: &Stream) -> Result<Vec<i32>> {
        let (_, draft_at, _, _) = squash_cuda::kernels::spec_layout();
        let c = dec.catch_up.len();
        let t0 = std::time::Instant::now();
        self.mtp[c].launch(s)?;
        s.synchronize()?;
        let t1 = std::time::Instant::now();
        dec.draft_secs += (t1 - t0).as_secs_f64();
        let mut verify = vec![dec.next];
        verify.extend((0..self.n).map(|j| self.rc_word(draft_at + j)));
        let p = state.len();
        forward::prepare_tokens(state, ws, &verify)?;
        self.verify_before_ple.launch(s)?;
        let bytes = forward::prepare_ple(model, state, ws, self.n + 1)?;
        debug_assert_eq!(bytes, self.bytes);
        self.verify_from_ple.launch(s)?;
        s.synchronize()?;
        dec.verify_secs += t1.elapsed().as_secs_f64();
        let a = self.rc_word(3) as usize;
        state.tokens.truncate(p + a + 1);
        let committed = verify[..=a].to_vec();
        dec.catch_up = committed.clone();
        dec.next = self.rc_word(1);
        dec.rounds += 1;
        dec.accepted += a;
        Ok(committed)
    }
}
