//! MTP draft-quality probe: greedy decoding with the target, drafting three tokens ahead with the MTP
//! layer at every step (without verification), and measuring how often the drafts match what the
//! target then produces.

use std::path::Path;

use anyhow::Result;
use squash_cuda::DeviceBuffer;
use squash_engine::forward::{self, HeadWorkspace, SequenceState, Workspace};
use squash_engine::model::{D, H, Precision};

const DRAFTS: usize = 3;

pub fn probe(model_dir: &Path, precision: Precision, fixture: &Path, steps: usize, chunk: usize) -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Sequence {
        tokens: Vec<i32>,
        prompt_len: usize,
    }
    let seq: Sequence = serde_json::from_slice(&std::fs::read(fixture.join("sequence.json"))?)?;
    let (model, s) = crate::load(model_dir, precision, true)?;
    let prompt = &seq.tokens[..seq.prompt_len];
    let cap = prompt.len() + steps + DRAFTS + 8;
    let mut ws = Workspace::new(chunk, cap)?;
    let hw = HeadWorkspace::new(1, 1)?;
    let mut state = SequenceState::with_mtp(cap, true)?;
    let pending = DeviceBuffer::zeroed(H * 4)?;
    let row = (H * 4) as u64;

    // Prefill: the MTP layer sees each prompt token paired with the previous position's hidden row.
    for c0 in (0..prompt.len()).step_by(chunk) {
        let c = &prompt[c0..(c0 + chunk).min(prompt.len())];
        let t = c.len();
        forward::forward(&model, &mut state, &mut ws, &s, c)?;
        s.copy_d2d(ws.mtp_hin, pending.addr(), H * 4)?;
        s.copy_d2d(ws.mtp_hin + row, ws.h, (t - 1) * H * 4)?;
        s.copy_d2d(pending.addr(), ws.h + (t - 1) as u64 * row, H * 4)?;
        forward::prepare_mtp(&mut ws, c0, c)?;
        forward::forward_mtp(&model, &state, &ws, &s, t, ws.mtp_hin, None)?;
        s.synchronize()?;
    }
    // The prompt's last logits give the first token; ws.out was overwritten by the MTP step, so recompute
    // from the target: run the last prompt token's head on the target output saved before. Simpler: use
    // the fixture's first completion token as the next token (the reference is greedy).
    let mut next = seq.tokens[seq.prompt_len];
    let mut target: Vec<i32> = Vec::new(); // target greedy tokens after `next`, in order
    let mut drafts: Vec<[i32; DRAFTS]> = Vec::new();
    for _ in 0..steps {
        let p = state.len();
        // Draft chain from (h_{p-1}, next) at p.
        let mut d = [0i32; DRAFTS];
        s.copy_d2d(ws.mtp_hin, pending.addr(), H * 4)?;
        let mut tok = next;
        for (j, slot) in d.iter_mut().enumerate() {
            forward::prepare_mtp(&mut ws, p + j, &[tok])?;
            forward::forward_mtp(&model, &state, &ws, &s, 1, ws.mtp_hin, None)?;
            *slot = forward::head_stats(&model, &hw, &s, ws.out, &[0], 1)?.top_ids[0];
            s.copy_d2d(ws.mtp_hin, ws.h, H * 4)?;
            tok = *slot;
        }
        drafts.push(d);
        // Target step on `next`.
        forward::forward(&model, &mut state, &mut ws, &s, &[next])?;
        s.copy_d2d(pending.addr(), ws.h, H * 4)?;
        next = forward::head_stats(&model, &hw, &s, ws.out, &[0], 1)?.top_ids[0];
        target.push(next);
    }
    // drafts[i][j] predicts target[i + j].
    let mut prefix = [0usize; DRAFTS + 1];
    let mut rounds = 0;
    for i in 0..steps.saturating_sub(DRAFTS) {
        rounds += 1;
        let mut a = 0;
        while a < DRAFTS && drafts[i][a] == target[i + a] {
            a += 1;
        }
        prefix[a] += 1;
    }
    let n = rounds as f64;
    let at_least = |k: usize| prefix[k..].iter().sum::<usize>() as f64 / n;
    println!("{} rounds: P(accept >= 1) {:.3}, >= 2 {:.3}, >= 3 {:.3}", rounds, at_least(1), at_least(2), at_least(3));
    let mean: f64 = (0..=DRAFTS).map(|a| a as f64 * prefix[a] as f64).sum::<f64>() / n;
    println!("expected tokens per verification round with 3 drafts: {:.2}", 1.0 + mean);
    let _ = D;
    Ok(())
}
