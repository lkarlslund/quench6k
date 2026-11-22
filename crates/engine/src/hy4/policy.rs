//! How many of the drafter's tokens a step verifies: the count with the most expected tokens per unit of
//! time, from running estimates of each draft position's acceptance (given the ones before it) and of
//! the cost of verification steps by width.

/// Most drafts a step can verify.
pub const MAX_DRAFTS: usize = 8;

pub struct DraftPolicy {
    /// Drafts available per pass.
    drafts: usize,
    /// accept[k]: chance draft k + 1 is accepted when the ones before it were.
    accept: [f64; MAX_DRAFTS],
    /// cost[w]: seconds of a verification step of w tokens (index 0 unused).
    cost: [f64; MAX_DRAFTS + 2],
    /// Seconds of a drafter pass.
    draft: f64,
    steps: usize,
    /// Calibration of drafters that state each draft's probability: kept and judged drafts (pseudo-counts
    /// included) per tenth of the stated probability.
    calib: [(f64, f64); 10],
    /// Tokens per second of recent drafted steps (drafter pass included).
    drafted_rate: f64,
}

impl DraftPolicy {
    /// `drafts` per pass; `step` and `per_token`: initial guesses of a one-token step's time and of each
    /// further token's, `draft` of a drafter pass (all refined as steps are timed).
    pub fn new(drafts: usize, step: f64, per_token: f64, draft: f64) -> Self {
        let drafts = drafts.min(MAX_DRAFTS);
        let mut cost = [0.0; MAX_DRAFTS + 2];
        for (w, c) in cost.iter_mut().enumerate().skip(1) {
            *c = step + per_token * (w - 1) as f64;
        }
        Self {
            drafts,
            accept: [0.8; MAX_DRAFTS],
            cost,
            draft,
            steps: 0,
            // Prior: stated probabilities are taken at 80 %, worth 4 drafts per tenth.
            calib: std::array::from_fn(|i| (4.0 * 0.8 * (i as f64 + 0.5) / 10.0, 4.0)),
            drafted_rate: 2.0 / step,
        }
    }

    fn calibrated(&self, conf: f32) -> f64 {
        let (kept, judged) = self.calib[((conf.max(0.0) * 10.0) as usize).min(9)];
        kept / judged
    }

    /// Whether to run the drafter for the next step: while drafted steps have recently produced more
    /// tokens per second than plain steps would (every 8th step drafts regardless, to keep that current).
    pub fn want_draft(&self) -> bool {
        self.drafts > 0 && (self.drafted_rate >= 1.0 / self.cost[1] || self.steps % 8 == 7)
    }

    /// Drafts to verify given the drafter's stated probability of each: the count with the most expected
    /// tokens per second (the drafter pass, already spent, counted in every option).
    pub fn drafts_conf(&self, conf: &[f32]) -> usize {
        let n = conf.len().min(self.drafts);
        let (mut best, mut best_rate) = (0, 1.0 / (self.cost[1] + self.draft));
        let (mut tokens, mut p) = (1.0, 1.0);
        for (d, &c) in conf[..n].iter().enumerate() {
            p *= self.calibrated(c);
            tokens += p;
            let rate = tokens / (self.cost[d + 2] + self.draft);
            if rate > best_rate {
                best = d + 1;
                best_rate = rate;
            }
        }
        best
    }

    /// Records a step after `drafts_conf`: the stated probabilities of the `drafted` verified drafts, of
    /// which the first `accepted` were taken, the verification's `seconds` and the drafter pass's.
    pub fn update_conf(&mut self, conf: &[f32], drafted: usize, accepted: usize, seconds: f64, draft_seconds: Option<f64>) {
        for (k, &c) in conf.iter().enumerate().take(drafted.min(accepted + 1)) {
            let b = &mut self.calib[((c.max(0.0) * 10.0) as usize).min(9)];
            // Recent drafts weigh more: decay the bucket's counts beyond 200 judged.
            if b.1 > 200.0 {
                b.0 *= 0.995;
                b.1 *= 0.995;
            }
            b.0 += (k < accepted) as u8 as f64;
            b.1 += 1.0;
        }
        if let Some(d) = draft_seconds {
            let rate = (accepted + 1) as f64 / (seconds + d);
            self.drafted_rate += 0.1 * (rate - self.drafted_rate);
        }
        self.update(drafted, accepted, seconds, draft_seconds);
    }

    /// Expected tokens per second verifying `d` drafts.
    fn rate(&self, d: usize) -> f64 {
        let (mut tokens, mut p) = (1.0, 1.0);
        for k in 0..d {
            p *= self.accept[k];
            tokens += p;
        }
        tokens / (self.cost[d + 1] + if d > 0 { self.draft } else { 0.0 })
    }

    /// Drafts to verify in the next step. Every 8th step that would verify fewer than all of them verifies
    /// one more, so the estimate of the next position (and of that width's cost) stays current without
    /// paying for a full-width step.
    pub fn drafts(&self) -> usize {
        let best = (0..=self.drafts)
            .max_by(|&a, &b| self.rate(a).total_cmp(&self.rate(b)))
            .unwrap_or(0);
        if best < self.drafts && self.steps % 8 == 7 { best + 1 } else { best }
    }

    /// Records a step that verified `drafted` drafts, of which the first `accepted` were taken, in
    /// `seconds` (the verification step, sampling included), after a drafter pass of `draft_seconds`.
    pub fn update(&mut self, drafted: usize, accepted: usize, seconds: f64, draft_seconds: Option<f64>) {
        const A: f64 = 0.1;
        for k in 0..drafted.min(accepted + 1) {
            let hit = if k < accepted { 1.0 } else { 0.0 };
            self.accept[k] += A * (hit - self.accept[k]);
        }
        let w = (drafted + 1).min(MAX_DRAFTS + 1);
        self.cost[w] += 0.2 * (seconds - self.cost[w]);
        if let Some(d) = draft_seconds {
            self.draft += 0.2 * (d - self.draft);
        }
        self.steps += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confident_drafts_are_verified_and_doubtful_ones_dropped() {
        let mut p = DraftPolicy::new(7, 0.0045, 0.0008, 0.0006);
        // Drafts stated at 0.95 are kept, at 0.3 never.
        for i in 0..300 {
            let conf = if i % 2 == 0 { [0.95, 0.95, 0.3, 0.3, 0.3, 0.3, 0.3] } else { [0.3; 7] };
            // Every draft verified (as a probe would), the 0.95 ones kept.
            let kept = if i % 2 == 0 { 2 } else { 0 };
            p.update_conf(&conf, 7, kept, 0.0045 + 0.0008 * 7.0, Some(0.0006));
        }
        assert_eq!(p.drafts_conf(&[0.95, 0.95, 0.3, 0.3, 0.3, 0.3, 0.3]), 2);
        assert_eq!(p.drafts_conf(&[0.3; 7]), 0);
    }

    #[test]
    fn verifies_fewer_drafts_when_later_ones_fail() {
        let mut p = DraftPolicy::new(5, 0.010, 0.0018, 0.0024);
        assert_eq!(p.drafts(), 5);
        // Only the first draft is ever accepted.
        for _ in 0..200 {
            let d = p.drafts();
            p.update(d, d.min(1), 0.010 + 0.0018 * d as f64, (d > 0).then_some(0.0024));
        }
        let d = p.drafts();
        assert!((1..=2).contains(&d) || p.steps % 8 == 7, "{d}");
    }

    #[test]
    fn no_drafts_when_none_are_accepted() {
        let mut p = DraftPolicy::new(5, 0.010, 0.0018, 0.0024);
        for _ in 0..200 {
            let d = p.drafts();
            p.update(d, 0, 0.010 + 0.0018 * d as f64, (d > 0).then_some(0.0024));
        }
        p.steps = 0;
        assert_eq!(p.drafts(), 0);
    }
}
