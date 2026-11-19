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
        }
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
    /// all, so the estimates of later positions stay current.
    pub fn drafts(&self) -> usize {
        let best = (0..=self.drafts)
            .max_by(|&a, &b| self.rate(a).total_cmp(&self.rate(b)))
            .unwrap_or(0);
        if best < self.drafts && self.steps % 8 == 7 { self.drafts } else { best }
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
