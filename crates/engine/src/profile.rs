//! Per-op GPU timing for eager steps: CUDA timing events around each named op, summed by name.

use std::collections::BTreeMap;

use anyhow::Result;
use quench6k_cuda::{Event, Stream};

#[derive(Default)]
pub struct Profiler {
    pending: Vec<(&'static str, Event, Event)>,
    totals: BTreeMap<&'static str, (usize, f64)>,
}

impl Profiler {
    /// Times `op` on `s` under `name`.
    pub fn time<T>(
        &mut self,
        name: &'static str,
        s: &Stream,
        op: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let (a, b) = (Event::timing()?, Event::timing()?);
        a.record(s)?;
        let out = op()?;
        b.record(s)?;
        self.pending.push((name, a, b));
        Ok(out)
    }

    /// Resolves recorded events (synchronizing on the last).
    pub fn collect(&mut self) -> Result<()> {
        for (name, a, b) in self.pending.drain(..) {
            b.synchronize()?;
            let e = self.totals.entry(name).or_default();
            e.0 += 1;
            e.1 += Event::elapsed_ms(&a, &b)? as f64;
        }
        Ok(())
    }

    /// Lines of `name calls total_ms share`, largest first.
    pub fn report(&self) -> String {
        let total: f64 = self.totals.values().map(|v| v.1).sum();
        let mut rows: Vec<_> = self.totals.iter().collect();
        rows.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap());
        let mut out = format!("{:<22} {:>7} {:>10} {:>7}\n", "op", "calls", "ms", "share");
        for (name, (calls, ms)) in rows {
            out += &format!(
                "{name:<22} {calls:>7} {ms:>10.3} {:>6.1}%\n",
                100.0 * ms / total
            );
        }
        out + &format!("{:<22} {:>7} {total:>10.3}\n", "total", "")
    }
}
