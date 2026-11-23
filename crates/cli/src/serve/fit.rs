//! The context a slot engine with growable states can serve: `--context`, or fewer positions when one
//! sequence alone cannot hold that many in the device memory left after the weights and the workspace.
//! The server advertises and enforces the result, so that clients sizing prompts from `/v1/models` do
//! not send prompts that are refused for lack of memory.

use anyhow::{Result, bail};

use super::slots::{SlotEngine, positions_in};

/// Rounding of a capped context when the engine's prefill chunk is unknown.
const ROUND: usize = 1024;

/// The context to serve from `avail` device bytes for one sequence (free memory with its state's
/// memory counted as free, as the scheduler's admission does), less `slack` (mapping granularity):
/// `context` when that fits, else the positions that fit ([`positions_in`], the scheduler's rule)
/// rounded down to whole prefill chunks of `chunk`.
pub(super) fn capped_context(avail: usize, slack: usize, chunk: usize, context: usize, plan: impl Fn(usize) -> usize) -> usize {
    let fit = positions_in(avail.saturating_sub(slack), context, plan);
    if fit >= context {
        return context;
    }
    let chunk = if chunk == 0 { ROUND } else { chunk };
    let rounded = fit / chunk * chunk;
    if rounded > 0 { rounded } else { fit }
}

/// [`capped_context`] measured on the device, right after the weights and the workspace are loaded:
/// with every slot's sampler and one sequence state allocated (and its first position mapped, which
/// shows the mapping granularity's cost), then freed again. Logs when the context is capped.
pub(super) fn fit_context<E: SlotEngine>(e: &mut E, slots: usize, context: usize) -> Result<usize> {
    if !e.growable() {
        // Fixed-size states hold `context` positions or fail to allocate.
        return Ok(context);
    }
    let samplers = (0..slots).map(|_| e.new_sampler()).collect::<Result<Vec<_>>>()?;
    let mut st = e.new_state(context)?;
    e.reserve(&mut st, 1)?;
    let held = e.device_bytes(&st);
    let slack = held.saturating_sub(e.plan_bytes(1));
    let free = e.free_bytes();
    e.sync()?;
    drop(st);
    drop(samplers);
    let avail = free + held;
    let capped = capped_context(avail, slack, e.chunk(), context, |p| e.plan_bytes(p));
    if capped == 0 {
        bail!(
            "no room for one sequence in device memory: {:.2} GiB free after the weights and workspace, {} bytes per position",
            free as f64 / (1u64 << 30) as f64,
            e.bytes_per_position()
        );
    }
    if capped < context {
        eprintln!(
            "{} worker: context capped to {capped} positions (--context {context}): one sequence fits about {} positions in the {:.2} GiB of device memory left after the weights and workspace ({:.0} MiB per sequence, {} bytes per position)",
            e.name(),
            positions_in(avail.saturating_sub(slack), context, |p| e.plan_bytes(p)),
            avail as f64 / (1u64 << 30) as f64,
            e.plan_bytes(0) as f64 / (1u64 << 20) as f64,
            e.bytes_per_position()
        );
    }
    Ok(capped)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: f64 = (1u64 << 30) as f64;

    fn gib(x: f64) -> usize {
        (x * GIB) as usize
    }

    /// One mapping grain (2 MiB) per growable buffer.
    fn grains(n: usize) -> usize {
        n * (2 << 20)
    }

    #[test]
    fn hy4_is_capped_to_whole_chunks() {
        // Hy4-flash with eight slots: 4.46 GiB left, 17,664 bytes per position, about 106 MB per sequence.
        let (per, fixed) = (17_664, 106 << 20);
        let plan = |p: usize| fixed + p * per;
        let ctx = capped_context(gib(4.46), grains(3), 2048, 262_144, plan);
        assert_eq!(ctx % 2048, 0);
        assert!(ctx < 262_144);
        // The capped context fits the scheduler's rule with the slack, a chunk more does not.
        let avail = gib(4.46) - grains(3);
        assert!(positions_in(avail, 262_144, plan) >= ctx);
        assert!(positions_in(avail, 262_144, plan) < ctx + 2048);
        // (4.46 GiB - 1 GiB headroom - 106 MiB - 6 MiB) / 17,664 = 203,673 -> 202,752 (99 chunks).
        assert_eq!(ctx, 202_752);
    }

    #[test]
    fn inkling_and_flint_fit_their_contexts() {
        // Inkling 2: 75.46 GiB, 36 KiB per position, 3.09 GiB fixed: 1M positions fit.
        let ink = |p: usize| gib(3.09) + p * 36 * 1024;
        assert_eq!(capped_context(gib(75.46), grains(64), 2048, 1 << 20, ink), 1 << 20);
        // Flint: 70.98 GiB, 28,672 bytes per position (rings up to 560 MB): 1M positions fit.
        let rings = 560 << 20;
        let flint = |p: usize| p * 28_672 + rings * p.min(4096) / 4096;
        assert_eq!(capped_context(gib(70.98), grains(32), 4096, 1 << 20, flint), 1 << 20);
        // A smaller --context stays as it is.
        assert_eq!(capped_context(gib(4.46), grains(3), 2048, 65_536, |p| p * 17_664), 65_536);
    }

    #[test]
    fn edge_cases() {
        // Nothing fits: 0 (the server refuses to start).
        assert_eq!(capped_context(gib(1.0), 0, 2048, 4096, |p| 1 + p), 0);
        // Fewer positions than one chunk: not rounded to nothing.
        assert_eq!(capped_context((1 << 30) + 1000 * 100, 0, 2048, 4096, |p| p * 100), 1000);
        // Unknown chunk: rounded to 1024.
        assert_eq!(capped_context((1 << 30) + 5000 * 100, 0, 0, 8192, |p| p * 100), 4096);
        // Exactly fitting --context is kept.
        assert_eq!(capped_context((1 << 30) + 4096 * 100, 0, 2048, 4096, |p| p * 100), 4096);
    }
}
