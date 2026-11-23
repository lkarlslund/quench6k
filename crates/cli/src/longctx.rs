//! `quench6k longctx-check`: the engines at the far end of their context without prefilling it. Sequences
//! of the server's capacity are placed at a high position (their full-length caches mapped up to it and
//! zero-filled, the token history made up), then decode steps run there: alone and batched (every row's
//! logits compared bitwise), one wide step against the same tokens one at a time, the drafter alone and
//! batched; logits must be finite. Layouts of n sequences at capacity / n each (1 x 1M, 2 x 512K, ...)
//! show what the device holds at once; then one sequence's step time against its position.
//!
//! The caches hold zeros, not real keys: this checks addressing (no fault, no 32-bit offsets, kernels
//! read the mapped rows) and batch invariance at those positions, not the model's quality there.

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};
use quench6k_cuda::{DeviceBuffer, Stream};

const GIB: f64 = (1u64 << 30) as f64;

fn gib(b: usize) -> f64 {
    b as f64 / GIB
}

fn free() -> usize {
    quench6k_cuda::mem_info().map_or(0, |(f, _)| f)
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// One engine under test.
trait Probe {
    type St;
    fn name(&self) -> &'static str;
    fn s(&self) -> &Stream;
    fn max_context(&self) -> usize;
    fn vocab(&self) -> usize;
    fn max_seqs(&self) -> usize;
    fn new_state(&mut self, capacity: usize) -> Result<Self::St>;
    /// Makes the state hold `pos` positions (made-up tokens; caches as mapped).
    fn place(&self, st: &mut Self::St, pos: usize);
    fn reserve(&self, st: &mut Self::St, positions: usize) -> Result<()>;
    fn truncate(&self, st: &mut Self::St, pos: usize);
    fn bytes(&self, st: &Self::St) -> usize;
    /// A batched step; the logits rows (sequence by sequence) at the returned address.
    fn step(&mut self, states: &mut [&mut Self::St], tokens: &[&[i32]]) -> Result<u64>;
    /// Drafts after a step (None: no drafter).
    fn draft(&mut self, states: &mut [&mut Self::St], pending: &[i32], rows: &[usize]) -> Result<Option<Vec<Vec<i32>>>>;
}

/// Zero-fills free device memory (all but `keep`) and gives it back, so that caches mapped next read
/// zeros rather than whatever the memory held (the driver hands back the same pages in practice; finite
/// logits are checked anyway).
fn scrub(s: &Stream, keep: usize) -> Result<()> {
    let n = free().saturating_sub(keep);
    if n > 0 {
        let b = DeviceBuffer::new(n)?;
        s.memset(b.addr(), 0, n)?;
        s.synchronize()?;
    }
    Ok(())
}

/// Hashes of `rows` logits rows at `src`, and whether all are finite.
fn rows_of<P: Probe>(p: &P, src: u64, rows: usize) -> Result<(Vec<u64>, bool)> {
    let v = p.vocab();
    let x: Vec<f32> = p.s().download(src, rows * v)?;
    let finite = x.iter().all(|f| f.is_finite());
    let bytes: Vec<u8> = x.iter().flat_map(|f| f.to_le_bytes()).collect();
    Ok((bytes.chunks(v * 4).map(fnv).collect(), finite))
}

fn tok(i: usize, j: usize) -> i32 {
    (1000 + 7 * i + 13 * j) as i32
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

/// n sequences of capacity `cap` at position `at` each: alone vs batched, wide vs narrow, drafts.
/// Returns false on a mismatch or non-finite logits.
#[allow(clippy::needless_range_loop)]
fn layout<P: Probe>(p: &mut P, cap: usize, n: usize, at: usize, steps: usize) -> Result<bool> {
    let f0 = free();
    let mut states = Vec::new();
    for i in 0..n {
        let r = (|| -> Result<P::St> {
            let mut st = p.new_state(cap)?;
            p.reserve(&mut st, at + steps + 64)?;
            Ok(st)
        })();
        match r {
            Ok(mut st) => {
                p.place(&mut st, at);
                states.push(st);
            }
            Err(e) => {
                println!(
                    "{} {n} x {at}: sequence {i} does not fit ({e:#}); {:.2} GiB free before, {:.2} GiB each",
                    p.name(),
                    gib(f0),
                    states.first().map_or(0.0, |st| gib(p.bytes(st)))
                );
                return Ok(true);
            }
        }
    }
    p.s().synchronize()?;
    let held = states.first().map_or(0, |st| p.bytes(st));
    let mut ok = true;
    // Alone: `steps` one-token steps each, logits hashes per step; then back to `at`.
    let mut alone: Vec<Vec<u64>> = Vec::new();
    let mut alone_ms = Vec::new();
    for (i, st) in states.iter_mut().enumerate() {
        let mut h = Vec::new();
        for j in 0..steps {
            let t = Instant::now();
            let src = p.step(&mut [&mut *st], &[&[tok(i, j)]])?;
            p.s().synchronize()?;
            alone_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let (r, finite) = rows_of(p, src, 1)?;
            if !finite {
                println!("  sequence {i} step {j}: non-finite logits");
                ok = false;
            }
            h.push(r[0]);
        }
        p.truncate(st, at);
        alone.push(h);
    }
    // Batched.
    let mut batch_ms = Vec::new();
    let mut bad = 0;
    for j in 0..steps {
        let toks: Vec<[i32; 1]> = (0..n).map(|i| [tok(i, j)]).collect();
        let refs: Vec<&[i32]> = toks.iter().map(|t| t.as_slice()).collect();
        let mut sts: Vec<&mut P::St> = states.iter_mut().collect();
        let t = Instant::now();
        let src = p.step(&mut sts, &refs)?;
        p.s().synchronize()?;
        batch_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let (r, _) = rows_of(p, src, n)?;
        for i in 0..n {
            if r[i] != alone[i][j] {
                bad += 1;
                println!("  step {j} sequence {i}: batched logits differ from alone");
            }
        }
    }
    // The drafter after the last batched step, batched, then alone (the same step's features).
    let pending: Vec<i32> = (0..n).map(|i| tok(i, steps)).collect();
    let rows = vec![0usize; n];
    let mut sts: Vec<&mut P::St> = states.iter_mut().collect();
    let drafted = p.draft(&mut sts, &pending, &rows)?;
    drop(sts);
    let mut draft_note = String::from("no drafter");
    if let Some(batched) = drafted {
        let mut same = 0;
        for (i, st) in states.iter_mut().enumerate() {
            let a = p.draft(&mut [&mut *st], &pending[i..=i], &rows[i..=i])?.unwrap_or_default();
            if a.first() == batched.get(i) {
                same += 1;
            } else {
                bad += 1;
                println!("  sequence {i}: drafts {:?} batched, {:?} alone", batched.get(i), a.first());
            }
        }
        let lens: Vec<usize> = batched.iter().map(Vec::len).collect();
        draft_note = format!("drafts {same}/{n} the same (lengths {lens:?})");
    }
    for st in states.iter_mut() {
        p.truncate(st, at);
    }
    // One wide step of all `steps` tokens against the one-token steps.
    let wide = steps.min(8);
    for (i, st) in states.iter_mut().enumerate() {
        let toks: Vec<i32> = (0..wide).map(|j| tok(i, j)).collect();
        let src = p.step(&mut [&mut *st], &[&toks])?;
        let (r, _) = rows_of(p, src, wide)?;
        if r[..] != alone[i][..wide] {
            bad += 1;
            println!("  sequence {i}: a {wide}-token step differs from {wide} one-token steps");
        }
        p.truncate(st, at);
    }
    ok &= bad == 0;
    println!(
        "{} {n} x cap {cap} at {at}: {} ({} rows alone == batched, {wide}-wide == narrow, {draft_note}); {:.2} GiB per sequence, {:.2} GiB free left; step {:.2} ms alone, {:.2} ms batched (median)",
        p.name(),
        if ok { "OK" } else { "MISMATCH" },
        n * steps,
        gib(held),
        gib(free()),
        median(alone_ms),
        median(batch_ms)
    );
    Ok(ok)
}

/// One sequence's one-token step time at each position.
fn timing<P: Probe>(p: &mut P, cap: usize, positions: &[usize], steps: usize) -> Result<()> {
    let mut st = p.new_state(cap)?;
    for &at in positions {
        if at + steps + 64 > cap {
            continue;
        }
        p.reserve(&mut st, at + steps + 64)?;
        p.place(&mut st, at);
        let mut ms = Vec::new();
        for j in 0..=steps {
            let t = Instant::now();
            let src = p.step(&mut [&mut st], &[&[tok(0, j)]])?;
            p.s().synchronize()?;
            if j > 0 {
                ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let (_, finite) = rows_of(p, src, 1)?;
            if !finite {
                println!("  position {at}: non-finite logits");
            }
        }
        p.truncate(&mut st, at);
        let lo = ms.iter().copied().fold(f64::MAX, f64::min);
        let hi = ms.iter().copied().fold(0.0, f64::max);
        println!(
            "{} one sequence at {at:>8}: step {:.2} ms median ({lo:.2}..{hi:.2}), {:.2} GiB mapped",
            p.name(),
            median(ms),
            gib(p.bytes(&st))
        );
    }
    Ok(())
}

fn check<P: Probe>(p: &mut P, cap: Option<usize>, seqs: &[usize], steps: usize, positions: &[usize]) -> Result<()> {
    let cap = cap.unwrap_or(p.max_context()).min(p.max_context());
    let (f, t) = quench6k_cuda::mem_info()?;
    println!("{}: capacity {cap}; after weights and workspace {:.2} GiB free of {:.2} GiB", p.name(), gib(f), gib(t));
    scrub(p.s(), 1 << 30)?;
    let mut ok = true;
    for &n in seqs {
        if n == 0 || n > p.max_seqs() {
            continue;
        }
        let at = cap / n - steps - 64;
        ok &= layout(p, cap, n, at, steps)?;
    }
    let positions: Vec<usize> = if positions.is_empty() {
        vec![4096, cap / 8, cap / 4, cap / 2, cap - steps - 128]
    } else {
        positions.to_vec()
    };
    timing(p, cap, &positions, steps)?;
    if !ok {
        bail!("{}: mismatches or non-finite logits", p.name());
    }
    Ok(())
}

mod ink {
    use super::*;
    use quench6k_engine::inkling::InklingModel;
    use quench6k_engine::inkling::forward::{BATCH_ROWS, CANON_CHUNK, MAX_SEQS, State, Workspace};

    pub struct P<'m> {
        pub m: &'m InklingModel,
        pub s: &'m Stream,
        pub ws: Workspace,
    }

    impl Probe for P<'_> {
        type St = State;
        fn name(&self) -> &'static str {
            "inkling"
        }
        fn s(&self) -> &Stream {
            self.s
        }
        fn max_context(&self) -> usize {
            self.m.cfg.max_context
        }
        fn vocab(&self) -> usize {
            self.m.cfg.vocab
        }
        fn max_seqs(&self) -> usize {
            MAX_SEQS
        }
        fn new_state(&mut self, capacity: usize) -> Result<State> {
            State::new(self.m, capacity, self.s)
        }
        fn place(&self, st: &mut State, pos: usize) {
            st.history = vec![1000; pos];
            st.pos = pos;
        }
        fn reserve(&self, st: &mut State, positions: usize) -> Result<()> {
            st.reserve(positions)
        }
        fn truncate(&self, st: &mut State, pos: usize) {
            st.truncate(pos);
        }
        fn bytes(&self, st: &State) -> usize {
            st.device_bytes()
        }
        fn step(&mut self, states: &mut [&mut State], tokens: &[&[i32]]) -> Result<u64> {
            self.m.forward_batch(states, &mut self.ws, self.s, tokens)?;
            Ok(self.ws.out_logits)
        }
        fn draft(&mut self, states: &mut [&mut State], pending: &[i32], rows: &[usize]) -> Result<Option<Vec<Vec<i32>>>> {
            if self.m.drafter.is_none() {
                return Ok(None);
            }
            Ok(Some(self.m.draft_batch(states, &mut self.ws, self.s, pending, rows)?))
        }
    }

    pub fn run(dir: &Path, cap: Option<usize>, seqs: &[usize], steps: usize, positions: &[usize]) -> Result<()> {
        let (m, s) = crate::inkling::load(dir)?;
        let ws = Workspace::new(&m, CANON_CHUNK, BATCH_ROWS)?;
        check(&mut P { m: &m, s: &s, ws }, cap, seqs, steps, positions)
    }
}

mod hy4 {
    use super::*;
    use quench6k_engine::hy4::Hy4Model;
    use quench6k_engine::hy4::forward::{CANON_CHUNK, MAX_SEQS, MAX_STEP, State, Workspace};

    pub struct P<'m> {
        pub m: &'m Hy4Model,
        pub s: &'m Stream,
        pub ws: Workspace,
    }

    impl Probe for P<'_> {
        type St = State;
        fn name(&self) -> &'static str {
            "hy4"
        }
        fn s(&self) -> &Stream {
            self.s
        }
        fn max_context(&self) -> usize {
            self.m.cfg.max_positions
        }
        fn vocab(&self) -> usize {
            self.m.cfg.vocab
        }
        fn max_seqs(&self) -> usize {
            MAX_SEQS
        }
        fn new_state(&mut self, capacity: usize) -> Result<State> {
            State::new(self.m, capacity, self.s)
        }
        fn place(&self, st: &mut State, pos: usize) {
            st.history = vec![1000; pos];
            st.pos = pos;
        }
        fn reserve(&self, st: &mut State, positions: usize) -> Result<()> {
            st.reserve(positions)
        }
        fn truncate(&self, st: &mut State, pos: usize) {
            st.truncate(pos);
        }
        fn bytes(&self, st: &State) -> usize {
            st.device_bytes()
        }
        fn step(&mut self, states: &mut [&mut State], tokens: &[&[i32]]) -> Result<u64> {
            self.m.forward_batch(states, &mut self.ws, self.s, tokens)?;
            Ok(self.ws.out_logits)
        }
        fn draft(&mut self, states: &mut [&mut State], pending: &[i32], rows: &[usize]) -> Result<Option<Vec<Vec<i32>>>> {
            if self.m.dfly.is_none() {
                return Ok(None);
            }
            Ok(Some(self.m.draft_batch(states, &mut self.ws, self.s, pending, rows)?))
        }
    }

    pub fn run(dir: &Path, cap: Option<usize>, seqs: &[usize], steps: usize, positions: &[usize]) -> Result<()> {
        let (m, s) = crate::hy4::load(dir)?;
        let ws = Workspace::new(&m, CANON_CHUNK, 0, MAX_SEQS * MAX_STEP)?;
        check(&mut P { m: &m, s: &s, ws }, cap, seqs, steps, positions)
    }
}

mod flint {
    use super::*;
    use quench6k_engine::flint as fl;

    pub struct P<'m> {
        pub m: &'m fl::FlintModel,
        pub s: &'m Stream,
        pub ws: fl::Workspace,
    }

    impl Probe for P<'_> {
        type St = fl::FlintState;
        fn name(&self) -> &'static str {
            "flint"
        }
        fn s(&self) -> &Stream {
            self.s
        }
        fn max_context(&self) -> usize {
            self.m.max_context
        }
        fn vocab(&self) -> usize {
            fl::VOCAB
        }
        fn max_seqs(&self) -> usize {
            fl::MAX_SEQS
        }
        fn new_state(&mut self, capacity: usize) -> Result<fl::FlintState> {
            fl::FlintState::new(self.m, capacity, self.s)
        }
        fn place(&self, st: &mut fl::FlintState, pos: usize) {
            st.tokens = vec![1000; pos];
        }
        fn reserve(&self, st: &mut fl::FlintState, positions: usize) -> Result<()> {
            st.reserve(positions)
        }
        fn truncate(&self, st: &mut fl::FlintState, pos: usize) {
            st.truncate(pos);
        }
        fn bytes(&self, st: &fl::FlintState) -> usize {
            st.device_bytes()
        }
        fn step(&mut self, states: &mut [&mut fl::FlintState], tokens: &[&[i32]]) -> Result<u64> {
            self.m.forward_batch(states, &mut self.ws, self.s, tokens)?;
            Ok(self.ws.logits)
        }
        fn draft(&mut self, _: &mut [&mut fl::FlintState], _: &[i32], _: &[usize]) -> Result<Option<Vec<Vec<i32>>>> {
            Ok(None)
        }
    }

    pub fn run(dir: &Path, cap: Option<usize>, seqs: &[usize], steps: usize, positions: &[usize]) -> Result<()> {
        let (m, s) = crate::flint::load(dir)?;
        let ws = fl::Workspace::new(fl::MAX_ROWS)?;
        check(&mut P { m: &m, s: &s, ws }, cap, seqs, steps, positions)
    }
}

pub fn run(dir: &Path, cap: Option<usize>, seqs: &[usize], steps: usize, positions: &[usize]) -> Result<()> {
    if quench6k_engine::inkling::InklingModel::is_inkling(dir) {
        ink::run(dir, cap, seqs, steps, positions)
    } else if quench6k_engine::hy4::Hy4Model::is_hy4(dir) {
        hy4::run(dir, cap, seqs, steps, positions)
    } else if quench6k_engine::flint::FlintModel::is_flint(dir) {
        flint::run(dir, cap, seqs, steps, positions)
    } else {
        bail!("{}: not an Inkling 2, Hy4-flash or Flint checkpoint", dir.display())
    }
}
