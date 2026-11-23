//! The slot scheduler over a CPU mock engine: a "model" whose next token is a hash of the sequence so
//! far, states that count device bytes against a budget, a drafter that guesses two tokens right and the
//! third wrong, and failures on request.

use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc as tmpsc;

use super::*;
use crate::serve::{Metrics, QueuedJob};

/// The mock model: the token after a sequence is a function of a running hash of it (states keep the
/// hash of every prefix, so long sequences cost nothing per step).
const H0: u64 = 0xcbf2_9ce4_8422_2325;

fn fold(x: u64, t: i32) -> u64 {
    (x ^ t as u64).wrapping_mul(0x100_0000_01b3)
}

fn tok_of(x: u64) -> i32 {
    (x % 40) as i32 + 1
}

/// Greedy output for `prompt`: `n` tokens, or up to the first stop token.
fn expected(prompt: &[i32], n: usize, stop: &[i32]) -> Vec<i32> {
    let mut x = prompt.iter().fold(H0, |x, &t| fold(x, t));
    let mut out = Vec::new();
    while out.len() < n {
        let t = tok_of(x);
        x = fold(x, t);
        out.push(t);
        if stop.contains(&t) {
            break;
        }
    }
    out
}

const GIB: usize = 1 << 30;

/// A prompt token whose sequence fails on its third decode step.
const FAIL: i32 = 999;

struct MState {
    id: usize,
    hist: Vec<i32>,
    /// pre[i]: the hash of hist[..i].
    pre: Vec<u64>,
    cap: usize,
    /// Device bytes held (fixed part, mapped positions, snapshot).
    bytes: usize,
    /// Positions mapped (growable states).
    mapped: usize,
    /// Whether the snapshot buffer is allocated.
    snap: bool,
    last_t: usize,
    steps: usize,
    used: Arc<AtomicUsize>,
    log: Arc<Mutex<Vec<String>>>,
}

impl MState {
    fn truncate(&mut self, pos: usize) {
        self.hist.truncate(pos);
        self.pre.truncate(self.hist.len() + 1);
    }
}

impl Drop for MState {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::Relaxed);
        self.log.lock().unwrap().push(format!("drop {}", self.id));
    }
}

struct MSampler {
    rows: [i32; 8],
}

impl SlotSampler for MSampler {
    type Src = [i32; 8];
    fn fetch(&mut self, src: [i32; 8], _rows: usize, _p: &GenParams) -> Result<()> {
        self.rows = src;
        Ok(())
    }
    fn pick(&mut self, row: usize, _p: &GenParams, _generated: &[i32]) -> i32 {
        self.rows[row]
    }
}

#[derive(Clone)]
struct Opts {
    /// Device bytes.
    total: usize,
    /// Free bytes the engine reports beyond what is free (the default: the scheduler's headroom, so
    /// that it can use all of `total`; 0 for a real device's view).
    bonus: usize,
    /// Device bytes per position, of a new state (growable: before any position), and of a snapshot
    /// allocated when a sequence first reaches `snap_at` positions (0: none).
    per: usize,
    fixed: usize,
    snap: usize,
    snap_at: usize,
    /// Positions mapped at once (growable states).
    grain: usize,
    growable: bool,
    drafter: Drafter,
    chunk: usize,
    /// Sleep per prefill chunk and per sequence step.
    prefill_ms: u64,
    step_ms: u64,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            total: 1 << 20,
            bonus: HEADROOM,
            per: 1,
            fixed: 0,
            snap: 0,
            snap_at: 0,
            grain: 1,
            growable: false,
            drafter: Drafter::None,
            chunk: 4,
            prefill_ms: 0,
            step_ms: 0,
        }
    }
}

impl Opts {
    /// Hy4-flash as measured (longctx-check): 4.46 GiB free after the weights and workspace, 17,664
    /// bytes per position, 0.15 GiB per state (rings, drafter buffers, snapshots), mapping 1024
    /// positions ahead.
    fn hy4() -> Self {
        Self {
            total: (4.46 * GIB as f64) as usize,
            bonus: 0,
            per: 17_664,
            fixed: (0.15 * GIB as f64) as usize,
            grain: 1024,
            growable: true,
            chunk: 2048,
            ..Default::default()
        }
    }

    /// Flint as measured (longctx-check): 70.98 GiB free after the weights and workspace, 28,672 bytes
    /// per position, its rings (0.55 GiB once grown) counted as a state's fixed part.
    fn flint() -> Self {
        Self {
            total: (70.98 * GIB as f64) as usize,
            bonus: 0,
            per: 28_672,
            fixed: (0.55 * GIB as f64) as usize,
            grain: 512,
            growable: true,
            chunk: 4096,
            ..Default::default()
        }
    }

    /// Inkling 2 Mini as measured on the device (longctx-check): free memory after the weights and
    /// workspace, 36 KiB per position, 3.09 GiB of rings per sequence and a 0.17 GiB snapshot at the
    /// first canonical boundary (which the plans do not count), mapping 2 MiB per cache (1024 positions).
    fn inkling() -> Self {
        Self {
            total: (75.46 * GIB as f64) as usize,
            bonus: 0,
            per: 36 * 1024,
            fixed: (3.09 * GIB as f64) as usize,
            snap: (0.17 * GIB as f64) as usize,
            snap_at: 2048,
            grain: 1024,
            growable: true,
            chunk: 2048,
            ..Default::default()
        }
    }
}

struct Mock {
    o: Opts,
    /// Sleep per sequence step (ms), changeable while the scheduler runs.
    pace: Arc<AtomicU64>,
    used: Arc<AtomicUsize>,
    /// Events: "prefill <state>", "step <states>", "new <state>".
    log: Arc<Mutex<Vec<String>>>,
    ids: usize,
    capacity: usize,
    /// The state of the last forward (a draft must follow its own sequence's step).
    last: usize,
}

impl Mock {
    /// Logits of the last (up to 8) fed tokens: the token after each.
    fn rows(st: &MState, toks: &[i32]) -> [i32; 8] {
        let n = toks.len().min(8);
        let mut x = *st.pre.last().unwrap();
        let mut rows = [0; 8];
        for (i, &t) in toks.iter().enumerate() {
            x = fold(x, t);
            if i + n >= toks.len() {
                rows[i + n - toks.len()] = tok_of(x);
            }
        }
        rows
    }

    /// Takes `bytes` of device memory.
    fn take(&self, bytes: usize, what: &str) -> Result<()> {
        if self.used.load(Ordering::Relaxed) + bytes > self.o.total {
            anyhow::bail!("mock {what}: out of memory");
        }
        self.used.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }

    fn feed(&mut self, st: &mut MState, toks: &[i32]) -> Result<[i32; 8]> {
        anyhow::ensure!(st.hist.len() + toks.len() <= st.cap, "context full");
        anyhow::ensure!(!self.o.growable || st.hist.len() + toks.len() <= st.mapped, "positions not reserved");
        if self.o.snap > 0 && !st.snap && st.hist.len() + toks.len() >= self.o.snap_at {
            self.take(self.o.snap, "snapshot")?;
            st.snap = true;
            st.bytes += self.o.snap;
        }
        let rows = Self::rows(st, toks);
        for &t in toks {
            let x = fold(*st.pre.last().unwrap(), t);
            st.pre.push(x);
        }
        st.hist.extend_from_slice(toks);
        st.last_t = toks.len();
        self.last = st.id;
        Ok(rows)
    }
}

impl SlotEngine for Mock {
    type State = MState;
    type Sampler = MSampler;

    fn name(&self) -> &'static str {
        "mock"
    }
    fn chunk(&self) -> usize {
        self.o.chunk
    }
    fn max_step(&self) -> usize {
        8
    }
    fn eos(&self) -> &[i32] {
        &[]
    }
    fn drafting(&self) -> Drafting {
        Drafting { kind: self.o.drafter, drafts: 3, costs: (0.001, 0.0001, 0.0001), lookup: true }
    }
    fn new_sampler(&self) -> Result<MSampler> {
        Ok(MSampler { rows: [0; 8] })
    }
    fn new_state(&mut self, capacity: usize) -> Result<MState> {
        let bytes = self.o.fixed + if self.o.growable { 0 } else { capacity * self.o.per };
        self.take(bytes, "alloc")?;
        self.ids += 1;
        self.log.lock().unwrap().push(format!("new {}", self.ids));
        Ok(MState {
            id: self.ids,
            hist: Vec::new(),
            pre: vec![H0],
            cap: capacity,
            bytes,
            mapped: 0,
            snap: false,
            last_t: 0,
            steps: 0,
            used: self.used.clone(),
            log: self.log.clone(),
        })
    }
    fn growable(&self) -> bool {
        self.o.growable
    }
    fn batched(&self) -> bool {
        true
    }
    fn bytes_per_position(&self) -> usize {
        self.o.per
    }
    fn reserve(&mut self, st: &mut MState, positions: usize) -> Result<()> {
        if !self.o.growable {
            return Ok(());
        }
        let want = positions.next_multiple_of(self.o.grain).min(st.cap);
        if want > st.mapped {
            let add = (want - st.mapped) * self.o.per;
            self.take(add, "map")?;
            st.bytes += add;
            st.mapped = want;
        }
        Ok(())
    }
    fn trim(&mut self, st: &mut MState, positions: usize) {
        let keep = positions.next_multiple_of(self.o.grain);
        if self.o.growable && keep < st.mapped {
            let sub = (st.mapped - keep) * self.o.per;
            self.used.fetch_sub(sub, Ordering::Relaxed);
            st.bytes -= sub;
            st.mapped = keep;
        }
        st.truncate(positions);
    }
    fn device_bytes(&self, st: &MState) -> usize {
        st.bytes
    }
    fn plan_bytes(&self, positions: usize) -> usize {
        self.o.fixed + if self.o.growable { positions } else { self.capacity } * self.o.per
    }
    fn free_bytes(&self) -> usize {
        self.o.total - self.used.load(Ordering::Relaxed) + self.o.bonus
    }
    fn history<'a>(&self, st: &'a MState) -> &'a [i32] {
        &st.hist
    }
    fn pos(&self, st: &MState) -> usize {
        st.hist.len()
    }
    fn last_t(&self, st: &MState) -> usize {
        st.last_t
    }
    fn rewind(&mut self, st: &mut MState, prompt: &[i32]) -> Result<usize> {
        let common = st.hist.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let keep = common.min(prompt.len() - 1);
        st.truncate(keep);
        Ok(keep)
    }
    fn snapshot(&mut self, _st: &mut MState) -> Result<()> {
        Ok(())
    }
    fn reset(&mut self, st: &mut MState) {
        st.truncate(0);
    }
    fn sync(&self) -> Result<()> {
        Ok(())
    }
    fn prefill(&mut self, st: &mut MState, tokens: &[i32], _last: bool) -> Result<[i32; 8]> {
        self.log.lock().unwrap().push(format!("prefill {}", st.id));
        self.log.lock().unwrap().push(format!("chunk {} {}..{}", st.id, st.hist.len(), st.hist.len() + tokens.len()));
        std::thread::sleep(Duration::from_millis(self.o.prefill_ms));
        let rows = self.feed(st, tokens)?;
        // The last token's logits in row 0.
        Ok([rows[tokens.len().min(8) - 1]; 8])
    }
    fn draft(&mut self, st: &mut MState, token: i32, row: usize) -> Result<Draft> {
        anyhow::ensure!(self.last == st.id, "draft after another sequence's step");
        anyhow::ensure!(row < st.last_t, "draft row {row} of a {}-token step", st.last_t);
        let mut x = fold(*st.pre.last().unwrap(), token);
        let mut tokens = Vec::new();
        for _ in 0..2 {
            let t = tok_of(x);
            x = fold(x, t);
            tokens.push(t);
        }
        tokens.push(tok_of(x) % 40 + 1);
        Ok(Draft { tokens, conf: vec![0.95, 0.95, 0.2], secs: 0.0001 })
    }
    fn step(&mut self, states: &mut [&mut MState], tokens: &[&[i32]], after: &mut After<'_, [i32; 8]>) -> Vec<Result<Option<Draft>>> {
        let ids: Vec<String> = states.iter().map(|s| s.id.to_string()).collect();
        self.log.lock().unwrap().push(format!("step {}", ids.join(",")));
        step_each(
            self,
            states,
            tokens,
            after,
            |e, st, toks| {
                let ms = e.pace.load(Ordering::Relaxed);
                if ms > 0 {
                    std::thread::sleep(Duration::from_millis(ms));
                }
                st.steps += 1;
                anyhow::ensure!(!(st.hist.contains(&FAIL) && st.steps >= 3), "injected failure");
                e.feed(st, toks)
            },
            |st, pos| st.truncate(pos),
        )
    }
}

fn params(max_tokens: usize, stop: Vec<i32>) -> GenParams {
    GenParams {
        max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        seed: 0,
        stop_tokens: stop,
        steering_strength: 0.0,
        adapter_scale: 0.0,
        boundaries: Vec::new(),
        capture: false,
        images: Vec::new(),
        progress: None,
        edits: None,
        record: None,
        fresh: false,
    }
}

struct Server {
    jobs: mpsc::Sender<Work>,
    log: Arc<Mutex<Vec<String>>>,
    pace: Arc<AtomicU64>,
    used: Arc<AtomicUsize>,
    worker: std::thread::JoinHandle<bool>,
}

/// The scheduler over a mock on a thread; joins true when it ended with Exit::Unload.
fn server(slots: usize, context: usize, o: Opts) -> Server {
    server_with(slots, context, o, 1.0)
}

fn server_with(slots: usize, context: usize, o: Opts, prefill_share: f64) -> Server {
    server_cfg(Config { slots, context, plan_tokens: 16384, prefill_share, unload_idle: None }, o)
}

fn server_cfg(cfg: Config, o: Opts) -> Server {
    let (jobs, rx) = mpsc::channel();
    let log = Arc::new(Mutex::new(Vec::new()));
    let pace = Arc::new(AtomicU64::new(o.step_ms));
    let used = Arc::new(AtomicUsize::new(0));
    let context = cfg.context;
    let mock = Mock { o, pace: pace.clone(), used: used.clone(), log: log.clone(), ids: 0, capacity: context, last: 0 };
    let worker = std::thread::spawn(move || {
        let metrics = Metrics::new(cfg.slots);
        matches!(schedule(mock, &cfg, &rx, &metrics, Vec::new()), Exit::Unload(_))
    });
    Server { jobs, log, pace, used, worker }
}

type Reply = tmpsc::UnboundedReceiver<Msg>;

impl Server {
    fn send(&self, ids: Vec<i32>, max_tokens: usize, mtp: bool) -> Reply {
        self.send_p(ids, params(max_tokens, Vec::new()), mtp)
    }
    fn send_p(&self, ids: Vec<i32>, params: GenParams, mtp: bool) -> Reply {
        let (tx, rx) = tmpsc::unbounded_channel();
        let job = Job { ids, params, mtp, tx, capture: None };
        self.jobs.send(Work::Generate(QueuedJob { job, cache_key: Vec::new() })).unwrap();
        rx
    }
    fn log(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }
    fn stop(self) -> bool {
        drop(self.jobs);
        self.worker.join().unwrap()
    }
}

/// Tokens and the end: Ok((finish, reused)) or the error.
fn collect(mut rx: Reply) -> (Vec<i32>, std::result::Result<(Finish, usize), String>) {
    let mut toks = Vec::new();
    loop {
        match rx.blocking_recv() {
            Some(Msg::Token(t)) => toks.push(t),
            Some(Msg::Progress(..)) => {}
            Some(Msg::Done(f, reused)) => return (toks, Ok((f, reused))),
            Some(Msg::Error(e)) => return (toks, Err(e)),
            None => return (toks, Err("channel closed".into())),
        }
    }
}

fn prompt(seed: i32, len: usize) -> Vec<i32> {
    (0..len as i32).map(|i| 100 + (seed * 37 + i * 11) % 200).collect()
}

/// Concurrent requests give the same tokens as alone, with and without drafts of every kind, and
/// really share steps.
#[test]
fn concurrent_requests_match_alone() {
    for drafter in [Drafter::None, Drafter::Count, Drafter::Conf] {
        for mtp in [false, true] {
            let srv = server(4, 512, Opts { drafter, ..Default::default() });
            let prompts: Vec<Vec<i32>> = (0..4).map(|i| prompt(i, 5 + 7 * i as usize)).collect();
            let rxs: Vec<Reply> = prompts.iter().enumerate().map(|(i, p)| srv.send(p.clone(), 40 + 10 * i, mtp)).collect();
            for (i, rx) in rxs.into_iter().enumerate() {
                let (toks, end) = collect(rx);
                assert_eq!(end, Ok((Finish::Length, 0)), "{drafter:?} mtp {mtp} request {i}");
                assert_eq!(toks, expected(&prompts[i], 40 + 10 * i, &[]), "{drafter:?} mtp {mtp} request {i}");
            }
            let log = srv.log();
            assert!(log.iter().any(|l| l.starts_with("step ") && l.contains(',')), "no shared step: {log:?}");
            srv.stop();
        }
    }
}

/// A request goes to the free slot holding the longest prefix of its prompt, else to the least
/// recently used one.
#[test]
fn prefix_slot_and_lru_choice() {
    let srv = server(2, 512, Opts::default());
    let a = prompt(1, 9);
    let b = prompt(2, 11);
    let (ga, _) = collect(srv.send(a.clone(), 6, false));
    let (gb, _) = collect(srv.send(b.clone(), 6, false));
    // Unrelated: the least recently used slot (a's).
    let (_, end) = collect(srv.send(prompt(3, 5), 4, false));
    assert_eq!(end, Ok((Finish::Length, 0)));
    // b's conversation goes on: b's slot, its prompt and output but the last token (never fed) reused.
    let mut b2 = [b.clone(), gb.clone(), vec![7, 8]].concat();
    let (g, end) = collect(srv.send(b2.clone(), 5, false));
    assert_eq!(end, Ok((Finish::Length, b.len() + gb.len() - 1)));
    assert_eq!(g, expected(&b2, 5, &[]));
    // a's slot was overwritten.
    let a2 = [a.clone(), ga.clone(), vec![3]].concat();
    let (g, end) = collect(srv.send(a2.clone(), 5, false));
    assert_eq!(end, Ok((Finish::Length, 0)));
    assert_eq!(g, expected(&a2, 5, &[]));
    // b's conversation still cached.
    b2.extend(&expected(&b2, 4, &[]));
    let (_, end) = collect(srv.send(b2.clone(), 3, false));
    assert!(matches!(end, Ok((Finish::Length, r)) if r > b.len()), "{end:?}");
    srv.stop();
}

/// Fixed-size states: a request on a slot without a state drops the least recently used idle slot's
/// state when the device is full.
#[test]
fn idle_state_evicted_for_new_request() {
    let srv = server(3, 100, Opts { total: 250, ..Default::default() });
    let a = prompt(1, 9);
    let b = prompt(2, 9);
    let (ga, _) = collect(srv.send(a.clone(), 5, false));
    let (gb, _) = collect(srv.send(b.clone(), 5, false));
    let c = prompt(3, 9);
    let (g, end) = collect(srv.send(c.clone(), 5, false));
    assert_eq!(end, Ok((Finish::Length, 0)));
    assert_eq!(g, expected(&c, 5, &[]));
    // a's state (least recently used) went; b's stayed.
    let b2 = [b.clone(), gb.clone(), vec![1]].concat();
    let (_, end) = collect(srv.send(b2, 3, false));
    assert_eq!(end, Ok((Finish::Length, b.len() + gb.len() - 1)));
    let a2 = [a, ga, vec![1]].concat();
    let (g, end) = collect(srv.send(a2.clone(), 3, false));
    assert_eq!((g, end), (expected(&a2, 3, &[]), Ok((Finish::Length, 0))));
    assert_eq!(srv.log().iter().filter(|l| l.starts_with("new ")).count(), 4);
    srv.stop();
}

/// A request whose plan does not fit next to a running one waits for it, then takes its memory.
#[test]
fn request_waits_for_memory() {
    let srv = server(2, 100, Opts { total: 150, step_ms: 1, ..Default::default() });
    let a = prompt(1, 9);
    let b = prompt(2, 9);
    let ra = srv.send(a.clone(), 60, true);
    let rb = srv.send(b.clone(), 10, true);
    let (ga, ea) = collect(ra);
    let (gb, eb) = collect(rb);
    assert_eq!((ga, ea), (expected(&a, 60, &[]), Ok((Finish::Length, 0))));
    assert_eq!((gb, eb), (expected(&b, 10, &[]), Ok((Finish::Length, 0))));
    let log = srv.log();
    assert!(!log.iter().any(|l| l.starts_with("step ") && l.contains(',')), "{log:?}");
    // b's state was made after a's last step.
    let last_a = log.iter().rposition(|l| l == "step 1").unwrap();
    let new_b = log.iter().position(|l| l == "new 2").unwrap();
    assert!(new_b > last_a, "{log:?}");
    srv.stop();
}

/// Growable states: idle slots' mapped memory goes first; a request bigger than the device fails.
#[test]
fn growable_memory_is_shared() {
    let srv = server(2, 200, Opts { total: 220, growable: true, ..Default::default() });
    let a = prompt(1, 20);
    let (ga, end) = collect(srv.send(a.clone(), 30, true));
    assert_eq!(end, Ok((Finish::Length, 0)));
    // b needs most of the device: a's idle memory is unmapped.
    let b = prompt(2, 120);
    let (gb, end) = collect(srv.send(b.clone(), 60, true));
    assert_eq!(end, Ok((Finish::Length, 0)));
    assert_eq!(gb, expected(&b, 60, &[]));
    let a2 = [a, ga, vec![5]].concat();
    let (g, end) = collect(srv.send(a2.clone(), 5, true));
    assert_eq!(end, Ok((Finish::Length, 0)));
    assert_eq!(g, expected(&a2, 5, &[]));
    srv.stop();
    // Nothing else holds memory and it still does not fit: an error.
    let srv = server(2, 400, Opts { total: 100, growable: true, ..Default::default() });
    let (_, end) = collect(srv.send(prompt(1, 150), 5, true));
    assert!(end.as_ref().is_err_and(|e| e.contains("does not fit in device memory") && e.contains("about 100 positions")), "{end:?}");
    // The prompt fits, its completion runs out of memory: an error that says so.
    let (g, end) = collect(srv.send(prompt(3, 80), 50, true));
    assert!(end.as_ref().is_err_and(|e| e.contains("device memory is full at")), "{end:?}");
    assert!(!g.is_empty() && g.len() < 50 && g == expected(&prompt(3, 80), g.len(), &[]));
    // The slot works afterwards.
    let p = prompt(2, 10);
    let (g, end) = collect(srv.send(p.clone(), 5, true));
    assert_eq!((g, end), (expected(&p, 5, &[]), Ok((Finish::Length, 0))));
    srv.stop();
}

/// Fixed-size states bigger than the device: an error, and the server goes on.
#[test]
fn state_bigger_than_device_fails() {
    let srv = server(2, 100, Opts { total: 50, ..Default::default() });
    let (_, end) = collect(srv.send(prompt(1, 10), 5, false));
    assert!(end.as_ref().is_err_and(|e| e.contains("out of memory")), "{end:?}");
    srv.stop();
}

/// A request whose client went away ends (queued: before starting); the others go on.
#[test]
fn cancellation() {
    let srv = server(2, 4096, Opts { step_ms: 1, ..Default::default() });
    drop(srv.send(prompt(9, 10), 50, false));
    let a = prompt(1, 10);
    let mut ra = srv.send(a.clone(), 3000, false);
    let b = prompt(2, 10);
    let rb = srv.send(b.clone(), 40, false);
    for _ in 0..5 {
        assert!(matches!(ra.blocking_recv(), Some(Msg::Token(_))));
    }
    drop(ra);
    let (gb, eb) = collect(rb);
    assert_eq!((gb, eb), (expected(&b, 40, &[]), Ok((Finish::Length, 0))));
    let c = prompt(3, 10);
    let (gc, ec) = collect(srv.send(c.clone(), 10, false));
    assert_eq!((gc, ec), (expected(&c, 10, &[]), Ok((Finish::Length, 0))));
    let log = srv.log();
    // The dropped queued request never got a state; a stopped long before its 3000 tokens.
    assert_eq!(log.iter().filter(|l| l.starts_with("new ")).count(), 2, "{log:?}");
    let a_steps = log.iter().filter(|l| l.starts_with("step ") && l[5..].split(',').any(|s| s == "1")).count();
    assert!(a_steps < 1000, "{a_steps} steps of the cancelled request");
    srv.stop();
}

/// An error in one slot ends only its request; the slot serves the next one.
#[test]
fn error_isolation() {
    for mtp in [false, true] {
        let srv = server(2, 512, Opts { drafter: Drafter::Count, ..Default::default() });
        let bad = [prompt(1, 6), vec![FAIL], prompt(4, 3)].concat();
        let rbad = srv.send(bad, 50, mtp);
        let good = prompt(2, 12);
        let rgood = srv.send(good.clone(), 50, mtp);
        let (_, e) = collect(rbad);
        assert!(e.as_ref().is_err_and(|e| e.contains("injected failure")), "{e:?}");
        let (g, e) = collect(rgood);
        assert_eq!((g, e), (expected(&good, 50, &[]), Ok((Finish::Length, 0))));
        let p = prompt(3, 8);
        let (g, e) = collect(srv.send(p.clone(), 20, mtp));
        assert_eq!((g, e), (expected(&p, 20, &[]), Ok((Finish::Length, 0))));
        srv.stop();
    }
}

/// Stop tokens, budgets and the context limit.
#[test]
fn stops_budgets_and_context() {
    let srv = server(2, 64, Opts::default());
    let p = prompt(1, 10);
    let full = expected(&p, 20, &[]);
    let (g, e) = collect(srv.send_p(p.clone(), params(20, vec![full[4]]), true));
    let stopped = expected(&p, 20, &[full[4]]);
    assert_eq!((g, e), (stopped.clone(), Ok((Finish::Stop, 0))));
    assert!(stopped.len() <= 5);
    // max_tokens beyond the context: the context ends it.
    let (g, e) = collect(srv.send(prompt(2, 50), 100, true));
    assert_eq!(g.len(), 14);
    assert_eq!(e, Ok((Finish::Length, 0)));
    let (g, e) = collect(srv.send(prompt(3, 64), 10, true));
    assert!(g.is_empty() && e.as_ref().is_err_and(|e| e.contains("exceeds the context")), "{e:?}");
    let (_, e) = collect(srv.send(Vec::new(), 10, true));
    assert!(e.is_err());
    srv.stop();
}

/// An unload request waits for the running requests.
#[test]
fn unload_waits_for_running() {
    let srv = server(2, 512, Opts { step_ms: 1, ..Default::default() });
    let p = prompt(1, 10);
    let rx = srv.send(p.clone(), 50, false);
    let (ack, acked) = mpsc::channel();
    srv.jobs.send(Work::Unload(ack)).unwrap();
    let (g, e) = collect(rx);
    assert_eq!((g, e), (expected(&p, 50, &[]), Ok((Finish::Length, 0))));
    assert!(srv.worker.join().unwrap());
    drop(acked);
}

/// A prompt prefilled while another request decodes takes turns with its steps (--prefill-share).
#[test]
fn prefill_interleaves_with_decode() {
    let srv = server_with(2, 16384, Opts { chunk: 128, prefill_ms: 1, step_ms: 1, ..Default::default() }, 0.5);
    let a = prompt(1, 4);
    let mut ra = srv.send(a.clone(), 1000, true);
    let mut ga = Vec::new();
    for _ in 0..3 {
        let Some(Msg::Token(t)) = ra.blocking_recv() else { panic!() };
        ga.push(t);
    }
    let b = prompt(2, 10000);
    let (gb, eb) = collect(srv.send(b.clone(), 10, true));
    assert_eq!((gb, eb), (expected(&b, 10, &[]), Ok((Finish::Length, 0))));
    let (rest, ea) = collect(ra);
    ga.extend(rest);
    assert_eq!((ga, ea), (expected(&a, 1000, &[]), Ok((Finish::Length, 0))));
    let log = srv.log();
    let first = log.iter().position(|l| l == "prefill 2").unwrap();
    let last = log.iter().rposition(|l| l == "prefill 2").unwrap();
    let between = log[first..last].iter().filter(|l| l.starts_with("step ")).count();
    assert!(between > 5, "{between} steps during the prefill: {:?}", &log[first..=last]);
    srv.stop();
}

/// The step-cost fit follows observed steps; a sequence's shares add up to the step.
#[test]
fn step_cost_shares() {
    let mut c = StepCost::new(0.010, 0.0018);
    for i in 0..400 {
        let r = [1.0, 2.0, 4.0, 8.0, 12.0, 32.0][i % 6];
        c.add(r, 0.0078 + 0.002 * r);
    }
    let (a, b) = c.fit();
    assert!((a - 0.0078).abs() < 0.0003 && (b - 0.002).abs() < 0.0001, "{a} {b}");
    assert_eq!(c.share(1, 4, 4), 1.0);
    let widths = [1, 4, 2, 1];
    let total: f64 = widths.iter().map(|&w| c.share(4, w, 8)).sum();
    assert!((total - 1.0).abs() < 1e-9, "{total}");
    // Eight sequences: a plain row costs (a/8 + b), three more drafts 3b more.
    let plain = c.share(8, 1, 8) * (a + 8.0 * b);
    assert!((plain - (a / 8.0 + b)).abs() < 1e-9);
}

/// A new conversation sharing only a beginning (a system prompt) with an idle slot's takes a free
/// slot; the idle conversation keeps its own.
#[test]
fn shared_beginning_does_not_take_a_slot() {
    let srv = server(2, 512, Opts::default());
    let system = prompt(7, 6);
    let a = [system.clone(), prompt(1, 30)].concat();
    let (ga, _) = collect(srv.send(a.clone(), 5, false));
    let b = [system.clone(), prompt(2, 30)].concat();
    let (gb, end) = collect(srv.send(b.clone(), 5, false));
    assert_eq!((gb, end), (expected(&b, 5, &[]), Ok((Finish::Length, 0))));
    let a2 = [a.clone(), ga.clone(), vec![9]].concat();
    let (_, end) = collect(srv.send(a2, 3, false));
    assert_eq!(end, Ok((Finish::Length, a.len() + ga.len() - 1)));
    srv.stop();
}

/// Prefill chunks end at multiples of the chunk size whatever was reused.
#[test]
fn prefill_chunks_are_aligned() {
    let srv = server(1, 512, Opts { chunk: 8, ..Default::default() });
    let a = prompt(1, 13);
    let (ga, _) = collect(srv.send(a.clone(), 4, false));
    // Continues a's conversation: 13 + 3 tokens reused, the rest from position 16 in chunks of 8.
    let a2 = [a.clone(), ga.clone(), prompt(2, 20)].concat();
    let (_, end) = collect(srv.send(a2.clone(), 2, false));
    assert_eq!(end, Ok((Finish::Length, a.len() + ga.len() - 1)));
    let chunks: Vec<String> = srv.log().into_iter().filter(|l| l.starts_with("chunk ")).collect();
    assert_eq!(chunks, ["chunk 1 0..8", "chunk 1 8..13", "chunk 1 16..24", "chunk 1 24..32", "chunk 1 32..37"]);
    srv.stop();
}

/// The most sequences any step ran together.
fn widest(log: &[String]) -> usize {
    log.iter().filter_map(|l| l.strip_prefix("step ")).map(|l| l.split(',').count()).max().unwrap_or(0)
}

/// Requests of the given prompt lengths sent at once (32 tokens each, no drafts): all complete with
/// their own tokens; returns the most that stepped together.
fn run_layout(srv: &Server, lens: &[usize]) -> usize {
    let prompts: Vec<Vec<i32>> = lens.iter().enumerate().map(|(i, &n)| prompt(i as i32, n)).collect();
    let rxs: Vec<Reply> = prompts.iter().map(|p| srv.send(p.clone(), 32, false)).collect();
    for (i, rx) in rxs.into_iter().enumerate() {
        let (g, e) = collect(rx);
        assert_eq!(e, Ok((Finish::Length, 0)), "{lens:?}: request {i}");
        assert!(g == expected(&prompts[i], 32, &[]), "{lens:?}: request {i}: wrong tokens");
    }
    widest(&srv.log())
}

/// Inkling's catalog setting (8 slots, --context 1048576) on the measured device: one conversation of
/// 1M, two of 512K, four of 256K, eight of 128K, or a mix, run at the same time (each with its plan:
/// prompt + completion + margin); two of 1M take turns.
#[test]
fn inkling_conversations_share_the_device() {
    let ctx = 1 << 20;
    for n in [1usize, 2, 4, 8] {
        let srv = server(8, ctx, Opts::inkling());
        let len = ctx / n - PLAN_MARGIN - 64;
        assert_eq!(run_layout(&srv, &vec![len; n]), n, "{n} x {len}");
        srv.stop();
    }
    let srv = server(8, ctx, Opts::inkling());
    let mix = [600_000, 150_000, 150_000, 50_000, 50_000, 50_000, 20_000, 20_000];
    assert_eq!(run_layout(&srv, &mix), 8, "{mix:?}");
    srv.stop();
    let srv = server(8, ctx, Opts::inkling());
    assert_eq!(run_layout(&srv, &[ctx - 8192, ctx - 8192]), 1);
    srv.stop();
}

/// Flint (8 slots, --context 262144): eight conversations of 256K fit at once.
#[test]
fn flint_conversations_share_the_device() {
    let ctx = 1 << 18;
    for n in [1usize, 2, 4, 8] {
        let srv = server(8, ctx, Opts::flint());
        let len = ctx / n - PLAN_MARGIN - 64;
        assert_eq!(run_layout(&srv, &vec![len; n]), n, "{n} x {len}");
        srv.stop();
    }
    let srv = server(8, ctx, Opts::flint());
    assert_eq!(run_layout(&srv, &[ctx - PLAN_MARGIN - 64; 8]), 8);
    srv.stop();
}

/// Hy4-flash (8 slots, --context 262144): the device holds about 200K positions in all, so a 250K
/// prompt fails at once with an error that says how much fits; 190K runs alone, two of 90K, four of 38K
/// and eight of 12K run at once (plans: prompt + completion + PLAN_MARGIN, and 0.15 GiB per state), two
/// of 120K take turns.
#[test]
fn hy4_conversations_share_what_fits() {
    let ctx = 1 << 18;
    let srv = server(8, ctx, Opts::hy4());
    let (g, e) = collect(srv.send(prompt(1, 250_000), 8, false));
    assert!(g.is_empty() && e.as_ref().is_err_and(|e| e.contains("does not fit in device memory") && e.contains("one sequence fits about")), "{e:?}");
    assert_eq!(run_layout(&srv, &[190_000]), 1);
    srv.stop();
    for (n, len) in [(2usize, 90_000usize), (4, 38_000), (8, 12_000)] {
        let srv = server(8, ctx, Opts::hy4());
        assert_eq!(run_layout(&srv, &vec![len; n]), n, "{n} x {len}");
        srv.stop();
    }
    let srv = server(8, ctx, Opts::hy4());
    assert_eq!(run_layout(&srv, &[120_000, 120_000]), 1);
    srv.stop();
}

/// Requests at --context: a prompt of the whole context fails with an error; one token short of it
/// gets its one token.
#[test]
fn inkling_context_limit() {
    let ctx = 1 << 20;
    let srv = server(8, ctx, Opts::inkling());
    let (g, e) = collect(srv.send(prompt(1, ctx), 8, false));
    assert!(g.is_empty() && e.as_ref().is_err_and(|e| e.contains("exceeds the context of 1048576")), "{e:?}");
    let p = prompt(2, ctx - 1);
    let (g, e) = collect(srv.send(p.clone(), 8, false));
    assert_eq!((g, e), (expected(&p, 1, &[]), Ok((Finish::Length, 0))));
    srv.stop();
}

/// Idle conversations give their whole state up (caches and rings), least recently used first, for a
/// new conversation of 1M; the most recent ones stay cached.
#[test]
fn idle_conversations_give_way_to_a_long_one() {
    let ctx = 1 << 20;
    let srv = server(8, ctx, Opts::inkling());
    let convs: Vec<Vec<i32>> = (0..8).map(|i| prompt(i, 100_000)).collect();
    let mut outs = Vec::new();
    for c in &convs {
        let (g, e) = collect(srv.send(c.clone(), 8, false));
        assert_eq!(e, Ok((Finish::Length, 0)));
        outs.push(g);
    }
    let long = prompt(99, ctx - 8192);
    let (g, e) = collect(srv.send(long.clone(), 32, false));
    assert_eq!((g, e), (expected(&long, 32, &[]), Ok((Finish::Length, 0))));
    // Conversation 0's slot (least recently used) took the long one; then 1, 2, .. gave way in order.
    let dropped: Vec<usize> = srv.log().iter().filter_map(|l| l.strip_prefix("drop ")?.parse().ok()).collect();
    assert!(!dropped.is_empty() && dropped.iter().enumerate().all(|(k, &id)| id == k + 2), "{dropped:?}");
    // The most recent conversation goes on from its cache; the first evicted one starts over.
    let go_on = |i: usize| {
        let p = [convs[i].clone(), outs[i].clone(), vec![5]].concat();
        let (g, e) = collect(srv.send(p.clone(), 4, false));
        assert_eq!(g, expected(&p, 4, &[]));
        e.unwrap().1
    };
    assert_eq!(go_on(7), convs[7].len() + outs[7].len() - 1);
    assert_eq!(go_on(1), 0);
    srv.stop();
}

/// A decoding conversation that grows into memory an idle one holds takes it (the idle one is dropped
/// while the request decodes) and completes.
#[test]
fn decode_growth_takes_idle_memory() {
    let ctx = 1 << 20;
    let o = Opts::inkling();
    let srv = server(2, ctx, o.clone());
    let idle = prompt(1, ctx - 8192);
    let (_, e) = collect(srv.send(idle, 8, false));
    assert_eq!(e, Ok((Finish::Length, 0)));
    // A prompt whose prefill fits beside the idle conversation, whose completion does not.
    let avail = o.total - srv.used.load(Ordering::Relaxed) - o.fixed - o.snap;
    let fits = avail / o.per / o.grain * o.grain;
    let p = prompt(2, fits - 2 * o.grain);
    let n = 4 * o.grain;
    let (g, e) = collect(srv.send(p.clone(), n, false));
    assert_eq!(e, Ok((Finish::Length, 0)));
    assert!(g == expected(&p, n, &[]));
    let log = srv.log();
    let dropped = log.iter().position(|l| l == "drop 1").expect("idle state dropped");
    let last_chunk = log.iter().rposition(|l| l.starts_with("chunk 2 ")).unwrap();
    assert!(dropped > last_chunk, "dropped during the prefill, not the decode");
    srv.stop();
}

/// Running requests that together outgrow the device: the newest one beyond its plan ends with an
/// error that says so; the older one completes.
#[test]
fn outgrowing_requests_newest_gives_way() {
    let cfg = Config { slots: 2, context: 100_000, plan_tokens: 100, prefill_share: 1.0, unload_idle: None };
    // Plans: 4000 + 100 + 4096 and 5000 + 100 + 4096 positions; both grow past them.
    let srv = server_cfg(cfg, Opts { total: 20_000, growable: true, step_ms: 1, ..Default::default() });
    let a = prompt(1, 4000);
    let mut ra = srv.send(a.clone(), 14_000, false);
    let Some(Msg::Token(first)) = ra.blocking_recv() else { panic!() };
    let b = prompt(2, 5000);
    let mut rb = srv.send(b.clone(), 8000, false);
    let Some(Msg::Token(b_first)) = rb.blocking_recv() else { panic!() };
    srv.pace.store(0, Ordering::Relaxed);
    let (gb, eb) = collect(rb);
    let (ga, ea) = collect(ra);
    assert_eq!(([vec![first], ga].concat(), ea), (expected(&a, 14_000, &[]), Ok((Finish::Length, 0))));
    assert!(eb.as_ref().is_err_and(|e| e.contains("newest") && e.contains("beyond its plan")), "{eb:?}");
    let gb = [vec![b_first], gb].concat();
    assert!(gb.len() > 4196 && gb.len() < 8000 && gb == expected(&b, gb.len(), &[]), "{}", gb.len());
    srv.stop();
}

/// An older request beyond its plan gives way to a newer one within its plan.
#[test]
fn outgrowing_requests_plans_are_kept() {
    let cfg = Config { slots: 2, context: 100_000, plan_tokens: 100, prefill_share: 1.0, unload_idle: None };
    let srv = server_cfg(cfg, Opts { total: 20_000, growable: true, step_ms: 1, ..Default::default() });
    // A's plan: 1000 + 100 + 4096 positions; it grows to 3500 before B (plan 9000 + 100 + 4096) starts.
    let a = prompt(1, 1000);
    let mut ra = srv.send(a.clone(), 16_000, false);
    let mut ga = Vec::new();
    while ga.len() < 2500 {
        let Some(Msg::Token(t)) = ra.blocking_recv() else { panic!() };
        ga.push(t);
    }
    let b = prompt(2, 9000);
    let mut rb = srv.send(b.clone(), 4000, false);
    let Some(Msg::Token(b_first)) = rb.blocking_recv() else { panic!() };
    srv.pace.store(0, Ordering::Relaxed);
    let (gb, eb) = collect(rb);
    assert_eq!(([vec![b_first], gb].concat(), eb), (expected(&b, 4000, &[]), Ok((Finish::Length, 0))));
    let (rest, ea) = collect(ra);
    ga.extend(rest);
    assert!(ea.as_ref().is_err_and(|e| e.contains("beyond its plan")), "{ea:?}");
    assert!(ga.len() > 4196 && ga.len() < 16_000 && ga == expected(&a, ga.len(), &[]), "{}", ga.len());
    srv.stop();
}
