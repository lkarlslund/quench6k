//! The slot scheduler over a CPU mock engine: a "model" whose next token is a hash of the sequence so
//! far, states that count device bytes against a budget, a drafter that guesses two tokens right and the
//! third wrong, and failures on request.

use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc as tmpsc;

use super::*;
use crate::serve::{Metrics, QueuedJob};

/// The mock model: the token after `h`.
fn next(h: &[i32]) -> i32 {
    let mut x = 0xcbf2_9ce4_8422_2325u64;
    for &t in h {
        x = (x ^ t as u64).wrapping_mul(0x100_0000_01b3);
    }
    (x % 40) as i32 + 1
}

/// Greedy output for `prompt`: `n` tokens, or up to the first stop token.
fn expected(prompt: &[i32], n: usize, stop: &[i32]) -> Vec<i32> {
    let mut h = prompt.to_vec();
    let mut out = Vec::new();
    while out.len() < n {
        let t = next(&h);
        h.push(t);
        out.push(t);
        if stop.contains(&t) {
            break;
        }
    }
    out
}

/// A prompt token whose sequence fails on its third decode step.
const FAIL: i32 = 999;

struct MState {
    id: usize,
    hist: Vec<i32>,
    cap: usize,
    bytes: usize,
    last_t: usize,
    steps: usize,
    used: Arc<AtomicUsize>,
}

impl Drop for MState {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::Relaxed);
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
    /// Device bytes (one byte per position).
    total: usize,
    growable: bool,
    drafter: Drafter,
    chunk: usize,
    /// Sleep per prefill chunk and per sequence step.
    prefill_ms: u64,
    step_ms: u64,
}

impl Default for Opts {
    fn default() -> Self {
        Self { total: 1 << 20, growable: false, drafter: Drafter::None, chunk: 4, prefill_ms: 0, step_ms: 0 }
    }
}

struct Mock {
    o: Opts,
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
        let h = [st.hist.as_slice(), toks].concat();
        let n = toks.len().min(8);
        let mut rows = [0; 8];
        for (r, row) in rows.iter_mut().enumerate().take(n) {
            *row = next(&h[..h.len() - n + r + 1]);
        }
        rows
    }

    fn feed(&mut self, st: &mut MState, toks: &[i32]) -> Result<[i32; 8]> {
        anyhow::ensure!(st.hist.len() + toks.len() <= st.cap, "context full");
        anyhow::ensure!(!self.o.growable || st.hist.len() + toks.len() <= st.bytes, "positions not reserved");
        let rows = Self::rows(st, toks);
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
        let bytes = if self.o.growable { 0 } else { capacity };
        if self.used.load(Ordering::Relaxed) + bytes > self.o.total {
            anyhow::bail!("mock alloc: out of memory");
        }
        self.used.fetch_add(bytes, Ordering::Relaxed);
        self.ids += 1;
        self.log.lock().unwrap().push(format!("new {}", self.ids));
        Ok(MState { id: self.ids, hist: Vec::new(), cap: capacity, bytes, last_t: 0, steps: 0, used: self.used.clone() })
    }
    fn growable(&self) -> bool {
        self.o.growable
    }
    fn batched(&self) -> bool {
        true
    }
    fn bytes_per_position(&self) -> usize {
        1
    }
    fn reserve(&mut self, st: &mut MState, positions: usize) -> Result<()> {
        if !self.o.growable {
            return Ok(());
        }
        let want = positions.min(st.cap);
        if want > st.bytes {
            if self.used.load(Ordering::Relaxed) + want - st.bytes > self.o.total {
                anyhow::bail!("mock map: out of memory");
            }
            self.used.fetch_add(want - st.bytes, Ordering::Relaxed);
            st.bytes = want;
        }
        Ok(())
    }
    fn trim(&mut self, st: &mut MState, positions: usize) {
        if self.o.growable && positions < st.bytes {
            self.used.fetch_sub(st.bytes - positions, Ordering::Relaxed);
            st.bytes = positions;
        }
        st.hist.truncate(positions);
    }
    fn device_bytes(&self, st: &MState) -> usize {
        st.bytes
    }
    fn plan_bytes(&self, positions: usize) -> usize {
        if self.o.growable { positions } else { self.capacity }
    }
    fn free_bytes(&self) -> usize {
        // The scheduler keeps HEADROOM free.
        self.o.total - self.used.load(Ordering::Relaxed) + HEADROOM
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
        st.hist.truncate(keep);
        Ok(keep)
    }
    fn snapshot(&mut self, _st: &mut MState) -> Result<()> {
        Ok(())
    }
    fn reset(&mut self, st: &mut MState) {
        st.hist.clear();
    }
    fn sync(&self) -> Result<()> {
        Ok(())
    }
    fn prefill(&mut self, st: &mut MState, tokens: &[i32], _last: bool) -> Result<[i32; 8]> {
        self.log.lock().unwrap().push(format!("prefill {}", st.id));
        std::thread::sleep(Duration::from_millis(self.o.prefill_ms));
        let rows = self.feed(st, tokens)?;
        // The last token's logits in row 0.
        Ok([rows[tokens.len().min(8) - 1]; 8])
    }
    fn draft(&mut self, st: &mut MState, token: i32, row: usize) -> Result<Draft> {
        anyhow::ensure!(self.last == st.id, "draft after another sequence's step");
        anyhow::ensure!(row < st.last_t, "draft row {row} of a {}-token step", st.last_t);
        let mut h = st.hist.clone();
        h.push(token);
        let mut tokens = Vec::new();
        for _ in 0..2 {
            let t = next(&h);
            h.push(t);
            tokens.push(t);
        }
        tokens.push(next(&h) % 40 + 1);
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
                std::thread::sleep(Duration::from_millis(e.o.step_ms));
                st.steps += 1;
                anyhow::ensure!(!(st.hist.contains(&FAIL) && st.steps >= 3), "injected failure");
                e.feed(st, toks)
            },
            |st, pos| st.hist.truncate(pos),
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
    worker: std::thread::JoinHandle<bool>,
}

/// The scheduler over a mock on a thread; joins true when it ended with Exit::Unload.
fn server(slots: usize, context: usize, o: Opts) -> Server {
    server_with(slots, context, o, 1.0)
}

fn server_with(slots: usize, context: usize, o: Opts, prefill_share: f64) -> Server {
    let (jobs, rx) = mpsc::channel();
    let log = Arc::new(Mutex::new(Vec::new()));
    let mock = Mock { o, used: Arc::new(AtomicUsize::new(0)), log: log.clone(), ids: 0, capacity: context, last: 0 };
    let worker = std::thread::spawn(move || {
        let metrics = Metrics::new(slots);
        let cfg = Config { slots, context, plan_tokens: 16384, prefill_share, unload_idle: None };
        matches!(schedule(mock, &cfg, &rx, &metrics, Vec::new()), Exit::Unload(_))
    });
    Server { jobs, log, worker }
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
