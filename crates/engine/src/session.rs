//! A reusable generation session for serving: one sequence's state, workspace and captured graphs,
//! reset between requests. Greedy requests decode speculatively (MTP, adaptive draft length); sampled
//! requests decode one token per step and sample on the host from the LM head's top-k logits.

use anyhow::{Result, ensure};
use squash_cuda::{DeviceBuffer, Stream};

use crate::forward::{self, BatchGraph, DecodeGraph, HeadWorkspace, SequenceState, Workspace};
use crate::model::{Model, VOCAB};
use crate::context_store::{ContextStore, StoreOptions};
use crate::spec::{SpecDecoder, SpecGraphs};

/// Sampling and stopping for one request. `temperature == 0` is greedy.
#[derive(Clone, Debug)]
pub struct GenParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    /// Drops candidates below min_p times the most likely one's probability (0: off).
    pub min_p: f32,
    pub seed: u64,
    /// Generation ends after any of these tokens (which is emitted).
    pub stop_tokens: Vec<i32>,
    pub steering_strength: f32,
    /// Prompt positions worth a recurrent-state snapshot for prefix reuse (e.g. the start of the final
    /// assistant turn, which a follow-up turn shares).
    pub boundaries: Vec<usize>,
    /// Record activations at the final prompt token and the final generated token (plain decoding).
    pub capture: bool,
    /// Images of the prompt, in order; each fills a run of placeholder tokens.
    pub images: Vec<std::sync::Arc<ImageInput>>,
    /// Called during prefill with (prompt tokens computed so far, of which served from cache).
    pub progress: Option<Progress>,
}

/// A prefill progress callback: (prompt tokens computed so far, of which served from cache).
#[derive(Clone)]
pub struct Progress(pub std::sync::Arc<dyn Fn(usize, usize) + Send + Sync>);

impl std::fmt::Debug for Progress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Progress")
    }
}

/// A preprocessed image or video. Each temporal group (one for an image) fills its own run of
/// placeholder tokens, starting at the matching prompt index of `starts`.
pub struct ImageInput {
    pub starts: Vec<usize>,
    pub input: crate::vision::VisionInput,
}

/// A content key per image span start: (prompt index, a negative pseudo-token ID from a hash of the
/// pixels and grid). Real token IDs are non-negative, so a key never matches a token.
fn image_marks(images: &[std::sync::Arc<ImageInput>]) -> Vec<(usize, i32)> {
    let mut marks = Vec::new();
    for im in images {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in im.input.grid.iter().map(|&g| g as u32).chain(im.input.pixels.iter().map(|p| p.to_bits())) {
            h = (h ^ v as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        for (k, &start) in im.starts.iter().enumerate() {
            let hk = (h ^ (k as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)).wrapping_mul(0xff51_afd7_ed55_8ccd);
            marks.push((start, -1 - ((hk >> 33) as i32 & 0x3fff_ffff)));
        }
    }
    marks
}

/// `tokens` with the marked positions replaced by their keys.
fn keyed(tokens: &[i32], marks: &[(usize, i32)]) -> Vec<i32> {
    let mut v = tokens.to_vec();
    for &(i, key) in marks {
        if i < v.len() {
            v[i] = key;
        }
    }
    v
}

impl std::fmt::Debug for ImageInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImageInput {{ starts: {:?}, grid: {:?} }}", self.starts, self.input.grid)
    }
}


/// Activation capture of one request: per site (prompt_last, completion_last) the per-layer records
/// (`forward::CaptureLayout`, 48 layers), the absolute position and token, and the top-20 next-token
/// ids with log-probabilities.
pub struct CaptureResult {
    pub sites: Vec<CaptureSite>,
}

pub struct CaptureSite {
    pub name: &'static str,
    pub position: usize,
    pub token: i32,
    pub records: Vec<u8>,
    pub top_ids: Vec<i32>,
    pub top_logprobs: Vec<f32>,
}

/// Top-20 ids and log-probabilities of a logits row.
fn top20(logits: &[f32]) -> (Vec<i32>, Vec<f32>) {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + logits.iter().map(|v| ((v - m) as f64).exp()).sum::<f64>().ln() as f32;
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(20, |&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(20);
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
    (idx.iter().map(|&i| i as i32).collect(), idx.iter().map(|&i| logits[i] - lse).collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// A stop token was generated.
    Stop,
    /// `max_tokens` reached, or the context is full.
    Length,
    /// The token callback asked to stop.
    Cancelled,
}

pub struct Session {
    ws: Workspace,
    state: SequenceState,
    hw: HeadWorkspace,
    graph: DecodeGraph,
    dec: SpecDecoder,
    spec: SpecGraphs,
    chunk: usize,
    capacity: usize,
    top_idx: DeviceBuffer,
    top_val: DeviceBuffer,
    /// Tokens whose attention K/V are cached at their positions (the committed sequence).
    kv_tokens: Vec<i32>,
    /// Recurrent-state snapshots at prompt positions, oldest first.
    snapshots: Vec<Snapshot>,
    /// Prompt tokens served from a snapshot in the last request.
    pub reused: usize,
    /// Steering strength the cached state was computed with (reuse needs the same strength).
    kv_strength: f32,
    /// Host and disk tiers of stored contexts.
    store: ContextStore,
    /// The second, plain-decoding slot for a concurrent request.
    aux: Option<Aux>,
    /// Activation capture records (two sites x 48 layers), and the prompt_last site of the current run.
    capture_buf: Option<DeviceBuffer>,
    prompt_site: Option<CaptureSite>,
    vision: Option<crate::vision::VisionModel>,
    /// Prompt positions whose `kv_tokens` entry is an image's content key instead of its placeholder
    /// ID (the first placeholder of each image), so the cache matches images by content.
    kv_marks: Vec<(usize, i32)>,
}

struct Aux {
    state: SequenceState,
    graph: DecodeGraph,
    batch: BatchGraph,
    hw2: HeadWorkspace,
    _out2: DeviceBuffer,
}

const CAPTURE_LAYER_BYTES: usize = forward::CaptureLayout::BYTES;

/// A request in progress on a slot (0: the main slot with MTP and the prefix cache; 1: the plain slot).
pub struct Run {
    pub slot: usize,
    pub params: GenParams,
    pub produced: usize,
    /// Speculative (slot 0, greedy, alone) or plain one-token steps.
    spec: bool,
    /// Plain mode: the token to feed next (already emitted).
    pending: i32,
    rng: Rng,
    budget: usize,
}

impl Run {
    /// Accounts an emitted token; returns why the run ends, if it does.
    pub fn accept(&mut self, tok: i32) -> Option<Finish> {
        self.produced += 1;
        if self.params.stop_tokens.contains(&tok) {
            return Some(Finish::Stop);
        }
        if self.produced >= self.budget {
            return Some(Finish::Length);
        }
        None
    }

    pub fn is_spec(&self) -> bool {
        self.spec
    }
}

/// The recurrent state at a position: every GDN layer's conv history and delta-rule state, the PLE conv
/// history and the speculative decoder's pending row (attention caches are position-indexed and stay).
struct Snapshot {
    pos: usize,
    buf: DeviceBuffer,
    /// Taken at a chunk boundary inside a long prompt rather than at a turn boundary; these are
    /// recycled first.
    periodic: bool,
}

const MAX_SNAPSHOTS: usize = 12;
/// Device memory a new snapshot leaves free; below it, an older snapshot is recycled instead.
const SNAPSHOT_HEADROOM: usize = 1 << 30;
/// Periodic snapshots per prompt (at most): a long prompt edited in the middle re-prefills at most
/// about 1/8 of itself.
const PERIODIC_SNAPSHOTS: usize = 8;

/// The largest top-k the session samples from.
pub const MAX_TOP_K: usize = 64;

impl Session {
    /// `capacity`: longest prompt + completion; `chunk`: prefill tokens per forward pass; `max_drafts`:
    /// longest speculative draft (adaptive 1..=max_drafts).
    pub fn new(model: &Model, s: &Stream, capacity: usize, chunk: usize, max_drafts: usize, store: StoreOptions) -> Result<Self> {
        let slack = 2 * (max_drafts + 2) + 16;
        let mut ws = Workspace::for_model(&model, chunk, capacity + slack)?;
        let mut state = SequenceState::with_mtp(capacity + slack, true)?;
        let hw = HeadWorkspace::new(1, MAX_TOP_K)?;
        // A one-token prompt to bring the sequence into a valid state for capture.
        let dec = SpecDecoder::prefill(model, &mut state, &mut ws, s, &[0], max_drafts, chunk)?;
        let graph = DecodeGraph::capture(model, &state, &ws, &hw, s)?;
        let spec = SpecGraphs::capture(model, &state, &mut ws, &dec, s, true)?;
        Ok(Self {
            ws,
            state,
            hw,
            graph,
            dec,
            spec,
            chunk,
            capacity,
            top_idx: DeviceBuffer::new(MAX_TOP_K * 4)?,
            top_val: DeviceBuffer::new(MAX_TOP_K * 4)?,
            kv_tokens: Vec::new(),
            snapshots: Vec::new(),
            reused: 0,
            kv_strength: 0.0,
            store: ContextStore::open(store)?,
            aux: None,
            capture_buf: None,
            prompt_site: None,
            vision: None,
            kv_marks: Vec::new(),
        })
    }

    fn capture_buf(&mut self) -> Result<u64> {
        if self.capture_buf.is_none() {
            self.capture_buf = Some(DeviceBuffer::zeroed(2 * crate::model::LAYERS * CAPTURE_LAYER_BYTES)?);
        }
        Ok(self.capture_buf.as_ref().unwrap().addr())
    }

    fn read_site(&self, s: &Stream, name: &'static str, buf: u64, position: usize, token: i32, logits: u64) -> Result<CaptureSite> {
        let records: Vec<u8> = s.download(buf, crate::model::LAYERS * CAPTURE_LAYER_BYTES)?;
        let row: Vec<f32> = s.download(logits, VOCAB)?;
        let (top_ids, top_logprobs) = top20(&row);
        Ok(CaptureSite { name, position, token, records, top_ids, top_logprobs })
    }

    /// Ends a capture run: feeds its final generated token through the model once, recording the
    /// completion_last site, and returns both sites.
    pub fn finish_capture(&mut self, model: &Model, s: &Stream, run: &Run) -> Result<Option<CaptureResult>> {
        let Some(prompt_site) = self.prompt_site.take() else { return Ok(None) };
        if !run.params.capture || run.spec {
            return Ok(None);
        }
        let buf = self.capture_buf()? + (crate::model::LAYERS * CAPTURE_LAYER_BYTES) as u64;
        let state = if run.slot == 0 { &mut self.state } else { &mut self.aux.as_mut().expect("second slot").state };
        let pos = state.len();
        self.ws.capture = Some(forward::CaptureTarget { pos, buf });
        let r = forward::forward(model, state, &mut self.ws, s, &[run.pending]);
        self.ws.capture = None;
        r?;
        forward::head_logits(model, &self.hw, s, self.ws.out)?;
        if run.slot == 0 {
            self.sync_kv_tokens();
        }
        let site = self.read_site(s, "completion_last", buf, pos, run.pending, self.hw.logits_addr())?;
        Ok(Some(CaptureResult { sites: vec![prompt_site, site] }))
    }

    /// Adds the second slot (plain decoding, batched with the main slot when both are busy).
    pub fn enable_second_slot(&mut self, model: &Model, s: &Stream) -> Result<()> {
        let mut state = SequenceState::new(self.state.capacity)?;
        forward::forward(model, &mut state, &mut self.ws, s, &[0])?;
        let graph = DecodeGraph::capture(model, &state, &self.ws, &self.hw, s)?;
        let hw2 = HeadWorkspace::new(2, MAX_TOP_K)?;
        let out2 = DeviceBuffer::new(8)?;
        let batch = BatchGraph::capture(model, &[&self.state, &state], &self.ws, &hw2, s, out2.addr())?;
        self.aux = Some(Aux { state, graph, batch, hw2, _out2: out2 });
        Ok(())
    }

    /// Enables image input with the given vision tower.
    pub fn set_vision(&mut self, vision: crate::vision::VisionModel) {
        self.vision = Some(vision);
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// Encodes the request's images. When device memory runs out, GPU snapshots are freed (periodic
    /// ones first; others are saved to the context store first) and the encoding retried.
    /// Image spans for a prefill from position `from` (see [`VisionModel::spans`](crate::vision::VisionModel::spans)).
    fn encode_images(&mut self, s: &Stream, images: &[std::sync::Arc<ImageInput>], from: usize) -> Result<Vec<std::sync::Arc<forward::MmSpan>>> {
        loop {
            match self.encode_images_once(s, images, from) {
                Err(e) if format!("{e:#}").contains("out of memory") && !self.snapshots.is_empty() => {
                    let i = self.snapshots.iter().position(|sn| sn.periodic).unwrap_or(0);
                    let pos = self.snapshots[i].pos;
                    if !self.snapshots[i].periodic {
                        self.stash(pos)?;
                    }
                    self.snapshots.remove(i);
                    eprintln!("image encoder out of memory: freed the snapshot at {pos}, retrying");
                }
                r => return r,
            }
        }
    }

    fn encode_images_once(&self, s: &Stream, images: &[std::sync::Arc<ImageInput>], from: usize) -> Result<Vec<std::sync::Arc<forward::MmSpan>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let vm = self.vision.as_ref().ok_or_else(|| anyhow::anyhow!("image input needs the vision tower (--vision)"))?;
        let mut spans = Vec::new();
        for im in images {
            spans.extend(vm.spans(s, &im.input, &im.starts, from)?.into_iter().map(std::sync::Arc::new));
        }
        spans.sort_by_key(|sp| sp.start);
        Ok(spans)
    }

    /// Speculative rounds and accepted drafts of the slot-0 run since its prefill.
    pub fn spec_stats(&self) -> (usize, usize) {
        (self.dec.rounds, self.dec.accepted)
    }

    pub fn has_second_slot(&self) -> bool {
        self.aux.is_some()
    }

    /// Starts a request on `slot`: prefills and returns the run with the tokens to emit now (none for a
    /// speculative run, whose first round emits).
    pub fn begin(&mut self, model: &Model, s: &Stream, slot: usize, prompt: &[i32], p: &GenParams, speculate: bool) -> Result<(Run, Vec<i32>)> {
        ensure!(!prompt.is_empty(), "empty prompt");
        ensure!(prompt.len() < self.capacity, "prompt of {} tokens exceeds the context of {}", prompt.len(), self.capacity);
        let budget = p.max_tokens.min(self.capacity - prompt.len());
        let mut run = Run { slot, params: p.clone(), produced: 0, spec: false, pending: 0, rng: Rng::new(p.seed), budget };
        let speculate = speculate && !p.capture;
        self.prompt_site = None;
        let capture = if p.capture { Some(forward::CaptureTarget { pos: prompt.len() - 1, buf: self.capture_buf()? }) } else { None };
        if slot == 0 {
            self.state.reset(s)?;
            self.state.set_steering_strength(s, p.steering_strength)?;
            self.ws.capture = capture;
            let r = self.prefill(model, s, prompt, &p.images, &p.boundaries, p.steering_strength, p.progress.as_ref());
            self.ws.capture = None;
            r?;
            if let Some(c) = capture {
                self.prompt_site = Some(self.read_site(s, "prompt_last", c.buf, c.pos, prompt[c.pos], self.dec.last_logits_addr())?);
            }
            if speculate {
                // Sampled runs draw the first token here; the first round commits it.
                if p.temperature > 0.0 {
                    self.dec.next = self.sample(s, self.dec.last_logits_addr(), p, &mut run.rng)?;
                }
                self.spec.set_sampling(s, sampling(p))?;
                self.spec.restart(&self.state, &self.dec, s)?;
                run.spec = true;
                return Ok((run, Vec::new()));
            }
            let tok = if p.temperature <= 0.0 { self.dec.next } else { self.sample(s, self.dec.last_logits_addr(), p, &mut run.rng)? };
            run.pending = tok;
            return Ok((run, vec![tok]));
        }
        let spans = self.encode_images(s, &p.images, 0)?;
        let aux = self.aux.as_mut().expect("second slot");
        aux.state.reset(s)?;
        aux.state.mm = spans;
        aux.state.set_steering_strength(s, p.steering_strength)?;
        self.ws.capture = capture;
        // The same step boundaries as a fresh slot-0 prefill (chunks, cut at turn boundaries), so
        // both slots compute identical results for the same prompt.
        let (ws, chunk) = (&mut self.ws, self.chunk);
        let r = with_ple_warm(model, prompt, 0, || -> Result<usize> {
            let mut c0 = 0;
            let mut last_rows = 0;
            while c0 < prompt.len() {
                let mut end = (c0 + chunk).min(prompt.len());
                if let Some(&b) = p.boundaries.iter().filter(|&&b| b > c0 && b < prompt.len()).min() {
                    end = end.min(b);
                }
                forward::forward(model, &mut aux.state, ws, s, &prompt[c0..end])?;
                last_rows = end - c0;
                c0 = end;
                if let Some(f) = &p.progress {
                    s.synchronize()?;
                    (f.0)(end, 0);
                }
            }
            Ok(last_rows)
        });
        self.ws.capture = None;
        let last_rows = r?;
        let last = self.ws.out + ((last_rows - 1) * crate::model::D * 2) as u64;
        forward::head_logits(model, &self.hw, s, last)?;
        if let Some(c) = capture {
            self.prompt_site = Some(self.read_site(s, "prompt_last", c.buf, c.pos, prompt[c.pos], self.hw.logits_addr())?);
        }
        let tok = self.sample(s, self.hw.logits_addr(), p, &mut run.rng)?;
        run.pending = tok;
        Ok((run, vec![tok]))
    }

    /// Switches a speculative run to plain steps (when another request starts): commits the deferred
    /// state and returns the target's next token, which is emitted now.
    pub fn to_plain(&mut self, model: &Model, s: &Stream, run: &mut Run) -> Result<Vec<i32>> {
        if !run.spec {
            return Ok(Vec::new());
        }
        forward::flush_spec(model, &mut self.state, &self.ws, s)?;
        run.spec = false;
        run.pending = self.dec.next;
        Ok(vec![self.dec.next])
    }

    /// One step of the given runs (one, or one per slot, both plain); returns each run's new tokens.
    pub fn step(&mut self, model: &Model, s: &Stream, runs: &mut [&mut Run]) -> Result<Vec<Vec<i32>>> {
        match runs {
            [run] if run.spec => {
                let committed = self.spec.round(model, &mut self.state, &mut self.ws, &mut self.dec, s)?;
                self.sync_kv_tokens();
                Ok(vec![committed])
            }
            [run] => {
                if run.slot == 0 {
                    self.graph.step(model, &mut self.state, &mut self.ws, &self.hw, s, run.pending)?;
                    self.sync_kv_tokens();
                } else {
                    let aux = self.aux.as_mut().expect("second slot");
                    aux.graph.step(model, &mut aux.state, &mut self.ws, &self.hw, s, run.pending)?;
                }
                let tok = self.sample(s, self.hw.logits_addr(), &run.params.clone(), &mut run.rng)?;
                run.pending = tok;
                Ok(vec![vec![tok]])
            }
            [a, b] => {
                ensure!(!a.spec && !b.spec && a.slot == 0 && b.slot == 1, "batched steps need both slots in plain mode");
                let aux = self.aux.as_mut().expect("second slot");
                aux.batch.step(model, &mut [&mut self.state, &mut aux.state], &mut self.ws, s, &[a.pending, b.pending])?;
                s.synchronize()?;
                self.sync_kv_tokens();
                let logits = self.aux.as_ref().unwrap().hw2.logits_addr();
                let ta = self.sample(s, logits, &a.params.clone(), &mut a.rng)?;
                let tb = self.sample(s, logits + (VOCAB * 4) as u64, &b.params.clone(), &mut b.rng)?;
                a.pending = ta;
                b.pending = tb;
                Ok(vec![vec![ta], vec![tb]])
            }
            _ => anyhow::bail!("step takes one run, or one per slot"),
        }
    }

    /// Stored contexts (host and disk tiers).
    pub fn stored_contexts(&self) -> usize {
        self.store.len()
    }

    /// Writes host-tier contexts to the disk tier (persistence).
    pub fn persist(&mut self) -> Result<()> {
        if self.kv_strength == 0.0 {
            self.stash(usize::MAX)?;
        }
        self.store.persist()
    }

    /// The live cache pieces holding positions [0, len): position table, then per attention cache (the
    /// target's QSA layers, then the drafter's) K, V, index keys and pooled index groups.
    fn kv_pieces(&self, len: usize) -> Vec<(u64, usize)> {
        use crate::forward::QsaCache;
        let mut v = vec![(self.state.pos3.addr(), len * 3 * 4)];
        let caches: Vec<&QsaCache> = self.state.qsa.iter().flatten().chain(self.state.mtp_qsa.iter()).collect();
        for c in caches {
            let per = |b: &DeviceBuffer, rows: usize, total_rows: usize| (b.addr(), b.bytes() / total_rows * rows);
            let cap = self.state.capacity;
            v.push(per(&c.k, len, cap));
            v.push(per(&c.v, len, cap));
            v.push(per(&c.raw, len, cap));
            v.push(per(&c.pooled, len / 4, cap / 4));
        }
        v
    }

    /// Saves the GPU context at the largest snapshot position (at most `limit`) to the store unless it
    /// is stored already.
    fn stash(&mut self, limit: usize) -> Result<()> {
        if !self.store.enabled() {
            return Ok(());
        }
        let Some(snap) = self.snapshots.iter().filter(|sn| sn.pos <= limit && sn.pos <= self.kv_tokens.len()).max_by_key(|sn| sn.pos) else { return Ok(()) };
        let tokens = self.kv_tokens[..snap.pos].to_vec();
        if self.store.contains(&tokens) {
            return Ok(());
        }
        let rec: usize = Self::pieces(&self.state, self.dec.pending_addr()).iter().map(|p| p.1).sum();
        let kv = self.kv_pieces(snap.pos);
        let total = rec + kv.iter().map(|p| p.1).sum::<usize>();
        let t0 = std::time::Instant::now();
        let buf = squash_cuda::PinnedBuffer::new(total)?;
        let t1 = std::time::Instant::now();
        let s = Stream::new()?;
        unsafe {
            s.copy_d2h(buf.as_ptr(), snap.buf.addr(), rec)?;
            let mut off = rec;
            for (addr, bytes) in kv {
                s.copy_d2h(buf.as_ptr().add(off), addr, bytes)?;
                off += bytes;
            }
        }
        s.synchronize()?;
        let t2 = std::time::Instant::now();
        self.store.insert(tokens, buf)?;
        if std::env::var_os("SQUASH_TIMING").is_some() {
            eprintln!("stash {} tokens, {} MiB: alloc {:.3}s copy {:.3}s insert {:.3}s", snap.pos, total >> 20, (t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), t2.elapsed().as_secs_f64());
        }
        Ok(())
    }

    /// Loads stored entry `i` into the live caches and makes it the GPU context (one snapshot at its
    /// end).
    fn load_entry(&mut self, s: &Stream, i: usize) -> Result<()> {
        let t0 = std::time::Instant::now();
        let rec: usize = Self::pieces(&self.state, self.dec.pending_addr()).iter().map(|p| p.1).sum();
        let (tokens, _) = self.store.get(i)?;
        let kv = self.kv_pieces(tokens.len());
        let snap_buf = match self.snapshots.pop() {
            Some(sn) if sn.buf.bytes() >= rec => sn.buf,
            _ => DeviceBuffer::new(rec)?,
        };
        self.snapshots.clear();
        let (_, data) = self.store.get(i)?;
        unsafe {
            s.copy_h2d(snap_buf.addr(), data.as_ptr(), rec)?;
            let mut off = rec;
            for (addr, bytes) in kv {
                s.copy_h2d(addr, data.as_ptr().add(off), bytes)?;
                off += bytes;
            }
        }
        s.synchronize()?;
        if std::env::var_os("SQUASH_TIMING").is_some() {
            eprintln!("load {} tokens from the store: {:.3}s", tokens.len(), t0.elapsed().as_secs_f64());
        }
        self.snapshots.push(Snapshot { pos: tokens.len(), buf: snap_buf, periodic: false });
        self.kv_tokens = tokens;
        Ok(())
    }

    /// `kv_tokens` for the slot-0 sequence: its tokens with image placeholders keyed by content.
    fn sync_kv_tokens(&mut self) {
        self.kv_tokens = keyed(&self.state.tokens, &self.kv_marks);
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The pieces of the recurrent state, as (device address, bytes); `pending` is the speculative
    /// decoder's pending row.
    fn pieces(state: &SequenceState, pending: u64) -> Vec<(u64, usize)> {
        let mut v: Vec<(u64, usize)> = state.gdn.iter().flatten().flat_map(|g| [(g.conv.addr(), g.conv.bytes()), (g.recurrent.addr(), g.recurrent.bytes())]).collect();
        v.push((state.ple_conv.addr(), state.ple_conv.bytes()));
        v.push((pending, crate::model::H * 4));
        v
    }

    fn snapshot_bytes(state: &SequenceState, pending: u64) -> usize {
        Self::pieces(state, pending).iter().map(|p| p.1).sum()
    }

    fn save_snapshot(state: &SequenceState, pending: u64, s: &Stream, pos: usize, recycled: Option<DeviceBuffer>) -> Result<Snapshot> {
        let pieces = Self::pieces(state, pending);
        let total: usize = pieces.iter().map(|p| p.1).sum();
        let buf = match recycled {
            Some(b) if b.bytes() >= total => b,
            _ => DeviceBuffer::new(total)?,
        };
        let mut off = 0u64;
        for (addr, bytes) in pieces {
            s.copy_d2d(buf.addr() + off, addr, bytes)?;
            off += bytes as u64;
        }
        Ok(Snapshot { pos, buf, periodic: false })
    }

    fn restore_snapshot(&self, s: &Stream, snap: &Snapshot) -> Result<()> {
        let mut off = 0u64;
        for (addr, bytes) in Self::pieces(&self.state, self.dec.pending_addr()) {
            s.copy_d2d(addr, snap.buf.addr() + off, bytes)?;
            off += bytes as u64;
        }
        Ok(())
    }

    /// Positions the state, target and MTP caches on the longest snapshot whose prefix the caches still
    /// hold, prefills the rest of `prompt`, and takes new snapshots at `boundaries`.
    fn prefill(&mut self, model: &Model, s: &Stream, prompt: &[i32], images: &[std::sync::Arc<ImageInput>], boundaries: &[usize], strength: f32, progress: Option<&Progress>) -> Result<()> {
        // Steered state is never reused across strengths, and the store holds unsteered state only.
        if self.kv_strength != strength {
            self.kv_tokens.clear();
            self.snapshots.clear();
        }
        let marks = image_marks(images);
        let key = keyed(prompt, &marks);
        let mut lcp = self.kv_tokens.iter().zip(&key).take_while(|(a, b)| a == b).count();
        if strength == 0.0 && self.store.enabled() {
            // The new prompt overwrites the cached conversation past `lcp`: keep its boundary state.
            if self.snapshots.iter().any(|sn| sn.pos > lcp) {
                self.stash(usize::MAX)?;
            }
            let gpu_best = self.snapshots.iter().filter(|sn| sn.pos <= lcp && sn.pos < prompt.len()).map(|sn| sn.pos).max().unwrap_or(0);
            if let Some((i, len)) = self.store.best(&key) {
                if len > gpu_best {
                    self.load_entry(s, i)?;
                    lcp = len;
                }
            }
        }
        self.kv_strength = strength;
        // Snapshots past the common prefix describe tokens the new prompt overwrites.
        self.snapshots.retain(|sn| sn.pos <= lcp);
        self.state.reset(s)?;
        self.kv_marks = marks;
        let best = self.snapshots.iter().enumerate().filter(|(_, sn)| sn.pos < prompt.len()).max_by_key(|(_, sn)| sn.pos).map(|(i, _)| i);
        let start = match best {
            Some(i) => {
                let snap = self.snapshots.remove(i);
                self.restore_snapshot(s, &snap)?;
                let pos = snap.pos;
                self.snapshots.push(snap); // most recently used last
                self.state.tokens = prompt[..pos].to_vec();
                pos
            }
            None => {
                s.memset(self.dec.pending_addr(), 0, crate::model::H * 4)?;
                0
            }
        };
        self.reused = start;
        // Images are encoded only from the restored position on.
        self.state.mm = self.encode_images(s, images, start)?;
        if let Some(f) = progress {
            (f.0)(start, start);
        }
        let report = progress.map(|f| move |pos: usize| (f.0)(pos, start));
        let mut pauses: Vec<usize> = boundaries.iter().copied().filter(|&b| b > start && b < prompt.len() && !self.snapshots.iter().any(|sn| sn.pos == b)).collect();
        // Periodic snapshots on chunk boundaries (no extra prefill steps), at most PERIODIC_SNAPSHOTS
        // per prompt, so a prefix shared up to any point is reusable to within one interval of it.
        let interval = self.chunk * prompt.len().div_ceil(self.chunk * PERIODIC_SNAPSHOTS).max(1);
        let periodic: Vec<usize> = (1..).map(|k| k * interval).take_while(|&p| p < prompt.len()).filter(|&p| p > start && !pauses.contains(&p) && !self.snapshots.iter().any(|sn| sn.pos == p)).collect();
        pauses.extend(&periodic);
        pauses.sort_unstable();
        pauses.dedup();
        let snapshots = &mut self.snapshots;
        let (dec, state, ws, chunk) = (&mut self.dec, &mut self.state, &mut self.ws, self.chunk);
        with_ple_warm(model, prompt, start, || dec.prefill_from(model, state, ws, s, prompt, start, chunk, &pauses, |state, pos, pending| {
            // Recycle the oldest periodic snapshot first, then the oldest. Snapshots only speed up later
            // prompts: when device memory is short, an older one is recycled, or none is taken.
            let oldest = |snapshots: &mut Vec<Snapshot>| {
                let i = snapshots.iter().position(|sn| sn.periodic).unwrap_or(0);
                snapshots.remove(i).buf
            };
            let buf = if snapshots.len() >= MAX_SNAPSHOTS {
                Some(oldest(snapshots))
            } else {
                // Leave headroom for per-request buffers (vision encoder, sampling).
                let bytes = Self::snapshot_bytes(state, pending);
                let room = squash_cuda::mem_info().map_or(false, |(free, _)| free >= bytes + SNAPSHOT_HEADROOM);
                match if room { DeviceBuffer::new(bytes) } else { Err(anyhow::anyhow!("under {} MiB free", SNAPSHOT_HEADROOM >> 20)) } {
                    Ok(b) => Some(b),
                    Err(_) if !snapshots.is_empty() => Some(oldest(snapshots)),
                    Err(e) => {
                        eprintln!("snapshot at {pos} skipped: {e:#}");
                        None
                    }
                }
            };
            if let Some(buf) = buf {
                let mut snap = Self::save_snapshot(state, pending, s, pos, Some(buf))?;
                snap.periodic = periodic.contains(&pos);
                snapshots.push(snap);
            }
            Ok(())
        }, report.as_ref().map(|f| f as &(dyn Fn(usize) + Send + Sync))))?;
        self.kv_tokens = key;
        Ok(())
    }

    /// Generates from `prompt`, calling `on_token` for every generated token (return false to stop).
    pub fn generate(&mut self, model: &Model, s: &Stream, prompt: &[i32], p: &GenParams, mut on_token: impl FnMut(i32) -> bool) -> Result<(Finish, usize)> {
        ensure!(!prompt.is_empty(), "empty prompt");
        ensure!(prompt.len() < self.capacity, "prompt of {} tokens exceeds the context of {}", prompt.len(), self.capacity);
        self.state.reset(s)?;
        self.state.set_steering_strength(s, p.steering_strength)?;
        let budget = p.max_tokens.min(self.capacity - prompt.len());
        let mut produced = 0usize;
        let mut emit = |tok: i32, produced: &mut usize| -> Option<Finish> {
            *produced += 1;
            if !on_token(tok) {
                return Some(Finish::Cancelled);
            }
            if p.stop_tokens.contains(&tok) {
                return Some(Finish::Stop);
            }
            if *produced >= budget {
                return Some(Finish::Length);
            }
            None
        };
        if budget == 0 {
            return Ok((Finish::Length, 0));
        }
        self.prefill(model, s, prompt, &p.images, &p.boundaries, p.steering_strength, p.progress.as_ref())?;
        if p.temperature <= 0.0 || !p.capture {
            if p.temperature > 0.0 {
                let mut rng = Rng::new(p.seed);
                self.dec.next = self.sample(s, self.dec.last_logits_addr(), p, &mut rng)?;
            }
            self.spec.set_sampling(s, sampling(p))?;
            self.spec.restart(&self.state, &self.dec, s)?;
            // The target's first token, then speculative rounds (each commits the previous `next`
            // and the accepted drafts; the round's own `next` is emitted by the following round).
            loop {
                let committed = self.spec.round(model, &mut self.state, &mut self.ws, &mut self.dec, s)?;
                for &tok in &committed {
                    if let Some(f) = emit(tok, &mut produced) {
                        self.sync_kv_tokens();
                        return Ok((f, produced));
                    }
                }
                if self.state.len() + 2 * (self.dec.drafts + 2) >= self.state.capacity {
                    self.sync_kv_tokens();
                    return Ok((Finish::Length, produced));
                }
            }
        }
        // Sampled: the first token from the last prompt row's logits, then one token per step.
        let mut rng = Rng::new(p.seed);
        let mut tok = self.sample(s, self.dec.last_logits_addr(), p, &mut rng)?;
        loop {
            if let Some(f) = emit(tok, &mut produced) {
                self.sync_kv_tokens();
                return Ok((f, produced));
            }
            self.graph.step(model, &mut self.state, &mut self.ws, &self.hw, s, tok)?;
            tok = self.sample(s, self.hw.logits_addr(), p, &mut rng)?;
        }
    }

    /// Samples from a logits row (VOCAB f32 at `logits`): top-k, temperature, top-p.
    fn sample(&self, s: &Stream, logits: u64, p: &GenParams, rng: &mut Rng) -> Result<i32> {
        let k = if p.temperature <= 0.0 { 1 } else if p.top_k == 0 { MAX_TOP_K } else { p.top_k.min(MAX_TOP_K) };
        squash_cuda::kernels::topk(s, logits, 1, VOCAB, VOCAB, k, false, self.top_idx.addr(), self.top_val.addr())?;
        let ids: Vec<i32> = s.download(self.top_idx.addr(), k)?;
        let vals: Vec<f32> = s.download(self.top_val.addr(), k)?;
        let t = p.temperature.max(1e-4);
        let m = vals[0];
        let mut probs: Vec<f64> = vals.iter().map(|v| (((v - m) / t) as f64).exp()).collect();
        let total: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|x| *x /= total);
        let mut keep = k;
        if p.top_p < 1.0 {
            let mut acc = 0.0;
            for (i, x) in probs.iter().enumerate() {
                acc += x;
                if acc >= p.top_p as f64 {
                    keep = i + 1;
                    break;
                }
            }
        }
        // min_p: a prefix of the sorted list too (ratios to the first are unchanged by renormalizing).
        keep = keep.min(probs.iter().take_while(|&&x| x >= p.min_p as f64 * probs[0]).count().max(1));
        let mass: f64 = probs[..keep].iter().sum();
        let mut r = rng.next() * mass;
        for i in 0..keep {
            r -= probs[i];
            if r <= 0.0 {
                return Ok(ids[i]);
            }
        }
        Ok(ids[keep - 1])
    }
}

/// Runs `f` (a prefill of `prompt[start..]`) while threads fault the prompt's PLE rows into the page
/// cache ahead of it.
fn with_ple_warm<R>(model: &Model, prompt: &[i32], start: usize, f: impl FnOnce() -> R) -> R {
    let threads = std::env::var("SQUASH_PLE_WARM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    if prompt.len() - start < 256 || threads == 0 {
        return f();
    }
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|sc| {
        sc.spawn(|| forward::warm_ple(model, prompt, start, threads, &stop));
        let r = f();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        r
    })
}

/// The speculative rounds' sampling parameters for a request (None: greedy).
fn sampling(p: &GenParams) -> Option<crate::spec::Sampling> {
    (p.temperature > 0.0).then(|| crate::spec::Sampling { temperature: p.temperature, top_p: p.top_p, top_k: p.top_k, min_p: p.min_p, seed: p.seed ^ 0x5bd1_e995_7f4a_7c15 })
}

struct Rng(u64);
impl Rng {
    /// A generator whose stream depends on every bit of `seed` (xorshift alone, seeded with nearby
    /// values, starts with nearly identical outputs).
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Self((z ^ (z >> 31)) | 1)
    }

    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}
