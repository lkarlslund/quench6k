//! Neuron atlases: activation statistics of every expert neuron over a corpus (an uploaded text, a
//! file under `--atlas-dir`, or text the model writes about a topic), and queries over them: neurons
//! ranked by how strongly they fire (against a baseline atlas: how much more), experts by how often
//! they are routed, and each neuron's strongest example in context.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use quench6k_engine::atlas::{NeuronId, Recorder, Stats};
use quench6k_engine::inspect::Pushes;
use quench6k_engine::model::{EXPERT_WIDTH, EXPERTS, LAYERS};
use quench6k_engine::session::{GenParams, Prefill, Session, Variant};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{mpsc as tmpsc, oneshot};

use super::{App, Msg, Work, error, prepare};

/// Corpus tokens prefilled per sequence: documents are cut into segments of at most this many tokens,
/// each computed from position 0.
const SEGMENT: usize = 2048;
/// Atlases kept (each holds about 190 MB of statistics).
const MAX_ATLASES: usize = 6;

/// A corpus scan for the inference thread: its segments are prefilled one at a time with the recorder
/// attached.
pub(super) struct ScanJob {
    tokens: Arc<Vec<i32>>,
    /// Corpus token ranges.
    pub segments: Vec<(usize, usize)>,
    pub next: usize,
    pub variant: Variant,
    params: GenParams,
    pub recorder: Option<Arc<Recorder>>,
    progress: Arc<AtomicUsize>,
    /// Keeps the segments' prefill channel open (nothing is sent on it).
    pub keep: (tmpsc::UnboundedSender<Msg>, tmpsc::UnboundedReceiver<Msg>),
    pub tx: oneshot::Sender<Result<Stats, String>>,
}

impl ScanJob {
    pub fn segment_len(&self) -> usize {
        let (a, b) = self.segments[self.next];
        b - a
    }

    /// Starts the next segment on `slot`.
    pub fn start(
        &mut self,
        session: &mut Session,
        s: &quench6k_cuda::Stream,
        slot: usize,
    ) -> anyhow::Result<Prefill> {
        if self.recorder.is_none() {
            self.recorder = Some(Arc::new(Recorder::new()?));
        }
        let (a, b) = self.segments[self.next];
        let mut p = self.params.clone();
        p.record = Some((self.recorder.clone().unwrap(), a));
        let pf = session.start(s, slot, &self.tokens[a..b], &p, false)?;
        self.next += 1;
        Ok(pf)
    }

    pub fn done(&self, tokens: usize) {
        self.progress.fetch_add(tokens, Ordering::Relaxed);
    }
}

pub(super) struct LensJob {
    pub neurons: Vec<NeuronId>,
    pub tx: oneshot::Sender<Result<Vec<Pushes>, String>>,
}

#[derive(Default)]
struct Corpus {
    tokens: Arc<Vec<i32>>,
    /// Start of each document.
    docs: Vec<usize>,
}

pub(super) struct Atlas {
    name: String,
    source: String,
    model: String,
    created: u64,
    /// "generating", "scanning", "ready" or "failed: ...".
    phase: Mutex<String>,
    /// Tokens generated or scanned so far, of `total`.
    progress: Arc<AtomicUsize>,
    total: AtomicUsize,
    corpus: Mutex<Corpus>,
    stats: OnceLock<Stats>,
}

impl Atlas {
    fn summary(&self) -> Value {
        json!({
            "name": self.name, "source": self.source, "model": self.model, "created": self.created,
            "phase": *self.phase.lock().unwrap(), "progress": self.progress.load(Ordering::Relaxed),
            "total": self.total.load(Ordering::Relaxed),
            "tokens": self.corpus.lock().unwrap().tokens.len(),
        })
    }

    fn fail(&self, e: impl std::fmt::Display) {
        *self.phase.lock().unwrap() = format!("failed: {e}");
    }

    /// Corpus tokens around `token` in its document: (ids, index of `token` in them).
    fn context(&self, token: usize) -> (Vec<i32>, usize) {
        let c = self.corpus.lock().unwrap();
        let d = c.docs.partition_point(|&s| s <= token).saturating_sub(1);
        let (start, end) = (
            c.docs[d],
            c.docs.get(d + 1).copied().unwrap_or(c.tokens.len()),
        );
        let a = token.saturating_sub(24).max(start);
        let b = (token + 9).min(end);
        (c.tokens[a..b].to_vec(), token - a)
    }
}

pub(super) type Atlases = Mutex<Vec<Arc<Atlas>>>;

fn find(app: &App, name: &str) -> Result<Arc<Atlas>, Response> {
    app.atlases
        .lock()
        .unwrap()
        .iter()
        .find(|a| a.name == name)
        .cloned()
        .ok_or_else(|| error(StatusCode::NOT_FOUND, format!("no atlas {name:?}")).into_response())
}

fn ready(app: &App, name: &str) -> Result<Arc<Atlas>, Response> {
    let a = find(app, name)?;
    if a.stats.get().is_none() {
        return Err(
            error(StatusCode::CONFLICT, format!("atlas {name:?} is not ready")).into_response(),
        );
    }
    Ok(a)
}

/// `GET /v1/atlas`: the atlases and their progress.
pub(super) async fn list(State(app): State<Arc<App>>) -> Json<Value> {
    let list: Vec<Value> = app
        .atlases
        .lock()
        .unwrap()
        .iter()
        .map(|a| a.summary())
        .collect();
    Json(json!({"data": list, "atlas_dir": app.atlas_dir.is_some()}))
}

/// `DELETE /v1/atlas/{name}`.
pub(super) async fn delete(State(app): State<Arc<App>>, Path(name): Path<String>) -> Response {
    let mut list = app.atlases.lock().unwrap();
    let n = list.len();
    list.retain(|a| a.name != name);
    if list.len() == n {
        return error(StatusCode::NOT_FOUND, format!("no atlas {name:?}")).into_response();
    }
    Json(json!({"deleted": name})).into_response()
}

/// `POST /v1/atlas`: builds an atlas named "name" from "text", a "file" under --atlas-dir, or "topic"
/// (the model writes "samples" texts of up to "max_tokens" about it, at "temperature", default 1).
/// "model" picks the variant.
pub(super) async fn create(State(app): State<Arc<App>>, Json(req): Json<Value>) -> Response {
    if app.flint {
        return error(
            StatusCode::BAD_REQUEST,
            "atlases are not supported for this model",
        )
        .into_response();
    }
    let bad = |m: String| error(StatusCode::BAD_REQUEST, m).into_response();
    let name = req
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if name.is_empty()
        || name.len() > 64
        || !name
            .chars()
            .all(|c| c.is_alphanumeric() || "-_. ".contains(c))
    {
        return bad("name: up to 64 letters, digits, spaces and -_.".into());
    }
    // The variant (and a check of the request's options) from an empty chat.
    let probe = json!({"model": req.get("model"), "messages": [{"role": "user", "content": "."}]});
    let base = match prepare(&app, &probe) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    let source = if let Some(t) = req.get("text").and_then(Value::as_str) {
        Source::Text(
            t.to_string(),
            req.get("filename")
                .and_then(Value::as_str)
                .unwrap_or("uploaded text")
                .to_string(),
        )
    } else if let Some(f) = req.get("file").and_then(Value::as_str) {
        let Some(dir) = &app.atlas_dir else {
            return bad("reading server files needs --atlas-dir; upload the text instead".into());
        };
        let path = match dir.join(f).canonicalize() {
            Ok(p) if p.starts_with(dir) => p,
            _ => return bad(format!("no file {f:?} in the atlas directory")),
        };
        match std::fs::read_to_string(&path) {
            Ok(t) => Source::Text(t, f.to_string()),
            Err(e) => return bad(format!("{f}: {e}")),
        }
    } else if let Some(topic) = req
        .get("topic")
        .and_then(Value::as_str)
        .filter(|t| !t.trim().is_empty())
    {
        let samples = req
            .get("samples")
            .and_then(Value::as_u64)
            .unwrap_or(16)
            .clamp(1, 256) as usize;
        let max_tokens = req
            .get("max_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(512)
            .clamp(16, 8192) as usize;
        let temperature = req
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        Source::Topic {
            topic: topic.to_string(),
            samples,
            max_tokens,
            temperature,
        }
    } else {
        return bad("give \"text\", \"file\" or \"topic\"".into());
    };
    let atlas = Arc::new(Atlas {
        source: match &source {
            Source::Text(_, label) => label.clone(),
            Source::Topic { topic, samples, .. } => format!("{samples} texts about: {topic}"),
        },
        name: name.clone(),
        model: base.model.clone(),
        created: super::now(),
        phase: Mutex::new("generating".into()),
        progress: Arc::new(AtomicUsize::new(0)),
        total: AtomicUsize::new(0),
        corpus: Mutex::new(Corpus::default()),
        stats: OnceLock::new(),
    });
    {
        let mut list = app.atlases.lock().unwrap();
        if list.iter().any(|a| a.name == name) {
            return error(StatusCode::CONFLICT, format!("atlas {name:?} exists")).into_response();
        }
        if list.len() >= MAX_ATLASES {
            return error(
                StatusCode::CONFLICT,
                format!("at most {MAX_ATLASES} atlases; delete one first"),
            )
            .into_response();
        }
        list.push(atlas.clone());
    }
    tokio::spawn(build(app.clone(), atlas.clone(), source, base.params));
    (StatusCode::ACCEPTED, Json(atlas.summary())).into_response()
}

enum Source {
    Text(String, String),
    Topic {
        topic: String,
        samples: usize,
        max_tokens: usize,
        temperature: f64,
    },
}

async fn build(app: Arc<App>, atlas: Arc<Atlas>, source: Source, params: GenParams) {
    let docs: Vec<Vec<i32>> = match source {
        Source::Text(text, _) => {
            // Documents: paragraphs (blank-line separated), so segments start at a paragraph.
            let mut docs = Vec::new();
            for para in text.split("\n\n").filter(|p| !p.trim().is_empty()) {
                match app.chat.encode(para) {
                    Ok(ids) if !ids.is_empty() => docs.push(ids),
                    Ok(_) => {}
                    Err(e) => return atlas.fail(e),
                }
            }
            docs
        }
        Source::Topic {
            topic,
            samples,
            max_tokens,
            temperature,
        } => {
            atlas.total.store(samples * max_tokens, Ordering::Relaxed);
            let req = json!({"model": atlas.model, "messages": [{"role": "user", "content": topic}],
                             "max_tokens": max_tokens, "temperature": temperature,
                             "chat_template_kwargs": {"enable_thinking": false}});
            let mut runs = Vec::new();
            for i in 0..samples {
                let mut r = req.clone();
                r["seed"] = json!(i as u64 + super::now());
                let p = match prepare(&app, &r) {
                    Ok(p) => p,
                    Err(e) => return atlas.fail(e.message),
                };
                let (tx, rx) = tmpsc::unbounded_channel();
                let job = super::Job {
                    ids: p.ids,
                    params: p.params,
                    mtp: true,
                    tx,
                    capture: None,
                };
                if app.enqueue(job).is_err() {
                    return atlas.fail("worker stopped");
                }
                let progress = atlas.progress.clone();
                runs.push(tokio::spawn(async move {
                    let mut rx = rx;
                    let mut out = Vec::new();
                    while let Some(m) = rx.recv().await {
                        match m {
                            Msg::Token(t) => {
                                out.push(t);
                                progress.fetch_add(1, Ordering::Relaxed);
                            }
                            Msg::Progress(..) => {}
                            Msg::Done(..) => break,
                            Msg::Error(e) => return Err(e),
                        }
                    }
                    Ok(out)
                }));
            }
            let mut docs = Vec::new();
            for r in runs {
                match r.await {
                    Ok(Ok(mut d)) => {
                        // Drop the end-of-turn token.
                        if d.last().is_some_and(|t| app.chat.stop_tokens.contains(t)) {
                            d.pop();
                        }
                        if !d.is_empty() {
                            docs.push(d);
                        }
                    }
                    Ok(Err(e)) => return atlas.fail(e),
                    Err(e) => return atlas.fail(e),
                }
            }
            docs
        }
    };
    let mut tokens = Vec::new();
    let mut starts = Vec::new();
    let mut segments = Vec::new();
    for d in &docs {
        starts.push(tokens.len());
        for chunk in d.chunks(SEGMENT) {
            segments.push((tokens.len(), tokens.len() + chunk.len()));
            tokens.extend_from_slice(chunk);
        }
    }
    if tokens.is_empty() {
        return atlas.fail("empty corpus");
    }
    let tokens = Arc::new(tokens);
    *atlas.corpus.lock().unwrap() = Corpus {
        tokens: tokens.clone(),
        docs: starts,
    };
    *atlas.phase.lock().unwrap() = "scanning".into();
    atlas.progress.store(0, Ordering::Relaxed);
    atlas.total.store(tokens.len(), Ordering::Relaxed);
    let mut p = params;
    p.max_tokens = 1;
    p.temperature = 0.0;
    p.boundaries.clear();
    p.fresh = true;
    let (tx, rx) = oneshot::channel();
    let job = ScanJob {
        tokens,
        segments,
        next: 0,
        variant: p.variant(),
        params: p,
        recorder: None,
        progress: atlas.progress.clone(),
        keep: tmpsc::unbounded_channel(),
        tx,
    };
    if app.jobs.send(Work::Scan(job)).is_err() {
        return atlas.fail("worker stopped");
    }
    match rx.await {
        Ok(Ok(stats)) => {
            let _ = atlas.stats.set(stats);
            *atlas.phase.lock().unwrap() = "ready".into();
        }
        Ok(Err(e)) => atlas.fail(e),
        Err(_) => atlas.fail("worker stopped"),
    }
}

#[derive(Deserialize)]
pub(super) struct RankQuery {
    /// A baseline atlas: rank by the difference.
    base: Option<String>,
    /// "mean" (|activation| per corpus token, default) or "peak" (strongest activation).
    by: Option<String>,
    limit: Option<usize>,
    layer: Option<usize>,
}

fn name_of((layer, expert, unit): NeuronId) -> Value {
    json!({"layer": layer, "expert": expert, "unit": unit})
}

/// Token texts of `ids`, keyed by id.
fn texts(app: &App, ids: impl IntoIterator<Item = i32>) -> serde_json::Map<String, Value> {
    let set: BTreeSet<i32> = ids.into_iter().collect();
    set.into_iter()
        .map(|id| {
            (
                id.to_string(),
                json!(app.chat.decode(&[id], false).unwrap_or_default()),
            )
        })
        .collect()
}

/// `GET /v1/atlas/{name}/neurons`: neurons ranked by how strongly they fire on the corpus (against
/// `base`: how much more than on the baseline), each with its strongest example.
pub(super) async fn neurons(
    State(app): State<Arc<App>>,
    Path(name): Path<String>,
    Query(q): Query<RankQuery>,
) -> Response {
    let atlas = match ready(&app, &name) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let base = match q
        .base
        .as_deref()
        .filter(|b| !b.is_empty())
        .map(|b| ready(&app, b))
    {
        Some(Err(r)) => return r,
        Some(Ok(b)) => Some(b),
        None => None,
    };
    let peak = q.by.as_deref() == Some("peak");
    let limit = q.limit.unwrap_or(50).clamp(1, 500);
    let stats = atlas.stats.get().unwrap();
    let n = atlas.corpus.lock().unwrap().tokens.len().max(1) as f32;
    let base_stats = base.as_ref().map(|b| {
        (
            b.stats.get().unwrap(),
            b.corpus.lock().unwrap().tokens.len().max(1) as f32,
        )
    });
    let value = |st: &Stats, tokens: f32, id: NeuronId| -> f32 {
        if peak {
            st.peak(id).map_or(0.0, |p| p.value.abs())
        } else {
            st.sum(id) / tokens
        }
    };
    let mut scored: Vec<(f32, NeuronId)> = Stats::neurons()
        .filter(|id| q.layer.is_none_or(|l| l == id.0))
        .map(|id| {
            let v = value(stats, n, id) - base_stats.map_or(0.0, |(b, bn)| value(b, bn, id));
            (v, id)
        })
        .collect();
    let k = limit.min(scored.len());
    if k < scored.len() {
        scored.select_nth_unstable_by(k, |a, b| b.0.total_cmp(&a.0));
        scored.truncate(k);
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut ids = Vec::new();
    let rows: Vec<Value> = scored
        .iter()
        .map(|&(score, id)| {
            let mut row = name_of(id);
            row["score"] = json!(score);
            row["mean"] = json!(stats.sum(id) / n);
            row["routed"] = json!(id.1.map_or(n as u32, |e| stats.routed(id.0, e)));
            if let Some((b, bn)) = base_stats {
                row["base_mean"] = json!(b.sum(id) / bn);
                row["base_peak"] = json!(b.peak(id).map(|p| p.value));
            }
            if let Some(p) = stats.peak(id) {
                let (ctx, at) = atlas.context(p.token);
                ids.extend_from_slice(&ctx);
                row["peak"] = json!({"value": p.value, "context": ctx, "at": at});
            }
            row
        })
        .collect();
    Json(json!({"atlas": name, "base": q.base, "by": if peak { "peak" } else { "mean" }, "tokens": n as usize,
                "neurons": rows, "texts": texts(&app, ids)}))
    .into_response()
}

/// `GET /v1/atlas/{name}/experts`: experts ranked by how often they are routed (against `base`: how
/// much more often than on the baseline).
pub(super) async fn experts(
    State(app): State<Arc<App>>,
    Path(name): Path<String>,
    Query(q): Query<RankQuery>,
) -> Response {
    let atlas = match ready(&app, &name) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let base = match q
        .base
        .as_deref()
        .filter(|b| !b.is_empty())
        .map(|b| ready(&app, b))
    {
        Some(Err(r)) => return r,
        Some(Ok(b)) => Some(b),
        None => None,
    };
    let stats = atlas.stats.get().unwrap();
    let n = atlas.corpus.lock().unwrap().tokens.len().max(1) as f32;
    let base_stats = base.as_ref().map(|b| {
        (
            b.stats.get().unwrap(),
            b.corpus.lock().unwrap().tokens.len().max(1) as f32,
        )
    });
    let mut rows: Vec<(f32, usize, usize, f32, Option<f32>)> = (0..LAYERS)
        .filter(|l| q.layer.is_none_or(|x| x == *l))
        .flat_map(|l| (0..EXPERTS).map(move |e| (l, e)))
        .map(|(l, e)| {
            let rate = stats.routed(l, e) as f32 / n;
            let base_rate = base_stats.map(|(b, bn)| b.routed(l, e) as f32 / bn);
            (rate - base_rate.unwrap_or(0.0), l, e, rate, base_rate)
        })
        .collect();
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    rows.truncate(q.limit.unwrap_or(50).clamp(1, 500));
    let rows: Vec<Value> = rows
        .iter()
        .map(|&(score, l, e, rate, base_rate)| json!({"layer": l, "expert": e, "score": score, "rate": rate, "base_rate": base_rate}))
        .collect();
    Json(json!({"atlas": name, "base": q.base, "experts": rows})).into_response()
}

/// `POST /v1/lens`: what neurons write toward ("neurons": [{"layer", "expert" or "shared": true,
/// "unit"}]), as tokens their output raises and lowers.
pub(super) async fn lens(State(app): State<Arc<App>>, Json(req): Json<Value>) -> Response {
    let Some(list) = req
        .get("neurons")
        .and_then(Value::as_array)
        .filter(|l| !l.is_empty() && l.len() <= 64)
    else {
        return error(StatusCode::BAD_REQUEST, "neurons: a list of 1 to 64").into_response();
    };
    let mut neurons = Vec::new();
    for v in list {
        let int = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x as usize);
        let (Some(layer), Some(unit)) = (int("layer"), int("unit")) else {
            return error(
                StatusCode::BAD_REQUEST,
                "each neuron needs a layer and a unit",
            )
            .into_response();
        };
        if layer >= LAYERS || unit >= EXPERT_WIDTH {
            return error(StatusCode::BAD_REQUEST, "neuron out of range").into_response();
        }
        neurons.push((layer, int("expert"), unit));
    }
    let (tx, rx) = oneshot::channel();
    if app
        .jobs
        .send(Work::Lens(LensJob {
            neurons: neurons.clone(),
            tx,
        }))
        .is_err()
    {
        return error(StatusCode::SERVICE_UNAVAILABLE, "worker stopped").into_response();
    }
    match rx.await {
        Ok(Ok(pushes)) => {
            let ids = pushes
                .iter()
                .flat_map(|p| p.up.iter().chain(&p.down).map(|t| t.id))
                .collect::<Vec<_>>();
            let rows: Vec<Value> = neurons
                .iter()
                .zip(&pushes)
                .map(|(&id, p)| {
                    let mut row = name_of(id);
                    row["pushes"] = json!(p);
                    row
                })
                .collect();
            Json(json!({"neurons": rows, "texts": texts(&app, ids)})).into_response()
        }
        Ok(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
        Err(_) => error(StatusCode::SERVICE_UNAVAILABLE, "worker stopped").into_response(),
    }
}
