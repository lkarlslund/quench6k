//! `quench6k serve-models`: one endpoint for several models, loaded on demand. Nothing loads at start;
//! `/v1/models` lists every model of the catalog (answered without loading). A request names its model
//! (the body's "model"): when another model is loaded, the server switches only once that model has been
//! idle (no request running) for the settle period; until then the request gets an error (503 with
//! Retry-After), so two clients asking for different models cannot make it load and unload in turns.
//! Reads (GET) never load a model.
//!
//! Each model runs as its own `quench6k serve` child process on a local port (ending it frees all of its
//! GPU memory); requests are passed through to it, streams included.
//!
//! The catalog is a JSON file:
//!
//! ```json
//! {"settle_secs": 60,
//!  "models": [
//!    {"args": ["--model", "/models/a", "--name", "a", "--context", "131072", "--kv", "fp8"]},
//!    {"args": ["--model", "/models/b", "--name", "b"], "env": {"QUENCH6K_INK_GUESS": "1"}}
//!  ]}
//! ```
//!
//! `args` are `quench6k serve` arguments without `--listen`; a model's names (its `--name`, plus the
//! adapted or steered name with `--adapter` or `--steering`) and context come from them. A model's server
//! may serve less context than its `--context` (what one sequence fits in device memory): once it has
//! loaded, its `/v1/models` context is listed instead, and remembered across restarts in a state file
//! (`$XDG_STATE_HOME/quench6k/serve-models-contexts.json`), keyed by the entry's args and env.
//!
//! By hand: `POST /v1/load {"model": name}` loads a model without waiting for the settle period (only
//! while no request is running), `POST /v1/unload` stops the loaded one; the loaded model's dashboard
//! has a model picker for both, and `/` shows the picker alone while no model is loaded.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use axum::Router;
use axum::body::Body;
use axum::extract::{Path, Request, State};
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use serde_json::{Value, json};
use tokio::sync::{Mutex, watch};

pub struct Options {
    pub catalog: PathBuf,
    pub listen: SocketAddr,
    /// Overrides the catalog's settle_secs.
    pub settle: Option<f64>,
}

/// One catalog model: its serve arguments and what they serve.
struct Entry {
    args: Vec<String>,
    env: HashMap<String, String>,
    names: Vec<String>,
    /// The context listed: the catalog's, or (learned) the one its server serves.
    context: AtomicUsize,
    /// What learned contexts are stored under ([`entry_key`]).
    key: String,
}

/// The running (or starting) child.
struct Backend {
    entry: usize,
    child: Option<tokio::process::Child>,
    /// None while loading; Ok(port) when serving; Err when the load failed.
    status: watch::Receiver<Option<std::result::Result<u16, String>>>,
}

struct Shared {
    backend: Option<Backend>,
    /// Requests passed to the backend and not finished.
    running: usize,
    /// When the last request to the backend started or ended.
    active: Instant,
}

struct App {
    entries: Vec<Entry>,
    /// Where learned contexts are kept (none without a state directory).
    state: Option<PathBuf>,
    settle: Duration,
    exe: PathBuf,
    started: u64,
    shared: Mutex<Shared>,
    client: hyper_util::client::legacy::Client<hyper_util::client::legacy::connect::HttpConnector, Body>,
}

pub fn run(opts: Options, identity: impl Fn(&[String]) -> Result<(Vec<String>, usize)>) -> Result<()> {
    let text = std::fs::read_to_string(&opts.catalog).with_context(|| format!("reading {}", opts.catalog.display()))?;
    let cat: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", opts.catalog.display()))?;
    let settle = opts.settle.or_else(|| cat["settle_secs"].as_f64()).unwrap_or(60.0);
    let mut entries = Vec::new();
    for m in cat["models"].as_array().context("catalog: no \"models\" array")? {
        let args: Vec<String> = m["args"]
            .as_array()
            .context("catalog: a model without \"args\"")?
            .iter()
            .map(|a| a.as_str().map(str::to_string).context("catalog: args must be strings"))
            .collect::<Result<_>>()?;
        if args.iter().any(|a| a == "--listen" || a.starts_with("--listen=")) {
            bail!("catalog: --listen is chosen by serve-models");
        }
        let env = m["env"]
            .as_object()
            .map(|o| o.iter().filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string()))).collect())
            .unwrap_or_default();
        let (names, context) = identity(&args)?;
        let key = entry_key(&args, &env);
        entries.push(Entry { args, env, names, context: AtomicUsize::new(context), key });
    }
    let state = state_file(std::env::var_os("XDG_STATE_HOME"), std::env::var_os("HOME"));
    if let Some(path) = &state {
        apply_learned(&entries, &read_learned(path));
    }
    let mut seen = std::collections::HashSet::new();
    for n in entries.iter().flat_map(|e| &e.names) {
        if !seen.insert(n.clone()) {
            bail!("catalog: model name {n} appears twice");
        }
    }
    for e in &entries {
        println!("model {} (context {})", e.names.join(", "), e.context.load(Ordering::Relaxed));
    }
    let app = Arc::new(App {
        entries,
        state,
        settle: Duration::from_secs_f64(settle),
        exe: std::env::current_exe()?,
        started: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()),
        shared: Mutex::new(Shared { backend: None, running: 0, active: Instant::now() }),
        client: hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
            .build(hyper_util::client::legacy::connect::HttpConnector::new()),
    });
    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/models/{*id}", get(model))
        .route("/v1/unload", axum::routing::post(unload))
        .route("/v1/load", axum::routing::post(load))
        .fallback(pass)
        .layer(axum::extract::DefaultBodyLimit::max(512 << 20))
        .with_state(app.clone());
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(opts.listen).await?;
        println!("serving {} models on http://{} (switching after {settle} s idle)", app.entries.len(), opts.listen);
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        let mut sh = app.shared.lock().await;
        if let Some(b) = sh.backend.take() {
            stop(b).await;
        }
        println!("stopped");
        Ok(())
    })
}

fn model_object(app: &App, e: &Entry, name: &str, loaded: bool) -> Value {
    let context = e.context.load(Ordering::Relaxed);
    json!({"id": name, "object": "model", "created": app.started, "owned_by": "quench6k",
           "max_model_len": context, "context_length": context, "loaded": loaded})
}

/// The learned-context state file: `$XDG_STATE_HOME/quench6k/serve-models-contexts.json`, or under
/// `~/.local/state` without (or with a relative) `XDG_STATE_HOME`.
fn state_file(xdg_state_home: Option<std::ffi::OsString>, home: Option<std::ffi::OsString>) -> Option<PathBuf> {
    let base = xdg_state_home
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| home.filter(|h| !h.is_empty()).map(|h| PathBuf::from(h).join(".local/state")))?;
    Some(base.join("quench6k").join("serve-models-contexts.json"))
}

/// Lists the learned contexts of the entries that have one (never more than the catalog's).
fn apply_learned(entries: &[Entry], learned: &HashMap<String, usize>) {
    for e in entries {
        if let Some(&c) = learned.get(&e.key) {
            e.context.fetch_min(c, Ordering::Relaxed);
        }
    }
}

/// A catalog entry's identity for learned contexts: its args and env (a changed entry learns again).
fn entry_key(args: &[String], env: &HashMap<String, String>) -> String {
    let mut env: Vec<_> = env.iter().collect();
    env.sort();
    json!({"args": args, "env": env}).to_string()
}

/// Learned contexts by entry key (empty when the file is missing or unreadable).
fn read_learned(path: &FsPath) -> HashMap<String, usize> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return HashMap::new();
    };
    let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    v["contexts"]
        .as_object()
        .map(|o| o.iter().filter_map(|(k, c)| Some((k.clone(), c.as_u64()? as usize))).collect())
        .unwrap_or_default()
}

/// Stores a learned context (the file's other entries kept), replacing the file atomically.
fn write_learned(path: &FsPath, key: &str, context: usize) -> Result<()> {
    let mut learned = read_learned(path);
    if learned.get(key) == Some(&context) {
        return Ok(());
    }
    learned.insert(key.to_string(), context);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&json!({"contexts": learned}))? + "\n")
        .with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

/// The context a loaded model's server serves (its `/v1/models`).
async fn served_context(app: &App, port: u16) -> Result<usize> {
    let req = axum::http::Request::get(format!("http://127.0.0.1:{port}/v1/models")).body(Body::empty())?;
    let resp = app.client.request(req).await?;
    anyhow::ensure!(resp.status().is_success(), "/v1/models: HTTP {}", resp.status());
    let body = axum::body::to_bytes(Body::new(resp.into_body()), 1 << 20).await?;
    let v: Value = serde_json::from_slice(&body)?;
    v["data"][0]["max_model_len"].as_u64().map(|c| c as usize).context("/v1/models: no max_model_len")
}

/// Lists (and stores) the context entry `i`'s server serves, once it has loaded.
async fn learn_context(app: &App, i: usize, port: u16) {
    let e = &app.entries[i];
    let served = match served_context(app, port).await {
        Ok(c) => c,
        Err(err) => {
            eprintln!("{}: reading its context: {err:#}", e.names[0]);
            return;
        }
    };
    let before = e.context.swap(served, Ordering::Relaxed);
    if served != before {
        println!("model {} serves context {served} (listed {before} before)", e.names[0]);
    }
    if let Some(path) = &app.state
        && let Err(err) = write_learned(path, &e.key, served)
    {
        eprintln!("storing the learned context of {}: {err:#}", e.names[0]);
    }
}

async fn loaded_entry(app: &App) -> Option<usize> {
    let sh = app.shared.lock().await;
    sh.backend.as_ref().filter(|b| matches!(*b.status.borrow(), Some(Ok(_)))).map(|b| b.entry)
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    let loaded = loaded_entry(&app).await;
    let data: Vec<Value> = app
        .entries
        .iter()
        .enumerate()
        .flat_map(|(i, e)| e.names.iter().map(move |n| (i, e, n)))
        .map(|(i, e, n)| model_object(&app, e, n, loaded == Some(i)))
        .collect();
    Json(json!({"object": "list", "data": data}))
}

async fn model(State(app): State<Arc<App>>, Path(id): Path<String>) -> Response {
    let loaded = loaded_entry(&app).await;
    match find(&app, &id) {
        Some(i) => Json(model_object(&app, &app.entries[i], &id, loaded == Some(i))).into_response(),
        None => error(StatusCode::NOT_FOUND, Some("model_not_found"), format!("model '{id}' not found"), false),
    }
}

/// GET /health: the loaded (or loading) model and its state, the running requests, and the catalog's
/// models (each entry's first name: what the dashboard's model picker offers).
async fn health(State(app): State<Arc<App>>) -> Json<Value> {
    let sh = app.shared.lock().await;
    let (model, state, failure) = match &sh.backend {
        None => (Value::Null, "unloaded", Value::Null),
        Some(b) => {
            let (state, failure) = match &*b.status.borrow() {
                None => ("loading", Value::Null),
                Some(Ok(_)) => ("loaded", Value::Null),
                Some(Err(e)) => ("failed", json!(e)),
            };
            (json!(app.entries[b.entry].names[0]), state, failure)
        }
    };
    let models: Vec<&str> = app.entries.iter().map(|e| e.names[0].as_str()).collect();
    Json(json!({"status": "ok", "model": model, "state": state, "error": failure, "running": sh.running,
                "models": models}))
}

/// POST /v1/unload: stops the loaded model once no request is running (not while one is loading).
async fn unload(State(app): State<Arc<App>>) -> Response {
    let mut sh = app.shared.lock().await;
    if let Some(r) = busy(&app, &sh) {
        return with_anthropic(r, false);
    }
    if let Some(b) = sh.backend.take() {
        stop(b).await;
    }
    Json(json!({"status": "unloaded"})).into_response()
}

/// POST /v1/load {"model": name}: loads a model by hand, switching from the loaded one without the
/// settle period, once no request is running. Answers at once (202 while loading); `/health` follows it.
async fn load(State(app): State<Arc<App>>, body: axum::body::Bytes) -> Response {
    let Some(name) = serde_json::from_slice::<Value>(&body).ok().and_then(|v| v["model"].as_str().map(str::to_string)) else {
        return error(StatusCode::BAD_REQUEST, None, "the body must be {\"model\": name}".into(), false);
    };
    let Some(want) = find(&app, &name) else {
        return error(StatusCode::NOT_FOUND, Some("model_not_found"), format!("model '{name}' not found"), false);
    };
    let mut sh = app.shared.lock().await;
    if let Some(b) = &sh.backend
        && b.entry == want
    {
        match &*b.status.borrow() {
            Some(Ok(_)) => return Json(json!({"status": "loaded", "model": name})).into_response(),
            None => return (StatusCode::ACCEPTED, Json(json!({"status": "loading", "model": name}))).into_response(),
            Some(Err(_)) => {} // a failed load: try again
        }
    }
    if let Some(r) = busy(&app, &sh) {
        return with_anthropic(r, false);
    }
    begin_load(&app, &mut sh, want);
    (StatusCode::ACCEPTED, Json(json!({"status": "loading", "model": name}))).into_response()
}

/// Why the loaded model cannot be stopped by hand now: requests are running, or a model is loading.
fn busy(app: &App, sh: &Shared) -> Option<Rejection> {
    let b = sh.backend.as_ref()?;
    let current = &app.entries[b.entry].names[0];
    let message = if b.status.borrow().is_none() {
        format!("model {current} is loading; wait until it has loaded")
    } else if sh.running > 0 {
        format!("model {current} is serving {} request(s); retry when they have finished", sh.running)
    } else {
        return None;
    };
    Some(Rejection { status: StatusCode::CONFLICT, code: "model_busy", message, retry_after: None })
}

fn find(app: &App, name: &str) -> Option<usize> {
    app.entries.iter().position(|e| e.names.iter().any(|n| n == name))
}

/// An error response, Anthropic-style for /v1/messages paths.
fn error(status: StatusCode, code: Option<&str>, msg: String, anthropic: bool) -> Response {
    let body = if anthropic {
        let kind = match status {
            StatusCode::NOT_FOUND => "not_found_error",
            StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
            s if s.is_server_error() => "api_error",
            _ => "invalid_request_error",
        };
        json!({"type": "error", "error": {"type": kind, "message": msg}})
    } else {
        let kind = if status.is_server_error() { "server_error" } else { "invalid_request_error" };
        json!({"error": {"message": msg, "type": kind, "code": code}})
    };
    (status, Json(body)).into_response()
}

/// Every other route: requests naming a model go to it (loading it when allowed); the rest go to the
/// loaded model, if any. Reads (GET, HEAD) never load a model: with no model loaded, or another model
/// than the one named (`?model=`) loaded, they get 404, the answer clients expect from a server that
/// does not offer an endpoint (llama.cpp's /props, say) - not an error that makes them give up probing.
async fn pass(State(app): State<Arc<App>>, req: Request) -> Response {
    let anthropic = req.uri().path().starts_with("/v1/messages");
    let (parts, body) = req.into_parts();
    let bytes = match axum::body::to_bytes(body, 512 << 20).await {
        Ok(b) => b,
        Err(e) => return error(StatusCode::BAD_REQUEST, None, format!("reading the request: {e}"), anthropic),
    };
    let named = serde_json::from_slice::<Value>(&bytes).ok().and_then(|v| v["model"].as_str().map(str::to_string));
    if parts.method == Method::GET || parts.method == Method::HEAD {
        let asked = named.or_else(|| {
            axum::extract::Query::<HashMap<String, String>>::try_from_uri(&parts.uri).ok()?.0.remove("model")
        });
        let loaded = loaded_entry(&app).await;
        let serves = match (&asked, loaded) {
            (_, None) => false,
            (None, Some(_)) => true,
            (Some(name), Some(l)) => find(&app, name) == Some(l),
        };
        if !serves {
            if asked.is_none() && parts.uri.path() == "/" && parts.method == Method::GET {
                return picker_page();
            }
            let what = asked.map_or_else(|| "no model is loaded".to_string(), |n| format!("'{n}' is not loaded"));
            let msg = format!("{} is answered by a loaded model only, and {what}; model details are at /v1/models", parts.uri.path());
            return error(StatusCode::NOT_FOUND, Some("not_found"), msg, anthropic);
        }
        return forward(&app, None, parts, bytes, anthropic).await;
    }
    let entry = match &named {
        Some(name) => match find(&app, name) {
            Some(i) => Some(i),
            None => return error(StatusCode::NOT_FOUND, Some("model_not_found"), format!("model '{name}' not found"), anthropic),
        },
        None => None,
    };
    forward(&app, entry, parts, bytes, anthropic).await
}

/// Passes a request to the model of `entry` (the loaded one for `None`), loading it when allowed.
async fn forward(app: &Arc<App>, entry: Option<usize>, parts: axum::http::request::Parts, bytes: axum::body::Bytes, anthropic: bool) -> Response {
    let port = match acquire(app, entry).await {
        Ok(p) => p,
        Err(r) => return with_anthropic(r, anthropic),
    };
    let guard = Running(app.clone());
    let uri = format!("http://127.0.0.1:{port}{}", parts.uri.path_and_query().map_or("/", |p| p.as_str()));
    let mut out = axum::http::Request::builder().method(parts.method).uri(uri);
    for (k, v) in parts.headers.iter() {
        if k != header::HOST && k != header::CONTENT_LENGTH {
            out = out.header(k, v);
        }
    }
    let out = out.header(header::CONTENT_LENGTH, bytes.len()).body(Body::from(bytes)).expect("request");
    match app.client.request(out).await {
        Ok(resp) => {
            let (parts, body) = resp.into_parts();
            Response::from_parts(parts, Body::new(Tracked { inner: body, _guard: guard }))
        }
        Err(e) => error(StatusCode::BAD_GATEWAY, None, format!("the model's server: {e}"), anthropic),
    }
}

fn with_anthropic(r: Rejection, anthropic: bool) -> Response {
    let mut resp = error(r.status, Some(r.code), r.message, anthropic);
    if let Some(s) = r.retry_after {
        resp.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(s));
    }
    resp
}

struct Rejection {
    status: StatusCode,
    code: &'static str,
    message: String,
    retry_after: Option<u64>,
}

/// Counts a passed-through request until its response body ends or is dropped.
struct Running(Arc<App>);

impl Drop for Running {
    fn drop(&mut self) {
        let app = self.0.clone();
        tokio::spawn(async move {
            let mut sh = app.shared.lock().await;
            sh.running -= 1;
            sh.active = Instant::now();
        });
    }
}

struct Tracked<B> {
    inner: B,
    _guard: Running,
}

impl<B: hyper::body::Body + Unpin> hyper::body::Body for Tracked<B> {
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<std::result::Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
        std::pin::Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> hyper::body::SizeHint {
        self.inner.size_hint()
    }
}

/// The port of the model `entry` (None: whichever is loaded) with one more running request counted:
/// loads it when nothing is loaded, or switches to it once the loaded model has settled.
async fn acquire(app: &Arc<App>, entry: Option<usize>) -> std::result::Result<u16, Rejection> {
    loop {
        let mut sh = app.shared.lock().await;
        let want = match (entry, &sh.backend) {
            (Some(i), _) => i,
            (None, Some(b)) => b.entry,
            (None, None) => {
                return Err(Rejection {
                    status: StatusCode::BAD_REQUEST,
                    code: "no_model_loaded",
                    message: "no model is loaded; name one (see /v1/models)".into(),
                    retry_after: None,
                });
            }
        };
        if let Some(b) = &sh.backend {
            if b.entry == want {
                let status = b.status.borrow().clone();
                match status {
                    Some(Ok(port)) => {
                        sh.running += 1;
                        sh.active = Instant::now();
                        return Ok(port);
                    }
                    Some(Err(e)) => {
                        // A failed load: report it, and let the next request try again.
                        sh.backend = None;
                        return Err(Rejection { status: StatusCode::SERVICE_UNAVAILABLE, code: "model_load_failed", message: e, retry_after: None });
                    }
                    None => {
                        let mut rx = b.status.clone();
                        drop(sh);
                        let _ = rx.changed().await;
                        continue;
                    }
                }
            }
            // Another model: switch only when it is loaded and has been idle for the settle period.
            let current = &app.entries[b.entry].names[0];
            let loading = b.status.borrow().is_none();
            let idle = sh.active.elapsed();
            if loading || sh.running > 0 || idle < app.settle {
                let wait = if loading || sh.running > 0 { app.settle } else { app.settle - idle };
                let why = if loading {
                    "is loading".to_string()
                } else if sh.running > 0 {
                    format!("is serving {} request(s)", sh.running)
                } else {
                    format!("was used {:.0} s ago", idle.as_secs_f64())
                };
                return Err(Rejection {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    code: "model_switch_blocked",
                    message: format!(
                        "model {current} {why}; switching to {} needs it idle for {:.0} s (retry later)",
                        app.entries[want].names[0],
                        app.settle.as_secs_f64()
                    ),
                    retry_after: Some(wait.as_secs_f64().ceil() as u64),
                });
            }
        }
        // Load `want` (after stopping the loaded model) in the background; this request waits for it.
        begin_load(app, &mut sh, want);
    }
}

/// Makes `want` the backend, loading it in the background after stopping the current one.
fn begin_load(app: &Arc<App>, sh: &mut Shared, want: usize) {
    let old = sh.backend.take();
    let (tx, rx) = watch::channel(None);
    sh.backend = Some(Backend { entry: want, child: None, status: rx });
    let app = app.clone();
    tokio::spawn(async move {
        if let Some(b) = old {
            stop(b).await;
        }
        let result = start(&app, want).await;
        if let Ok((_, port)) = &result {
            learn_context(&app, want, *port).await;
        }
        let mut sh = app.shared.lock().await;
        let status = match result {
            Ok((child, port)) => {
                if let Some(b) = sh.backend.as_mut().filter(|b| b.entry == want) {
                    b.child = Some(child);
                }
                Ok(port)
            }
            Err(e) => {
                eprintln!("loading {}: {e:#}", app.entries[want].names[0]);
                Err(format!("loading {}: {e:#}", app.entries[want].names[0]))
            }
        };
        sh.active = Instant::now();
        let _ = tx.send(Some(status));
    });
}

/// The model picker shown at `/` while no model serves (the loaded model's dashboard has its own).
fn picker_page() -> Response {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")],
        include_str!("multi.html"),
    )
        .into_response()
}

/// Starts a model's server and waits until it answers.
async fn start(app: &App, i: usize) -> Result<(tokio::process::Child, u16)> {
    let e = &app.entries[i];
    let port = std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
    println!("loading {} on port {port}", e.names[0]);
    let t0 = Instant::now();
    let mut child = tokio::process::Command::new(&app.exe)
        .arg("serve")
        .args(&e.args)
        .arg("--listen")
        .arg(format!("127.0.0.1:{port}"))
        .envs(&e.env)
        .kill_on_drop(true)
        .spawn()
        .context("starting quench6k serve")?;
    loop {
        if let Some(status) = child.try_wait()? {
            bail!("the model's server exited ({status}) while loading");
        }
        if tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
            println!("{} loaded in {:.1} s", e.names[0], t0.elapsed().as_secs_f64());
            return Ok((child, port));
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// Stops a model's server (SIGINT: it keeps its stored contexts), killing it after a minute.
async fn stop(b: Backend) {
    let Some(mut child) = b.child else { return };
    if let Some(pid) = child.id() {
        let _ = tokio::process::Command::new("kill").arg("-INT").arg(pid.to_string()).status().await;
    }
    match tokio::time::timeout(Duration::from_secs(60), child.wait()).await {
        Ok(_) => {}
        Err(_) => {
            let _ = child.kill().await;
        }
    }
    println!("model unloaded");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(args: &[&str], context: usize) -> Entry {
        let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        let env = HashMap::new();
        let key = entry_key(&args, &env);
        Entry { args, env, names: vec!["m".into()], context: AtomicUsize::new(context), key }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("quench6k-multi-{tag}-{}-{nanos}", std::process::id()))
    }

    #[test]
    fn state_file_follows_xdg() {
        let f = |x: Option<&str>, h: Option<&str>| state_file(x.map(Into::into), h.map(Into::into));
        let want = |base: &str| Some(PathBuf::from(base).join("quench6k/serve-models-contexts.json"));
        assert_eq!(f(Some("/x/state"), Some("/h")), want("/x/state"));
        // A relative XDG_STATE_HOME is ignored (the XDG spec), as an unset one.
        assert_eq!(f(Some("rel"), Some("/h")), want("/h/.local/state"));
        assert_eq!(f(None, Some("/h")), want("/h/.local/state"));
        assert_eq!(f(None, None), None);
    }

    #[test]
    fn entry_keys_follow_args_and_env() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let env = |kv: &[(&str, &str)]| kv.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect::<HashMap<_, _>>();
        let a = args(&["--model", "/m", "--context", "262144"]);
        let k = entry_key(&a, &env(&[("A", "1"), ("B", "2")]));
        assert_eq!(k, entry_key(&a, &env(&[("B", "2"), ("A", "1")])));
        assert_ne!(k, entry_key(&a, &env(&[("A", "1")])));
        assert_ne!(k, entry_key(&args(&["--model", "/m", "--context", "131072"]), &env(&[("A", "1"), ("B", "2")])));
    }

    #[test]
    fn learned_contexts_persist_per_entry() {
        let dir = temp_dir("learned");
        let path = dir.join("quench6k/serve-models-contexts.json");
        assert!(read_learned(&path).is_empty());
        let hy4 = entry(&["--model", "/hy4", "--slots", "8", "--context", "262144"], 262_144);
        let ink = entry(&["--model", "/ink", "--context", "1048576"], 1 << 20);
        write_learned(&path, &hy4.key, 202_752).unwrap();
        write_learned(&path, &ink.key, 1 << 20).unwrap();
        // A restart lists the learned contexts before any load.
        let fresh = [
            entry(&["--model", "/hy4", "--slots", "8", "--context", "262144"], 262_144),
            entry(&["--model", "/ink", "--context", "1048576"], 1 << 20),
            // A changed entry (other args) is not matched: it lists its catalog context.
            entry(&["--model", "/hy4", "--slots", "4", "--context", "262144"], 262_144),
        ];
        apply_learned(&fresh, &read_learned(&path));
        let listed: Vec<usize> = fresh.iter().map(|e| e.context.load(Ordering::Relaxed)).collect();
        assert_eq!(listed, [202_752, 1 << 20, 262_144]);
        // Relearning replaces the value and keeps the other entries.
        write_learned(&path, &hy4.key, 200_704).unwrap();
        let learned = read_learned(&path);
        assert_eq!(learned.len(), 2);
        assert_eq!(learned[&hy4.key], 200_704);
        // A learned value never lists more than the catalog's context.
        let small = entry(&["--model", "/ink", "--context", "1048576"], 65_536);
        apply_learned(std::slice::from_ref(&small), &learned);
        assert_eq!(small.context.load(Ordering::Relaxed), 65_536);
        // An unreadable file counts as empty and is replaced on the next write.
        std::fs::write(&path, "not json").unwrap();
        assert!(read_learned(&path).is_empty());
        write_learned(&path, &hy4.key, 202_752).unwrap();
        assert_eq!(read_learned(&path)[&hy4.key], 202_752);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn named(name: &str) -> Entry {
        let mut e = entry(&["--model", &format!("/{name}")], 4096);
        e.names = vec![name.into(), format!("{name}-alias")];
        e
    }

    fn test_app(entries: Vec<Entry>) -> Arc<App> {
        Arc::new(App {
            entries,
            state: None,
            settle: Duration::from_secs(60),
            // No executable: every load fails at once (what the hand-load path does with a failure).
            exe: PathBuf::new(),
            started: 0,
            shared: Mutex::new(Shared { backend: None, running: 0, active: Instant::now() }),
            client: hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                .build(hyper_util::client::legacy::connect::HttpConnector::new()),
        })
    }

    async fn call(app: &Arc<App>, path: &str, body: &str) -> (StatusCode, Value) {
        let resp = if path == "/health" {
            health(State(app.clone())).await.into_response()
        } else if path == "/v1/unload" {
            unload(State(app.clone())).await
        } else {
            load(State(app.clone()), axum::body::Bytes::from(body.to_string())).await
        };
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    /// A backend in a given state (Some(Ok(port)): loaded), without a child process.
    fn backend(entry: usize, status: Option<std::result::Result<u16, String>>) -> Backend {
        let (_tx, rx) = watch::channel(status);
        Backend { entry, child: None, status: rx }
    }

    #[test]
    fn hand_load_and_unload() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let app = test_app(vec![named("a"), named("b")]);
            let (s, h) = call(&app, "/health", "").await;
            assert_eq!(s, StatusCode::OK);
            assert_eq!(h["models"], json!(["a", "b"]));
            assert_eq!(h["state"], "unloaded");
            assert_eq!(call(&app, "/v1/load", "{}").await.0, StatusCode::BAD_REQUEST);
            assert_eq!(call(&app, "/v1/load", r#"{"model": "c"}"#).await.0, StatusCode::NOT_FOUND);

            // Loaded: the same model (any of its names) is a no-op; another one switches at once (no settle).
            app.shared.lock().await.backend = Some(backend(0, Some(Ok(1))));
            app.shared.lock().await.active = Instant::now();
            let (s, v) = call(&app, "/v1/load", r#"{"model": "a-alias"}"#).await;
            assert_eq!((s, v["status"].as_str()), (StatusCode::OK, Some("loaded")));
            // ...but not while requests run, and neither does unload.
            app.shared.lock().await.running = 2;
            let (s, v) = call(&app, "/v1/load", r#"{"model": "b"}"#).await;
            assert_eq!(s, StatusCode::CONFLICT);
            assert!(v["error"]["message"].as_str().unwrap().contains("serving 2 request(s)"), "{v}");
            assert_eq!(call(&app, "/v1/unload", "").await.0, StatusCode::CONFLICT);
            app.shared.lock().await.running = 0;
            // Loading: unload and other loads wait; the same model reports loading.
            app.shared.lock().await.backend = Some(backend(0, None));
            assert_eq!(call(&app, "/v1/unload", "").await.0, StatusCode::CONFLICT);
            assert_eq!(call(&app, "/v1/load", r#"{"model": "b"}"#).await.0, StatusCode::CONFLICT);
            assert_eq!(call(&app, "/v1/load", r#"{"model": "a"}"#).await.0, StatusCode::ACCEPTED);
            // Unload of a loaded model.
            app.shared.lock().await.backend = Some(backend(0, Some(Ok(1))));
            let (s, _) = call(&app, "/v1/unload", "").await;
            assert_eq!(s, StatusCode::OK);
            assert!(app.shared.lock().await.backend.is_none());

            // A hand load runs in the background; its failure shows on /health, and loading again retries.
            let (s, v) = call(&app, "/v1/load", r#"{"model": "b-alias"}"#).await;
            assert_eq!((s, v["status"].as_str()), (StatusCode::ACCEPTED, Some("loading")));
            let mut h = Value::Null;
            for _ in 0..100 {
                h = call(&app, "/health", "").await.1;
                if h["state"] != "loading" {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            assert_eq!(h["state"], "failed");
            assert_eq!(h["model"], "b");
            assert!(h["error"].as_str().unwrap().starts_with("loading b:"), "{h}");
            assert_eq!(call(&app, "/v1/load", r#"{"model": "b"}"#).await.0, StatusCode::ACCEPTED);
        });
    }

    #[test]
    fn loaded_server_context_is_learned_and_stored() {
        let dir = temp_dir("serve");
        let path = dir.join("contexts.json");
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // A model server that serves less than its catalog context.
            let fake = Router::new().route(
                "/v1/models",
                get(|| async { Json(json!({"object": "list", "data": [{"id": "m", "max_model_len": 198_656}]})) }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            tokio::spawn(async move { axum::serve(listener, fake).await });
            let app = App {
                entries: vec![entry(&["--model", "/hy4", "--context", "262144"], 262_144)],
                state: Some(path.clone()),
                settle: Duration::from_secs(60),
                exe: PathBuf::new(),
                started: 0,
                shared: Mutex::new(Shared { backend: None, running: 0, active: Instant::now() }),
                client: hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                    .build(hyper_util::client::legacy::connect::HttpConnector::new()),
            };
            learn_context(&app, 0, port).await;
            let listed = model_object(&app, &app.entries[0], "m", true);
            assert_eq!(listed["max_model_len"], 198_656);
            assert_eq!(listed["context_length"], 198_656);
            assert_eq!(read_learned(&path)[&app.entries[0].key], 198_656);
        });
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
