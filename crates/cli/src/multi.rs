//! `quench6k serve-models`: one endpoint for several models, loaded on demand. Nothing loads at start;
//! `/v1/models` lists every model of the catalog (answered without loading). A request names its model
//! (the body's "model"): when another model is loaded, the server switches only once that model has been
//! idle (no request running) for the settle period; until then the request gets an error (503 with
//! Retry-After), so two clients asking for different models cannot make it load and unload in turns.
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
//! adapted or steered name with `--adapter` or `--steering`) and context come from them.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use axum::Router;
use axum::body::Body;
use axum::extract::{Path, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
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
    context: usize,
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
        entries.push(Entry { args, env, names, context });
    }
    let mut seen = std::collections::HashSet::new();
    for n in entries.iter().flat_map(|e| &e.names) {
        if !seen.insert(n.clone()) {
            bail!("catalog: model name {n} appears twice");
        }
    }
    for e in &entries {
        println!("model {} (context {})", e.names.join(", "), e.context);
    }
    let app = Arc::new(App {
        entries,
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
    json!({"id": name, "object": "model", "created": app.started, "owned_by": "quench6k",
           "max_model_len": e.context, "context_length": e.context, "loaded": loaded})
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

async fn health(State(app): State<Arc<App>>) -> Json<Value> {
    let sh = app.shared.lock().await;
    let (model, state) = match &sh.backend {
        None => (Value::Null, "unloaded"),
        Some(b) => (
            json!(app.entries[b.entry].names[0]),
            match *b.status.borrow() {
                None => "loading",
                Some(Ok(_)) => "loaded",
                Some(Err(_)) => "failed",
            },
        ),
    };
    Json(json!({"status": "ok", "model": model, "state": state, "running": sh.running}))
}

/// POST /v1/unload: stops the loaded model once no request is running.
async fn unload(State(app): State<Arc<App>>) -> Response {
    let mut sh = app.shared.lock().await;
    if sh.running > 0 {
        return error(StatusCode::CONFLICT, Some("model_busy"), "requests are running".into(), false);
    }
    if let Some(b) = sh.backend.take() {
        stop(b).await;
    }
    Json(json!({"status": "unloaded"})).into_response()
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
/// loaded model, if any.
async fn pass(State(app): State<Arc<App>>, req: Request) -> Response {
    let anthropic = req.uri().path().starts_with("/v1/messages");
    let (parts, body) = req.into_parts();
    let bytes = match axum::body::to_bytes(body, 512 << 20).await {
        Ok(b) => b,
        Err(e) => return error(StatusCode::BAD_REQUEST, None, format!("reading the request: {e}"), anthropic),
    };
    let named = serde_json::from_slice::<Value>(&bytes).ok().and_then(|v| v["model"].as_str().map(str::to_string));
    let entry = match &named {
        Some(name) => match find(&app, name) {
            Some(i) => Some(i),
            None => return error(StatusCode::NOT_FOUND, Some("model_not_found"), format!("model '{name}' not found"), anthropic),
        },
        None => None,
    };
    let port = match acquire(&app, entry).await {
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
                    status: StatusCode::SERVICE_UNAVAILABLE,
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
        let old = sh.backend.take();
        let (tx, rx) = watch::channel(None);
        sh.backend = Some(Backend { entry: want, child: None, status: rx });
        drop(sh);
        let app2 = app.clone();
        tokio::spawn(async move {
            if let Some(b) = old {
                stop(b).await;
            }
            let result = start(&app2, want).await;
            let mut sh = app2.shared.lock().await;
            let status = match result {
                Ok((child, port)) => {
                    if let Some(b) = sh.backend.as_mut().filter(|b| b.entry == want) {
                        b.child = Some(child);
                    }
                    Ok(port)
                }
                Err(e) => {
                    eprintln!("loading {}: {e:#}", app2.entries[want].names[0]);
                    Err(format!("loading {}: {e:#}", app2.entries[want].names[0]))
                }
            };
            sh.active = Instant::now();
            let _ = tx.send(Some(status));
        });
    }
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
