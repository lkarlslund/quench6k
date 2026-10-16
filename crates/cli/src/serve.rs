//! OpenAI-compatible HTTP serving (`squash serve`): `/v1/models` and `/v1/chat/completions` with
//! streaming, reasoning (`reasoning_content`) and tool calls. One GPU worker thread owns the model and
//! the generation session; requests queue for it.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use squash_engine::model::Precision;
use squash_engine::session::{Finish, GenParams, Progress, Session};
use tokio::sync::mpsc as tmpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::chat::{Chat, TextStream};

pub struct ServeOptions {
    pub model: PathBuf,
    pub listen: SocketAddr,
    pub context: usize,
    pub drafts: usize,
    pub name: String,
    pub steer: crate::SteerArgs,
    pub cache_host_gib: f64,
    pub cache_dir: Option<PathBuf>,
    pub chat_template: Option<PathBuf>,
    pub tool_retries: usize,
    pub default_max_tokens: usize,
    /// Concurrent requests decoded together (1 or 2).
    pub slots: usize,
    pub capture_dir: Option<PathBuf>,
    pub vision: bool,
    pub image_max_pixels: usize,
    pub video_max_frames: usize,
}

/// Identity of the served model and numerics for stored contexts: config.json plus the profile.
pub fn model_id(model_dir: &std::path::Path) -> String {
    let config = std::fs::read(model_dir.join("config.json")).unwrap_or_default();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let kv: &[u8] = if squash_engine::forward::KV_FP8.load(std::sync::atomic::Ordering::Relaxed) { b"/kv8" } else { b"" };
    for b in config.iter().chain(b"int8g/squash-1".iter()).chain(kv) {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

enum Work {
    Generate(Job),
    /// Write stored contexts to disk, then acknowledge.
    Persist(mpsc::Sender<()>),
}

/// A queued generation and where its tokens go.
struct Job {
    ids: Vec<i32>,
    params: GenParams,
    tx: tmpsc::UnboundedSender<Msg>,
    /// Activation capture: the output file and metadata to embed.
    capture: Option<(PathBuf, Value)>,
}

enum Msg {
    Token(i32),
    /// Prefill progress: (prompt tokens computed, of which served from cache).
    Progress(usize, usize),
    /// Finished, with the prompt tokens served from the prefix cache.
    Done(Finish, usize),
    Error(String),
}

struct App {
    chat: Chat,
    jobs: mpsc::Sender<Work>,
    name: String,
    /// Steering strength used when a request carries none (the loaded pack's, or 0).
    default_strength: f32,
    steering: bool,
    tool_retries: usize,
    default_max_tokens: usize,
    tool_call_token: Option<i32>,
    capture_dir: Option<PathBuf>,
    model_id: String,
    vision: bool,
    image_max_pixels: usize,
    video_max_frames: usize,
}

pub fn run(opts: ServeOptions) -> Result<()> {
    let chat = Chat::load(&opts.model, opts.chat_template.as_deref())?;
    let tool_call_token = chat.token_id("<tool_call>");
    let (jobs_tx, jobs_rx) = mpsc::channel::<Work>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<f32>>();
    let (model_dir, context, drafts, steer, slots) = (opts.model.clone(), opts.context, opts.drafts, opts.steer.clone(), opts.slots);
    let store = squash_engine::context_store::StoreOptions {
        host_bytes: (opts.cache_host_gib * 1073741824.0) as usize,
        dir: opts.cache_dir.clone(),
        model_id: model_id(&opts.model),
    };
    let steering = steer.steering.is_some();
    let vision = opts.vision;
    std::thread::spawn(move || {
        let setup = (|| -> Result<_> {
            let (mut model, stream) = crate::load(&model_dir, Precision::Int8Group, true)?;
            let strength = steer.attach(&mut model, &stream)?;
            let mut session = Session::new(&model, &stream, context, 8192, drafts, store)?;
            if vision {
                let ckpt = squash_engine::checkpoint::Checkpoint::open(&model_dir)?;
                session.set_vision(squash_engine::vision::VisionModel::load(&ckpt, squash_engine::loader::LoadOptions::default(), &stream)?);
            }
            println!("context store: {} stored contexts", session.stored_contexts());
            Ok((model, stream, session, strength))
        })();
        let (model, stream, mut session) = match setup {
            Ok((m, s, sess, strength)) => {
                let _ = ready_tx.send(Ok(strength));
                (m, s, sess)
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
                return;
            }
        };
        if slots > 1 {
            if let Err(e) = session.enable_second_slot(&model, &stream) {
                eprintln!("second slot unavailable: {e:#}");
            }
        }
        schedule(&model, &stream, &mut session, &jobs_rx);
    });
    let default_strength = ready_rx.recv().context("worker thread")??;
    let app = Arc::new(App { chat, jobs: jobs_tx.clone(), name: opts.name, default_strength, steering, tool_retries: opts.tool_retries, default_max_tokens: opts.default_max_tokens, tool_call_token, capture_dir: opts.capture_dir.clone(), model_id: model_id(&opts.model), vision, image_max_pixels: opts.image_max_pixels, video_max_frames: opts.video_max_frames });
    let router = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .with_state(app);
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(opts.listen).await?;
        println!("serving on http://{}", opts.listen);
        axum::serve(listener, router).with_graceful_shutdown(async { let _ = tokio::signal::ctrl_c().await; }).await?;
        // Keep stored contexts across the restart.
        let (ack_tx, ack_rx) = mpsc::channel();
        if jobs_tx.send(Work::Persist(ack_tx)).is_ok() {
            let _ = ack_rx.recv();
        }
        println!("stopped");
        // Exit without tearing down the GPU worker: destroying its CUDA graphs while the runtime
        // shuts down crashes, and the driver frees everything at exit anyway.
        use std::io::Write;
        let _ = std::io::stdout().flush();
        std::process::exit(0)
    })
}

/// The GPU worker: admits queued requests into free slots and steps the running ones, one alone
/// (speculatively when it started alone) or both batched.
fn schedule(model: &squash_engine::model::Model, s: &squash_cuda::Stream, session: &mut Session, jobs: &mpsc::Receiver<Work>) {
    use squash_engine::session::Run;
    struct Active {
        run: Run,
        tx: tmpsc::UnboundedSender<Msg>,
        reused: usize,
        capture: Option<(PathBuf, Value)>,
        generated: Vec<i32>,
        /// Prompt tokens, prefill seconds, and when decoding started.
        prompt: usize,
        prefill: f64,
        decode_start: std::time::Instant,
    }
    // Sends tokens; Some when the run is over (Cancelled when the client went away).
    fn emit(a: &mut Active, toks: &[i32]) -> Option<Finish> {
        for &t in toks {
            if a.tx.send(Msg::Token(t)).is_err() {
                return Some(Finish::Cancelled);
            }
            a.generated.push(t);
            if let Some(f) = a.run.accept(t) {
                return Some(f);
            }
        }
        None
    }
    // Completes a finished run: writes its capture, then reports.
    let complete = |a: Active, f: Finish, session: &mut Session| {
        if let Some((path, meta)) = &a.capture {
            let result = session.finish_capture(model, s, &a.run).and_then(|c| match c {
                Some(c) => write_capture(path, meta, &a.generated, &c),
                None => Ok(()),
            });
            if let Err(e) = result {
                let _ = a.tx.send(Msg::Error(format!("capture: {e:#}")));
                return;
            }
        }
        let secs = a.decode_start.elapsed().as_secs_f64();
        let n = a.generated.len();
        eprintln!("done: slot {}, prompt {} ({} reused) prefill {:.2}s ({:.0} tok/s), {} generated in {:.2}s ({:.1} tok/s), {:?}",
                  a.run.slot, a.prompt, a.reused, a.prefill, (a.prompt - a.reused) as f64 / a.prefill.max(1e-9), n, secs,
                  n.saturating_sub(1) as f64 / secs.max(1e-9), f);
        if let Some(path) = std::env::var_os("SQUASH_TRACE") {
            use std::io::Write;
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(file, "{}", json!({"slot": a.run.slot, "finish": format!("{f:?}"), "generated": a.generated}));
            }
        }
        let _ = a.tx.send(Msg::Done(f, a.reused));
    };
    let nslots = if session.has_second_slot() { 2 } else { 1 };
    let mut slots: [Option<Active>; 2] = [None, None];
    let mut queue: std::collections::VecDeque<Job> = std::collections::VecDeque::new();
    let take = |w: Work, queue: &mut std::collections::VecDeque<Job>, session: &mut Session| match w {
        Work::Generate(job) => queue.push_back(job),
        Work::Persist(ack) => {
            if let Err(e) = session.persist() {
                eprintln!("persisting contexts: {e:#}");
            }
            let _ = ack.send(());
        }
    };
    loop {
        if slots.iter().all(Option::is_none) && queue.is_empty() {
            match jobs.recv() {
                Ok(w) => take(w, &mut queue, session),
                Err(_) => return,
            }
        }
        while let Ok(w) = jobs.try_recv() {
            take(w, &mut queue, session);
        }
        // Admit queued requests into free slots (the main slot first: MTP and the prefix cache).
        while let Some(free) = (0..nslots).find(|&i| slots[i].is_none()) {
            let Some(job) = queue.pop_front() else { break };
            let other = 1 - free;
            if let Some(a) = slots[other].as_mut() {
                if a.run.is_spec() {
                    match session.to_plain(model, s, &mut a.run) {
                        Ok(toks) => {
                            if let Some(f) = emit(a, &toks) {
                                complete(slots[other].take().unwrap(), f, session);
                            }
                        }
                        Err(e) => {
                            let _ = a.tx.send(Msg::Error(format!("{e:#}")));
                            slots[other] = None;
                        }
                    }
                }
            }
            let speculate = slots[other].is_none() && queue.is_empty();
            let t0 = std::time::Instant::now();
            match session.begin(model, s, free, &job.ids, &job.params, speculate) {
                Ok((run, toks)) => {
                    let mut a = Active { run, tx: job.tx, reused: if free == 0 { session.reused } else { 0 }, capture: job.capture, generated: Vec::new(),
                                         prompt: job.ids.len(), prefill: t0.elapsed().as_secs_f64(), decode_start: std::time::Instant::now() };
                    if let Some(path) = std::env::var_os("SQUASH_TRACE") {
                        use std::io::Write;
                        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                            let _ = writeln!(f, "{}", json!({"slot": free, "reused": a.reused, "ids": job.ids}));
                        }
                    }
                    match emit(&mut a, &toks) {
                        Some(f) => complete(a, f, session),
                        None => slots[free] = Some(a),
                    }
                }
                Err(e) => {
                    let _ = job.tx.send(Msg::Error(format!("{e:#}")));
                }
            }
        }
        // Step the running requests.
        let result = match &mut slots {
            [Some(a), Some(b)] => session.step(model, s, &mut [&mut a.run, &mut b.run]),
            [Some(a), None] | [None, Some(a)] => session.step(model, s, &mut [&mut a.run]),
            [None, None] => continue,
        };
        match result {
            Ok(toks) => {
                let mut k = 0;
                for slot in slots.iter_mut() {
                    if let Some(a) = slot {
                        let done = emit(a, &toks[k]);
                        k += 1;
                        if let Some(f) = done {
                            complete(slot.take().unwrap(), f, session);
                        }
                    }
                }
            }
            Err(e) => {
                for slot in slots.iter_mut() {
                    if let Some(a) = slot.take() {
                        let _ = a.tx.send(Msg::Error(format!("{e:#}")));
                    }
                }
            }
        }
    }
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": app.name, "object": "model", "owned_by": "squash"}]}))
}

fn error(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"message": msg.into(), "type": "invalid_request_error"}}))).into_response()
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// What the request asks for, after rendering and tokenizing.
struct Prepared {
    capture: Option<(PathBuf, Value)>,
    ids: Vec<i32>,
    params: GenParams,
    stops: Vec<String>,
    thinking: bool,
    tools: Option<Value>,
    stream: bool,
    /// Stream prefill progress chunks ("return_progress", as llama.cpp's server).
    progress: bool,
}

fn prepare(app: &App, req: &Value) -> Result<Prepared, Response> {
    let messages = req.get("messages").filter(|m| m.is_array()).ok_or_else(|| error(StatusCode::BAD_REQUEST, "messages must be an array"))?;
    let kwargs = req.get("chat_template_kwargs").cloned().unwrap_or(Value::Null);
    let enable_thinking = kwargs.get("enable_thinking").and_then(Value::as_bool);
    let effort = kwargs.get("reasoning_effort").or_else(|| req.get("reasoning_effort")).and_then(Value::as_str);
    let tools = req.get("tools").filter(|t| t.as_array().is_some_and(|a| !a.is_empty())).cloned();
    let (messages, images, videos) = extract_images(app, messages)?;
    let text = app.chat.render(&messages, tools.as_ref(), enable_thinking, effort).map_err(|e| error(StatusCode::BAD_REQUEST, format!("chat template: {e:#}")))?;
    let thinking = text.ends_with("<think>\n");
    let ids = app.chat.encode(&text).map_err(|e| error(StatusCode::BAD_REQUEST, format!("{e:#}")))?;
    let (mut ids, images) = app.chat.expand_media(&ids, images, videos).map_err(|e| error(StatusCode::BAD_REQUEST, format!("{e:#}")))?;
    let mut thinking = thinking;
    // Debugging: "prompt_ids" replays a traced prompt (text-only) in place of the rendered messages.
    if let Some(raw) = req.get("prompt_ids").and_then(Value::as_array) {
        ids = raw.iter().filter_map(Value::as_i64).map(|v| v as i32).collect();
        let tail = app.chat.decode(&ids[ids.len().saturating_sub(4)..], false).unwrap_or_default();
        thinking = tail.ends_with("<think>\n");
    }
    let max_tokens = req.get("max_completion_tokens").or_else(|| req.get("max_tokens")).and_then(Value::as_u64).map_or(app.default_max_tokens, |n| n as usize);
    let temperature = req.get("temperature").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    let top_p = req.get("top_p").and_then(Value::as_f64).unwrap_or(0.95) as f32;
    let top_k = req.get("top_k").and_then(Value::as_u64).unwrap_or(20) as usize;
    let seed = req.get("seed").and_then(Value::as_u64).unwrap_or_else(|| now() ^ 0x5eed);
    let stops = match req.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect(),
        _ => Vec::new(),
    };
    let steering_strength = match req.get("steering").and_then(|s| s.get("steering_strength")).and_then(Value::as_f64) {
        Some(_) if !app.steering => return Err(error(StatusCode::BAD_REQUEST, "steering requested but no pack is loaded")),
        Some(a) => a as f32,
        None => app.default_strength,
    };
    let capture = match req.get("capture") {
        None | Some(Value::Null) => None,
        Some(c) => {
            let name = c.get("name").and_then(Value::as_str).unwrap_or("");
            if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") || name.starts_with('.') {
                return Err(error(StatusCode::BAD_REQUEST, "capture.name must be a plain file name"));
            }
            let Some(dir) = &app.capture_dir else { return Err(error(StatusCode::BAD_REQUEST, "capture requested but the server has no --capture-dir")) };
            let meta = json!({"engine": "squash", "model_id": app.model_id, "prompt_ids": ids, "temperature": temperature, "top_p": top_p,
                              "top_k": top_k, "seed": seed, "steering_strength": steering_strength, "request": c});
            Some((dir.join(format!("{name}.safetensors")), meta))
        }
    };
    let params = GenParams { max_tokens, temperature, top_p, top_k, seed, stop_tokens: app.chat.stop_tokens.clone(), steering_strength, boundaries: app.chat.boundaries(&ids), capture: capture.is_some(), images, progress: None };
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let progress = stream && req.get("return_progress").and_then(Value::as_bool).unwrap_or(false);
    Ok(Prepared { capture, ids, params, stops, thinking, tools, stream, progress })
}

/// Splits generated text into (reasoning, content, tool calls).
/// Replaces image and video parts of message contents (OpenAI-style `image_url` / `video_url` with
/// a data URL) by template placeholders and preprocesses them, in order.
#[allow(clippy::type_complexity)]
fn extract_images(app: &App, messages: &Value) -> Result<(Value, Vec<squash_engine::vision::VisionInput>, Vec<squash_engine::vision::VideoClip>), Response> {
    use base64::Engine;
    let mut out = messages.clone();
    let (mut images, mut videos) = (Vec::new(), Vec::new());
    for m in out.as_array_mut().into_iter().flatten() {
        let Some(parts) = m.get_mut("content").and_then(Value::as_array_mut) else { continue };
        for part in parts.iter_mut() {
            let kind = part.get("type").and_then(Value::as_str).unwrap_or("").to_owned();
            let url = match kind.as_str() {
                "image_url" | "video_url" => part.get(kind.as_str()).and_then(|u| u.get("url").or(Some(u))).and_then(Value::as_str).map(str::to_owned),
                "image" | "video" => part.get(kind.as_str()).and_then(Value::as_str).map(str::to_owned),
                _ => continue,
            };
            if !app.vision {
                return Err(error(StatusCode::BAD_REQUEST, "this server has no vision tower (started with --no-vision)"));
            }
            let url = url.ok_or_else(|| error(StatusCode::BAD_REQUEST, "image part without a URL"))?;
            let data = url.strip_prefix("data:").and_then(|r| r.split_once(";base64,")).map(|(_, d)| d)
                .ok_or_else(|| error(StatusCode::BAD_REQUEST, "images must be data URLs (data:<type>;base64,...)"))?;
            let bytes = base64::engine::general_purpose::STANDARD.decode(data.trim()).map_err(|e| error(StatusCode::BAD_REQUEST, format!("media base64: {e}")))?;
            if kind.starts_with("video") {
                // ffmpeg reads the clip from a private temporary file.
                let path = std::env::temp_dir().join(format!("squash-video-{}-{}", std::process::id(), now() ^ (bytes.len() as u64).wrapping_mul(0x9e37_79b9)));
                std::fs::write(&path, &bytes).map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, format!("video: {e}")))?;
                let clip = squash_engine::vision::preprocess_video(&path, 4096, 25165824, app.video_max_frames);
                let _ = std::fs::remove_file(&path);
                videos.push(clip.map_err(|e| error(StatusCode::BAD_REQUEST, format!("video: {e:#}")))?);
                *part = json!({"type": "video"});
            } else {
                let input = squash_engine::vision::preprocess_image(&bytes, 65536, app.image_max_pixels).map_err(|e| error(StatusCode::BAD_REQUEST, format!("image: {e:#}")))?;
                images.push(input);
                *part = json!({"type": "image"});
            }
        }
    }
    Ok((out, images, videos))
}

fn parse_output(text: &str, thinking: bool, tools: Option<&Value>) -> (Option<String>, String, Vec<Value>) {
    let (reasoning, rest) = if thinking {
        match think_end(text) {
            Some((i, j)) => (Some(text[..i].trim().to_string()), text[j..].trim_start()),
            None => (Some(text.trim().to_string()), ""),
        }
    } else {
        (None, text)
    };
    let mut calls = Vec::new();
    let content_end = rest.find("<tool_call>").unwrap_or(rest.len());
    let mut cursor = content_end;
    while let Some(start) = rest[cursor..].find("<tool_call>") {
        let body_start = cursor + start + "<tool_call>".len();
        let Some(end) = rest[body_start..].find("</tool_call>") else { break };
        if let Some(call) = parse_tool_call(&rest[body_start..body_start + end], tools) {
            calls.push(json!({"id": format!("call_{}_{}", now(), calls.len()), "type": "function", "function": call}));
        }
        cursor = body_start + end + "</tool_call>".len();
    }
    // Leading whitespace after </think> is layout; trailing whitespace is kept (it can precede a stop
    // string) unless tool calls follow.
    let content = rest[..content_end].trim_start();
    let content = if calls.is_empty() { content } else { content.trim_end() };
    (reasoning, content.to_string(), calls)
}

/// `<function=NAME>` then `<parameter=ARG>\nvalue\n</parameter>` blocks (or a JSON object). Values are kept as strings
/// when the tool schema says so, otherwise parsed as JSON when they are valid JSON.
fn parse_tool_call(body: &str, tools: Option<&Value>) -> Option<Value> {
    // JSON style (hermes): {"name": ..., "arguments": {...}}.
    if body.trim_start().starts_with('{') {
        let v: Value = serde_json::from_str(body.trim()).ok()?;
        let name = v.get("name")?.as_str()?.to_string();
        let args = match v.get("arguments") {
            Some(Value::String(a)) => serde_json::from_str::<Value>(a).ok()?,
            Some(a @ Value::Object(_)) => a.clone(),
            None => json!({}),
            _ => return None,
        };
        return Some(json!({"name": name, "arguments": args.to_string()}));
    }
    let f = body.find("<function=")? + "<function=".len();
    let name_end = body[f..].find('>')? + f;
    let name = body[f..name_end].trim().to_string();
    let schema = tools.and_then(|t| t.as_array()).and_then(|a| {
        a.iter().find(|t| t.get("function").and_then(|f| f.get("name")).and_then(Value::as_str) == Some(name.as_str()))
    });
    let mut args = serde_json::Map::new();
    let mut cursor = name_end;
    while let Some(p) = body[cursor..].find("<parameter=") {
        let ps = cursor + p + "<parameter=".len();
        let pe = body[ps..].find('>')? + ps;
        let arg = body[ps..pe].trim().to_string();
        let ve = body[pe + 1..].find("</parameter>")? + pe + 1;
        let raw = body[pe + 1..ve].strip_prefix('\n').unwrap_or(&body[pe + 1..ve]);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let ty = schema.and_then(|s| s.pointer(&format!("/function/parameters/properties/{arg}/type"))).and_then(Value::as_str);
        let value = if ty == Some("string") { Value::String(raw.to_string()) } else { serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string())) };
        args.insert(arg, value);
        cursor = ve + "</parameter>".len();
    }
    Some(json!({"name": name, "arguments": Value::Object(args).to_string()}))
}

fn finish_reason(finish: Finish, tool_calls: bool) -> &'static str {
    match finish {
        _ if tool_calls => "tool_calls",
        Finish::Length => "length",
        Finish::Stop | Finish::Cancelled => "stop",
    }
}

async fn chat_completions(State(app): State<Arc<App>>, Json(req): Json<Value>) -> Response {
    let p = match prepare(&app, &req) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let id = format!("chatcmpl-{}", now());
    if p.stream { stream_response(app, p, id).await } else { full_response(app, p, id).await }
}

/// Appends `piece` to `text` and reports whether a stop string now occurs (text cut before it).
fn apply_stops(text: &mut String, stops: &[String]) -> bool {
    for s in stops.iter().filter(|s| !s.is_empty()) {
        if let Some(i) = text.find(s.as_str()) {
            text.truncate(i);
            return true;
        }
    }
    false
}

async fn full_response(app: Arc<App>, p: Prepared, id: String) -> Response {
    let (seg, n, finish, cached) = match generate(&app, &p, |_| {}).await {
        Ok(r) => r,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    let prompt_tokens = p.ids.len();
    let (reasoning, content, calls) = parse_output(&seg.text, p.thinking, p.tools.as_ref());
    let mut message = json!({"role": "assistant", "content": if content.is_empty() && !calls.is_empty() { Value::Null } else { Value::String(content) }});
    if let Some(r) = reasoning {
        message["reasoning_content"] = Value::String(r);
    }
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls.clone());
    }
    let reason = if seg.stopped { "stop" } else { finish_reason(finish, !calls.is_empty()) };
    Json(json!({
        "id": id, "object": "chat.completion", "created": now(), "model": app.name,
        "choices": [{"index": 0, "message": message, "finish_reason": reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    }))
    .into_response()
}

/// The end of the reasoning in a thinking turn's output: (start, end) of the `</think>` tag. The model
/// sometimes opens its reasoning with a literal `<thinking>` and closes it with `</thinking>` text,
/// never emitting `</think>`; then that closing tag ends the reasoning.
fn think_end(text: &str) -> Option<(usize, usize)> {
    let at = |i: usize, tag: &str| (i, i + tag.len());
    if let Some(i) = text.find("</think>") {
        return Some(at(i, "</think>"));
    }
    if text.trim_start().starts_with("<thinking>") {
        return text.find("</thinking>").map(|i| at(i, "</thinking>"));
    }
    None
}

/// A piece of streamed output.
enum Seg {
    Reasoning(String),
    Content(String),
    /// Prefill progress: (prompt tokens computed, total, served from cache, milliseconds since start).
    Prompt(usize, usize, usize, u64),
    /// Bytes of tool-call text generated so far (held back until the call is validated).
    ToolBytes(usize),
}

/// Splits the growing output text into streamable pieces: reasoning until `</think>` (thinking turns),
/// then content; text from `<tool_call>` on is held back for the final tool calls, and possible partial
/// tags and trailing reasoning whitespace wait for more text.
struct Segmenter {
    text: String,
    sent: usize,
    thinking: bool,
    in_reasoning: bool,
    content_started: bool,
    stopped: bool,
    tool_bytes: usize,
}

impl Segmenter {
    fn new(thinking: bool) -> Self {
        Self { text: String::new(), sent: 0, thinking, in_reasoning: thinking, content_started: false, stopped: false, tool_bytes: 0 }
    }

    fn feed(&mut self, piece: &str, stops: &[String]) -> Vec<Seg> {
        let mut out = Vec::new();
        if self.stopped {
            return out;
        }
        self.text.push_str(piece);
        if apply_stops(&mut self.text, stops) {
            self.stopped = true;
        }
        let (text, mut sent) = (&self.text, self.sent);
        loop {
            if self.in_reasoning {
                let tag = think_end(text).filter(|&(i, _)| i >= sent);
                let (end, done) = match tag {
                    Some((i, _)) => (i, true),
                    None => (text.len().saturating_sub("</thinking>".len()).max(sent), false),
                };
                let end = (sent..=end).rev().find(|&i| text.is_char_boundary(i)).unwrap_or(sent);
                let end = sent + text[sent..end].trim_end().len();
                if end > sent {
                    out.push(Seg::Reasoning(text[sent..end].to_string()));
                    sent = end;
                }
                if done {
                    sent = tag.map_or(sent, |(_, j)| j);
                    self.in_reasoning = false;
                    continue;
                }
            } else if !text[sent..].contains("<tool_call>") {
                if !self.content_started {
                    sent += text[sent..].len() - text[sent..].trim_start().len();
                }
                let end = text.len().saturating_sub("<tool_call>".len()).max(sent);
                let end = (sent..=end).rev().find(|&i| text.is_char_boundary(i) && !text[i..].starts_with('<')).unwrap_or(sent);
                let end = if self.stopped { text.len() } else { end };
                if end > sent {
                    out.push(Seg::Content(text[sent..end].to_string()));
                    sent = end;
                    self.content_started = true;
                }
            } else {
                let held = text.len() - (sent + text[sent..].find("<tool_call>").unwrap_or(0));
                if held != self.tool_bytes {
                    self.tool_bytes = held;
                    out.push(Seg::ToolBytes(held));
                }
            }
            break;
        }
        self.sent = sent;
        out
    }

    /// The content held back until the end (as a possible tag start), given the parsed content.
    fn tail(&self, content: &str) -> Option<String> {
        if self.in_reasoning {
            return None;
        }
        let text = &self.text;
        let body = if self.thinking { think_end(text).map(|(_, j)| &text[j..]).unwrap_or("") } else { text.as_str() };
        let lead = body.len() - body.trim_start().len();
        let body_start = text.len() - body.len() + lead;
        let shown = self.sent.saturating_sub(body_start).min(content.len());
        (shown < content.len() && content.is_char_boundary(shown)).then(|| content[shown..].to_string())
    }
}

/// Receives one generation job's tokens, feeding decoded text through the segmenter to `on_seg`;
/// returns (tokens, finish, cached prompt tokens) or an error message.
async fn drive(app: &App, rx: &mut tmpsc::UnboundedReceiver<Msg>, total_prompt: usize, stops: &[String], seg: &mut Segmenter, stream: &mut TextStream, on_seg: &mut impl FnMut(Seg)) -> Result<(usize, Finish, usize), String> {
    let mut n = 0usize;
    let start = std::time::Instant::now();
    while let Some(msg) = rx.recv().await {
        match msg {
            Msg::Token(t) => {
                n += 1;
                if seg.stopped {
                    continue;
                }
                let piece = stream.push(&app.chat, t).map_err(|e| format!("{e:#}"))?;
                for s in seg.feed(&piece, stops) {
                    on_seg(s);
                }
                if seg.stopped {
                    rx.close();
                }
            }
            Msg::Progress(done, cached) => on_seg(Seg::Prompt(done, total_prompt, cached, start.elapsed().as_millis() as u64)),
            Msg::Done(f, c) => return Ok((n, f, c)),
            Msg::Error(e) => return Err(e),
        }
    }
    Ok((n, Finish::Cancelled, 0))
}

/// Runs a prepared request to completion. When the output contains tool calls that fail validation,
/// the text before the first `<tool_call>` is kept and the call is regenerated from there (sampled, a
/// new seed per attempt) up to `tool_retries` times, so a malformed call is retried instead of being
/// returned as text. Returns (segmenter, generated tokens, finish, cached prompt tokens).
async fn generate(app: &App, p: &Prepared, mut on_seg: impl FnMut(Seg)) -> Result<(Segmenter, usize, Finish, usize), String> {
    let mut seg = Segmenter::new(p.thinking);
    let mut stream = TextStream::default();
    let (mut total, mut cached_first) = (0usize, None);
    let mut params = p.params.clone();
    let mut ids = p.ids.clone();
    for attempt in 0..=app.tool_retries {
        let (tx, mut rx) = tmpsc::unbounded_channel();
        if p.progress {
            let ptx = tx.clone();
            params.progress = Some(Progress(Arc::new(move |done, cached| {
                let _ = ptx.send(Msg::Progress(done, cached));
            })));
        }
        app.jobs.send(Work::Generate(Job { ids: ids.clone(), params: params.clone(), tx, capture: p.capture.clone() })).map_err(|_| "generation worker stopped".to_string())?;
        let (n, finish, cached) = drive(app, &mut rx, ids.len(), &p.stops, &mut seg, &mut stream, &mut on_seg).await?;
        total += n;
        cached_first.get_or_insert(cached);
        let (Some(tools), Some(tc)) = (p.tools.as_ref(), app.tool_call_token) else { return Ok((seg, total, finish, cached_first.unwrap_or(0))) };
        let body = if p.thinking { think_end(&seg.text).map(|(_, j)| &seg.text[j..]).unwrap_or("") } else { seg.text.as_str() };
        let problem = match validate_calls(body, tools) {
            // SQUASH_TEST_TOOL_RETRY: treat the first attempt's tool calls as invalid (exercises retries).
            Ok(()) if attempt == 0 && body.contains("<tool_call>") && std::env::var_os("SQUASH_TEST_TOOL_RETRY").is_some() => Some("forced by SQUASH_TEST_TOOL_RETRY".to_string()),
            Ok(()) => None,
            Err(e) => Some(e),
        };
        let Some(problem) = problem else { return Ok((seg, total, finish, cached_first.unwrap_or(0))) };
        let generated = stream.ids().to_vec();
        let Some(at) = generated.iter().position(|&t| t == tc) else { return Ok((seg, total, finish, cached_first.unwrap_or(0))) };
        if attempt == app.tool_retries || params.max_tokens <= n {
            eprintln!("tool call still invalid after {attempt} retries: {problem}");
            return Ok((seg, total, finish, cached_first.unwrap_or(0)));
        }
        eprintln!("tool call invalid ({problem}); regenerating it (attempt {})", attempt + 1);
        // Rewind to just before the tool call and continue from there.
        let prefix = &generated[..at];
        // Everything generated before <tool_call> (reasoning included) stays as shown.
        seg.text = app.chat.decode(prefix, true).map_err(|e| format!("{e:#}"))?;
        seg.sent = seg.sent.min(seg.text.len());
        seg.stopped = false;
        stream = TextStream::after(&app.chat, prefix).map_err(|e| format!("{e:#}"))?;
        ids = p.ids.iter().chain(prefix).copied().collect();
        params.max_tokens -= at.min(params.max_tokens);
        params.temperature = params.temperature.max(0.7);
        params.seed = params.seed.wrapping_add(0x9e37_79b9 * (attempt as u64 + 1));
    }
    unreachable!()
}

/// Checks every `<tool_call>` block: it parses, is closed, names a declared function, has the required
/// parameters, and its values match the schema types.
fn validate_calls(body: &str, tools: &Value) -> Result<(), String> {
    let mut cursor = 0;
    while let Some(start) = body[cursor..].find("<tool_call>") {
        let b0 = cursor + start + "<tool_call>".len();
        let end = body[b0..].find("</tool_call>").ok_or("unterminated <tool_call>")?;
        let block = &body[b0..b0 + end];
        let call = parse_tool_call(block, Some(tools)).ok_or("unparsable tool call")?;
        if !block.contains("</function>") && !block.trim_start().starts_with('{') {
            return Err("missing </function>".into());
        }
        let name = call["name"].as_str().unwrap_or("");
        let schema = tools
            .as_array()
            .and_then(|a| a.iter().find(|t| t.pointer("/function/name").and_then(Value::as_str) == Some(name)))
            .ok_or_else(|| format!("unknown function {name:?}"))?;
        let args: Value = serde_json::from_str(call["arguments"].as_str().unwrap_or("{}")).map_err(|e| format!("arguments: {e}"))?;
        let params = schema.pointer("/function/parameters");
        for req in params.and_then(|p| p.get("required")).and_then(Value::as_array).into_iter().flatten() {
            let r = req.as_str().unwrap_or("");
            if args.get(r).is_none() {
                return Err(format!("{name}: missing required parameter {r:?}"));
            }
        }
        if let Some(props) = params.and_then(|p| p.get("properties")).and_then(Value::as_object) {
            for (k, v) in args.as_object().into_iter().flatten() {
                let Some(ty) = props.get(k).and_then(|p| p.get("type")).and_then(Value::as_str) else { continue };
                let ok = match ty {
                    "string" => v.is_string(),
                    "integer" => v.is_i64() || v.is_u64(),
                    "number" => v.is_number(),
                    "boolean" => v.is_boolean(),
                    "array" => v.is_array(),
                    "object" => v.is_object(),
                    _ => true,
                };
                if !ok {
                    return Err(format!("{name}: parameter {k:?} is not a {ty}"));
                }
            }
        }
        cursor = b0 + end + "</tool_call>".len();
    }
    Ok(())
}

async fn stream_response(app: Arc<App>, p: Prepared, id: String) -> Response {
    let (out_tx, out_rx) = tmpsc::unbounded_channel::<Result<Event, std::convert::Infallible>>();
    tokio::spawn(async move {
        let created = now();
        let chunk = |delta: Value, finish: Option<&str>| {
            let data = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": app.name,
                              "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
            Ok(Event::default().data(data.to_string()))
        };
        let _ = out_tx.send(chunk(json!({"role": "assistant"}), None));
        let result = generate(&app, &p, |s| {
            let delta = match s {
                Seg::Reasoning(t) => json!({"reasoning_content": t}),
                Seg::Content(t) => json!({"content": t}),
                // llama.cpp-style extensions (requested with "return_progress"; harmless otherwise).
                Seg::Prompt(processed, total, cache, ms) => {
                    let c = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": app.name, "choices": [],
                                   "prompt_progress": {"total": total, "cache": cache, "processed": processed, "time_ms": ms}});
                    let _ = out_tx.send(Ok(Event::default().data(c.to_string())));
                    return;
                }
                Seg::ToolBytes(b) => {
                    let c = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": app.name,
                                   "choices": [{"index": 0, "delta": {}, "finish_reason": Value::Null}], "tool_call_progress": {"bytes": b}});
                    let _ = out_tx.send(Ok(Event::default().data(c.to_string())));
                    return;
                }
            };
            let _ = out_tx.send(chunk(delta, None));
        })
        .await;
        let (seg, n, finish, cached) = match result {
            Ok(r) => r,
            Err(e) => {
                let _ = out_tx.send(Ok(Event::default().data(json!({"error": {"message": e}}).to_string())));
                return;
            }
        };
        let (thinking, tools, prompt_tokens) = (p.thinking, p.tools.clone(), p.ids.len());
        let (_, content, calls) = parse_output(&seg.text, thinking, tools.as_ref());
        if let Some(t) = seg.tail(&content) {
            let _ = out_tx.send(chunk(json!({"content": t}), None));
        }
        if !calls.is_empty() {
            let deltas: Vec<Value> = calls.iter().enumerate().map(|(i, c)| json!({"index": i, "id": c["id"], "type": "function", "function": c["function"]})).collect();
            let _ = out_tx.send(chunk(json!({"tool_calls": deltas}), None));
        }
        let reason = if seg.stopped { "stop" } else { finish_reason(finish, !calls.is_empty()) };
        let mut last = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": app.name,
                              "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]});
        last["usage"] = json!({"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n,
                               "prompt_tokens_details": {"cached_tokens": cached}});
        let _ = out_tx.send(Ok(Event::default().data(last.to_string())));
        let _ = out_tx.send(Ok(Event::default().data("[DONE]")));
    });
    Sse::new(UnboundedReceiverStream::new(out_rx)).keep_alive(KeepAlive::default()).into_response()
}

// ---- Anthropic Messages API ---------------------------------------------------------------------

/// Text of an Anthropic content value (a string or text blocks).
fn anthropic_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks.iter().filter_map(|b| b.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

/// An Anthropic `/v1/messages` request as the equivalent chat-completions request.
fn from_anthropic(req: &Value) -> Value {
    let mut messages = Vec::new();
    if let Some(system) = req.get("system") {
        messages.push(json!({"role": "system", "content": anthropic_text(system)}));
    }
    for m in req.get("messages").and_then(Value::as_array).into_iter().flatten() {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        match m.get("content") {
            Some(Value::Array(blocks)) => {
                let (mut text, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
                // User content with images keeps its parts in order (OpenAI content-part form).
                let mut parts: Vec<Value> = Vec::new();
                for b in blocks {
                    match b.get("type").and_then(Value::as_str) {
                        Some("image") => {
                            if !text.is_empty() {
                                parts.push(json!({"type": "text", "text": std::mem::take(&mut text)}));
                            }
                            let src = b.get("source").cloned().unwrap_or(Value::Null);
                            let url = match src.get("type").and_then(Value::as_str) {
                                Some("base64") => format!("data:{};base64,{}", src.get("media_type").and_then(Value::as_str).unwrap_or("image/png"), src.get("data").and_then(Value::as_str).unwrap_or("")),
                                _ => src.get("url").and_then(Value::as_str).unwrap_or("").to_owned(),
                            };
                            parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
                        }
                        Some("text") => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
                        Some("thinking") => reasoning.push_str(b.get("thinking").and_then(Value::as_str).unwrap_or("")),
                        Some("tool_use") => calls.push(json!({"id": b["id"], "type": "function",
                            "function": {"name": b["name"], "arguments": b.get("input").cloned().unwrap_or(json!({}))}})),
                        Some("tool_result") => {
                            if !text.is_empty() {
                                messages.push(json!({"role": role, "content": std::mem::take(&mut text)}));
                            }
                            messages.push(json!({"role": "tool", "tool_call_id": b["tool_use_id"],
                                                 "content": anthropic_text(b.get("content").unwrap_or(&Value::Null))}));
                        }
                        _ => {}
                    }
                }
                if role == "assistant" {
                    let mut msg = json!({"role": "assistant", "content": text});
                    if !reasoning.is_empty() {
                        msg["reasoning_content"] = Value::String(reasoning);
                    }
                    if !calls.is_empty() {
                        msg["tool_calls"] = Value::Array(calls);
                    }
                    messages.push(msg);
                } else if !parts.is_empty() {
                    if !text.is_empty() {
                        parts.push(json!({"type": "text", "text": text}));
                    }
                    messages.push(json!({"role": role, "content": parts}));
                } else if !text.is_empty() {
                    messages.push(json!({"role": role, "content": text}));
                }
            }
            Some(c) => messages.push(json!({"role": role, "content": anthropic_text(c)})),
            None => {}
        }
    }
    let tools: Vec<Value> = req.get("tools").and_then(Value::as_array).into_iter().flatten()
        .map(|t| json!({"type": "function", "function": {"name": t["name"], "description": t.get("description").cloned().unwrap_or(Value::Null),
                                                         "parameters": t.get("input_schema").cloned().unwrap_or(json!({"type": "object"}))}}))
        .collect();
    let thinking = req.get("thinking").and_then(|t| t.get("type")).and_then(Value::as_str) == Some("enabled");
    let mut out = json!({"messages": messages,
                         "chat_template_kwargs": {"enable_thinking": thinking}, "stream": req.get("stream").cloned().unwrap_or(json!(false))});
    if !tools.is_empty() {
        out["tools"] = Value::Array(tools);
    }
    for (from, to) in [("max_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("stop_sequences", "stop"), ("steering", "steering")] {
        if let Some(v) = req.get(from) {
            out[to] = v.clone();
        }
    }
    out
}

fn stop_reason(finish: Finish, stopped: bool, tool_calls: bool) -> &'static str {
    match finish {
        _ if tool_calls => "tool_use",
        _ if stopped => "stop_sequence",
        Finish::Length => "max_tokens",
        _ => "end_turn",
    }
}

fn tool_use_blocks(calls: &[Value]) -> Vec<Value> {
    calls.iter().map(|c| {
        let input: Value = c["function"]["arguments"].as_str().and_then(|a| serde_json::from_str(a).ok()).unwrap_or(json!({}));
        json!({"type": "tool_use", "id": c["id"].as_str().unwrap_or("").replace("call_", "toolu_"), "name": c["function"]["name"], "input": input})
    }).collect()
}

async fn messages(State(app): State<Arc<App>>, Json(req): Json<Value>) -> Response {
    let converted = from_anthropic(&req);
    let p = match prepare(&app, &converted) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let prompt_tokens = p.ids.len();
    let id = format!("msg_{}", now());
    let (thinking, tools) = (p.thinking, p.tools.clone());
    if !p.stream {
        let (seg, n, finish, cached) = match generate(&app, &p, |_| {}).await {
            Ok(r) => r,
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        };
        let (reasoning, content, calls) = parse_output(&seg.text, thinking, tools.as_ref());
        let mut blocks = Vec::new();
        if let Some(r) = reasoning.filter(|r| !r.is_empty()) {
            blocks.push(json!({"type": "thinking", "thinking": r, "signature": ""}));
        }
        if !content.is_empty() {
            blocks.push(json!({"type": "text", "text": content}));
        }
        blocks.extend(tool_use_blocks(&calls));
        return Json(json!({"id": id, "type": "message", "role": "assistant", "model": app.name, "content": blocks,
                           "stop_reason": stop_reason(finish, seg.stopped, !calls.is_empty()), "stop_sequence": Value::Null,
                           "usage": {"input_tokens": prompt_tokens - cached, "cache_read_input_tokens": cached, "output_tokens": n}}))
        .into_response();
    }
    let (out_tx, out_rx) = tmpsc::unbounded_channel::<Result<Event, std::convert::Infallible>>();
    tokio::spawn(async move {
        let ev = |name: &str, data: Value| Ok(Event::default().event(name).data(data.to_string()));
        let _ = out_tx.send(ev("message_start", json!({"type": "message_start", "message": {"id": id, "type": "message", "role": "assistant",
            "model": app.name, "content": [], "stop_reason": Value::Null, "stop_sequence": Value::Null,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0}}})));
        // Blocks open as their kind changes: (index, kind) of the open block.
        let mut open: Option<(usize, bool)> = None;
        let mut next_index = 0usize;
        let mut emit = |reasoning: bool, text: String| {
            if open.map(|(_, r)| r) != Some(reasoning) {
                if let Some((i, _)) = open {
                    let _ = out_tx.send(ev("content_block_stop", json!({"type": "content_block_stop", "index": i})));
                }
                let block = if reasoning { json!({"type": "thinking", "thinking": ""}) } else { json!({"type": "text", "text": ""}) };
                let _ = out_tx.send(ev("content_block_start", json!({"type": "content_block_start", "index": next_index, "content_block": block})));
                open = Some((next_index, reasoning));
                next_index += 1;
            }
            let i = open.unwrap().0;
            let delta = if reasoning { json!({"type": "thinking_delta", "thinking": text}) } else { json!({"type": "text_delta", "text": text}) };
            let _ = out_tx.send(ev("content_block_delta", json!({"type": "content_block_delta", "index": i, "delta": delta})));
        };
        let result = generate(&app, &p, |s| match s {
            Seg::Reasoning(t) => emit(true, t),
            Seg::Content(t) => emit(false, t),
            Seg::Prompt(..) | Seg::ToolBytes(_) => {}
        })
        .await;
        let (seg, n, finish, _) = match result {
            Ok(r) => r,
            Err(e) => {
                let _ = out_tx.send(ev("error", json!({"type": "error", "error": {"type": "api_error", "message": e}})));
                return;
            }
        };
        let (_, content, calls) = parse_output(&seg.text, thinking, tools.as_ref());
        if let Some(t) = seg.tail(&content) {
            emit(false, t);
        }
        if let Some((i, _)) = open {
            let _ = out_tx.send(ev("content_block_stop", json!({"type": "content_block_stop", "index": i})));
        }
        for (k, block) in tool_use_blocks(&calls).into_iter().enumerate() {
            let i = next_index + k;
            let input = block["input"].to_string();
            let mut start = block.clone();
            start["input"] = json!({});
            let _ = out_tx.send(ev("content_block_start", json!({"type": "content_block_start", "index": i, "content_block": start})));
            let _ = out_tx.send(ev("content_block_delta", json!({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": input}})));
            let _ = out_tx.send(ev("content_block_stop", json!({"type": "content_block_stop", "index": i})));
        }
        let _ = out_tx.send(ev("message_delta", json!({"type": "message_delta", "delta": {"stop_reason": stop_reason(finish, seg.stopped, !calls.is_empty()), "stop_sequence": Value::Null},
                                                        "usage": {"output_tokens": n}})));
        let _ = out_tx.send(ev("message_stop", json!({"type": "message_stop"})));
    });
    Sse::new(UnboundedReceiverStream::new(out_rx)).keep_alive(KeepAlive::default()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Value {
        json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}])
    }

    #[test]
    fn literal_thinking_tags_end_reasoning() {
        let text = "<thinking>\nplan it\n</thinking>\n## Goal\nsummary";
        let (r, c, _) = parse_output(text, true, None);
        assert_eq!(r.as_deref(), Some("<thinking>\nplan it"));
        assert_eq!(c, "## Goal\nsummary");
        let mut seg = Segmenter::new(true);
        let mut streamed = String::new();
        for piece in text.split_inclusive(' ').flat_map(|w| w.split_inclusive('\n')) {
            for sg in seg.feed(piece, &[]) {
                if let Seg::Content(t) = sg {
                    streamed.push_str(&t);
                }
            }
        }
        streamed.push_str(&seg.tail(&c).unwrap_or_default());
        assert_eq!(streamed, c);
        // The real tag wins, and a literal tag without a literal opening stays reasoning.
        assert_eq!(parse_output("a</thinking>b</think>c", true, None).1, "c");
        assert_eq!(parse_output("a</thinking>b", true, None).1, "");
    }

    #[test]
    fn tool_call_validation() {
        let ok = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        assert!(validate_calls(ok, &tools()).is_ok());
        assert!(validate_calls("plain answer", &tools()).is_ok());
        let unterminated = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>";
        assert!(validate_calls(unterminated, &tools()).is_err());
        let unknown = "<tool_call>\n<function=get_time>\n</function>\n</tool_call>";
        assert!(validate_calls(unknown, &tools()).unwrap_err().contains("unknown function"));
        let missing = "<tool_call>\n<function=get_weather>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        assert!(validate_calls(missing, &tools()).unwrap_err().contains("missing required"));
        let mistyped = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n<parameter=days>\nthree\n</parameter>\n</function>\n</tool_call>";
        assert!(validate_calls(mistyped, &tools()).unwrap_err().contains("not a integer"));
        let json_style = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call>";
        assert!(validate_calls(json_style, &tools()).is_ok(), "{:?}", validate_calls(json_style, &tools()));
    }
}

/// Writes an activation capture as safetensors: per site and layer `layers.{L}.{name}.{site}` tensors
/// (premix F32 [4, 2560], mix BF16 [2560], block F32 [2560], moe F32 [2560], router_logits F32 [512],
/// router_top_ids I32 [10], router_top_weights F32 [10]), `top_ids.{site}` / `top_logprobs.{site}` [20],
/// and JSON metadata (request, prompt and completion ids, positions, fire count).
fn write_capture(path: &std::path::Path, meta: &Value, generated: &[i32], c: &squash_engine::session::CaptureResult) -> Result<()> {
    use squash_engine::forward::CaptureLayout as L;
    let fields: [(&str, &str, usize, Vec<usize>, usize); 7] = [
        ("premix", "F32", L::PREMIX, vec![4, 2560], L::MIX - L::PREMIX),
        ("mix", "BF16", L::MIX, vec![2560], L::BLOCK - L::MIX),
        ("block", "F32", L::BLOCK, vec![2560], L::MOE - L::BLOCK),
        ("moe", "F32", L::MOE, vec![2560], L::ROUTER - L::MOE),
        ("router_logits", "F32", L::ROUTER, vec![512], L::TOP_I - L::ROUTER),
        ("router_top_ids", "I32", L::TOP_I, vec![10], L::TOP_W - L::TOP_I),
        ("router_top_weights", "F32", L::TOP_W, vec![10], L::BYTES - L::TOP_W),
    ];
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    let add = |name: String, dtype: &str, shape: Vec<usize>, bytes: &[u8], header: &mut serde_json::Map<String, Value>, data: &mut Vec<u8>| {
        header.insert(name, json!({"dtype": dtype, "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]}));
        data.extend_from_slice(bytes);
    };
    let layers = squash_engine::model::LAYERS;
    for site in &c.sites {
        for l in 0..layers {
            let rec = &site.records[l * L::BYTES..(l + 1) * L::BYTES];
            for (name, dtype, off, shape, len) in &fields {
                add(format!("layers.{l}.{name}.{}", site.name), dtype, shape.clone(), &rec[*off..off + len], &mut header, &mut data);
            }
        }
        let ids: Vec<u8> = site.top_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        let lps: Vec<u8> = site.top_logprobs.iter().flat_map(|v| v.to_le_bytes()).collect();
        add(format!("top_ids.{}", site.name), "I32", vec![site.top_ids.len()], &ids, &mut header, &mut data);
        add(format!("top_logprobs.{}", site.name), "F32", vec![site.top_logprobs.len()], &lps, &mut header, &mut data);
    }
    let mut m = meta.clone();
    m["completion_ids"] = json!(generated);
    m["sites"] = Value::Array(c.sites.iter().map(|s| json!({"site": s.name, "position": s.position, "token": s.token})).collect());
    m["fire_count"] = json!(layers * c.sites.len());
    header.insert("__metadata__".into(), json!({"squash.capture": m.to_string()}));
    let mut h = Value::Object(header).to_string().into_bytes();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let tmp = path.with_extension("tmp");
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(&(h.len() as u64).to_le_bytes())?;
        f.write_all(&h)?;
        f.write_all(&data)?;
        f.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}
