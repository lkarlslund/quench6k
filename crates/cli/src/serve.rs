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
use squash_engine::session::{Finish, GenParams, Session};
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
}

/// A queued generation and where its tokens go.
struct Job {
    ids: Vec<i32>,
    params: GenParams,
    tx: tmpsc::UnboundedSender<Msg>,
}

enum Msg {
    Token(i32),
    /// Finished, with the prompt tokens served from the prefix cache.
    Done(Finish, usize),
    Error(String),
}

struct App {
    chat: Chat,
    jobs: mpsc::Sender<Job>,
    name: String,
    /// Steering strength used when a request carries none (the loaded pack's, or 0).
    default_strength: f32,
    steering: bool,
}

pub fn run(opts: ServeOptions) -> Result<()> {
    let chat = Chat::load(&opts.model)?;
    let (jobs_tx, jobs_rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<f32>>();
    let (model_dir, context, drafts, steer) = (opts.model.clone(), opts.context, opts.drafts, opts.steer.clone());
    let steering = steer.steering.is_some();
    std::thread::spawn(move || {
        let setup = (|| -> Result<_> {
            let (mut model, stream) = crate::load(&model_dir, Precision::Int8Group, true)?;
            let strength = steer.attach(&mut model, &stream)?;
            let session = Session::new(&model, &stream, context, 8192, drafts)?;
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
        for job in jobs_rx {
            let tx = job.tx.clone();
            let result = session.generate(&model, &stream, &job.ids, &job.params, |tok| tx.send(Msg::Token(tok)).is_ok());
            let _ = match result {
                Ok((finish, _)) => job.tx.send(Msg::Done(finish, session.reused)),
                Err(e) => job.tx.send(Msg::Error(format!("{e:#}"))),
            };
        }
    });
    let default_strength = ready_rx.recv().context("worker thread")??;
    let app = Arc::new(App { chat, jobs: jobs_tx, name: opts.name, default_strength, steering });
    let router = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .with_state(app);
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(opts.listen).await?;
        println!("serving on http://{}", opts.listen);
        axum::serve(listener, router).await?;
        Ok(())
    })
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
    ids: Vec<i32>,
    params: GenParams,
    stops: Vec<String>,
    thinking: bool,
    tools: Option<Value>,
    stream: bool,
}

fn prepare(app: &App, req: &Value) -> Result<Prepared, Response> {
    let messages = req.get("messages").filter(|m| m.is_array()).ok_or_else(|| error(StatusCode::BAD_REQUEST, "messages must be an array"))?;
    let kwargs = req.get("chat_template_kwargs").cloned().unwrap_or(Value::Null);
    let enable_thinking = kwargs.get("enable_thinking").and_then(Value::as_bool);
    let effort = kwargs.get("reasoning_effort").or_else(|| req.get("reasoning_effort")).and_then(Value::as_str);
    let tools = req.get("tools").filter(|t| t.as_array().is_some_and(|a| !a.is_empty())).cloned();
    let text = app.chat.render(messages, tools.as_ref(), enable_thinking, effort).map_err(|e| error(StatusCode::BAD_REQUEST, format!("chat template: {e:#}")))?;
    let thinking = text.ends_with("<think>\n");
    let ids = app.chat.encode(&text).map_err(|e| error(StatusCode::BAD_REQUEST, format!("{e:#}")))?;
    let max_tokens = req.get("max_completion_tokens").or_else(|| req.get("max_tokens")).and_then(Value::as_u64).unwrap_or(4096) as usize;
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
    let params = GenParams { max_tokens, temperature, top_p, top_k, seed, stop_tokens: app.chat.stop_tokens.clone(), steering_strength, boundaries: app.chat.boundaries(&ids) };
    Ok(Prepared { ids, params, stops, thinking, tools, stream: req.get("stream").and_then(Value::as_bool).unwrap_or(false) })
}

/// Splits generated text into (reasoning, content, tool calls).
fn parse_output(text: &str, thinking: bool, tools: Option<&Value>) -> (Option<String>, String, Vec<Value>) {
    let (reasoning, rest) = if thinking {
        match text.find("</think>") {
            Some(i) => (Some(text[..i].trim().to_string()), text[i + "</think>".len()..].trim_start()),
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

/// `<function=NAME>` then `<parameter=ARG>\nvalue\n</parameter>` blocks. Values are kept as strings
/// when the tool schema says so, otherwise parsed as JSON when they are valid JSON.
fn parse_tool_call(body: &str, tools: Option<&Value>) -> Option<Value> {
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
    let (tx, rx) = tmpsc::unbounded_channel();
    let prompt_tokens = p.ids.len();
    if app.jobs.send(Job { ids: p.ids, params: p.params, tx }).is_err() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "generation worker stopped");
    }
    let id = format!("chatcmpl-{}", now());
    if p.stream { stream_response(app, rx, id, prompt_tokens, p.stops, p.thinking, p.tools).await } else { full_response(app, rx, id, prompt_tokens, p.stops, p.thinking, p.tools).await }
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

async fn full_response(app: Arc<App>, mut rx: tmpsc::UnboundedReceiver<Msg>, id: String, prompt_tokens: usize, stops: Vec<String>, thinking: bool, tools: Option<Value>) -> Response {
    let mut stream = TextStream::default();
    let (mut text, mut n, mut finish) = (String::new(), 0usize, Finish::Stop);
    let mut stopped = false;
    let mut cached = 0usize;
    while let Some(msg) = rx.recv().await {
        match msg {
            Msg::Token(t) => {
                n += 1;
                if !stopped {
                    match stream.push(&app.chat, t) {
                        Ok(piece) => text.push_str(&piece),
                        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
                    }
                    if apply_stops(&mut text, &stops) {
                        stopped = true;
                        rx.close();
                    }
                }
            }
            Msg::Done(f, c) => {
                finish = f;
                cached = c;
                break;
            }
            Msg::Error(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    }
    let (reasoning, content, calls) = parse_output(&text, thinking, tools.as_ref());
    let mut message = json!({"role": "assistant", "content": if content.is_empty() && !calls.is_empty() { Value::Null } else { Value::String(content) }});
    if let Some(r) = reasoning {
        message["reasoning_content"] = Value::String(r);
    }
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls.clone());
    }
    let reason = if stopped { "stop" } else { finish_reason(finish, !calls.is_empty()) };
    Json(json!({
        "id": id, "object": "chat.completion", "created": now(), "model": app.name,
        "choices": [{"index": 0, "message": message, "finish_reason": reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    }))
    .into_response()
}

/// A piece of streamed output.
enum Seg {
    Reasoning(String),
    Content(String),
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
}

impl Segmenter {
    fn new(thinking: bool) -> Self {
        Self { text: String::new(), sent: 0, thinking, in_reasoning: thinking, content_started: false, stopped: false }
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
                let (end, done) = match text[sent..].find("</think>") {
                    Some(i) => (sent + i, true),
                    None => (text.len().saturating_sub("</think>".len()).max(sent), false),
                };
                let end = (sent..=end).rev().find(|&i| text.is_char_boundary(i)).unwrap_or(sent);
                let end = sent + text[sent..end].trim_end().len();
                if end > sent {
                    out.push(Seg::Reasoning(text[sent..end].to_string()));
                    sent = end;
                }
                if done {
                    sent = text[sent..].find("</think>").map(|i| sent + i).unwrap_or(sent) + "</think>".len();
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
        let body = if self.thinking { text.find("</think>").map(|i| &text[i + "</think>".len()..]).unwrap_or("") } else { text.as_str() };
        let lead = body.len() - body.trim_start().len();
        let body_start = text.len() - body.len() + lead;
        let shown = self.sent.saturating_sub(body_start).min(content.len());
        (shown < content.len() && content.is_char_boundary(shown)).then(|| content[shown..].to_string())
    }
}

/// Receives the generation, feeding decoded text to `on_seg`; returns (segmenter, tokens, finish,
/// cached prompt tokens) or an error message.
async fn drive(app: &App, rx: &mut tmpsc::UnboundedReceiver<Msg>, stops: &[String], thinking: bool, mut on_seg: impl FnMut(Seg)) -> Result<(Segmenter, usize, Finish, usize), String> {
    let mut stream = TextStream::default();
    let mut seg = Segmenter::new(thinking);
    let mut n = 0usize;
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
            Msg::Done(f, c) => return Ok((seg, n, f, c)),
            Msg::Error(e) => return Err(e),
        }
    }
    Ok((seg, n, Finish::Cancelled, 0))
}

async fn stream_response(app: Arc<App>, mut rx: tmpsc::UnboundedReceiver<Msg>, id: String, prompt_tokens: usize, stops: Vec<String>, thinking: bool, tools: Option<Value>) -> Response {
    let (out_tx, out_rx) = tmpsc::unbounded_channel::<Result<Event, std::convert::Infallible>>();
    tokio::spawn(async move {
        let created = now();
        let chunk = |delta: Value, finish: Option<&str>| {
            let data = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": app.name,
                              "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
            Ok(Event::default().data(data.to_string()))
        };
        let _ = out_tx.send(chunk(json!({"role": "assistant"}), None));
        let result = drive(&app, &mut rx, &stops, thinking, |s| {
            let delta = match s {
                Seg::Reasoning(t) => json!({"reasoning_content": t}),
                Seg::Content(t) => json!({"content": t}),
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
                for b in blocks {
                    match b.get("type").and_then(Value::as_str) {
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
    let mut out = json!({"messages": messages, "max_tokens": req.get("max_tokens").cloned().unwrap_or(json!(4096)),
                         "chat_template_kwargs": {"enable_thinking": thinking}, "stream": req.get("stream").cloned().unwrap_or(json!(false))});
    if !tools.is_empty() {
        out["tools"] = Value::Array(tools);
    }
    for (from, to) in [("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("stop_sequences", "stop"), ("steering", "steering")] {
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
    let (tx, mut rx) = tmpsc::unbounded_channel();
    let prompt_tokens = p.ids.len();
    if app.jobs.send(Job { ids: p.ids, params: p.params, tx }).is_err() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "generation worker stopped");
    }
    let id = format!("msg_{}", now());
    let (thinking, tools, stops) = (p.thinking, p.tools, p.stops);
    if !p.stream {
        let (seg, n, finish, cached) = match drive(&app, &mut rx, &stops, thinking, |_| {}).await {
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
        let result = drive(&app, &mut rx, &stops, thinking, |s| match s {
            Seg::Reasoning(t) => emit(true, t),
            Seg::Content(t) => emit(false, t),
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
