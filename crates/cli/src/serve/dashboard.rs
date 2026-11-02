//! Embedded dashboard and bounded, server-side wall-clock telemetry. The sampler never touches CUDA.
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{App, Metrics, model_names};

const HISTORY_SECONDS: usize = 3600;
const PHASES: [&str; 5] = ["idle", "prefill", "decode", "error", "unavailable"];

#[derive(Default)]
pub(super) struct LiveSlot {
    phase: AtomicU64,
    prompt: AtomicU64,
    cached: AtomicU64,
    progress: AtomicU64,
    generated: AtomicU64,
    // Cumulative live counters; completed-generation counters retain their existing semantics.
    prefilled_total: AtomicU64,
    generated_total: AtomicU64,
}

impl LiveSlot {
    pub(super) fn begin(&self, prompt: usize) {
        self.prompt.store(prompt as u64, Ordering::Relaxed);
        self.cached.store(0, Ordering::Relaxed);
        self.progress.store(0, Ordering::Relaxed);
        self.generated.store(0, Ordering::Relaxed);
        self.phase.store(1, Ordering::Relaxed);
    }

    pub(super) fn prefill(&self, done: usize, cached: usize) {
        let previous = self.progress.swap(done as u64, Ordering::Relaxed);
        self.cached.store(cached as u64, Ordering::Relaxed);
        self.prefilled_total.fetch_add(
            (done as u64).saturating_sub(previous.max(cached as u64)),
            Ordering::Relaxed,
        );
    }

    pub(super) fn decoding(&self) {
        self.phase.store(2, Ordering::Relaxed);
    }

    pub(super) fn output(&self) {
        self.generated.fetch_add(1, Ordering::Relaxed);
        self.generated_total.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn end(&self, failed: bool) {
        self.phase
            .store(if failed { 3 } else { 0 }, Ordering::Relaxed);
    }

    pub(super) fn unavailable(&self) {
        self.phase.store(4, Ordering::Relaxed);
    }
}

#[derive(Clone, Serialize)]
struct Sample {
    timestamp: u64,
    output_tps: Vec<f64>,
    prefill_tps: Vec<f64>,
    kv_bytes: Vec<u64>,
    prefilling: usize,
    decoding: usize,
    queued: u64,
}

#[derive(Default)]
pub(super) struct History {
    samples: VecDeque<Sample>,
    previous: Vec<(u64, u64)>,
    elapsed: f64,
}

impl History {
    fn record(&mut self, metrics: &Metrics, timestamp: u64, elapsed: f64) {
        let dt = elapsed - self.elapsed;
        let mut sample = Sample {
            timestamp,
            output_tps: Vec::new(),
            prefill_tps: Vec::new(),
            kv_bytes: Vec::new(),
            prefilling: 0,
            decoding: 0,
            queued: metrics.queued.load(Ordering::Relaxed),
        };
        let mut counters = Vec::with_capacity(metrics.live.len());
        for (i, slot) in metrics.live.iter().enumerate() {
            let output = slot.generated_total.load(Ordering::Relaxed);
            let prefill = slot.prefilled_total.load(Ordering::Relaxed);
            let (old_output, old_prefill) =
                self.previous.get(i).copied().unwrap_or((output, prefill));
            let rate = |current: u64, old: u64| {
                if dt > 0.0 {
                    (current.saturating_sub(old) as f64 / dt * 100.0).round() / 100.0
                } else {
                    0.0
                }
            };
            sample.output_tps.push(rate(output, old_output));
            sample.prefill_tps.push(rate(prefill, old_prefill));
            sample
                .kv_bytes
                .push(metrics.kv_bytes[i].load(Ordering::Relaxed));
            match slot.phase.load(Ordering::Relaxed) {
                1 => sample.prefilling += 1,
                2 => sample.decoding += 1,
                _ => {}
            }
            counters.push((output, prefill));
        }
        self.previous = counters;
        self.elapsed = elapsed;
        self.samples.push_back(sample);
        while self.samples.len() > HISTORY_SECONDS + 1 {
            self.samples.pop_front();
        }
    }
}

fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub(super) fn start(metrics: Arc<Metrics>) {
    std::thread::Builder::new()
        .name("dashboard-sampler".into())
        .spawn(move || {
            let epoch = Instant::now();
            loop {
                // No locks are taken by the inference worker; only this thread and HTTP readers use history.
                metrics
                    .history
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .record(&metrics, timestamp(), epoch.elapsed().as_secs_f64());
                std::thread::sleep(Duration::from_secs(1));
            }
        })
        .expect("dashboard sampler thread");
}

#[derive(Default, Deserialize)]
pub(super) struct Since {
    since: Option<u64>,
}

pub(super) async fn stats(State(app): State<Arc<App>>, Query(query): Query<Since>) -> Response {
    let metrics = &app.metrics;
    let (samples, latest) = {
        let history = metrics.history.lock().unwrap_or_else(|e| e.into_inner());
        let samples: Vec<_> = history
            .samples
            .iter()
            .filter(|s| query.since.is_none_or(|since| s.timestamp > since))
            .cloned()
            .collect();
        (samples, history.samples.back().cloned())
    };
    let slots: Vec<_> = metrics
        .live
        .iter()
        .enumerate()
        .map(|(i, slot)| {
            let phase = slot.phase.load(Ordering::Relaxed).min(4) as usize;
            json!({
                "id": i, "phase": PHASES[phase],
                "prompt_tokens": slot.prompt.load(Ordering::Relaxed),
                "cached_tokens": slot.cached.load(Ordering::Relaxed),
                "prefilled_tokens": slot.progress.load(Ordering::Relaxed),
                "generated_tokens": slot.generated.load(Ordering::Relaxed),
                "kv_bytes": metrics.kv_bytes[i].load(Ordering::Relaxed),
                "output_tps": latest.as_ref().map_or(0.0, |s| s.output_tps[i]),
                "prefill_tps": latest.as_ref().map_or(0.0, |s| s.prefill_tps[i]),
            })
        })
        .collect();
    let sum = |values: &[AtomicU64]| {
        values
            .iter()
            .map(|v| v.load(Ordering::Relaxed))
            .sum::<u64>()
    };
    let value = json!({
        "server": {
            "name": app.name, "models": model_names(&app).collect::<Vec<_>>(),
            "started_at": app.started * 1000, "context": app.context,
            "slots": metrics.live.len(), "history_seconds": HISTORY_SECONDS,
            "sample_interval_ms": 1000, "kv_memory_available": !app.flint,
            "kv_format": if quench6k_engine::forward::KV_FP8.load(Ordering::Relaxed) { "FP8" } else { "BF16" },
        },
        "totals": {
            "generations": sum(&metrics.generations),
            "prompt_tokens": sum(&metrics.prompt_tokens), "cached_tokens": sum(&metrics.cached_tokens),
            "generated_tokens": metrics.live.iter().map(|s| s.generated_total.load(Ordering::Relaxed)).sum::<u64>(),
            "prefilled_tokens": metrics.live.iter().map(|s| s.prefilled_total.load(Ordering::Relaxed)).sum::<u64>(),
            "errors": metrics.errors.load(Ordering::Relaxed),
        },
        "current": { "timestamp": timestamp(), "queued": metrics.queued.load(Ordering::Relaxed), "slots": slots },
        "samples": samples,
    });
    ([(header::CACHE_CONTROL, "no-store")], Json(value)).into_response()
}

fn asset_response(content_type: &'static str, body: &'static [u8]) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "no-cache"),
            (
                header::HeaderName::from_static("x-content-type-options"),
                "nosniff",
            ),
        ],
        body,
    )
        .into_response()
}

pub(super) async fn index() -> Response {
    asset_response("text/html; charset=utf-8", include_bytes!("ui/index.html"))
}

pub(super) async fn asset(Path(file): Path<String>) -> Response {
    match file.as_str() {
        "dashboard.css" => asset_response(
            "text/css; charset=utf-8",
            include_bytes!("ui/dashboard.css"),
        ),
        "dashboard.js" => asset_response(
            "text/javascript; charset=utf-8",
            include_bytes!("ui/dashboard.js"),
        ),
        "alpine.min.js" => asset_response(
            "text/javascript; charset=utf-8",
            include_bytes!("ui/vendor/alpine.min.js"),
        ),
        "echarts.min.js" => asset_response(
            "text/javascript; charset=utf-8",
            include_bytes!("ui/vendor/echarts.min.js"),
        ),
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_rates_exclude_reused_tokens_and_include_inflight_output() {
        let metrics = Metrics::new(2);
        let mut history = History::default();
        history.record(&metrics, 1000, 0.0);
        metrics.live[0].begin(1000);
        metrics.live[0].prefill(900, 900);
        metrics.live[0].prefill(1000, 900);
        metrics.live[0].decoding();
        for _ in 0..10 {
            metrics.live[0].output();
        }
        metrics.live[1].begin(200);
        metrics.live[1].prefill(200, 0);
        metrics.queued.store(3, Ordering::Relaxed);
        history.record(&metrics, 3000, 2.0);
        let sample = history.samples.back().unwrap();
        assert_eq!(sample.output_tps, [5.0, 0.0]);
        assert_eq!(sample.prefill_tps, [50.0, 100.0]);
        assert_eq!(
            (sample.prefilling, sample.decoding, sample.queued),
            (1, 1, 3)
        );
        assert_eq!(metrics.generations[0].load(Ordering::Relaxed), 0);
        // Request reuse resets only current-request state, never the cumulative live counters.
        metrics.live[0].end(false);
        metrics.live[0].begin(100);
        metrics.live[0].prefill(100, 100);
        history.record(&metrics, 4000, 3.0);
        assert_eq!(history.samples.back().unwrap().prefill_tps[0], 0.0);
    }

    #[test]
    fn history_is_bounded_and_idle_slots_return_to_zero() {
        let metrics = Metrics::new(1);
        let mut history = History::default();
        for t in 0..HISTORY_SECONDS + 5 {
            history.record(&metrics, t as u64 * 1000, t as f64);
        }
        assert_eq!(history.samples.len(), HISTORY_SECONDS + 1);
        assert_eq!(history.samples.front().unwrap().timestamp, 4000);
        let sample = history.samples.back().unwrap();
        assert_eq!(sample.output_tps, [0.0]);
        assert_eq!((sample.prefilling, sample.decoding), (0, 0));
    }
}
