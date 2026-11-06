//! Durable minute telemetry. Only the sampler and blocking HTTP queries access storage.
use super::{Metrics, Sample, timestamp};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tsink::{
    Aggregation, DataPoint, Label, QueryOptions, Row, Storage, StorageBuilder, TimestampPrecision,
    WalSyncMode,
};

macro_rules! totals {
    ($($field:ident),+ $(,)?) => {
        #[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
        #[serde(default)]
        pub(super) struct Totals { $(pub $field: u64),+ }
        impl Totals {
            pub fn add(&self, other: &Self) -> Self { Self { $($field: self.$field.saturating_add(other.$field)),+ } }
            pub fn difference(&self, old: &Self) -> Self { Self { $($field: self.$field.saturating_sub(old.$field)),+ } }
            fn rows(&self, t: i64) -> Vec<Row> { vec![$(Row::new(concat!("total_", stringify!($field)), DataPoint::new(t, self.$field))),+] }
            fn set(&mut self, name: &str, value: u64) { match name { $(concat!("total_", stringify!($field)) => self.$field = value,)+ _ => {} } }
        }
    }
}
totals!(
    requests,
    generations,
    prompt_tokens,
    cached_tokens,
    generated_tokens,
    prefilled_tokens,
    errors
);
impl Totals {
    pub fn read(metrics: &Metrics) -> Self {
        let sum = |values: &[std::sync::atomic::AtomicU64]| {
            values.iter().map(|v| v.load(Ordering::Relaxed)).sum()
        };
        Self {
            requests: metrics.requests.load(Ordering::Relaxed),
            generations: sum(&metrics.generations),
            prompt_tokens: sum(&metrics.prompt_tokens),
            cached_tokens: sum(&metrics.cached_tokens),
            generated_tokens: metrics
                .live
                .iter()
                .map(|s| s.generated_total.load(Ordering::Relaxed))
                .sum(),
            prefilled_tokens: metrics
                .live
                .iter()
                .map(|s| s.prefilled_total.load(Ordering::Relaxed))
                .sum(),
            errors: metrics.errors.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Default, Deserialize, Serialize)]
struct Bucket {
    sample: Sample,
    seconds: f64,
}
impl Bucket {
    fn push(&mut self, s: &Sample, seconds: f64) {
        self.sample.timestamp = s.timestamp / 60000 * 60000;
        self.seconds += seconds;
        for (dest, values) in [
            (&mut self.sample.output_tps, &s.output_tps),
            (&mut self.sample.prefill_tps, &s.prefill_tps),
        ] {
            dest.resize(dest.len().max(values.len()), 0.0);
            for (i, value) in values.iter().enumerate() {
                dest[i] += value * seconds;
            }
        }
        self.sample
            .kv_bytes
            .resize(self.sample.kv_bytes.len().max(s.kv_bytes.len()), 0);
        for (i, v) in s.kv_bytes.iter().enumerate() {
            self.sample.kv_bytes[i] = self.sample.kv_bytes[i].max(*v);
        }
        self.sample.prefilling = self.sample.prefilling.max(s.prefilling);
        self.sample.decoding = self.sample.decoding.max(s.decoding);
        self.sample.queued = self.sample.queued.max(s.queued);
        self.sample.totals = self.sample.totals.add(&s.totals);
    }
    fn sample(&self) -> Sample {
        let mut s = self.sample.clone();
        s.seconds = self.seconds;
        for values in [&mut s.output_tps, &mut s.prefill_tps] {
            for v in values {
                *v /= self.seconds.max(1e-9);
            }
        }
        s
    }
}

#[derive(Default, Deserialize, Serialize)]
struct Checkpoint {
    version: u32,
    since: u64,
    totals: Totals,
    pending: Option<Bucket>,
}

pub(in super::super) struct Archive {
    db: Arc<dyn Storage>,
    _lock: std::fs::File,
    known_slots: std::sync::atomic::AtomicUsize,
    background_errors: std::sync::atomic::AtomicU64,
    pub(super) base: Totals,
    pub since: u64,
    pub retention_seconds: u64,
    pub error: Mutex<Option<String>>,
    pub queries: Arc<tokio::sync::Semaphore>,
    pending: Mutex<Option<Sample>>,
}

pub(super) struct Writer {
    archive: Arc<Archive>,
    dir: PathBuf,
    checkpoint: Checkpoint,
}

pub(super) fn default_dir() -> Result<PathBuf> {
    let root = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))
        .context("set --stats-dir or HOME/XDG_STATE_HOME for durable statistics")?;
    Ok(root.join("quench6k/stats"))
}

impl Writer {
    pub(super) fn open(dir: PathBuf, days: u32) -> Result<(Arc<Archive>, Self)> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create stats directory {}", dir.display()))?;
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(dir.join("LOCK"))?;
        lock.try_lock().context(
            "statistics directory is already in use; choose a distinct --stats-dir for each server",
        )?;
        let mut checkpoint = match std::fs::read(dir.join("checkpoint.json")) {
            Ok(data) => {
                let c: Checkpoint =
                    serde_json::from_slice(&data).context("read statistics checkpoint")?;
                anyhow::ensure!(
                    c.version == 1,
                    "unsupported statistics checkpoint version {}",
                    c.version
                );
                c
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Checkpoint {
                version: 1,
                since: timestamp(),
                ..Default::default()
            },
            Err(e) => return Err(e.into()),
        };
        let retention_seconds = u64::from(days) * 86400;
        // A dormant server can outlive its graph retention; lifetime totals still survive.
        if checkpoint.pending.as_ref().is_some_and(|b| {
            b.sample.timestamp < timestamp().saturating_sub(retention_seconds * 1000)
        }) {
            checkpoint.pending = None;
        }
        let db = StorageBuilder::new()
            .with_data_path(dir.join("series"))
            .with_timestamp_precision(TimestampPrecision::Milliseconds)
            .with_retention(Duration::from_secs(retention_seconds))
            .with_memory_limit(32 << 20)
            .with_max_writers(1)
            .with_wal_size_limit(64 << 20)
            .with_wal_sync_mode(WalSyncMode::PerAppend)
            // A failed flush or compaction pass is retried on the next pass; by default tsink
            // instead refuses all later writes until it is reopened.
            .with_background_fail_fast(false)
            .build()
            .context("open performance time-series database")?;
        let stored_slots = db
            .list_metrics()?
            .iter()
            .flat_map(|s| s.labels.iter())
            .filter(|l| l.name == "slot")
            .filter_map(|l| l.value.parse::<usize>().ok())
            .map(|i| i + 1)
            .max()
            .unwrap_or(0);
        let known_slots = stored_slots.max(
            checkpoint
                .pending
                .as_ref()
                .map_or(0, |b| b.sample.output_tps.len()),
        );
        let archive = Arc::new(Archive {
            db,
            _lock: lock,
            known_slots: std::sync::atomic::AtomicUsize::new(known_slots),
            background_errors: std::sync::atomic::AtomicU64::new(0),
            base: checkpoint.totals.clone(),
            since: checkpoint.since,
            retention_seconds,
            error: Mutex::new(None),
            queries: Arc::new(tokio::sync::Semaphore::new(2)),
            pending: Mutex::new(checkpoint.pending.as_ref().map(Bucket::sample)),
        });
        let writer = Self {
            archive: archive.clone(),
            dir,
            checkpoint,
        };
        writer.save()?;
        Ok((archive, writer))
    }

    pub(super) fn record(&mut self, sample: &Sample, runtime: &Totals, seconds: f64) -> Result<()> {
        let minute = sample.timestamp / 60000 * 60000;
        if self
            .checkpoint
            .pending
            .as_ref()
            .is_some_and(|b| b.sample.timestamp != minute)
        {
            self.flush()?;
        }
        self.checkpoint
            .pending
            .get_or_insert_with(Bucket::default)
            .push(sample, seconds);
        self.checkpoint.totals = self.archive.base.add(runtime);
        *self
            .archive
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner()) =
            self.checkpoint.pending.as_ref().map(Bucket::sample);
        self.save()
    }

    fn flush(&mut self) -> Result<()> {
        self.archive.log_background_errors();
        if let Some(bucket) = &self.checkpoint.pending {
            self.archive.write(&bucket.sample())?;
        }
        self.checkpoint.pending = None;
        Ok(())
    }

    fn save(&self) -> Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(self.dir.join("checkpoint.tmp"))?;
        file.write_all(&serde_json::to_vec(&self.checkpoint)?)?;
        file.sync_all()?;
        std::fs::rename(
            self.dir.join("checkpoint.tmp"),
            self.dir.join("checkpoint.json"),
        )?;
        #[cfg(unix)]
        std::fs::File::open(&self.dir)?.sync_all()?;
        Ok(())
    }

    pub(super) fn finish(&mut self, runtime: &Totals) -> Result<()> {
        // Partial minutes live in the durable checkpoint; only finalized minutes enter the TSDB.
        self.checkpoint.totals = self.archive.base.add(runtime);
        self.save()?;
        self.archive.db.close()?;
        Ok(())
    }
}

impl Archive {
    pub(super) fn report(&self, result: Result<()>) {
        let mut error = self.error.lock().unwrap_or_else(|e| e.into_inner());
        match result {
            Err(e) => {
                let message = format!("{e:#}");
                if error.as_ref() != Some(&message) {
                    eprintln!("persisting performance history: {message}");
                }
                *error = Some(message);
            }
            Ok(()) => *error = None,
        }
    }

    /// tsink reports its flush and compaction failures only through `tracing`, which we don't
    /// subscribe to.
    fn log_background_errors(&self) {
        let health = self.db.observability_snapshot().health;
        let seen = self
            .background_errors
            .swap(health.background_errors_total, Ordering::Relaxed);
        if health.background_errors_total > seen {
            eprintln!(
                "performance history storage: {} background errors, last: {}",
                health.background_errors_total - seen,
                health.last_background_error.as_deref().unwrap_or("?")
            );
        }
    }

    fn write(&self, sample: &Sample) -> Result<()> {
        // A removed slot contributes zero in later minutes, including downsampled averages.
        let mut sample = sample.clone();
        let count = sample
            .output_tps
            .len()
            .max(sample.prefill_tps.len())
            .max(sample.kv_bytes.len());
        let count = self
            .known_slots
            .fetch_max(count, Ordering::Relaxed)
            .max(count);
        sample.output_tps.resize(count, 0.0);
        sample.prefill_tps.resize(count, 0.0);
        sample.kv_bytes.resize(count, 0);
        let t = sample.timestamp as i64;
        let mut rows = sample.totals.rows(t);
        rows.push(Row::new(
            "sample_seconds",
            DataPoint::new(t, sample.seconds),
        ));
        for (name, values) in [
            ("output_tps", &sample.output_tps),
            ("prefill_tps", &sample.prefill_tps),
        ] {
            rows.extend(values.iter().enumerate().map(|(slot, v)| {
                Row::with_labels(
                    name,
                    vec![Label::new("slot", slot.to_string())],
                    DataPoint::new(t, *v * sample.seconds),
                )
            }));
        }
        rows.extend(sample.kv_bytes.iter().enumerate().map(|(slot, v)| {
            Row::with_labels(
                "kv_bytes",
                vec![Label::new("slot", slot.to_string())],
                DataPoint::new(t, *v),
            )
        }));
        for (name, v) in [
            ("prefilling", sample.prefilling as u64),
            ("decoding", sample.decoding as u64),
            ("queued", sample.queued),
        ] {
            rows.push(Row::new(name, DataPoint::new(t, v)));
        }
        // tsink preserves duplicate timestamps. Finalized minutes are immutable: on recovery
        // after WAL commit but before checkpoint replacement, only insert missing metric rows.
        // Also handles a failed batch that made only a prefix of its rows visible.
        let mut missing = Vec::new();
        for row in rows {
            let existing = self.db.select(row.metric(), row.labels(), t, t + 1)?;
            anyhow::ensure!(
                existing.iter().all(|p| p == row.data_point()),
                "conflicting persisted minute {} for {}",
                t,
                row.metric()
            );
            if existing.is_empty() {
                missing.push(row);
            }
        }
        if !missing.is_empty() {
            self.db.insert_rows(&missing)?;
        }
        Ok(())
    }

    pub(super) fn query(&self, from: u64, end: u64, interval: u64) -> Result<Vec<Sample>> {
        let mut buckets = BTreeMap::<u64, Sample>::new();
        let pending = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        // Pending replaces its persisted timestamp after restart, never double counting that minute.
        let persisted_end = pending
            .as_ref()
            .filter(|p| p.timestamp < end)
            .map_or(end, |p| p.timestamp);
        for series in self.db.list_metrics()? {
            let slot = series
                .labels
                .iter()
                .find(|l| l.name == "slot")
                .and_then(|l| l.value.parse::<usize>().ok());
            let aggregation = match series.name.as_str() {
                "output_tps" | "prefill_tps" | "sample_seconds" => Aggregation::Sum,
                n if n.starts_with("total_") => Aggregation::Sum,
                _ => Aggregation::Max,
            };
            let options = QueryOptions::new(from as i64, persisted_end as i64)
                .with_labels(series.labels)
                .with_downsample(interval as i64, aggregation);
            if from >= persisted_end {
                continue;
            }
            for p in self.db.select_with_options(&series.name, options)? {
                let t = p.timestamp as u64;
                let s = buckets.entry(t).or_insert_with(|| Sample {
                    timestamp: t,
                    ..Default::default()
                });
                let v = p.value.as_f64().unwrap_or(0.0);
                match (series.name.as_str(), slot) {
                    ("output_tps", Some(i)) => {
                        s.output_tps.resize(s.output_tps.len().max(i + 1), 0.0);
                        s.output_tps[i] = v;
                    }
                    ("prefill_tps", Some(i)) => {
                        s.prefill_tps.resize(s.prefill_tps.len().max(i + 1), 0.0);
                        s.prefill_tps[i] = v;
                    }
                    ("kv_bytes", Some(i)) => {
                        s.kv_bytes.resize(s.kv_bytes.len().max(i + 1), 0);
                        s.kv_bytes[i] = p.value.as_u64().unwrap_or(0);
                    }
                    ("sample_seconds", _) => s.seconds = v,
                    ("prefilling", _) => s.prefilling = v as usize,
                    ("decoding", _) => s.decoding = v as usize,
                    ("queued", _) => s.queued = v as u64,
                    (name, _) => s.totals.set(name, p.value.as_u64().unwrap_or(0)),
                }
            }
        }
        let mut samples = buckets.into_values().collect::<Vec<_>>();
        for sample in &mut samples {
            // Sum token work and measured time, so changing slot counts contribute zeros for
            // absent slots and longer samples receive their actual weight.
            for rates in [&mut sample.output_tps, &mut sample.prefill_tps] {
                for rate in rates {
                    *rate /= sample.seconds.max(1e-9);
                }
            }
        }
        if let Some(pending) = pending.filter(|p| p.timestamp >= from && p.timestamp < end) {
            samples.push(pending);
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            Self(std::env::temp_dir().join(format!(
                "quench6k-stats-{}-{}-{}",
                std::process::id(),
                timestamp(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            )))
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn sample(t: u64, slots: usize, requests: u64, tokens: u64) -> Sample {
        Sample {
            timestamp: t,
            seconds: 1.0,
            output_tps: vec![10.0; slots],
            prefill_tps: vec![30.0; slots],
            kv_bytes: vec![1234; slots],
            decoding: slots,
            totals: Totals {
                requests,
                generated_tokens: tokens,
                ..Default::default()
            },
            ..Default::default()
        }
    }
    #[test]
    fn restart_preserves_totals_partial_minutes_and_variable_slot_counts() {
        let dir = Temp::new();
        let t = (timestamp() / 60000 - 5) * 60000;
        let (archive, mut writer) = Writer::open(dir.0.clone(), 31).unwrap();
        let first = sample(t + 1000, 13, 2, 20);
        writer.record(&first, &first.totals, 2.0).unwrap();
        writer.finish(&first.totals).unwrap();
        drop(writer);
        drop(archive);
        let (archive, mut writer) = Writer::open(dir.0.clone(), 31).unwrap();
        assert_eq!(archive.base.requests, 2);
        let mut second = sample(t + 2000, 3, 1, 30);
        second.output_tps = vec![20.0; 3];
        writer.record(&second, &second.totals, 1.0).unwrap();
        let result = archive.query(t, t + 60000, 60000).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].totals.requests, 3);
        assert_eq!(result[0].totals.generated_tokens, 50);
        assert_eq!(result[0].output_tps.len(), 13);
        assert!((result[0].output_tps[0] - 40.0 / 3.0).abs() < 1e-9);
        assert!((result[0].output_tps[12] - 20.0 / 3.0).abs() < 1e-9);
        let third = sample(t + 61000, 1, 4, 100);
        let runtime = second.totals.add(&third.totals);
        writer.record(&third, &runtime, 1.0).unwrap();
        writer.finish(&runtime).unwrap();
        drop(writer);
        drop(archive);
        let (archive, mut writer) = Writer::open(dir.0.clone(), 31).unwrap();
        assert_eq!(archive.base.requests, 7);
        assert_eq!(archive.base.generated_tokens, 150);
        let result = archive.query(t, t + 120000, 60000).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(
            result
                .iter()
                .map(|s| s.totals.generated_tokens)
                .sum::<u64>(),
            150
        );
        assert_eq!(result[0].output_tps.len(), 13);
        writer.finish(&Totals::default()).unwrap();
    }
    #[test]
    fn native_downsampling_keeps_integer_counts_and_peak_concurrency() {
        let dir = Temp::new();
        let (archive, mut writer) = Writer::open(dir.0.clone(), 31).unwrap();
        let t = (timestamp() / 600000 - 2) * 600000;
        let huge = (1u64 << 53) + 1;
        archive.write(&sample(t, 1, 2, huge)).unwrap();
        archive.write(&sample(t + 60000, 12, 3, 5)).unwrap();
        archive.write(&sample(t, 1, 2, huge)).unwrap();
        let result = archive.query(t, t + 120000, 600000).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].totals.generated_tokens, huge + 5);
        assert_eq!(result[0].totals.requests, 5);
        assert_eq!(result[0].decoding, 12);
        assert_eq!(result[0].kv_bytes.len(), 12);
        assert_eq!(result[0].output_tps[11], 5.0);
        writer.finish(&Totals::default()).unwrap();
    }
    #[test]
    fn refuses_shared_directories_and_invalid_checkpoints() {
        let dir = Temp::new();
        let (archive, mut writer) = Writer::open(dir.0.clone(), 31).unwrap();
        assert!(Writer::open(dir.0.clone(), 31).is_err());
        writer.finish(&Totals::default()).unwrap();
        drop(writer);
        drop(archive);
        std::fs::write(dir.0.join("checkpoint.json"), b"broken").unwrap();
        assert!(Writer::open(dir.0.clone(), 31).is_err());
    }
}
