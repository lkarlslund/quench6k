//! Durable minute telemetry. Only the sampler and blocking HTTP queries access storage.
use super::{Metrics, Sample, timestamp};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tsink::{
    DataPoint, Label, Row, Storage, StorageBuilder, TimestampPrecision, Value, WalSyncMode,
};

/// tsink persists every insert as a segment of its own within 250 ms, merges segments only up
/// to its top level (16 inserts each) and keeps ~20 KB of index memory per segment for our 35
/// series (8 slots). Inserting every minute grew that by ~1.9 MB a day without bound inside the
/// retention; inserting each hour once it has ended keeps 31 days at ~50 segments.
const BATCH_MS: u64 = 3_600_000;
/// Finished minutes not yet in tsink, oldest first; rewritten once a minute, not every second.
const UNWRITTEN: &str = "unwritten.json";
/// While tsink refuses writes, keep retrying at most this many minutes (a day).
const MAX_UNWRITTEN: usize = 1440;

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

/// Minutes queries read from the writer rather than from tsink.
#[derive(Clone, Default)]
struct Recent {
    /// Finished minutes, oldest first, all before `pending`.
    unwritten: Vec<Sample>,
    pending: Option<Sample>,
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
    recent: Mutex<Recent>,
    #[cfg(test)]
    refuse_writes: std::sync::atomic::AtomicBool,
}

pub(super) struct Writer {
    archive: Arc<Archive>,
    dir: PathBuf,
    checkpoint: Checkpoint,
    unwritten: Vec<Sample>,
    unwritten_dirty: bool,
}

pub(super) fn default_dir() -> Result<PathBuf> {
    let root = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))
        .context("set --stats-dir or HOME/XDG_STATE_HOME for durable statistics")?;
    Ok(root.join("quench6k/stats"))
}

/// Replaces `dir/name` with `data` so that a crash leaves the old or the new contents.
fn store(dir: &Path, name: &str, data: &[u8]) -> Result<()> {
    use std::io::Write;
    let tmp = dir.join(format!("{name}.tmp"));
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(data)?;
    file.sync_all()?;
    std::fs::rename(&tmp, dir.join(name))?;
    #[cfg(unix)]
    std::fs::File::open(dir)?.sync_all()?;
    Ok(())
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
        let mut unwritten: Vec<Sample> = match std::fs::read(dir.join(UNWRITTEN)) {
            Ok(data) => serde_json::from_slice(&data).context("read unwritten statistics")?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e.into()),
        };
        let retention_seconds = u64::from(days) * 86400;
        // A dormant server can outlive its graph retention; lifetime totals still survive.
        let cutoff = timestamp().saturating_sub(retention_seconds * 1000);
        unwritten.retain(|s| s.timestamp >= cutoff);
        if checkpoint.pending.as_ref().is_some_and(|b| {
            b.sample.timestamp < cutoff
                // Saved as finished before the checkpoint that dropped it.
                || unwritten.last().is_some_and(|u| u.timestamp >= b.sample.timestamp)
        }) {
            checkpoint.pending = None;
        }
        let db = StorageBuilder::new()
            .with_data_path(dir.join("series"))
            .with_timestamp_precision(TimestampPrecision::Milliseconds)
            .with_retention(Duration::from_secs(retention_seconds))
            // No memory limit: tsink refuses every write beyond it and can evict only unflushed
            // chunks, never the index of persisted segments, so a limit only discards history.
            // Hourly inserts bound that index by the retention instead.
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
        let known_slots = unwritten
            .iter()
            .map(|s| s.output_tps.len())
            .chain(
                checkpoint
                    .pending
                    .as_ref()
                    .map(|b| b.sample.output_tps.len()),
            )
            .fold(stored_slots, usize::max);
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
            recent: Mutex::new(Recent {
                unwritten: unwritten.clone(),
                pending: checkpoint.pending.as_ref().map(Bucket::sample),
            }),
            #[cfg(test)]
            refuse_writes: Default::default(),
        });
        let writer = Self {
            archive: archive.clone(),
            dir,
            checkpoint,
            unwritten,
            unwritten_dirty: false,
        };
        writer.save()?;
        Ok((archive, writer))
    }

    pub(super) fn record(&mut self, sample: &Sample, runtime: &Totals, seconds: f64) -> Result<()> {
        let minute = sample.timestamp / 60000 * 60000;
        let mut result = Ok(());
        if self
            .checkpoint
            .pending
            .as_ref()
            .is_some_and(|b| b.sample.timestamp != minute)
        {
            result = self.finish_minute(minute);
        }
        // Even when tsink refuses finished minutes, the current one keeps accumulating.
        self.checkpoint
            .pending
            .get_or_insert_with(Bucket::default)
            .push(sample, seconds);
        self.checkpoint.totals = self.archive.base.add(runtime);
        self.archive.recent().pending = self.checkpoint.pending.as_ref().map(Bucket::sample);
        self.save()?;
        result
    }

    /// Moves the pending minute to the unwritten ones and writes every hour before `minute`.
    fn finish_minute(&mut self, minute: u64) -> Result<()> {
        self.archive.log_background_errors();
        if let Some(bucket) = self.checkpoint.pending.take() {
            let mut sample = bucket.sample();
            self.archive.pad(&mut sample);
            self.unwritten.push(sample);
            self.unwritten_dirty = true;
        }
        // tsink rejects a whole insert holding a point older than its retention.
        let cutoff = timestamp().saturating_sub(self.archive.retention_seconds * 1000) + 60000;
        let before = self.unwritten.len();
        self.unwritten.retain(|s| s.timestamp >= cutoff);
        let mut dropped = before - self.unwritten.len();
        let mut result = Ok(());
        while let Some(first) = self.unwritten.first() {
            let batch = first.timestamp / BATCH_MS;
            if batch >= minute / BATCH_MS {
                break;
            }
            let n = self
                .unwritten
                .iter()
                .take_while(|s| s.timestamp / BATCH_MS == batch)
                .count();
            if let Err(e) = self.archive.write(&self.unwritten[..n]) {
                result = Err(e);
                break;
            }
            self.unwritten.drain(..n);
            self.unwritten_dirty = true;
        }
        if self.unwritten.len() > MAX_UNWRITTEN {
            let excess = self.unwritten.len() - MAX_UNWRITTEN;
            self.unwritten.drain(..excess);
            dropped += excess;
            let message = format!("dropped the {excess} oldest unwritten minutes");
            result = Err(match result {
                Ok(()) => anyhow::anyhow!(message),
                Err(e) => e.context(message),
            });
        }
        self.unwritten_dirty |= dropped > 0;
        let saved = if self.unwritten_dirty {
            serde_json::to_vec(&self.unwritten)
                .map_err(anyhow::Error::from)
                .and_then(|data| store(&self.dir, UNWRITTEN, &data))
                .context("save unwritten statistics")
        } else {
            Ok(())
        };
        self.unwritten_dirty = saved.is_err();
        self.archive.recent().unwritten = self.unwritten.clone();
        saved.and(result)
    }

    fn save(&self) -> Result<()> {
        store(
            &self.dir,
            "checkpoint.json",
            &serde_json::to_vec(&self.checkpoint)?,
        )
    }

    pub(super) fn finish(&mut self, runtime: &Totals) -> Result<()> {
        // Unfinished minutes and hours live in the durable checkpoint and unwritten minutes;
        // only finished hours enter the TSDB.
        self.checkpoint.totals = self.archive.base.add(runtime);
        self.save()?;
        if self.unwritten_dirty {
            store(&self.dir, UNWRITTEN, &serde_json::to_vec(&self.unwritten)?)?;
            self.unwritten_dirty = false;
        }
        self.archive.db.close()?;
        Ok(())
    }
}

/// tsink rows of one finished minute. Rates are stored as token sums so that downsampling sums
/// work and time.
fn rows(sample: &Sample) -> Vec<Row> {
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
    rows
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

    fn recent(&self) -> std::sync::MutexGuard<'_, Recent> {
        self.recent.lock().unwrap_or_else(|e| e.into_inner())
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

    /// A removed slot contributes zero in later minutes, including downsampled averages.
    fn pad(&self, sample: &mut Sample) {
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
    }

    /// Inserts finished minutes in one batch, so tsink persists them as one segment.
    fn write(&self, samples: &[Sample]) -> Result<()> {
        #[cfg(test)]
        anyhow::ensure!(
            !self.refuse_writes.load(Ordering::Relaxed),
            "refused by test"
        );
        let mut series = HashMap::<(String, Vec<Label>), Vec<Row>>::new();
        for sample in samples {
            let mut sample = sample.clone();
            self.pad(&mut sample);
            for row in rows(&sample) {
                series
                    .entry((row.metric().to_string(), row.labels().to_vec()))
                    .or_default()
                    .push(row);
            }
        }
        // tsink preserves duplicate timestamps. Finished minutes are immutable: after a crash
        // between the insert and saving the unwritten minutes, or after a failed insert that
        // made only some rows visible, insert only the rows tsink lacks.
        let mut missing = Vec::new();
        for ((metric, labels), rows) in series {
            let times = rows.iter().map(|r| r.data_point().timestamp);
            let (first, last) = (times.clone().min().unwrap(), times.max().unwrap());
            let stored = self
                .db
                .select(&metric, &labels, first, last + 1)?
                .into_iter()
                .map(|p| p.timestamp)
                .collect::<HashSet<_>>();
            missing.extend(
                rows.into_iter()
                    .filter(|r| !stored.contains(&r.data_point().timestamp)),
            );
        }
        if !missing.is_empty() {
            self.db.insert_rows(&missing)?;
        }
        Ok(())
    }

    pub(super) fn query(&self, from: u64, end: u64, interval: u64) -> Result<Vec<Sample>> {
        let recent = self.recent().clone();
        let unwritten = recent
            .unwritten
            .iter()
            .filter(|s| s.timestamp >= from && s.timestamp < end)
            .collect::<Vec<_>>();
        // Minutes the writer still holds replace any copy tsink already has (a restart can
        // leave one in both), so tsink is read only before the first of them.
        let stored_end = recent
            .unwritten
            .first()
            .or(recent.pending.as_ref())
            .map_or(end, |s| s.timestamp.min(end));
        let mut series = HashMap::<(String, Vec<Label>), Vec<DataPoint>>::new();
        if from < stored_end {
            for s in self.db.list_metrics()? {
                // Not tsink's downsampling: with an aggregation, select_with_options (0.10.2)
                // holds its visibility read lock and takes it again to read the series. When a
                // background retention pass queues for the write lock in between, both wait
                // forever.
                let points = self
                    .db
                    .select(&s.name, &s.labels, from as i64, stored_end as i64)?;
                series.entry((s.name, s.labels)).or_default().extend(points);
            }
        }
        for row in unwritten.iter().flat_map(|s| rows(s)) {
            series
                .entry((row.metric().to_string(), row.labels().to_vec()))
                .or_default()
                .push(row.data_point().clone());
        }
        let mut buckets = BTreeMap::<u64, Sample>::new();
        for ((name, labels), points) in series {
            let slot = labels
                .iter()
                .find(|l| l.name == "slot")
                .and_then(|l| l.value.parse::<usize>().ok());
            let sum = matches!(
                name.as_str(),
                "output_tps" | "prefill_tps" | "sample_seconds"
            ) || name.starts_with("total_");
            for (t, value) in downsample(points, from, interval, sum) {
                let s = buckets.entry(t).or_insert_with(|| Sample {
                    timestamp: t,
                    ..Default::default()
                });
                let v = value.as_f64().unwrap_or(0.0);
                match (name.as_str(), slot) {
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
                        s.kv_bytes[i] = value.as_u64().unwrap_or(0);
                    }
                    ("sample_seconds", _) => s.seconds = v,
                    ("prefilling", _) => s.prefilling = v as usize,
                    ("decoding", _) => s.decoding = v as usize,
                    ("queued", _) => s.queued = v as u64,
                    (name, _) => s.totals.set(name, value.as_u64().unwrap_or(0)),
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
        if let Some(pending) = recent
            .pending
            .filter(|p| p.timestamp >= from && p.timestamp < end)
        {
            samples.push(pending);
        }
        Ok(samples)
    }
}

/// Sums or maxima over buckets starting at `from + k * interval`, as tsink's downsampling.
/// Integer values stay exact (sums saturate); NaN points are skipped.
fn downsample(points: Vec<DataPoint>, from: u64, interval: u64, sum: bool) -> BTreeMap<u64, Value> {
    let interval = interval.max(1);
    let mut buckets = BTreeMap::<u64, Value>::new();
    for p in points {
        if matches!(p.value, Value::F64(v) if v.is_nan()) {
            continue;
        }
        let t = from + (p.timestamp.max(0) as u64).saturating_sub(from) / interval * interval;
        let merged = match (buckets.remove(&t), p.value) {
            (None, v) => v,
            (Some(Value::U64(a)), Value::U64(b)) => {
                Value::U64(if sum { a.saturating_add(b) } else { a.max(b) })
            }
            (Some(a), b) => {
                let (a, b) = (a.as_f64().unwrap_or(0.0), b.as_f64().unwrap_or(0.0));
                Value::F64(if sum { a + b } else { a.max(b) })
            }
        };
        buckets.insert(t, merged);
    }
    buckets
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
        archive.write(&[sample(t, 1, 2, huge)]).unwrap();
        archive.write(&[sample(t + 60000, 12, 3, 5)]).unwrap();
        archive.write(&[sample(t, 1, 2, huge)]).unwrap();
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

    const H: u64 = BATCH_MS;
    /// Waits until tsink has persisted every point inserted so far, as its 250 ms background
    /// flush does between two minutes of the production sampler.
    fn settle(archive: &Archive) {
        let start = std::time::Instant::now();
        loop {
            let o = archive.db.observability_snapshot();
            if o.flush.persisted_points_total >= o.wal.append_points_total {
                return;
            }
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "tsink did not flush"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn record_minute(writer: &mut Writer, runtime: &mut Totals, t: u64) -> Result<()> {
        let s = sample(t, 8, 1, 10);
        *runtime = runtime.add(&s.totals);
        writer.record(&s, runtime, 1.0)
    }
    #[test]
    fn eight_slot_minutes_reach_tsink_once_per_hour() {
        // tsink persists every insert as a segment of its own and never merges its top level
        // (16 inserts per segment); each segment holds ~20 KB of index memory for these 35
        // series. Inserting every minute outgrew the former 32 MiB budget after ~16 days.
        let dir = Temp::new();
        let hours = 4;
        let t0 = (timestamp() / H - hours - 2) * H;
        let end = t0 + (hours * 60 + 1) * 60000;
        let (archive, mut writer) = Writer::open(dir.0.clone(), 31).unwrap();
        let mut runtime = Totals::default();
        for m in 0..=hours * 60 {
            record_minute(&mut writer, &mut runtime, t0 + m * 60000 + 1000).unwrap();
            settle(&archive);
        }
        let segments = archive
            .db
            .observability_snapshot()
            .flush
            .persisted_segments_total;
        assert!(
            segments <= hours,
            "{segments} tsink segments for {hours} hours"
        );
        let check = |archive: &Archive| {
            let result = archive.query(t0, end, 60000).unwrap();
            assert_eq!(result.len() as u64, hours * 60 + 1);
            for (m, s) in result.iter().enumerate() {
                assert_eq!(s.timestamp, t0 + m as u64 * 60000);
                assert_eq!(s.totals.requests, 1);
                assert_eq!(s.kv_bytes, vec![1234; 8]);
                assert_eq!(s.output_tps, vec![10.0; 8]);
            }
            let hourly = archive.query(t0, end, H).unwrap();
            assert_eq!(
                hourly.iter().map(|s| s.totals.requests).sum::<u64>(),
                hours * 60 + 1
            );
        };
        check(&archive);
        writer.finish(&runtime).unwrap();
        drop(writer);
        drop(archive);
        let (archive, mut writer) = Writer::open(dir.0.clone(), 31).unwrap();
        check(&archive);
        writer.finish(&runtime).unwrap();
    }
    #[test]
    fn refused_writes_keep_every_minute_and_retry() {
        let dir = Temp::new();
        let t0 = (timestamp() / H - 5) * H;
        let (archive, mut writer) = Writer::open(dir.0.clone(), 31).unwrap();
        let mut runtime = Totals::default();
        for m in 0..60 {
            record_minute(&mut writer, &mut runtime, t0 + m * 60000).unwrap();
        }
        archive.refuse_writes.store(true, Ordering::Relaxed);
        let failures = (60..150)
            .filter(|m| record_minute(&mut writer, &mut runtime, t0 + m * 60000).is_err())
            .count();
        assert!(failures > 0);
        let result = archive.query(t0, t0 + 150 * 60000, 60000).unwrap();
        assert_eq!(result.len(), 150);
        // The backlog survives a restart while storage still refuses it.
        writer.finish(&runtime).unwrap();
        drop(writer);
        drop(archive);
        let (archive, mut writer) = Writer::open(dir.0.clone(), 31).unwrap();
        for m in 150..=180 {
            record_minute(&mut writer, &mut runtime, t0 + m * 60000).unwrap();
        }
        let result = archive.query(t0, t0 + 181 * 60000, 60000).unwrap();
        assert_eq!(result.len(), 181);
        assert!(result.iter().all(|s| s.totals.requests == 1));
        assert_eq!(
            result
                .iter()
                .map(|s| s.totals.generated_tokens)
                .sum::<u64>(),
            1810
        );
        let stored = archive
            .db
            .select("total_requests", &[], t0 as i64, (t0 + 180 * 60000) as i64)
            .unwrap();
        assert_eq!(stored.len(), 180);
        writer.finish(&runtime).unwrap();
    }
    /// The production failure: ~18 days of minutes inserted one at a time (one tsink segment
    /// each, as tsink's 250 ms flush makes them) left the former 32 MiB budget unable to admit
    /// the next minute. Slow (each reopen loads every segment so far; about half an hour):
    /// `cargo test --release -p quench6k-cli -- --ignored`.
    #[test]
    #[ignore]
    fn extends_history_stored_one_segment_per_minute() {
        let dir = Temp::new();
        let minutes = 26000u64;
        let t0 = (timestamp() / 60000 - minutes - 120) * 60000;
        let numbers = [
            "total_requests",
            "total_generations",
            "total_prompt_tokens",
            "total_cached_tokens",
            "total_generated_tokens",
            "total_prefilled_tokens",
            "total_errors",
            "prefilling",
            "decoding",
            "queued",
        ];
        for m in 0..minutes {
            let t = (t0 + m * 60000) as i64;
            let mut rows: Vec<Row> = numbers
                .iter()
                .map(|n| Row::new(*n, DataPoint::new(t, m % 7)))
                .collect();
            rows.push(Row::new("sample_seconds", DataPoint::new(t, 60.0)));
            for slot in 0..8 {
                let labels = vec![Label::new("slot", slot.to_string())];
                rows.push(Row::with_labels(
                    "output_tps",
                    labels.clone(),
                    DataPoint::new(t, 600.0),
                ));
                rows.push(Row::with_labels(
                    "prefill_tps",
                    labels.clone(),
                    DataPoint::new(t, 0.0),
                ));
                rows.push(Row::with_labels(
                    "kv_bytes",
                    labels,
                    DataPoint::new(t, 163577856u64),
                ));
            }
            let db = StorageBuilder::new()
                .with_data_path(dir.0.join("series"))
                .with_timestamp_precision(TimestampPrecision::Milliseconds)
                .with_retention(Duration::from_secs(31 * 86400))
                .with_max_writers(1)
                .build()
                .unwrap();
            db.insert_rows(&rows).unwrap();
            db.close().unwrap();
        }
        let (archive, mut writer) = Writer::open(dir.0.clone(), 31).unwrap();
        eprintln!(
            "tsink memory after {minutes} one-minute inserts: {:?}",
            archive.db.observability_snapshot().memory
        );
        // The second minute starts an hour, so the first one is written to tsink.
        let t = timestamp() / H * H - 60000;
        let mut runtime = Totals::default();
        record_minute(&mut writer, &mut runtime, t).unwrap();
        record_minute(&mut writer, &mut runtime, t + 60000).unwrap();
        let stored = archive
            .db
            .select("total_requests", &[], t as i64, t as i64 + 1)
            .unwrap();
        assert_eq!(stored.len(), 1);
        writer.finish(&runtime).unwrap();
    }
}
