//! Renders benchmark records (JSON lines from `quench6k bench --json`, plus baseline records of other
//! engines) as an SVG chart and as the Markdown tables of BENCHMARK.md.
//!
//!     benchplot <records.jsonl>... --svg bench/results.svg --markdown BENCHMARK.md
//!
//! Later records replace earlier ones with the same engine, mode, drafts, adaptive and sampled flags,
//! KV format and context. The Markdown file's text between `<!-- bench:begin -->` and `<!-- bench:end -->` is
//! replaced with the tables.

use std::collections::BTreeMap;
use std::fmt::Write;

use anyhow::{Context, Result, bail};
use serde_json::Value;

#[derive(Clone, Debug)]
struct Rec {
    engine: String,
    mode: String,
    drafts: u64,
    adaptive: bool,
    sampled: bool,
    kv: String,
    ctx: u64,
    prefill: Option<f64>,
    decode: Option<f64>,
}

impl Rec {
    fn key(&self) -> (String, String, u64, bool, bool, String, u64) {
        (self.engine.clone(), self.mode.clone(), self.drafts, self.adaptive, self.sampled, self.kv.clone(), self.ctx)
    }
}

/// A plotted or tabulated series: records matching an engine, mode and KV format.
struct Series {
    label: &'static str,
    engine: &'static str,
    mode: &'static str,
    drafts: u64,
    adaptive: bool,
    sampled: bool,
    kv: &'static str,
    color: &'static str,
    dashed: bool,
}

const DECODE: &[Series] = &[
    Series { label: "quench6k MTP sampled (adaptive, up to 6 drafts)", engine: "quench6k", mode: "mtp", drafts: 6, adaptive: true, sampled: true, kv: "bf16", color: "#2f9e44", dashed: false },
    Series { label: "quench6k MTP greedy (adaptive, up to 6 drafts)", engine: "quench6k", mode: "mtp", drafts: 6, adaptive: true, sampled: false, kv: "bf16", color: "#d9480f", dashed: false },
    Series { label: "quench6k MTP greedy (3 drafts)", engine: "quench6k", mode: "mtp", drafts: 3, adaptive: false, sampled: false, kv: "bf16", color: "#f08c00", dashed: false },
    Series { label: "NInfer 6000 MTP (3 drafts, LM-head draft)", engine: "ninfer", mode: "mtp", drafts: 3, adaptive: false, sampled: false, kv: "bf16", color: "#868e96", dashed: true },
    Series { label: "quench6k plain", engine: "quench6k", mode: "plain", drafts: 0, adaptive: false, sampled: false, kv: "bf16", color: "#1971c2", dashed: false },
    Series { label: "NInfer 6000 plain", engine: "ninfer", mode: "plain", drafts: 0, adaptive: false, sampled: false, kv: "bf16", color: "#adb5bd", dashed: true },
];

const PREFILL: &[Series] = &[
    Series { label: "quench6k", engine: "quench6k", mode: "plain", drafts: 0, adaptive: false, sampled: false, kv: "bf16", color: "#1971c2", dashed: false },
    Series { label: "NInfer 6000", engine: "ninfer", mode: "plain", drafts: 0, adaptive: false, sampled: false, kv: "bf16", color: "#adb5bd", dashed: true },
];

fn parse(line: &str) -> Result<Rec> {
    let v: Value = serde_json::from_str(line)?;
    let s = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_owned).with_context(|| format!("missing {k}"));
    Ok(Rec {
        engine: s("engine")?,
        mode: s("mode")?,
        drafts: v.get("drafts").and_then(Value::as_u64).unwrap_or(0),
        adaptive: v.get("adaptive").and_then(Value::as_bool).unwrap_or(false),
        sampled: v.get("sampled").and_then(Value::as_bool).unwrap_or(false),
        kv: s("kv")?,
        ctx: v.get("ctx").and_then(Value::as_u64).context("missing ctx")?,
        prefill: v.get("prefill_tok_s").and_then(Value::as_f64),
        decode: v.get("decode_tok_s").and_then(Value::as_f64),
    })
}

fn find<'a>(recs: &'a [Rec], s: &Series, kv: &str, ctx: u64) -> Option<&'a Rec> {
    recs.iter().find(|r| r.engine == s.engine && r.mode == s.mode && r.drafts == s.drafts && r.adaptive == s.adaptive && r.sampled == s.sampled && r.kv == kv && r.ctx == ctx)
}

/// "512", "8K", ...; within 3% of a power of two it is named after it (260,000 tokens: "256K").
fn ctx_label(ctx: u64) -> String {
    let p = 2f64.powf((ctx as f64).log2().round());
    let c = if (ctx as f64 - p).abs() / p < 0.03 { p as u64 } else { ctx };
    if c < 1000 { c.to_string() } else { format!("{}K", (c as f64 / 1024.0).round() as u64) }
}

fn num(v: f64) -> String {
    if v >= 1000.0 {
        let n = v.round() as u64;
        format!("{},{:03}", n / 1000, n % 1000)
    } else {
        format!("{v:.1}")
    }
}

/// One chart panel: tok/s against context (log scale) for `series`.
fn panel(svg: &mut String, recs: &[Rec], ctxs: &[u64], series: &[Series], value: fn(&Rec) -> Option<f64>, x0: f64, title: &str) {
    let (w, h, top, left, bottom) = (430.0, 300.0, 40.0, 58.0, 36.0);
    let pts = |s: &Series, kv: &str| -> Vec<(u64, f64)> { ctxs.iter().filter_map(|&c| find(recs, s, kv, c).and_then(value).map(|v| (c, v))).collect() };
    // Each quench6k series is drawn twice: BF16 KV as listed, FP8 KV dotted with open markers.
    let lines: Vec<(&Series, &str)> = series.iter().flat_map(|s| if s.engine == "quench6k" { vec![(s, s.kv), (s, "fp8")] } else { vec![(s, s.kv)] }).collect();
    let vmax = lines.iter().flat_map(|&(s, kv)| pts(s, kv)).map(|p| p.1).fold(0.0, f64::max);
    // A round axis maximum with 4-6 gridlines.
    let step = (0..7).flat_map(|e| [1.0, 2.0, 2.5, 5.0].map(|m| m * 10f64.powi(e))).find(|st| vmax / st <= 5.0).unwrap_or(1.0);
    let ymax = (vmax / step).ceil() * step;
    let (lx, ly) = (x0 + left, top);
    let (pw, ph) = (w - left - 12.0, h - top - bottom);
    let (lmin, lmax) = ((ctxs[0] as f64).log2(), (*ctxs.last().unwrap() as f64).log2());
    let px = |c: u64| lx + ((c as f64).log2() - lmin) / (lmax - lmin).max(1e-9) * pw;
    let py = |v: f64| ly + ph - v / ymax * ph;
    let _ = writeln!(svg, r#"<text x="{}" y="24" class="title">{title}</text>"#, x0 + left);
    let mut v = 0.0;
    while v <= ymax + 1e-9 {
        let y = py(v);
        let _ = writeln!(svg, r#"<line x1="{lx}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}" class="grid"/><text x="{:.1}" y="{:.1}" class="ytick">{}</text>"#, lx + pw, lx - 6.0, y + 4.0, num(v).trim_end_matches(".0"));
        v += step;
    }
    for &c in ctxs {
        let _ = writeln!(svg, r#"<text x="{:.1}" y="{:.1}" class="xtick">{}</text>"#, px(c), ly + ph + 18.0, ctx_label(c));
    }
    let _ = writeln!(svg, r#"<text x="{:.1}" y="{:.1}" class="axis">context (tokens)</text>"#, lx + pw / 2.0, ly + ph + 33.0);
    for &(s, kv) in &lines {
        let p = pts(s, kv);
        if p.is_empty() {
            continue;
        }
        // One polyline per run of consecutive contexts: no line across a context without data.
        let fp8 = kv == "fp8";
        let dash = if fp8 { r#" stroke-dasharray="0.1 4.5" stroke-linecap="round""# } else if s.dashed { r#" stroke-dasharray="5 4""# } else { "" };
        let mut runs: Vec<Vec<(u64, f64)>> = Vec::new();
        for &(c, v) in &p {
            let i = ctxs.iter().position(|&x| x == c).unwrap();
            match runs.last_mut() {
                Some(run) if i > 0 && run.last().map(|l| l.0) == Some(ctxs[i - 1]) => run.push((c, v)),
                _ => runs.push(vec![(c, v)]),
            }
        }
        for run in runs.iter().filter(|r| r.len() > 1) {
            let path: Vec<String> = run.iter().map(|&(c, v)| format!("{:.1},{:.1}", px(c), py(v))).collect();
            let _ = writeln!(svg, r#"<polyline points="{}" fill="none" stroke="{}" stroke-width="2.2"{dash}/>"#, path.join(" "), s.color);
        }
        for &(c, v) in &p {
            let fill = if fp8 { "#ffffff" } else { s.color };
            let _ = writeln!(svg, r#"<circle cx="{:.1}" cy="{:.1}" r="3" fill="{fill}" stroke="{}" stroke-width="1.5"/>"#, px(c), py(v), s.color);
        }
    }
}

fn legend(svg: &mut String, series: &[Series], x: f64, y: f64) {
    for (i, s) in series.iter().enumerate() {
        let yy = y + i as f64 * 18.0;
        let dash = if s.dashed { r#" stroke-dasharray="5 4""# } else { "" };
        let _ = writeln!(svg, r#"<line x1="{x}" y1="{yy}" x2="{}" y2="{yy}" stroke="{}" stroke-width="2.2"{dash}/><text x="{}" y="{}" class="legend">{}</text>"#, x + 22.0, s.color, x + 28.0, yy + 4.0, s.label);
    }
}

fn svg(recs: &[Rec], ctxs: &[u64]) -> String {
    let mut s = String::new();
    let (w, h) = (880.0, 440.0);
    let _ = writeln!(s, r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">"#);
    s.push_str(r#"<style>.title{font-size:15px;font-weight:600;fill:#212529}.grid{stroke:#e9ecef}.ytick{font-size:11px;fill:#495057;text-anchor:end}.xtick{font-size:11px;fill:#495057;text-anchor:middle}.axis{font-size:11px;fill:#868e96;text-anchor:middle}.legend{font-size:12px;fill:#343a40}</style>"#);
    let _ = writeln!(s, r##"<rect width="{w}" height="{h}" fill="#ffffff"/>"##);
    panel(&mut s, recs, ctxs, DECODE, |r| r.decode, 0.0, "Decode, tokens/s (one request)");
    panel(&mut s, recs, ctxs, PREFILL, |r| r.prefill, 440.0, "Prefill, tokens/s");
    legend(&mut s, DECODE, 60.0, 318.0);
    legend(&mut s, PREFILL, 500.0, 318.0);
    let (x, y) = (500.0, 318.0 + (PREFILL.len() as f64 + 0.6) * 18.0);
    let _ = writeln!(s, r##"<line x1="{x}" y1="{y}" x2="{}" y2="{y}" stroke="#495057" stroke-width="2.2"/><circle cx="{}" cy="{y}" r="3" fill="#495057"/><text x="{}" y="{}" class="legend">quench6k with BF16 KV cache</text>"##, x + 22.0, x + 11.0, x + 28.0, y + 4.0);
    let y = y + 18.0;
    let _ = writeln!(s, r##"<line x1="{x}" y1="{y}" x2="{}" y2="{y}" stroke="#495057" stroke-width="2.2" stroke-dasharray="0.1 4.5" stroke-linecap="round"/><circle cx="{}" cy="{y}" r="3" fill="#ffffff" stroke="#495057" stroke-width="1.5"/><text x="{}" y="{}" class="legend">quench6k with FP8 KV cache (--kv fp8)</text>"##, x + 22.0, x + 11.0, x + 28.0, y + 4.0);
    s.push_str("</svg>\n");
    s
}

fn table(recs: &[Rec], ctxs: &[u64], kv: &str, cols: &[(&str, &Series, fn(&Rec) -> Option<f64>)]) -> String {
    let mut t = String::from("| Context |");
    for (h, _, _) in cols {
        let _ = write!(t, " {h} |");
    }
    t.push_str("\n|---:|");
    t.push_str(&"---:|".repeat(cols.len()));
    t.push('\n');
    for &c in ctxs {
        let cells: Vec<String> = cols.iter().map(|(_, s, f)| find(recs, s, if s.engine == "quench6k" { kv } else { s.kv }, c).and_then(f).map_or("–".into(), num)).collect();
        if cells.iter().all(|x| x == "–") {
            continue;
        }
        let _ = writeln!(t, "| {} | {} |", ctx_label(c), cells.join(" | "));
    }
    t
}

fn markdown(recs: &[Rec], ctxs: &[u64]) -> String {
    let (q_sampled, q_mtp6, q_mtp3, n_mtp, q_plain, n_plain) = (&DECODE[0], &DECODE[1], &DECODE[2], &DECODE[3], &DECODE[4], &DECODE[5]);
    let d: fn(&Rec) -> Option<f64> = |r| r.decode;
    let p: fn(&Rec) -> Option<f64> = |r| r.prefill;
    let mut m = String::from("BF16 KV cache (the NInfer 6000 baseline's configuration):\n\n");
    m.push_str(&table(recs, ctxs, "bf16", &[("Decode", q_plain, d), ("NInfer decode", n_plain, d), ("MTP greedy, 3 drafts", q_mtp3, d), ("NInfer MTP, 3 drafts", n_mtp, d), ("MTP greedy, adaptive ≤ 6", q_mtp6, d), ("MTP sampled, adaptive ≤ 6", q_sampled, d), ("Prefill", q_plain, p), ("NInfer prefill", n_plain, p)]));
    m.push_str("\nFP8 KV cache (`--kv fp8`, the serving configuration):\n\n");
    m.push_str(&table(recs, ctxs, "fp8", &[("Decode", q_plain, d), ("MTP greedy, adaptive ≤ 6", q_mtp6, d), ("MTP sampled, adaptive ≤ 6", q_sampled, d), ("Prefill", q_plain, p)]));
    m
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut inputs, mut svg_out, mut md_out) = (Vec::new(), None, None);
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--svg" => svg_out = it.next(),
            "--markdown" => md_out = it.next(),
            _ => inputs.push(a),
        }
    }
    if inputs.is_empty() {
        bail!("usage: benchplot <records.jsonl>... [--svg out.svg] [--markdown BENCHMARK.md]");
    }
    let mut latest: BTreeMap<_, Rec> = BTreeMap::new();
    for path in &inputs {
        let text = std::fs::read_to_string(path).with_context(|| path.clone())?;
        for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let r = parse(line).with_context(|| format!("{path}:{}", i + 1))?;
            latest.insert(r.key(), r);
        }
    }
    let recs: Vec<Rec> = latest.into_values().collect();
    let mut ctxs: Vec<u64> = recs.iter().filter(|r| r.engine == "quench6k").map(|r| r.ctx).collect();
    ctxs.sort();
    ctxs.dedup();
    if ctxs.is_empty() {
        bail!("no quench6k records");
    }
    if let Some(path) = svg_out {
        std::fs::write(&path, svg(&recs, &ctxs))?;
    }
    let md = markdown(&recs, &ctxs);
    match md_out {
        Some(path) => {
            let text = std::fs::read_to_string(&path)?;
            let (b, e) = ("<!-- bench:begin -->", "<!-- bench:end -->");
            let (i, j) = (text.find(b).context("no bench:begin marker")?, text.find(e).context("no bench:end marker")?);
            std::fs::write(&path, format!("{}{b}\n{md}{}", &text[..i], &text[j..]))?;
        }
        None => print!("{md}"),
    }
    Ok(())
}
