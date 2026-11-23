//! Inkling routed experts of small steps (ignored by default):
//! `cargo test --release -p quench6k-cuda --test inkling_experts -- --ignored --nocapture`.
//!
//! Random NVFP4 expert stacks (128 experts of gate/up 2304 x 2 x 1152 and down 1152 x 2304, several layers:
//! more than L2 holds) and random routing of T rows; each kernel variant's activations and outputs are
//! compared bitwise with the per-pair kernels', and timed per layer in a CUDA graph.

use quench6k_cuda::inkling::{self as ik, ExpertStack};
use quench6k_cuda::{DeviceBuffer, Event, Graph, Stream};

const D: usize = 2304;
const FF: usize = 1152;
const E: usize = 128;
const TOPK: usize = 8;
const LAYERS: usize = 6;

fn random_bytes(s: &Stream, n: usize, seed: u32, lo: u8, span: u8) -> DeviceBuffer {
    let mut x = seed | 1;
    let v: Vec<u8> = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            if span == 0 { x as u8 } else { lo + (x % span as u32) as u8 }
        })
        .collect();
    let b = DeviceBuffer::new(n).unwrap();
    s.upload(&b, &v).unwrap();
    b
}

fn stack(s: &Stream, rows: usize, k: usize, seed: u32, keep: &mut Vec<DeviceBuffer>) -> ExpertStack {
    let w = random_bytes(s, E * rows * k / 2, seed, 0, 0);
    // E4M3 block scales between 2^-4 and 2^-1 (finite, no NaN).
    let sc = random_bytes(s, E * rows * k / 16, seed ^ 0x55, 0x18, 0x18);
    let s2 = DeviceBuffer::new(E * 4).unwrap();
    s.upload(&s2, &vec![0.37f32; E]).unwrap();
    let st = ExpertStack { w: w.addr(), sc: sc.addr(), s2: s2.addr() };
    keep.extend([w, sc, s2]);
    st
}

/// Microseconds per graph replay, median of five.
fn time_graph(s: &Stream, g: &Graph) -> f64 {
    let (a, b) = (Event::timing().unwrap(), Event::timing().unwrap());
    g.launch(s).unwrap();
    s.synchronize().unwrap();
    let mut v: Vec<f64> = (0..5)
        .map(|_| {
            a.record(s).unwrap();
            g.launch(s).unwrap();
            b.record(s).unwrap();
            b.synchronize().unwrap();
            Event::elapsed_ms(&a, &b).unwrap() as f64 * 1000.0
        })
        .collect();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v[2]
}

#[test]
#[ignore]
fn inkling_experts() {
    quench6k_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    let mut keep = Vec::new();
    let layers: Vec<(ExpertStack, ExpertStack)> =
        (0..LAYERS).map(|l| (stack(&s, 2 * FF, D, 11 + l as u32 * 7, &mut keep), stack(&s, D, FF, 1001 + l as u32 * 13, &mut keep))).collect();
    let rows_max = 64;
    let pairs_max = rows_max * TOPK;
    // BF16 activations around +-1.
    let x = {
        let mut v = Vec::with_capacity(rows_max * D);
        let mut r = 0x1234_5678u32;
        for _ in 0..rows_max * D {
            r ^= r << 13;
            r ^= r >> 17;
            r ^= r << 5;
            let f = (r % 2001) as f32 / 1000.0 - 1.0;
            v.push((f.to_bits() >> 16) as u16);
        }
        let b = DeviceBuffer::new(v.len() * 2).unwrap();
        s.upload(&b, &v).unwrap();
        b
    };
    let logits: Vec<DeviceBuffer> = (0..LAYERS)
        .map(|l| {
            let mut r = 0x9e37_79b9u32 ^ (l as u32 * 77);
            let v: Vec<f32> = (0..rows_max * E)
                .map(|_| {
                    r ^= r << 13;
                    r ^= r >> 17;
                    r ^= r << 5;
                    (r % 10007) as f32 / 1000.0 - 5.0
                })
                .collect();
            let b = DeviceBuffer::new(v.len() * 4).unwrap();
            s.upload(&b, &v).unwrap();
            b
        })
        .collect();
    let bias = DeviceBuffer::zeroed(E * 4).unwrap();
    let buf = |bytes: usize| DeviceBuffer::zeroed(bytes).unwrap();
    // Per layer: routing (so a graph's layers keep their own) and outputs per variant.
    let sel_e: Vec<DeviceBuffer> = (0..LAYERS).map(|_| buf(pairs_max * 4)).collect();
    let sel_w: Vec<DeviceBuffer> = (0..LAYERS).map(|_| buf(pairs_max * 4)).collect();
    let order: Vec<DeviceBuffer> = (0..LAYERS).map(|_| buf(pairs_max * 4)).collect();
    let groups: Vec<DeviceBuffer> = (0..LAYERS).map(|_| buf((1 + 2 * pairs_max) * 4)).collect();
    let variants = variants();
    let act: Vec<DeviceBuffer> = (0..variants.len()).map(|_| buf(pairs_max * FF * 2)).collect();
    let y: Vec<DeviceBuffer> = (0..variants.len()).map(|_| buf(pairs_max * D * 4)).collect();
    let ts: Vec<usize> = std::env::var("INK_T").ok().map_or(vec![1, 2, 4, 8, 12, 16, 24, 32, 48, 64], |v| v.split(',').map(|x| x.parse().unwrap()).collect());
    for t in ts {
        let pairs = t * TOPK;
        let mut line = format!("T {t:2}:");
        let mut distinct = 0usize;
        for l in 0..LAYERS {
            ik::route(&s, logits[l].addr(), bias.addr(), 2.5, t, sel_e[l].addr(), sel_w[l].addr(), order[l].addr(), groups[l].addr()).unwrap();
            let e: Vec<i32> = s.download(sel_e[l].addr(), pairs).unwrap();
            let mut seen = [false; E];
            for &x in &e {
                seen[x as usize] = true;
            }
            distinct += seen.iter().filter(|&&b| b).count();
        }
        let floor_bytes = distinct as f64 / LAYERS as f64 * 3.0 * (FF * D) as f64 * (0.5 + 1.0 / 16.0);
        let mut base: Option<(Vec<u16>, Vec<u32>)> = None;
        for (vi, (name, var)) in variants.iter().enumerate() {
            if !var.applies(t) {
                continue;
            }
            let run = |s: &Stream| -> anyhow::Result<()> {
                for (l, (gu, dn)) in layers.iter().enumerate() {
                    var.run(s, t, x.addr(), sel_e[l].addr(), sel_w[l].addr(), order[l].addr(), groups[l].addr(), *gu, *dn, act[vi].addr(), y[vi].addr())?;
                }
                Ok(())
            };
            // Bits: the last layer's activations and outputs against the first variant's.
            run(&s).unwrap();
            let a: Vec<u16> = s.download(act[vi].addr(), pairs * FF).unwrap();
            let yy: Vec<u32> = s.download::<f32>(y[vi].addr(), pairs * D).unwrap().iter().map(|f| f.to_bits()).collect();
            let same = match &base {
                None => {
                    // A digest of the base outputs, to compare builds.
                    let mut h = 0xcbf29ce484222325u64;
                    for b in a.iter().flat_map(|v| v.to_le_bytes()).chain(yy.iter().flat_map(|v| v.to_le_bytes())) {
                        h ^= b as u64;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                    base = Some((a, yy));
                    format!("base {h:016x}")
                }
                Some((ba, by)) => {
                    let da = ba.iter().zip(&a).filter(|(p, q)| p != q).count();
                    let dy = by.iter().zip(&yy).filter(|(p, q)| p != q).count();
                    if da == 0 && dy == 0 { "same".to_string() } else { format!("DIFF act {da} y {dy}") }
                }
            };
            let g = Graph::capture(&s, run).unwrap();
            let us = time_graph(&s, &g) / LAYERS as f64;
            line += &format!("  {name} {us:6.1} us ({:4.2} TB/s, {same})", floor_bytes / us / 1e6);
        }
        println!("{line}  [{:.1} distinct experts]", distinct as f64 / LAYERS as f64);
    }
}

struct Variant {
    groups: bool,
    staged: bool,
    min_t: usize,
}

impl Variant {
    fn applies(&self, t: usize) -> bool {
        t >= self.min_t
    }
    #[allow(clippy::too_many_arguments)]
    fn run(&self, s: &Stream, t: usize, x: u64, sel_e: u64, sel_w: u64, order: u64, groups: u64, gu: ExpertStack, dn: ExpertStack, act: u64, y: u64) -> anyhow::Result<()> {
        let g = if self.groups { groups } else { 0 };
        ik::experts(s, x, sel_e, sel_w, true, gu, dn, 4.0, 25.0, t, order, true, g, self.staged, act, y, 0)
    }
}

fn variants() -> Vec<(&'static str, Variant)> {
    vec![
        ("pair", Variant { groups: false, staged: false, min_t: 1 }),
        ("group", Variant { groups: true, staged: false, min_t: 2 }),
        ("staged", Variant { groups: false, staged: true, min_t: 1 }),
    ]
}

/// The LM head (FP8 block-128, 200064 x 2304) for M rows: the shared GEMV in launches of at most 16 rows
/// against ik::lm_rows (one pass), bitwise and timed.
#[test]
#[ignore]
fn inkling_lm_rows() {
    use quench6k_cuda::kernels::{self as k, WeightFormat};
    quench6k_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    let (n, kk) = (200064usize, D);
    let w = random_bytes(&s, n * kk, 77, 0, 0x7e);
    let sc = {
        let v: Vec<f32> = (0..n.div_ceil(128) * (kk / 128)).map(|i| 0.001 + (i % 97) as f32 * 1e-4).collect();
        let b = DeviceBuffer::new(v.len() * 4).unwrap();
        s.upload(&b, &v).unwrap();
        b
    };
    let rows_max = 64;
    let x = random_bytes(&s, rows_max * 2 * D * 2, 5, 0, 0);
    // BF16 inputs: random bytes with the exponent bytes bounded (|x| < 4, no NaN).
    {
        let mut v: Vec<u16> = s.download(x.addr(), rows_max * 2 * D).unwrap();
        for e in &mut v {
            *e = (*e & 0x807f) | (((*e >> 7) & 0x1f) + 0x70) << 7;
        }
        s.upload(&x, &v).unwrap();
    }
    let ldx = 2 * D;
    let y0 = DeviceBuffer::zeroed(rows_max * n * 4).unwrap();
    let y1 = DeviceBuffer::zeroed(rows_max * n * 4).unwrap();
    let g = k::gemv_max_m();
    let ms: Vec<usize> = std::env::var("INK_M").ok().map_or(vec![1, 4, 8, 16, 32, 56, 64], |v| v.split(',').map(|x| x.parse().unwrap()).collect());
    for m in ms {
        let gemv = |s: &Stream| -> anyhow::Result<()> {
            for r0 in (0..m).step_by(g) {
                let mm = g.min(m - r0);
                k::gemv(s, mm, x.addr() + (r0 * ldx * 2) as u64, ldx, WeightFormat::Fp8Block, w.addr(), sc.addr(), y0.addr() + (r0 * n * 4) as u64, n, 0, n, kk, 1, 1.0, false)?;
            }
            Ok(())
        };
        let lm = |s: &Stream| -> anyhow::Result<()> { ik::lm_rows(s, x.addr(), ldx, m, 0, 0, w.addr(), sc.addr(), y1.addr(), n, n, kk) };
        gemv(&s).unwrap();
        lm(&s).unwrap();
        let a: Vec<u32> = s.download::<f32>(y0.addr(), m * n).unwrap().iter().map(|f| f.to_bits()).collect();
        let b: Vec<u32> = s.download::<f32>(y1.addr(), m * n).unwrap().iter().map(|f| f.to_bits()).collect();
        let diff = a.iter().zip(&b).filter(|(p, q)| p != q).count();
        let ga = Graph::capture(&s, gemv).unwrap();
        let gb = Graph::capture(&s, lm).unwrap();
        let (ta, tb) = (time_graph(&s, &ga), time_graph(&s, &gb));
        println!("M {m:2}: gemv {ta:7.1} us, lm_rows {tb:7.1} us ({:.2} TB/s), {}", (n * kk) as f64 / tb / 1e6, if diff == 0 { "bitwise same".to_string() } else { format!("{diff} DIFFER") });
        if m % 8 == 0 {
            // The drafter's blocks of 8 rows without their first rows: m / 8 * 7 rows.
            let mb = m / 8 * 7;
            s.memset(y1.addr(), 0, m * n * 4).unwrap();
            ik::lm_rows(&s, x.addr(), ldx, mb, 8, 7, w.addr(), sc.addr(), y1.addr(), n, n, kk).unwrap();
            let b: Vec<u32> = s.download::<f32>(y1.addr(), m * n).unwrap().iter().map(|f| f.to_bits()).collect();
            let diff = (0..m).filter(|r| r % 8 != 0).map(|r| (0..n).filter(|&c| a[r * n + c] != b[r * n + c]).count()).sum::<usize>();
            let zero = (0..m).filter(|r| r % 8 == 0).all(|r| b[r * n..(r + 1) * n].iter().all(|&v| v == 0));
            let gb = Graph::capture(&s, |s| ik::lm_rows(s, x.addr(), ldx, mb, 8, 7, w.addr(), sc.addr(), y1.addr(), n, n, kk)).unwrap();
            println!("  {mb} rows of blocks of 8 (first rows skipped): {:7.1} us, {}", time_graph(&s, &gb), if diff == 0 && zero { "bitwise same".to_string() } else { format!("{diff} DIFFER, skipped rows untouched {zero}") });
            // Only the first 3 rows after each block's first (a drafter pass for 3 drafts).
            let mp = m / 8 * 3;
            s.memset(y1.addr(), 0, m * n * 4).unwrap();
            ik::lm_rows(&s, x.addr(), ldx, mp, 8, 3, w.addr(), sc.addr(), y1.addr(), n, n, kk).unwrap();
            let b: Vec<u32> = s.download::<f32>(y1.addr(), m * n).unwrap().iter().map(|f| f.to_bits()).collect();
            let used = |r: usize| (1..=3).contains(&(r % 8));
            let diff = (0..m).filter(|&r| used(r)).map(|r| (0..n).filter(|&c| a[r * n + c] != b[r * n + c]).count()).sum::<usize>();
            let zero = (0..m).filter(|&r| !used(r)).all(|r| b[r * n..(r + 1) * n].iter().all(|&v| v == 0));
            let gb = Graph::capture(&s, |s| ik::lm_rows(s, x.addr(), ldx, mp, 8, 3, w.addr(), sc.addr(), y1.addr(), n, n, kk)).unwrap();
            println!("  {mp} rows of blocks of 8 (3 after the first): {:7.1} us, {}", time_graph(&s, &gb), if diff == 0 && zero { "bitwise same".to_string() } else { format!("{diff} DIFFER, other rows untouched {zero}") });
        }
    }
}

/// Top-k of vocabulary rows whose length is not a multiple of the slice count (Inkling: 200058): the
/// two-stage path against one block per row (QUENCH6K_TOPK_ONE_PASS), bitwise, with ties; and timed.
#[test]
#[ignore]
fn topk_odd_rows() {
    use quench6k_cuda::kernels as k;
    quench6k_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    for &n in &[200058usize, 151936, 65537, 131071] {
        let rows = 8;
        let mut r = 0x2545_f491u32 ^ n as u32;
        let v: Vec<f32> = (0..rows * n)
            .map(|i| {
                r ^= r << 13;
                r ^= r >> 17;
                r ^= r << 5;
                // Coarse values: many exact ties, also among the largest.
                if i % 7 == 0 { 9.0 } else { (r % 4001) as f32 / 400.0 - 5.0 }
            })
            .collect();
        let x = DeviceBuffer::new(v.len() * 4).unwrap();
        s.upload(&x, &v).unwrap();
        let idx = DeviceBuffer::zeroed(rows * 512 * 4).unwrap();
        let val = DeviceBuffer::zeroed(rows * 512 * 4).unwrap();
        for &kk in &[1usize, 2, 20, 64, 256] {
            for &by_index in &[false, true] {
                let run = |one: bool| -> (Vec<i32>, Vec<u32>, f64) {
                    if one {
                        unsafe { std::env::set_var("QUENCH6K_TOPK_ONE_PASS", "1") };
                    } else {
                        unsafe { std::env::remove_var("QUENCH6K_TOPK_ONE_PASS") };
                    }
                    k::topk(&s, x.addr(), rows, n, n, kk, by_index, idx.addr(), val.addr()).unwrap();
                    let a: Vec<i32> = s.download(idx.addr(), rows * kk).unwrap();
                    let b: Vec<u32> = s.download::<f32>(val.addr(), rows * kk).unwrap().iter().map(|f| f.to_bits()).collect();
                    let g = Graph::capture(&s, |s| k::topk(s, x.addr(), 1, n, n, kk, by_index, idx.addr(), val.addr())).unwrap();
                    (a, b, time_graph(&s, &g))
                };
                let (a1, b1, t1) = run(true);
                let (a2, b2, t2) = run(false);
                println!("n {n} k {kk:3} by_index {by_index}: one pass {t1:6.1} us, two stages {t2:6.1} us (1 row): {}", if a1 == a2 && b1 == b2 { "bitwise same" } else { "DIFFER" });
            }
        }
    }
    unsafe { std::env::remove_var("QUENCH6K_TOPK_ONE_PASS") };
}
