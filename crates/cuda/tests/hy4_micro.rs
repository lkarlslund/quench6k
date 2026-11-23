//! Hy4-flash small-step kernels at several sequences' rows (ignored by default):
//! `cargo test --release -p quench6k-cuda --test hy4_micro -- --ignored --nocapture --test-threads=1`.
//!
//! Each case checks bitwise that a batched launch gives every row the result it gets alone (one row per
//! launch), then times it inside a graph over distinct weight copies (more than L2 holds).

use quench6k_cuda::kernels::{self, WeightFormat};
use quench6k_cuda::{DeviceBuffer, Event, Graph, Stream};

/// Microseconds per call, median of five replays of a graph of `calls` calls.
fn time_graph(s: &Stream, g: &Graph, calls: usize) -> f64 {
    let (a, b) = (Event::timing().unwrap(), Event::timing().unwrap());
    g.launch(s).unwrap();
    s.synchronize().unwrap();
    let mut v: Vec<f64> = (0..5)
        .map(|_| {
            a.record(s).unwrap();
            g.launch(s).unwrap();
            b.record(s).unwrap();
            b.synchronize().unwrap();
            Event::elapsed_ms(&a, &b).unwrap() as f64 * 1000.0 / calls as f64
        })
        .collect();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v[2]
}

/// Deterministic pseudo-random bytes (xorshift).
fn bytes(n: usize, seed: u64, f: impl Fn(u8) -> u8) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            f((x >> 24) as u8)
        })
        .collect()
}

fn upload(s: &Stream, v: &[u8]) -> DeviceBuffer {
    let b = DeviceBuffer::new(v.len().max(256)).unwrap();
    s.upload(&b, v).unwrap();
    b
}

/// BF16 values of moderate magnitude: sign, exponent 120..=127, random mantissa.
fn bf16_bytes(n: usize, seed: u64) -> Vec<u8> {
    let r = bytes(n * 2, seed, |b| b);
    r.chunks_exact(2)
        .flat_map(|c| {
            let v: u16 = ((c[1] as u16 & 0x80) << 8) | ((120 + (c[1] as u16 & 7)) << 7) | (c[0] as u16 & 0x7f);
            v.to_le_bytes()
        })
        .collect()
}

struct Shape {
    name: &'static str,
    n: usize,
    k: usize,
    fmt: WeightFormat,
}

const SHAPES: &[Shape] = &[
    Shape { name: "o_proj sliding", n: 3072, k: 16384, fmt: WeightFormat::Fp8Block },
    Shape { name: "gate sliding", n: 16384, k: 3072, fmt: WeightFormat::Fp8Block },
    Shape { name: "q_b sliding", n: 16384, k: 1024, fmt: WeightFormat::Fp8Block },
    Shape { name: "shared gate", n: 1536, k: 3072, fmt: WeightFormat::Fp8Block },
    Shape { name: "up_lat", n: 3072, k: 1536, fmt: WeightFormat::Fp8Block },
    Shape { name: "q_a", n: 1024, k: 3072, fmt: WeightFormat::Bf16 },
    Shape { name: "kv_a", n: 576, k: 3072, fmt: WeightFormat::Bf16 },
    Shape { name: "router", n: 512, k: 3072, fmt: WeightFormat::Bf16 },
    Shape { name: "attn_in sliding", n: 1600, k: 3072, fmt: WeightFormat::Bf16 },
    Shape { name: "attn_in indexed", n: 1744, k: 3072, fmt: WeightFormat::Bf16 },
    Shape { name: "mlp_in", n: 4608, k: 3072, fmt: WeightFormat::Fp8Block },
    Shape { name: "o_proj indexed", n: 3072, k: 8192, fmt: WeightFormat::Fp8Block },
    Shape { name: "q_b indexed", n: 8192, k: 1024, fmt: WeightFormat::Fp8Block },
    Shape { name: "gate indexed", n: 8192, k: 3072, fmt: WeightFormat::Fp8Block },
    Shape { name: "idx q", n: 2048, k: 1024, fmt: WeightFormat::Bf16 },
    Shape { name: "dense gate", n: 13824, k: 3072, fmt: WeightFormat::Bf16 },
    Shape { name: "dense down", n: 3072, k: 13824, fmt: WeightFormat::Bf16 },
    Shape { name: "engram kv", n: 15360, k: 1536, fmt: WeightFormat::Bf16 },
    Shape { name: "lm_head", n: 120832, k: 3072, fmt: WeightFormat::Fp8Block },
];

struct Weights {
    w: Vec<DeviceBuffer>,
    sc: Vec<DeviceBuffer>,
}

fn weights(s: &Stream, sh: &Shape, copies: usize) -> Weights {
    let (n, k) = (sh.n, sh.k);
    let mut w = Vec::new();
    let mut sc = Vec::new();
    for c in 0..copies {
        match sh.fmt {
            WeightFormat::Bf16 => {
                w.push(upload(s, &bf16_bytes(n * k, 7 + c as u64)));
                sc.push(DeviceBuffer::new(256).unwrap());
            }
            _ => {
                // E4M3 without NaN codes (0x7f / 0xff).
                w.push(upload(s, &bytes(n * k, 11 + c as u64, |b| if b & 0x7f == 0x7f { b & 0xfe } else { b })));
                let blocks = n.div_ceil(128) * (k / 128);
                let v: Vec<f32> = (0..blocks).map(|i| 0.001 + (i % 7) as f32 * 0.0003).collect();
                let b = DeviceBuffer::new(blocks * 4).unwrap();
                s.upload(&b, &v).unwrap();
                sc.push(b);
            }
        }
    }
    Weights { w, sc }
}

/// Today's small-step launches: rows in launches of gemv_max_m.
fn launch_old(s: &Stream, m: usize, x: u64, sh: &Shape, w: u64, sc: u64, y: u64) {
    let gm = kernels::gemv_max_m();
    for r0 in (0..m).step_by(gm) {
        let rows = gm.min(m - r0);
        kernels::gemv(s, rows, x + (r0 * sh.k * 2) as u64, sh.k, sh.fmt, w, sc, y + (r0 * sh.n * 4) as u64, sh.n, 0, sh.n, sh.k, 1, 1.0, false)
            .unwrap();
    }
}

#[test]
#[ignore]
fn hy4_gemv_grouped() {
    let s = Stream::new().unwrap();
    let only: Option<String> = std::env::var("HY4_MICRO_SHAPE").ok();
    let ms: Vec<usize> = std::env::var("HY4_MICRO_M")
        .ok()
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![1, 3, 8, 12, 16, 20, 32, 48, 64]);
    // Grouped launches of the shared GEMV (mg, r), then the wide kernel (mt, r) marked w.
    // HY4_MICRO_CFG: e.g. "g8x4,w8x2" (g: grouped shared GEMV, w: wide).
    let configs: Vec<(usize, usize, bool)> = std::env::var("HY4_MICRO_CFG")
        .ok()
        .map(|v| {
            v.split(',')
                .map(|c| {
                    let (mg, r) = c[1..].split_once('x').unwrap();
                    (mg.parse().unwrap(), r.parse().unwrap(), c.starts_with('w'))
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(8, 4, false), (16, 2, false), (8, 4, true), (16, 4, true)]);
    let mmax = 64;
    for sh in SHAPES {
        if only.as_deref().is_some_and(|o| !sh.name.contains(o)) {
            continue;
        }
        let bytes_per = sh.n * sh.k * if sh.fmt == WeightFormat::Bf16 { 2 } else { 1 };
        let copies = (400 << 20) / bytes_per + 1;
        let copies = copies.clamp(2, 64);
        let wt = weights(&s, sh, copies);
        let x = upload(&s, &bf16_bytes(mmax * sh.k, 3));
        let y_ref = DeviceBuffer::new(mmax * sh.n * 4).unwrap();
        let y = DeviceBuffer::new(mmax * sh.n * 4).unwrap();
        // Reference: one row per launch.
        for m in 0..mmax {
            kernels::gemv(
                &s,
                1,
                x.addr() + (m * sh.k * 2) as u64,
                sh.k,
                sh.fmt,
                wt.w[0].addr(),
                wt.sc[0].addr(),
                y_ref.addr() + (m * sh.n * 4) as u64,
                sh.n,
                0,
                sh.n,
                sh.k,
                1,
                1.0,
                false,
            )
            .unwrap();
        }
        let want: Vec<u32> = s.download(y_ref.addr(), mmax * sh.n).unwrap();
        println!("{} [{} x {}] {:?}, {} copies", sh.name, sh.n, sh.k, sh.fmt as i32, copies);
        for &m in &ms {
            let calls = copies;
            let g = Graph::capture(&s, |s| {
                for c in 0..calls {
                    launch_old(s, m, x.addr(), sh, wt.w[c].addr(), wt.sc[c].addr(), y.addr());
                }
                Ok(())
            })
            .unwrap();
            let old = time_graph(&s, &g, calls);
            let mut line = format!("  m {m:>2}: old {old:7.1} us |");
            for &(mg, r, wide) in &configs {
                let run = |s: &Stream, c: usize| {
                    if wide {
                        quench6k_cuda::hy4::gemv_wide(s, m, mg, r, x.addr(), sh.k, sh.fmt, wt.w[c].addr(), wt.sc[c].addr(), y.addr(), sh.n, sh.n, sh.k)
                    } else {
                        kernels::gemv_grouped(s, m, mg, r, x.addr(), sh.k, sh.fmt, wt.w[c].addr(), wt.sc[c].addr(), y.addr(), sh.n, sh.n, sh.k)
                    }
                };
                s.memset(y.addr(), 0xff, mmax * sh.n * 4).unwrap();
                run(&s, 0).unwrap();
                let got: Vec<u32> = s.download(y.addr(), m * sh.n).unwrap();
                let same = got == want[..m * sh.n];
                let g = Graph::capture(&s, |s| {
                    for c in 0..calls {
                        run(s, c)?;
                    }
                    Ok(())
                })
                .unwrap();
                let t = time_graph(&s, &g, calls);
                line += &format!(" {}{mg}x{r} {t:6.1}{}", if wide { "w" } else { "" }, if same { "" } else { " DIFF" });
                assert!(same, "{}: m {m} mg {mg} r {r}: rows differ from one-row launches", sh.name);
            }
            println!("{line}");
        }
    }
}

/// The sorting router against the 16-round argmax router: random logits and biases, many of them tied
/// (values on a coarse grid), every row's experts and weights bitwise.
#[test]
#[ignore]
fn hy4_route_sort() {
    let s = Stream::new().unwrap();
    const E: usize = 512;
    for (case, grid) in [(0u64, 0.0f32), (1, 0.25), (2, 1.0), (3, 4.0)] {
        let t = 64;
        let r = bytes(t * E * 2 + E * 2, 101 + case, |b| b);
        let val = |i: usize| -> f32 {
            let x = (u16::from_le_bytes([r[2 * i], r[2 * i + 1]]) as f32 / 65535.0 - 0.5) * 8.0;
            if grid > 0.0 { (x * grid).round() / grid } else { x }
        };
        let logits: Vec<f32> = (0..t * E).map(val).collect();
        let bias: Vec<f32> = (0..E).map(|i| val(t * E + i) * 0.01).collect();
        let lb = DeviceBuffer::new(t * E * 4).unwrap();
        let bb = DeviceBuffer::new(E * 4).unwrap();
        s.upload(&lb, &logits).unwrap();
        s.upload(&bb, &bias).unwrap();
        let run = |legacy: bool| -> (Vec<i32>, Vec<u32>) {
            let e = DeviceBuffer::new(t * 16 * 4).unwrap();
            let w = DeviceBuffer::new(t * 16 * 4).unwrap();
            quench6k_cuda::hy4::set_legacy(legacy);
            quench6k_cuda::hy4::route(&s, lb.addr(), E, bb.addr(), 3.998, t, e.addr(), w.addr()).unwrap();
            s.synchronize().unwrap();
            quench6k_cuda::hy4::set_legacy(false);
            (s.download(e.addr(), t * 16).unwrap(), s.download(w.addr(), t * 16).unwrap())
        };
        let (a, b) = (run(true), run(false));
        println!("grid {grid}: experts {}, weights {}", if a.0 == b.0 { "same" } else { "DIFF" }, if a.1 == b.1 { "same" } else { "DIFF" });
        assert!(a == b, "grid {grid}: the sorting router differs");
    }
}
