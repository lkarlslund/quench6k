//! Kernel latency microbenchmarks (ignored by default): `cargo test --release -p quench6k-cuda --test micro -- --ignored --nocapture`.
//!
//! Each case captures a graph of many dependent launches over distinct weights (more than L2 holds),
//! so per-call times include weight streaming the way a decode step sees it.

use quench6k_cuda::kernels::{self, WeightFormat};
use quench6k_cuda::{DeviceBuffer, Event, Graph, Stream};

const D: usize = 2560;
const H: usize = 4 * D;
const RANK: usize = 320;
const CALLS: usize = 96;

fn filled(s: &Stream, bytes: usize, byte: u8) -> DeviceBuffer {
    let b = DeviceBuffer::new(bytes).unwrap();
    s.memset(b.addr(), byte, bytes).unwrap();
    b
}

fn scales(s: &Stream, n: usize) -> DeviceBuffer {
    let b = DeviceBuffer::new(n * 4).unwrap();
    s.upload(&b, &vec![0.01f32; n]).unwrap();
    b
}

/// Microseconds per graph-body repetition, median of five replays.
fn time_graph(s: &Stream, g: &Graph, reps: usize) -> f64 {
    let (a, b) = (Event::timing().unwrap(), Event::timing().unwrap());
    g.launch(s).unwrap();
    s.synchronize().unwrap();
    let mut v: Vec<f64> = (0..5)
        .map(|_| {
            a.record(s).unwrap();
            g.launch(s).unwrap();
            b.record(s).unwrap();
            b.synchronize().unwrap();
            Event::elapsed_ms(&a, &b).unwrap() as f64 * 1000.0 / reps as f64
        })
        .collect();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v[2]
}

struct HcWeights {
    down: DeviceBuffer,
    down_s: DeviceBuffer,
    inj: DeviceBuffer,
    inj_s: DeviceBuffer,
    up: DeviceBuffer,
    up_s: DeviceBuffer,
    g: DeviceBuffer,
}

fn hc_weights(s: &Stream) -> HcWeights {
    let gr = kernels::int8_group();
    HcWeights {
        down: filled(s, RANK * H, 3),
        down_s: scales(s, RANK * H / gr),
        inj: filled(s, 4 * H, 3),
        inj_s: scales(s, 4 * H / gr),
        up: filled(s, H * RANK, 3),
        up_s: scales(s, H * RANK / gr),
        g: filled(s, 4 * D * 2, 0),
    }
}

struct Gemv {
    w: DeviceBuffer,
    s: DeviceBuffer,
    n: usize,
}

fn gemv_weights(s: &Stream, n: usize) -> Gemv {
    Gemv {
        w: filled(s, n * D, 3),
        s: scales(s, n * D / kernels::int8_group()),
        n,
    }
}

#[test]
#[ignore]
fn hc_latency() {
    quench6k_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    let hcs: Vec<HcWeights> = (0..CALLS).map(|_| hc_weights(&s)).collect();
    let gemvs: Vec<Gemv> = (0..CALLS).map(|_| gemv_weights(&s, 16480)).collect();
    let small: Vec<Gemv> = (0..CALLS).map(|_| gemv_weights(&s, D)).collect();
    let t_max = 16;
    let h = filled(&s, t_max * H * 4, 0);
    let y = filled(&s, t_max * D * 4, 0);
    let inj_prev = filled(&s, t_max * 4 * 4, 0);
    let scratch = DeviceBuffer::zeroed(kernels::hc_small_scratch_floats(t_max) * 4).unwrap();
    let z = filled(&s, t_max * RANK * 2, 0);
    let r = filled(&s, t_max * 16, 0);
    let inj = filled(&s, t_max * 16, 0);
    let m = filled(&s, t_max * D * 2, 0);
    let out = filled(&s, t_max * 16480 * 4, 0);
    let partial = filled(&s, 1 << 24, 0);
    let fmt = WeightFormat::Int8Group;
    let hc = |s: &Stream, w: &HcWeights, t: usize| {
        kernels::hc_small(
            s,
            h.addr(),
            t,
            y.addr(),
            inj_prev.addr(),
            (fmt, w.down.addr(), w.down_s.addr()),
            Some((fmt, w.inj.addr(), w.inj_s.addr())),
            (fmt, w.up.addr(), w.up_s.addr()),
            w.g.addr(),
            scratch.addr(),
            z.addr(),
            r.addr(),
            inj.addr(),
            m.addr(),
        )
    };
    let gemv = |s: &Stream, w: &Gemv, t: usize, x: u64| {
        kernels::gemv(
            s,
            t,
            x,
            D,
            fmt,
            w.w.addr(),
            w.s.addr(),
            out.addr(),
            w.n,
            partial.addr(),
            w.n,
            D,
            1,
            1.0,
            false,
        )
    };
    for t in [1usize, 4] {
        let g = Graph::capture(&s, |s| hcs.iter().try_for_each(|w| hc(s, w, t))).unwrap();
        let hc_only = time_graph(&s, &g, CALLS);
        let g = Graph::capture(&s, |s| {
            gemvs.iter().try_for_each(|w| gemv(s, w, t, m.addr()))
        })
        .unwrap();
        let big_only = time_graph(&s, &g, CALLS);
        let g = Graph::capture(&s, |s| {
            small.iter().try_for_each(|w| gemv(s, w, t, m.addr()))
        })
        .unwrap();
        let small_only = time_graph(&s, &g, CALLS);
        let g = Graph::capture(&s, |s| {
            hcs.iter().zip(&gemvs).try_for_each(|(a, b)| {
                hc(s, a, t)?;
                gemv(s, b, t, m.addr())
            })
        })
        .unwrap();
        let hc_big = time_graph(&s, &g, CALLS);
        let g = Graph::capture(&s, |s| {
            hcs.iter().zip(&small).try_for_each(|(a, b)| {
                gemv(s, b, t, m.addr())?;
                hc(s, a, t)
            })
        })
        .unwrap();
        let small_hc = time_graph(&s, &g, CALLS);
        println!(
            "T={t}: hc {hc_only:.2} us, gemv 16480 {big_only:.2}, gemv 2560 {small_only:.2}, hc+gemv16480 {hc_big:.2}, gemv2560+hc {small_hc:.2}"
        );
    }
}

/// Routed experts for small batches: route + gate/up + down over 40 layers of random routing across
/// 512 distinct NVFP4 experts (1.4 GB, far beyond L2). Reports time per layer and the bandwidth of the
/// distinct experts' weights.
#[test]
#[ignore]
fn expert_latency() {
    quench6k_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    const E: usize = 512;
    const W: usize = 640;
    const LAYERS: usize = 40;
    const TOPK: usize = 10;
    // Codes 0x22 (small FP4 values) and FP8 scales 0x38 (1.0) everywhere.
    let gate_codes: Vec<DeviceBuffer> = (0..E).map(|_| filled(&s, W * D / 2, 0x22)).collect();
    let gate_scales: Vec<DeviceBuffer> = (0..E).map(|_| filled(&s, W * D / 16, 0x38)).collect();
    let up_codes: Vec<DeviceBuffer> = (0..E).map(|_| filled(&s, W * D / 2, 0x22)).collect();
    let up_scales: Vec<DeviceBuffer> = (0..E).map(|_| filled(&s, W * D / 16, 0x38)).collect();
    let down_codes: Vec<DeviceBuffer> = (0..E).map(|_| filled(&s, D * W / 2, 0x22)).collect();
    let down_scales: Vec<DeviceBuffer> = (0..E).map(|_| filled(&s, D * W / 16, 0x38)).collect();
    let table = |v: &[DeviceBuffer]| {
        let b = DeviceBuffer::new(E * 8).unwrap();
        s.upload(&b, &v.iter().map(|x| x.addr()).collect::<Vec<u64>>())
            .unwrap();
        b
    };
    let ones = {
        let b = DeviceBuffer::new(E * 4).unwrap();
        s.upload(&b, &vec![1.0f32; E]).unwrap();
        b
    };
    let (gc, gs, uc, us, dc, ds) = (
        table(&gate_codes),
        table(&gate_scales),
        table(&up_codes),
        table(&up_scales),
        table(&down_codes),
        table(&down_scales),
    );
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 40) as f32 / (1u64 << 24) as f32
    };
    let t_max = 16;
    let logits: Vec<DeviceBuffer> = (0..LAYERS)
        .map(|_| {
            let b = DeviceBuffer::new(t_max * E * 4).unwrap();
            s.upload(
                &b,
                &(0..t_max * E).map(|_| next() * 8.0).collect::<Vec<f32>>(),
            )
            .unwrap();
            b
        })
        .collect();
    let m = filled(&s, t_max * D * 2, 0);
    let pairs = t_max * TOPK;
    let (top_i, top_w, pair_row, row_token) = (
        filled(&s, pairs * 4, 0),
        filled(&s, pairs * 4, 0),
        filled(&s, pairs * 4, 0),
        filled(&s, pairs * 4, 0),
    );
    let (offsets, active, num_active) = (
        filled(&s, (E + 1) * 4, 0),
        filled(&s, E * 4, 0),
        filled(&s, 4, 0),
    );
    let act = filled(&s, pairs * W * 2, 0);
    let yd = filled(&s, pairs * D * 4, 0);
    for t in [1usize, 2, 4, 5] {
        let max_active = (t * TOPK).min(E);
        let layer = |s: &Stream, l: usize| -> anyhow::Result<()> {
            kernels::moe_route_small(
                s,
                logits[l].addr(),
                E,
                t,
                top_i.addr(),
                top_w.addr(),
                offsets.addr(),
                pair_row.addr(),
                row_token.addr(),
                active.addr(),
                num_active.addr(),
            )?;
            kernels::moe_gate_up_small(
                s,
                m.addr(),
                row_token.addr(),
                (gc.addr(), gs.addr(), ones.addr()),
                (uc.addr(), us.addr(), ones.addr()),
                offsets.addr(),
                active.addr(),
                num_active.addr(),
                max_active,
                t,
                act.addr(),
            )?;
            kernels::moe_down_small(
                s,
                act.addr(),
                (dc.addr(), ds.addr(), ones.addr()),
                offsets.addr(),
                active.addr(),
                num_active.addr(),
                max_active,
                t,
                yd.addr(),
            )
        };
        let mut distinct = 0usize;
        for l in 0..LAYERS {
            layer(&s, l).unwrap();
            distinct += s.download::<i32>(num_active.addr(), 1).unwrap()[0] as usize;
        }
        let g = Graph::capture(&s, |s| (0..LAYERS).try_for_each(|l| layer(s, l))).unwrap();
        let us_layer = time_graph(&s, &g, LAYERS);
        let bytes = distinct as f64 / LAYERS as f64 * (3 * W * D) as f64 * (0.5 + 1.0 / 16.0);
        println!(
            "T={t}: {us_layer:.2} us per layer, {:.1} experts, {:.2} TB/s",
            distinct as f64 / LAYERS as f64,
            bytes / us_layer / 1e6
        );
    }
}

/// INT8G GEMV chains (96 distinct weights each) for the decode projection shapes; splits > 1 adds
/// the partial-sum reduction kernel.
#[test]
#[ignore]
fn gemv_latency() {
    quench6k_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    let fmt = WeightFormat::Int8Group;
    let x = filled(&s, 16 * 16480 * 2, 0);
    let out = filled(&s, 16 * 16480 * 4, 0);
    let partial = filled(&s, 1 << 24, 0);
    for &(n, k, splits) in &[
        (2560usize, 6144usize, 1usize),
        (2560, 6144, 2),
        (2560, 640, 1),
        (16480, 2560, 1),
        (13952, 2560, 1),
        (3600, 2560, 1),
    ] {
        let ws: Vec<(DeviceBuffer, DeviceBuffer)> = (0..CALLS)
            .map(|_| {
                (
                    filled(&s, n * k, 3),
                    scales(&s, n * k / kernels::int8_group()),
                )
            })
            .collect();
        let mut line = format!("N={n} K={k} splits={splits}:");
        for t in [1usize, 4, 8, 16] {
            let g = Graph::capture(&s, |s| {
                ws.iter().try_for_each(|(w, sc)| {
                    kernels::gemv(
                        s,
                        t,
                        x.addr(),
                        k,
                        fmt,
                        w.addr(),
                        sc.addr(),
                        out.addr(),
                        n,
                        partial.addr(),
                        n,
                        k,
                        splits,
                        1.0,
                        false,
                    )
                })
            })
            .unwrap();
            let us = time_graph(&s, &g, CALLS);
            let bytes = (n * k) as f64 * (1.0 + 4.0 / kernels::int8_group() as f64);
            line += &format!("  T={t} {us:.2} us {:.2} TB/s", bytes / us / 1e6);
        }
        println!("{line}");
    }
}

/// QSA group selection for four rows at several context lengths (continuous and tied scores).
#[test]
#[ignore]
fn select_latency() {
    quench6k_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    let mut rng = 0x1234_5678_9abc_def1u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 40) as f32 / (1u64 << 24) as f32
    };
    let t = 4;
    for &ctx in &[8192usize, 65536, 131072, 262144] {
        let ld = ctx / 4;
        for coarse in [false, true] {
            let v: Vec<f32> = (0..t * ld)
                .map(|_| {
                    if coarse {
                        (next() * 64.0).round() / 4.0
                    } else {
                        next() * 3.0
                    }
                })
                .collect();
            let scores = DeviceBuffer::new(v.len() * 4).unwrap();
            s.upload(&scores, &v).unwrap();
            let start = DeviceBuffer::new(4).unwrap();
            s.upload(&start, &[(ctx - t) as i32]).unwrap();
            let sel = DeviceBuffer::new(t * 512 * 4).unwrap();
            let g = Graph::capture(&s, |s| {
                (0..CALLS).try_for_each(|_| {
                    kernels::qsa_select(s, scores.addr(), t, ld, start.addr(), sel.addr())
                })
            })
            .unwrap();
            println!(
                "ctx {ctx} coarse {coarse}: {:.2} us",
                time_graph(&s, &g, CALLS)
            );
        }
    }
}

/// Dense INT8G and BF16 GEMM throughput on prefill-sized shapes (M = 4096 tokens).
#[test]
#[ignore]
fn gemm_throughput() {
    quench6k_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    let m = 4096;
    for &(n, k) in &[
        (16480usize, 2560usize),
        (2560, 6144),
        (13952, 2560),
        (324, 10240),
        (10240, 320),
    ] {
        let x = filled(&s, m * k * 2, 0);
        let y = filled(&s, m * n * 4, 0);
        let w8 = filled(&s, n * k, 3);
        let sc = scales(&s, n * k / kernels::int8_group());
        let wb = filled(&s, n * k * 2, 0);
        let mut line = format!("M={m} N={n} K={k}:");
        for (fmt, w, scale, name) in [
            (WeightFormat::Int8Group, &w8, sc.addr(), "int8g"),
            (WeightFormat::Bf16, &wb, 0, "bf16"),
        ] {
            let g = Graph::capture(&s, |s| {
                (0..8).try_for_each(|_| {
                    kernels::gemm_f32(
                        s,
                        x.addr(),
                        k,
                        fmt,
                        w.addr(),
                        k,
                        scale,
                        y.addr(),
                        n,
                        m,
                        n,
                        k,
                        1.0,
                        false,
                    )
                })
            })
            .unwrap();
            let us = time_graph(&s, &g, 8);
            line += &format!(
                "  {name} {us:.0} us {:.0} TFLOPS",
                2.0 * (m * n * k) as f64 / us / 1e6
            );
        }
        println!("{line}");
    }
}

/// The MTP draft head (1-bit sketch shortlist, NVFP4 rescoring) over the full vocabulary, per call.
#[test]
#[ignore]
fn draft_head_latency() {
    quench6k_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    const V: usize = 248_320;
    let mut seed = 0x2545F4914F6CDD1Du64;
    let w: Vec<u16> = (0..V * D)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            quench6k_cuda::bf16::from_f32(((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.05)
        })
        .collect();
    let wb = DeviceBuffer::new(V * D * 2).unwrap();
    s.upload(&wb, &w).unwrap();
    drop(w);
    let codes = DeviceBuffer::new(V * D / 2).unwrap();
    let sc = DeviceBuffer::new(V * D / 16).unwrap();
    let s2 = DeviceBuffer::new(4).unwrap();
    let amax = DeviceBuffer::new(4).unwrap();
    kernels::quantize_nvfp4(
        &s,
        wb.addr(),
        1,
        V,
        D,
        codes.addr(),
        sc.addr(),
        s2.addr(),
        amax.addr(),
    )
    .unwrap();
    let sketch = DeviceBuffer::new(kernels::draft_sketch_bytes(V)).unwrap();
    kernels::build_draft_sketch(&s, wb.addr(), sketch.addr(), V).unwrap();
    s.synchronize().unwrap();
    drop(wb);
    let x = filled(&s, D * 2, 0x3c);
    let ys = DeviceBuffer::new(V * 4).unwrap();
    let pick = DeviceBuffer::new(kernels::draft_pick_words(V) * 4).unwrap();
    let out = DeviceBuffer::new(64).unwrap();
    let reps = 50;
    let g = Graph::capture(&s, |s| {
        for _ in 0..reps {
            kernels::draft_head_argmax(
                s,
                x.addr(),
                sketch.addr(),
                (codes.addr(), sc.addr(), s2.addr()),
                ys.addr(),
                pick.addr(),
                V,
                out.addr(),
                0,
                0,
            )?;
        }
        Ok(())
    })
    .unwrap();
    println!(
        "draft head: {:.1} us per call ({} rows, sketch {:.1} MB)",
        time_graph(&s, &g, reps),
        V,
        kernels::draft_sketch_bytes(V) as f64 / 1e6
    );
}

/// Chunked GDN prefill (prep + state walk) for one layer at 8192 tokens.
#[test]
#[ignore]
fn gdn_chunked_tc_8k() {
    quench6k_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    const T: usize = 8192;
    const C: usize = 10240;
    const VH: usize = 48;
    const DK: usize = 128;
    let mut x = 0x1234_5678u32;
    let mut rnd = || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        (x as f32 / u32::MAX as f32) - 0.5
    };
    let qkv: Vec<f32> = (0..T * C).map(|_| rnd() * 0.18).collect();
    let beta: Vec<f32> = (0..T * VH).map(|_| 0.5 + rnd() * 0.9).collect();
    let g: Vec<f32> = (0..T * VH).map(|_| -0.45 + rnd() * 0.5).collect();
    let up = |v: &Vec<f32>| -> DeviceBuffer {
        let b = DeviceBuffer::new(v.len() * 4).unwrap();
        s.upload(&b, v.as_slice()).unwrap();
        b
    };
    let (dq, db, dg) = (up(&qkv), up(&beta), up(&g));
    let st = DeviceBuffer::zeroed(VH * DK * DK * 4).unwrap();
    let out = DeviceBuffer::new(T * VH * DK * 4).unwrap();
    let scratch = DeviceBuffer::new(kernels::gdn_tc_scratch_floats(T) * 4).unwrap();
    let run = || {
        kernels::gdn_chunked_tc(
            &s,
            dq.addr(),
            db.addr(),
            dg.addr(),
            st.addr(),
            out.addr(),
            T,
            scratch.addr(),
        )
        .unwrap()
    };
    run();
    s.synchronize().unwrap();
    let (a, b) = (Event::timing().unwrap(), Event::timing().unwrap());
    let mut v: Vec<f64> = (0..7)
        .map(|_| {
            a.record(&s).unwrap();
            for _ in 0..10 {
                run();
            }
            b.record(&s).unwrap();
            b.synchronize().unwrap();
            Event::elapsed_ms(&a, &b).unwrap() as f64 * 100.0
        })
        .collect();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    s.memset(st.addr(), 0, VH * DK * DK * 4).unwrap();
    run();
    let o: Vec<f32> = s.download(out.addr(), T * VH * DK).unwrap();
    let st_out: Vec<f32> = s.download(st.addr(), VH * DK * DK).unwrap();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in o.iter().chain(&st_out) {
        h = (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3);
    }
    let sum = h;
    println!(
        "{} ",
        std::env::var("QUENCH6K_VARIANTS").unwrap_or_default()
    );
    println!(
        "gdn_chunked_tc T={T}: {:.1} us per layer (bits hash {sum:016x})",
        v[3]
    );
}

/// NVFP4 activation quantization of a prefill chunk: the 8192 x 2560 expert input and the
/// 81920 x 640 down-projection input.
#[test]
#[ignore]
fn fp4_quant_rows_prefill() {
    quench6k_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    for &(rows, k) in &[(8192usize, 2560usize), (81920, 640)] {
        let mut x = 0x9e37_79b9u32;
        let vals: Vec<u16> = (0..rows * k)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                // BF16 of a heavy-tailed value: occasional outliers per row
                let u = (x as f32 / u32::MAX as f32) - 0.5;
                let v = if i % 997 == 0 {
                    u * 40.0
                } else {
                    u * u * u * 8.0
                };
                (v.to_bits() >> 16) as u16
            })
            .collect();
        let xb = DeviceBuffer::new(rows * k * 2).unwrap();
        s.upload(&xb, &vals).unwrap();
        let bytes = kernels::fp4_rows_bytes(rows, k);
        let out = DeviceBuffer::zeroed(bytes).unwrap();
        let run = || kernels::fp4_quant_rows(&s, xb.addr(), rows, k, out.addr()).unwrap();
        run();
        s.synchronize().unwrap();
        let (a, b) = (Event::timing().unwrap(), Event::timing().unwrap());
        let mut v: Vec<f64> = (0..7)
            .map(|_| {
                a.record(&s).unwrap();
                for _ in 0..20 {
                    run();
                }
                b.record(&s).unwrap();
                b.synchronize().unwrap();
                Event::elapsed_ms(&a, &b).unwrap() as f64 * 50.0
            })
            .collect();
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let o: Vec<u8> = s.download(out.addr(), bytes).unwrap();
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &c in &o {
            h = (h ^ c as u64).wrapping_mul(0x100_0000_01b3);
        }
        println!(
            "fp4_quant_rows {rows} x {k}: {:.1} us (bytes hash {h:016x})",
            v[3]
        );
    }
}

/// Device bytes from a xorshift stream; `lo..lo+span` bounds each byte (E4M3 scales stay finite).
fn random_bytes(s: &Stream, n: usize, seed: u32, lo: u8, span: u8) -> DeviceBuffer {
    let mut x = seed | 1;
    let v: Vec<u8> = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            if span == 0 {
                x as u8
            } else {
                lo + (x % span as u32) as u8
            }
        })
        .collect();
    let b = DeviceBuffer::new(n).unwrap();
    s.upload(&b, &v).unwrap();
    b
}

/// Routed-expert W4A4 GEMMs of one 8K prefill chunk (512 experts, top 10, 81920 routed rows):
/// gate/up with SwiGLU (N 640, K 2560) and down (N 2560, K 640), on a skewed synthetic routing.
#[test]
#[ignore]
fn fp4_expert_gemm_prefill() {
    quench6k_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    const E: usize = 512;
    const T: usize = 8192;
    const TOPK: usize = 10;
    const I: usize = 640;
    let pairs = T * TOPK;
    // Skewed expert popularity: weight ~ 1 / (rank + 40) over a fixed permutation.
    let mut x = 0x2545_f491u32;
    let mut rnd = || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x
    };
    let w: Vec<f64> = (0..E)
        .map(|e| 1.0 / ((e * 197 % E) as f64 + 40.0))
        .collect();
    let total: f64 = w.iter().sum();
    let mut top: Vec<i32> = Vec::with_capacity(pairs);
    for _ in 0..T {
        let mut chosen: Vec<i32> = Vec::new();
        while chosen.len() < TOPK {
            let mut u = rnd() as f64 / u32::MAX as f64 * total;
            let mut e = 0;
            while e + 1 < E && u > w[e] {
                u -= w[e];
                e += 1;
            }
            if !chosen.contains(&(e as i32)) {
                chosen.push(e as i32);
            }
        }
        top.extend(chosen);
    }
    let mut counts = vec![0usize; E];
    for &e in &top {
        counts[e as usize] += 1;
    }
    let bm = kernels::fp4_tile_rows();
    let tiles_needed: usize = counts.iter().map(|c| c.div_ceil(bm)).sum();
    let max_tiles = pairs.div_ceil(bm) + E;
    let top_i = DeviceBuffer::new(pairs * 4).unwrap();
    s.upload(&top_i, &top).unwrap();
    let ib = |n: usize| DeviceBuffer::zeroed(n * 4).unwrap();
    let (cnt, cur, off, te, tr, nt, pr) = (
        ib(E),
        ib(E),
        ib(E + 1),
        ib(max_tiles),
        ib(max_tiles),
        ib(1),
        ib(pairs),
    );
    kernels::moe_plan(
        &s,
        top_i.addr(),
        T,
        cnt.addr(),
        cur.addr(),
        off.addr(),
        te.addr(),
        tr.addr(),
        nt.addr(),
        pr.addr(),
        bm,
    )
    .unwrap();
    // Expert weights: codes [N, K/2], E4M3 block scales of 1.0 [N, K/16], scale2 per expert.
    let proj = |n: usize,
                k: usize|
     -> (
        DeviceBuffer,
        DeviceBuffer,
        DeviceBuffer,
        DeviceBuffer,
        DeviceBuffer,
    ) {
        let wq = random_bytes(&s, E * n * k / 2, (n * 31 + k) as u32, 0, 0);
        let sq = random_bytes(&s, E * n * k / 16, (n * 7 + k * 3) as u32, 0x30, 16);
        let wp: Vec<u64> = (0..E).map(|e| wq.addr() + (e * n * k / 2) as u64).collect();
        let sp: Vec<u64> = (0..E)
            .map(|e| sq.addr() + (e * n * k / 16) as u64)
            .collect();
        let (wpb, spb) = (
            DeviceBuffer::new(E * 8).unwrap(),
            DeviceBuffer::new(E * 8).unwrap(),
        );
        s.upload(&wpb, &wp).unwrap();
        s.upload(&spb, &sp).unwrap();
        (wq, sq, wpb, spb, scales(&s, E))
    };
    let gate = proj(I, D);
    let up = proj(I, D);
    let down = proj(D, I);
    let a_in = DeviceBuffer::zeroed(kernels::fp4_rows_bytes(pairs, D)).unwrap();
    let a_dn = DeviceBuffer::zeroed(kernels::fp4_rows_bytes(pairs, I)).unwrap();
    for (buf, k) in [(&a_in, D), (&a_dn, I)] {
        let (c, sc, r) = kernels::fp4_rows_layout(buf.addr(), pairs, k);
        let codes = random_bytes(&s, pairs * k / 2, k as u32 * 11, 0, 0);
        let bscale = random_bytes(&s, pairs * k / 16, k as u32 * 13, 0x30, 16);
        s.copy_d2d(c, codes.addr(), pairs * k / 2).unwrap();
        s.copy_d2d(sc, bscale.addr(), pairs * k / 16).unwrap();
        s.upload_to(r, &vec![0.01f32; pairs]).unwrap();
    }
    let act = DeviceBuffer::zeroed(pairs * I * 2).unwrap();
    let y = DeviceBuffer::zeroed(pairs * D * 2).unwrap();
    let only = std::env::var("QUENCH6K_MICRO_ONLY").unwrap_or_default();
    let gu = || {
        kernels::moe_fp4_gate_up(
            &s,
            a_in.addr(),
            pairs,
            (gate.2.addr(), gate.3.addr(), gate.4.addr()),
            (up.2.addr(), up.3.addr(), up.4.addr()),
            off.addr(),
            te.addr(),
            tr.addr(),
            nt.addr(),
            max_tiles,
            act.addr(),
            I,
            D,
        )
        .unwrap()
    };
    let dn = || {
        kernels::moe_fp4_gemm_out(
            &s,
            a_dn.addr(),
            pairs,
            down.2.addr(),
            down.3.addr(),
            down.4.addr(),
            off.addr(),
            te.addr(),
            tr.addr(),
            nt.addr(),
            max_tiles,
            y.addr(),
            D,
            I,
            true,
        )
        .unwrap()
    };
    let time = |f: &dyn Fn(), reps: usize| -> f64 {
        f();
        s.synchronize().unwrap();
        let (a, b) = (Event::timing().unwrap(), Event::timing().unwrap());
        let mut v: Vec<f64> = (0..7)
            .map(|_| {
                a.record(&s).unwrap();
                for _ in 0..reps {
                    f();
                }
                b.record(&s).unwrap();
                b.synchronize().unwrap();
                Event::elapsed_ms(&a, &b).unwrap() as f64 * 1000.0 / reps as f64
            })
            .collect();
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        v[3]
    };
    let hash = |addr: u64, n: usize| {
        let o: Vec<u16> = s.download(addr, n).unwrap();
        o.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &v| {
            (h ^ v as u64).wrapping_mul(0x100_0000_01b3)
        })
    };
    let flops = |n: usize, k: usize| 2.0 * (pairs * n * k) as f64;
    println!(
        "routing: {} tiles of {bm} rows ({:.0}% full), largest expert {} rows, smallest {}",
        tiles_needed,
        100.0 * pairs as f64 / (tiles_needed * bm) as f64,
        counts.iter().max().unwrap(),
        counts.iter().min().unwrap()
    );
    if only.is_empty() || only == "gate_up" {
        let us = time(&gu, 10);
        println!(
            "gate_up: {us:.1} us, {:.0} TFLOPS (act hash {:016x})",
            flops(2 * I, D) / us / 1e6,
            hash(act.addr(), pairs * I)
        );
    }
    if only.is_empty() || only == "down" {
        let us = time(&dn, 10);
        println!(
            "down: {us:.1} us, {:.0} TFLOPS (y hash {:016x})",
            flops(D, I) / us / 1e6,
            hash(y.addr(), pairs * D)
        );
    }
}
