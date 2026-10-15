//! Kernel latency microbenchmarks (ignored by default): `cargo test --release -p squash-cuda --test micro -- --ignored --nocapture`.
//!
//! Each case captures a graph of many dependent launches over distinct weights (more than L2 holds),
//! so per-call times include weight streaming the way a decode step sees it.

use squash_cuda::kernels::{self, WeightFormat};
use squash_cuda::{DeviceBuffer, Event, Graph, Stream};

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
    Gemv { w: filled(s, n * D, 3), s: scales(s, n * D / kernels::int8_group()), n }
}

#[test]
#[ignore]
fn hc_latency() {
    squash_cuda::set_device(0).unwrap();
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
        kernels::gemv(s, t, x, D, fmt, w.w.addr(), w.s.addr(), out.addr(), w.n, partial.addr(), w.n, D, 1, 1.0, false)
    };
    for t in [1usize, 4] {
        let g = Graph::capture(&s, |s| hcs.iter().try_for_each(|w| hc(s, w, t))).unwrap();
        let hc_only = time_graph(&s, &g, CALLS);
        let g = Graph::capture(&s, |s| gemvs.iter().try_for_each(|w| gemv(s, w, t, m.addr()))).unwrap();
        let big_only = time_graph(&s, &g, CALLS);
        let g = Graph::capture(&s, |s| small.iter().try_for_each(|w| gemv(s, w, t, m.addr()))).unwrap();
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
    squash_cuda::set_device(0).unwrap();
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
        s.upload(&b, &v.iter().map(|x| x.addr()).collect::<Vec<u64>>()).unwrap();
        b
    };
    let ones = {
        let b = DeviceBuffer::new(E * 4).unwrap();
        s.upload(&b, &vec![1.0f32; E]).unwrap();
        b
    };
    let (gc, gs, uc, us, dc, ds) = (table(&gate_codes), table(&gate_scales), table(&up_codes), table(&up_scales), table(&down_codes), table(&down_scales));
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
            s.upload(&b, &(0..t_max * E).map(|_| next() * 8.0).collect::<Vec<f32>>()).unwrap();
            b
        })
        .collect();
    let m = filled(&s, t_max * D * 2, 0);
    let pairs = t_max * TOPK;
    let (top_i, top_w, pair_row, row_token) = (filled(&s, pairs * 4, 0), filled(&s, pairs * 4, 0), filled(&s, pairs * 4, 0), filled(&s, pairs * 4, 0));
    let (offsets, active, num_active) = (filled(&s, (E + 1) * 4, 0), filled(&s, E * 4, 0), filled(&s, 4, 0));
    let act = filled(&s, pairs * W * 2, 0);
    let yd = filled(&s, pairs * D * 4, 0);
    for t in [1usize, 2, 4, 5] {
        let max_active = (t * TOPK).min(E);
        let layer = |s: &Stream, l: usize| -> anyhow::Result<()> {
            kernels::moe_route_small(s, logits[l].addr(), E, t, top_i.addr(), top_w.addr(), offsets.addr(), pair_row.addr(), row_token.addr(), active.addr(), num_active.addr())?;
            kernels::moe_gate_up_small(s, m.addr(), row_token.addr(), (gc.addr(), gs.addr(), ones.addr()), (uc.addr(), us.addr(), ones.addr()), offsets.addr(), active.addr(), num_active.addr(), max_active, t, act.addr())?;
            kernels::moe_down_small(s, act.addr(), (dc.addr(), ds.addr(), ones.addr()), offsets.addr(), active.addr(), num_active.addr(), max_active, t, yd.addr())
        };
        let mut distinct = 0usize;
        for l in 0..LAYERS {
            layer(&s, l).unwrap();
            distinct += s.download::<i32>(num_active.addr(), 1).unwrap()[0] as usize;
        }
        let g = Graph::capture(&s, |s| (0..LAYERS).try_for_each(|l| layer(s, l))).unwrap();
        let us_layer = time_graph(&s, &g, LAYERS);
        let bytes = distinct as f64 / LAYERS as f64 * (3 * W * D) as f64 * (0.5 + 1.0 / 16.0);
        println!("T={t}: {us_layer:.2} us per layer, {:.1} experts, {:.2} TB/s", distinct as f64 / LAYERS as f64, bytes / us_layer / 1e6);
    }
}

/// INT8G GEMV chains (96 distinct weights each) for the decode projection shapes; splits > 1 adds
/// the partial-sum reduction kernel.
#[test]
#[ignore]
fn gemv_latency() {
    squash_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    let fmt = WeightFormat::Int8Group;
    let x = filled(&s, 16 * 16480 * 2, 0);
    let out = filled(&s, 16 * 16480 * 4, 0);
    let partial = filled(&s, 1 << 24, 0);
    for &(n, k, splits) in &[(2560usize, 6144usize, 1usize), (2560, 6144, 2), (2560, 640, 1), (16480, 2560, 1), (13952, 2560, 1), (3600, 2560, 1)] {
        let ws: Vec<(DeviceBuffer, DeviceBuffer)> =
            (0..CALLS).map(|_| (filled(&s, n * k, 3), scales(&s, n * k / kernels::int8_group()))).collect();
        let mut line = format!("N={n} K={k} splits={splits}:");
        for t in [1usize, 4] {
            let g = Graph::capture(&s, |s| {
                ws.iter().try_for_each(|(w, sc)| kernels::gemv(s, t, x.addr(), k, fmt, w.addr(), sc.addr(), out.addr(), n, partial.addr(), n, k, splits, 1.0, false))
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
    squash_cuda::set_device(0).unwrap();
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
            let v: Vec<f32> = (0..t * ld).map(|_| if coarse { (next() * 64.0).round() / 4.0 } else { next() * 3.0 }).collect();
            let scores = DeviceBuffer::new(v.len() * 4).unwrap();
            s.upload(&scores, &v).unwrap();
            let start = DeviceBuffer::new(4).unwrap();
            s.upload(&start, &[(ctx - t) as i32]).unwrap();
            let sel = DeviceBuffer::new(t * 512 * 4).unwrap();
            let g = Graph::capture(&s, |s| (0..CALLS).try_for_each(|_| kernels::qsa_select(s, scores.addr(), t, ld, start.addr(), sel.addr()))).unwrap();
            println!("ctx {ctx} coarse {coarse}: {:.2} us", time_graph(&s, &g, CALLS));
        }
    }
}
