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
