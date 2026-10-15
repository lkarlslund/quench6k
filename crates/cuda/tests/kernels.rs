//! Kernel tests against straightforward CPU implementations.

use squash_cuda::kernels::WeightFormat;
use squash_cuda::{DeviceBuffer, Stream, bf16, kernels};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn to_bf16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}

fn upload<T: Copy>(s: &Stream, v: &[T]) -> DeviceBuffer {
    let b = DeviceBuffer::new(std::mem::size_of_val(v)).unwrap();
    s.upload(&b, v).unwrap();
    b
}

#[test]
fn gemm_matches_cpu() {
    squash_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for &(m, n, k) in &[(1, 320, 10240), (7, 4, 2560), (130, 300, 64), (64, 128, 32), (200, 1000, 2560)] {
        let x = rng.vec(m * k).iter().map(|&v| bf16::round(v)).collect::<Vec<_>>();
        let w = rng.vec(n * k).iter().map(|&v| bf16::round(v)).collect::<Vec<_>>();
        let (dx, dw) = (upload(&s, &to_bf16(&x)), upload(&s, &to_bf16(&w)));
        let y0 = rng.vec(m * n);
        let dy = upload(&s, &y0);
        kernels::gemm_f32(&s, dx.addr(), k, WeightFormat::Bf16, dw.addr(), k, 0, dy.addr(), n, m, n, k, 0.5, true).unwrap();
        let y: Vec<f32> = s.download(dy.addr(), m * n).unwrap();
        let mut worst = 0f64;
        for i in 0..m {
            for j in 0..n {
                let dot: f64 = (0..k).map(|t| x[i * k + t] as f64 * w[j * k + t] as f64).sum();
                let want = y0[i * n + j] as f64 + 0.5 * dot;
                let scale = (k as f64).sqrt() * 0.5;
                worst = worst.max((y[i * n + j] as f64 - want).abs() / scale);
            }
        }
        assert!(worst < 1e-4, "gemm {m}x{n}x{k}: relative error {worst}");
    }
}

fn cpu_matmul(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut y = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            y[i * n + j] = (0..k).map(|t| x[i * k + t] as f64 * w[j * k + t] as f64).sum();
        }
    }
    y
}

fn max_rel(got: &[f32], want: &[f64], k: usize) -> f64 {
    got.iter().zip(want).map(|(g, w)| (*g as f64 - w).abs() / (k as f64).sqrt()).fold(0.0, f64::max)
}

#[test]
fn gemv_and_int8_group_paths_match_cpu() {
    squash_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for &(m, n, k, splits) in &[(1, 320, 10240, 16), (3, 10240, 320, 1), (16, 2560, 6144, 3), (2, 33, 2560, 1), (5, 48, 2560, 4)] {
        let x: Vec<f32> = rng.vec(m * k).iter().map(|&v| bf16::round(v)).collect();
        let w: Vec<f32> = rng.vec(n * k).iter().map(|&v| bf16::round(v)).collect();
        let (dx, dw) = (upload(&s, &to_bf16(&x)), upload(&s, &to_bf16(&w)));
        let dy = DeviceBuffer::new(m * n * 4).unwrap();
        let partial = DeviceBuffer::new(splits * m * n * 4).unwrap();

        // BF16 GEMV
        kernels::gemv(&s, m, dx.addr(), k, WeightFormat::Bf16, dw.addr(), 0, dy.addr(), n, partial.addr(), n, k, splits, 1.0, false).unwrap();
        let got: Vec<f32> = s.download(dy.addr(), m * n).unwrap();
        let e = max_rel(&got, &cpu_matmul(&x, &w, m, n, k), k);
        assert!(e < 1e-4, "bf16 gemv {m}x{n}x{k}/{splits}: {e}");

        let dq = DeviceBuffer::new(n * k).unwrap();
        // Group-scaled INT8 through both paths
        if k % kernels::int8_group() == 0 {
            let g = kernels::int8_group();
            let dgs = DeviceBuffer::new(n * (k / g) * 4).unwrap();
            kernels::quantize_int8_groups(&s, dw.addr(), n, k, dq.addr(), dgs.addr()).unwrap();
            let q: Vec<u8> = s.download(dq.addr(), n * k).unwrap();
            let gs: Vec<f32> = s.download(dgs.addr(), n * (k / g)).unwrap();
            let wq: Vec<f32> = (0..n * k).map(|i| q[i] as i8 as f32 * gs[(i / k) * (k / g) + (i % k) / g]).collect();
            let want = cpu_matmul(&x, &wq, m, n, k);
            kernels::gemv(&s, m, dx.addr(), k, WeightFormat::Int8Group, dq.addr(), dgs.addr(), dy.addr(), n, partial.addr(), n, k, splits, 1.0, false).unwrap();
            let got: Vec<f32> = s.download(dy.addr(), m * n).unwrap();
            let e = max_rel(&got, &want, k);
            assert!(e < 1e-4, "int8g gemv {m}x{n}x{k}/{splits}: {e}");
            kernels::gemm_f32(&s, dx.addr(), k, WeightFormat::Int8Group, dq.addr(), k, dgs.addr(), dy.addr(), n, m, n, k, 1.0, false).unwrap();
            let got: Vec<f32> = s.download(dy.addr(), m * n).unwrap();
            // The tensor-core path rounds q * scale to BF16 while staging.
            let e = max_rel(&got, &want, k);
            assert!(e < 3e-3, "int8g gemm {m}x{n}x{k}: {e}");
        }
    }
}

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn fp8_e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f32;
    if e == 0 { sign * m / 8.0 * 2f32.powi(-6) } else { sign * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

#[test]
fn nvfp4_quantization_round_trips() {
    squash_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    let mut rng = Rng(0x0dd_ba11);
    let (count, n, k) = (3, 64, 256);
    let w: Vec<f32> = rng.vec(count * n * k).iter().enumerate().map(|(i, &v)| bf16::round(v * if i % 97 == 0 { 8.0 } else { 1.0 })).collect();
    let dw = upload(&s, &to_bf16(&w));
    let codes = DeviceBuffer::new(count * n * k / 2).unwrap();
    let scales = DeviceBuffer::new(count * n * k / 16).unwrap();
    let scale2 = DeviceBuffer::new(count * 4).unwrap();
    let amax = DeviceBuffer::new(count * 4).unwrap();
    kernels::quantize_nvfp4(&s, dw.addr(), count, n, k, codes.addr(), scales.addr(), scale2.addr(), amax.addr()).unwrap();
    let c: Vec<u8> = s.download(codes.addr(), count * n * k / 2).unwrap();
    let sc: Vec<u8> = s.download(scales.addr(), count * n * k / 16).unwrap();
    let s2: Vec<f32> = s.download(scale2.addr(), count).unwrap();
    for m in 0..count {
        let per = n * k;
        let amax = w[m * per..(m + 1) * per].iter().fold(0f32, |a, &v| a.max(v.abs()));
        assert!((s2[m] - amax / (6.0 * 448.0)).abs() <= 1e-6 * s2[m], "scale2 {m}");
        for b in 0..per / 16 {
            let bs = fp8_e4m3(sc[m * per / 16 + b]) * s2[m];
            for i in 0..16 {
                let idx = m * per + b * 16 + i;
                let byte = c[idx / 2];
                let code = if idx % 2 == 0 { byte & 15 } else { byte >> 4 };
                let q = E2M1[(code & 7) as usize] * if code & 8 != 0 { -1.0 } else { 1.0 } * bs;
                // Nearest grid point: error at most half the largest grid step (1.0) times the scale.
                assert!((q - w[idx]).abs() <= 1.0 * bs + 1e-6, "value {idx}: {} vs {}", q, w[idx]);
            }
        }
    }
}

fn nvfp4_value(code: u8) -> f32 {
    E2M1[(code & 7) as usize] * if code & 8 != 0 { -1.0 } else { 1.0 }
}

/// Small-batch expert gate/up (+SwiGLU) against the CPU, two tokens routed to overlapping experts.
#[test]
fn expert_gate_up_small_matches_cpu() {
    squash_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    let mut rng = Rng(0xfeed_beef);
    const E: usize = 512;
    let (n, k) = (640usize, 2560usize);
    for t in [2usize, 4, 8] {
    let mut rng = Rng(0xfeed_beef + t as u64);
    // Two experts with random NVFP4 weights for gate and up.
    let mut make = |rng: &mut Rng| -> (Vec<u8>, Vec<u8>, f32) {
        let codes: Vec<u8> = (0..n * k / 2).map(|_| ((rng.next() * 127.0 + 128.0) as u32 & 0xff) as u8).collect();
        let scales: Vec<u8> = (0..n * k / 16).map(|_| 0x30 + ((rng.next().abs() * 8.0) as u8)).collect();
        (codes, scales, 0.01)
    };
    let experts: Vec<[(Vec<u8>, Vec<u8>, f32); 2]> = (0..2).map(|_| [make(&mut rng), make(&mut rng)]).collect();
    let dev: Vec<[(DeviceBuffer, DeviceBuffer); 2]> = experts.iter().map(|e| [(upload(&s, &e[0].0), upload(&s, &e[0].1)), (upload(&s, &e[1].0), upload(&s, &e[1].1))]).collect();
    let ids = [7usize, 300];
    let mut tables = vec![0u64; 4 * E];
    let mut s2 = vec![0f32; 2 * E];
    for (i, &e) in ids.iter().enumerate() {
        for j in 0..2 {
            tables[2 * j * E + e] = dev[i][j].0.addr();
            tables[(2 * j + 1) * E + e] = dev[i][j].1.addr();
            s2[j * E + e] = experts[i][j].2;
        }
    }
    let dt = upload(&s, &tables);
    let ds2 = upload(&s, &s2);
    // Token 0 -> experts 7 and 300, token 1 -> expert 300: rows grouped by expert.
    let x: Vec<f32> = rng.vec(t * k).iter().map(|&v| bf16::round(v)).collect();
    let dx = upload(&s, &to_bf16(&x));
    let mut offsets = vec![3i32; E + 1];
    for (e, o) in offsets.iter_mut().enumerate() {
        *o = if e <= 7 { 0 } else if e <= 300 { 1 } else { 3 };
    }
    let row_token = [0i32, 0, (t - 1) as i32];
    let active = [7i32, 300];
    let (d_off, d_rt, d_act, d_na) = (upload(&s, &offsets), upload(&s, &row_token), upload(&s, &active), upload(&s, &[2i32]));
    let out = DeviceBuffer::new(3 * n * 2).unwrap();
    let a = |j: usize| (dt.addr() + (2 * j * E * 8) as u64, dt.addr() + ((2 * j + 1) * E * 8) as u64, ds2.addr() + (j * E * 4) as u64);
    kernels::moe_gate_up_small(&s, dx.addr(), d_rt.addr(), a(0), a(1), d_off.addr(), d_act.addr(), d_na.addr(), 2, t, out.addr()).unwrap();
    let got: Vec<u16> = s.download(out.addr(), 3 * n).unwrap();
    let row_expert = [0usize, 1, 1];
    for r in 0..3 {
        let xi = &x[row_token[r] as usize * k..][..k];
        let ex = &experts[row_expert[r]];
        for f in [0usize, 1, 2, 3, 100, 333, 639] {
            let dot = |w: &(Vec<u8>, Vec<u8>, f32)| -> f64 {
                (0..k).map(|c| {
                    let byte = w.0[f * k / 2 + c / 2];
                    let code = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
                    let sc = fp8_e4m3(w.1[f * k / 16 + c / 16]);
                    (nvfp4_value(code) * sc) as f64 * xi[c] as f64
                }).sum::<f64>() * w.2 as f64
            };
            let (g, u) = (dot(&ex[0]), dot(&ex[1]));
            let want = g / (1.0 + (-g).exp()) * u;
            let have = bf16::to_f32(got[r * n + f]) as f64;
            assert!((have - want).abs() <= 1e-2 * want.abs().max(1e-3) + 1e-4, "t {t} row {r} feature {f}: {have} vs {want}");
        }
    }
    }
}

/// The small-batch HyperConnection gives every token the same bits whatever the batch size.
#[test]
fn hc_small_is_batch_invariant() {
    squash_cuda::set_device(0).unwrap();
    let s = Stream::new().unwrap();
    let mut rng = Rng(0x5eed_0f_4c);
    const D: usize = 2560;
    const H: usize = 4 * D;
    const RANK: usize = 320;
    let gr = kernels::int8_group();
    let bytes = |n: usize, rng: &mut Rng| -> Vec<i8> { (0..n).map(|_| (rng.next() * 100.0) as i8).collect() };
    let pos = |n: usize, rng: &mut Rng| -> Vec<f32> { (0..n).map(|_| 0.002 + 0.002 * rng.next().abs()).collect() };
    let (wd, sd) = (upload(&s, &bytes(RANK * H, &mut rng)), upload(&s, &pos(RANK * H / gr, &mut rng)));
    let (wi, si) = (upload(&s, &bytes(4 * H, &mut rng)), upload(&s, &pos(4 * H / gr, &mut rng)));
    let (wu, su) = (upload(&s, &bytes(H * RANK, &mut rng)), upload(&s, &pos(H * RANK / gr, &mut rng)));
    let g = upload(&s, &to_bf16(&rng.vec(4 * D).iter().map(|v| v * 0.1).collect::<Vec<_>>()));
    let fmt = WeightFormat::Int8Group;
    for (with_y, with_inj) in [(true, true), (false, true), (true, false), (false, false)] {
    let run = |t: usize, h0: &[f32], y0: &[f32], ip: &[f32]| -> (Vec<f32>, Vec<u16>, Vec<f32>) {
        let h = upload(&s, h0);
        let y = upload(&s, y0);
        let inj_prev = upload(&s, ip);
        let scratch = DeviceBuffer::zeroed(kernels::hc_small_scratch_floats(t) * 4).unwrap();
        let z = DeviceBuffer::new(t * RANK * 2).unwrap();
        let r = DeviceBuffer::new(t * 16).unwrap();
        let inj = DeviceBuffer::new(t * 16).unwrap();
        let m = DeviceBuffer::new(t * D * 2).unwrap();
        s.memset(inj.addr(), 0, t * 16).unwrap();
        let y = if with_y { y.addr() } else { 0 };
        let inject = with_inj.then(|| (fmt, wi.addr(), si.addr()));
        kernels::hc_small(&s, h.addr(), t, y, inj_prev.addr(), (fmt, wd.addr(), sd.addr()), inject,
            (fmt, wu.addr(), su.addr()), g.addr(), scratch.addr(), z.addr(), r.addr(), inj.addr(), m.addr()).unwrap();
        (s.download(h.addr(), t * H).unwrap(), s.download(m.addr(), t * D).unwrap(), s.download(inj.addr(), t * 4).unwrap())
    };
    for &t in &[2usize, 4, 5, 8, 16] {
        let h0 = rng.vec(t * H);
        let y0 = rng.vec(t * D);
        let ip: Vec<f32> = (0..t * 4).map(|_| 1.0 + rng.next() * 0.5).collect();
        let (hb, mb, ib) = run(t, &h0, &y0, &ip);
        for i in 0..t {
            let (h1, m1, i1) = run(1, &h0[i * H..(i + 1) * H], &y0[i * D..(i + 1) * D], &ip[i * 4..(i + 1) * 4]);
            assert!(h1.iter().zip(&hb[i * H..]).all(|(a, b)| a.to_bits() == b.to_bits()), "T={t} token {i}: residual differs");
            assert!(i1.iter().zip(&ib[i * 4..]).all(|(a, b)| a.to_bits() == b.to_bits()), "T={t} token {i}: injection differs");
            let bad = m1.iter().zip(&mb[i * D..]).filter(|(a, b)| a != b).count();
            assert!(bad == 0, "T={t} token {i} (y {with_y}, inj {with_inj}): {bad} block-input values differ");
        }
    }
    }
}
