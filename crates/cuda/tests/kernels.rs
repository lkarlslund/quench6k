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

fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f32;
    if e == 0 { sign * m / 8.0 * 2f32.powi(-6) } else { sign * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
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
fn gemv_and_fp8_paths_match_cpu() {
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

        // FP8 quantization, then GEMV and tensor-core GEMM on the FP8 weights
        let dq = DeviceBuffer::new(n * k).unwrap();
        let dsc = DeviceBuffer::new(n * 4).unwrap();
        kernels::quantize_fp8_rows(&s, dw.addr(), n, k, dq.addr(), dsc.addr()).unwrap();
        let q: Vec<u8> = s.download(dq.addr(), n * k).unwrap();
        let sc: Vec<f32> = s.download(dsc.addr(), n).unwrap();
        let wq: Vec<f32> = (0..n * k).map(|i| fp8_e4m3_to_f32(q[i]) * sc[i / k]).collect();
        let quant_err = w.iter().zip(&wq).map(|(a, b)| (a - b).abs() / sc[0].max(1e-9)).fold(0f32, f32::max);
        assert!(quant_err.is_finite());
        let want = cpu_matmul(&x, &wq, m, n, k);
        kernels::gemv(&s, m, dx.addr(), k, WeightFormat::Fp8, dq.addr(), dsc.addr(), dy.addr(), n, partial.addr(), n, k, splits, 1.0, false).unwrap();
        let got: Vec<f32> = s.download(dy.addr(), m * n).unwrap();
        let e = max_rel(&got, &want, k);
        assert!(e < 1e-4, "fp8 gemv {m}x{n}x{k}/{splits}: {e}");
        kernels::gemm_f32(&s, dx.addr(), k, WeightFormat::Fp8, dq.addr(), k, dsc.addr(), dy.addr(), n, m, n, k, 1.0, false).unwrap();
        let got: Vec<f32> = s.download(dy.addr(), m * n).unwrap();
        // The tensor-core path rounds dequantized weights to BF16 (exact for E4M3 values) before the MMA.
        let e = max_rel(&got, &want, k);
        assert!(e < 1e-4, "fp8 gemm {m}x{n}x{k}: {e}");

        // INT8 quantization through both paths
        kernels::quantize_int8_rows(&s, dw.addr(), n, k, dq.addr(), dsc.addr()).unwrap();
        let q: Vec<u8> = s.download(dq.addr(), n * k).unwrap();
        let sc: Vec<f32> = s.download(dsc.addr(), n).unwrap();
        let wq: Vec<f32> = (0..n * k).map(|i| q[i] as i8 as f32 * sc[i / k]).collect();
        let want = cpu_matmul(&x, &wq, m, n, k);
        kernels::gemv(&s, m, dx.addr(), k, WeightFormat::Int8, dq.addr(), dsc.addr(), dy.addr(), n, partial.addr(), n, k, splits, 1.0, false).unwrap();
        let got: Vec<f32> = s.download(dy.addr(), m * n).unwrap();
        let e = max_rel(&got, &want, k);
        assert!(e < 1e-4, "int8 gemv {m}x{n}x{k}/{splits}: {e}");
        kernels::gemm_f32(&s, dx.addr(), k, WeightFormat::Int8, dq.addr(), k, dsc.addr(), dy.addr(), n, m, n, k, 1.0, false).unwrap();
        let got: Vec<f32> = s.download(dy.addr(), m * n).unwrap();
        let e = max_rel(&got, &want, k);
        assert!(e < 1e-4, "int8 gemm {m}x{n}x{k}: {e}");

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
