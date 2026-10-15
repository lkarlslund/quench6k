//! Kernel tests against straightforward CPU implementations.

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
        kernels::gemm_f32(&s, dx.addr(), k, dw.addr(), k, dy.addr(), n, m, n, k, 0.5, true).unwrap();
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
