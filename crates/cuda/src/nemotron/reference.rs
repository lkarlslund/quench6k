//! CPU references for the Nemotron-H kernels (tests only; slow).
//!
//! `*_emu` functions mirror the kernels' FP32 arithmetic and order (FMA placement, warp reduction trees,
//! BF16 rounding points), so a GPU result differs from them only where CUDA's expf / log1pf / exp2f
//! differ from the host's by an ulp. `*_f64` functions compute the same model semantics (the same BF16
//! rounding points of the checkpoint's model) in FP64 to measure accuracy.

use super::*;
use crate::bf16;

/// Deterministic xorshift generator for test data.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [lo, hi).
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }
    pub fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }
}

pub fn rb(x: f32) -> f32 {
    bf16::round(x)
}

/// The warp_sum / warp_sum_slots tree: v[l] += v[l ^ off] for off = 16, 8, 4, 2, 1; lane 0's value.
pub fn warp_tree(v: &[f32; 32]) -> f32 {
    let mut a = *v;
    for off in [16, 8, 4, 2, 1] {
        let b = a;
        for l in 0..32 {
            a[l] = b[l] + b[l ^ off];
        }
    }
    a[0]
}

/// common.cuh block_sum over `vals` (one per thread, blockDim = vals.len(), a multiple of 32).
pub fn block_sum(vals: &[f32]) -> f32 {
    let warps = vals.len() / 32;
    let mut ws = [0.0f32; 32];
    for w in 0..warps {
        let mut v = [0.0f32; 32];
        v.copy_from_slice(&vals[w * 32..w * 32 + 32]);
        ws[w] = warp_tree(&v);
    }
    warp_tree(&ws)
}

pub fn softplus_f32(v: f32) -> f32 {
    if v <= 20.0 { v.exp().ln_1p() } else { v }
}

/// nemotron::rms as the kernel computes it (FP32 output; round with [`rb`] for the BF16 variant).
pub fn rms_emu(x: &[f32], w_bf16: &[f32], eps: f32) -> Vec<f32> {
    let per = HIDDEN / 256;
    let partial: Vec<f32> = (0..256)
        .map(|t| {
            let mut ss = 0.0f32;
            for k in 0..per {
                let v = x[k * 256 + t];
                ss = v.mul_add(v, ss);
            }
            ss
        })
        .collect();
    let ss = block_sum(&partial);
    let rs = 1.0 / (ss / HIDDEN as f32 + eps).sqrt();
    (0..HIDDEN).map(|c| w_bf16[c] * (x[c] * rs)).collect()
}

// ---- Mamba-2 ---------------------------------------------------------------------------------------

/// One layer's Mamba parameters (FP32 host copies of the device vectors).
#[derive(Clone)]
pub struct MambaParams {
    pub conv_w: Vec<f32>,
    pub conv_b: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub a_log: Vec<f32>,
    pub d: Vec<f32>,
    pub norm_w: Vec<f32>,
    pub dt_min: f32,
    pub dt_max: f32,
}

impl MambaParams {
    /// Random parameters of realistic magnitudes (dt ~ 0.002 .. 0.15, A in -1 .. -20), BF16-exact.
    pub fn random(rng: &mut Rng) -> Self {
        let r = |v: Vec<f32>| v.into_iter().map(rb).collect::<Vec<_>>();
        Self {
            conv_w: r(rng.vec(CONV_DIM * CONV_TAPS, -0.5, 0.5)),
            conv_b: r(rng.vec(CONV_DIM, -0.2, 0.2)),
            dt_bias: r(rng.vec(HEADS, -6.0, -2.0)),
            a_log: r(rng.vec(HEADS, 0.0, 3.0)),
            d: r(rng.vec(HEADS, 0.5, 1.5)),
            norm_w: r(rng.vec(INNER, 0.5, 1.5)),
            dt_min: 0.0,
            dt_max: f32::INFINITY,
        }
    }
}

/// One record row (as the kernels store it).
#[derive(Clone)]
pub struct RecRow {
    pub dt: Vec<f32>,
    pub raw: Vec<f32>,
    pub conv: Vec<f32>,
}

/// A sequence's Mamba layer state as the kernels keep it: SSM state, conv state, two record parities and
/// the header (replay, parity).
#[derive(Clone)]
pub struct MambaSeq {
    pub ssm: Vec<f32>,
    pub conv: Vec<f32>,
    pub rec: [Vec<RecRow>; 2],
    pub replay: usize,
    pub parity: usize,
}

impl Default for MambaSeq {
    fn default() -> Self {
        Self { ssm: vec![0.0; HEADS * HEAD_DIM * STATE], conv: vec![0.0; 3 * CONV_DIM], rec: [Vec::new(), Vec::new()], replay: 0, parity: 0 }
    }
}

/// Outputs of one step of a sequence: convolved xBC (BF16 values), dt, y (BF16 values), gated norm out.
pub struct MambaOut {
    pub xbc: Vec<f32>,
    pub dt: Vec<f32>,
    pub y: Vec<f32>,
    pub out: Vec<f32>,
}

fn raw_xbc(proj: &[f32], r: usize, c: usize) -> f32 {
    rb(proj[r * PROJ + XBC_COL + c])
}

impl MambaSeq {
    fn window(&self, k: usize, c: usize) -> f32 {
        let j = self.replay + k;
        if j >= 3 { self.rec[self.parity][j - 3].raw[c] } else { self.conv[j * CONV_DIM + c] }
    }

    /// One step of `rows` rows (proj [rows, PROJ]) exactly as mamba_conv, mamba_conv_commit, mamba_scan and
    /// mamba_gated_norm compute it (eps 1e-5).
    pub fn step_emu(&mut self, w: &MambaParams, proj: &[f32], rows: usize, defer: bool) -> MambaOut {
        // Conv + dt.
        let mut xbc = vec![0.0f32; rows * CONV_DIM];
        let mut dt = vec![0.0f32; rows * HEADS];
        for c in 0..CONV_DIM {
            let wt = &w.conv_w[c * 4..c * 4 + 4];
            for i in 0..rows {
                let hist = |j: usize| -> f32 {
                    // input at row i - 3 + j
                    let idx = i as isize - 3 + j as isize;
                    if idx >= 0 { raw_xbc(proj, idx as usize, c) } else { self.window((3 + idx) as usize, c) }
                };
                let mut acc = w.conv_b[c];
                acc = wt[0].mul_add(hist(0), acc);
                acc = wt[1].mul_add(hist(1), acc);
                acc = wt[2].mul_add(hist(2), acc);
                acc = wt[3].mul_add(raw_xbc(proj, i, c), acc);
                xbc[i * CONV_DIM + c] = rb(acc / (1.0 + (-acc).exp()));
            }
        }
        for i in 0..rows {
            for h in 0..HEADS {
                let v = rb(proj[i * PROJ + DT_COL + h]) + w.dt_bias[h];
                dt[i * HEADS + h] = softplus_f32(v).max(w.dt_min).min(w.dt_max);
            }
        }
        // Commit of the conv state (replay first).
        let eff: Vec<f32> = (0..3).flat_map(|k| (0..CONV_DIM).map(move |c| (k, c))).map(|(k, c)| self.window(k, c)).collect();
        let replay_rows: Vec<RecRow> = self.rec[self.parity][..self.replay].to_vec();
        if defer {
            self.conv.copy_from_slice(&eff);
            let next = self.parity ^ 1;
            self.rec[next] = (0..rows)
                .map(|i| RecRow {
                    dt: dt[i * HEADS..(i + 1) * HEADS].to_vec(),
                    raw: (0..CONV_DIM).map(|c| raw_xbc(proj, i, c)).collect(),
                    conv: xbc[i * CONV_DIM..(i + 1) * CONV_DIM].to_vec(),
                })
                .collect();
        } else {
            for k in 0..3 {
                for c in 0..CONV_DIM {
                    let j = rows + k;
                    self.conv[k * CONV_DIM + c] = if j < 3 { eff[j * CONV_DIM + c] } else { raw_xbc(proj, j - 3, c) };
                }
            }
        }
        // Scan.
        let mut y = vec![0.0f32; rows * INNER];
        let tokens: Vec<(&[f32], &[f32])> = replay_rows
            .iter()
            .map(|r| (&r.conv[..], &r.dt[..]))
            .chain((0..rows).map(|i| (&xbc[i * CONV_DIM..(i + 1) * CONV_DIM], &dt[i * HEADS..(i + 1) * HEADS])))
            .collect();
        let replay = self.replay;
        let mut committed = self.ssm.clone();
        for h in 0..HEADS {
            let g = h / (HEADS / GROUPS);
            let a = -w.a_log[h].exp();
            for p in 0..HEAD_DIM {
                let s = &mut self.ssm[(h * HEAD_DIM + p) * STATE..(h * HEAD_DIM + p + 1) * STATE];
                for (i, (conv, d)) in tokens.iter().enumerate() {
                    let d = d[h];
                    let da = (d * a).exp();
                    let x = conv[h * HEAD_DIM + p];
                    for n in 0..STATE {
                        let db = conv[INNER + g * STATE + n] * d;
                        s[n] = s[n].mul_add(da, db * x);
                    }
                    if i + 1 == replay {
                        committed[(h * HEAD_DIM + p) * STATE..(h * HEAD_DIM + p + 1) * STATE].copy_from_slice(s);
                    }
                    if i >= replay {
                        let mut lanes = [0.0f32; 32];
                        for (l, lane) in lanes.iter_mut().enumerate() {
                            let c = &conv[INNER + GROUPS * STATE + g * STATE + 4 * l..];
                            let mut acc = s[4 * l] * c[0];
                            acc = s[4 * l + 1].mul_add(c[1], acc);
                            acc = s[4 * l + 2].mul_add(c[2], acc);
                            *lane = s[4 * l + 3].mul_add(c[3], acc);
                        }
                        let v = warp_tree(&lanes);
                        y[(i - replay) * INNER + h * HEAD_DIM + p] = rb(v + x * w.d[h]);
                    }
                }
            }
        }
        if defer {
            self.ssm = committed;
        }
        // Gated norm.
        let mut out = vec![0.0f32; rows * INNER];
        for i in 0..rows {
            for g in 0..GROUPS {
                let mut vals = vec![0.0f32; NORM_GROUP];
                let mut partial = [0.0f32; 256];
                for (t, part) in partial.iter_mut().enumerate() {
                    let mut ss = 0.0f32;
                    for k in 0..NORM_GROUP / 256 {
                        let c = g * NORM_GROUP + k * 256 + t;
                        let z = rb(proj[i * PROJ + c]);
                        let gate = z * (1.0 / (1.0 + (-z).exp()));
                        let v = y[i * INNER + c] * gate;
                        vals[k * 256 + t] = v;
                        ss = v.mul_add(v, ss);
                    }
                    *part = ss;
                }
                let ss = block_sum(&partial);
                let rstd = 1.0 / (ss / NORM_GROUP as f32 + 1e-5).sqrt();
                for k in 0..NORM_GROUP / 256 {
                    for t in 0..256 {
                        let c = g * NORM_GROUP + k * 256 + t;
                        out[i * INNER + c] = rb(vals[k * 256 + t] * rstd * w.norm_w[c]);
                    }
                }
            }
        }
        MambaOut { xbc, dt, y, out }
    }

    /// mamba_set_ctl for this sequence after a step of `rows` rows.
    pub fn set_ctl(&mut self, rows: usize, accepted: usize, deferred: bool) {
        if deferred {
            self.replay = if accepted > 0 { accepted.min(rows) } else { rows };
            self.parity ^= 1;
        } else {
            self.replay = 0;
        }
    }
}

/// FP64 recurrence of a whole sequence from a zero state with the model's rounding points (BF16 in_proj,
/// BF16 conv output, BF16 y): returns y before its BF16 rounding [rows, INNER] and the final state.
pub fn mamba_scan_f64(w: &MambaParams, proj: &[f32], rows: usize) -> (Vec<f64>, Vec<f64>) {
    let mut seq = MambaSeq::default();
    let o = seq.step_emu(w, proj, rows, false); // conv and dt as the model computes them
    let mut st = vec![0.0f64; HEADS * HEAD_DIM * STATE];
    let mut y = vec![0.0f64; rows * INNER];
    for h in 0..HEADS {
        let g = h / (HEADS / GROUPS);
        let a = -(w.a_log[h] as f64).exp();
        for i in 0..rows {
            let d = {
                let v = rb(proj[i * PROJ + DT_COL + h]) as f64 + w.dt_bias[h] as f64;
                if v <= 20.0 { v.exp().ln_1p() } else { v }
            };
            let da = (d * a).exp();
            let conv = &o.xbc[i * CONV_DIM..(i + 1) * CONV_DIM];
            for p in 0..HEAD_DIM {
                let x = conv[h * HEAD_DIM + p] as f64;
                let s = &mut st[(h * HEAD_DIM + p) * STATE..(h * HEAD_DIM + p + 1) * STATE];
                let mut acc = 0.0f64;
                for n in 0..STATE {
                    s[n] = s[n] * da + conv[INNER + g * STATE + n] as f64 * d * x;
                    acc += s[n] * conv[INNER + GROUPS * STATE + g * STATE + n] as f64;
                }
                y[i * INNER + h * HEAD_DIM + p] = acc + x * w.d[h] as f64;
            }
        }
    }
    (y, st)
}

/// Random in_proj rows [rows, PROJ] of moderate magnitude.
pub fn random_proj(rng: &mut Rng, rows: usize) -> Vec<f32> {
    rng.vec(rows * PROJ, -2.0, 2.0)
}

// ---- MoE -------------------------------------------------------------------------------------------

/// moe_route as the kernel computes it: (experts ascending, weights).
pub fn route_emu(logits: &[f32], bias: &[f32], scaling: f32) -> (Vec<i32>, Vec<f32>) {
    let score: Vec<f32> = logits[..EXPERTS].iter().map(|&l| 1.0 / (1.0 + (-l).exp())).collect();
    let choice: Vec<f32> = score.iter().zip(bias).map(|(s, b)| {
        let c = s + b;
        if c.is_nan() { f32::NEG_INFINITY } else { c }
    }).collect();
    let mut idx: Vec<usize> = (0..EXPERTS).collect();
    idx.sort_by(|&a, &b| choice[b].partial_cmp(&choice[a]).unwrap().then(a.cmp(&b)));
    let mut sel: Vec<usize> = idx[..TOPK].to_vec();
    sel.sort();
    let mut sum = 0.0f32;
    for &e in &sel {
        sum += score[e];
    }
    let w = sel.iter().map(|&e| score[e] / (sum + 1e-20) * scaling).collect();
    (sel.into_iter().map(|e| e as i32).collect(), w)
}

pub fn e2m1_value(c: u8) -> f32 {
    let m = (c & 7) as usize;
    let a = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][m];
    if c & 8 != 0 { -a } else { a }
}

pub fn e4m3_value(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f32;
    if e == 15 && (b & 7) == 7 {
        return f32::NAN;
    }
    if e == 0 { s * m / 8.0 * 2f32.powi(-6) } else { s * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

/// E4M3 code of x, round to nearest even, saturating to +-448 (as __nv_fp8_e4m3(float)).
pub fn e4m3_encode(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x < 0.0 || (x == 0.0 && x.is_sign_negative()) { 0x80u8 } else { 0 };
    let a = x.abs();
    if a >= 448.0 {
        return sign | 0x7e;
    }
    // Nearest of the 127 non-negative finite codes; ties to the even code.
    let mut best = 0u8;
    let mut best_d = f32::INFINITY;
    for c in 0u8..=0x7e {
        let d = (e4m3_value(c) - a).abs();
        if d < best_d || (d == best_d && c & 1 == 0) {
            best = c;
            best_d = d;
        }
    }
    sign | best
}

/// Dequantized row n of one expert projection [N, K] (codes [N][K/2], scales [N][K/16]) times s2, in f64.
pub fn nvfp4_row(codes: &[u8], scales: &[u8], k: usize, n: usize) -> Vec<f64> {
    (0..k)
        .map(|i| {
            let b = codes[n * k / 2 + i / 2];
            let c = if i % 2 == 0 { b & 15 } else { b >> 4 };
            e2m1_value(c) as f64 * e4m3_value(scales[n * k / 16 + i / 16]) as f64
        })
        .collect()
}

/// Host copy of a layer's routed experts (contiguous over experts), for tests.
pub struct NvExpertsHost {
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    pub s2: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

impl NvExpertsHost {
    /// Random codes, block scales around `scale` (E4M3 codes 0x28..0x38) and s2 near 1e-2.
    pub fn random(rng: &mut Rng, experts: usize, n: usize, k: usize) -> Self {
        let codes = (0..experts * n * k / 2).map(|_| rng.next_u64() as u8).collect();
        let scales = (0..experts * n * k / 16).map(|_| 0x28 + (rng.next_u64() % 16) as u8).collect();
        let s2 = rng.vec(experts, 0.005, 0.02);
        Self { codes, scales, s2, n, k }
    }
    pub fn dot_f64(&self, e: usize, n: usize, x: &[f32]) -> f64 {
        let (cpe, spe) = (self.n * self.k / 2, self.n * self.k / 16);
        let row = nvfp4_row(&self.codes[e * cpe..(e + 1) * cpe], &self.scales[e * spe..(e + 1) * spe], self.k, n);
        row.iter().zip(x).map(|(w, &x)| w * x as f64).sum::<f64>() * self.s2[e] as f64
    }
}

/// relu² of a BF16 Linear output.
pub fn relu2_emu(x: f32) -> f32 {
    let v = rb(x).max(0.0);
    rb(v * v)
}

/// moe_combine as the kernel computes it, y rows indexed by pair (rows * 22 of them).
pub fn combine_emu(y: &[f32], sel_w: &[f32], rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * LATENT];
    for r in 0..rows {
        for n in 0..LATENT {
            let mut acc = 0.0f32;
            for k in 0..TOPK {
                let pair = r * TOPK + k;
                acc += rb(y[pair * LATENT + n]) * sel_w[pair];
            }
            out[r * LATENT + n] = rb(acc);
        }
    }
    out
}

// ---- Attention -------------------------------------------------------------------------------------

/// attn_prep's cache row of one position from its BF16-rounded k [2][128], v [2][128].
pub fn kv_row_emu(k: &[f32], v: &[f32], fp8: bool) -> Vec<u8> {
    if !fp8 {
        let mut out = Vec::with_capacity(KV_ROW_BYTES_BF16);
        for x in k.iter().chain(v) {
            out.extend_from_slice(&bf16::from_f32(rb(*x)).to_le_bytes());
        }
        return out;
    }
    let mut codes = vec![0u8; 2 * KVH * HD];
    let mut sc = [0.0f32; 2 * KVH];
    for (part, src) in [k, v].iter().enumerate() {
        for g in 0..KVH {
            let vals: Vec<f32> = src[g * HD..(g + 1) * HD].iter().map(|&x| rb(x)).collect();
            let amax = vals.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let s = if amax > 0.0 { amax / 448.0 } else { 1.0 };
            for (d, x) in vals.iter().enumerate() {
                codes[part * KVH * HD + g * HD + d] = e4m3_encode(x / s);
            }
            sc[part * KVH + g] = s;
        }
    }
    let mut out = codes;
    for s in sc {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Dequantized (k, v) [2][128] each of a cache row.
pub fn kv_row_decode(row: &[u8], fp8: bool) -> (Vec<f64>, Vec<f64>) {
    let mut k = vec![0.0f64; KVH * HD];
    let mut v = vec![0.0f64; KVH * HD];
    for g in 0..KVH {
        for d in 0..HD {
            let i = g * HD + d;
            if fp8 {
                let sc = |j: usize| f32::from_le_bytes(row[2 * KVH * HD + 4 * j..2 * KVH * HD + 4 * j + 4].try_into().unwrap()) as f64;
                k[i] = e4m3_value(row[i]) as f64 * sc(g);
                v[i] = e4m3_value(row[KVH * HD + i]) as f64 * sc(KVH + g);
            } else {
                let b = |o: usize| bf16::to_f32(u16::from_le_bytes([row[2 * o], row[2 * o + 1]])) as f64;
                k[i] = b(i);
                v[i] = b(KVH * HD + i);
            }
        }
    }
    (k, v)
}

/// Exact softmax attention (FP64) of one row: q [32][128] (BF16 values) over cached rows 0 ..= pos.
pub fn attention_f64(q: &[f32], rows: &[Vec<u8>], pos: usize, fp8: bool) -> Vec<f64> {
    let kv: Vec<(Vec<f64>, Vec<f64>)> = rows[..=pos].iter().map(|r| kv_row_decode(r, fp8)).collect();
    let mut out = vec![0.0f64; QH * HD];
    for h in 0..QH {
        let g = h / (QH / KVH);
        let s: Vec<f64> = kv
            .iter()
            .map(|(k, _)| (0..HD).map(|d| q[h * HD + d] as f64 * k[g * HD + d]).sum::<f64>() / (HD as f64).sqrt())
            .collect();
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let p: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
        let l: f64 = p.iter().sum();
        for d in 0..HD {
            out[h * HD + d] = p.iter().zip(&kv).map(|(p, (_, v))| p * v[g * HD + d]).sum::<f64>() / l;
        }
    }
    out
}

/// A cache row's K and V as the kernels see them: values before the per-key scale (BF16 values or E4M3
/// codes' values) and the scales (1 for BF16).
pub struct KvTile {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub ks: [f32; KVH],
    pub vs: [f32; KVH],
}

pub fn kv_row_unscaled(row: &[u8], fp8: bool) -> KvTile {
    let mut t = KvTile { k: vec![0.0; KVH * HD], v: vec![0.0; KVH * HD], ks: [1.0; KVH], vs: [1.0; KVH] };
    for i in 0..KVH * HD {
        if fp8 {
            t.k[i] = e4m3_value(row[i]);
            t.v[i] = e4m3_value(row[KVH * HD + i]);
        } else {
            t.k[i] = bf16::to_f32(u16::from_le_bytes([row[2 * i], row[2 * i + 1]]));
            t.v[i] = bf16::to_f32(u16::from_le_bytes([row[2 * (KVH * HD + i)], row[2 * (KVH * HD + i) + 1]]));
        }
    }
    if fp8 {
        for g in 0..KVH {
            let sc = |j: usize| f32::from_le_bytes(row[2 * KVH * HD + 4 * j..2 * KVH * HD + 4 * j + 4].try_into().unwrap());
            t.ks[g] = sc(g);
            t.vs[g] = sc(KVH + g);
        }
    }
    t
}

/// The canonical tiled algorithm in FP32 with sequential dot products (not the MMAs' order): checks the
/// tile / group / fold structure and the masking against [`attention_f64`].
pub fn attention_canonical_cpu(q: &[f32], rows: &[KvTile], pos: usize) -> Vec<f32> {
    const C: f32 = 0.088_388_35_f32 * std::f32::consts::LOG2_E;
    let tiles_per_group = ATTN_GROUP / ATTN_TILE;
    let mut out = vec![0.0f32; QH * HD];
    for h in 0..QH {
        let g = h / (QH / KVH);
        let mut total: Option<(f32, f32, Vec<f32>)> = None;
        for gi in 0..=pos / ATTN_GROUP {
            let (mut m, mut l, mut acc) = (f32::NEG_INFINITY, 0.0f32, vec![0.0f32; HD]);
            let t_last = (gi * tiles_per_group + tiles_per_group - 1).min(pos / ATTN_TILE);
            for t in gi * tiles_per_group..=t_last {
                let mut s = [f32::NEG_INFINITY; ATTN_TILE];
                let mut pv = [0.0f32; ATTN_TILE];
                for (kj, sv) in s.iter_mut().enumerate() {
                    let j = t * ATTN_TILE + kj;
                    if j <= pos {
                        let dot: f32 = (0..HD).map(|d| q[h * HD + d] * rows[j].k[g * HD + d]).sum();
                        *sv = dot * (rows[j].ks[g] * C);
                    }
                }
                let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mn = m.max(mx);
                let alpha = (m - mn).exp2();
                m = mn;
                let mut sum = 0.0f32;
                for kj in 0..ATTN_TILE {
                    let p = (s[kj] - mn).exp2();
                    sum += p;
                    let j = t * ATTN_TILE + kj;
                    pv[kj] = if j <= pos { rb(p * rows[j].vs[g]) } else { 0.0 };
                }
                l = l * alpha + sum;
                for (d, a) in acc.iter_mut().enumerate() {
                    *a *= alpha;
                    for (kj, p) in pv.iter().enumerate() {
                        let j = t * ATTN_TILE + kj;
                        if j <= pos {
                            *a += p * rows[j].v[g * HD + d];
                        }
                    }
                }
            }
            total = Some(match total {
                None => (m, l, acc),
                Some((mt, lt, at)) => {
                    let mm = mt.max(m);
                    let (a, b) = ((mt - mm).exp2(), (m - mm).exp2());
                    (mm, lt * a + l * b, at.iter().zip(&acc).map(|(x, y)| x * a + y * b).collect())
                }
            });
        }
        let (_, l, acc) = total.unwrap();
        for d in 0..HD {
            out[h * HD + d] = rb(acc[d] / l);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_rel(a: &[f64], b: &[f64]) -> (f64, f64) {
        let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let max_abs = a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        (max_abs, max_abs / scale)
    }

    /// The FP32 recurrence (kernel order) against FP64: 300 tokens (beyond two 128-token chunks).
    #[test]
    fn mamba_f32_vs_f64() {
        let mut rng = Rng::new(1);
        let w = MambaParams::random(&mut rng);
        let rows = 300;
        let proj = random_proj(&mut rng, rows);
        let mut seq = MambaSeq::default();
        let o = seq.step_emu(&w, &proj, rows, false);
        let (y64, st64) = mamba_scan_f64(&w, &proj, rows);
        let st32: Vec<f64> = seq.ssm.iter().map(|&x| x as f64).collect();
        let (sa, sr) = max_rel(&st32, &st64);
        // y in BF16: compare against the FP64 y rounded the same way, in BF16 ulps.
        let mut flips = 0usize;
        let ymax = y64.iter().fold(0.0f64, |m, x| m.max(x.abs())) as f32;
        let mut worst = 0.0f32;
        for (a, b) in o.y.iter().zip(&y64) {
            let r = rb(*b as f32);
            if *a != r {
                flips += 1;
                // One BF16 ulp of the larger of |y| and 1e-4 of the largest |y| (cancellation).
                let e = (a - r).abs() / r.abs().max(1e-4 * ymax);
                worst = worst.max(e);
                assert!(e < 0.0079, "y {a} vs {r}");
            }
        }
        eprintln!("y: max |y| {ymax:.3}, worst flip {worst:.2e} relative");
        eprintln!("mamba f32 vs f64: state max abs {sa:.3e} (rel to max {sr:.3e}); y BF16 flips {flips} of {}", o.y.len());
        assert!(sr < 1e-5, "state error {sr}");
        assert!((flips as f64) < 0.01 * o.y.len() as f64);
    }

    /// Prefill in one step, in chunks, and token by token give the same bits (the emulation of the
    /// kernels' sequential recurrence).
    #[test]
    fn mamba_split_invariance() {
        let mut rng = Rng::new(2);
        let w = MambaParams::random(&mut rng);
        let rows = 40;
        let proj = random_proj(&mut rng, rows);
        let mut a = MambaSeq::default();
        let oa = a.step_emu(&w, &proj, rows, false);
        let mut b = MambaSeq::default();
        let mut out_b = Vec::new();
        for (r0, n) in [(0, 13), (13, 1), (14, 2), (16, 24)] {
            let o = b.step_emu(&w, &proj[r0 * PROJ..(r0 + n) * PROJ], n, false);
            out_b.extend(o.out);
        }
        assert_eq!(a.ssm, b.ssm);
        assert_eq!(a.conv, b.conv);
        assert_eq!(oa.out, out_b);
    }

    /// Deferred steps with accepted prefixes replayed by the next step equal plain processing of the
    /// accepted tokens.
    #[test]
    fn mamba_deferred_steps() {
        let mut rng = Rng::new(3);
        let w = MambaParams::random(&mut rng);
        // Steps of (rows, accepted); the plain run sees only the accepted rows.
        let plan = [(5usize, 2usize), (4, 4), (3, 1), (6, 3), (1, 1)];
        let mut spec = MambaSeq::default();
        let mut plain = MambaSeq::default();
        // A prefill first.
        let pre = random_proj(&mut rng, 7);
        spec.step_emu(&w, &pre, 7, false);
        spec.set_ctl(7, 0, false);
        plain.step_emu(&w, &pre, 7, false);
        for (rows, acc) in plan {
            let proj = random_proj(&mut rng, rows);
            let os = spec.step_emu(&w, &proj, rows, true);
            spec.set_ctl(rows, acc, true);
            let op = plain.step_emu(&w, &proj[..acc * PROJ], acc, false);
            assert_eq!(&os.out[..acc * INNER], &op.out[..], "outputs of accepted rows");
        }
        // Flush: a step without rows.
        spec.step_emu(&w, &[], 0, false);
        spec.set_ctl(0, 0, false);
        assert_eq!(spec.ssm, plain.ssm);
        assert_eq!(spec.conv, plain.conv);
    }

    #[test]
    fn rms_vs_f64() {
        let mut rng = Rng::new(6);
        let x = rng.vec(HIDDEN, -30.0, 30.0);
        let w: Vec<f32> = rng.vec(HIDDEN, 0.2, 2.0).into_iter().map(rb).collect();
        let got = rms_emu(&x, &w, 1e-5);
        let ms = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / HIDDEN as f64;
        let r = 1.0 / (ms + 1e-5).sqrt();
        let worst = (0..HIDDEN).fold(0.0f64, |m, c| {
            let want = w[c] as f64 * x[c] as f64 * r;
            m.max((got[c] as f64 - want).abs() / want.abs().max(1e-3))
        });
        assert!(worst < 1e-6, "rms rel error {worst}");
    }

    #[test]
    fn route_semantics() {
        let mut rng = Rng::new(4);
        let logits = rng.vec(EXPERTS, -4.0, 4.0);
        let mut bias = rng.vec(EXPERTS, -0.1, 0.1);
        bias[7] = 10.0; // always chosen
        let (e, w) = route_emu(&logits, &bias, 5.0);
        assert_eq!(e.len(), TOPK);
        assert!(e.windows(2).all(|p| p[0] < p[1]));
        assert!(e.contains(&7));
        let sum: f32 = w.iter().sum();
        assert!((sum - 5.0).abs() < 1e-4);
        // Weights come from the sigmoid scores, not score + bias.
        let s7 = 1.0 / (1.0 + (-logits[7]).exp());
        let i7 = e.iter().position(|&x| x == 7).unwrap();
        let total: f32 = e.iter().map(|&x| 1.0 / (1.0 + (-logits[x as usize]).exp())).sum();
        assert!((w[i7] - s7 / total * 5.0).abs() < 1e-5);
    }

    #[test]
    fn e4m3_codec() {
        for c in 0u8..=0x7e {
            let v = e4m3_value(c);
            assert_eq!(e4m3_encode(v), c, "code {c:#x} value {v}");
            assert_eq!(e4m3_encode(-v), if c == 0 { 0x80 } else { c | 0x80 });
        }
        assert_eq!(e4m3_value(0x7e), 448.0);
        assert_eq!(e4m3_encode(1000.0), 0x7e);
        // Tie between 1.0 (0x38) and 1.125 (0x39): even code.
        assert_eq!(e4m3_encode(1.0625), 0x38);
    }

    /// The canonical tiled algorithm (CPU, FP32) against exact attention, across two groups, both formats.
    #[test]
    fn attention_canonical_vs_f64() {
        let mut rng = Rng::new(5);
        let len = 1100;
        for fp8 in [false, true] {
            let rows: Vec<Vec<u8>> = (0..len)
                .map(|_| {
                    let k = rng.vec(KVH * HD, -2.0, 2.0);
                    let v = rng.vec(KVH * HD, -2.0, 2.0);
                    kv_row_emu(&k, &v, fp8)
                })
                .collect();
            let q: Vec<f32> = rng.vec(QH * HD, -1.0, 1.0).into_iter().map(rb).collect();
            let tiles: Vec<KvTile> = rows.iter().map(|r| kv_row_unscaled(r, fp8)).collect();
            for pos in [0usize, 63, 64, 1023, 1024, 1099] {
                let a = attention_canonical_cpu(&q, &tiles, pos);
                let b = attention_f64(&q, &rows, pos, fp8);
                let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
                let (ma, mr) = max_rel(&a64, &b);
                assert!(mr < 2e-2, "fp8 {fp8} pos {pos}: max abs {ma} rel {mr}");
            }
        }
    }
}
