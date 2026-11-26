//! Nemotron-H kernels on the GPU against the CPU references (ignored by default; needs the GPU):
//! `cargo test --release -p quench6k-cuda --test nemotron -- --ignored --nocapture --test-threads=1`.
//!
//! Each test checks the kernels against `nemotron::reference` (FP32 emulation within an ulp of CUDA's
//! transcendental functions, or FP64) and checks bitwise invariances on the GPU alone: prefill split
//! points, batch composition, deferred speculative steps against plain ones, prefill against decode.

use quench6k_cuda::bf16;
use quench6k_cuda::kernels;
use quench6k_cuda::nemotron::reference::*;
use quench6k_cuda::nemotron::*;
use quench6k_cuda::{DeviceBuffer, Stream};

fn up<T: Copy>(s: &Stream, v: &[T]) -> DeviceBuffer {
    let b = DeviceBuffer::new(std::mem::size_of_val(v).max(256)).unwrap();
    s.upload(&b, v).unwrap();
    b
}

fn bf16s(s: &Stream, addr: u64, n: usize) -> Vec<f32> {
    s.download::<u16>(addr, n).unwrap().into_iter().map(bf16::to_f32).collect()
}

fn f32s(s: &Stream, addr: u64, n: usize) -> Vec<f32> {
    s.download::<f32>(addr, n).unwrap()
}

fn seqs_buf(s: &Stream, q: &Seqs) -> DeviceBuffer {
    up(s, q.as_bytes())
}

/// Count of differing values and the largest difference relative to max(|b|, floor).
fn compare(a: &[f32], b: &[f32], floor: f32) -> (usize, f32) {
    let mut n = 0;
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        if x != y {
            n += 1;
            worst = worst.max((x - y).abs() / y.abs().max(floor));
        }
    }
    (n, worst)
}

/// The block RMSNorm against its emulation (bitwise: no transcendental functions), both outputs.
#[test]
#[ignore]
fn rms_vs_reference() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(51);
    let rows = 9;
    let ldx = HIDDEN + 32;
    let x = rng.vec(rows * ldx, -3.0, 3.0);
    let w: Vec<u16> = rng.vec(HIDDEN, 0.2, 2.0).into_iter().map(bf16::from_f32).collect();
    let wf: Vec<f32> = w.iter().map(|&v| bf16::to_f32(v)).collect();
    let (xb, wb) = (up(&s, &x), up(&s, &w));
    let (o16, o32) = (DeviceBuffer::new(rows * HIDDEN * 2).unwrap(), DeviceBuffer::new(rows * HIDDEN * 4).unwrap());
    rms(&s, xb.addr(), ldx, wb.addr(), 1e-5, rows, o16.addr(), HIDDEN, false).unwrap();
    rms(&s, xb.addr(), ldx, wb.addr(), 1e-5, rows, o32.addr(), HIDDEN, true).unwrap();
    let (g16, g32) = (bf16s(&s, o16.addr(), rows * HIDDEN), f32s(&s, o32.addr(), rows * HIDDEN));
    for r in 0..rows {
        let want = rms_emu(&x[r * ldx..r * ldx + HIDDEN], &wf, 1e-5);
        assert_eq!(&g32[r * HIDDEN..(r + 1) * HIDDEN], &want[..], "row {r} FP32");
        let want16: Vec<f32> = want.iter().map(|&v| rb(v)).collect();
        assert_eq!(&g16[r * HIDDEN..(r + 1) * HIDDEN], &want16[..], "row {r} BF16");
    }
    eprintln!("rms: bitwise equal to the emulation (BF16 and FP32 outputs)");
}

// ---- Mamba -----------------------------------------------------------------------------------------

const LAYERS: usize = 2;
const LAYER: usize = 1;
const REC: usize = 8;
const LD: usize = PROJ + 64;

struct MambaGpu {
    w: MambaW,
    _keep: Vec<DeviceBuffer>,
}

fn mamba_weights(s: &Stream, p: &MambaParams) -> MambaGpu {
    let bufs: Vec<DeviceBuffer> = [&p.conv_w, &p.conv_b, &p.dt_bias, &p.a_log, &p.d, &p.norm_w].iter().map(|v| up(s, v)).collect();
    let w = MambaW {
        conv_w: bufs[0].addr(),
        conv_b: bufs[1].addr(),
        dt_bias: bufs[2].addr(),
        a_log: bufs[3].addr(),
        d: bufs[4].addr(),
        norm_w: bufs[5].addr(),
        dt_min: p.dt_min,
        dt_max: p.dt_max,
    };
    MambaGpu { w, _keep: bufs }
}

fn new_block() -> DeviceBuffer {
    DeviceBuffer::zeroed(mamba_seq_bytes(LAYERS, REC)).unwrap()
}

fn layer_off() -> usize {
    MAMBA_HEADER_BYTES + LAYER * mamba_layer_bytes(REC)
}

/// One Mamba step of several sequences (rows per sequence from `projs`), returns each sequence's out rows.
fn mamba_step(s: &Stream, w: &MambaGpu, blocks: &[&DeviceBuffer], projs: &[&[f32]], defer: bool) -> Vec<Vec<f32>> {
    mamba_step_with(s, w, blocks, projs, defer, false)
}

/// [`mamba_step`], with `small` through [`mamba_mixer_small`].
fn mamba_step_with(s: &Stream, w: &MambaGpu, blocks: &[&DeviceBuffer], projs: &[&[f32]], defer: bool, small: bool) -> Vec<Vec<f32>> {
    let mut q = Seqs::default();
    let mut rows = 0;
    let mut host = Vec::new();
    for (i, p) in projs.iter().enumerate() {
        let n = p.len() / PROJ;
        q.mamba[i] = blocks[i].addr();
        q.row0[i] = rows as i32;
        for r in 0..n {
            host.extend_from_slice(&p[r * PROJ..(r + 1) * PROJ]);
            host.extend(std::iter::repeat_n(0.0f32, LD - PROJ));
        }
        rows += n;
    }
    q.nseq = projs.len() as i32;
    q.row0[projs.len()] = rows as i32;
    let qb = seqs_buf(s, &q);
    let proj = up(s, &host);
    let r = rows.max(1);
    let (xbc, dt, y) = (
        DeviceBuffer::new(r * CONV_DIM * 2).unwrap(),
        DeviceBuffer::new(r * HEADS * 4).unwrap(),
        DeviceBuffer::new(r * INNER * 2).unwrap(),
    );
    let scratch = MambaScratch { xbc: xbc.addr(), dt: dt.addr(), y: y.addr() };
    let out = DeviceBuffer::new(r * INNER * 2).unwrap();
    if small {
        mamba_mixer_small(s, proj.addr(), LD, rows, qb.addr(), projs.len(), LAYER, REC, &w.w, defer, &scratch, out.addr()).unwrap();
    } else {
        mamba_mixer(s, proj.addr(), LD, rows, qb.addr(), projs.len(), LAYER, REC, &w.w, defer, &scratch, out.addr()).unwrap();
    }
    s.synchronize().unwrap();
    let all = bf16s(s, out.addr(), rows * INNER);
    let mut res = Vec::new();
    for i in 0..projs.len() {
        res.push(all[q.row0[i] as usize * INNER..q.row0[i + 1] as usize * INNER].to_vec());
    }
    res
}

fn set_ctl(s: &Stream, blocks: &[&DeviceBuffer], rows: &[usize], accepted: &[i32], deferred: bool) {
    let mut q = Seqs::default();
    let mut r = 0;
    for (i, n) in rows.iter().enumerate() {
        q.mamba[i] = blocks[i].addr();
        q.row0[i] = r;
        r += *n as i32;
    }
    q.row0[rows.len()] = r;
    q.nseq = rows.len() as i32;
    let qb = seqs_buf(s, &q);
    let acc = up(s, accepted);
    mamba_set_ctl(s, qb.addr(), rows.len(), acc.addr(), deferred).unwrap();
    s.synchronize().unwrap();
}

fn flush(s: &Stream, w: &MambaGpu, blocks: &[&DeviceBuffer]) {
    let mut q = Seqs::default();
    for (i, b) in blocks.iter().enumerate() {
        q.mamba[i] = b.addr();
    }
    q.nseq = blocks.len() as i32;
    let qb = seqs_buf(s, &q);
    mamba_flush(s, qb.addr(), blocks.len(), LAYER, REC, &w.w).unwrap();
    s.synchronize().unwrap();
    set_ctl(s, blocks, &vec![0; blocks.len()], &vec![0; blocks.len()], false);
}

fn ssm_of(s: &Stream, b: &DeviceBuffer) -> Vec<f32> {
    f32s(s, b.addr() + layer_off() as u64, HEADS * HEAD_DIM * STATE)
}

fn conv_of(s: &Stream, b: &DeviceBuffer) -> Vec<f32> {
    f32s(s, b.addr() + (layer_off() + SSM_BYTES) as u64, 3 * CONV_DIM)
}

fn check_against_emu(s: &Stream, what: &str, gpu_out: &[f32], emu_out: &[f32], block: &DeviceBuffer, emu: &MambaSeq) {
    let (n, worst) = compare(gpu_out, emu_out, 1e-2);
    let ssm = ssm_of(s, block);
    let smax = emu.ssm.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let sdiff = ssm.iter().zip(&emu.ssm).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let conv_eq = conv_of(s, block) == emu.conv;
    eprintln!("{what}: out differs in {n} of {} (worst {worst:.2e}); state max diff {sdiff:.2e} (max |S| {smax:.2e}); conv state equal {conv_eq}", gpu_out.len());
    assert!(n * 200 < gpu_out.len().max(200), "{what}: too many output differences");
    assert!(worst < 0.02, "{what}: output difference {worst}");
    assert!(sdiff <= 1e-5 * smax, "{what}: state difference {sdiff}");
    assert!(conv_eq, "{what}: conv state");
}

/// Kernels against the CPU emulation: a prefill, a deferred multi-sequence step, a replaying step.
#[test]
#[ignore]
fn mamba_vs_reference() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(11);
    let p = MambaParams::random(&mut rng);
    let w = mamba_weights(&s, &p);
    let blocks: Vec<DeviceBuffer> = (0..3).map(|_| new_block()).collect();
    let br: Vec<&DeviceBuffer> = blocks.iter().collect();
    let mut emu: Vec<MambaSeq> = (0..3).map(|_| MambaSeq::default()).collect();
    // Step 1: prefill-sized rows for sequence 0, small for 1 and 2.
    let rows = [150usize, 5, 1];
    let projs: Vec<Vec<f32>> = rows.iter().map(|&n| random_proj(&mut rng, n)).collect();
    let pr: Vec<&[f32]> = projs.iter().map(|v| &v[..]).collect();
    let outs = mamba_step(&s, &w, &br, &pr, false);
    for i in 0..3 {
        let o = emu[i].step_emu(&p, &projs[i], rows[i], false);
        check_against_emu(&s, &format!("step 1 seq {i}"), &outs[i], &o.out, &blocks[i], &emu[i]);
    }
    set_ctl(&s, &br, &rows, &[0, 0, 0], false);
    // Step 2: deferred, accepted 2 / 3 / 1.
    let rows = [4usize, 3, 2];
    let acc = [2i32, 3, 1];
    let projs: Vec<Vec<f32>> = rows.iter().map(|&n| random_proj(&mut rng, n)).collect();
    let pr: Vec<&[f32]> = projs.iter().map(|v| &v[..]).collect();
    let outs = mamba_step(&s, &w, &br, &pr, true);
    for i in 0..3 {
        let o = emu[i].step_emu(&p, &projs[i], rows[i], true);
        check_against_emu(&s, &format!("step 2 (deferred) seq {i}"), &outs[i], &o.out, &blocks[i], &emu[i]);
        emu[i].set_ctl(rows[i], acc[i] as usize, true);
    }
    set_ctl(&s, &br, &rows, &acc, true);
    // Step 3: plain, replaying the accepted rows first.
    let rows = [1usize, 2, 1];
    let projs: Vec<Vec<f32>> = rows.iter().map(|&n| random_proj(&mut rng, n)).collect();
    let pr: Vec<&[f32]> = projs.iter().map(|v| &v[..]).collect();
    let outs = mamba_step(&s, &w, &br, &pr, false);
    for i in 0..3 {
        let o = emu[i].step_emu(&p, &projs[i], rows[i], false);
        check_against_emu(&s, &format!("step 3 (replay) seq {i}"), &outs[i], &o.out, &blocks[i], &emu[i]);
    }
}

/// Bitwise GPU invariances: prefill splits and single-row steps, batch composition, deferred steps.
#[test]
#[ignore]
fn mamba_invariance() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(12);
    let p = MambaParams::random(&mut rng);
    let w = mamba_weights(&s, &p);
    let n = 100;
    let proj = random_proj(&mut rng, n);
    // One launch.
    let a = new_block();
    let out_a = mamba_step(&s, &w, &[&a], &[&proj], false).remove(0);
    // Chunks of 37, 1, 2, 60.
    let b = new_block();
    let mut out_b = Vec::new();
    for (r0, k) in [(0usize, 37usize), (37, 1), (38, 2), (40, 60)] {
        out_b.extend(mamba_step(&s, &w, &[&b], &[&proj[r0 * PROJ..(r0 + k) * PROJ]], false).remove(0));
    }
    // Row by row.
    let c = new_block();
    let mut out_c = Vec::new();
    for r in 0..n {
        out_c.extend(mamba_step(&s, &w, &[&c], &[&proj[r * PROJ..(r + 1) * PROJ]], false).remove(0));
    }
    assert!(out_a == out_b && out_a == out_c, "outputs depend on the prefill split");
    assert!(ssm_of(&s, &a) == ssm_of(&s, &b) && ssm_of(&s, &a) == ssm_of(&s, &c), "state depends on the split");
    assert!(conv_of(&s, &a) == conv_of(&s, &b) && conv_of(&s, &a) == conv_of(&s, &c));
    eprintln!("prefill split / row-by-row: bitwise equal");

    // Batch composition: sequence x alone vs with two others.
    let px = random_proj(&mut rng, 3);
    let po = random_proj(&mut rng, 7);
    let (x1, x2, o1, o2) = (new_block(), new_block(), new_block(), new_block());
    let alone = mamba_step(&s, &w, &[&x1], &[&px], false).remove(0);
    let together = mamba_step(&s, &w, &[&o1, &x2, &o2], &[&po, &px, &po[..PROJ]], false).remove(1);
    assert_eq!(alone, together, "batch composition changes a sequence's output");
    assert_eq!(ssm_of(&s, &x1), ssm_of(&s, &x2));
    eprintln!("batch composition: bitwise equal");

    // Deferred steps with accepted prefixes (then a flush) vs plain steps over the accepted rows.
    let (sp, pl) = (new_block(), new_block());
    let pre = random_proj(&mut rng, 9);
    mamba_step(&s, &w, &[&sp], &[&pre], false);
    mamba_step(&s, &w, &[&pl], &[&pre], false);
    for (rows, acc) in [(5usize, 2usize), (4, 4), (3, 1), (6, 3), (2, 1)] {
        let pj = random_proj(&mut rng, rows);
        let os = mamba_step(&s, &w, &[&sp], &[&pj], true).remove(0);
        set_ctl(&s, &[&sp], &[rows], &[acc as i32], true);
        let op = mamba_step(&s, &w, &[&pl], &[&pj[..acc * PROJ]], false).remove(0);
        assert_eq!(&os[..acc * INNER], &op[..], "accepted rows' outputs");
    }
    flush(&s, &w, &[&sp]);
    assert_eq!(ssm_of(&s, &sp), ssm_of(&s, &pl), "state after deferred steps");
    assert_eq!(conv_of(&s, &sp), conv_of(&s, &pl), "conv state after deferred steps");
    eprintln!("deferred steps + flush: bitwise equal to plain steps");
}

// ---- MoE -------------------------------------------------------------------------------------------

const TEST_EXPERTS: usize = 40;

#[test]
#[ignore]
fn moe_route_vs_reference() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(21);
    let rows = 37;
    let logits = rng.vec(rows * EXPERTS, -5.0, 5.0);
    let bias = rng.vec(EXPERTS, -0.05, 0.05);
    let (lb, bb) = (up(&s, &logits), up(&s, &bias));
    let (e, wv) = (DeviceBuffer::new(rows * TOPK * 4).unwrap(), DeviceBuffer::new(rows * TOPK * 4).unwrap());
    moe_route(&s, lb.addr(), EXPERTS, bb.addr(), 5.0, rows, e.addr(), wv.addr()).unwrap();
    let ge: Vec<i32> = s.download(e.addr(), rows * TOPK).unwrap();
    let gw = f32s(&s, wv.addr(), rows * TOPK);
    let mut worst = 0.0f32;
    for r in 0..rows {
        let (ce, cw) = route_emu(&logits[r * EXPERTS..(r + 1) * EXPERTS], &bias, 5.0);
        assert_eq!(&ge[r * TOPK..(r + 1) * TOPK], &ce[..], "row {r} experts");
        for k in 0..TOPK {
            worst = worst.max((gw[r * TOPK + k] - cw[k]).abs() / cw[k]);
        }
    }
    eprintln!("route: experts equal, weights worst rel diff {worst:.2e}");
    assert!(worst < 1e-5);
}

struct ExpertsGpu {
    up: NvExperts,
    down: NvExperts,
    up_tabs: (DeviceBuffer, DeviceBuffer),
    down_tabs: (DeviceBuffer, DeviceBuffer),
    _keep: Vec<DeviceBuffer>,
}

fn experts_gpu(s: &Stream, h_up: &NvExpertsHost, h_down: &NvExpertsHost) -> ExpertsGpu {
    let mut keep = Vec::new();
    let mut mk = |h: &NvExpertsHost| -> (NvExperts, (DeviceBuffer, DeviceBuffer)) {
        let (c, sc, s2) = (up(s, &h.codes), up(s, &h.scales), up(s, &h.s2));
        let ct: Vec<u64> = (0..EXPERTS).map(|e| c.addr() + (e.min(TEST_EXPERTS - 1) * h.n * h.k / 2) as u64).collect();
        let st: Vec<u64> = (0..EXPERTS).map(|e| sc.addr() + (e.min(TEST_EXPERTS - 1) * h.n * h.k / 16) as u64).collect();
        let nv = NvExperts { codes: c.addr(), scales: sc.addr(), s2: s2.addr() };
        let tabs = (up(s, &ct), up(s, &st));
        keep.extend([c, sc, s2]);
        (nv, tabs)
    };
    let (upn, up_tabs) = mk(h_up);
    let (downn, down_tabs) = mk(h_down);
    ExpertsGpu { up: upn, down: downn, up_tabs, down_tabs, _keep: keep }
}

/// Random routing over the first TEST_EXPERTS experts (22 distinct, ascending) with weights.
fn routing(rng: &mut Rng, rows: usize) -> (Vec<i32>, Vec<f32>) {
    let mut e = Vec::new();
    let mut w = Vec::new();
    for _ in 0..rows {
        let mut pick: Vec<i32> = (0..TEST_EXPERTS as i32).collect();
        for i in (1..pick.len()).rev() {
            pick.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
        }
        let mut sel = pick[..TOPK].to_vec();
        sel.sort();
        e.extend(sel);
        w.extend(rng.vec(TOPK, 0.05, 0.5));
    }
    (e, w)
}

fn experts_small_run(s: &Stream, ex: &ExpertsGpu, x: &[u16], sel: &[i32], rows: usize) -> (Vec<f32>, Vec<f32>) {
    let (xb, eb) = (up(s, x), up(s, sel));
    let pairs = rows * TOPK;
    let order = DeviceBuffer::new(pairs * 4).unwrap();
    let act = DeviceBuffer::new(pairs * MOE_INTER * 2).unwrap();
    let y = DeviceBuffer::new(pairs * LATENT * 4).unwrap();
    moe_experts_small(s, xb.addr(), eb.addr(), &ex.up, &ex.down, rows, order.addr(), act.addr(), y.addr()).unwrap();
    s.synchronize().unwrap();
    (bf16s(s, act.addr(), pairs * MOE_INTER), f32s(s, y.addr(), pairs * LATENT))
}

#[test]
#[ignore]
fn moe_experts_vs_reference() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(22);
    let h_up = NvExpertsHost::random(&mut rng, TEST_EXPERTS, MOE_INTER, LATENT);
    let h_down = NvExpertsHost::random(&mut rng, TEST_EXPERTS, LATENT, MOE_INTER);
    let ex = experts_gpu(&s, &h_up, &h_down);
    let rows = 5;
    let x: Vec<u16> = rng.vec(rows * LATENT, -1.0, 1.0).into_iter().map(bf16::from_f32).collect();
    let xf: Vec<f32> = x.iter().map(|&v| bf16::to_f32(v)).collect();
    let (sel, sw) = routing(&mut rng, rows);
    let (act, y) = experts_small_run(&s, &ex, &x, &sel, rows);
    // Activations against FP64 dots with the model's rounding.
    let mut flips = 0;
    for pair in 0..rows * TOPK {
        let (r, e) = (pair / TOPK, sel[pair] as usize);
        for n in 0..MOE_INTER {
            let v = h_up.dot_f64(e, n, &xf[r * LATENT..(r + 1) * LATENT]) as f32;
            let want = relu2_emu(v);
            let got = act[pair * MOE_INTER + n];
            if got != want {
                flips += 1;
                assert!((got - want).abs() <= 0.016 * want.abs().max(1e-3), "act pair {pair} n {n}: {got} vs {want}");
            }
        }
    }
    // Down outputs against FP64 dots of the GPU's activations.
    let mut worst = 0.0f64;
    let ymax = y.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
    for pair in 0..rows * TOPK {
        let e = sel[pair] as usize;
        for n in 0..LATENT {
            let want = h_down.dot_f64(e, n, &act[pair * MOE_INTER..(pair + 1) * MOE_INTER]);
            worst = worst.max((y[pair * LATENT + n] as f64 - want).abs() / ymax);
        }
    }
    eprintln!("experts small: act BF16 flips {flips} of {}; down max diff {worst:.2e} of max |y|", act.len());
    assert!(flips * 100 < act.len());
    assert!(worst < 1e-5);
    // Batch invariance: each row alone.
    for r in 0..rows {
        let (a1, y1) = experts_small_run(&s, &ex, &x[r * LATENT..(r + 1) * LATENT], &sel[r * TOPK..(r + 1) * TOPK], 1);
        assert_eq!(&a1[..], &act[r * TOPK * MOE_INTER..(r + 1) * TOPK * MOE_INTER], "row {r} act alone vs batched");
        assert_eq!(&y1[..], &y[r * TOPK * LATENT..(r + 1) * TOPK * LATENT], "row {r} y alone vs batched");
    }
    // Combine against the emulation (bitwise).
    let (yb, wb, out) = (up(&s, &y), up(&s, &sw), DeviceBuffer::new(rows * LATENT * 2).unwrap());
    moe_combine(&s, yb.addr(), 0, wb.addr(), rows, out.addr()).unwrap();
    let got = bf16s(&s, out.addr(), rows * LATENT);
    assert_eq!(got, combine_emu(&y, &sw, rows), "combine");
    eprintln!("experts small: batch invariant; combine bitwise equal to the emulation");

    // Prefill path: plan, gather, W4A16 grouped GEMMs, relu², combine; close to the small path.
    let rows_p = 48;
    let xp: Vec<u16> = rng.vec(rows_p * LATENT, -1.0, 1.0).into_iter().map(bf16::from_f32).collect();
    let (selp, swp) = routing(&mut rng, rows_p);
    let pairs = rows_p * TOPK;
    let tile = kernels::moe_tile_rows();
    let max_tiles = pairs / tile + EXPERTS;
    let ib = |n: usize| DeviceBuffer::new(n * 4).unwrap();
    let (counts, offsets, cursor, te, tr, nt, pr) = (ib(EXPERTS), ib(EXPERTS + 1), ib(EXPERTS), ib(max_tiles), ib(max_tiles), ib(1), ib(pairs));
    let (xb, eb, wb) = (up(&s, &xp), up(&s, &selp), up(&s, &swp));
    let xs = DeviceBuffer::new(pairs * LATENT * 2).unwrap();
    let yu = DeviceBuffer::new(pairs * MOE_INTER * 4).unwrap();
    let act_p = DeviceBuffer::new(pairs * MOE_INTER * 2).unwrap();
    let yd = DeviceBuffer::new(pairs * LATENT * 4).unwrap();
    let outp = DeviceBuffer::new(rows_p * LATENT * 2).unwrap();
    moe_plan(&s, eb.addr(), rows_p, tile, counts.addr(), offsets.addr(), cursor.addr(), te.addr(), tr.addr(), nt.addr(), pr.addr()).unwrap();
    moe_gather(&s, xb.addr(), pr.addr(), rows_p, xs.addr()).unwrap();
    kernels::moe_nvfp4_gemm(&s, xs.addr(), ex.up_tabs.0.addr(), ex.up_tabs.1.addr(), ex.up.s2, offsets.addr(), te.addr(), tr.addr(), nt.addr(), max_tiles, yu.addr(), MOE_INTER, LATENT).unwrap();
    relu2(&s, yu.addr(), MOE_INTER, MOE_INTER, pairs, act_p.addr(), MOE_INTER).unwrap();
    kernels::moe_nvfp4_gemm(&s, act_p.addr(), ex.down_tabs.0.addr(), ex.down_tabs.1.addr(), ex.down.s2, offsets.addr(), te.addr(), tr.addr(), nt.addr(), max_tiles, yd.addr(), LATENT, MOE_INTER).unwrap();
    moe_combine(&s, yd.addr(), pr.addr(), wb.addr(), rows_p, outp.addr()).unwrap();
    let got_p = bf16s(&s, outp.addr(), rows_p * LATENT);
    // The same rows through the small path, 16 at a time.
    let mut got_s = Vec::new();
    for c in 0..rows_p / 16 {
        let (_, yc) = experts_small_run(&s, &ex, &xp[c * 16 * LATENT..(c + 1) * 16 * LATENT], &selp[c * 16 * TOPK..(c + 1) * 16 * TOPK], 16);
        got_s.extend(combine_emu(&yc, &swp[c * 16 * TOPK..(c + 1) * 16 * TOPK], 16));
    }
    let omax = got_s.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let (n, _) = compare(&got_p, &got_s, 1.0);
    let worst = got_p.iter().zip(&got_s).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / omax;
    eprintln!("experts prefill vs small: {n} of {} differ, max diff {worst:.2e} of max |out|", got_p.len());
    assert!(worst < 2e-2);
}

/// PDL ordering of the small-step expert chain: routing (moe_route) -> grouping -> up -> down, with many pairs per
/// expert (22 of 24 favoured experts per row, 16 rows), many steps enqueued back to back without host
/// synchronization; each step must equal the same step run alone (synchronized). A kernel that reads the
/// routing or the grouping before its dependency wait sees an earlier step's values here.
#[test]
#[ignore]
fn moe_experts_pdl_chain() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(71);
    let h_up = NvExpertsHost::random(&mut rng, TEST_EXPERTS, MOE_INTER, LATENT);
    let h_down = NvExpertsHost::random(&mut rng, TEST_EXPERTS, LATENT, MOE_INTER);
    let ex = experts_gpu(&s, &h_up, &h_down);
    let rows = 16;
    let steps = 24;
    let pairs = rows * TOPK;
    let mut bias = vec![-10.0f32; EXPERTS];
    for b in bias.iter_mut().take(24) {
        *b = 10.0;
    }
    let bb = up(&s, &bias);
    let logits: Vec<DeviceBuffer> = (0..steps).map(|_| up(&s, &rng.vec(rows * EXPERTS, -4.0, 4.0))).collect();
    let xs: Vec<DeviceBuffer> = (0..steps)
        .map(|_| up(&s, &rng.vec(rows * LATENT, -1.0, 1.0).into_iter().map(bf16::from_f32).collect::<Vec<u16>>()))
        .collect();
    let (sel_e, sel_w, order, act) = (
        DeviceBuffer::new(pairs * 4).unwrap(),
        DeviceBuffer::new(pairs * 4).unwrap(),
        DeviceBuffer::new(pairs * 4).unwrap(),
        DeviceBuffer::new(pairs * MOE_INTER * 2).unwrap(),
    );
    let ys: Vec<DeviceBuffer> = (0..steps).map(|_| DeviceBuffer::new(pairs * LATENT * 4).unwrap()).collect();
    let step = |i: usize| {
        moe_route(&s, logits[i].addr(), EXPERTS, bb.addr(), 5.0, rows, sel_e.addr(), sel_w.addr()).unwrap();
        moe_experts_small(&s, xs[i].addr(), sel_e.addr(), &ex.up, &ex.down, rows, order.addr(), act.addr(), ys[i].addr()).unwrap();
    };
    // Reference: one step at a time.
    let mut want = Vec::new();
    for i in 0..steps {
        step(i);
        s.synchronize().unwrap();
        want.push(f32s(&s, ys[i].addr(), pairs * LATENT));
    }
    for round in 0..5 {
        for y in &ys {
            s.memset(y.addr(), 0, pairs * LATENT * 4).unwrap();
        }
        for i in 0..steps {
            step(i);
        }
        s.synchronize().unwrap();
        let bad = (0..steps).filter(|&i| f32s(&s, ys[i].addr(), pairs * LATENT) != want[i]).count();
        eprintln!("experts PDL chain round {round}: {bad} of {steps} steps differ from the synchronized run");
        assert_eq!(bad, 0, "experts chain reads stale routing / grouping");
    }
}

// ---- Attention -------------------------------------------------------------------------------------

/// One attention layer's cache (one sequence, `len` positions) filled by attn_prep from random q/k/v,
/// with a second layer row in between (off = row bytes) to exercise ld/off. Returns (cache, kv, q BF16).
fn attn_fill(s: &Stream, rng: &mut Rng, len: usize, fp8: bool, kvq: &mut Vec<f32>) -> (DeviceBuffer, KvRows, DeviceBuffer) {
    let row = if fp8 { KV_ROW_BYTES_FP8 } else { KV_ROW_BYTES_BF16 };
    let kv = KvRows { off: row, ld: 2 * row, fp8 };
    let cache = DeviceBuffer::zeroed(len * kv.ld).unwrap();
    let ld = QH * HD + 2 * KVH * HD + 8;
    let qkv = rng.vec(len * ld, -2.0, 2.0);
    *kvq = qkv.clone();
    let qkvb = up(s, &qkv);
    let mut q = Seqs::default();
    q.kv[0] = cache.addr();
    q.row0[1] = len as i32;
    q.nseq = 1;
    let qb = seqs_buf(s, &q);
    let qout = DeviceBuffer::new(len * QH * HD * 2).unwrap();
    attn_prep(s, qkvb.addr(), ld, 0, QH * HD, QH * HD + KVH * HD, len, qb.addr(), kv, qout.addr()).unwrap();
    s.synchronize().unwrap();
    (cache, kv, qout)
}

/// attention_small of single rows at the given positions (each its own "sequence" over the same cache).
fn attn_decode(s: &Stream, cache: &DeviceBuffer, kv: KvRows, q_all: &[u16], positions: &[usize]) -> Vec<f32> {
    let mut q = Seqs::default();
    let mut qrows = Vec::new();
    for (i, &p) in positions.iter().enumerate() {
        q.kv[i] = cache.addr();
        q.start[i] = p as i32;
        q.row0[i] = i as i32;
        qrows.extend_from_slice(&q_all[p * QH * HD..(p + 1) * QH * HD]);
    }
    q.row0[positions.len()] = positions.len() as i32;
    q.nseq = positions.len() as i32;
    let qb = seqs_buf(s, &q);
    let qd = up(s, &qrows);
    let max_groups = positions.iter().max().unwrap() / ATTN_GROUP + 1;
    let part = DeviceBuffer::new(attn_part_floats(positions.len(), max_groups) * 4).unwrap();
    let out = DeviceBuffer::new(positions.len() * QH * HD * 2).unwrap();
    attention_small(s, qd.addr(), positions.len(), qb.addr(), kv, max_groups, part.addr(), out.addr()).unwrap();
    s.synchronize().unwrap();
    bf16s(s, out.addr(), positions.len() * QH * HD)
}

/// attention_prefill of rows r0 .. r0 + n of the sequence.
fn attn_prefill(s: &Stream, cache: &DeviceBuffer, kv: KvRows, q_all: &[u16], r0: usize, n: usize) -> Vec<f32> {
    let mut q = Seqs::default();
    q.kv[0] = cache.addr();
    q.start[0] = r0 as i32;
    q.row0[1] = n as i32;
    q.nseq = 1;
    let qb = seqs_buf(s, &q);
    let qd = up(s, &q_all[r0 * QH * HD..(r0 + n) * QH * HD]);
    let out = DeviceBuffer::new(n * QH * HD * 2).unwrap();
    attention_prefill(s, qd.addr(), n, qb.addr(), kv, out.addr()).unwrap();
    s.synchronize().unwrap();
    bf16s(s, out.addr(), n * QH * HD)
}

#[test]
#[ignore]
fn attention_vs_reference() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(31);
    let len = 2100;
    for fp8 in [false, true] {
        let mut qkv = Vec::new();
        let (cache, kv, qout) = attn_fill(&s, &mut rng, len, fp8, &mut qkv);
        let ld = QH * HD + 2 * KVH * HD + 8;
        // Prep: q and cache rows bitwise against the emulation.
        let q_all: Vec<u16> = s.download(qout.addr(), len * QH * HD).unwrap();
        let bytes: Vec<u8> = s.download(cache.addr(), len * kv.ld).unwrap();
        let row_b = if fp8 { KV_ROW_BYTES_FP8 } else { KV_ROW_BYTES_BF16 };
        let mut rows = Vec::new();
        for p in 0..len {
            let r = &qkv[p * ld..(p + 1) * ld];
            for i in 0..QH * HD {
                assert_eq!(q_all[p * QH * HD + i], bf16::from_f32(r[i]), "q");
            }
            let want = kv_row_emu(&r[QH * HD..QH * HD + KVH * HD], &r[QH * HD + KVH * HD..QH * HD + 2 * KVH * HD], fp8);
            let got = &bytes[p * kv.ld + kv.off..p * kv.ld + kv.off + row_b];
            assert_eq!(got, &want[..], "fp8 {fp8} cache row {p}");
            rows.push(got.to_vec());
        }
        // Prefill of all rows, prefill in two parts, decode of selected rows: bitwise equal.
        let full = attn_prefill(&s, &cache, kv, &q_all, 0, len);
        let mut parts = attn_prefill(&s, &cache, kv, &q_all, 0, 1029);
        parts.extend(attn_prefill(&s, &cache, kv, &q_all, 1029, len - 1029));
        assert!(full == parts, "fp8 {fp8}: prefill depends on the split");
        let positions = [0usize, 1, 63, 64, 1023, 1024, 1500, 2047, 2048, 2099];
        let dec = attn_decode(&s, &cache, kv, &q_all, &positions);
        for (i, &p) in positions.iter().enumerate() {
            assert_eq!(&dec[i * QH * HD..(i + 1) * QH * HD], &full[p * QH * HD..(p + 1) * QH * HD], "fp8 {fp8}: decode vs prefill at {p}");
        }
        // Batch invariance of decode: one row alone.
        let one = attn_decode(&s, &cache, kv, &q_all, &[1500]);
        assert_eq!(&one[..], &dec[6 * QH * HD..7 * QH * HD], "fp8 {fp8}: decode row alone vs batched");
        // Against exact attention.
        let mut worst = 0.0f64;
        for &p in &positions {
            let qf: Vec<f32> = q_all[p * QH * HD..(p + 1) * QH * HD].iter().map(|&v| bf16::to_f32(v)).collect();
            let want = attention_f64(&qf, &rows, p, fp8);
            let m = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            for i in 0..QH * HD {
                worst = worst.max((full[p * QH * HD + i] as f64 - want[i]).abs() / m);
            }
        }
        eprintln!("attention fp8 {fp8}: prep bitwise, prefill split / decode / batch bitwise; max diff vs FP64 {worst:.2e} of max |out|");
        assert!(worst < 2e-2);
    }
}

/// A prefill launch whose CTAs mix two sequences equals one launch per sequence.
#[test]
#[ignore]
fn attention_prefill_mixed_sequences() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(32);
    let mut qkv = Vec::new();
    let (cache_a, kv, qa) = attn_fill(&s, &mut rng, 70, false, &mut qkv);
    let (cache_b, _, qb_) = attn_fill(&s, &mut rng, 70, false, &mut qkv);
    let qa: Vec<u16> = s.download(qa.addr(), 70 * QH * HD).unwrap();
    let qbv: Vec<u16> = s.download(qb_.addr(), 70 * QH * HD).unwrap();
    // Sequence a rows 60..65 (5 rows), sequence b rows 40..47 (7 rows) in one launch.
    let mut q = Seqs::default();
    q.kv[0] = cache_a.addr();
    q.kv[1] = cache_b.addr();
    q.start[0] = 60;
    q.start[1] = 40;
    q.row0[1] = 5;
    q.row0[2] = 12;
    q.nseq = 2;
    let qd = seqs_buf(&s, &q);
    let mut rowsq = qa[60 * QH * HD..65 * QH * HD].to_vec();
    rowsq.extend_from_slice(&qbv[40 * QH * HD..47 * QH * HD]);
    let qbuf = up(&s, &rowsq);
    let out = DeviceBuffer::new(12 * QH * HD * 2).unwrap();
    attention_prefill(&s, qbuf.addr(), 12, qd.addr(), kv, out.addr()).unwrap();
    let mixed = bf16s(&s, out.addr(), 12 * QH * HD);
    let a = attn_prefill(&s, &cache_a, kv, &qa, 60, 5);
    let b = attn_prefill(&s, &cache_b, kv, &qbv, 40, 7);
    assert_eq!(&mixed[..5 * QH * HD], &a[..]);
    assert_eq!(&mixed[5 * QH * HD..], &b[..]);
    eprintln!("prefill with mixed sequences: bitwise equal to separate launches");
}

#[test]
#[ignore]
fn layout() {
    check_layout().unwrap();
}

// ---- Timing (decode-sized steps, inside CUDA graphs, working sets beyond L2) ------------------------

fn time_graph(s: &Stream, body: impl FnOnce(&Stream) -> anyhow::Result<()>, calls: usize) -> f64 {
    use quench6k_cuda::{Event, Graph};
    let g = Graph::capture(s, body).unwrap();
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

/// Fast random bytes (8 per generator step).
fn rand_bytes(rng: &mut Rng, n: usize, f: impl Fn(u8) -> u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(n + 8);
    while v.len() < n {
        v.extend(rng.next_u64().to_le_bytes().map(&f));
    }
    v.truncate(n);
    v
}

fn experts_dev(s: &Stream, rng: &mut Rng, e: usize, n: usize, k: usize, keep: &mut Vec<DeviceBuffer>) -> (NvExperts, u64, u64) {
    let codes = up(s, &rand_bytes(rng, e * n * k / 2, |b| b));
    let scales = up(s, &rand_bytes(rng, e * n * k / 16, |b| 0x28 + (b & 15)));
    let s2 = up(s, &rng.vec(e, 0.005, 0.02));
    let ct: Vec<u64> = (0..EXPERTS).map(|i| codes.addr() + (i.min(e - 1) * n * k / 2) as u64).collect();
    let st: Vec<u64> = (0..EXPERTS).map(|i| scales.addr() + (i.min(e - 1) * n * k / 16) as u64).collect();
    let (ctb, stb) = (up(s, &ct), up(s, &st));
    let nv = NvExperts { codes: codes.addr(), scales: scales.addr(), s2: s2.addr() };
    let r = (nv, ctb.addr(), stb.addr());
    keep.extend([codes, scales, s2, ctb, stb]);
    r
}

fn rand_sel(rng: &mut Rng, rows: usize, experts: usize) -> Vec<i32> {
    let mut v = Vec::new();
    for _ in 0..rows {
        let mut pick: Vec<i32> = (0..experts as i32).collect();
        for i in (1..pick.len()).rev() {
            pick.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
        }
        let mut sel = pick[..TOPK].to_vec();
        sel.sort();
        v.extend(sel);
    }
    v
}

#[test]
#[ignore]
fn time_mamba() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(41);
    let p = MambaParams::random(&mut rng);
    let w = mamba_weights(&s, &p);
    // Decode: one row per sequence, 16 layers' states per sequence.
    const TL: usize = 16;
    for nseq in [1usize, 8] {
        let blocks: Vec<DeviceBuffer> = (0..nseq).map(|_| DeviceBuffer::zeroed(mamba_seq_bytes(TL, REC)).unwrap()).collect();
        let mut q = Seqs::default();
        for (i, b) in blocks.iter().enumerate() {
            q.mamba[i] = b.addr();
            q.row0[i + 1] = i as i32 + 1;
        }
        q.nseq = nseq as i32;
        let qb = seqs_buf(&s, &q);
        let proj = up(&s, &random_proj(&mut rng, nseq));
        let (xbc, dt, y, out) = (
            DeviceBuffer::new(nseq * CONV_DIM * 2).unwrap(),
            DeviceBuffer::new(nseq * HEADS * 4).unwrap(),
            DeviceBuffer::new(nseq * INNER * 2).unwrap(),
            DeviceBuffer::new(nseq * INNER * 2).unwrap(),
        );
        let sc = MambaScratch { xbc: xbc.addr(), dt: dt.addr(), y: y.addr() };
        let us = time_graph(
            &s,
            |st| {
                for l in 0..TL {
                    mamba_mixer(st, proj.addr(), PROJ, nseq, qb.addr(), nseq, l, REC, &w.w, false, &sc, out.addr())?;
                }
                Ok(())
            },
            TL,
        );
        let bytes = (2 * SSM_BYTES * nseq) as f64;
        eprintln!("TIME mamba decode {nseq} seq: {us:.1} us/layer ({:.0} GB/s of state r+w)", bytes / us / 1e3);
    }
    // Prefill: 4096 rows of one sequence (each kernel separately and the mixer).
    let rows = 4096;
    let block = new_block();
    let mut q = Seqs::default();
    q.mamba[0] = block.addr();
    q.row0[1] = rows as i32;
    q.nseq = 1;
    let qb = seqs_buf(&s, &q);
    let proj = up(&s, &random_proj(&mut rng, rows));
    let (xbc, dt, y, out) = (
        DeviceBuffer::new(rows * CONV_DIM * 2).unwrap(),
        DeviceBuffer::new(rows * HEADS * 4).unwrap(),
        DeviceBuffer::new(rows * INNER * 2).unwrap(),
        DeviceBuffer::new(rows * INNER * 2).unwrap(),
    );
    let sc = MambaScratch { xbc: xbc.addr(), dt: dt.addr(), y: y.addr() };
    let t_conv = time_graph(&s, |st| mamba_conv(st, proj.addr(), PROJ, rows, qb.addr(), 1, LAYER, REC, &w.w, xbc.addr(), dt.addr()), 1);
    let t_commit = time_graph(&s, |st| mamba_conv_commit(st, proj.addr(), PROJ, rows, qb.addr(), 1, LAYER, REC, xbc.addr(), dt.addr(), false), 1);
    let t_scan = time_graph(&s, |st| mamba_scan(st, xbc.addr(), dt.addr(), rows, qb.addr(), 1, LAYER, REC, &w.w, false, y.addr()), 1);
    let t_norm = time_graph(&s, |st| mamba_gated_norm(st, y.addr(), proj.addr(), PROJ, rows, w.w.norm_w, 1e-5, out.addr()), 1);
    let t_all = time_graph(&s, |st| mamba_mixer(st, proj.addr(), PROJ, rows, qb.addr(), 1, LAYER, REC, &w.w, false, &sc, out.addr()), 1);
    eprintln!(
        "TIME mamba prefill {rows} rows: conv {t_conv:.0} commit {t_commit:.0} scan {t_scan:.0} norm {t_norm:.0} mixer {t_all:.0} us ({:.0} tok/s per layer)",
        rows as f64 / t_all * 1e6
    );
}

#[test]
#[ignore]
fn time_experts() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(42);
    let mut keep = Vec::new();
    // All 512 experts (3.2 GB of up + down weights), 16 routings cycled per graph.
    let (upn, upc, ups) = experts_dev(&s, &mut rng, EXPERTS, MOE_INTER, LATENT, &mut keep);
    let (downn, downc, downs) = experts_dev(&s, &mut rng, EXPERTS, LATENT, MOE_INTER, &mut keep);
    for rows in [1usize, 2, 4, 8, 16] {
        let x: Vec<u16> = rng.vec(rows * LATENT, -1.0, 1.0).into_iter().map(bf16::from_f32).collect();
        let xb = up(&s, &x);
        let sels: Vec<DeviceBuffer> = (0..16).map(|_| up(&s, &rand_sel(&mut rng, rows, EXPERTS))).collect();
        let pairs = rows * TOPK;
        let (order, act, y) = (
            DeviceBuffer::new(pairs * 4).unwrap(),
            DeviceBuffer::new(pairs * MOE_INTER * 2).unwrap(),
            DeviceBuffer::new(pairs * LATENT * 4).unwrap(),
        );
        let us = time_graph(
            &s,
            |st| {
                for sel in &sels {
                    moe_experts_small(st, xb.addr(), sel.addr(), &upn, &downn, rows, order.addr(), act.addr(), y.addr())?;
                }
                Ok(())
            },
            16,
        );
        // Distinct experts' bytes (codes + scales of up and down).
        let mut distinct = 0usize;
        for sel in &sels {
            let v: Vec<i32> = s.download(sel.addr(), pairs).unwrap();
            let mut u = v.clone();
            u.sort();
            u.dedup();
            distinct += u.len();
        }
        let bytes = distinct as f64 / 16.0 * 2.0 * (MOE_INTER * LATENT) as f64 * (0.5 + 1.0 / 16.0);
        eprintln!("TIME experts small {rows} rows: {us:.1} us/layer ({:.0} GB/s of distinct expert weights)", bytes / us / 1e3);
    }
    // Prefill path at 2048 tokens (QUENCH6K_NEM_TIME_ROWS: another row count).
    let rows: usize = std::env::var("QUENCH6K_NEM_TIME_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let pairs = rows * TOPK;
    let x: Vec<u16> = rng.vec(rows * LATENT, -1.0, 1.0).into_iter().map(bf16::from_f32).collect();
    let xb = up(&s, &x);
    let sel = up(&s, &rand_sel(&mut rng, rows, EXPERTS));
    let sw = up(&s, &rng.vec(pairs, 0.05, 0.5));
    let tile = kernels::moe_tile_rows();
    let max_tiles = pairs / tile + EXPERTS;
    let ib = |n: usize| DeviceBuffer::new(n * 4).unwrap();
    let (counts, offsets, cursor, te, tr, nt, pr) = (ib(EXPERTS), ib(EXPERTS + 1), ib(EXPERTS), ib(max_tiles), ib(max_tiles), ib(1), ib(pairs));
    let xs = DeviceBuffer::new(pairs * LATENT * 2).unwrap();
    let yu = DeviceBuffer::new(pairs * MOE_INTER * 4).unwrap();
    let act = DeviceBuffer::new(pairs * MOE_INTER * 2).unwrap();
    let yd = DeviceBuffer::new(pairs * LATENT * 4).unwrap();
    let out = DeviceBuffer::new(rows * LATENT * 2).unwrap();
    let plan = |st: &Stream| -> anyhow::Result<()> {
        moe_plan(st, sel.addr(), rows, tile, counts.addr(), offsets.addr(), cursor.addr(), te.addr(), tr.addr(), nt.addr(), pr.addr())?;
        moe_gather(st, xb.addr(), pr.addr(), rows, xs.addr())
    };
    let t_plan = time_graph(&s, plan, 1);
    let t_up = time_graph(&s, |st| kernels::moe_nvfp4_gemm(st, xs.addr(), upc, ups, upn.s2, offsets.addr(), te.addr(), tr.addr(), nt.addr(), max_tiles, yu.addr(), MOE_INTER, LATENT), 1);
    let t_relu = time_graph(&s, |st| relu2(st, yu.addr(), MOE_INTER, MOE_INTER, pairs, act.addr(), MOE_INTER), 1);
    let t_down = time_graph(&s, |st| kernels::moe_nvfp4_gemm(st, act.addr(), downc, downs, downn.s2, offsets.addr(), te.addr(), tr.addr(), nt.addr(), max_tiles, yd.addr(), LATENT, MOE_INTER), 1);
    let t_comb = time_graph(&s, |st| moe_combine(st, yd.addr(), pr.addr(), sw.addr(), rows, out.addr()), 1);
    let flops = 2.0 * 2.0 * (pairs * MOE_INTER * LATENT) as f64;
    // The fused path (its own tile rows).
    let tile2 = moe_prefill_tile_rows();
    let max2 = pairs / tile2 + EXPERTS;
    let (te2, tr2) = (ib(max2), ib(max2));
    let yb = DeviceBuffer::new(pairs * LATENT * 2).unwrap();
    let plan2 = |st: &Stream| -> anyhow::Result<()> {
        moe_plan(st, sel.addr(), rows, tile2, counts.addr(), offsets.addr(), cursor.addr(), te2.addr(), tr2.addr(), nt.addr(), pr.addr())?;
        moe_gather(st, xb.addr(), pr.addr(), rows, xs.addr())
    };
    let t_plan2 = time_graph(&s, plan2, 1);
    let t_gemms = time_graph(&s, |st| moe_experts_prefill(st, xs.addr(), &upn, &downn, offsets.addr(), te2.addr(), tr2.addr(), nt.addr(), max2, act.addr(), yb.addr()), 1);
    let t_comb2 = time_graph(&s, |st| moe_combine_bf16(st, yb.addr(), pr.addr(), sw.addr(), rows, out.addr()), 1);
    // Thin 32-row tiles.
    let max3 = pairs / 32 + EXPERTS;
    let (te3, tr3) = (ib(max3), ib(max3));
    moe_plan(&s, sel.addr(), rows, 32, counts.addr(), offsets.addr(), cursor.addr(), te3.addr(), tr3.addr(), nt.addr(), pr.addr()).unwrap();
    let t_thin = time_graph(&s, |st| moe_experts_prefill_tiles(st, xs.addr(), &upn, &downn, offsets.addr(), te3.addr(), tr3.addr(), nt.addr(), max3, 32, act.addr(), yb.addr()), 1);
    eprintln!("TIME experts prefill thin {rows} tokens (tile 32): gemms {t_thin:.0} us/layer ({:.0} TFLOPS)", flops / t_thin / 1e6);
    let _ = time_graph(&s, plan2, 1);
    eprintln!(
        "TIME experts prefill fused {rows} tokens (tile {tile2}): plan+gather {t_plan2:.0} gemms {t_gemms:.0} combine {t_comb2:.0} = {:.0} us/layer ({:.0} TFLOPS in the GEMMs)",
        t_plan2 + t_gemms + t_comb2,
        flops / t_gemms / 1e6
    );
    // The unfused path with the shared GEMM (re-plan with its tile rows).
    let _ = time_graph(&s, plan, 1);
    let total = t_plan + t_up + t_relu + t_down + t_comb;
    eprintln!(
        "TIME experts prefill {rows} tokens: plan+gather {t_plan:.0} up {t_up:.0} relu2 {t_relu:.0} down {t_down:.0} combine {t_comb:.0} = {total:.0} us/layer ({:.0} TFLOPS in the GEMMs)",
        flops / (t_up + t_down) / 1e6
    );
}

#[test]
#[ignore]
fn time_attention() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(43);
    let ctx = 32768usize;
    for fp8 in [false, true] {
        let row = if fp8 { KV_ROW_BYTES_FP8 } else { KV_ROW_BYTES_BF16 };
        let kv = KvRows { off: 0, ld: row, fp8 };
        let caches: Vec<DeviceBuffer> = (0..8).map(|_| DeviceBuffer::zeroed(ctx * row).unwrap()).collect();
        let mut q = Seqs::default();
        for (i, c) in caches.iter().enumerate() {
            q.kv[i] = c.addr();
            q.start[i] = (ctx - 1) as i32;
            q.row0[i + 1] = i as i32 + 1;
        }
        q.nseq = 8;
        let qb = seqs_buf(&s, &q);
        let qv: Vec<u16> = rng.vec(8 * QH * HD, -1.0, 1.0).into_iter().map(bf16::from_f32).collect();
        let qd = up(&s, &qv);
        let groups = ctx / ATTN_GROUP;
        let part = DeviceBuffer::new(attn_part_floats(8, groups) * 4).unwrap();
        let out = DeviceBuffer::new(8 * QH * HD * 2).unwrap();
        let us = time_graph(&s, |st| attention_small(st, qd.addr(), 8, qb.addr(), kv, groups, part.addr(), out.addr()), 1);
        eprintln!("TIME attention decode fp8 {fp8}, 8 seq at {ctx}: {us:.1} us ({:.0} GB/s of cache)", (8 * ctx * row) as f64 / us / 1e3);
        // Prefill: 4096 rows of the first cache.
        let rows = 4096;
        let mut q1 = Seqs::default();
        q1.kv[0] = caches[0].addr();
        q1.row0[1] = rows as i32;
        q1.nseq = 1;
        let qb1 = seqs_buf(&s, &q1);
        let qp: Vec<u16> = rng.vec(rows * QH * HD, -1.0, 1.0).into_iter().map(bf16::from_f32).collect();
        let qpd = up(&s, &qp);
        let outp = DeviceBuffer::new(rows * QH * HD * 2).unwrap();
        let us = time_graph(&s, |st| attention_prefill(st, qpd.addr(), rows, qb1.addr(), kv, outp.addr()), 1);
        let flops = 4.0 * (rows * rows / 2 * QH * HD) as f64;
        eprintln!("TIME attention prefill fp8 {fp8}, {rows} rows: {us:.0} us ({:.0} TFLOPS)", flops / us / 1e6);
    }
}

/// Decode attention in the engine's cache layout (9 layers' rows per position): 1 and 8 sequences at 8K and
/// 32K positions, by L2 prefetch distance.
#[test]
#[ignore]
fn time_attention_decode() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(44);
    const LAYERS_PER_ROW: usize = 9;
    for fp8 in [false, true] {
        let row = if fp8 { KV_ROW_BYTES_FP8 } else { KV_ROW_BYTES_BF16 };
        let kv = KvRows { off: 4 * row, ld: LAYERS_PER_ROW * row, fp8 };
        let ctx_max = 32768usize;
        // Random bytes (FP8: no NaN codes; BF16: some NaNs, the speed is the same), one 64 MiB piece copied over.
        let piece = up(&s, &rand_bytes(&mut rng, 64 << 20, |b| if fp8 { b & 0x77 } else { b & 0x7f }));
        let bytes = ctx_max * LAYERS_PER_ROW * row;
        let caches: Vec<DeviceBuffer> = (0..8)
            .map(|_| {
                let c = DeviceBuffer::new(bytes).unwrap();
                for o in (0..bytes).step_by(64 << 20) {
                    s.copy_d2d(c.addr() + o as u64, piece.addr(), (64usize << 20).min(bytes - o)).unwrap();
                }
                c
            })
            .collect();
        for nseq in [1usize, 8] {
            for ctx in [8192usize, 32768] {
                let mut q = Seqs::default();
                for (i, c) in caches.iter().take(nseq).enumerate() {
                    q.kv[i] = c.addr();
                    q.start[i] = (ctx - 1) as i32;
                    q.row0[i + 1] = i as i32 + 1;
                }
                q.nseq = nseq as i32;
                let qb = seqs_buf(&s, &q);
                let qv: Vec<u16> = rng.vec(nseq * QH * HD, -1.0, 1.0).into_iter().map(bf16::from_f32).collect();
                let qd = up(&s, &qv);
                let groups = ctx / ATTN_GROUP;
                let part = DeviceBuffer::new(attn_part_floats(nseq, groups) * 4).unwrap();
                let out = DeviceBuffer::new(nseq * QH * HD * 2).unwrap();
                let mut line = format!("TIME attention decode fp8 {fp8}, {nseq} seq at {ctx}:");
                for pf in [0usize, 1, 2, 3, 4, 6] {
                    set_attn_prefetch(pf);
                    let us = time_graph(&s, |st| attention_small(st, qd.addr(), nseq, qb.addr(), kv, groups, part.addr(), out.addr()), 1);
                    line += &format!(" pf {pf} {us:.1} us ({:.0} GB/s);", (nseq * ctx * 2 * KVH * HD * if fp8 { 1 } else { 2 }) as f64 / us / 1e3);
                }
                set_attn_prefetch(3);
                eprintln!("{line}");
            }
        }
    }
}

/// The engine's projection shapes (rows x columns, stored format) through the shared GEMV and every wide
/// form, at 1, 2, 4 and 8 input rows, weights rotated through 1 GiB (beyond L2).
#[test]
#[ignore]
fn time_gemv_shapes() {
    use kernels::WeightFormat as F;
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(45);
    const POOL: usize = 1 << 30;
    let pool = DeviceBuffer::new(POOL).unwrap();
    let piece = up(&s, &rand_bytes(&mut rng, 64 << 20, |b| b & 0xF7));
    for o in (0..POOL).step_by(64 << 20) {
        s.copy_d2d(pool.addr() + o as u64, piece.addr(), 64 << 20).unwrap();
    }
    const SPOOL: usize = 128 << 20;
    let spool = up(&s, &rng.vec(SPOOL / 4, 0.001, 0.02));
    let x: Vec<u16> = rng.vec(64 * 8192, -1.0, 1.0).into_iter().map(bf16::from_f32).collect();
    let xb = up(&s, &x);
    let y = DeviceBuffer::new(64 * 131072 * 4).unwrap();
    let shapes: [(&str, usize, usize, F); 10] = [
        ("in_proj", 18560, 4096, F::Fp8Block),
        ("out_proj", 4096, 8192, F::Fp8Block),
        ("qkv", 4608, 4096, F::Int8Group),
        ("o_proj", 4096, 4096, F::Int8Group),
        ("shared_up", 5376, 4096, F::Fp8Block),
        ("shared_down", 4096, 5376, F::Fp8Block),
        ("gate", 512, 4096, F::Bf16),
        ("fc1", 1024, 4096, F::Int8Group),
        ("fc2", 4096, 1024, F::Int8Group),
        ("lm_head", 131072, 4096, F::Int8Group),
    ];
    let forms = [(16usize, 4usize), (16, 2), (8, 4), (8, 2), (8, 1), (4, 4), (4, 2), (4, 1), (2, 4), (2, 2)];
    for (name, n, k, fmt) in shapes {
        let wb = n * k * if fmt == F::Bf16 { 2 } else { 1 };
        let sb = match fmt {
            F::Fp8Block => n.div_ceil(128) * (k / 128) * 4,
            F::Int8Group => n * (k / kernels::int8_group()) * 4,
            _ => 0,
        };
        let calls = (POOL / 2 / wb).clamp(4, 64);
        let wa = |j: usize| pool.addr() + ((j * wb) % (POOL - wb) / 256 * 256) as u64;
        let sa = |j: usize| if sb == 0 { 0 } else { spool.addr() + ((j * sb) % (SPOOL - sb) / 256 * 256) as u64 };
        // QUENCH6K_TEST_MS: the row counts (comma-separated; default 1,2,4,8).
        let ms: Vec<usize> = std::env::var("QUENCH6K_TEST_MS").map_or(vec![1, 2, 4, 8], |v| v.split(',').map(|x| x.trim().parse().unwrap()).collect());
        for &m in &ms {
            let us = time_graph(
                &s,
                |st| {
                    for j in 0..calls {
                        for r0 in (0..m).step_by(16) {
                            let rows = 16.min(m - r0);
                            kernels::gemv(st, rows, xb.addr() + (r0 * k * 2) as u64, k, fmt, wa(j), sa(j), y.addr() + (r0 * n * 4) as u64, n, 0, n, k, 1, 1.0, false)?;
                        }
                    }
                    Ok(())
                },
                calls,
            );
            let mut line = format!("TIME gemv {name} {n}x{k} {fmt:?} m {m}: shared {us:.1} us ({:.0} GB/s)", (wb + sb) as f64 / us / 1e3);
            for &(mt, r) in &forms {
                if mt > 2 * m.max(1) || m < 2 || (fmt == F::Bf16 && mt * r > 32) {
                    continue;
                }
                let us = time_graph(
                    &s,
                    |st| {
                        for j in 0..calls {
                            gemv_wide(st, m, mt, r, xb.addr(), k, fmt, wa(j), sa(j), y.addr(), n, n, k, 1.0, false)?;
                        }
                        Ok(())
                    },
                    calls,
                );
                line += &format!("; {mt}x{r} {us:.1} ({:.0})", (wb + sb) as f64 / us / 1e3);
            }
            eprintln!("{line}");
        }
    }
}

// ---- Output digest (bit-identity of optimizations) ---------------------------------------------------

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

fn digest_of<T: Copy>(v: &[T]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
    fnv(&mut h, b);
    h
}

/// Prints one hash per kernel family over fixed inputs; compare before / after a speed change.
#[test]
#[ignore]
fn digest() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(61);
    // Mamba: prefill of 700 rows (several staging chunks), decode steps, a deferred step, a replay.
    let p = MambaParams::random(&mut rng);
    let w = mamba_weights(&s, &p);
    let (a, b) = (new_block(), new_block());
    let mut outs = Vec::new();
    let pa = random_proj(&mut rng, 700);
    let pb = random_proj(&mut rng, 3);
    outs.extend(mamba_step(&s, &w, &[&a, &b], &[&pa, &pb], false).concat());
    for _ in 0..3 {
        let (x, y) = (random_proj(&mut rng, 1), random_proj(&mut rng, 2));
        outs.extend(mamba_step(&s, &w, &[&a, &b], &[&x, &y], false).concat());
    }
    let (x, y) = (random_proj(&mut rng, 4), random_proj(&mut rng, 5));
    outs.extend(mamba_step(&s, &w, &[&a, &b], &[&x, &y], true).concat());
    set_ctl(&s, &[&a, &b], &[4, 5], &[2, 5], true);
    let (x, y) = (random_proj(&mut rng, 1), random_proj(&mut rng, 1));
    outs.extend(mamba_step(&s, &w, &[&a, &b], &[&x, &y], false).concat());
    let mut st = ssm_of(&s, &a);
    st.extend(ssm_of(&s, &b));
    st.extend(conv_of(&s, &a));
    eprintln!("digest mamba out {:016x} state {:016x}", digest_of(&outs), digest_of(&st));

    // Routed experts, small path at 1, 8, 16 rows, and the prefill path at 300 rows (over 40 experts).
    let h_up = NvExpertsHost::random(&mut rng, TEST_EXPERTS, MOE_INTER, LATENT);
    let h_down = NvExpertsHost::random(&mut rng, TEST_EXPERTS, LATENT, MOE_INTER);
    let ex = experts_gpu(&s, &h_up, &h_down);
    for rows in [1usize, 8, 16] {
        let x: Vec<u16> = rng.vec(rows * LATENT, -1.0, 1.0).into_iter().map(bf16::from_f32).collect();
        let (sel, _) = routing(&mut rng, rows);
        let (act, y) = experts_small_run(&s, &ex, &x, &sel, rows);
        eprintln!("digest experts small {rows} rows: act {:016x} y {:016x}", digest_of(&act), digest_of(&y));
    }
    let rows_p = 300;
    let xp: Vec<u16> = rng.vec(rows_p * LATENT, -1.0, 1.0).into_iter().map(bf16::from_f32).collect();
    let (selp, swp) = routing(&mut rng, rows_p);
    let pairs = rows_p * TOPK;
    let tile = kernels::moe_tile_rows();
    let max_tiles = pairs / tile + EXPERTS;
    let ib = |n: usize| DeviceBuffer::new(n * 4).unwrap();
    let (counts, offsets, cursor, te, tr, nt, pr) = (ib(EXPERTS), ib(EXPERTS + 1), ib(EXPERTS), ib(max_tiles), ib(max_tiles), ib(1), ib(pairs));
    let (xb, eb, wb) = (up(&s, &xp), up(&s, &selp), up(&s, &swp));
    let xs = DeviceBuffer::new(pairs * LATENT * 2).unwrap();
    let yu = DeviceBuffer::new(pairs * MOE_INTER * 4).unwrap();
    let act_p = DeviceBuffer::new(pairs * MOE_INTER * 2).unwrap();
    let yd = DeviceBuffer::new(pairs * LATENT * 4).unwrap();
    let outp = DeviceBuffer::new(rows_p * LATENT * 2).unwrap();
    moe_plan(&s, eb.addr(), rows_p, tile, counts.addr(), offsets.addr(), cursor.addr(), te.addr(), tr.addr(), nt.addr(), pr.addr()).unwrap();
    moe_gather(&s, xb.addr(), pr.addr(), rows_p, xs.addr()).unwrap();
    kernels::moe_nvfp4_gemm(&s, xs.addr(), ex.up_tabs.0.addr(), ex.up_tabs.1.addr(), ex.up.s2, offsets.addr(), te.addr(), tr.addr(), nt.addr(), max_tiles, yu.addr(), MOE_INTER, LATENT).unwrap();
    relu2(&s, yu.addr(), MOE_INTER, MOE_INTER, pairs, act_p.addr(), MOE_INTER).unwrap();
    kernels::moe_nvfp4_gemm(&s, act_p.addr(), ex.down_tabs.0.addr(), ex.down_tabs.1.addr(), ex.down.s2, offsets.addr(), te.addr(), tr.addr(), nt.addr(), max_tiles, yd.addr(), LATENT, MOE_INTER).unwrap();
    moe_combine(&s, yd.addr(), pr.addr(), wb.addr(), rows_p, outp.addr()).unwrap();
    eprintln!("digest experts prefill out {:016x}", digest_of(&bf16s(&s, outp.addr(), rows_p * LATENT)));
    // The fused prefill path: the same output bits.
    let tile2 = moe_prefill_tile_rows();
    let max2 = pairs / tile2 + EXPERTS;
    let (te2, tr2) = (ib(max2), ib(max2));
    let yb = DeviceBuffer::new(pairs * LATENT * 2).unwrap();
    moe_plan(&s, eb.addr(), rows_p, tile2, counts.addr(), offsets.addr(), cursor.addr(), te2.addr(), tr2.addr(), nt.addr(), pr.addr()).unwrap();
    moe_gather(&s, xb.addr(), pr.addr(), rows_p, xs.addr()).unwrap();
    moe_experts_prefill(&s, xs.addr(), &ex.up, &ex.down, offsets.addr(), te2.addr(), tr2.addr(), nt.addr(), max2, act_p.addr(), yb.addr()).unwrap();
    moe_combine_bf16(&s, yb.addr(), pr.addr(), wb.addr(), rows_p, outp.addr()).unwrap();
    eprintln!("digest experts prefill (fused) out {:016x}", digest_of(&bf16s(&s, outp.addr(), rows_p * LATENT)));
    // Thin 32-row tiles (small steps): the same bits.
    let max3 = pairs / 32 + EXPERTS;
    let (te3, tr3) = (ib(max3), ib(max3));
    moe_plan(&s, eb.addr(), rows_p, 32, counts.addr(), offsets.addr(), cursor.addr(), te3.addr(), tr3.addr(), nt.addr(), pr.addr()).unwrap();
    moe_gather(&s, xb.addr(), pr.addr(), rows_p, xs.addr()).unwrap();
    moe_experts_prefill_tiles(&s, xs.addr(), &ex.up, &ex.down, offsets.addr(), te3.addr(), tr3.addr(), nt.addr(), max3, 32, act_p.addr(), yb.addr()).unwrap();
    moe_combine_bf16(&s, yb.addr(), pr.addr(), wb.addr(), rows_p, outp.addr()).unwrap();
    eprintln!("digest experts prefill (thin) out {:016x}", digest_of(&bf16s(&s, outp.addr(), rows_p * LATENT)));

    // Attention, both formats: prefill of 2100 rows and decode rows.
    for fp8 in [false, true] {
        let mut qkv = Vec::new();
        let (cache, kv, qout) = attn_fill(&s, &mut rng, 2100, fp8, &mut qkv);
        let q_all: Vec<u16> = s.download(qout.addr(), 2100 * QH * HD).unwrap();
        let full = attn_prefill(&s, &cache, kv, &q_all, 0, 2100);
        let dec = attn_decode(&s, &cache, kv, &q_all, &[5, 1024, 2099]);
        let bytes: Vec<u8> = s.download(cache.addr(), 2100 * kv.ld).unwrap();
        eprintln!("digest attention fp8 {fp8}: cache {:016x} prefill {:016x} decode {:016x}", digest_of(&bytes), digest_of(&full), digest_of(&dec));
    }
}

/// Prefill projections: the FP8-block / INT8-group GEMM against expanding the weights to BF16 and running
/// the BF16 GEMM (bitwise equal outputs; times per call, M = QUENCH6K_NEM_TIME_ROWS or 2048 and 6144).
#[test]
#[ignore]
fn time_dense() {
    use quench6k_cuda::kernels::{self as kk, WeightFormat};
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(43);
    let ms: Vec<usize> = match std::env::var("QUENCH6K_NEM_TIME_ROWS").ok().and_then(|v| v.parse().ok()) {
        Some(m) => vec![m],
        None => vec![2048, 6144],
    };
    for &(n, k) in &[(18560usize, 4096usize), (4096, 8192), (5376, 4096), (4096, 5376), (4096, 1024)] {
        for fmt in [WeightFormat::Fp8Block, WeightFormat::Int8Group] {
            // E4M3 codes without NaN (0x7f / 0xff); INT8 any byte.
            let codes = up(&s, &rand_bytes(&mut rng, n * k, |b| if fmt == WeightFormat::Fp8Block && b & 0x7f == 0x7f { b & 0xf7 } else { b }));
            let nsc = if fmt == WeightFormat::Fp8Block { n.div_ceil(128) * (k / 128) } else { n * (k / kk::int8_group()) };
            let sc = up(&s, &rng.vec(nsc, 0.001, 0.01));
            let wb = DeviceBuffer::new(n * k * 2).unwrap();
            for &m in &ms {
                let x: Vec<u16> = rng.vec(m * k, -1.0, 1.0).into_iter().map(bf16::from_f32).collect();
                let xb = up(&s, &x);
                let (ya, yb) = (DeviceBuffer::new(m * n * 4).unwrap(), DeviceBuffer::new(m * n * 4).unwrap());
                kk::gemm_f32(&s, xb.addr(), k, fmt, codes.addr(), k, sc.addr(), ya.addr(), n, m, n, k, 0.5, false).unwrap();
                expand_bf16(&s, fmt, codes.addr(), sc.addr(), n, k, wb.addr()).unwrap();
                kk::gemm_f32(&s, xb.addr(), k, WeightFormat::Bf16, wb.addr(), k, 0, yb.addr(), n, m, n, k, 0.5, false).unwrap();
                let (a, b): (Vec<u32>, Vec<u32>) = (s.download(ya.addr(), m * n).unwrap(), s.download(yb.addr(), m * n).unwrap());
                assert!(a == b, "{fmt:?} M={m} N={n} K={k}: expanded BF16 GEMM differs");
                let t8 = time_graph(&s, |st| kk::gemm_f32(st, xb.addr(), k, fmt, codes.addr(), k, sc.addr(), ya.addr(), n, m, n, k, 0.5, false), 1);
                let te = time_graph(&s, |st| expand_bf16(st, fmt, codes.addr(), sc.addr(), n, k, wb.addr()), 1);
                let tb = time_graph(&s, |st| kk::gemm_f32(st, xb.addr(), k, WeightFormat::Bf16, wb.addr(), k, 0, yb.addr(), n, m, n, k, 0.5, false), 1);
                let tf = |us: f64| 2.0 * (m * n * k) as f64 / us / 1e6;
                eprintln!(
                    "TIME dense {fmt:?} M={m} N={n} K={k}: 8-bit {t8:.0} us ({:.0} TFLOPS), expand {te:.0} + BF16 {tb:.0} ({:.0} TFLOPS) = {:.0} us; bitwise equal",
                    tf(t8),
                    tf(tb),
                    te + tb
                );
            }
        }
    }
}

/// The small steps' alternative launches give the bits of the ones they replace: the router by sorting
/// (ties, NaN logits), the wide GEMV (every form and format, output factor, accumulation, 2 to 64 rows),
/// the Mamba convolution and its commit in one launch (plain, deferred and replaying steps of several
/// sequences), the attention combine per query head (and decode rows against prefill rows).
#[test]
#[ignore]
fn decode_forms_equal() {
    let s = Stream::new().unwrap();
    let mut rng = Rng::new(77);
    // Router: random rows, rows of few distinct values (ties), a row with NaN logits.
    let rows = 40;
    let mut logits = rng.vec(rows * EXPERTS, -5.0, 5.0);
    for r in 20..36 {
        for e in 0..EXPERTS {
            logits[r * EXPERTS + e] = (logits[r * EXPERTS + e] * 2.0).round() / 2.0;
        }
    }
    for e in (0..EXPERTS).step_by(3) {
        logits[37 * EXPERTS + e] = f32::NAN;
    }
    let mut bias = rng.vec(EXPERTS, -0.05, 0.05);
    for b in bias.iter_mut().skip(100).take(200) {
        *b = 0.0;
    }
    let (lb, bb) = (up(&s, &logits), up(&s, &bias));
    let buf = || DeviceBuffer::new(rows * TOPK * 4).unwrap();
    let (e1, w1, e2, w2) = (buf(), buf(), buf(), buf());
    moe_route(&s, lb.addr(), EXPERTS, bb.addr(), 5.0, rows, e1.addr(), w1.addr()).unwrap();
    moe_route_sort(&s, lb.addr(), EXPERTS, bb.addr(), 5.0, rows, e2.addr(), w2.addr()).unwrap();
    assert_eq!(s.download::<i32>(e1.addr(), rows * TOPK).unwrap(), s.download::<i32>(e2.addr(), rows * TOPK).unwrap(), "router experts");
    assert_eq!(s.download::<u32>(w1.addr(), rows * TOPK).unwrap(), s.download::<u32>(w2.addr(), rows * TOPK).unwrap(), "router weights");
    eprintln!("router by sorting: equal");

    // Wide GEMV against the shared GEMV.
    use kernels::WeightFormat as F;
    let (n, k) = (1000usize, 1536usize);
    let wbf: Vec<u16> = rng.vec(n * k, -1.0, 1.0).into_iter().map(bf16::from_f32).collect();
    let wq: Vec<u8> = (0..n * k).map(|_| (rng.next_u64() % 255) as u8).collect();
    let i8s = rng.vec(n * (k / kernels::int8_group()), 0.001, 0.02);
    let w8: Vec<u8> = (0..n * k).map(|_| (rng.next_u64() % 256) as u8 & 0xF7).collect();
    let fbs = rng.vec(n.div_ceil(128) * (k / 128), 0.5, 2.0);
    let mats = [
        (F::Bf16, up(&s, &wbf), None),
        (F::Int8Group, up(&s, &wq), Some(up(&s, &i8s))),
        (F::Fp8Block, up(&s, &w8), Some(up(&s, &fbs))),
    ];
    let mmax = 64;
    let x: Vec<u16> = rng.vec(mmax * k, -1.0, 1.0).into_iter().map(bf16::from_f32).collect();
    let xb = up(&s, &x);
    let y0 = rng.vec(mmax * n, -1.0, 1.0);
    let forms = [(16usize, 4usize), (16, 2), (8, 4), (8, 2), (8, 1), (4, 4), (4, 2), (4, 1), (2, 4), (2, 2)];
    for (fmt, w, sc) in &mats {
        let sc = sc.as_ref().map_or(0, |b| b.addr());
        for m in [2usize, 3, 4, 5, 6, 7, 8, 9, 16, 17, 25, 32, 40, 57, 64] {
            for (alpha, acc) in [(1.0f32, false), (0.37, true)] {
                let ya = up(&s, &y0);
                for r0 in (0..m).step_by(16) {
                    let rows = 16.min(m - r0);
                    kernels::gemv(&s, rows, xb.addr() + (r0 * k * 2) as u64, k, *fmt, w.addr(), sc, ya.addr() + (r0 * n * 4) as u64, n, 0, n, k, 1, alpha, acc)
                        .unwrap();
                }
                let want = s.download::<u32>(ya.addr(), m * n).unwrap();
                for &(mt, r) in &forms {
                    let yb = up(&s, &y0);
                    gemv_wide(&s, m, mt, r, xb.addr(), k, *fmt, w.addr(), sc, yb.addr(), n, n, k, alpha, acc).unwrap();
                    assert!(s.download::<u32>(yb.addr(), m * n).unwrap() == want, "wide GEMV {fmt:?} m {m} form {mt}x{r} alpha {alpha} acc {acc}");
                }
            }
        }
    }
    eprintln!("wide GEMV: equal");

    // Mamba convolution and commit in one launch.
    let p = MambaParams::random(&mut rng);
    let w = mamba_weights(&s, &p);
    let (a1, b1, a2, b2) = (new_block(), new_block(), new_block(), new_block());
    type Step = (Vec<f32>, Vec<f32>, bool, Option<[i32; 2]>);
    let mut steps: Vec<Step> = Vec::new();
    for _ in 0..3 {
        steps.push((random_proj(&mut rng, 1), random_proj(&mut rng, 2), false, None));
    }
    steps.push((random_proj(&mut rng, 4), random_proj(&mut rng, 5), true, Some([2, 5])));
    steps.push((random_proj(&mut rng, 3), random_proj(&mut rng, 1), true, Some([3, 1])));
    steps.push((random_proj(&mut rng, 1), random_proj(&mut rng, 1), false, None));
    for (i, (x, y, defer, acc)) in steps.iter().enumerate() {
        let o1 = mamba_step_with(&s, &w, &[&a1, &b1], &[x, y], *defer, false);
        let o2 = mamba_step_with(&s, &w, &[&a2, &b2], &[x, y], *defer, true);
        assert!(o1 == o2, "mamba step {i}: outputs");
        let rows = [x.len() / PROJ, y.len() / PROJ];
        match acc {
            Some(a) => {
                set_ctl(&s, &[&a1, &b1], &rows, a, true);
                set_ctl(&s, &[&a2, &b2], &rows, a, true);
            }
            None => {
                set_ctl(&s, &[&a1, &b1], &rows, &[0, 0], false);
                set_ctl(&s, &[&a2, &b2], &rows, &[0, 0], false);
            }
        }
        for (u, v) in [(&a1, &a2), (&b1, &b2)] {
            let bytes = mamba_seq_bytes(LAYERS, REC);
            assert!(s.download::<u8>(u.addr(), bytes).unwrap() == s.download::<u8>(v.addr(), bytes).unwrap(), "mamba step {i}: state blocks");
        }
    }
    eprintln!("Mamba convolution and commit in one launch: equal");

    // The attention combine with a CTA per query head, and decode rows against prefill rows.
    for fp8 in [false, true] {
        let mut qkv = Vec::new();
        let len = 9000;
        let (cache, kv, qout) = attn_fill(&s, &mut rng, len, fp8, &mut qkv);
        let q_all: Vec<u16> = s.download(qout.addr(), len * QH * HD).unwrap();
        let pos = [5usize, 1023, 1024, 4100, 8191, 8999];
        set_attn_combine(false);
        let old = attn_decode(&s, &cache, kv, &q_all, &pos);
        set_attn_combine(true);
        let new = attn_decode(&s, &cache, kv, &q_all, &pos);
        assert!(old.iter().zip(&new).all(|(a, b)| a.to_bits() == b.to_bits()), "attention combine fp8 {fp8}");
        for pf in [0usize, 1, 5] {
            set_attn_prefetch(pf);
            let other = attn_decode(&s, &cache, kv, &q_all, &pos);
            assert!(other.iter().zip(&new).all(|(a, b)| a.to_bits() == b.to_bits()), "attention prefetch {pf} fp8 {fp8}");
        }
        set_attn_prefetch(3);
        let full = attn_prefill(&s, &cache, kv, &q_all, 0, len);
        for (i, &p) in pos.iter().enumerate() {
            let a = &new[i * QH * HD..(i + 1) * QH * HD];
            let b = &full[p * QH * HD..(p + 1) * QH * HD];
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "attention decode vs prefill fp8 {fp8} position {p}");
        }
    }
    eprintln!("attention combine per query head, L2 prefetch distances: equal (and decode rows equal prefill rows)");
}

