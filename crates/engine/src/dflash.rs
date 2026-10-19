//! The DFlash 2 block drafter (incoai/Qwen3.8-Flash-Next-DFlash2): drafts 7 tokens in one pass of five
//! dense layers over the block [x, mask × 7], conditioned on the target's features — the completed
//! outputs of target layers 5, 15, 25, 35 and 45, each averaged over the 4 residual streams.
//!
//! The features of committed positions are projected once into per-layer K/V, kept in a ring of the
//! last [`RING`] positions (the drafter's attention window). Each drafting step first adds the rows
//! committed by the last verification (captured during it, see [`Taps`]), then runs the block, and a
//! candidate selector walks one path through each position's top 16 tokens. Verification is the
//! MTP one: only the drafts differ, so the output does not.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use quench6k_cuda::{DeviceBuffer, Stream, kernels as k};

use crate::checkpoint::{Checkpoint, DType};
use crate::forward::{self, HeadWorkspace};
use crate::loader::DeviceWeights;
use crate::model::{D, Linear, Model, VOCAB};
use crate::profile::Profiler;

/// Target layers whose outputs condition the drafter.
pub const TAPS: [usize; 5] = [5, 15, 25, 35, 45];
pub const BLOCK: usize = 8;
pub const DRAFTS: usize = BLOCK - 1;
/// Positions of context K/V kept (the sliding window).
pub const RING: usize = 2048;
const LAYERS: usize = 5;
const QH: usize = 32;
const KVH: usize = 8;
const HD: usize = 128;
const FF: usize = 7680;
const RANK: usize = 256;
const CAND: usize = 16;
/// Feature row: the 5 taps, BF16.
pub const FEAT_BYTES: usize = TAPS.len() * D * 2;
/// Header words (see dflash.cu): LO, CTX, BLK, then the block tokens.
const HDR_LO: usize = 0;
const HDR_CTX: usize = 1;
const HDR_BLK: usize = 2;
const HDR_INTS: usize = 3 + BLOCK;

/// Feature capture of a target step: rows `first .. first + rows` of the step go to `buf` rows 0 ..
/// (BF16 [rows, 5, 2560]).
#[derive(Clone, Copy)]
pub struct Taps {
    pub buf: u64,
    pub first: usize,
    pub rows: usize,
}

struct DLayer {
    in_norm: u64,
    post_norm: u64,
    attn_conv: (Linear, u64),
    mlp_conv: (Linear, u64),
    /// q_proj | k_proj | v_proj as one projection, and its k | v rows.
    qkv: Linear,
    kv: Linear,
    o: Linear,
    q_norm: u64,
    k_norm: u64,
    /// gate_proj | up_proj.
    gate_up: Linear,
    down: Linear,
}

pub struct DFlash {
    _weights: DeviceWeights,
    _bufs: Vec<DeviceBuffer>,
    _staged: Option<DeviceWeights>,
    fc: Linear,
    hidden_norm: u64,
    norm: u64,
    layers: Vec<DLayer>,
    select_proj: Linear,
    pred_cb: u64,
    succ_cb: u64,
    mask: i32,
}

impl DFlash {
    /// Loads the drafter from `dir` (config.json and BF16 safetensors) and checks it fits this engine.
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).with_context(|| format!("reading {}/config.json", dir.display()))?)?;
        let dc = &cfg["dflash_config"];
        let num = |v: &serde_json::Value| v.as_u64().map(|x| x as usize);
        let arch = cfg["architectures"][0].as_str().unwrap_or("");
        ensure!(arch == "DFlash2DraftModel", "{}: architecture {arch:?}, expected DFlash2DraftModel", dir.display());
        let expect = [
            ("hidden_size", num(&cfg["hidden_size"]), D),
            ("num_hidden_layers", num(&cfg["num_hidden_layers"]), LAYERS),
            ("num_attention_heads", num(&cfg["num_attention_heads"]), QH),
            ("num_key_value_heads", num(&cfg["num_key_value_heads"]), KVH),
            ("head_dim", num(&cfg["head_dim"]), HD),
            ("intermediate_size", num(&cfg["intermediate_size"]), FF),
            ("vocab_size", num(&cfg["vocab_size"]), VOCAB),
            ("sliding_window", num(&cfg["sliding_window"]), RING),
            ("block_size", num(&dc["block_size"]), BLOCK),
            ("conv_kernel_size", num(&dc["conv_kernel_size"]), 2),
            ("conv_group_size", num(&dc["conv_group_size"]), 16),
            ("selector_rank", num(&dc["selector_rank"]), RANK),
            ("selector_top_k", num(&dc["selector_top_k"]), CAND),
        ];
        for (name, got, want) in expect {
            ensure!(got == Some(want), "drafter {name} {got:?}, expected {want}");
        }
        let taps: Vec<usize> = dc["target_layer_ids"].as_array().map(|a| a.iter().filter_map(num).collect()).unwrap_or_default();
        ensure!(taps == TAPS, "drafter target layers {taps:?}, expected {TAPS:?}");
        ensure!(cfg["is_causal"].as_bool() == Some(false), "the drafter must be non-causal");
        ensure!(cfg["rope_parameters"]["rope_theta"].as_f64() == Some(10000.0) && cfg["rope_parameters"]["rope_type"].as_str() == Some("default"), "drafter RoPE must be default with theta 10000");
        for (key, want) in [("input_embedding_scale", 1.0), ("output_multiplier", 1.0)] {
            ensure!(dc[key].as_f64().unwrap_or(want) == want, "drafter {key} {:?} is not supported", dc[key]);
        }
        ensure!(cfg["final_logit_softcapping"].is_null(), "drafter logit softcapping is not supported");
        let mask = dc["mask_token_id"].as_i64().context("drafter config lacks mask_token_id")? as i32;

        let ckpt = Checkpoint::open(dir)?;
        // Projections are staged in BF16 and stored as INT8 with a scale per 64 weights (as the target's
        // projections; QUENCH6K_DFLASH_BF16=1 keeps BF16), q/k/v and gate/up fused into one each.
        let int8 = std::env::var("QUENCH6K_DFLASH_BF16").as_deref() != Ok("1");
        let is_proj = |n: &str| n.ends_with("_proj.weight") || n == "fc.weight" || n.ends_with("projection.weight");
        let opts = crate::loader::LoadOptions::default();
        let (weights, stats) = crate::loader::upload(&ckpt, |n| !is_proj(n), opts)?;
        let (staged, staged_stats) = crate::loader::upload(&ckpt, is_proj, opts)?;
        let stream = Stream::new()?;
        let mut bufs = Vec::new();
        let mut finish = |parts: &[Linear]| -> Result<Linear> {
            let (n, kk) = (parts.iter().map(|l| l.n).sum::<usize>(), parts[0].k);
            let fused = if parts.len() == 1 {
                parts[0]
            } else {
                let b = DeviceBuffer::new(n * kk * 2)?;
                let mut off = 0;
                for l in parts {
                    stream.copy_d2d(b.addr() + off as u64, l.addr, l.n * kk * 2)?;
                    off += l.n * kk * 2;
                }
                let l = Linear { fmt: k::WeightFormat::Bf16, addr: b.addr(), scale: 0, n, k: kk };
                bufs.push(b);
                l
            };
            if !int8 {
                return Ok(fused);
            }
            let q = DeviceBuffer::new(n * kk)?;
            let sc = DeviceBuffer::new(n * kk / k::int8_group() * 4)?;
            k::quantize_int8_groups(&stream, fused.addr, n, kk, q.addr(), sc.addr())?;
            let l = Linear { fmt: k::WeightFormat::Int8Group, addr: q.addr(), scale: sc.addr(), n, k: kk };
            bufs.push(q);
            bufs.push(sc);
            Ok(l)
        };
        let w = |name: &str| -> Result<u64> {
            let t = weights.get(name)?;
            ensure!(t.dtype == DType::BF16, "{name}: {:?}, expected BF16", t.dtype);
            Ok(t.addr)
        };
        let lin = |name: &str, n: usize, kk: usize| -> Result<Linear> {
            let t = staged.get(name)?;
            ensure!(t.dtype == DType::BF16 && t.shape == [n, kk], "{name}: {:?} {:?}, expected BF16 [{n}, {kk}]", t.dtype, t.shape);
            Ok(Linear { fmt: k::WeightFormat::Bf16, addr: t.addr, scale: 0, n, k: kk })
        };
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("layers.{i}");
            let mut conv = |c: &str| -> Result<(Linear, u64)> {
                let base = format!("{p}.{c}.base_kernel");
                ensure!(weights.get(&base)?.shape == [2, 2, D], "{base}: unexpected shape");
                Ok((finish(&[lin(&format!("{p}.{c}.kernel_projection.weight"), 4 * D / 16, D)?])?, w(&base)?))
            };
            let (attn_conv, mlp_conv) = (conv("attention_conv")?, conv("mlp_conv")?);
            layers.push(DLayer {
                in_norm: w(&format!("{p}.input_layernorm.weight"))?,
                post_norm: w(&format!("{p}.post_attention_layernorm.weight"))?,
                attn_conv,
                mlp_conv,
                qkv: finish(&[
                    lin(&format!("{p}.self_attn.q_proj.weight"), QH * HD, D)?,
                    lin(&format!("{p}.self_attn.k_proj.weight"), KVH * HD, D)?,
                    lin(&format!("{p}.self_attn.v_proj.weight"), KVH * HD, D)?,
                ])?,
                kv: Linear { fmt: k::WeightFormat::Bf16, addr: 0, scale: 0, n: 0, k: 0 },
                o: finish(&[lin(&format!("{p}.self_attn.o_proj.weight"), D, QH * HD)?])?,
                q_norm: w(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: w(&format!("{p}.self_attn.k_norm.weight"))?,
                gate_up: finish(&[lin(&format!("{p}.mlp.gate_proj.weight"), FF, D)?, lin(&format!("{p}.mlp.up_proj.weight"), FF, D)?])?,
                down: finish(&[lin(&format!("{p}.mlp.down_proj.weight"), D, FF)?])?,
            });
            // The k | v rows of the fused projection (context rows need no queries).
            let l = layers.last_mut().unwrap();
            let r0 = QH * HD;
            let bytes = if l.qkv.fmt == k::WeightFormat::Bf16 { 2 } else { 1 };
            let scale = if l.qkv.scale == 0 { 0 } else { l.qkv.scale + (r0 * D / k::int8_group() * 4) as u64 };
            l.kv = Linear { fmt: l.qkv.fmt, addr: l.qkv.addr + (r0 * D * bytes) as u64, scale, n: 2 * KVH * HD, k: D };
        }
        let codebook = |name: &str| -> Result<u64> {
            let t = weights.get(name)?;
            ensure!(t.dtype == DType::BF16 && t.shape == [VOCAB, RANK], "{name}: unexpected layout");
            Ok(t.addr)
        };
        let fc = finish(&[lin("fc.weight", D, TAPS.len() * D)?])?;
        let select_proj = finish(&[lin("candidate_selector.hidden_projection.weight", RANK, D)?])?;
        stream.synchronize()?;
        // INT8: the staged BF16 projections are no longer needed.
        let staged = if int8 { drop(staged); None } else { Some(staged) };
        let d = Self {
            fc,
            hidden_norm: w("hidden_norm.weight")?,
            norm: w("norm.weight")?,
            select_proj,
            pred_cb: codebook("candidate_selector.predecessor_codebook")?,
            succ_cb: codebook("candidate_selector.successor_codebook")?,
            layers,
            mask,
            _weights: weights,
            _bufs: bufs,
            _staged: staged,
        };
        println!(
            "DFlash drafter {}: {:.2} GiB in {:.1}s, projections {}",
            dir.display(),
            (stats.bytes + staged_stats.bytes) as f64 / 1073741824.0,
            stats.seconds + staged_stats.seconds,
            if int8 { "INT8 (a scale per 64)" } else { "BF16" }
        );
        Ok(d)
    }
}

/// A sequence's drafter state: the context K/V ring ([layer][K, V][RING, 8, 128] BF16), the features of
/// the last verification's rows, and the header.
pub struct DState {
    kv: DeviceBuffer,
    pub feat: DeviceBuffer,
    pub hdr: DeviceBuffer,
}

const RING_BYTES: usize = RING * KVH * HD * 2;

impl DState {
    /// `rows`: verification rows to keep features of.
    pub fn new(rows: usize) -> Result<Self> {
        Ok(Self { kv: DeviceBuffer::zeroed(LAYERS * 2 * RING_BYTES)?, feat: DeviceBuffer::zeroed(rows * FEAT_BYTES)?, hdr: DeviceBuffer::zeroed(HDR_INTS * 4)? })
    }

    fn ring(&self, layer: usize) -> (u64, u64) {
        let base = self.kv.addr() + (layer * 2 * RING_BYTES) as u64;
        (base, base + RING_BYTES as u64)
    }

    /// Marks the ring as holding positions from `lo` on (after a prefill).
    pub fn set_lo(&self, s: &Stream, lo: usize) -> Result<()> {
        s.upload_to(self.hdr.addr() + (HDR_LO * 4) as u64, &[lo as i32])
    }
}

/// Scratch of the drafter, shared by the sequences of a workspace.
pub struct Scratch {
    _bufs: Vec<DeviceBuffer>,
    /// Prefill features (RING rows) and verification features (up to 16 rows).
    pub feat: u64,
    pub vfeat: u64,
    ctx_f: u64,
    ctx_bf: u64,
    ctx_kv: u64,
    h: u64,
    hn: u64,
    hc: u64,
    dynb: u64,
    qkv: u64,
    q: u64,
    bk: u64,
    bv: u64,
    att: u64,
    part: u64,
    o: u64,
    gu: u64,
    act: u64,
    out: u64,
    hp: u64,
    ids: u64,
    vals: u64,
    partial: u64,
    head: HeadWorkspace,
}

/// Rows of the verification feature buffer (a pair round's 2 × 8).
pub const VFEAT_ROWS: usize = 16;

impl Scratch {
    pub fn new() -> Result<Self> {
        let mut bufs = Vec::new();
        let mut alloc = |bytes: usize| -> Result<u64> {
            let b = DeviceBuffer::zeroed(bytes)?;
            let a = b.addr();
            bufs.push(b);
            Ok(a)
        };
        let b = BLOCK;
        Ok(Self {
            feat: alloc(RING * FEAT_BYTES)?,
            vfeat: alloc(VFEAT_ROWS * FEAT_BYTES)?,
            ctx_f: alloc(RING * D * 4)?,
            ctx_bf: alloc(RING * D * 2)?,
            ctx_kv: alloc(RING * 2 * KVH * HD * 4)?,
            h: alloc(b * D * 4)?,
            hn: alloc(b * D * 2)?,
            hc: alloc(b * D * 2)?,
            dynb: alloc(b * 4 * D / 16 * 4)?,
            qkv: alloc(b * (QH + 2 * KVH) * HD * 4)?,
            q: alloc(b * QH * HD * 4)?,
            bk: alloc(b * KVH * HD * 2)?,
            bv: alloc(b * KVH * HD * 2)?,
            att: alloc(b * QH * HD * 2)?,
            part: alloc(k::dflash_attention_floats() * 4)?,
            o: alloc(b * D * 4)?,
            gu: alloc(b * 2 * FF * 4)?,
            act: alloc(b * FF * 2)?,
            out: alloc(b * D * 2)?,
            hp: alloc(b * RANK * 4)?,
            ids: alloc(b * CAND * 4)?,
            vals: alloc(b * CAND * 4)?,
            partial: alloc(forward::PARTIAL_FLOATS * 4)?,
            head: HeadWorkspace::new(DRAFTS, 1)?,
            _bufs: bufs,
        })
    }
}

/// Where the context rows' positions come from: a device word (0: none) plus a host offset.
#[derive(Clone, Copy)]
pub struct Pos(pub u64, pub usize);

/// Runs `$call` on `$s`, timed under `$name` when profiling.
macro_rules! op {
    ($prof:ident, $name:expr, $s:expr, $call:expr) => {
        match $prof.as_deref_mut() {
            Some(p) => p.time($name, $s, || $call)?,
            None => $call?,
        }
    };
}

/// Adds `rows` (≤ RING) feature rows at `feat` (positions `pos` ..) to the sequence's context ring.
pub fn context(d: &DFlash, sc: &Scratch, st: &DState, s: &Stream, feat: u64, rows: usize, pos: Pos) -> Result<()> {
    context_profiled(d, sc, st, s, feat, rows, pos, None)
}

#[allow(clippy::too_many_arguments)]
fn context_profiled(d: &DFlash, sc: &Scratch, st: &DState, s: &Stream, feat: u64, rows: usize, pos: Pos, mut prof: Option<&mut Profiler>) -> Result<()> {
    ensure!(rows <= RING, "{rows} context rows exceed the ring");
    if rows == 0 {
        return Ok(());
    }
    op!(prof, "ctx fc", s, forward::linear_rows(s, sc.partial, feat, TAPS.len() * D, &d.fc, sc.ctx_f, D, rows));
    op!(prof, "ctx norm", s, k::dflash_rmsnorm(s, sc.ctx_f, D, d.hidden_norm, sc.ctx_bf, D, rows));
    let ld = 2 * KVH * HD;
    for (i, l) in d.layers.iter().enumerate() {
        op!(prof, "ctx kv", s, forward::linear_rows(s, sc.partial, sc.ctx_bf, D, &l.kv, sc.ctx_kv, ld, rows));
        let (rk, rv) = st.ring(i);
        op!(prof, "ctx kv prep", s, k::dflash_qkv(s, sc.ctx_kv, ld, (0, 0, KVH * HD), 0, l.k_norm, pos.0, pos.1, rows, 0, 0, rk, rv, true));
    }
    Ok(())
}

/// One drafting step (graph-capturable): `c` context rows from `st.feat` (the last verification's
/// committed rows), then the block from the round control's position `rc_p` and next token `rc_x`.
/// The 7 drafts go to `drafts` (i32), their distributions to `qs` when `params` asks for sampled drafts.
#[allow(clippy::too_many_arguments)]
pub fn draft(model: &Model, d: &DFlash, sc: &Scratch, st: &DState, s: &Stream, c: usize, rc_p: u64, rc_x: u64, drafts: u64, qs: u64, params: u64, mut prof: Option<&mut Profiler>) -> Result<()> {
    let hdr = st.hdr.addr();
    op!(prof, "header", s, k::dflash_header(s, rc_p, rc_x, c, hdr, d.mask));
    context_profiled(d, sc, st, s, st.feat.addr(), c, Pos(hdr + (HDR_CTX * 4) as u64, 0), prof.as_deref_mut())?;
    let lin = |x: u64, ldx: usize, w: &Linear, y: u64, ldy: usize, rows: usize| forward::linear_rows(s, sc.partial, x, ldx, w, y, ldy, rows);
    let b = BLOCK;
    let ld_dyn = 4 * D / 16;
    op!(prof, "embed", s, k::dflash_embed(s, hdr, model.embed, sc.h));
    for (i, l) in d.layers.iter().enumerate() {
        // Attention: norm, convolution, projections, head norms and RoPE, block attention, output
        // projection, convolution added to the residual.
        op!(prof, "norm", s, k::dflash_rmsnorm(s, sc.h, D, l.in_norm, sc.hn, D, b));
        op!(prof, "proj attn_conv", s, lin(sc.hn, D, &l.attn_conv.0, sc.dynb, ld_dyn, b));
        op!(prof, "conv", s, k::dflash_conv(s, sc.hn, true, sc.dynb, ld_dyn, l.attn_conv.1, sc.hc, false));
        let ld = (QH + 2 * KVH) * HD;
        op!(prof, "proj qkv", s, lin(sc.hc, D, &l.qkv, sc.qkv, ld, b));
        op!(prof, "qkv prep", s, k::dflash_qkv(s, sc.qkv, ld, (0, QH * HD, (QH + KVH) * HD), l.q_norm, l.k_norm, hdr + (HDR_BLK * 4) as u64, 0, b, QH, sc.q, sc.bk, sc.bv, false));
        let (rk, rv) = st.ring(i);
        op!(prof, "attention", s, k::dflash_attention(s, sc.q, rk, rv, sc.bk, sc.bv, hdr, sc.part, 0));
        op!(prof, "attention combine", s, k::dflash_attention(s, 0, 0, 0, 0, 0, hdr, sc.part, sc.att));
        op!(prof, "proj o", s, lin(sc.att, QH * HD, &l.o, sc.o, D, b));
        op!(prof, "conv", s, k::dflash_conv(s, sc.o, false, sc.dynb + (2 * D / 16 * 4) as u64, ld_dyn, l.attn_conv.1 + (2 * D * 2) as u64, sc.h, true));
        // MLP, the same way.
        op!(prof, "norm", s, k::dflash_rmsnorm(s, sc.h, D, l.post_norm, sc.hn, D, b));
        op!(prof, "proj mlp_conv", s, lin(sc.hn, D, &l.mlp_conv.0, sc.dynb, ld_dyn, b));
        op!(prof, "conv", s, k::dflash_conv(s, sc.hn, true, sc.dynb, ld_dyn, l.mlp_conv.1, sc.hc, false));
        op!(prof, "proj gate_up", s, lin(sc.hc, D, &l.gate_up, sc.gu, 2 * FF, b));
        op!(prof, "swiglu", s, k::swiglu(s, sc.gu, sc.gu + (FF * 4) as u64, 2 * FF, FF, sc.act, b));
        op!(prof, "proj down", s, lin(sc.act, FF, &l.down, sc.o, D, b));
        op!(prof, "conv", s, k::dflash_conv(s, sc.o, false, sc.dynb + (2 * D / 16 * 4) as u64, ld_dyn, l.mlp_conv.1 + (2 * D * 2) as u64, sc.h, true));
    }
    op!(prof, "norm", s, k::dflash_rmsnorm(s, sc.h, D, d.norm, sc.out, D, b));
    // Rows 1 .. 7 (the mask positions): LM-head logits, their top 16, the selector's context gate.
    let rows = sc.out + (D * 2) as u64;
    op!(prof, "lm head", s, forward::head_logits_rows(model, &sc.head, s, rows, DRAFTS));
    op!(prof, "topk", s, k::topk(s, sc.head.logits_addr(), DRAFTS, VOCAB, VOCAB, CAND, false, sc.ids, sc.vals));
    op!(prof, "proj select_proj", s, lin(rows, D, &d.select_proj, sc.hp, RANK, DRAFTS));
    op!(prof, "select", s, k::dflash_select(s, sc.ids, sc.vals, sc.hp, d.pred_cb, d.succ_cb, rc_x, drafts, qs, params));
    Ok(())
}

