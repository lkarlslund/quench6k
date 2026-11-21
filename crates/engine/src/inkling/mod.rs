//! Inkling 2 Mini (`inkling2_loop`, thinkingmachines/Inkling-2-Mini) text model: configuration and weights
//! on the device. The forward pass is in [`forward`].
//!
//! Supported checkpoint layout (the original release): routed experts NVFP4 (`w13_weight` codes
//! `[128, 2304, 1152]` with gate and up rows interleaved, E4M3 block scales per 16, a per-expert factor
//! `amax / outer_scale_denominator`) or MXFP8 (E4M3 with a UE8M0 scale per 32 columns), everything else
//! BF16 or F32.

pub mod forward;

use std::path::Path;

use anyhow::{Context, Result, ensure};
use quench6k_cuda::inkling::{ExpertStack, MarkovW};
use quench6k_cuda::kernels::WeightFormat;
use quench6k_cuda::{DeviceBuffer, Stream};
use serde_json::Value;

use crate::checkpoint::{Checkpoint, DType};

pub const D: usize = 2304;
pub const H: usize = 24;
pub const KVH: usize = 8;
pub const HD: usize = 128;
pub const QD: usize = H * HD;
pub const KVD: usize = KVH * HD;
pub const DREL: usize = 16;
pub const EXT: usize = 1024;
pub const WINDOW: usize = 1024;
pub const EXPERTS: usize = 128;
pub const TOPK: usize = 8;
pub const FF: usize = 1152;
pub const BLOCK: usize = 6;
pub const RANK: usize = 64;
pub const CONVK: usize = 4;
/// Columns of the fused query-side projection: q, output gate, relative states, sinks.
pub const QG: usize = 2 * QD + H * DREL + H;
pub const GATE_OFF: usize = QD;

pub struct Config {
    pub layers: usize,
    /// Logits columns (the unembedding's unpadded rows).
    pub vocab: usize,
    /// Embedding rows.
    pub embed_rows: usize,
    pub sliding: Vec<bool>,
    pub loop_start: usize,
    pub loop_end: usize,
    pub passes: usize,
    pub eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub gate_cap: f32,
    pub up_cap: f32,
    pub logit_cap: f32,
    pub route_scale: f32,
    pub eos: Vec<i32>,
}

fn num(v: &Value, key: &str) -> Result<f64> {
    v.get(key).and_then(Value::as_f64).with_context(|| format!("config: {key} missing"))
}

impl Config {
    pub fn read(dir: &Path) -> Result<Self> {
        let all: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let t = all.get("text_config").context("config: text_config missing")?;
        let arch = t.get("model_type").and_then(Value::as_str).unwrap_or("");
        ensure!(arch == "inkling2_loop_text", "config: model_type {arch:?} is not inkling2_loop_text");
        let layers = num(t, "num_hidden_layers")? as usize;
        for (key, want) in [
            ("hidden_size", D),
            ("num_attention_heads", H),
            ("num_key_value_heads", KVH),
            ("head_dim", HD),
            ("swa_num_attention_heads", H),
            ("swa_num_key_value_heads", KVH),
            ("swa_head_dim", HD),
            ("d_rel", DREL),
            ("rel_extent", EXT),
            ("sliding_window_size", WINDOW),
            ("n_routed_experts", EXPERTS),
            ("num_experts_per_tok", TOPK),
            ("n_shared_experts", 0),
            ("intermediate_size", FF),
            ("n_layers_per_attn_res_block", BLOCK),
            ("sconv_kernel_size", CONVK),
            ("log_scaling_n_floor", 8192),
        ] {
            let v = num(t, key)? as usize;
            ensure!(v == want, "config: {key} = {v}, this engine supports {want}");
        }
        ensure!((num(t, "log_scaling_alpha")? - 0.1).abs() < 1e-9, "config: log_scaling_alpha");
        let local: Vec<usize> = t
            .get("local_layer_ids")
            .and_then(Value::as_array)
            .context("config: local_layer_ids")?
            .iter()
            .filter_map(Value::as_u64)
            .map(|v| v as usize)
            .collect();
        let sliding: Vec<bool> = (0..layers).map(|i| local.contains(&i)).collect();
        for (i, &s) in sliding.iter().enumerate() {
            // The transformers model types layers by position: every 6th is full attention.
            ensure!(s != (i + 1).is_multiple_of(BLOCK), "config: layer {i} attention type");
        }
        let lc = t.get("loop_config").context("config: loop_config")?;
        let loop_start = num(lc, "core_start")? as usize;
        let loop_end = num(lc, "core_end")? as usize;
        let passes = num(lc, "num_passes")? as usize;
        ensure!(num(lc, "gate_rank")? as usize == RANK, "config: loop gate_rank");
        ensure!(
            lc.get("is_varloop").and_then(Value::as_bool) != Some(true),
            "config: variable loop counts are not supported"
        );
        ensure!(
            loop_start > 0 && loop_start.is_multiple_of(BLOCK) && loop_end.is_multiple_of(BLOCK) && loop_start < loop_end && loop_end <= layers && passes >= 1,
            "config: loop {loop_start}..{loop_end} x {passes}"
        );
        // One block of the loop's last pass feeds the next pass's messages (plus its partial sum).
        ensure!(loop_end - loop_start == 2 * BLOCK, "config: loops of other than two blocks are not supported");
        // Block sums kept: embedding, one per block before the loop and in it, and after it.
        ensure!(layers / BLOCK < forward::SLOTS - 2, "config: too many blocks");
        let eos = match all.get("eos_token_id").or_else(|| t.get("eos_token_id")) {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_i64).map(|v| v as i32).collect(),
            Some(Value::Number(n)) => n.as_i64().map(|v| v as i32).into_iter().collect(),
            _ => Vec::new(),
        };
        Ok(Self {
            layers,
            vocab: num(t, "unpadded_vocab_size")? as usize,
            embed_rows: num(t, "vocab_size")? as usize,
            sliding,
            loop_start,
            loop_end,
            passes,
            eps: num(t, "rms_norm_eps")? as f32,
            qk_clip: num(t, "qk_l2_clip_threshold")? as f32,
            rel_clip: num(t, "rel_l2_clip_threshold")? as f32,
            gate_cap: num(t, "mlp_gate_softcap")? as f32,
            up_cap: num(t, "mlp_up_softcap")? as f32,
            logit_cap: num(t, "final_logit_softcapping")? as f32,
            route_scale: num(t, "route_scale")? as f32,
            eos,
        })
    }

    pub fn looped(&self, i: usize) -> bool {
        (self.loop_start..self.loop_end).contains(&i)
    }

    /// Layer visits of one forward pass: (layer, pass) for the layers before the loop, each pass of the
    /// loop, then the layers after it.
    pub fn visits(&self) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = (0..self.loop_start).map(|i| (i, 0)).collect();
        for p in 0..self.passes {
            v.extend((self.loop_start..self.loop_end).map(|i| (i, p)));
        }
        v.extend((self.loop_end..self.layers).map(|i| (i, 0)));
        v
    }
}

/// A projection `[n, k]` on the device.
#[derive(Clone, Copy, Debug)]
pub struct Linear {
    pub w: u64,
    pub scale: u64,
    pub fmt: WeightFormat,
    pub n: usize,
    pub k: usize,
}

pub struct Attn {
    /// [q (3072) | output gate (3072) | relative states (384) | sinks (24)] x 2304.
    pub qg: Linear,
    /// Keys [1024] (+ the full layers' per-token convolution taps [32]) x 2304; values likewise.
    pub kg: Linear,
    pub vg: Linear,
    pub o: Linear,
    pub qscale: u64,
    pub rscale: u64,
    /// Relative position profiles BF16 [16, 1024].
    pub proj: u64,
    /// Sliding layers: taps BF16 [1024, 4]; full layers: static taps BF16 [4, 1024].
    pub kconv: u64,
    pub vconv: u64,
}

/// Message gate of a looped layer: down [64, 4608], forget | gain [4608, 64].
pub struct MsgGate {
    pub down: Linear,
    pub fg: Linear,
}

pub struct Layer {
    pub sliding: bool,
    pub looped: bool,
    /// Input norms of the query, key and value inputs (one shared norm in sliding layers).
    pub in_norm: [u64; 3],
    /// Queries of the attention input's block residual F32 [1 or 3, 2304] (0 for layer 0).
    pub attn_query: u64,
    pub mlp_query: u64,
    pub post_norm: u64,
    pub attn: Attn,
    pub attn_conv: u64,
    pub mlp_conv: u64,
    pub router: Linear,
    pub bias: u64,
    pub fp4: bool,
    pub gate_up: ExpertStack,
    pub down: ExpertStack,
    /// NVFP4 layers: per-expert (codes, scales) pointer tables of gate/up and down for the grouped GEMM.
    pub tables: Option<[(u64, u64); 2]>,
    /// Query, key, value and MLP message gates of looped layers.
    pub gates: Option<[MsgGate; 4]>,
}

/// One layer of the DSpark drafter: pre-norm attention (over context keys from the target's features
/// and the block itself) and a dense SwiGLU MLP.
pub struct DraftLayer {
    pub local: bool,
    pub in_norm: u64,
    pub post_norm: u64,
    /// Norm of the layer's mixture of the target features (its context input).
    pub mix_norm: u64,
    /// [q (3072) | k (1024) | v (1024)] x 2304.
    pub qkv: Linear,
    /// Context [k (1024) | v (1024)] x 2304.
    pub kv_ctx: Linear,
    pub o: Linear,
    pub q_norm: u64,
    pub k_norm: u64,
    pub k_norm_ctx: u64,
    /// [gate (9216) | up (9216)] x 2304.
    pub gate_up: Linear,
    pub down: Linear,
}

/// The checkpoint's DSpark drafter (`mtp.safetensors`): a block of the accepted token and mask tokens
/// through `layers`, then a recurrent Markov head that picks the drafts in order, biasing the target's LM
/// head logits of each block row.
pub struct Drafter {
    pub layers: Vec<DraftLayer>,
    pub norm: u64,
    pub mask: u64,
    /// F32 [layers + 1, 4]: softmax(fuse_weight) rows, the mixture weights of the target features per layer
    /// and (last) for the Markov head.
    pub alpha: u64,
    /// Target layer visits whose inputs are the first features (the last is the final norm's output).
    pub visits: Vec<usize>,
    pub markov: MarkovW,
    /// rnn_proj's hidden and f_rnn columns [768, 4608].
    pub markov_hf: Linear,
    pub markov_norm: u64,
    pub theta: f32,
    pub window: usize,
    /// Block rows: the accepted token and block - 1 masks (the drafts).
    pub block: usize,
}

pub struct InklingModel {
    pub cfg: Config,
    pub layers: Vec<Layer>,
    /// Device BF16 [embed_rows, 2304].
    pub embed: u64,
    pub embed_norm: u64,
    /// Message queries F32 [loop layers, 4 (query, key, value, MLP), 2304].
    pub msg_queries: u64,
    pub final_query: u64,
    pub norm: u64,
    pub lm_head: Linear,
    /// The DSpark drafter (QUENCH6K_INK_DRAFT=0: not loaded).
    pub drafter: Option<Drafter>,
    pub device_bytes: usize,
    _buffers: Vec<DeviceBuffer>,
}

impl InklingModel {
    pub fn is_inkling(dir: &Path) -> bool {
        std::fs::read(dir.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .is_some_and(|v| v.get("model_type").and_then(Value::as_str) == Some("inkling2_loop"))
    }

    pub fn load(dir: &Path, s: &Stream) -> Result<Self> {
        let cfg = Config::read(dir)?;
        let ckpt = Checkpoint::open(dir)?;
        let t0 = std::time::Instant::now();
        let largest = ckpt.tensors.values().map(|t| t.bytes as usize).max().unwrap_or(0);
        let mut l = Loader {
            ckpt: &ckpt,
            s,
            staging: quench6k_cuda::PinnedBuffer::new(largest)?,
            buffers: Vec::new(),
            bytes: 0,
        };
        let m = |x: &str| format!("model.llm.{x}");
        let embed = l.raw(&m("embed.weight"), DType::BF16, Some(&[cfg.embed_rows, D]))?;
        let embed_norm = l.raw(&m("embed_norm.weight"), DType::BF16, Some(&[D]))?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let n = |x: &str| m(&format!("layers.{i}.{x}"));
            let a = |x: &str| n(&format!("attn.{x}"));
            let sliding = cfg.sliding[i];
            let looped = cfg.looped(i);
            let in_norm = if sliding {
                let w = l.raw(&n("attn_norm.weight"), DType::BF16, Some(&[D]))?;
                [w, w, w]
            } else {
                [
                    l.raw(&n("q_input_norm.weight"), DType::BF16, Some(&[D]))?,
                    l.raw(&n("k_input_norm.weight"), DType::BF16, Some(&[D]))?,
                    l.raw(&n("v_input_norm.weight"), DType::BF16, Some(&[D]))?,
                ]
            };
            let qg = l.concat(&[a("wq_du.weight"), a("gate_attn.weight"), a("wr_du.weight"), a("attn_sink_proj.weight")], D)?;
            ensure!(qg.n == QG, "layer {i}: query-side projections have {} rows", qg.n);
            let (kg, vg, kconv, vconv) = if sliding {
                (
                    l.concat(&[a("wk_dv.weight")], D)?,
                    l.concat(&[a("wv_dv.weight")], D)?,
                    l.raw(&a("k_sconv.weight"), DType::BF16, Some(&[KVD, 1, CONVK]))?,
                    l.raw(&a("v_sconv.weight"), DType::BF16, Some(&[KVD, 1, CONVK]))?,
                )
            } else {
                (
                    l.concat(&[a("wk_dv.weight"), a("k_sconv.weight_proj")], D)?,
                    l.concat(&[a("wv_dv.weight"), a("v_sconv.weight_proj")], D)?,
                    l.raw(&a("k_sconv.static_w"), DType::BF16, Some(&[CONVK, KVD]))?,
                    l.raw(&a("v_sconv.static_w"), DType::BF16, Some(&[CONVK, KVD]))?,
                )
            };
            let attn = Attn {
                qg,
                kg,
                vg,
                o: l.concat(&[a("wo_ud.weight")], QD)?,
                qscale: l.raw(&a("q_l2_clip.scale"), DType::BF16, Some(&[H, HD]))?,
                rscale: l.raw(&a("rel_l2_clip.scale"), DType::BF16, Some(&[H, DREL]))?,
                proj: l.raw(&a("rel_logits_proj.proj"), DType::BF16, Some(&[DREL, EXT]))?,
                kconv,
                vconv,
            };
            let nq = if sliding { 1 } else { 3 };
            let attn_query = if i == 0 {
                0
            } else {
                let name = m(&format!("attn_res.attn_queries.{}", i - 1));
                let shape: &[usize] = if nq == 1 { &[D] } else { &[3, D] };
                l.raw(&name, DType::F32, Some(shape))?
            };
            let ex = n("mlp.experts.w13_weight");
            let fp4 = ckpt.tensors.contains_key(&format!("{ex}_amax"));
            let (gate_up, down) = if fp4 {
                (l.nvfp4_experts(&ex, 2 * FF, D)?, l.nvfp4_experts(&n("mlp.experts.w2_weight"), D, FF)?)
            } else {
                (l.mxfp8_experts(&ex, 2 * FF, D)?, l.mxfp8_experts(&n("mlp.experts.w2_weight"), D, FF)?)
            };
            let tables = if fp4 {
                let mut t = [(0, 0); 2];
                for (slot, (st, rows, cols)) in [(gate_up, 2 * FF, D), (down, D, FF)].into_iter().enumerate() {
                    let c: Vec<u64> = (0..EXPERTS).map(|e| st.w + (e * rows * cols / 2) as u64).collect();
                    let sc: Vec<u64> = (0..EXPERTS).map(|e| st.sc + (e * rows * cols / 16) as u64).collect();
                    t[slot] = (l.upload_slice(&c)?, l.upload_slice(&sc)?);
                }
                Some(t)
            } else {
                None
            };
            let gates = if looped {
                let g = |l: &mut Loader, c: &str| -> Result<MsgGate> {
                    let x = |y: &str| n(&format!("controllers.{c}.{y}"));
                    Ok(MsgGate {
                        down: l.concat(&[x("down_weight")], 2 * D)?,
                        fg: l.concat(&[x("forget_weight"), x("gain_weight")], RANK)?,
                    })
                };
                Some([g(&mut l, "attn")?, g(&mut l, "k")?, g(&mut l, "v")?, g(&mut l, "mlp")?])
            } else {
                None
            };
            layers.push(Layer {
                sliding,
                looped,
                in_norm,
                attn_query,
                mlp_query: l.raw(&m(&format!("attn_res.mlp_queries.{i}")), DType::F32, Some(&[D]))?,
                post_norm: l.raw(&n("mlp_norm.weight"), DType::BF16, Some(&[D]))?,
                attn,
                attn_conv: l.raw(&n("attn_sconv.weight"), DType::BF16, Some(&[D, 1, CONVK]))?,
                mlp_conv: l.raw(&n("mlp_sconv.weight"), DType::BF16, Some(&[D, 1, CONVK]))?,
                router: l.concat(&[n("mlp.gate.weight")], D)?,
                bias: l.raw(&n("mlp.gate.bias"), DType::F32, Some(&[EXPERTS]))?,
                fp4,
                gate_up,
                down,
                tables,
                gates,
            });
        }
        // Message queries: per loop layer [query, mlp] and [key, value] in two tensors.
        let nloop = cfg.loop_end - cfg.loop_start;
        let qm = l.f32s(&m("middle_cycle_cross_pass_queries"), &[2 * nloop, D])?;
        let kv = l.f32s(&m("middle_cycle_cross_pass_kv_queries"), &[2 * nloop, D])?;
        let mut mq = Vec::with_capacity(nloop * 4 * D);
        for j in 0..nloop {
            for src in [&qm[2 * j * D..][..D], &kv[2 * j * D..][..D], &kv[(2 * j + 1) * D..][..D], &qm[(2 * j + 1) * D..][..D]] {
                mq.extend_from_slice(src);
            }
        }
        let msg_queries = l.upload_slice(&mq)?;
        let final_query = l.raw(&m("attn_res.output_query"), DType::F32, Some(&[D]))?;
        let norm = l.raw(&m("norm.weight"), DType::BF16, Some(&[D]))?;
        let mut lm_head = l.concat(&[m("unembed.weight")], D)?;
        ensure!(lm_head.n >= cfg.vocab, "unembed has {} rows", lm_head.n);
        lm_head.n = cfg.vocab;
        let drafter = if std::env::var("QUENCH6K_INK_DRAFT").is_ok_and(|v| v == "0") || !ckpt.tensors.contains_key("fuse_weight") {
            None
        } else {
            Some(l.drafter(dir, &cfg, &lm_head)?)
        };
        s.synchronize()?;
        let device_bytes = l.bytes;
        let buffers = std::mem::take(&mut l.buffers);
        eprintln!(
            "inkling: loaded in {:.1}s, {:.2} GiB on the device",
            t0.elapsed().as_secs_f64(),
            device_bytes as f64 / (1u64 << 30) as f64
        );
        Ok(Self {
            cfg,
            layers,
            embed,
            embed_norm,
            msg_queries,
            final_query,
            norm,
            lm_head,
            drafter,
            device_bytes,
            _buffers: buffers,
        })
    }
}

/// BF16 projections quantized to block-128 FP8 at load. QUENCH6K_INK_FP8: comma-separated name fragments
/// (e.g. `attn.w,unembed,layers.`), "all" for every projection except the router and message gates, or
/// "none". Default: the LM head (no measurable change in the teacher-forced distributions, while FP8
/// layer projections move them by about half the BF16 noise floor).
fn fp8_at_load(name: &str) -> bool {
    static SPEC: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let spec = SPEC.get_or_init(|| {
        std::env::var("QUENCH6K_INK_FP8")
            .unwrap_or_else(|_| "model.llm.unembed".into())
            .split(',')
            .filter(|s| !s.is_empty() && *s != "none")
            .map(str::to_string)
            .collect()
    });
    // The router stays BF16 (routing decisions); the message gates' kernels read BF16 weights.
    if name.ends_with("mlp.gate.weight") || name.contains("controllers.") {
        return false;
    }
    spec.iter().any(|f| f == "all" || name.contains(f.as_str()))
}

/// Projections quantized to INT8 with a scale per 64 weights at load (QUENCH6K_INK_INT8 overrides): the
/// layers' and the drafter's. On the gate's fixtures the generated positions stay as close to the FP32
/// reference as the BF16 reference is (FP8 with a scale per 128 x 128 block: about 50% farther).
const INT8_DEFAULT: &str = "model.llm.layers,.self_attn.,.mlp.";

/// Whether `name` matches the env list `var` (comma-separated name fragments, "all", "none"; `default`
/// when unset), never the router or the message gates (their kernels read BF16).
fn quantized_at_load(var: &str, default: &str, name: &str) -> bool {
    if name.ends_with("mlp.gate.weight") || name.contains("controllers.") {
        return false;
    }
    let spec = std::env::var(var).unwrap_or_else(|_| default.into());
    spec.split(',').filter(|f| !f.is_empty() && *f != "none").any(|f| f == "all" || name.contains(f))
}

struct Loader<'a> {
    ckpt: &'a Checkpoint,
    s: &'a Stream,
    staging: quench6k_cuda::PinnedBuffer,
    buffers: Vec<DeviceBuffer>,
    bytes: usize,
}

impl Loader<'_> {
    fn alloc(&mut self, bytes: usize) -> Result<u64> {
        let b = DeviceBuffer::new(bytes.max(16))?;
        let a = b.addr();
        self.buffers.push(b);
        self.bytes += bytes;
        Ok(a)
    }

    /// Copies a tensor from the checkpoint to device address `dst`; returns its size.
    fn copy_tensor(&mut self, name: &str, dst: u64) -> Result<usize> {
        let info = self.ckpt.get(name)?.clone();
        let len = info.bytes as usize;
        let buf = self.staging.as_mut_slice();
        crate::fileio::read_exact_at(&self.ckpt.files[info.file].file, &mut buf[..len], info.offset)?;
        unsafe { self.s.copy_h2d(dst, self.staging.as_ptr(), len)? };
        self.s.synchronize()?;
        Ok(len)
    }

    fn read(&mut self, name: &str) -> Result<Vec<u8>> {
        self.ckpt.read(name)
    }

    fn upload_slice<T: Copy>(&mut self, v: &[T]) -> Result<u64> {
        let a = self.alloc(std::mem::size_of_val(v))?;
        self.s.upload_to(a, v)?;
        self.s.synchronize()?;
        Ok(a)
    }

    /// A tensor uploaded as stored (dtype and, when given, shape checked).
    fn raw(&mut self, name: &str, dtype: DType, shape: Option<&[usize]>) -> Result<u64> {
        let info = self.ckpt.get(name)?;
        ensure!(info.dtype == dtype, "{name}: {:?}, expected {dtype:?}", info.dtype);
        if let Some(sh) = shape {
            ensure!(info.shape == sh, "{name}: shape {:?}, expected {sh:?}", info.shape);
        }
        let a = self.alloc(info.bytes as usize)?;
        self.copy_tensor(name, a)?;
        Ok(a)
    }

    fn f32s(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let info = self.ckpt.get(name)?;
        ensure!(info.dtype == DType::F32 && info.shape == shape, "{name}: {:?} {:?}", info.dtype, info.shape);
        Ok(self.read(name)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// BF16 matrices of `k` columns stacked row-wise into one projection; quantized at load to block-128
    /// FP8 when the first name matches QUENCH6K_INK_FP8 (see [`fp8_at_load`]).
    fn concat(&mut self, names: &[String], k: usize) -> Result<Linear> {
        let mut rows = 0;
        for name in names {
            let info = self.ckpt.get(name)?;
            ensure!(info.dtype == DType::BF16, "{name}: {:?}, expected BF16", info.dtype);
            ensure!(info.shape.len() == 2 && info.shape[1] == k, "{name}: shape {:?}, expected [_, {k}]", info.shape);
            rows += info.shape[0];
        }
        let group = quench6k_cuda::kernels::int8_group();
        let int8 = k.is_multiple_of(group) && quantized_at_load("QUENCH6K_INK_INT8", INT8_DEFAULT, &names[0]);
        let fp8 = !int8 && k.is_multiple_of(128) && fp8_at_load(&names[0]);
        let tmp = if fp8 || int8 { Some(DeviceBuffer::new(rows * k * 2)?) } else { None };
        let w = match &tmp {
            Some(b) => b.addr(),
            None => self.alloc(rows * k * 2)?,
        };
        let mut at = 0;
        for name in names {
            at += self.copy_tensor(name, w + at as u64)?;
        }
        if int8 {
            let q = self.alloc(rows * k)?;
            let scale = self.alloc(rows * (k / group) * 4)?;
            quench6k_cuda::kernels::quantize_int8_groups(self.s, w, rows, k, q, scale)?;
            self.s.synchronize()?;
            return Ok(Linear {
                w: q,
                scale,
                fmt: WeightFormat::Int8Group,
                n: rows,
                k,
            });
        }
        if fp8 {
            let q = self.alloc(rows * k)?;
            let scale = self.alloc(rows.div_ceil(128) * (k / 128) * 4)?;
            quench6k_cuda::hy4::quantize_fp8b(self.s, w, rows, k, q, scale)?;
            self.s.synchronize()?;
            return Ok(Linear {
                w: q,
                scale,
                fmt: WeightFormat::Fp8Block,
                n: rows,
                k,
            });
        }
        Ok(Linear {
            w,
            scale: 0,
            fmt: WeightFormat::Bf16,
            n: rows,
            k,
        })
    }

    /// NVFP4 expert stack `[128, rows, cols]`: codes, E4M3 scales and s2 = amax / outer_scale_denominator.
    fn nvfp4_experts(&mut self, base: &str, rows: usize, cols: usize) -> Result<ExpertStack> {
        let w = self.raw(base, DType::U8, Some(&[EXPERTS, rows, cols / 2]))?;
        let sc = self.raw(&format!("{base}_scale"), DType::F8E4M3, Some(&[EXPERTS, rows, cols / 16]))?;
        let amax = self.f32s(&format!("{base}_amax"), &[EXPERTS])?;
        let den = self.f32s(&format!("{base}_outer_scale_denominator"), &[1])?[0];
        let s2: Vec<f32> = amax.iter().map(|a| a / den).collect();
        Ok(ExpertStack {
            w,
            sc,
            s2: self.upload_slice(&s2)?,
        })
    }

    /// MXFP8 expert stack `[128, rows, cols]`: E4M3 values and UE8M0 scales per 32 columns.
    fn mxfp8_experts(&mut self, base: &str, rows: usize, cols: usize) -> Result<ExpertStack> {
        Ok(ExpertStack {
            w: self.raw(base, DType::F8E4M3, Some(&[EXPERTS, rows, cols]))?,
            sc: self.raw(&format!("{base}_scale"), DType::U8, Some(&[EXPERTS, rows, cols / 32]))?,
            s2: 0,
        })
    }

    fn bf16s(&mut self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        let info = self.ckpt.get(name)?;
        ensure!(info.dtype == DType::BF16 && info.shape == shape, "{name}: {:?} {:?}", info.dtype, info.shape);
        Ok(self.read(name)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }

    /// The DSpark drafter of `mtp.safetensors` (mtp_config of config.json).
    fn drafter(&mut self, dir: &Path, cfg: &Config, lm_head: &Linear) -> Result<Drafter> {
        let all: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let m = all.get("mtp_config").context("config: mtp_config")?;
        for (key, want) in [("hidden_size", D), ("num_attention_heads", H), ("num_key_value_heads", KVH), ("head_dim", HD), ("markov_rank", 256)] {
            ensure!(num(m, key)? as usize == want, "mtp_config: {key}");
        }
        ensure!(m.get("markov_head_type").and_then(Value::as_str) == Some("rnn_hidden"), "mtp_config: markov_head_type");
        let n_layers = num(m, "num_hidden_layers")? as usize;
        let ff = num(m, "intermediate_size")? as usize;
        let ints = |key: &str| -> Result<Vec<usize>> {
            Ok(m.get(key)
                .and_then(Value::as_array)
                .with_context(|| format!("mtp_config: {key}"))?
                .iter()
                .filter_map(Value::as_u64)
                .map(|v| v as usize)
                .collect())
        };
        let local = ints("local_layer_ids")?;
        let targets = ints("target_layer_ids")?;
        let n_visits = cfg.visits().len();
        ensure!(
            targets.len() == 4 && targets.last() == Some(&n_visits) && targets.windows(2).all(|w| w[0] < w[1]),
            "mtp_config: target_layer_ids {targets:?} (4 sorted visits, the last the final output {n_visits})"
        );
        let fuse = self.bf16s("fuse_weight", &[n_layers + 1, 4])?;
        let mut alpha = Vec::with_capacity(fuse.len());
        for row in fuse.chunks(4) {
            let v: Vec<f32> = row.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
            let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = v.iter().map(|x| (x - mx).exp()).collect();
            let sum: f32 = e.iter().sum();
            alpha.extend(e.iter().map(|x| x / sum));
        }
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let n = |x: &str| format!("layers.{i}.{x}");
            let a = |x: &str| n(&format!("self_attn.{x}"));
            let gate_up = self.concat(&[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")], D)?;
            ensure!(gate_up.n == 2 * ff, "drafter layer {i}: MLP width");
            layers.push(DraftLayer {
                local: local.contains(&i),
                in_norm: self.raw(&n("input_layernorm.weight"), DType::BF16, Some(&[D]))?,
                post_norm: self.raw(&n("post_attention_layernorm.weight"), DType::BF16, Some(&[D]))?,
                mix_norm: self.raw(&n("dflare_norm.weight"), DType::BF16, Some(&[D]))?,
                qkv: self.concat(&[a("q_proj.weight"), a("k_proj.weight"), a("v_proj.weight")], D)?,
                kv_ctx: self.concat(&[a("k_proj_ctx.weight"), a("v_proj_ctx.weight")], D)?,
                o: self.concat(&[a("o_proj.weight")], QD)?,
                q_norm: self.raw(&a("q_norm.weight"), DType::BF16, Some(&[HD]))?,
                k_norm: self.raw(&a("k_norm.weight"), DType::BF16, Some(&[HD]))?,
                k_norm_ctx: self.raw(&a("k_norm_ctx.weight"), DType::BF16, Some(&[HD]))?,
                gate_up,
                down: self.concat(&[n("mlp.down_proj.weight")], ff)?,
            });
        }
        // Markov head: rnn_proj [768, 5120] columns [state | previous token | hidden | f_rnn] split in two.
        const R: usize = 256;
        let rnn = self.bf16s("markov_head.rnn_proj.weight", &[3 * R, 2 * R + 2 * D])?;
        let (mut sp, mut hf) = (Vec::with_capacity(3 * R * 2 * R), Vec::with_capacity(3 * R * 2 * D));
        for row in rnn.chunks(2 * R + 2 * D) {
            sp.extend_from_slice(&row[..2 * R]);
            hf.extend_from_slice(&row[2 * R..]);
        }
        let markov_hf = Linear {
            w: self.upload_slice(&hf)?,
            scale: 0,
            fmt: WeightFormat::Bf16,
            n: 3 * R,
            k: 2 * D,
        };
        // hidden_out folded into the LM head: fold_w [vocab, 256] = head . hidden_out, fold_b = head . bias.
        let ho = self.bf16s("markov_head.hidden_out.weight", &[D, R])?;
        let mut ho_t = vec![0u16; R * D];
        for d in 0..D {
            for r in 0..R {
                ho_t[r * D + d] = ho[d * R + r];
            }
        }
        let ho_t = self.upload_slice(&ho_t)?;
        let ho_b = self.raw("markov_head.hidden_out.bias", DType::BF16, Some(&[D]))?;
        let vocab = cfg.vocab;
        // The fold reads the BF16 unembedding (the LM head itself may be FP8).
        let _ = lm_head;
        let head = DeviceBuffer::new(self.ckpt.get("model.llm.unembed.weight")?.bytes as usize)?;
        self.copy_tensor("model.llm.unembed.weight", head.addr())?;
        let tmp = DeviceBuffer::new(vocab * R * 4)?;
        quench6k_cuda::kernels::gemm_f32(self.s, head.addr(), D, WeightFormat::Bf16, ho_t, D, 0, tmp.addr(), R, vocab, R, D, 1.0, false)?;
        let fold_w = self.alloc(vocab * R * 2)?;
        quench6k_cuda::kernels::f32_to_bf16(self.s, tmp.addr(), fold_w, vocab * R)?;
        let fold_b = self.alloc(vocab * 4)?;
        quench6k_cuda::kernels::gemv(self.s, 1, ho_b, D, WeightFormat::Bf16, head.addr(), 0, fold_b, vocab, 0, vocab, D, 1, 1.0, false)?;
        self.s.synchronize()?;
        drop(tmp);
        drop(head);
        let markov = MarkovW {
            w: self.upload_slice(&sp)?,
            bias: self.raw("markov_head.rnn_proj.bias", DType::BF16, Some(&[3 * R]))?,
            w1: self.raw("markov_head.markov_w1.weight", DType::BF16, Some(&[cfg.embed_rows, R]))?,
            fold_w,
            fold_b,
        };
        Ok(Drafter {
            layers,
            norm: self.raw("norm.weight", DType::BF16, Some(&[D]))?,
            mask: self.raw("mask_embed", DType::BF16, Some(&[D]))?,
            alpha: self.upload_slice(&alpha)?,
            visits: targets[..3].to_vec(),
            markov,
            markov_hf,
            markov_norm: self.raw("markov_head.dflare_norm.weight", DType::BF16, Some(&[D]))?,
            theta: num(m, "rope_theta")? as f32,
            window: num(m, "sliding_window_size")? as usize,
            block: num(m, "block_size")? as usize,
        })
    }
}
