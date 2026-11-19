//! Hy4-flash (`hy_v4_flash`, tencent/Hy4-flash) text model: configuration, weights on the device and the
//! host-resident tables (token embedding, Engram n-gram tables). The forward pass is in [`forward`].
//!
//! Supported checkpoint layout: routed experts NVFP4 stacked per layer (`experts.{gate,up,down}_proj`
//! codes `[512, out, in/2]`, E4M3 block scales, one FP32 scale per expert), attention projections BF16 or
//! FP8 with a scale per 128 x 128 block (`weight_scale_inv`), Engram tables FP8 with one FP32 scale per row,
//! everything else BF16/F32 as in the original release (Avifenesh/Hy4-flash-NVFP4).

pub mod forward;
pub mod hash;

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use quench6k_cuda::hy4::Experts;
use quench6k_cuda::kernels::WeightFormat;
use quench6k_cuda::{DeviceBuffer, Stream};
use serde_json::Value;

use crate::checkpoint::{Checkpoint, DType};

pub const D: usize = 3072;
pub const HC: usize = 4;
pub const Q_LORA: usize = 1024;
pub const KV_LORA: usize = 512;
pub const ROPE: usize = 64;
pub const NOPE: usize = 192;
pub const QK: usize = NOPE + ROPE;
pub const VD: usize = 256;
pub const KROW: usize = KV_LORA + ROPE;
pub const LAT: usize = 1536;
pub const EXPERTS: usize = 512;
pub const TOPK: usize = 16;
pub const IH: usize = 16;
pub const ID: usize = 128;
pub const GROUP: usize = 4;
pub const TOPG: usize = 512;
pub const WINDOW: usize = 512;
pub const ENGRAM_HEADS: usize = 16;
pub const ENGRAM_DIM: usize = 96;
pub const ENGRAM_W: usize = ENGRAM_HEADS * ENGRAM_DIM;

pub struct Config {
    pub layers: usize,
    pub vocab: usize,
    pub sliding: Vec<bool>,
    pub dense_mlp: Vec<bool>,
    pub heads_indexed: usize,
    pub heads_sliding: usize,
    pub eps: f32,
    pub hc_eps: f32,
    pub hc_mag: f32,
    pub theta_sliding: f32,
    pub theta_indexed: f32,
    pub scaling: f32,
    pub swiglu_limit: f32,
    pub engram_layers: Vec<usize>,
    pub engram_seed: u64,
    pub engram_pad: i64,
    pub engram_max_ngram: usize,
    pub engram_sizes: std::collections::HashMap<usize, Vec<i64>>,
    pub image_token: Option<i64>,
    pub video_token: Option<i64>,
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
        ensure!(arch == "hy_v4_flash", "config: model_type {arch:?} is not hy_v4_flash");
        let list = |key: &str| -> Result<Vec<String>> {
            Ok(t.get(key)
                .and_then(Value::as_array)
                .with_context(|| format!("config: {key} missing"))?
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect())
        };
        let layers = num(t, "num_hidden_layers")? as usize;
        let lt = list("layer_types")?;
        let mt = list("mlp_layer_types")?;
        ensure!(lt.len() == layers && mt.len() == layers, "config: layer type lists");
        let sliding: Vec<bool> = lt.iter().map(|s| s == "sliding_attention").collect();
        for s in &lt {
            ensure!(
                matches!(s.as_str(), "sliding_attention" | "deepseek_sparse_attention" | "indexed_attention"),
                "config: unknown layer type {s}"
            );
        }
        let dense_mlp = mt.iter().map(|s| s == "dense").collect();
        for (key, want) in [
            ("hidden_size", D),
            ("hc_mult", HC),
            ("q_lora_rank", Q_LORA),
            ("kv_lora_rank", KV_LORA),
            ("qk_rope_head_dim", ROPE),
            ("qk_nope_head_dim", NOPE),
            ("v_head_dim", VD),
            ("swa_qk_rope_head_dim", ROPE),
            ("swa_qk_nope_head_dim", NOPE),
            ("swa_v_head_dim", VD),
            ("routed_expert_hidden_size", LAT),
            ("moe_intermediate_size", LAT),
            ("n_routed_experts", EXPERTS),
            ("num_experts_per_tok", TOPK),
            ("n_group", 1),
            ("compressed_indexer_num_attention_heads", IH),
            ("compressed_indexer_head_dim", ID),
            ("compressed_indexer_compress_ratio", GROUP),
            ("compressed_indexer_group_topk", TOPG),
            ("sliding_window", WINDOW),
            ("engram_n_head_per_ngram", ENGRAM_HEADS / 2),
            ("engram_n_embed_per_ngram", ENGRAM_DIM * ENGRAM_HEADS / 2),
            ("engram_max_ngram_size", 3),
        ] {
            let v = num(t, key)? as usize;
            ensure!(v == want, "config: {key} = {v}, this engine supports {want}");
        }
        ensure!(t.get("use_compressed_indexer").and_then(Value::as_bool) == Some(true), "config: indexer");
        ensure!(t.get("engram_fp8_embedding").and_then(Value::as_bool) == Some(true), "config: engram FP8");
        let rope = t.get("rope_parameters").context("config: rope_parameters")?;
        let theta = |k: &str| -> Result<f32> {
            let p = rope.get(k).with_context(|| format!("config: rope {k}"))?;
            ensure!(
                p.get("rope_type").and_then(Value::as_str) == Some("default"),
                "config: rope type of {k}"
            );
            Ok(num(p, "rope_theta")? as f32)
        };
        let theta_indexed = theta("deepseek_sparse_attention").or_else(|_| theta("indexed_attention"))?;
        let mut engram_sizes = std::collections::HashMap::new();
        let ids: Vec<usize> = t
            .get("engram_layer_ids")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
            .unwrap_or_default();
        let sizes = t.get("engram_layer_head_vocab_sizes");
        for id in &ids {
            let orders = sizes
                .and_then(|s| s.get(id.to_string()))
                .and_then(Value::as_array)
                .with_context(|| format!("config: engram sizes of layer {id}"))?;
            let flat: Vec<i64> = orders
                .iter()
                .flat_map(|o| o.as_array().into_iter().flatten().filter_map(Value::as_i64))
                .collect();
            ensure!(flat.len() == ENGRAM_HEADS, "config: engram sizes of layer {id}");
            engram_sizes.insert(*id, flat);
        }
        let eos = match t.get("eos_token_id") {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_i64).map(|v| v as i32).collect(),
            Some(v) => v.as_i64().map(|v| vec![v as i32]).unwrap_or_default(),
            None => vec![],
        };
        Ok(Self {
            layers,
            vocab: num(t, "vocab_size")? as usize,
            sliding,
            dense_mlp,
            heads_indexed: num(t, "num_attention_heads")? as usize,
            heads_sliding: num(t, "swa_num_attention_heads")? as usize,
            eps: num(t, "rms_norm_eps")? as f32,
            hc_eps: num(t, "hc_eps")? as f32,
            hc_mag: num(t, "hc_magnitude")? as f32,
            theta_sliding: theta("sliding_attention")?,
            theta_indexed,
            scaling: num(t, "routed_scaling_factor")? as f32,
            swiglu_limit: num(t, "swiglu_limit")? as f32,
            engram_layers: ids.iter().map(|id| id - 1).collect(),
            engram_seed: num(t, "engram_seed")? as u64,
            engram_pad: num(t, "engram_pad_id")? as i64,
            engram_max_ngram: 3,
            engram_sizes,
            image_token: all.get("image_token_id").and_then(Value::as_i64),
            video_token: all.get("video_token_id").and_then(Value::as_i64),
            eos,
        })
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

pub struct Hc {
    pub fn_: u64,
    pub base: u64,
    pub scale: u64,
}

pub struct Indexer {
    pub q: Linear,
    pub hw: Linear,
    pub tk: Linear,
    pub tk_norm: u64,
    pub wv: u64,
    pub wg: u64,
    pub ape: u64,
    pub key_norm: u64,
}

pub struct Attn {
    pub heads: usize,
    pub sliding: bool,
    pub q_a: Linear,
    pub q_a_norm: u64,
    pub q_b: Linear,
    pub kv_a: Linear,
    pub kv_a_norm: u64,
    /// kv_b_proj BF16 `[heads * 448, 512]`: per head 192 key rows then 256 value rows.
    pub kv_b: u64,
    /// The key rows of each head transposed: BF16 `[heads, 512, 192]`.
    pub wuk_t: u64,
    pub gate: Linear,
    pub o: Linear,
    pub post_norm: u64,
    pub sinks: u64,
    pub indexer: Option<Indexer>,
}

/// Routed experts of one layer plus the per-expert pointer tables of the grouped GEMM.
pub struct RoutedExperts {
    pub gate: Experts,
    pub up: Experts,
    pub down: Experts,
    /// (codes, scales) pointer tables, `[512]` u64 each, for gate, up and down.
    pub tables: [(u64, u64); 3],
}

pub struct Moe {
    pub router: Linear,
    pub bias: u64,
    pub down_lat: Linear,
    pub up_lat: Linear,
    pub experts: RoutedExperts,
    pub shared_gate: Linear,
    pub shared_up: Linear,
    pub shared_down: Linear,
    pub norm: u64,
}

pub struct Dense {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub norm: u64,
}

pub enum Mlp {
    Moe(Moe),
    Dense(Dense),
}

pub struct Engram {
    pub hash: hash::EngramHash,
    table: Mmap,
    table_off: usize,
    scales: Mmap,
    scales_off: usize,
    pub head_norm: u64,
    /// key_projs 0..3 and value_proj stacked: `[5 * 3072, 1536]` BF16, bias `[5 * 3072]`.
    pub kv: Linear,
    pub bias: u64,
    pub norm_key: u64,
    pub norm_query: u64,
    pub gamma: u64,
    pub hc_scale: u64,
}

impl Engram {
    /// Starts reading row `r` into the page cache (no-op elsewhere).
    pub fn prefetch(&self, r: usize) {
        #[cfg(unix)]
        {
            let _ = self.table.advise_range(memmap2::Advice::WillNeed, self.table_off + r * ENGRAM_DIM, ENGRAM_DIM);
            let _ = self.scales.advise_range(memmap2::Advice::WillNeed, self.scales_off + r * 4, 4);
        }
        #[cfg(not(unix))]
        let _ = r;
    }

    /// Reads the whole table and its scales into the page cache with `threads` threads.
    pub fn preload(&self, threads: usize) -> u8 {
        let total = self.hash.moduli.iter().sum::<i64>() as usize;
        let table = &self.table[self.table_off..self.table_off + total * ENGRAM_DIM];
        let sc = &self.scales[self.scales_off..self.scales_off + total * 4];
        let mut sink = 0u8;
        for part in [table, sc] {
            let chunk = part.len().div_ceil(threads).max(4096);
            sink ^= std::thread::scope(|scope| {
                let handles: Vec<_> = part
                    .chunks(chunk)
                    .map(|c| {
                        scope.spawn(move || {
                            let mut x = 0u8;
                            for i in (0..c.len()).step_by(4096) {
                                x ^= unsafe { std::ptr::read_volatile(&c[i]) };
                            }
                            x
                        })
                    })
                    .collect();
                handles.into_iter().fold(0u8, |a, h| a ^ h.join().unwrap())
            });
        }
        sink
    }

    /// FP8 row `r` (96 bytes) and its scale.
    pub fn row(&self, r: usize) -> (&[u8], f32) {
        let o = self.table_off + r * ENGRAM_DIM;
        let s = self.scales_off + r * 4;
        (
            &self.table[o..o + ENGRAM_DIM],
            f32::from_le_bytes(self.scales[s..s + 4].try_into().unwrap()),
        )
    }
}

pub struct Layer {
    pub attn_hc: Hc,
    pub mlp_hc: Hc,
    pub in_norm: u64,
    pub post_attn_norm: u64,
    pub attn: Attn,
    pub mlp: Mlp,
    pub engram: Option<Engram>,
}

pub struct Hy4Model {
    pub cfg: Config,
    /// Layers whose Engram memory is loaded (a subset of cfg.engram_layers when tables are missing).
    pub engram_layers: Vec<usize>,
    pub layers: Vec<Layer>,
    pub hc_head: Hc,
    pub norm: u64,
    pub lm_head: Linear,
    embed: Mmap,
    embed_off: usize,
    pub device_bytes: usize,
    _buffers: Vec<DeviceBuffer>,
}

impl Hy4Model {
    pub fn is_hy4(dir: &Path) -> bool {
        std::fs::read(dir.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .is_some_and(|v| v.get("model_type").and_then(Value::as_str) == Some("hy_v4_flash"))
    }

    /// BF16 embedding row of `token`.
    pub fn embed_row(&self, token: usize) -> &[u8] {
        let o = self.embed_off + token * D * 2;
        &self.embed[o..o + D * 2]
    }

    pub fn load(dir: &Path, s: &Stream) -> Result<Self> {
        let cfg = Config::read(dir)?;
        let ckpt = Checkpoint::open(dir)?;
        let t0 = std::time::Instant::now();
        // Weights are read once: direct I/O keeps them out of the page cache, which holds the Engram tables.
        let largest = ckpt
            .tensors
            .iter()
            .filter(|(n, _)| !n.contains("multi_head_embedding") && !n.contains("embed_tokens"))
            .map(|(_, t)| t.bytes as usize)
            .max()
            .unwrap_or(0);
        let direct: Option<Vec<std::fs::File>> = ckpt.files.iter().map(|f| crate::fileio::open_direct(&f.path)).collect();
        let mut l = Loader {
            ckpt: &ckpt,
            s,
            staging: quench6k_cuda::PinnedBuffer::new(largest + 2 * DIRECT_ALIGN)?,
            direct,
            buffers: Vec::new(),
            bytes: 0,
        };
        let (embed, embed_off) = l.map("model.embed_tokens.weight", DType::BF16, &[cfg.vocab, D])?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let n = |x: &str| format!("model.layers.{i}.{x}");
            let hc = |l: &mut Loader, which: &str| -> Result<Hc> {
                Ok(Hc {
                    fn_: l.raw(&n(&format!("{which}.hc_pre.hc_fn")), DType::F32, Some(&[2 * HC, HC * D]))?,
                    base: l.raw(&n(&format!("{which}.hc_pre.hc_base")), DType::F32, Some(&[2 * HC]))?,
                    scale: l.raw(&n(&format!("{which}.hc_pre.hc_scale")), DType::F32, Some(&[2]))?,
                })
            };
            let attn_hc = hc(&mut l, "hc_attn_layer")?;
            let mlp_hc = hc(&mut l, "hc_mlp_layer")?;
            let in_norm = l.raw(&n("input_layernorm.weight"), DType::BF16, Some(&[D]))?;
            let post_attn_norm = l.raw(&n("post_attention_layernorm.weight"), DType::BF16, Some(&[D]))?;
            let sliding = cfg.sliding[i];
            let heads = if sliding { cfg.heads_sliding } else { cfg.heads_indexed };
            let a = |x: &str| n(&format!("self_attn.{x}"));
            let kv_b_name = a("kv_b_proj.weight");
            let kv_b_bytes = l.expect(&kv_b_name, DType::BF16, &[heads * (NOPE + VD), KV_LORA])?;
            let kv_b = l.upload_bytes(&kv_b_bytes)?;
            // Transposed key rows: wuk_t[h][j][i] = kv_b[h * 448 + i][j].
            let src: Vec<u16> = kv_b_bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let mut wuk = vec![0u16; heads * KV_LORA * NOPE];
            for h in 0..heads {
                for i2 in 0..NOPE {
                    let row = &src[(h * (NOPE + VD) + i2) * KV_LORA..][..KV_LORA];
                    for (j, &v) in row.iter().enumerate() {
                        wuk[(h * KV_LORA + j) * NOPE + i2] = v;
                    }
                }
            }
            let wuk_t = l.upload_slice(&wuk)?;
            let indexer = if sliding {
                None
            } else {
                let x = |y: &str| a(&format!("compressed_indexer.{y}"));
                Some(Indexer {
                    q: l.linear(&x("query_projection.weight"), Some((IH * ID, Q_LORA)))?,
                    hw: l.linear(&x("head_weight_projection.weight"), Some((IH, D)))?,
                    tk: l.linear(&x("token_key_projection.weight"), Some((ID, D)))?,
                    tk_norm: l.raw(&x("token_key_norm.weight"), DType::BF16, Some(&[ID]))?,
                    wv: l.raw(&x("key_value_projection.weight"), DType::F32, Some(&[ID, ID]))?,
                    wg: l.raw(&x("key_gate_projection.weight"), DType::F32, Some(&[ID, ID]))?,
                    ape: l.raw(&x("intra_group_ape"), DType::F32, Some(&[GROUP, ID]))?,
                    key_norm: l.raw(&x("key_norm.weight"), DType::F32, Some(&[ID]))?,
                })
            };
            let attn = Attn {
                heads,
                sliding,
                q_a: l.linear(&a("q_a_proj.weight"), Some((Q_LORA, D)))?,
                q_a_norm: l.raw(&a("q_a_layernorm.weight"), DType::BF16, Some(&[Q_LORA]))?,
                q_b: l.linear(&a("q_b_proj.weight"), Some((heads * QK, Q_LORA)))?,
                kv_a: l.linear(&a("kv_a_proj_with_mqa.weight"), Some((KROW, D)))?,
                kv_a_norm: l.raw(&a("kv_a_layernorm.weight"), DType::BF16, Some(&[KV_LORA]))?,
                kv_b,
                wuk_t,
                gate: l.linear(&a("linear_gate.weight"), Some((heads * VD, D)))?,
                o: l.linear(&a("o_proj.weight"), Some((D, heads * VD)))?,
                post_norm: l.raw(&a("post_proj_norm.weight"), DType::BF16, Some(&[D]))?,
                sinks: l.raw(&a("learnable_sink_param"), DType::F32, Some(&[heads]))?,
                indexer,
            };
            let m = |x: &str| n(&format!("mlp.{x}"));
            let mlp = if cfg.dense_mlp[i] {
                Mlp::Dense(Dense {
                    gate: l.linear(&m("gate_proj.weight"), None)?,
                    up: l.linear(&m("up_proj.weight"), None)?,
                    down: l.linear(&m("down_proj.weight"), None)?,
                    norm: l.raw(&m("post_fc2_norm.weight"), DType::BF16, Some(&[D]))?,
                })
            } else {
                let ex = |l: &mut Loader, which: &str| -> Result<Experts> {
                    let b = m(&format!("experts.{which}_proj"));
                    Ok(Experts {
                        codes: l.raw(&format!("{b}.weight"), DType::U8, Some(&[EXPERTS, LAT, LAT / 2]))?,
                        scales: l.raw(&format!("{b}.weight_scale"), DType::F8E4M3, Some(&[EXPERTS, LAT, LAT / 16]))?,
                        s2: l.raw(&format!("{b}.weight_scale_2"), DType::F32, Some(&[EXPERTS]))?,
                    })
                };
                let (gate, up, down) = (ex(&mut l, "gate")?, ex(&mut l, "up")?, ex(&mut l, "down")?);
                let mut tables = [(0, 0); 3];
                for (slot, e) in [gate, up, down].iter().enumerate() {
                    let c: Vec<u64> = (0..EXPERTS).map(|x| e.codes + (x * LAT * LAT / 2) as u64).collect();
                    let sc: Vec<u64> = (0..EXPERTS).map(|x| e.scales + (x * LAT * LAT / 16) as u64).collect();
                    tables[slot] = (l.upload_slice(&c)?, l.upload_slice(&sc)?);
                }
                Mlp::Moe(Moe {
                    router: l.linear(&m("gate.weight"), Some((EXPERTS, D)))?,
                    bias: l.raw(&m("gate.e_score_correction_bias"), DType::F32, Some(&[EXPERTS]))?,
                    down_lat: l.linear(&m("routed_expert_down_proj.weight"), Some((LAT, D)))?,
                    up_lat: l.linear(&m("routed_expert_up_proj.weight"), Some((D, LAT)))?,
                    experts: RoutedExperts {
                        gate,
                        up,
                        down,
                        tables,
                    },
                    shared_gate: l.linear(&m("shared_experts.gate_proj.weight"), Some((LAT, D)))?,
                    shared_up: l.linear(&m("shared_experts.up_proj.weight"), Some((LAT, D)))?,
                    shared_down: l.linear(&m("shared_experts.down_proj.weight"), Some((D, LAT)))?,
                    norm: l.raw(&m("combine_output_norm.weight"), DType::BF16, Some(&[D]))?,
                })
            };
            let table = format!("model.layers.{i}.engram_layer.multi_head_embedding.embedding.weight");
            let engram = if !cfg.engram_layers.contains(&i) {
                None
            } else if ckpt.tensors.contains_key(&table) {
                Some(l.engram(&cfg, i)?)
            } else {
                eprintln!("hy4: WARNING: {table} is missing; layer {i} runs without its Engram memory");
                None
            };
            layers.push(Layer {
                attn_hc,
                mlp_hc,
                in_norm,
                post_attn_norm,
                attn,
                mlp,
                engram,
            });
            s.synchronize()?;
            if (i + 1) % 6 == 0 {
                eprintln!(
                    "hy4: loaded {} layers, {:.1} GiB, {:.1}s",
                    i + 1,
                    l.bytes as f64 / (1u64 << 30) as f64,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        let hc_head = Hc {
            fn_: l.raw("model.hc_head.hc_head_fn", DType::F32, Some(&[HC, HC * D]))?,
            base: l.raw("model.hc_head.hc_head_base", DType::F32, Some(&[HC]))?,
            scale: l.raw("model.hc_head.hc_head_scale", DType::F32, Some(&[1]))?,
        };
        let norm = l.raw("model.norm.weight", DType::BF16, Some(&[D]))?;
        let lm_head = l.linear("lm_head.weight", Some((cfg.vocab, D)))?;
        s.synchronize()?;
        let device_bytes = l.bytes;
        let buffers = std::mem::take(&mut l.buffers);
        eprintln!(
            "hy4: loaded in {:.1}s, {:.2} GiB on the device",
            t0.elapsed().as_secs_f64(),
            device_bytes as f64 / (1u64 << 30) as f64
        );
        let engram_layers: Vec<usize> = (0..layers.len()).filter(|&i| layers[i].engram.is_some()).collect();
        if std::env::var("QUENCH6K_HY4_ENGRAM_PRELOAD").is_ok_and(|v| v != "0") {
            let t1 = std::time::Instant::now();
            for &i in &engram_layers {
                std::hint::black_box(layers[i].engram.as_ref().unwrap().preload(16));
            }
            eprintln!("hy4: Engram tables read into the page cache in {:.1}s", t1.elapsed().as_secs_f64());
        }
        Ok(Self {
            cfg,
            engram_layers,
            layers,
            hc_head,
            norm,
            lm_head,
            embed,
            embed_off,
            device_bytes,
            _buffers: buffers,
        })
    }
}

/// BF16 projections quantized to block-128 FP8 at load. QUENCH6K_HY4_FP8: comma-separated name fragments,
/// "all" for every BF16 projection except the router, "none". Default: the shared experts, the latent
/// projections around the routed experts and the LM head (the set that passes the gate; adding q_a/kv_a or
/// the dense layer does not).
fn fp8_at_load(name: &str) -> bool {
    static SPEC: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let spec = SPEC.get_or_init(|| {
        std::env::var("QUENCH6K_HY4_FP8")
            .unwrap_or_else(|_| "shared_experts,routed_expert,lm_head".into())
            .split(',')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    });
    if name.ends_with("mlp.gate.weight") {
        return false;
    }
    spec.iter().any(|f| f == "all" || name.contains(f.as_str()))
}

const DIRECT_ALIGN: usize = 4096;

struct Loader<'a> {
    ckpt: &'a Checkpoint,
    s: &'a Stream,
    staging: quench6k_cuda::PinnedBuffer,
    direct: Option<Vec<std::fs::File>>,
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

    /// Reads a tensor into the pinned staging buffer: (offset of its first byte there, length).
    fn fetch(&mut self, name: &str) -> Result<(usize, usize)> {
        let info = self.ckpt.get(name)?.clone();
        let len = info.bytes as usize;
        let buf = self.staging.as_mut_slice();
        match &self.direct {
            Some(files) => {
                let begin = info.offset as usize / DIRECT_ALIGN * DIRECT_ALIGN;
                let end = (info.offset as usize + len).div_ceil(DIRECT_ALIGN) * DIRECT_ALIGN;
                let skip = info.offset as usize - begin;
                let window = &mut buf[..end - begin];
                let mut filled = 0;
                while filled < skip + len {
                    let n = crate::fileio::read_at(&files[info.file], &mut window[filled..], (begin + filled) as u64)?;
                    ensure!(n > 0, "unexpected end of {}", self.ckpt.files[info.file].path.display());
                    filled += n;
                }
                Ok((skip, len))
            }
            None => {
                crate::fileio::read_exact_at(&self.ckpt.files[info.file].file, &mut buf[..len], info.offset)?;
                Ok((0, len))
            }
        }
    }

    fn read(&mut self, name: &str) -> Result<Vec<u8>> {
        let (o, n) = self.fetch(name)?;
        Ok(self.staging.as_slice()[o..o + n].to_vec())
    }

    /// Copies a tensor from the checkpoint to device address `dst`.
    fn copy_tensor(&mut self, name: &str, dst: u64) -> Result<usize> {
        let (o, n) = self.fetch(name)?;
        unsafe { self.s.copy_h2d(dst, self.staging.as_ptr().add(o), n)? };
        self.s.synchronize()?;
        Ok(n)
    }

    fn upload_tensor(&mut self, name: &str) -> Result<u64> {
        let n = self.ckpt.get(name)?.bytes as usize;
        let a = self.alloc(n)?;
        self.copy_tensor(name, a)?;
        Ok(a)
    }

    fn upload_bytes(&mut self, v: &[u8]) -> Result<u64> {
        let a = self.alloc(v.len())?;
        self.s.upload_to(a, v)?;
        // The pageable upload reads `v` asynchronously; finish before the caller drops it.
        self.s.synchronize()?;
        Ok(a)
    }

    fn upload_slice<T: Copy>(&mut self, v: &[T]) -> Result<u64> {
        let a = self.alloc(std::mem::size_of_val(v))?;
        self.s.upload_to(a, v)?;
        self.s.synchronize()?;
        Ok(a)
    }

    fn expect(&mut self, name: &str, dtype: DType, shape: &[usize]) -> Result<Vec<u8>> {
        let info = self.ckpt.get(name)?;
        ensure!(
            info.dtype == dtype && info.shape == shape,
            "{name}: {:?} {:?}, expected {dtype:?} {shape:?}",
            info.dtype,
            info.shape
        );
        self.read(name)
    }

    /// A tensor uploaded as stored (dtype and, when given, shape checked).
    fn raw(&mut self, name: &str, dtype: DType, shape: Option<&[usize]>) -> Result<u64> {
        let info = self.ckpt.get(name)?;
        ensure!(info.dtype == dtype, "{name}: {:?}, expected {dtype:?}", info.dtype);
        if let Some(sh) = shape {
            ensure!(info.shape == sh, "{name}: shape {:?}, expected {sh:?}", info.shape);
        }
        self.upload_tensor(name)
    }

    /// A projection: BF16, or FP8 with `<name>_scale_inv` block scales.
    fn linear(&mut self, name: &str, shape: Option<(usize, usize)>) -> Result<Linear> {
        let info = self.ckpt.get(name)?.clone();
        ensure!(info.shape.len() == 2, "{name}: not a matrix");
        let (n, k) = (info.shape[0], info.shape[1]);
        if let Some((sn, sk)) = shape {
            ensure!((n, k) == (sn, sk), "{name}: shape [{n}, {k}], expected [{sn}, {sk}]");
        }
        if info.dtype == DType::BF16 && k % 128 == 0 && fp8_at_load(name) {
            // Quantized at load to block-128 FP8 (QUENCH6K_HY4_FP8).
            let tmp = DeviceBuffer::new(n * k * 2)?;
            self.copy_tensor(name, tmp.addr())?;
            let w = self.alloc(n * k)?;
            let scale = self.alloc(n.div_ceil(128) * (k / 128) * 4)?;
            quench6k_cuda::hy4::quantize_fp8b(self.s, tmp.addr(), n, k, w, scale)?;
            self.s.synchronize()?;
            return Ok(Linear { w, scale, fmt: WeightFormat::Fp8Block, n, k });
        }
        let (fmt, scale) = match info.dtype {
            DType::BF16 => (WeightFormat::Bf16, 0),
            DType::F8E4M3 => {
                let sname = format!("{name}_scale_inv");
                ensure!(
                    k % 128 == 0,
                    "{name}: block-FP8 needs K a multiple of 128"
                );
                let scale = self.raw(&sname, DType::F32, Some(&[n.div_ceil(128), k / 128]))?;
                (WeightFormat::Fp8Block, scale)
            }
            other => bail!("{name}: unsupported projection dtype {other:?}"),
        };
        let w = self.upload_tensor(name)?;
        Ok(Linear { w, scale, fmt, n, k })
    }

    /// Memory map of a host-resident tensor: (map of its file, byte offset of the tensor in it).
    fn map(&self, name: &str, dtype: DType, shape: &[usize]) -> Result<(Mmap, usize)> {
        let info = self.ckpt.get(name)?;
        ensure!(
            info.dtype == dtype && info.shape == shape,
            "{name}: {:?} {:?}, expected {dtype:?} {shape:?}",
            info.dtype,
            info.shape
        );
        let file = &self.ckpt.files[info.file];
        let map = unsafe { Mmap::map(&file.file) }.with_context(|| format!("mapping {}", file.path.display()))?;
        Ok((map, info.offset as usize))
    }

    /// Several BF16 tensors of equal width concatenated row-wise into one device buffer.
    fn concat_bf16(&mut self, names: &[String]) -> Result<u64> {
        let mut total = 0;
        for name in names {
            let info = self.ckpt.get(name)?;
            ensure!(info.dtype == DType::BF16, "{name}: expected BF16");
            total += info.bytes as usize;
        }
        let base = self.alloc(total)?;
        let mut at = 0;
        for name in names {
            at += self.copy_tensor(name, base + at as u64)?;
        }
        Ok(base)
    }

    fn engram(&mut self, cfg: &Config, i: usize) -> Result<Engram> {
        let p = format!("model.layers.{i}.engram_layer");
        let layer_id = i + 1;
        let lt = self.expect(&format!("{p}.lookup_table"), DType::I32, &[self.ckpt.get(&format!("{p}.lookup_table"))?.shape[0]])?;
        let lookup: Vec<i32> = lt.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let moduli = cfg.engram_sizes.get(&layer_id).context("engram sizes")?.clone();
        let rows: i64 = moduli.iter().sum();
        let hash = hash::EngramHash::new(lookup, cfg.engram_seed, layer_id as u64, cfg.engram_max_ngram, moduli, cfg.engram_pad);
        let tname = format!("{p}.multi_head_embedding.embedding.weight");
        let (table, table_off) = self.map(&tname, DType::F8E4M3, &[rows as usize, ENGRAM_DIM])?;
        let (scales, scales_off) = self.map(&format!("{tname}_scale"), DType::F32, &[rows as usize])?;
        #[cfg(unix)]
        {
            // Rows are 96 bytes read at random: readahead would only evict useful pages.
            table.advise(memmap2::Advice::Random)?;
            scales.advise(memmap2::Advice::Random)?;
        }
        let names = |f: &dyn Fn(usize) -> String, n: usize| (0..n).map(f).collect::<Vec<_>>();
        let head_norm = self.concat_bf16(&names(&|h| format!("{p}.lookup_per_head_norms.{h}.weight"), ENGRAM_HEADS))?;
        let mut kv_names = names(&|c| format!("{p}.key_projs.{c}.weight"), HC);
        kv_names.push(format!("{p}.value_proj.weight"));
        for name in &kv_names {
            let info = self.ckpt.get(name)?;
            ensure!(info.shape == [D, ENGRAM_W] && info.dtype == DType::BF16, "{name}: unexpected layout");
        }
        let kv_w = self.concat_bf16(&kv_names)?;
        let mut bias_names = names(&|c| format!("{p}.key_projs.{c}.bias"), HC);
        bias_names.push(format!("{p}.value_proj.bias"));
        let bias = self.concat_bf16(&bias_names)?;
        Ok(Engram {
            hash,
            table,
            table_off,
            scales,
            scales_off,
            head_norm,
            kv: Linear {
                w: kv_w,
                scale: 0,
                fmt: WeightFormat::Bf16,
                n: (HC + 1) * D,
                k: ENGRAM_W,
            },
            bias,
            norm_key: self.concat_bf16(&names(&|c| format!("{p}.norm_key.{c}.weight"), HC))?,
            norm_query: self.concat_bf16(&names(&|c| format!("{p}.norm_query.{c}.weight"), HC))?,
            gamma: self.raw(&format!("{p}.unified_pre_gamma"), DType::BF16, Some(&[D]))?,
            hc_scale: self.concat_bf16(&names(&|c| format!("{p}.unified_per_hc_scales.{c}"), HC))?,
        })
    }
}
