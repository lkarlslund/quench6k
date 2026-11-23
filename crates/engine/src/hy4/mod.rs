//! Hy4-flash (`hy_v4_flash`, tencent/Hy4-flash) text model: configuration, weights on the device and the
//! host-resident tables (token embedding, Engram n-gram tables). The forward pass is in [`forward`].
//!
//! Supported checkpoint layout: routed experts NVFP4 stacked per layer (`experts.{gate,up,down}_proj`
//! codes `[512, out, in/2]`, E4M3 block scales, one FP32 scale per expert), attention projections BF16 or
//! FP8 with a scale per 128 x 128 block (`weight_scale_inv`), Engram tables FP8 with one FP32 scale per row,
//! everything else BF16/F32 as in the original release (Avifenesh/Hy4-flash-NVFP4).

pub mod forward;
pub mod hash;
pub mod policy;

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
    /// Longest context the model supports (max_position_embeddings).
    pub max_positions: usize,
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
            max_positions: num(t, "max_position_embeddings")? as usize,
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
    /// q_a and kv_a (and the indexer's token key and head weight projections) stacked, rows
    /// [q_a | kv_a | tk | hw], when their formats agree: one GEMV for the small steps.
    pub proj_in: Option<Linear>,
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
    /// down_lat, shared_gate and shared_up stacked, rows [down_lat | shared_gate | shared_up], when their
    /// formats agree: one GEMV for the small steps.
    pub mlp_in: Option<Linear>,
}

pub struct Dense {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub norm: u64,
    /// gate and up stacked ([gate | up]) when their formats agree.
    pub gate_up: Option<Linear>,
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
    /// Pages of the table and of its scales a read-ahead hint was given for (one bit per 4 KiB page): such
    /// pages are taken as being in the page cache and get no further hint (a hint is a system call).
    hinted: [Vec<std::sync::atomic::AtomicU64>; 2],
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
    /// Starts reading row `r` into the page cache unless its pages were hinted before (no-op elsewhere).
    pub fn prefetch(&self, r: usize) {
        #[cfg(unix)]
        {
            for (k, (map, at, len)) in [(&self.table, self.table_off + r * ENGRAM_DIM, ENGRAM_DIM), (&self.scales, self.scales_off + r * 4, 4)]
                .into_iter()
                .enumerate()
            {
                // Pages counted from the mapping's first page (by address: the mapping may start inside a page).
                let base = map.as_ptr() as usize;
                if Self::first_hint(&self.hinted[k], (base & 4095) + at, len) {
                    let _ = map.advise_range(memmap2::Advice::WillNeed, at, len);
                }
            }
        }
        #[cfg(not(unix))]
        let _ = r;
    }

    /// Marks the pages of bytes [at, at + len) as hinted; whether any was not before.
    fn first_hint(bits: &[std::sync::atomic::AtomicU64], at: usize, len: usize) -> bool {
        use std::sync::atomic::Ordering;
        let mut first = false;
        for page in at >> 12..=(at + len - 1) >> 12 {
            let (w, b) = (page / 64, 1u64 << (page % 64));
            if let Some(word) = bits.get(w) {
                first |= word.fetch_or(b, Ordering::Relaxed) & b == 0;
            } else {
                first = true;
            }
        }
        first
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

/// One layer of the DFly drafter: pre-norm attention and a dense SwiGLU MLP on a plain residual stream.
pub struct DflyLayer {
    pub in_norm: u64,
    pub post_attn_norm: u64,
    pub attn: Attn,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    /// gate and up stacked ([gate | up]) when their formats agree.
    pub gate_up: Option<Linear>,
}

/// The checkpoint's DFly drafter (`model.dfly_layers`). After a target step it drafts the tokens that follow
/// the accepted one in a single pass of a query block (that token, then mask tokens) over the target's
/// caches, then picks them in order through the hidden correction and the target's LM head.
pub struct Dfly {
    pub layers: Vec<DflyLayer>,
    pub norm: u64,
    /// Fusion of the target's last hidden state into the first query: x0 = e + gate * eh([enorm e, hnorm h]).
    pub enorm: u64,
    pub hnorm: u64,
    pub eh: Linear,
    pub fuse_gate: u64,
    /// Hidden correction: h + down(swiglu(gate c, up c)), c = [hidden_norm h, embed_norm(previous token)].
    pub c_hidden_norm: u64,
    pub c_embed_norm: u64,
    pub c_gate: Linear,
    pub c_up: Linear,
    pub c_down: Linear,
    /// c_gate and c_up stacked ([c_gate | c_up]) when their formats agree.
    pub c_gate_up: Option<Linear>,
    /// The target's token embedding on the device (BF16 [vocab, 3072]).
    pub embed: u64,
    pub mask: i32,
    /// Per drafter layer: the target layer whose key lists it reuses, and the one whose cache it reads and
    /// writes its query block into.
    pub index_layers: Vec<usize>,
    pub kv_layers: Vec<usize>,
    /// Query block width: the accepted token plus the drafts.
    pub width: usize,
}

pub struct Hy4Model {
    pub cfg: Config,
    /// Layers whose Engram memory is loaded (a subset of cfg.engram_layers when tables are missing).
    pub engram_layers: Vec<usize>,
    pub layers: Vec<Layer>,
    pub hc_head: Hc,
    pub norm: u64,
    pub lm_head: Linear,
    /// The DFly drafter (QUENCH6K_HY4_DFLY=0: not loaded).
    pub dfly: Option<Dfly>,
    /// Per layer: its index among the layers of its kind (indexed or sliding), whose caches a sequence
    /// keeps interleaved per position (indexed) or one ring after another (sliding).
    pub cache_slot: Vec<usize>,
    pub n_indexed: usize,
    pub n_sliding: usize,
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
            let attn = l.attn(&n("self_attn"), heads, sliding, !sliding)?;
            let m = |x: &str| n(&format!("mlp.{x}"));
            let mlp = if cfg.dense_mlp[i] {
                let (gate_up, gu) = l.linears(&[(&m("gate_proj.weight"), None), (&m("up_proj.weight"), None)], false)?;
                ensure!(gu[0].n == gu[1].n, "{}: gate and up widths differ", m("gate_proj.weight"));
                Mlp::Dense(Dense {
                    gate: gu[0],
                    up: gu[1],
                    down: l.linear(&m("down_proj.weight"), None)?,
                    norm: l.raw(&m("post_fc2_norm.weight"), DType::BF16, Some(&[D]))?,
                    gate_up,
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
                let (mlp_in, v) = l.linears(
                    &[
                        (&m("routed_expert_down_proj.weight"), Some((LAT, D))),
                        (&m("shared_experts.gate_proj.weight"), Some((LAT, D))),
                        (&m("shared_experts.up_proj.weight"), Some((LAT, D))),
                    ],
                    false,
                )?;
                Mlp::Moe(Moe {
                    router: l.linear(&m("gate.weight"), Some((EXPERTS, D)))?,
                    bias: l.raw(&m("gate.e_score_correction_bias"), DType::F32, Some(&[EXPERTS]))?,
                    down_lat: v[0],
                    up_lat: l.linear(&m("routed_expert_up_proj.weight"), Some((D, LAT)))?,
                    experts: RoutedExperts {
                        gate,
                        up,
                        down,
                        tables,
                    },
                    shared_gate: v[1],
                    shared_up: v[2],
                    shared_down: l.linear(&m("shared_experts.down_proj.weight"), Some((D, LAT)))?,
                    norm: l.raw(&m("combine_output_norm.weight"), DType::BF16, Some(&[D]))?,
                    mlp_in,
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
        let dfly = if std::env::var("QUENCH6K_HY4_DFLY").is_ok_and(|v| v == "0") {
            None
        } else {
            l.dfly(dir, &cfg, &embed[embed_off..embed_off + cfg.vocab * D * 2])?
        };
        s.synchronize()?;
        let device_bytes = l.bytes;
        let buffers = std::mem::take(&mut l.buffers);
        eprintln!(
            "hy4: loaded in {:.1}s, {:.2} GiB on the device",
            t0.elapsed().as_secs_f64(),
            device_bytes as f64 / (1u64 << 30) as f64
        );
        let engram_layers: Vec<usize> = (0..layers.len()).filter(|&i| layers[i].engram.is_some()).collect();
        let (mut n_indexed, mut n_sliding) = (0, 0);
        let cache_slot = layers
            .iter()
            .map(|l| {
                let c = if l.attn.sliding { &mut n_sliding } else { &mut n_indexed };
                *c += 1;
                *c - 1
            })
            .collect();
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
            dfly,
            cache_slot,
            n_indexed,
            n_sliding,
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

/// Where a projection's device weights come from.
#[derive(Clone, Copy)]
enum Src {
    Bf16,
    /// BF16 quantized at load to block-128 FP8.
    Fp8AtLoad,
    /// FP8 with `<name>_scale_inv` block scales.
    Fp8,
    /// NVFP4 requantized to block-128 FP8.
    Nvfp4,
}

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

    /// How projection `name` is kept on the device: (source kind, format, n, k).
    fn plan(&self, name: &str, nvfp4: bool) -> Result<(Src, WeightFormat, usize, usize)> {
        let info = self.ckpt.get(name)?;
        ensure!(info.shape.len() == 2, "{name}: not a matrix");
        let (n, k) = (info.shape[0], info.shape[1]);
        if nvfp4 {
            return Ok((Src::Nvfp4, WeightFormat::Fp8Block, n, k * 2));
        }
        Ok(match info.dtype {
            // Quantized at load to block-128 FP8 (QUENCH6K_HY4_FP8).
            DType::BF16 if k % 128 == 0 && fp8_at_load(name) => (Src::Fp8AtLoad, WeightFormat::Fp8Block, n, k),
            DType::BF16 => (Src::Bf16, WeightFormat::Bf16, n, k),
            DType::F8E4M3 => {
                ensure!(k % 128 == 0, "{name}: block-FP8 needs K a multiple of 128");
                (Src::Fp8, WeightFormat::Fp8Block, n, k)
            }
            other => bail!("{name}: unsupported projection dtype {other:?}"),
        })
    }

    /// Writes projection `name` (planned as `src`) to device weights `w` and, for FP8, block scales `scale`
    /// ([ceil(n / 128), k / 128] F32).
    fn fill(&mut self, name: &str, src: Src, n: usize, k: usize, w: u64, scale: u64) -> Result<()> {
        match src {
            Src::Bf16 => {
                self.copy_tensor(name, w)?;
            }
            Src::Fp8AtLoad => {
                let tmp = DeviceBuffer::new(n * k * 2)?;
                self.copy_tensor(name, tmp.addr())?;
                quench6k_cuda::hy4::quantize_fp8b(self.s, tmp.addr(), n, k, w, scale)?;
                self.s.synchronize()?;
            }
            Src::Fp8 => {
                let sname = format!("{name}_scale_inv");
                let info = self.ckpt.get(&sname)?;
                let want = [n.div_ceil(128), k / 128];
                ensure!(info.dtype == DType::F32 && info.shape == want, "{sname}: {:?} {:?}, expected F32 {want:?}", info.dtype, info.shape);
                self.copy_tensor(name, w)?;
                self.copy_tensor(&sname, scale)?;
            }
            Src::Nvfp4 => {
                // NVFP4 (codes [n, k / 2], E4M3 scales per 16, scalar weight_scale_2) requantized to
                // block-128 FP8 (the GEMV has no NVFP4 path).
                let sname = format!("{name}_scale");
                for (t, dtype, shape) in [(name, DType::U8, [n, k / 2]), (sname.as_str(), DType::F8E4M3, [n, k / 16])] {
                    let info = self.ckpt.get(t)?;
                    ensure!(info.dtype == dtype && info.shape == shape, "{t}: {:?} {:?}", info.dtype, info.shape);
                }
                let codes = DeviceBuffer::new(n * k / 2)?;
                let scales = DeviceBuffer::new(n * k / 16)?;
                self.copy_tensor(name, codes.addr())?;
                self.copy_tensor(&sname, scales.addr())?;
                let s2b = self.read(&format!("{name}_scale_2"))?;
                ensure!(s2b.len() == 4, "{name}_scale_2: not a scalar");
                let s2 = f32::from_le_bytes(s2b[..4].try_into().unwrap());
                let tmp = DeviceBuffer::new(n * k * 2)?;
                quench6k_cuda::hy4::nvfp4_to_bf16(self.s, codes.addr(), scales.addr(), s2, n, k, tmp.addr())?;
                quench6k_cuda::hy4::quantize_fp8b(self.s, tmp.addr(), n, k, w, scale)?;
                self.s.synchronize()?;
            }
        }
        Ok(())
    }

    /// A projection: BF16, or FP8 with `<name>_scale_inv` block scales.
    fn linear(&mut self, name: &str, shape: Option<(usize, usize)>) -> Result<Linear> {
        Ok(self.linears(&[(name, shape)], false)?.1.pop().unwrap())
    }

    /// Projections of the same input stored as one matrix, rows in the given order, when their formats
    /// and widths agree (FP8 parts but the last a multiple of 128 rows): (the stacked matrix, a view of
    /// each part in it). A GEMV gives every row of the stacked matrix the bits of its part's own launch.
    /// Otherwise (or for a single part) each part on its own and no stacked matrix. `nvfp4`: NVFP4
    /// checkpoint projections requantized to block-128 FP8.
    fn linears(&mut self, parts: &[(&str, Option<(usize, usize)>)], nvfp4: bool) -> Result<(Option<Linear>, Vec<Linear>)> {
        let mut plans = Vec::with_capacity(parts.len());
        for &(name, shape) in parts {
            let p = self.plan(name, nvfp4)?;
            if let Some((sn, sk)) = shape {
                ensure!((p.2, p.3) == (sn, sk), "{name}: shape [{}, {}], expected [{sn}, {sk}]", p.2, p.3);
            }
            plans.push(p);
        }
        let (fmt, k) = (plans[0].1, plans[0].3);
        let fp8 = fmt == WeightFormat::Fp8Block;
        let stack = parts.len() > 1
            && plans.iter().all(|p| p.1 == fmt && p.3 == k)
            && (!fp8 || plans[..plans.len() - 1].iter().all(|p| p.2 % 128 == 0));
        let elem = if fp8 { 1 } else { 2 };
        let scale_bytes = |n: usize| if fp8 { n.div_ceil(128) * (k / 128) * 4 } else { 0 };
        let mut views = Vec::with_capacity(parts.len());
        if !stack {
            for (&(name, _), &(src, fmt, n, k)) in parts.iter().zip(&plans) {
                let w = self.alloc(n * k * elem)?;
                let scale = if fp8 { self.alloc(scale_bytes(n))? } else { 0 };
                self.fill(name, src, n, k, w, scale)?;
                views.push(Linear { w, scale, fmt, n, k });
            }
            return Ok((None, views));
        }
        let total: usize = plans.iter().map(|p| p.2).sum();
        let w = self.alloc(total * k * elem)?;
        let scale = if fp8 { self.alloc(scale_bytes(total))? } else { 0 };
        let mut row = 0;
        for (&(name, _), &(src, _, n, _)) in parts.iter().zip(&plans) {
            let (pw, ps) = (w + (row * k * elem) as u64, if fp8 { scale + scale_bytes(row) as u64 } else { 0 });
            self.fill(name, src, n, k, pw, ps)?;
            views.push(Linear { w: pw, scale: ps, fmt, n, k });
            row += n;
        }
        Ok((Some(Linear { w, scale, fmt, n: total, k }), views))
    }

    /// An MLA attention sublayer under `prefix` (`...self_attn`).
    fn attn(&mut self, prefix: &str, heads: usize, sliding: bool, indexed: bool) -> Result<Attn> {
        let a = |x: &str| format!("{prefix}.{x}");
        let x = |y: &str| a(&format!("compressed_indexer.{y}"));
        let mut parts = vec![(a("q_a_proj.weight"), Some((Q_LORA, D))), (a("kv_a_proj_with_mqa.weight"), Some((KROW, D)))];
        if indexed {
            parts.push((x("token_key_projection.weight"), Some((ID, D))));
            parts.push((x("head_weight_projection.weight"), Some((IH, D))));
        }
        let parts: Vec<(&str, Option<(usize, usize)>)> = parts.iter().map(|(n, sh)| (n.as_str(), *sh)).collect();
        let (proj_in, v) = self.linears(&parts, false)?;
        let indexer = if indexed {
            Some(Indexer {
                q: self.linear(&x("query_projection.weight"), Some((IH * ID, Q_LORA)))?,
                hw: v[3],
                tk: v[2],
                tk_norm: self.raw(&x("token_key_norm.weight"), DType::BF16, Some(&[ID]))?,
                wv: self.raw(&x("key_value_projection.weight"), DType::F32, Some(&[ID, ID]))?,
                wg: self.raw(&x("key_gate_projection.weight"), DType::F32, Some(&[ID, ID]))?,
                ape: self.raw(&x("intra_group_ape"), DType::F32, Some(&[GROUP, ID]))?,
                key_norm: self.raw(&x("key_norm.weight"), DType::F32, Some(&[ID]))?,
            })
        } else {
            None
        };
        let kv_b_bytes = self.expect(&a("kv_b_proj.weight"), DType::BF16, &[heads * (NOPE + VD), KV_LORA])?;
        let kv_b = self.upload_bytes(&kv_b_bytes)?;
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
        let wuk_t = self.upload_slice(&wuk)?;
        // Sinks as F32 (the drafter stores them in BF16).
        let sink_name = a("learnable_sink_param");
        let sinks = if self.ckpt.get(&sink_name)?.dtype == DType::BF16 {
            let b = self.expect(&sink_name, DType::BF16, &[heads])?;
            let v: Vec<f32> = b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
            self.upload_slice(&v)?
        } else {
            self.raw(&sink_name, DType::F32, Some(&[heads]))?
        };
        Ok(Attn {
            heads,
            sliding,
            q_a: v[0],
            q_a_norm: self.raw(&a("q_a_layernorm.weight"), DType::BF16, Some(&[Q_LORA]))?,
            q_b: self.linear(&a("q_b_proj.weight"), Some((heads * QK, Q_LORA)))?,
            kv_a: v[1],
            kv_a_norm: self.raw(&a("kv_a_layernorm.weight"), DType::BF16, Some(&[KV_LORA]))?,
            kv_b,
            wuk_t,
            gate: self.linear(&a("linear_gate.weight"), Some((heads * VD, D)))?,
            o: self.linear(&a("o_proj.weight"), Some((D, heads * VD)))?,
            post_norm: self.raw(&a("post_proj_norm.weight"), DType::BF16, Some(&[D]))?,
            sinks,
            indexer,
            proj_in,
        })
    }

    /// The DFly drafter, if the checkpoint has one.
    fn dfly(&mut self, dir: &Path, cfg: &Config, embed: &[u8]) -> Result<Option<Dfly>> {
        let all: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let n_layers = all.get("dfly_num_hidden_layers").and_then(Value::as_u64).unwrap_or(0) as usize;
        if n_layers == 0 || !self.ckpt.tensors.contains_key("model.dfly_layers.final_norm.weight") {
            return Ok(None);
        }
        let ints = |key: &str| -> Result<Vec<usize>> {
            let v: Vec<usize> = all
                .get(key)
                .and_then(Value::as_array)
                .with_context(|| format!("config: {key}"))?
                .iter()
                .filter_map(Value::as_u64)
                .map(|x| x as usize)
                .collect();
            ensure!(v.len() == n_layers, "config: {key} needs one target layer per drafter layer");
            for &i in &v {
                ensure!(i < cfg.layers && !cfg.sliding[i], "config: {key}: layer {i} is not an indexed layer");
            }
            Ok(v)
        };
        let index_layers = ints("dfly_target_index_layer")?;
        let kv_layers = ints("dfly_target_kv_layers")?;
        ensure!(
            all.get("dfly_context_local_window").is_none_or(Value::is_null),
            "config: dfly_context_local_window is not supported"
        );
        ensure!(
            all.get("dfly_mtp_fusion").and_then(Value::as_str).unwrap_or("slot0") == "slot0",
            "config: dfly_mtp_fusion"
        );
        let mask = num(&all, "dfly_mask_token_id")? as i32;
        ensure!((mask as usize) < cfg.vocab, "config: dfly_mask_token_id");
        let drafts = all
            .get("speculative_config")
            .and_then(|v| v.get("num_speculative_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(5) as usize;
        let fp4 = |l: &mut Self, names: &[&str]| -> Result<(Option<Linear>, Vec<Linear>)> {
            let parts: Vec<(&str, Option<(usize, usize)>)> = names.iter().map(|&n| (n, None)).collect();
            l.linears(&parts, true)
        };
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let n = |x: &str| format!("model.dfly_layers.{i}.{x}");
            let attn = self.attn(&n("self_attn"), cfg.heads_indexed, false, false)?;
            let (gate_up, gu) = fp4(self, &[&n("mlp.gate_proj.weight"), &n("mlp.up_proj.weight")])?;
            ensure!(gu[0].n == gu[1].n, "drafter layer {i}: gate and up widths differ");
            layers.push(DflyLayer {
                in_norm: self.raw(&n("input_layernorm.weight"), DType::BF16, Some(&[D]))?,
                post_attn_norm: self.raw(&n("post_attention_layernorm.weight"), DType::BF16, Some(&[D]))?,
                attn,
                gate: gu[0],
                up: gu[1],
                down: fp4(self, &[&n("mlp.down_proj.weight")])?.1[0],
                gate_up,
            });
        }
        let x = |y: &str| format!("model.dfly_layers.{y}");
        let (c_gate_up, cgu) = fp4(self, &[&x("hidden_correction.gate_proj.weight"), &x("hidden_correction.up_proj.weight")])?;
        ensure!(cgu[0].n == cgu[1].n, "hidden_correction: gate and up widths differ");
        let dfly = Dfly {
            layers,
            norm: self.raw(&x("final_norm.weight"), DType::BF16, Some(&[D]))?,
            enorm: self.raw(&x("mtp_fusion.enorm.weight"), DType::BF16, Some(&[D]))?,
            hnorm: self.raw(&x("mtp_fusion.hnorm.weight"), DType::BF16, Some(&[D]))?,
            eh: self.linear(&x("mtp_fusion.eh_proj.weight"), Some((D, 2 * D)))?,
            fuse_gate: self.raw(&x("mtp_fusion.gate"), DType::BF16, Some(&[D]))?,
            c_hidden_norm: self.raw(&x("hidden_correction.hidden_norm.weight"), DType::BF16, Some(&[D]))?,
            c_embed_norm: self.raw(&x("hidden_correction.embed_norm.weight"), DType::BF16, Some(&[D]))?,
            c_gate: cgu[0],
            c_up: cgu[1],
            c_down: fp4(self, &[&x("hidden_correction.down_proj.weight")])?.1[0],
            c_gate_up,
            embed: self.upload_bytes(embed)?,
            mask,
            index_layers,
            kv_layers,
            width: drafts + 1,
        };
        ensure!(dfly.c_gate.k == 2 * D && dfly.c_down.n == D, "hidden_correction shapes");
        Ok(Some(dfly))
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
        let bits = |m: &Mmap| (0..(m.len() + 4096).div_ceil(4096 * 64)).map(|_| std::sync::atomic::AtomicU64::new(0)).collect::<Vec<_>>();
        let hinted = [bits(&table), bits(&scales)];
        Ok(Engram {
            hash,
            table,
            table_off,
            scales,
            scales_off,
            hinted,
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
