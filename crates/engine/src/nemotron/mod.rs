//! Nemotron-H (`nemotron_h`: NVIDIA Nemotron 3 Super 120B-A12B and finetunes such as Munin 2.0) text
//! model: configuration, weights on the device and the host-resident token embedding. The forward pass is
//! in [`forward`].
//!
//! The layers follow `hybrid_override_pattern`: Mamba-2 mixers (M), MoE blocks (E) and attention (*), each
//! a pre-norm residual block. Supported checkpoint layout: the ModelOpt MIXED_PRECISION release, whose
//! per-tensor formats the loader reads from the safetensors headers:
//! - routed experts NVFP4 per expert (`experts.{e}.{up,down}_proj`: codes, E4M3 scales per 16, an F32
//!   `weight_scale_2`), stacked per layer on the device as stored;
//! - FP8 E4M3 projections with one F32 `weight_scale` per tensor, kept as FP8 (every 128 x 128 block
//!   gets the tensor's scale, so the values are the checkpoint's);
//! - NVFP4 dense projections (one shared expert), expanded to BF16 code x block scale (exact) with the
//!   global scale applied to the output;
//! - BF16 projections, quantized at load to INT8 with an F32 scale per 64 weights (QUENCH6K_NEM_INT8);
//! - the MTP layer's BF16 experts NVFP4-quantized at load (the drafter only proposes tokens).
//!
//! The `input_scale` tensors are not used: activations stay BF16 / F32.

pub mod forward;

use quench6k_cuda::nemotron as nk;

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use quench6k_cuda::kernels::WeightFormat;
use quench6k_cuda::{DeviceBuffer, Stream};
use serde_json::Value;

use crate::checkpoint::{Checkpoint, DType};

pub const D: usize = 4096;
/// Attention: query heads, key/value heads, head width.
pub const H: usize = 32;
pub const KVH: usize = 2;
pub const HD: usize = 128;
pub const QKV: usize = (H + 2 * KVH) * HD;
/// Mamba-2: heads, head width, inner width, groups, state size, convolution taps.
pub const MH: usize = 128;
pub const MHD: usize = 64;
pub const DI: usize = MH * MHD;
pub const NG: usize = 8;
pub const NS: usize = 128;
pub const CONVK: usize = 4;
/// Convolution channels (x, B, C) and in_proj rows (z, xBC, dt).
pub const CONV: usize = DI + 2 * NG * NS;
pub const INP: usize = DI + CONV + MH;
/// MoE: routed experts, experts per token, expert width, latent width, shared expert width.
pub const EXPERTS: usize = 512;
pub const TOPK: usize = 22;
pub const FF: usize = 2688;
pub const LAT: usize = 1024;
pub const SH: usize = 5376;

/// Mixer of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Mamba,
    Moe,
    Attention,
}

pub struct Config {
    pub kinds: Vec<Kind>,
    pub mtp_kinds: Vec<Kind>,
    pub vocab: usize,
    pub eps: f32,
    pub routed_scaling: f32,
    /// dt clamp of the scan (time_step_limit; (0, inf) in the released configs).
    pub dt_limit: (f32, f32),
    pub eos: Vec<i32>,
    /// Longest context of the model card (max_position_embeddings).
    pub max_positions: usize,
}

fn num(v: &Value, key: &str) -> Result<f64> {
    v.get(key).and_then(Value::as_f64).with_context(|| format!("config: {key} missing"))
}

fn parse_kinds(pattern: &str) -> Result<Vec<Kind>> {
    pattern
        .chars()
        .map(|c| match c {
            'M' => Ok(Kind::Mamba),
            'E' => Ok(Kind::Moe),
            '*' => Ok(Kind::Attention),
            other => bail!("config: layer type {other:?} in the hybrid pattern is not supported"),
        })
        .collect()
}

impl Config {
    pub fn read(dir: &Path) -> Result<Self> {
        let c: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).context("config.json")?)?;
        Self::parse(&c, std::fs::read(dir.join("generation_config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()))
    }

    pub fn parse(c: &Value, generation: Option<Value>) -> Result<Self> {
        let arch = c.get("model_type").and_then(Value::as_str).unwrap_or("");
        ensure!(arch == "nemotron_h", "config: model_type {arch:?} is not nemotron_h");
        for (key, want) in [
            ("hidden_size", D),
            ("num_attention_heads", H),
            ("num_key_value_heads", KVH),
            ("head_dim", HD),
            ("mamba_num_heads", MH),
            ("mamba_head_dim", MHD),
            ("n_groups", NG),
            ("ssm_state_size", NS),
            ("conv_kernel", CONVK),
            ("n_routed_experts", EXPERTS),
            ("num_experts_per_tok", TOPK),
            ("moe_intermediate_size", FF),
            ("moe_latent_size", LAT),
            ("moe_shared_expert_intermediate_size", SH),
            ("n_shared_experts", 1),
            ("n_group", 1),
            ("topk_group", 1),
        ] {
            let v = num(c, key)? as usize;
            ensure!(v == want, "config: {key} = {v}, this engine supports {want}");
        }
        for (key, want) in [
            ("use_conv_bias", true),
            ("use_bias", false),
            ("mamba_proj_bias", false),
            ("attention_bias", false),
            ("mlp_bias", false),
            ("norm_topk_prob", true),
            ("tie_word_embeddings", false),
        ] {
            let v = c.get(key).and_then(Value::as_bool).unwrap_or(want);
            ensure!(v == want, "config: {key} = {v} is not supported");
        }
        for (key, want) in [("mamba_hidden_act", "silu"), ("mlp_hidden_act", "relu2")] {
            let v = c.get(key).and_then(Value::as_str).unwrap_or(want);
            ensure!(v == want, "config: {key} = {v:?} is not supported");
        }
        ensure!(c.get("sliding_window").is_none_or(Value::is_null), "config: sliding_window is not supported");
        let pattern = c.get("hybrid_override_pattern").and_then(Value::as_str).context("config: hybrid_override_pattern")?;
        let kinds = parse_kinds(pattern)?;
        if let Some(n) = c.get("num_hidden_layers").and_then(Value::as_u64) {
            ensure!(n as usize == kinds.len(), "config: num_hidden_layers {n} but {} layers in the pattern", kinds.len());
        }
        let mtp_kinds = match c.get("num_nextn_predict_layers").and_then(Value::as_u64).unwrap_or(0) {
            0 => Vec::new(),
            1 => parse_kinds(c.get("mtp_hybrid_override_pattern").and_then(Value::as_str).unwrap_or("*E"))?,
            n => bail!("config: {n} MTP layers (one is supported)"),
        };
        ensure!(
            mtp_kinds.is_empty() || mtp_kinds == [Kind::Attention, Kind::Moe],
            "config: MTP pattern {mtp_kinds:?} is not supported"
        );
        let dt_limit = match c.get("time_step_limit").and_then(Value::as_array) {
            Some(a) if a.len() == 2 => (
                a[0].as_f64().unwrap_or(0.0) as f32,
                a[1].as_f64().map_or(f32::INFINITY, |v| v as f32),
            ),
            _ => (0.0, f32::INFINITY),
        };
        let ids = |v: Option<&Value>| -> Vec<i32> {
            match v {
                Some(Value::Array(a)) => a.iter().filter_map(Value::as_i64).map(|v| v as i32).collect(),
                Some(v) => v.as_i64().map(|v| v as i32).into_iter().collect(),
                None => Vec::new(),
            }
        };
        let mut eos = ids(c.get("eos_token_id"));
        if let Some(g) = &generation {
            for e in ids(g.get("eos_token_id")) {
                if !eos.contains(&e) {
                    eos.push(e);
                }
            }
        }
        Ok(Self {
            kinds,
            mtp_kinds,
            vocab: num(c, "vocab_size")? as usize,
            eps: c.get("layer_norm_epsilon").or_else(|| c.get("norm_eps")).and_then(Value::as_f64).unwrap_or(1e-5) as f32,
            routed_scaling: num(c, "routed_scaling_factor")? as f32,
            dt_limit,
            eos,
            max_positions: num(c, "max_position_embeddings")? as usize,
        })
    }

    pub fn count(&self, k: Kind) -> usize {
        self.kinds.iter().filter(|&&x| x == k).count()
    }

    /// Per layer: its index among the layers of its kind.
    pub fn slots(&self) -> Vec<usize> {
        let mut n = [0usize; 3];
        self.kinds
            .iter()
            .map(|&k| {
                let i = k as usize;
                n[i] += 1;
                n[i] - 1
            })
            .collect()
    }
}

/// A projection `[n, k]` on the device: y = alpha * x . W^T.
#[derive(Clone, Copy, Debug)]
pub struct Linear {
    pub w: u64,
    pub scale: u64,
    pub fmt: WeightFormat,
    pub n: usize,
    pub k: usize,
    /// Output factor (the global scale of NVFP4 weights expanded to BF16; 1 otherwise).
    pub alpha: f32,
}

pub struct Mamba {
    pub in_proj: Linear,
    pub out_proj: Linear,
    /// The mixer's small parameters (F32 on the device: conv weight [CONV, 4] and bias, dt_bias, A_log,
    /// D, the gated norm's weight) and the dt limits.
    pub w: nk::MambaW,
}

pub struct Attn {
    /// [q | k | v] x D.
    pub qkv: Linear,
    pub o: Linear,
}

/// A layer's routed experts: NVFP4 stacks per projection and, for the grouped GEMM of large steps,
/// per-expert (codes, scales) pointer tables (u64 [EXPERTS] each) of up and down.
pub struct RoutedExperts {
    pub up: nk::NvExperts,
    pub down: nk::NvExperts,
    pub tables: [(u64, u64); 2],
}

pub struct Moe {
    /// Router BF16 [EXPERTS, D] (logits in F32) and its F32 correction bias [EXPERTS].
    pub gate: Linear,
    pub bias: u64,
    pub fc1: Linear,
    pub fc2: Linear,
    pub experts: RoutedExperts,
    pub shared_up: Linear,
    pub shared_down: Linear,
}

pub enum Mixer {
    Mamba(Mamba),
    Attn(Attn),
    Moe(Moe),
}

pub struct Layer {
    pub norm: u64,
    pub mixer: Mixer,
}

/// The MTP layer (`mtp.layers`): x0 = eh_proj([enorm(embed(next token)), hnorm(hidden)]), then an
/// attention block and an MoE block, final_layernorm, and the model's LM head.
pub struct Mtp {
    pub eh: Linear,
    pub enorm: u64,
    pub hnorm: u64,
    pub layers: Vec<Layer>,
    pub final_norm: u64,
}

pub struct NemotronModel {
    pub cfg: Config,
    pub layers: Vec<Layer>,
    /// Per layer: its index among the layers of its kind (Mamba state and attention cache slots).
    pub slot: Vec<usize>,
    pub n_mamba: usize,
    pub n_attn: usize,
    pub norm_f: u64,
    pub lm_head: Linear,
    pub mtp: Option<Mtp>,
    embed: Mmap,
    embed_off: usize,
    pub device_bytes: usize,
    _buffers: Vec<DeviceBuffer>,
}

impl NemotronModel {
    pub fn is_nemotron(dir: &Path) -> bool {
        std::fs::read(dir.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .is_some_and(|v| v.get("model_type").and_then(Value::as_str) == Some("nemotron_h"))
    }

    /// BF16 embedding row of `token`.
    pub fn embed_row(&self, token: usize) -> &[u8] {
        let o = self.embed_off + token * D * 2;
        &self.embed[o..o + D * 2]
    }

    pub fn load(dir: &Path, s: &Stream) -> Result<Self> {
        let cfg = Config::read(dir)?;
        check_shards(dir)?;
        let ckpt = Checkpoint::open(dir)?;
        let t0 = std::time::Instant::now();
        let largest = ckpt
            .tensors
            .iter()
            .filter(|(n, _)| *n != "backbone.embeddings.weight")
            .map(|(_, t)| t.bytes as usize)
            .max()
            .unwrap_or(0);
        let direct: Option<Vec<std::fs::File>> = ckpt.files.iter().map(|f| crate::fileio::open_direct(&f.path)).collect();
        let mut l = Loader::new(&ckpt, Some(s), Some(quench6k_cuda::PinnedBuffer::new(largest + 2 * DIRECT_ALIGN)?), direct);
        let (embed, embed_off) = l.map("backbone.embeddings.weight", DType::BF16, &[cfg.vocab, D])?;
        let parts = l.parts(&cfg, t0)?;
        s.synchronize()?;
        let device_bytes = l.bytes;
        let buffers = std::mem::take(&mut l.buffers);
        eprintln!(
            "nemotron: loaded in {:.1}s, {:.2} GiB on the device",
            t0.elapsed().as_secs_f64(),
            device_bytes as f64 / (1u64 << 30) as f64
        );
        let slot = cfg.slots();
        Ok(Self {
            n_mamba: cfg.count(Kind::Mamba),
            n_attn: cfg.count(Kind::Attention),
            cfg,
            layers: parts.layers,
            slot,
            norm_f: parts.norm_f,
            lm_head: parts.lm_head,
            mtp: parts.mtp,
            embed,
            embed_off,
            device_bytes,
            _buffers: buffers,
        })
    }

    /// What [`NemotronModel::load`] would do, from the safetensors headers and small tensors alone (no
    /// GPU): every tensor's presence, dtype and shape checked, the device bytes it would allocate and how
    /// each projection is kept.
    pub fn dry_run(dir: &Path) -> Result<DryRun> {
        let cfg = Config::read(dir)?;
        check_shards(dir)?;
        let ckpt = Checkpoint::open(dir)?;
        let mut l = Loader::new(&ckpt, None, None, None);
        let e = ckpt.get("backbone.embeddings.weight")?;
        ensure!(e.dtype == DType::BF16 && e.shape == [cfg.vocab, D], "embedding: {:?} {:?}", e.dtype, e.shape);
        let parts = l.parts(&cfg, std::time::Instant::now())?;
        let mut scales = Vec::new();
        for (i, layer) in parts.layers.iter().enumerate() {
            if let Mixer::Attn(_) = &layer.mixer {
                let m = |x: &str| format!("backbone.layers.{i}.mixer.{x}");
                let (ks, vs) = (m("k_proj.k_scale"), m("v_proj.v_scale"));
                if ckpt.tensors.contains_key(&ks) && ckpt.tensors.contains_key(&vs) {
                    scales.push((i, l.scalar(&ks)?, l.scalar(&vs)?));
                }
            }
        }
        Ok(DryRun {
            device_bytes: l.bytes,
            kept: l.kept,
            cache_scales: scales,
            mtp: parts.mtp.is_some(),
        })
    }
}

/// What a load would allocate and how it keeps the projections ([`NemotronModel::dry_run`]).
pub struct DryRun {
    pub device_bytes: usize,
    /// (stored, kept) -> (projections, device bytes).
    pub kept: std::collections::BTreeMap<(String, String), (usize, usize)>,
    /// The checkpoint's static cache scales (layer, k_scale, v_scale); not used (the cache is BF16, or FP8
    /// with a scale per position and head).
    pub cache_scales: Vec<(usize, f32, f32)>,
    pub mtp: bool,
}

struct Parts {
    layers: Vec<Layer>,
    norm_f: u64,
    lm_head: Linear,
    mtp: Option<Mtp>,
}

/// Every shard named by model.safetensors.index.json exists and is complete (its header's tensors fit in
/// the file), so a checkpoint still downloading is refused before anything is read.
pub fn check_shards(dir: &Path) -> Result<()> {
    let index: Value = serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).context("model.safetensors.index.json")?)?;
    let map = index.get("weight_map").and_then(Value::as_object).context("index: weight_map")?;
    let mut shards: Vec<&str> = map.values().filter_map(Value::as_str).collect();
    shards.sort_unstable();
    shards.dedup();
    for shard in shards {
        let path = dir.join(shard);
        let len = std::fs::metadata(&path).with_context(|| format!("shard {shard} is missing (still downloading?)"))?.len();
        let mut f = std::fs::File::open(&path)?;
        let mut hl = [0u8; 8];
        std::io::Read::read_exact(&mut f, &mut hl).with_context(|| format!("shard {shard}: no header"))?;
        let hl = u64::from_le_bytes(hl);
        ensure!(hl + 8 <= len, "shard {shard} is incomplete ({len} bytes)");
        let mut header = vec![0u8; hl as usize];
        std::io::Read::read_exact(&mut f, &mut header)?;
        let h: serde_json::Map<String, Value> = serde_json::from_slice(&header).with_context(|| format!("shard {shard}: header"))?;
        let end = h
            .values()
            .filter_map(|t| t.get("data_offsets").and_then(Value::as_array))
            .filter_map(|o| o.get(1).and_then(Value::as_u64))
            .max()
            .unwrap_or(0);
        ensure!(8 + hl + end == len, "shard {shard} is incomplete: {len} bytes, header says {}", 8 + hl + end);
    }
    for name in map.keys() {
        ensure!(!name.is_empty(), "index: empty tensor name");
    }
    Ok(())
}

/// How a checkpoint projection is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stored {
    Bf16,
    /// FP8 E4M3 with one F32 weight_scale.
    Fp8,
    /// NVFP4 codes, E4M3 scales per 16, F32 weight_scale_2.
    Nvfp4,
}

/// How a projection is kept on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kept {
    Bf16,
    /// INT8 with an F32 scale per 64 weights of a row (quantized at load from BF16).
    Int8,
    /// FP8 E4M3 as stored, the tensor's scale in every 128 x 128 block.
    Fp8,
    /// NVFP4 code x block scale as BF16 (exact), the global scale as the output factor.
    Nvfp4AsBf16,
}

/// BF16 projections quantized to INT8 groups at load: QUENCH6K_NEM_INT8 = comma-separated name fragments,
/// "all" (default) or "none". The router stays BF16 (its kernel reads BF16; routing decisions).
pub fn int8_at_load(name: &str) -> bool {
    if name.ends_with("mixer.gate.weight") {
        return false;
    }
    let spec = std::env::var("QUENCH6K_NEM_INT8").unwrap_or_else(|_| "all".into());
    spec.split(',').filter(|f| !f.is_empty() && *f != "none").any(|f| f == "all" || name.contains(f))
}

pub fn kept(stored: Stored, name: &str, k: usize) -> Kept {
    match stored {
        Stored::Bf16 if k.is_multiple_of(quench6k_cuda::kernels::int8_group()) && int8_at_load(name) => Kept::Int8,
        Stored::Bf16 => Kept::Bf16,
        Stored::Fp8 => Kept::Fp8,
        Stored::Nvfp4 => Kept::Nvfp4AsBf16,
    }
}

/// Device bytes of an `[n, k]` projection kept as `kept`.
pub fn linear_bytes(kept: Kept, n: usize, k: usize) -> usize {
    match kept {
        Kept::Bf16 | Kept::Nvfp4AsBf16 => n * k * 2,
        Kept::Int8 => n * k + n * (k / quench6k_cuda::kernels::int8_group()) * 4,
        Kept::Fp8 => n * k + n.div_ceil(128) * (k / 128) * 4,
    }
}

const DIRECT_ALIGN: usize = 4096;

/// Reads the checkpoint onto the device; without a stream (dry run) it only checks the tensors and counts
/// the bytes it would allocate.
struct Loader<'a> {
    ckpt: &'a Checkpoint,
    s: Option<&'a Stream>,
    staging: Option<quench6k_cuda::PinnedBuffer>,
    /// Pinned buffer where per-expert tensors are stacked before one upload.
    stack: Option<quench6k_cuda::PinnedBuffer>,
    direct: Option<Vec<std::fs::File>>,
    buffers: Vec<DeviceBuffer>,
    bytes: usize,
    /// (stored, kept) -> (projections, device bytes).
    kept: std::collections::BTreeMap<(String, String), (usize, usize)>,
}

impl<'a> Loader<'a> {
    fn new(ckpt: &'a Checkpoint, s: Option<&'a Stream>, staging: Option<quench6k_cuda::PinnedBuffer>, direct: Option<Vec<std::fs::File>>) -> Self {
        Self {
            ckpt,
            s,
            staging,
            stack: None,
            direct,
            buffers: Vec::new(),
            bytes: 0,
            kept: Default::default(),
        }
    }

    /// The model's layers, final norm, LM head and MTP layer.
    fn parts(&mut self, cfg: &Config, t0: std::time::Instant) -> Result<Parts> {
        let mtp_on = !cfg.mtp_kinds.is_empty()
            && self.ckpt.tensors.contains_key("mtp.layers.0.eh_proj.weight")
            && std::env::var("QUENCH6K_NEM_MTP").map_or(true, |v| v != "0");
        let mut layers = Vec::with_capacity(cfg.kinds.len());
        for (i, &kind) in cfg.kinds.iter().enumerate() {
            layers.push(self.layer(&format!("backbone.layers.{i}"), kind, cfg.dt_limit)?);
            self.sync()?;
            if self.s.is_some() && (i + 1) % 8 == 0 {
                eprintln!(
                    "nemotron: loaded {} layers, {:.1} GiB, {:.1}s",
                    i + 1,
                    self.bytes as f64 / (1u64 << 30) as f64,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        let norm_f = self.raw("backbone.norm_f.weight", DType::BF16, Some(&[D]))?;
        let lm_head = self.linear("lm_head.weight", (cfg.vocab, D))?;
        let mtp = if mtp_on {
            let p = "mtp.layers";
            let eh = self.linear(&format!("{p}.0.eh_proj.weight"), (D, 2 * D))?;
            let enorm = self.raw(&format!("{p}.0.enorm.weight"), DType::BF16, Some(&[D]))?;
            let hnorm = self.raw(&format!("{p}.0.hnorm.weight"), DType::BF16, Some(&[D]))?;
            let mut ml = Vec::new();
            for (j, &kind) in cfg.mtp_kinds.iter().enumerate() {
                ml.push(self.layer(&format!("{p}.{j}"), kind, cfg.dt_limit)?);
            }
            let final_norm = self.raw(&format!("{p}.{}.final_layernorm.weight", cfg.mtp_kinds.len() - 1), DType::BF16, Some(&[D]))?;
            Some(Mtp {
                eh,
                enorm,
                hnorm,
                layers: ml,
                final_norm,
            })
        } else {
            None
        };
        Ok(Parts { layers, norm_f, lm_head, mtp })
    }

    fn sync(&self) -> Result<()> {
        match self.s {
            Some(s) => s.synchronize(),
            None => Ok(()),
        }
    }

    fn alloc(&mut self, bytes: usize) -> Result<u64> {
        if self.s.is_none() {
            // Dry run: distinct, aligned, never dereferenced addresses.
            let a = (1u64 << 44) + self.bytes as u64;
            self.bytes += bytes.next_multiple_of(256);
            return Ok(a);
        }
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
        let buf = self.staging.as_mut().context("loader without staging")?.as_mut_slice();
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
        if self.staging.is_none() {
            return self.ckpt.read(name);
        }
        let (o, n) = self.fetch(name)?;
        Ok(self.staging.as_ref().unwrap().as_slice()[o..o + n].to_vec())
    }

    /// An F32 scalar tensor (shape [] or [1]).
    fn scalar(&mut self, name: &str) -> Result<f32> {
        let info = self.ckpt.get(name)?;
        ensure!(info.dtype == DType::F32 && info.shape.iter().product::<usize>() == 1, "{name}: {:?} {:?}, expected an F32 scalar", info.dtype, info.shape);
        let b = self.read(name)?;
        Ok(f32::from_le_bytes(b[..4].try_into().unwrap()))
    }

    /// A small tensor (BF16 or F32) as F32 values.
    fn f32s(&mut self, name: &str, count: usize) -> Result<Vec<f32>> {
        let info = self.ckpt.get(name)?.clone();
        ensure!(info.shape.iter().product::<usize>() == count, "{name}: shape {:?}, expected {count} values", info.shape);
        let b = self.read(name)?;
        Ok(match info.dtype {
            DType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            DType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            other => bail!("{name}: {other:?}, expected F32 or BF16"),
        })
    }

    /// Copies a tensor from the checkpoint to device address `dst`.
    fn copy_tensor(&mut self, name: &str, dst: u64) -> Result<usize> {
        let Some(s) = self.s else {
            return Ok(self.ckpt.get(name)?.bytes as usize);
        };
        let (o, n) = self.fetch(name)?;
        unsafe { s.copy_h2d(dst, self.staging.as_ref().unwrap().as_ptr().add(o), n)? };
        s.synchronize()?;
        Ok(n)
    }

    fn upload_slice<T: Copy>(&mut self, v: &[T]) -> Result<u64> {
        let a = self.alloc(std::mem::size_of_val(v))?;
        if let Some(s) = self.s {
            s.upload_to(a, v)?;
            s.synchronize()?;
        }
        Ok(a)
    }

    fn note(&mut self, stored: Stored, kept: Kept, bytes: usize) {
        let e = self.kept.entry((format!("{stored:?}"), format!("{kept:?}"))).or_default();
        e.0 += 1;
        e.1 += bytes;
    }

    /// A tensor uploaded as stored (dtype and, when given, shape checked).
    fn raw(&mut self, name: &str, dtype: DType, shape: Option<&[usize]>) -> Result<u64> {
        let info = self.ckpt.get(name)?;
        ensure!(info.dtype == dtype, "{name}: {:?}, expected {dtype:?}", info.dtype);
        if let Some(sh) = shape {
            ensure!(info.shape == sh, "{name}: shape {:?}, expected {sh:?}", info.shape);
        }
        let n = info.bytes as usize;
        let a = self.alloc(n)?;
        self.copy_tensor(name, a)?;
        Ok(a)
    }

    /// Small parameters as F32 on the device (exact for BF16 sources).
    fn raw_f32(&mut self, name: &str, count: usize) -> Result<u64> {
        let v = self.f32s(name, count)?;
        self.upload_slice(&v)
    }

    /// How projection `name` (`base.weight`) is stored: by its dtype and companion tensors.
    fn stored(&self, name: &str, shape: (usize, usize)) -> Result<Stored> {
        let info = self.ckpt.get(name)?;
        let (n, k) = shape;
        let base = name.strip_suffix(".weight").context("projection name")?;
        let st = match info.dtype {
            DType::BF16 => Stored::Bf16,
            DType::F8E4M3 => Stored::Fp8,
            DType::U8 if self.ckpt.tensors.contains_key(&format!("{base}.weight_scale_2")) => Stored::Nvfp4,
            other => bail!("{name}: unsupported projection dtype {other:?}"),
        };
        let want: [usize; 2] = if st == Stored::Nvfp4 { [n, k / 2] } else { [n, k] };
        ensure!(info.shape == want, "{name}: shape {:?}, expected {want:?}", info.shape);
        Ok(st)
    }

    /// A projection of shape [n, k] (input width k), kept as [`kept`] decides.
    fn linear(&mut self, name: &str, shape: (usize, usize)) -> Result<Linear> {
        self.linears(&[name], shape.1, Some(shape.0))
    }

    /// Projections of the same input stacked row-wise into one matrix (all stored alike); `rows`: the
    /// expected row count of a single part.
    fn linears(&mut self, names: &[&str], k: usize, rows: Option<usize>) -> Result<Linear> {
        let mut parts = Vec::with_capacity(names.len());
        for &name in names {
            let info = self.ckpt.get(name)?;
            ensure!(info.shape.len() == 2, "{name}: not a matrix");
            let n = info.shape[0];
            if let Some(r) = rows {
                ensure!(n == r, "{name}: {n} rows, expected {r}");
            }
            let st = self.stored(name, (n, k))?;
            parts.push((name, n, st));
        }
        let st = parts[0].2;
        ensure!(parts.iter().all(|p| p.2 == st), "{names:?}: stacked projections stored differently");
        let total: usize = parts.iter().map(|p| p.1).sum();
        let kept = kept(st, names[0], k);
        self.note(st, kept, linear_bytes(kept, total, k));
        let base = |name: &str| name.strip_suffix(".weight").unwrap().to_string();
        let Some(s) = self.s else {
            // Dry run: the allocations and the scales' checks of a load.
            if kept == Kept::Fp8 {
                for &(name, _, _) in &parts {
                    let sc = self.scalar(&format!("{}.weight_scale", base(name)))?;
                    ensure!(sc.is_finite() && sc > 0.0, "{name}: weight_scale {sc}");
                }
            }
            if kept == Kept::Nvfp4AsBf16 {
                let b = base(parts[0].0);
                let info = self.ckpt.get(&format!("{b}.weight_scale"))?;
                ensure!(info.dtype == DType::F8E4M3 && info.shape == [total, k / 16], "{b}.weight_scale: {:?} {:?}", info.dtype, info.shape);
                self.scalar(&format!("{b}.weight_scale_2"))?;
            }
            let fmt = match kept {
                Kept::Bf16 | Kept::Nvfp4AsBf16 => WeightFormat::Bf16,
                Kept::Int8 => WeightFormat::Int8Group,
                Kept::Fp8 => WeightFormat::Fp8Block,
            };
            let w = self.alloc(linear_bytes(kept, total, k))?;
            return Ok(Linear { w, scale: 0, fmt, n: total, k, alpha: 1.0 });
        };
        match kept {
            Kept::Bf16 | Kept::Int8 => {
                let tmp = DeviceBuffer::new(total * k * 2)?;
                let mut at = 0;
                for &(name, _, _) in &parts {
                    at += self.copy_tensor(name, tmp.addr() + at as u64)?;
                }
                if kept == Kept::Bf16 {
                    let w = self.alloc(total * k * 2)?;
                    s.copy_d2d(w, tmp.addr(), total * k * 2)?;
                    s.synchronize()?;
                    return Ok(Linear { w, scale: 0, fmt: WeightFormat::Bf16, n: total, k, alpha: 1.0 });
                }
                let g = quench6k_cuda::kernels::int8_group();
                let q = self.alloc(total * k)?;
                let scale = self.alloc(total * (k / g) * 4)?;
                quench6k_cuda::kernels::quantize_int8_groups(s, tmp.addr(), total, k, q, scale)?;
                s.synchronize()?;
                Ok(Linear { w: q, scale, fmt: WeightFormat::Int8Group, n: total, k, alpha: 1.0 })
            }
            Kept::Fp8 => {
                ensure!(k.is_multiple_of(128), "{names:?}: FP8 needs K a multiple of 128");
                // Block scales [ceil(total / 128), k / 128]: a block row may span two parts only when a part
                // ends inside it, which would mix their scales.
                ensure!(
                    parts.len() == 1 || parts[..parts.len() - 1].iter().all(|p| p.1.is_multiple_of(128)),
                    "{names:?}: stacked FP8 parts must have multiples of 128 rows"
                );
                let w = self.alloc(total * k)?;
                let kb = k / 128;
                let mut scales = vec![0f32; total.div_ceil(128) * kb];
                let mut row = 0;
                for &(name, n, _) in &parts {
                    self.copy_tensor(name, w + (row * k) as u64)?;
                    let sc = self.scalar(&format!("{}.weight_scale", base(name)))?;
                    ensure!(sc.is_finite() && sc > 0.0, "{name}: weight_scale {sc}");
                    for br in row / 128..(row + n).div_ceil(128) {
                        scales[br * kb..(br + 1) * kb].fill(sc);
                    }
                    row += n;
                }
                let scale = self.upload_slice(&scales)?;
                Ok(Linear { w, scale, fmt: WeightFormat::Fp8Block, n: total, k, alpha: 1.0 })
            }
            Kept::Nvfp4AsBf16 => {
                ensure!(parts.len() == 1, "{names:?}: stacked NVFP4 projections");
                let (name, n, _) = parts[0];
                let b = base(name);
                let sname = format!("{b}.weight_scale");
                let info = self.ckpt.get(&sname)?;
                ensure!(info.dtype == DType::F8E4M3 && info.shape == [n, k / 16], "{sname}: {:?} {:?}", info.dtype, info.shape);
                let s2 = self.scalar(&format!("{b}.weight_scale_2"))?;
                let codes = DeviceBuffer::new(n * k / 2)?;
                let sc = DeviceBuffer::new(n * k / 16)?;
                self.copy_tensor(name, codes.addr())?;
                self.copy_tensor(&sname, sc.addr())?;
                let w = self.alloc(n * k * 2)?;
                // Code x block scale is exact in BF16 (at most 6 significant bits); s2 scales the output.
                quench6k_cuda::hy4::nvfp4_to_bf16(s, codes.addr(), sc.addr(), 1.0, n, k, w)?;
                s.synchronize()?;
                Ok(Linear { w, scale: 0, fmt: WeightFormat::Bf16, n, k, alpha: s2 })
            }
        }
    }

    fn layer(&mut self, p: &str, kind: Kind, dt_limit: (f32, f32)) -> Result<Layer> {
        let m = |x: &str| format!("{p}.mixer.{x}");
        let norm = self.raw(&format!("{p}.norm.weight"), DType::BF16, Some(&[D]))?;
        let mixer = match kind {
            Kind::Mamba => Mixer::Mamba(Mamba {
                in_proj: self.linear(&m("in_proj.weight"), (INP, D))?,
                out_proj: self.linear(&m("out_proj.weight"), (D, DI))?,
                w: nk::MambaW {
                    conv_w: self.raw_f32(&m("conv1d.weight"), CONV * CONVK)?,
                    conv_b: self.raw_f32(&m("conv1d.bias"), CONV)?,
                    dt_bias: self.raw_f32(&m("dt_bias"), MH)?,
                    a_log: self.raw_f32(&m("A_log"), MH)?,
                    d: self.raw_f32(&m("D"), MH)?,
                    norm_w: self.raw_f32(&m("norm.weight"), DI)?,
                    dt_min: dt_limit.0,
                    dt_max: dt_limit.1,
                },
            }),
            Kind::Attention => {
                let (q, k, v) = (m("q_proj.weight"), m("k_proj.weight"), m("v_proj.weight"));
                let qkv = self.linears(&[&q, &k, &v], D, None)?;
                ensure!(qkv.n == QKV, "{p}: q/k/v have {} rows, expected {QKV}", qkv.n);
                Mixer::Attn(Attn {
                    qkv,
                    o: self.linear(&m("o_proj.weight"), (D, H * HD))?,
                })
            }
            Kind::Moe => {
                let gate = self.linear(&m("gate.weight"), (EXPERTS, D))?;
                ensure!(gate.fmt == WeightFormat::Bf16, "{p}: the router must stay BF16");
                let bias = self.raw_f32(&m("gate.e_score_correction_bias"), EXPERTS)?;
                let experts = if self.ckpt.get(&m("experts.0.up_proj.weight"))?.dtype == DType::BF16 {
                    self.experts_from_bf16(&m("experts"))?
                } else {
                    self.experts_nvfp4(&m("experts"))?
                };
                Mixer::Moe(Moe {
                    gate,
                    bias,
                    fc1: self.linear(&m("fc1_latent_proj.weight"), (LAT, D))?,
                    fc2: self.linear(&m("fc2_latent_proj.weight"), (D, LAT))?,
                    experts,
                    shared_up: self.linear(&m("shared_experts.up_proj.weight"), (SH, D))?,
                    shared_down: self.linear(&m("shared_experts.down_proj.weight"), (D, SH))?,
                })
            }
        };
        Ok(Layer { norm, mixer })
    }

    fn stack_buffer(&mut self, bytes: usize) -> Result<&mut quench6k_cuda::PinnedBuffer> {
        if self.stack.as_ref().is_none_or(|b| b.len() < bytes) {
            self.stack = None;
            self.stack = Some(quench6k_cuda::PinnedBuffer::new(bytes)?);
        }
        Ok(self.stack.as_mut().unwrap())
    }

    /// `names[e]` (equal-sized tensors of `dtype` and `shape`) stacked in expert order into one device buffer.
    fn stacked(&mut self, names: &[String], dtype: DType, shape: &[usize]) -> Result<u64> {
        let each: usize = shape.iter().product::<usize>() * dtype.size();
        for name in names {
            let info = self.ckpt.get(name)?;
            ensure!(info.dtype == dtype && info.shape == shape, "{name}: {:?} {:?}, expected {dtype:?} {shape:?}", info.dtype, info.shape);
        }
        let total = each * names.len();
        let Some(s) = self.s else {
            return self.alloc(total);
        };
        self.stack_buffer(total)?;
        for (e, name) in names.iter().enumerate() {
            let (o, n) = self.fetch(name)?;
            let src = &self.staging.as_ref().unwrap().as_slice()[o..o + n];
            self.stack.as_mut().unwrap().as_mut_slice()[e * each..(e + 1) * each].copy_from_slice(src);
        }
        let dst = self.alloc(total)?;
        unsafe { s.copy_h2d(dst, self.stack.as_ref().unwrap().as_ptr(), total)? };
        s.synchronize()?;
        Ok(dst)
    }

    /// The routed experts under `p` (`p.{e}.{up,down}_proj`) as stored (NVFP4).
    fn experts_nvfp4(&mut self, p: &str) -> Result<RoutedExperts> {
        let names = |proj: &str, suffix: &str| -> Vec<String> { (0..EXPERTS).map(|e| format!("{p}.{e}.{proj}.{suffix}")).collect() };
        let s2 = |l: &mut Self, proj: &str| -> Result<u64> {
            let v: Vec<f32> = names(proj, "weight_scale_2").iter().map(|n| l.scalar(n)).collect::<Result<_>>()?;
            ensure!(v.iter().all(|x| x.is_finite() && *x > 0.0), "{p}: {proj} weight_scale_2 out of range");
            l.upload_slice(&v)
        };
        let up = nk::NvExperts {
            codes: self.stacked(&names("up_proj", "weight"), DType::U8, &[FF, LAT / 2])?,
            scales: self.stacked(&names("up_proj", "weight_scale"), DType::F8E4M3, &[FF, LAT / 16])?,
            s2: s2(self, "up_proj")?,
        };
        let down = nk::NvExperts {
            codes: self.stacked(&names("down_proj", "weight"), DType::U8, &[LAT, FF / 2])?,
            scales: self.stacked(&names("down_proj", "weight_scale"), DType::F8E4M3, &[LAT, FF / 16])?,
            s2: s2(self, "down_proj")?,
        };
        self.routed(up, down)
    }

    /// The per-expert pointer tables of `up` and `down` (grouped GEMM of large steps).
    fn routed(&mut self, up: nk::NvExperts, down: nk::NvExperts) -> Result<RoutedExperts> {
        let mut tables = [(0, 0); 2];
        for (slot, (e, n, k)) in [(up, FF, LAT), (down, LAT, FF)].into_iter().enumerate() {
            let c: Vec<u64> = (0..EXPERTS).map(|x| e.codes + (x * n * k / 2) as u64).collect();
            let sc: Vec<u64> = (0..EXPERTS).map(|x| e.scales + (x * n * k / 16) as u64).collect();
            tables[slot] = (self.upload_slice(&c)?, self.upload_slice(&sc)?);
        }
        Ok(RoutedExperts { up, down, tables })
    }

    /// BF16 routed experts (the MTP layer's) NVFP4-quantized at load into the layout of
    /// [`Loader::experts_nvfp4`], in groups of experts (bounded temporary memory).
    fn experts_from_bf16(&mut self, p: &str) -> Result<RoutedExperts> {
        const GROUP: usize = 64;
        let mut out = [0u64; 6];
        for (slot, (proj, n, k)) in [("up_proj", FF, LAT), ("down_proj", LAT, FF)].into_iter().enumerate() {
            let codes = self.alloc(EXPERTS * n * k / 2)?;
            let scales = self.alloc(EXPERTS * n * k / 16)?;
            let s2 = self.alloc(EXPERTS * 4)?;
            for e in 0..EXPERTS {
                let name = format!("{p}.{e}.{proj}.weight");
                let info = self.ckpt.get(&name)?;
                ensure!(info.dtype == DType::BF16 && info.shape == [n, k], "{name}: {:?} {:?}", info.dtype, info.shape);
            }
            out[3 * slot] = codes;
            out[3 * slot + 1] = scales;
            out[3 * slot + 2] = s2;
            let Some(s) = self.s else {
                continue;
            };
            let tmp = DeviceBuffer::new(GROUP * n * k * 2)?;
            let amax = DeviceBuffer::new(GROUP * 4)?;
            for e0 in (0..EXPERTS).step_by(GROUP) {
                for e in e0..e0 + GROUP {
                    self.copy_tensor(&format!("{p}.{e}.{proj}.weight"), tmp.addr() + ((e - e0) * n * k * 2) as u64)?;
                }
                quench6k_cuda::kernels::quantize_nvfp4(
                    s,
                    tmp.addr(),
                    GROUP,
                    n,
                    k,
                    codes + (e0 * n * k / 2) as u64,
                    scales + (e0 * n * k / 16) as u64,
                    s2 + (e0 * 4) as u64,
                    amax.addr(),
                )?;
                s.synchronize()?;
            }
        }
        let up = nk::NvExperts { codes: out[0], scales: out[1], s2: out[2] };
        let down = nk::NvExperts { codes: out[3], scales: out[4], s2: out[5] };
        self.routed(up, down)
    }

    /// Memory map of a host-resident tensor: (map of its file, byte offset of the tensor in it).
    fn map(&self, name: &str, dtype: DType, shape: &[usize]) -> Result<(Mmap, usize)> {
        let info = self.ckpt.get(name)?;
        ensure!(info.dtype == dtype && info.shape == shape, "{name}: {:?} {:?}, expected {dtype:?} {shape:?}", info.dtype, info.shape);
        let file = &self.ckpt.files[info.file];
        let map = unsafe { Mmap::map(&file.file) }.with_context(|| format!("mapping {}", file.path.display()))?;
        Ok((map, info.offset as usize))
    }
}

/// Device memory of the weights by part, computed from config.json and hf_quant_config.json (no
/// safetensors needed): what [`NemotronModel::load`] allocates.
#[derive(Debug, Default, Clone)]
pub struct WeightPlan {
    pub experts: usize,
    pub mamba: usize,
    pub attention: usize,
    pub moe_dense: usize,
    pub router: usize,
    pub norms_small: usize,
    pub lm_head: usize,
    pub mtp: usize,
    /// The BF16 embedding (host-resident, memory-mapped).
    pub embed_host: usize,
}

impl WeightPlan {
    pub fn device(&self) -> usize {
        self.experts + self.mamba + self.attention + self.moe_dense + self.router + self.norms_small + self.lm_head + self.mtp
    }

    /// From the model directory's config.json and hf_quant_config.json.
    pub fn read(dir: &Path) -> Result<Self> {
        let cfg = Config::read(dir)?;
        let q: Value = serde_json::from_slice(&std::fs::read(dir.join("hf_quant_config.json")).context("hf_quant_config.json")?)?;
        let ql = q.pointer("/quantization/quantized_layers").and_then(Value::as_object).context("hf_quant_config: quantized_layers")?;
        let stored = |module: &str| -> Stored {
            match ql.get(module).and_then(|v| v.get("quant_algo")).and_then(Value::as_str) {
                Some("FP8") => Stored::Fp8,
                Some("NVFP4") => Stored::Nvfp4,
                _ => Stored::Bf16,
            }
        };
        Ok(Self::compute(&cfg, &stored))
    }

    pub fn compute(cfg: &Config, stored: &dyn Fn(&str) -> Stored) -> Self {
        let lin = |module: &str, n: usize, k: usize| linear_bytes(kept(stored(module), &format!("{module}.weight"), k), n, k);
        // Codes, scales, s2 and the two pointer tables of up and down.
        let ex_nvfp4 = EXPERTS * 2 * (FF * LAT / 2 + FF * LAT / 16 + 4 + 16);
        let mut p = Self::default();
        let layer = |p: &mut Self, pre: &str, kind: Kind, mtp: bool| {
            let m = |x: &str| format!("{pre}.mixer.{x}");
            let mut b = D * 2;
            let (dense, which) = match kind {
                Kind::Mamba => (lin(&m("in_proj"), INP, D) + lin(&m("out_proj"), D, DI) + (CONV * CONVK + CONV + 3 * MH + DI) * 4, 0),
                Kind::Attention => {
                    // q/k/v are stacked: their (common) format is q's.
                    let qkv = linear_bytes(kept(stored(&m("q_proj")), &m("q_proj.weight"), D), QKV, D);
                    (qkv + lin(&m("o_proj"), D, H * HD), 1)
                }
                Kind::Moe => {
                    p.router += EXPERTS * D * 2 + EXPERTS * 4;
                    if mtp {
                        b += ex_nvfp4;
                    } else {
                        p.experts += ex_nvfp4;
                    }
                    let dense = lin(&m("fc1_latent_proj"), LAT, D)
                        + lin(&m("fc2_latent_proj"), D, LAT)
                        + lin(&m("shared_experts.up_proj"), SH, D)
                        + lin(&m("shared_experts.down_proj"), D, SH);
                    (dense, 2)
                }
            };
            if mtp {
                p.mtp += dense + b;
            } else {
                *[&mut p.mamba, &mut p.attention, &mut p.moe_dense][which] += dense;
                p.norms_small += b;
            }
        };
        for (i, &kind) in cfg.kinds.iter().enumerate() {
            layer(&mut p, &format!("backbone.layers.{i}"), kind, false);
        }
        p.norms_small += D * 2;
        p.lm_head = lin("lm_head", cfg.vocab, D);
        if !cfg.mtp_kinds.is_empty() && std::env::var("QUENCH6K_NEM_MTP").map_or(true, |v| v != "0") {
            for (j, &kind) in cfg.mtp_kinds.iter().enumerate() {
                layer(&mut p, &format!("mtp.layers.{j}"), kind, true);
            }
            // The MTP router counts under mtp.
            p.router -= EXPERTS * D * 2 + EXPERTS * 4;
            p.mtp += EXPERTS * D * 2 + EXPERTS * 4;
            p.mtp += lin("mtp.layers.0.eh_proj", D, 2 * D) + 3 * D * 2;
        }
        p.embed_host = cfg.vocab * D * 2;
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Value {
        serde_json::json!({
            "model_type": "nemotron_h", "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 2,
            "head_dim": 128, "mamba_num_heads": 128, "mamba_head_dim": 64, "n_groups": 8, "ssm_state_size": 128,
            "conv_kernel": 4, "n_routed_experts": 512, "num_experts_per_tok": 22, "moe_intermediate_size": 2688,
            "moe_latent_size": 1024, "moe_shared_expert_intermediate_size": 5376, "n_shared_experts": 1,
            "n_group": 1, "topk_group": 1, "use_conv_bias": true, "norm_topk_prob": true,
            "hybrid_override_pattern": "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME",
            "num_hidden_layers": 88, "mtp_hybrid_override_pattern": "*E", "num_nextn_predict_layers": 1,
            "vocab_size": 131072, "layer_norm_epsilon": 1e-5, "routed_scaling_factor": 5.0, "eos_token_id": 2,
            "bos_token_id": 1, "max_position_embeddings": 262144, "mamba_hidden_act": "silu", "mlp_hidden_act": "relu2",
            "sliding_window": null, "tie_word_embeddings": false
        })
    }

    #[test]
    fn parses_the_released_config() {
        let generation = serde_json::json!({"eos_token_id": [2, 11]});
        let c = Config::parse(&config(), Some(generation)).unwrap();
        assert_eq!(c.kinds.len(), 88);
        assert_eq!((c.count(Kind::Mamba), c.count(Kind::Moe), c.count(Kind::Attention)), (40, 40, 8));
        assert_eq!(c.mtp_kinds, [Kind::Attention, Kind::Moe]);
        assert_eq!(c.eos, [2, 11]);
        assert_eq!(c.dt_limit, (0.0, f32::INFINITY));
        let slots = c.slots();
        assert_eq!(&slots[..8], &[0, 0, 1, 1, 2, 2, 3, 0]);
        assert_eq!(slots[87], 39);
        assert_eq!((INP, CONV, QKV), (18560, 10240, 4608));
    }

    #[test]
    fn refuses_other_shapes() {
        let mut c = config();
        c["moe_latent_size"] = 512.into();
        assert!(Config::parse(&c, None).is_err());
        let mut c = config();
        c["hybrid_override_pattern"] = "M-E".into();
        c["num_hidden_layers"] = 3.into();
        assert!(Config::parse(&c, None).is_err());
    }

    #[test]
    fn weight_plan_counts_formats() {
        let c = Config::parse(&config(), None).unwrap();
        // Everything BF16 (quantized to INT8 at load) except the routed experts.
        let p = WeightPlan::compute(&c, &|m: &str| if m.contains(".experts.") { Stored::Nvfp4 } else { Stored::Bf16 });
        let g = quench6k_cuda::kernels::int8_group();
        let i8b = |n: usize, k: usize| n * k + n * (k / g) * 4;
        assert_eq!(p.lm_head, i8b(131072, D));
        assert_eq!(p.experts, 40 * EXPERTS * 2 * (FF * LAT / 2 + FF * LAT / 16 + 4 + 16));
        assert_eq!(p.mamba, 40 * (i8b(INP, D) + i8b(D, DI) + (CONV * CONVK + CONV + 3 * MH + DI) * 4));
        // FP8 stays 8-bit with a block scale table.
        let f = WeightPlan::compute(&c, &|m: &str| if m.ends_with("in_proj") { Stored::Fp8 } else { Stored::Bf16 });
        assert!(f.mamba < p.mamba);
        assert_eq!(linear_bytes(Kept::Fp8, INP, D), INP * D + INP.div_ceil(128) * 32 * 4);
    }
}
