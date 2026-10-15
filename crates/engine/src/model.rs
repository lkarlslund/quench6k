//! The text model's weights, bound to device addresses (SPEC §2–§9).

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use squash_cuda::kernels::WeightFormat;
use squash_cuda::{DeviceBuffer, Stream};

use crate::checkpoint::{Checkpoint, DType};
use crate::loader::DeviceWeights;

pub const LAYERS: usize = 48;
pub const D: usize = 2560;
pub const S: usize = 4;
pub const H: usize = S * D;
pub const VOCAB: usize = 248_320;
pub const EXPERTS: usize = 512;
pub const TOP_K: usize = 10;
pub const EXPERT_WIDTH: usize = 640;
pub const HC_RANK: usize = 320;
pub const EOS: i32 = 248_044;

pub fn is_attention(layer: usize) -> bool {
    layer % 4 == 3
}

/// A projection weight `[n, k]`: BF16, or an 8-bit format with per-row FP32 scales at `scale`.
#[derive(Clone, Copy, Debug)]
pub struct Linear {
    pub fmt: WeightFormat,
    pub addr: u64,
    pub scale: u64,
    pub n: usize,
    pub k: usize,
}

/// Projections that share an input, computed into one output row per token: member `i` fills columns
/// `parts[i].1 ..` of a row of width `n`. With `fused`, the members are contiguous rows of one weight
/// and a single GEMV/GEMM computes all of them.
pub struct Proj {
    pub parts: Vec<(Linear, usize)>,
    pub fused: Option<Linear>,
    pub n: usize,
}

impl Proj {
    /// Output column where member `i` starts.
    pub fn col(&self, i: usize) -> usize {
        self.parts[i].1
    }
}

/// A HyperConnection's weights (SPEC §4). `down` holds the low-rank down projection and, except in the
/// final mixers, the 4-row injection projection after it; both have the `(1 + norm)` weight folded into
/// their columns at load.
pub struct Hc {
    pub norm: u64,
    pub down: Proj,
    pub up: Linear,
    pub injects: bool,
}

/// `proj`: in_proj_qkv | in_proj_z | in_proj_a | in_proj_b.
pub struct Gdn {
    pub proj: Proj,
    pub conv: u64,
    pub a_log: u64,
    pub dt_bias: u64,
    pub norm: u64,
    pub out: Linear,
}

/// `proj`: q_proj (query and gate per head) | k_proj | v_proj | indexer index_qk_proj.
pub struct Qsa {
    pub proj: Proj,
    pub o: Linear,
    pub q_norm: u64,
    pub k_norm: u64,
    pub index_q_norm: u64,
    pub index_k_norm: u64,
}

pub enum Block {
    Gdn(Gdn),
    Qsa(Qsa),
}

/// Per-projection device tables over the 512 experts: code and scale addresses, and `weight_scale_2`.
pub struct ExpertProj {
    pub weights: u64,
    pub scales: u64,
    pub scale2: u64,
}

/// `proj`: router | shared gate_proj | shared up_proj | shared_expert_gate.
pub struct Moe {
    pub proj: Proj,
    pub shared_down: Linear,
    pub gate: ExpertProj,
    pub up: ExpertProj,
    pub down: ExpertProj,
    _tables: DeviceBuffer,
}

pub struct Ple {
    pub key: Linear,
    pub value: Linear,
    pub norm_key: u64,
    pub norm_query: u64,
    pub norm_conv: u64,
    pub conv: u64,
    pub multipliers: [i64; 3],
    pub sizes: [i64; 16],
    pub offsets: [i64; 16],
}

pub struct Layer {
    pub attn_hc: Hc,
    pub mlp_hc: Hc,
    pub block: Block,
    pub moe: Moe,
    pub ple: Option<Ple>,
}

/// The host-resident PLE table: 128 memory-mapped shards of `[rows, 160]` BF16.
pub struct PleTable {
    maps: Vec<(Mmap, usize)>,
    rows_per_shard: usize,
}

impl PleTable {
    pub const WIDTH: usize = 160;

    fn open(ckpt: &Checkpoint) -> Result<Self> {
        let prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";
        let mut shards: Vec<(usize, &str)> = ckpt
            .tensors
            .keys()
            .filter_map(|n| n.strip_prefix(prefix).and_then(|r| r.strip_suffix(".weight")).map(|i| (i.parse().unwrap(), n.as_str())))
            .collect();
        shards.sort();
        ensure!(!shards.is_empty(), "checkpoint has no PLE table");
        let mut maps = Vec::with_capacity(shards.len());
        let mut rows_per_shard = 0;
        for (i, (index, name)) in shards.iter().enumerate() {
            ensure!(*index == i, "PLE shard {i} missing");
            let info = ckpt.get(name)?;
            ensure!(info.dtype == DType::BF16 && info.shape[1] == Self::WIDTH, "{name}: unexpected PLE shard layout");
            rows_per_shard = info.shape[0];
            let file = &ckpt.files[info.file];
            let map = unsafe { Mmap::map(&file.file) }.with_context(|| format!("mapping {}", file.path.display()))?;
            maps.push((map, info.offset as usize));
        }
        Ok(Self { maps, rows_per_shard })
    }

    /// Asks the kernel to start reading row `r` from disk if it is not resident, so that a batch of cold
    /// rows is fetched in parallel instead of one page fault at a time.
    pub fn prefetch(&self, r: usize) {
        let (map, base) = &self.maps[r / self.rows_per_shard];
        let start = base + (r % self.rows_per_shard) * Self::WIDTH * 2;
        let _ = map.advise_range(memmap2::Advice::WillNeed, start, Self::WIDTH * 2);
    }

    /// Appends row `r` (160 BF16 values) to `out`.
    pub fn gather(&self, r: usize, out: &mut Vec<u16>) {
        let (map, base) = &self.maps[r / self.rows_per_shard];
        let start = base + (r % self.rows_per_shard) * Self::WIDTH * 2;
        let bytes = &map[start..start + Self::WIDTH * 2];
        out.extend(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])));
    }
}

pub struct Model {
    pub weights: DeviceWeights,
    pub embed: u64,
    pub lm_head: Linear,
    pub final_hc: Hc,
    pub layers: Vec<Layer>,
    pub ple_table: PleTable,
}

fn read_i64s<const N: usize>(ckpt: &Checkpoint, name: &str) -> Result<[i64; N]> {
    let bytes = ckpt.read(name)?;
    ensure!(bytes.len() == 8 * N, "{name}: expected {N} I64 values");
    Ok(std::array::from_fn(|i| i64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap())))
}

impl Model {
    pub fn bind(ckpt: &Checkpoint, weights: DeviceWeights, stream: &Stream) -> Result<Self> {
        let w = |name: &str| -> Result<u64> { Ok(weights.get(name)?.addr) };
        let lin = |name: &str| -> Result<Linear> {
            let t = weights.get(name)?;
            ensure!(t.shape.len() == 2, "{name}: not a 2-D projection");
            let fmt = match t.dtype {
                DType::BF16 => WeightFormat::Bf16,
                DType::F8E4M3 => WeightFormat::Fp8,
                DType::I8 => WeightFormat::Int8,
                DType::I8G => WeightFormat::Int8Group,
                other => anyhow::bail!("{name}: {other:?} is not a projection weight format"),
            };
            Ok(Linear { fmt, addr: t.addr, scale: t.scale, n: t.shape[0], k: t.shape[1] })
        };
        let proj = |(group, members): (String, Vec<String>)| -> Result<Proj> {
            let mut parts = Vec::with_capacity(members.len());
            let mut n = 0;
            for m in &members {
                let l = lin(m)?;
                parts.push((l, n));
                n += l.n;
            }
            let fused = if weights.tensors.contains_key(&group) {
                Some(lin(&group)?)
            } else if parts.len() == 1 {
                Some(parts[0].0)
            } else {
                None
            };
            Ok(Proj { parts, fused, n })
        };
        let hc = |prefix: &str, injects: bool| -> Result<Hc> {
            Ok(Hc {
                norm: w(&format!("{prefix}.hc_norm.weight"))?,
                down: proj(hc_group(prefix, injects))?,
                up: lin(&format!("{prefix}.input_mix_weight_up.weight"))?,
                injects,
            })
        };

        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("model.language_model.layers.{i}");
            let block = if is_attention(i) {
                let a = format!("{p}.self_attn");
                Block::Qsa(Qsa {
                    proj: proj(qsa_group(&p))?,
                    o: lin(&format!("{a}.o_proj.weight"))?,
                    q_norm: w(&format!("{a}.q_norm.weight"))?,
                    k_norm: w(&format!("{a}.k_norm.weight"))?,
                    index_q_norm: w(&format!("{a}.indexer.q_layernorm.weight"))?,
                    index_k_norm: w(&format!("{a}.indexer.k_layernorm.weight"))?,
                })
            } else {
                let g = format!("{p}.linear_attn");
                Block::Gdn(Gdn {
                    proj: proj(gdn_group(&p))?,
                    conv: w(&format!("{g}.conv1d.weight"))?,
                    a_log: w(&format!("{g}.A_log"))?,
                    dt_bias: w(&format!("{g}.dt_bias"))?,
                    norm: w(&format!("{g}.norm.weight"))?,
                    out: lin(&format!("{g}.out_proj.weight"))?,
                })
            };

            // Expert tables: [gate codes, gate scales, up codes, up scales, down codes, down scales] x 512 u64,
            // then [gate, up, down] weight_scale_2 x 512 f32.
            let mut addrs = vec![0u64; 6 * EXPERTS];
            let mut scale2 = vec![0f32; 3 * EXPERTS];
            for e in 0..EXPERTS {
                for (j, proj) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                    let base = format!("{p}.mlp.experts.{e}.{proj}");
                    addrs[(2 * j) * EXPERTS + e] = w(&format!("{base}.weight"))?;
                    addrs[(2 * j + 1) * EXPERTS + e] = w(&format!("{base}.weight_scale"))?;
                    let s2 = ckpt.read(&format!("{base}.weight_scale_2"))?;
                    scale2[j * EXPERTS + e] = f32::from_le_bytes(s2[..4].try_into()?);
                }
            }
            let tables = DeviceBuffer::new(addrs.len() * 8 + scale2.len() * 4)?;
            stream.upload_to(tables.addr(), &addrs)?;
            stream.upload_to(tables.addr() + (addrs.len() * 8) as u64, &scale2)?;
            let t = |k: usize| tables.addr() + (k * EXPERTS * 8) as u64;
            let s2 = |j: usize| tables.addr() + (addrs.len() * 8 + j * EXPERTS * 4) as u64;
            let m = format!("{p}.mlp");
            let moe = Moe {
                proj: proj(moe_group(&p))?,
                shared_down: lin(&format!("{m}.shared_expert.down_proj.weight"))?,
                gate: ExpertProj { weights: t(0), scales: t(1), scale2: s2(0) },
                up: ExpertProj { weights: t(2), scales: t(3), scale2: s2(1) },
                down: ExpertProj { weights: t(4), scales: t(5), scale2: s2(2) },
                _tables: tables,
            };

            let ple = if i == 1 {
                let q = format!("{p}.ple");
                Some(Ple {
                    key: lin(&format!("{q}.key_proj.weight"))?,
                    value: lin(&format!("{q}.value_proj.weight"))?,
                    norm_key: w(&format!("{q}.norm_key.weight"))?,
                    norm_query: w(&format!("{q}.norm_query.weight"))?,
                    norm_conv: w(&format!("{q}.norm_conv.weight"))?,
                    conv: w(&format!("{q}.conv1d.weight"))?,
                    multipliers: read_i64s(ckpt, &format!("{q}.ple_embedding.layer_multipliers"))?,
                    sizes: read_i64s(ckpt, &format!("{q}.ple_embedding.ngram_heads_vocab_sizes"))?,
                    offsets: read_i64s(ckpt, &format!("{q}.ple_embedding.ngram_heads_offsets"))?,
                })
            } else {
                None
            };

            layers.push(Layer {
                attn_hc: hc(&format!("{p}.attn_hyper_connection"), true)?,
                mlp_hc: hc(&format!("{p}.mlp_hyper_connection"), true)?,
                block,
                moe,
                ple,
            });
        }

        Ok(Self {
            embed: w("model.language_model.embed_tokens.weight")?,
            lm_head: lin("lm_head.weight")?,
            final_hc: hc("model.language_model.hyper_connection_mixer", false)?,
            layers,
            ple_table: PleTable::open(ckpt)?,
            weights,
        })
    }
}

fn hc_group(prefix: &str, injects: bool) -> (String, Vec<String>) {
    let mut members = vec![format!("{prefix}.input_mix_weight_down.weight")];
    if injects {
        members.push(format!("{prefix}.block_inject_weight.weight"));
    }
    (format!("{prefix}.squash.down_inject"), members)
}

fn gdn_group(layer: &str) -> (String, Vec<String>) {
    let g = format!("{layer}.linear_attn");
    (format!("{g}.squash.in_proj"), ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"].iter().map(|m| format!("{g}.{m}.weight")).collect())
}

fn qsa_group(layer: &str) -> (String, Vec<String>) {
    let a = format!("{layer}.self_attn");
    (format!("{a}.squash.in_proj"), ["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"].iter().map(|m| format!("{a}.{m}.weight")).collect())
}

fn moe_group(layer: &str) -> (String, Vec<String>) {
    let m = format!("{layer}.mlp");
    (format!("{m}.squash.in_proj"), ["gate", "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert_gate"].iter().map(|x| format!("{m}.{x}.weight")).collect())
}

/// Every multi-member projection group of the text model; quantized profiles store each contiguously.
pub fn projection_groups() -> Vec<(String, Vec<String>)> {
    let mut groups = Vec::new();
    for i in 0..LAYERS {
        let p = format!("model.language_model.layers.{i}");
        groups.push(hc_group(&format!("{p}.attn_hyper_connection"), true));
        groups.push(hc_group(&format!("{p}.mlp_hyper_connection"), true));
        groups.push(if is_attention(i) { qsa_group(&p) } else { gdn_group(&p) });
        groups.push(moe_group(&p));
    }
    groups
}

/// Weight precision profile for the non-expert projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Checkpoint BF16 everywhere except the NVFP4 experts.
    Bf16,
    /// FP8 E4M3 (per-row scales) for the large projections, see `selection::eight_bit`.
    Fp8,
    /// INT8 (per-row scales) for the same projections.
    Int8,
    /// INT8 with a scale per 64 weights of a row, for the same projections.
    Int8Group,
}

/// Uploads the text model in the given precision and binds it.
pub fn load_text(ckpt: &Checkpoint, precision: Precision, opts: crate::loader::LoadOptions, stream: &Stream) -> Result<(Model, crate::loader::LoadStats)> {
    use crate::{loader, selection};
    let format = match precision {
        Precision::Bf16 => WeightFormat::Bf16,
        Precision::Fp8 => WeightFormat::Fp8,
        Precision::Int8 => WeightFormat::Int8,
        Precision::Int8Group => WeightFormat::Int8Group,
    };
    let quantized = |n: &str| format != WeightFormat::Bf16 && selection::eight_bit(n);
    let (mut weights, mut stats) = loader::upload(ckpt, |n| selection::text(n) && !quantized(n), opts)?;
    // Fold each HyperConnection's (1 + w) norm weight into its down and injection projections (SPEC §4),
    // so small batches can project the raw residual and apply the per-stream RMS factor afterwards.
    let fold = |name: &str, addr: u64, n: usize, k: usize| -> Result<()> {
        let prefix = name.strip_suffix(".input_mix_weight_down.weight").or_else(|| name.strip_suffix(".block_inject_weight.weight"));
        match prefix {
            Some(prefix) => squash_cuda::kernels::fold_norm(stream, addr, weights.get(&format!("{prefix}.hc_norm.weight"))?.addr, n, k),
            None => Ok(()),
        }
    };
    if format == WeightFormat::Bf16 {
        let names: Vec<String> = weights.tensors.keys().filter(|n| n.ends_with(".input_mix_weight_down.weight") || n.ends_with(".block_inject_weight.weight")).cloned().collect();
        for name in names {
            let t = weights.get(&name)?;
            fold(&name, t.addr, t.shape[0], t.shape[1])?;
        }
        stream.synchronize()?;
    } else {
        let (converted, s) = loader::upload_quantized(ckpt, |n| selection::text(n) && quantized(n), format, &projection_groups(), &fold, opts, stream)?;
        stats.bytes += s.bytes;
        stats.seconds += s.seconds;
        stats.segments += s.segments;
        weights.merge(converted);
    }
    Ok((Model::bind(ckpt, weights, stream)?, stats))
}
