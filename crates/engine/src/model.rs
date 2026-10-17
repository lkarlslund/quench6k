//! The text model's weights, bound to device addresses (SPEC §2–§9).

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use quench6k_cuda::kernels::WeightFormat;
use quench6k_cuda::{DeviceBuffer, Stream};

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
    _buffers: Vec<DeviceBuffer>,
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
            // Rows are read at random: without this every fault also reads ahead ~128 KiB around a
            // 320-byte row, and cold prompts become bound by disk bandwidth.
            map.advise(memmap2::Advice::Random).context("madvise(MADV_RANDOM) on the PLE table")?;
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

    /// Reads row `r`, faulting it into the page cache (blocking).
    pub fn touch(&self, r: usize) -> u8 {
        let (map, base) = &self.maps[r / self.rows_per_shard];
        let start = base + (r % self.rows_per_shard) * Self::WIDTH * 2;
        // The row may straddle a page boundary.
        unsafe { std::ptr::read_volatile(&map[start]) ^ std::ptr::read_volatile(&map[start + Self::WIDTH * 2 - 1]) }
    }

    /// Appends row `r` (160 BF16 values) to `out`.
    pub fn gather(&self, r: usize, out: &mut Vec<u16>) {
        let (map, base) = &self.maps[r / self.rows_per_shard];
        let start = base + (r % self.rows_per_shard) * Self::WIDTH * 2;
        let bytes = &map[start..start + Self::WIDTH * 2];
        out.extend(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])));
    }
}

/// The MTP drafter (SPEC §10): one QSA decoder layer with its own mixer, sharing the embedding and LM head.
pub struct Mtp {
    /// NVFP4 copy of the LM head for drafting: (codes, block scales, scale2) and their storage.
    pub draft_head: (u64, u64, u64),
    /// 1-bit sketch of the LM head that shortlists draft candidates (see `quench6k_draft_head_argmax`).
    pub draft_sketch: u64,
    pub _draft_head: Vec<DeviceBuffer>,
    pub fc_embedding: Linear,
    pub fc_hidden: Linear,
    pub norm_embedding: u64,
    pub norm_hidden: u64,
    pub layer: Layer,
    pub final_hc: Hc,
}

/// Device pointer tables over the 512 experts of one MoE block, plus any buffers they point into.
pub struct ExpertTables {
    pub gate: ExpertProj,
    pub up: ExpertProj,
    pub down: ExpertProj,
    pub buffers: Vec<DeviceBuffer>,
}

impl ExpertTables {
    /// `addrs`: [gate codes, gate scales, up codes, up scales, down codes, down scales][expert];
    /// `scale2`: [gate, up, down][expert].
    pub fn upload(addrs: &[[u64; EXPERTS]; 6], scale2: &[[f32; EXPERTS]; 3], mut buffers: Vec<DeviceBuffer>, stream: &Stream) -> Result<Self> {
        let flat_a: Vec<u64> = addrs.iter().flatten().copied().collect();
        let flat_s: Vec<f32> = scale2.iter().flatten().copied().collect();
        let tables = DeviceBuffer::new(flat_a.len() * 8 + flat_s.len() * 4)?;
        stream.upload_to(tables.addr(), &flat_a)?;
        stream.upload_to(tables.addr() + (flat_a.len() * 8) as u64, &flat_s)?;
        let t = |k: usize| tables.addr() + (k * EXPERTS * 8) as u64;
        let s2 = |j: usize| tables.addr() + (flat_a.len() * 8 + j * EXPERTS * 4) as u64;
        let (gate, up, down) = (
            ExpertProj { weights: t(0), scales: t(1), scale2: s2(0) },
            ExpertProj { weights: t(2), scales: t(3), scale2: s2(1) },
            ExpertProj { weights: t(4), scales: t(5), scale2: s2(2) },
        );
        buffers.push(tables);
        Ok(Self { gate, up, down, buffers })
    }
}

pub struct Model {
    pub weights: DeviceWeights,
    pub mtp: Option<Mtp>,
    pub embed: u64,
    pub lm_head: Linear,
    pub final_hc: Hc,
    pub layers: Vec<Layer>,
    pub ple_table: PleTable,
    /// Direction-removal steering pack, applied with each sequence's strength (see `steering`).
    pub steering: Option<crate::steering::Steering>,
}

fn read_i64s<const N: usize>(ckpt: &Checkpoint, name: &str) -> Result<[i64; N]> {
    let bytes = ckpt.read(name)?;
    ensure!(bytes.len() == 8 * N, "{name}: expected {N} I64 values");
    Ok(std::array::from_fn(|i| i64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap())))
}

impl Model {
    /// Elements of the largest INT8-group projection weight other than the LM head (which keeps its
    /// INT8 GEMM), 0 without any.
    pub fn max_int8g_elems(&self) -> usize {
        let head = self.lm_head.addr;
        self.weights.tensors.values().filter(|t| t.dtype == DType::I8G && t.addr != head).map(|t| t.shape.iter().product::<usize>()).max().unwrap_or(0)
    }

    /// Binds the first `layers` text layers (all 48 for the real model).
    pub fn bind(ckpt: &Checkpoint, weights: DeviceWeights, mtp_experts: Option<(ExpertTables, (u64, u64, u64), u64, Vec<DeviceBuffer>)>, layers: usize, stream: &Stream) -> Result<Self> {
        ensure!((2..=LAYERS).contains(&layers), "{layers} layers: between 2 (layer 1 holds the PLE) and {LAYERS}");
        let w = |name: &str| -> Result<u64> { Ok(weights.get(name)?.addr) };
        let lin = |name: &str| -> Result<Linear> {
            let t = weights.get(name)?;
            ensure!(t.shape.len() == 2, "{name}: not a 2-D projection");
            let fmt = match t.dtype {
                DType::BF16 => WeightFormat::Bf16,
                DType::I8G => WeightFormat::Int8Group,
                DType::MX8 => WeightFormat::Mxfp8,
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

        let block = |p: &str, attention: bool| -> Result<Block> {
            Ok(if attention {
                let a = format!("{p}.self_attn");
                Block::Qsa(Qsa {
                    proj: proj(qsa_group(p))?,
                    o: lin(&format!("{a}.o_proj.weight"))?,
                    q_norm: w(&format!("{a}.q_norm.weight"))?,
                    k_norm: w(&format!("{a}.k_norm.weight"))?,
                    index_q_norm: w(&format!("{a}.indexer.q_layernorm.weight"))?,
                    index_k_norm: w(&format!("{a}.indexer.k_layernorm.weight"))?,
                })
            } else {
                let g = format!("{p}.linear_attn");
                Block::Gdn(Gdn {
                    proj: proj(gdn_group(p))?,
                    conv: w(&format!("{g}.conv1d.weight"))?,
                    a_log: w(&format!("{g}.A_log"))?,
                    dt_bias: w(&format!("{g}.dt_bias"))?,
                    norm: w(&format!("{g}.norm.weight"))?,
                    out: lin(&format!("{g}.out_proj.weight"))?,
                })
            })
        };
        let moe = |p: &str, tables: ExpertTables| -> Result<Moe> {
            let m = format!("{p}.mlp");
            Ok(Moe {
                proj: proj(moe_group(p))?,
                shared_down: lin(&format!("{m}.shared_expert.down_proj.weight"))?,
                gate: tables.gate,
                up: tables.up,
                down: tables.down,
                _buffers: tables.buffers,
            })
        };

        let count = layers;
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let p = format!("model.language_model.layers.{i}");
            // Expert codes, scales and weight_scale_2 as stored per expert in the checkpoint.
            let mut addrs = [[0u64; EXPERTS]; 6];
            let mut scale2 = [[0f32; EXPERTS]; 3];
            for e in 0..EXPERTS {
                for (j, proj) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                    let base = format!("{p}.mlp.experts.{e}.{proj}");
                    addrs[2 * j][e] = w(&format!("{base}.weight"))?;
                    addrs[2 * j + 1][e] = w(&format!("{base}.weight_scale"))?;
                    let s2 = ckpt.read(&format!("{base}.weight_scale_2"))?;
                    scale2[j][e] = f32::from_le_bytes(s2[..4].try_into()?);
                }
            }
            let tables = ExpertTables::upload(&addrs, &scale2, Vec::new(), stream)?;

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
                block: block(&p, is_attention(i))?,
                moe: moe(&p, tables)?,
                ple,
            });
        }

        let mtp = match mtp_experts {
            None => None,
            Some((tables, head, sketch, head_bufs)) => Some(Mtp {
                draft_head: head,
                draft_sketch: sketch,
                _draft_head: head_bufs,
                fc_embedding: lin("mtp.fc_embedding.weight")?,
                fc_hidden: lin("mtp.fc_hidden.weight")?,
                norm_embedding: w("mtp.pre_fc_norm_embedding.weight")?,
                norm_hidden: w("mtp.pre_fc_norm_hidden.weight")?,
                layer: Layer {
                    attn_hc: hc("mtp.layers.0.attn_hyper_connection", true)?,
                    mlp_hc: hc("mtp.layers.0.mlp_hyper_connection", true)?,
                    block: block("mtp.layers.0", true)?,
                    moe: moe("mtp.layers.0", tables)?,
                    ple: None,
                },
                final_hc: hc("mtp.hyper_connection_mixer", false)?,
            }),
        };

        Ok(Self {
            embed: w("model.language_model.embed_tokens.weight")?,
            lm_head: lin("lm_head.weight")?,
            final_hc: hc("model.language_model.hyper_connection_mixer", false)?,
            layers,
            mtp,
            ple_table: PleTable::open(ckpt)?,
            weights,
            steering: None,
        })
    }
}

fn hc_group(prefix: &str, injects: bool) -> (String, Vec<String>) {
    let mut members = vec![format!("{prefix}.input_mix_weight_down.weight")];
    if injects {
        members.push(format!("{prefix}.block_inject_weight.weight"));
    }
    (format!("{prefix}.quench6k.down_inject"), members)
}

fn gdn_group(layer: &str) -> (String, Vec<String>) {
    let g = format!("{layer}.linear_attn");
    (format!("{g}.quench6k.in_proj"), ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"].iter().map(|m| format!("{g}.{m}.weight")).collect())
}

fn qsa_group(layer: &str) -> (String, Vec<String>) {
    let a = format!("{layer}.self_attn");
    (format!("{a}.quench6k.in_proj"), ["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"].iter().map(|m| format!("{a}.{m}.weight")).collect())
}

fn moe_group(layer: &str) -> (String, Vec<String>) {
    let m = format!("{layer}.mlp");
    (format!("{m}.quench6k.in_proj"), ["gate", "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert_gate"].iter().map(|x| format!("{m}.{x}.weight")).collect())
}

/// Every multi-member projection group of the text model; quantized profiles store each contiguously.
pub fn projection_groups(mtp: bool, layers: usize) -> Vec<(String, Vec<String>)> {
    let mut groups = Vec::new();
    for i in 0..layers {
        let p = format!("model.language_model.layers.{i}");
        groups.push(hc_group(&format!("{p}.attn_hyper_connection"), true));
        groups.push(hc_group(&format!("{p}.mlp_hyper_connection"), true));
        groups.push(if is_attention(i) { qsa_group(&p) } else { gdn_group(&p) });
        groups.push(moe_group(&p));
    }
    if mtp {
        let p = "mtp.layers.0";
        groups.push(hc_group(&format!("{p}.attn_hyper_connection"), true));
        groups.push(hc_group(&format!("{p}.mlp_hyper_connection"), true));
        groups.push(qsa_group(p));
        groups.push(moe_group(p));
    }
    groups
}

/// NVFP4 copy of the LM head for the drafter, quantized from the checkpoint's BF16 head.
fn load_draft_head(ckpt: &Checkpoint, opts: crate::loader::LoadOptions, stream: &Stream) -> Result<((u64, u64, u64), u64, Vec<DeviceBuffer>)> {
    let (staged, _) = crate::loader::upload(ckpt, |t| t == "lm_head.weight", opts)?;
    let src = staged.get("lm_head.weight")?;
    let codes = DeviceBuffer::new(VOCAB * D / 2)?;
    let scales = DeviceBuffer::new(VOCAB * D / 16)?;
    let s2 = DeviceBuffer::new(4)?;
    let amax = DeviceBuffer::new(4)?;
    quench6k_cuda::kernels::quantize_nvfp4(stream, src.addr, 1, VOCAB, D, codes.addr(), scales.addr(), s2.addr(), amax.addr())?;
    let sketch = DeviceBuffer::new(quench6k_cuda::kernels::draft_sketch_bytes(VOCAB))?;
    quench6k_cuda::kernels::build_draft_sketch(stream, src.addr, sketch.addr(), VOCAB)?;
    stream.synchronize()?;
    Ok(((codes.addr(), scales.addr(), s2.addr()), sketch.addr(), vec![codes, scales, s2, sketch]))
}

/// Uploads the MTP layer's BF16 experts one tensor at a time and NVFP4-quantizes them into the layout
/// of the main model's experts (gate_up [512, 1280, 2560] holds gate rows 0..640 then up rows per expert).
fn load_mtp_experts(ckpt: &Checkpoint, opts: crate::loader::LoadOptions, stream: &Stream) -> Result<ExpertTables> {
    use quench6k_cuda::kernels as k;
    let mut addrs = [[0u64; EXPERTS]; 6];
    let mut scale2 = [[0f32; EXPERTS]; 3];
    let mut buffers = Vec::new();
    // (tensor, matrices per expert, rows, cols, first projection index)
    for (name, per_expert, n, kk, proj) in [("mtp.layers.0.mlp.experts.gate_up_proj", 2, EXPERT_WIDTH, D, 0), ("mtp.layers.0.mlp.experts.down_proj", 1, D, EXPERT_WIDTH, 2)] {
        let (staged, _) = crate::loader::upload(ckpt, |t| t == name, opts)?;
        let src = staged.get(name)?;
        let count = EXPERTS * per_expert;
        let codes = DeviceBuffer::new(count * n * kk / 2)?;
        let scales = DeviceBuffer::new(count * n * kk / 16)?;
        let s2 = DeviceBuffer::new(count * 4)?;
        let amax = DeviceBuffer::new(count * 4)?;
        k::quantize_nvfp4(stream, src.addr, count, n, kk, codes.addr(), scales.addr(), s2.addr(), amax.addr())?;
        let s2_host: Vec<f32> = stream.download(s2.addr(), count)?;
        for e in 0..EXPERTS {
            for j in 0..per_expert {
                let m = e * per_expert + j;
                addrs[2 * (proj + j)][e] = codes.addr() + (m * n * kk / 2) as u64;
                addrs[2 * (proj + j) + 1][e] = scales.addr() + (m * n * kk / 16) as u64;
                scale2[proj + j][e] = s2_host[m];
            }
        }
        buffers.push(codes);
        buffers.push(scales);
    }
    ExpertTables::upload(&addrs, &scale2, buffers, stream)
}

/// Weight precision profile for the non-expert projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Checkpoint BF16 everywhere except the NVFP4 experts.
    Bf16,
    /// INT8 with a scale per 64 weights of a row for the projections in `selection::eight_bit`.
    /// (FP8 E4M3 and per-row INT8 were measured and fail the correctness gate.)
    Int8Group,
}

/// Uploads the text model (and with `mtp`, the MTP drafter) in the given precision and binds it.
pub fn load_text(ckpt: &Checkpoint, precision: Precision, mtp: bool, opts: crate::loader::LoadOptions, stream: &Stream) -> Result<(Model, crate::loader::LoadStats)> {
    use crate::{loader, selection};
    let format = match precision {
        Precision::Bf16 => WeightFormat::Bf16,
        Precision::Int8Group => WeightFormat::Int8Group,
    };
    // A truncated model (opts.layers < LAYERS) leaves out the later layers' tensors.
    let in_layers = |n: &str| match n.strip_prefix("model.language_model.layers.").and_then(|r| r.split('.').next()).and_then(|i| i.parse::<usize>().ok()) {
        Some(i) => i < opts.layers,
        None => true,
    };
    let wanted = |n: &str| (selection::text(n) && in_layers(n)) || (mtp && selection::mtp(n) && !n.contains(".experts."));
    let quantized = |n: &str| format != WeightFormat::Bf16 && selection::eight_bit(n);
    let (mut weights, mut stats) = loader::upload(ckpt, |n| wanted(n) && !quantized(n), opts)?;
    // Fold each HyperConnection's (1 + w) norm weight into its down and injection projections (SPEC §4),
    // so small batches can project the raw residual and apply the per-stream RMS factor afterwards.
    // Experiment: QUENCH6K_FAKE_FP4_W (comma-separated classes) rounds those BF16 weights to NVFP4 values
    // (per-row scale, least-error E4M3 block scales per 16) before they are quantized as usual.
    let fake_fp4 = std::env::var("QUENCH6K_FAKE_FP4_W").ok();
    let fp4_class = |n: &str| {
        fake_fp4.as_deref().is_some_and(|set| {
            set.split(',').any(|c| match c {
                "qkv" => n.ends_with(".linear_attn.in_proj_qkv.weight"),
                "z" => n.ends_with(".linear_attn.in_proj_z.weight"),
                "gdnout" => n.ends_with(".linear_attn.out_proj.weight"),
                "attnin" => n.contains(".self_attn.") && !n.ends_with(".o_proj.weight"),
                "attno" => n.ends_with(".self_attn.o_proj.weight"),
                "shared" => n.contains(".shared_expert.") && !n.ends_with("shared_expert_gate.weight"),
                "hc" => n.contains("hyper_connection."),
                "router" => n.ends_with(".mlp.gate.weight"),
                _ => false,
            }) && (set.split(',').any(|c| c == "mtp") || !n.starts_with("mtp."))
        })
    };
    let fold = |name: &str, addr: u64, n: usize, k: usize| -> Result<()> {
        let prefix = name.strip_suffix(".input_mix_weight_down.weight").or_else(|| name.strip_suffix(".block_inject_weight.weight"));
        if let Some(prefix) = prefix {
            quench6k_cuda::kernels::fold_norm(stream, addr, weights.get(&format!("{prefix}.hc_norm.weight"))?.addr, n, k)?;
        }
        if fp4_class(name) {
            // QUENCH6K_FAKE_BITS=b (2..8): b-bit integers with a scale per 64 instead of NVFP4.
            match std::env::var("QUENCH6K_FAKE_BITS").ok().and_then(|v| v.parse::<usize>().ok()) {
                Some(bits) => quench6k_cuda::kernels::fake_int(stream, addr, n, k, 64, bits)?,
                None => quench6k_cuda::kernels::fake_fp4_act(stream, addr, n, k)?,
            }
        }
        Ok(())
    };
    if format == WeightFormat::Bf16 {
        let names: Vec<String> = weights.tensors.keys().filter(|n| n.ends_with(".input_mix_weight_down.weight") || n.ends_with(".block_inject_weight.weight")).cloned().collect();
        for name in names {
            let t = weights.get(&name)?;
            fold(&name, t.addr, t.shape[0], t.shape[1])?;
        }
        // Experiment: the projections that would be 8-bit rounded to MXFP8 in place.
        // QUENCH6K_FAKE_MXFP8_SET: comma-separated classes (gdn, gdnout, attn) instead of everything.
        let set = std::env::var("QUENCH6K_FAKE_MXFP8_SET").ok();
        let in_set = |n: &str| match &set {
            None => true,
            Some(set) => set.split(',').any(|c| match c {
                "gdn" => n.contains(".linear_attn.in_proj"),
                "gdnout" => n.contains(".linear_attn.out_proj"),
                "attn" => n.contains(".self_attn."),
                "hc" => n.contains("hyper_connection."),
                "router" => n.ends_with(".mlp.gate.weight"),
                _ => false,
            }),
        };
        if std::env::var_os("QUENCH6K_FAKE_MXFP8").is_some() || set.is_some() {
            let names: Vec<String> = weights.tensors.keys().filter(|n| selection::eight_bit(n) && in_set(n)).cloned().collect();
            for name in names {
                let t = weights.get(&name)?;
                if t.shape.len() == 2 && t.shape[1] % 32 == 0 {
                    quench6k_cuda::kernels::fake_mxfp8(stream, t.addr, t.shape[0], t.shape[1], t.shape[1])?;
                }
            }
        }
        stream.synchronize()?;
    } else {
        let (converted, s) = loader::upload_quantized(ckpt, |n| wanted(n) && quantized(n), format, &projection_groups(mtp, opts.layers), &fold, opts, stream)?;
        stats.bytes += s.bytes;
        stats.seconds += s.seconds;
        stats.segments += s.segments;
        weights.merge(converted);
    }
    let mtp_experts = if mtp {
        let (head, sketch, bufs) = load_draft_head(ckpt, opts, stream)?;
        Some((load_mtp_experts(ckpt, opts, stream)?, head, sketch, bufs))
    } else {
        None
    };
    Ok((Model::bind(ckpt, weights, mtp_experts, opts.layers, stream)?, stats))
}
