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
/// `parts[i].1 ..` of a row of width `n`. With `fused` runs, each run `(weight, column)` holds several
/// members as contiguous rows of one weight (all of them, or those sharing a format), and one GEMV/GEMM
/// per run computes them; without, every member is computed on its own.
pub struct Proj {
    pub parts: Vec<(Linear, usize)>,
    pub fused: Vec<(Linear, usize)>,
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

/// How the host-resident PLE table stores its rows.
enum PleFormat {
    Bf16,
    /// FP8 E4M3 with a scale for the whole table (as in Qwen's FP8 release): each code's value times it.
    Fp8(Box<[f32; 256]>),
    /// NVFP4: 4-bit E2M1 codes (two per byte, low nibble first), an E4M3 scale per 16 values in a sibling
    /// `weight_scale` tensor per shard, and one FP32 `weight_scale_2` for the table.
    Nvfp4 { scale2: f32 },
}

/// The host-resident PLE table: 128 memory-mapped shards of `[rows, 160]` values (BF16, FP8 E4M3 or
/// NVFP4); quantized rows are dequantized to BF16 when gathered.
pub struct PleTable {
    /// Per shard: the row codes' mapping and offset, and for NVFP4 the block scales' mapping and offset.
    maps: Vec<(Mmap, usize, Option<(Mmap, usize)>)>,
    rows_per_shard: usize,
    format: PleFormat,
}

/// E2M1 values by code.
const FP4: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

impl PleTable {
    pub const WIDTH: usize = 160;

    fn open(ckpt: &Checkpoint) -> Result<Self> {
        let prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.";
        let mut shards: Vec<(usize, &str)> = ckpt
            .tensors
            .keys()
            .filter_map(|n| {
                n.strip_prefix(prefix)
                    .and_then(|r| r.strip_prefix("shard_"))
                    .and_then(|r| r.strip_suffix(".weight"))
                    .map(|i| (i.parse().unwrap(), n.as_str()))
            })
            .collect();
        shards.sort();
        ensure!(!shards.is_empty(), "checkpoint has no PLE table");
        let dtype = ckpt.get(shards[0].1)?.dtype;
        let read_scalar = |name: &str| -> Result<f32> {
            let info = ckpt.get(name)?;
            let b = ckpt.read(name)?;
            Ok(match (info.dtype, b.len()) {
                (DType::BF16, 2) => f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16),
                (DType::F32, 4) => f32::from_le_bytes(b[..4].try_into().unwrap()),
                _ => anyhow::bail!("{name}: expected one BF16 or F32 value"),
            })
        };
        let (format, width) = match dtype {
            DType::BF16 => (PleFormat::Bf16, Self::WIDTH),
            DType::F8E4M3 => {
                let scale = read_scalar(&format!("{prefix}weight_scale"))
                    .context("FP8 PLE table without a weight_scale")?;
                (
                    PleFormat::Fp8(Box::new(std::array::from_fn(|c| e4m3(c as u8) * scale))),
                    Self::WIDTH,
                )
            }
            DType::U8 => {
                let scale2 = read_scalar(&format!("{prefix}weight_scale_2"))
                    .context("NVFP4 PLE table without a weight_scale_2")?;
                (PleFormat::Nvfp4 { scale2 }, Self::WIDTH / 2)
            }
            other => anyhow::bail!("PLE table: {other:?} is not supported (BF16, F8_E4M3 or NVFP4)"),
        };
        let map = |info: &crate::checkpoint::TensorInfo| -> Result<(Mmap, usize)> {
            let file = &ckpt.files[info.file];
            let map = unsafe { Mmap::map(&file.file) }
                .with_context(|| format!("mapping {}", file.path.display()))?;
            // Rows are read at random: without this every fault also reads ahead ~128 KiB around a
            // 320-byte row, and cold prompts become bound by disk bandwidth.
            #[cfg(unix)]
            map.advise(memmap2::Advice::Random)
                .context("madvise(MADV_RANDOM) on the PLE table")?;
            Ok((map, info.offset as usize))
        };
        let mut maps = Vec::with_capacity(shards.len());
        let mut rows_per_shard = 0;
        for (i, (index, name)) in shards.iter().enumerate() {
            ensure!(*index == i, "PLE shard {i} missing");
            let info = ckpt.get(name)?;
            ensure!(
                info.dtype == dtype
                    && info.shape[1] == width
                    && (i == 0 || info.shape[0] == rows_per_shard),
                "{name}: unexpected PLE shard layout"
            );
            rows_per_shard = info.shape[0];
            let scales = match format {
                PleFormat::Nvfp4 { .. } => {
                    let sname = format!("{prefix}shard_{i}.weight_scale");
                    let s = ckpt.get(&sname)?;
                    ensure!(
                        s.dtype == DType::F8E4M3 && s.shape == [rows_per_shard, Self::WIDTH / 16],
                        "{sname}: unexpected NVFP4 scale layout"
                    );
                    Some(map(s)?)
                }
                _ => None,
            };
            let (m, off) = map(info)?;
            maps.push((m, off, scales));
        }
        Ok(Self {
            maps,
            rows_per_shard,
            format,
        })
    }

    /// Bytes per row of codes.
    fn row_bytes(&self) -> usize {
        match self.format {
            PleFormat::Bf16 => 2 * Self::WIDTH,
            PleFormat::Fp8(_) => Self::WIDTH,
            PleFormat::Nvfp4 { .. } => Self::WIDTH / 2,
        }
    }

    /// The row's codes and, for NVFP4, its block scales.
    fn row(&self, r: usize) -> (&[u8], Option<&[u8]>) {
        let (map, base, scales) = &self.maps[r / self.rows_per_shard];
        let i = r % self.rows_per_shard;
        let start = base + i * self.row_bytes();
        let codes = &map[start..start + self.row_bytes()];
        let scales = scales.as_ref().map(|(m, b)| {
            let s = b + i * (Self::WIDTH / 16);
            &m[s..s + Self::WIDTH / 16]
        });
        (codes, scales)
    }

    /// Asks the kernel to start reading row `r` from disk if it is not resident, so that a batch of cold
    /// rows is fetched in parallel instead of one page fault at a time.
    pub fn prefetch(&self, r: usize) {
        #[cfg(unix)]
        {
            let (map, base, scales) = &self.maps[r / self.rows_per_shard];
            let i = r % self.rows_per_shard;
            let _ = map.advise_range(
                memmap2::Advice::WillNeed,
                base + i * self.row_bytes(),
                self.row_bytes(),
            );
            if let Some((m, b)) = scales {
                let _ = m.advise_range(
                    memmap2::Advice::WillNeed,
                    b + i * (Self::WIDTH / 16),
                    Self::WIDTH / 16,
                );
            }
        }
        #[cfg(not(unix))]
        let _ = r;
    }

    /// Reads row `r`, faulting it into the page cache (blocking).
    pub fn touch(&self, r: usize) -> u8 {
        let (codes, scales) = self.row(r);
        // The row may straddle a page boundary.
        let mut x = unsafe {
            std::ptr::read_volatile(&codes[0]) ^ std::ptr::read_volatile(&codes[codes.len() - 1])
        };
        if let Some(s) = scales {
            x ^= unsafe { std::ptr::read_volatile(&s[0]) ^ std::ptr::read_volatile(&s[s.len() - 1]) };
        }
        x
    }

    /// Appends row `r` (160 BF16 values) to `out`.
    pub fn gather(&self, r: usize, out: &mut Vec<u16>) {
        let (bytes, scales) = self.row(r);
        match &self.format {
            PleFormat::Bf16 => out.extend(
                bytes
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]])),
            ),
            PleFormat::Fp8(lut) => out.extend(bytes.iter().map(|&c| bf16_rne(lut[c as usize]))),
            PleFormat::Nvfp4 { scale2 } => {
                let scales = scales.expect("NVFP4 PLE rows have scales");
                out.extend(nvfp4_row(bytes, scales, *scale2));
            }
        }
    }
}

/// One NVFP4 row of `2 * codes.len()` values as BF16: E2M1 value times (E4M3 block scale times `scale2`).
fn nvfp4_row<'a>(codes: &'a [u8], scales: &'a [u8], scale2: f32) -> impl Iterator<Item = u16> + 'a {
    codes.iter().enumerate().flat_map(move |(j, &c)| {
        let s = e4m3(scales[j / 8]) * scale2;
        [bf16_rne(FP4[(c & 15) as usize] * s), bf16_rne(FP4[(c >> 4) as usize] * s)]
    })
}

/// A projection weight that the checkpoint stores quantized, dequantized to BF16 on the host: MXFP8
/// (E4M3 codes with a sibling `weight_scale` of UE8M0 bytes per 32) or NVFP4 (packed E2M1 codes with a
/// sibling `weight_scale` of E4M3 bytes per 16 and an FP32 `weight_scale_2`). `None` for other tensors.
pub fn dequantize_bf16(ckpt: &Checkpoint, name: &str) -> Result<Option<(Vec<u16>, [usize; 2])>> {
    let Some(sname) = name
        .strip_suffix(".weight")
        .map(|b| format!("{b}.weight_scale"))
    else {
        return Ok(None);
    };
    let (Some(w), Some(sc)) = (ckpt.tensors.get(name), ckpt.tensors.get(&sname)) else {
        return Ok(None);
    };
    if w.shape.len() != 2 {
        return Ok(None);
    }
    if crate::loader::native_mxfp8(ckpt, name) {
        let (n, k) = (w.shape[0], w.shape[1]);
        let (codes, scales) = (ckpt.read(name)?, ckpt.read(&sname)?);
        let out = codes
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let s = f32::from_bits((scales[(i / k) * (k / 32) + (i % k) / 32] as u32) << 23);
                bf16_rne(e4m3(c) * s)
            })
            .collect();
        return Ok(Some((out, [n, k])));
    }
    let s2name = format!("{}_2", sname);
    if w.dtype == DType::U8 && sc.dtype == DType::F8E4M3 && sc.shape == [w.shape[0], w.shape[1] / 8] {
        let (n, k) = (w.shape[0], 2 * w.shape[1]);
        let s2 = ckpt.read(&s2name)?;
        ensure!(s2.len() == 4, "{s2name}: expected one F32 value");
        let scale2 = f32::from_le_bytes(s2[..4].try_into().unwrap());
        let (codes, scales) = (ckpt.read(name)?, ckpt.read(&sname)?);
        let mut out = Vec::with_capacity(n * k);
        for r in 0..n {
            out.extend(nvfp4_row(
                &codes[r * k / 2..(r + 1) * k / 2],
                &scales[r * k / 16..(r + 1) * k / 16],
                scale2,
            ));
        }
        return Ok(Some((out, [n, k])));
    }
    Ok(None)
}

/// FP8 E4M3 (fn: no infinities, 0x7f/0xff NaN) to f32.
fn e4m3(c: u8) -> f32 {
    let (s, e, m) = ((c >> 7) as u32, ((c >> 3) & 15) as i32, (c & 7) as f32);
    let v = match (e, c & 0x7f) {
        (_, 0x7f) => f32::NAN,
        (0, _) => m / 8.0 * 2f32.powi(-6),
        _ => (1.0 + m / 8.0) * 2f32.powi(e - 7),
    };
    if s == 1 { -v } else { v }
}

/// f32 to BF16, rounding to nearest even (the product of two BF16 values as PyTorch rounds it).
fn bf16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
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
    /// Tables over the NVFP4 experts of layer `prefix` as stored per expert in the checkpoint (codes,
    /// block scales and `weight_scale_2`), already on the device in `weights`.
    pub fn bind(ckpt: &Checkpoint, weights: &DeviceWeights, prefix: &str, stream: &Stream) -> Result<Self> {
        let mut addrs = [[0u64; EXPERTS]; 6];
        let mut scale2 = [[0f32; EXPERTS]; 3];
        for e in 0..EXPERTS {
            for (j, proj) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                let base = format!("{prefix}.mlp.experts.{e}.{proj}");
                addrs[2 * j][e] = weights.get(&format!("{base}.weight"))?.addr;
                addrs[2 * j + 1][e] = weights.get(&format!("{base}.weight_scale"))?.addr;
                let s2 = ckpt.read(&format!("{base}.weight_scale_2"))?;
                scale2[j][e] = f32::from_le_bytes(s2[..4].try_into()?);
            }
        }
        Self::upload(&addrs, &scale2, Vec::new(), stream)
    }

    /// `addrs`: [gate codes, gate scales, up codes, up scales, down codes, down scales][expert];
    /// `scale2`: [gate, up, down][expert].
    pub fn upload(
        addrs: &[[u64; EXPERTS]; 6],
        scale2: &[[f32; EXPERTS]; 3],
        mut buffers: Vec<DeviceBuffer>,
        stream: &Stream,
    ) -> Result<Self> {
        let flat_a: Vec<u64> = addrs.iter().flatten().copied().collect();
        let flat_s: Vec<f32> = scale2.iter().flatten().copied().collect();
        let tables = DeviceBuffer::new(flat_a.len() * 8 + flat_s.len() * 4)?;
        stream.upload_to(tables.addr(), &flat_a)?;
        stream.upload_to(tables.addr() + (flat_a.len() * 8) as u64, &flat_s)?;
        let t = |k: usize| tables.addr() + (k * EXPERTS * 8) as u64;
        let s2 = |j: usize| tables.addr() + (flat_a.len() * 8 + j * EXPERTS * 4) as u64;
        let (gate, up, down) = (
            ExpertProj {
                weights: t(0),
                scales: t(1),
                scale2: s2(0),
            },
            ExpertProj {
                weights: t(2),
                scales: t(3),
                scale2: s2(1),
            },
            ExpertProj {
                weights: t(4),
                scales: t(5),
                scale2: s2(2),
            },
        );
        buffers.push(tables);
        Ok(Self {
            gate,
            up,
            down,
            buffers,
        })
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
    /// Rank-1 output-projection adapter, applied with each sequence's adapter scale (see `adapter`).
    pub adapter: Option<crate::adapter::Adapter>,
    /// The DFlash block drafter, used for speculative rounds instead of the MTP layer when present.
    pub dflash: Option<crate::dflash::DFlash>,
}

fn read_i64s<const N: usize>(ckpt: &Checkpoint, name: &str) -> Result<[i64; N]> {
    let bytes = ckpt.read(name)?;
    ensure!(bytes.len() == 8 * N, "{name}: expected {N} I64 values");
    Ok(std::array::from_fn(|i| {
        i64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap())
    }))
}

impl Model {
    /// Elements of the largest INT8-group projection weight other than the LM head (which keeps its
    /// INT8 GEMM), 0 without any.
    pub fn max_int8g_elems(&self) -> usize {
        let head = self.lm_head.addr;
        self.weights
            .tensors
            .values()
            .filter(|t| t.dtype == DType::I8G && t.addr != head)
            .map(|t| t.shape.iter().product::<usize>())
            .max()
            .unwrap_or(0)
    }

    /// Binds the first `layers` text layers (all 48 for the real model).
    pub fn bind(
        ckpt: &Checkpoint,
        weights: DeviceWeights,
        mtp_experts: Option<(ExpertTables, (u64, u64, u64), u64, Vec<DeviceBuffer>)>,
        layers: usize,
        stream: &Stream,
    ) -> Result<Self> {
        ensure!(
            (2..=LAYERS).contains(&layers),
            "{layers} layers: between 2 (layer 1 holds the PLE) and {LAYERS}"
        );
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
            Ok(Linear {
                fmt,
                addr: t.addr,
                scale: t.scale,
                n: t.shape[0],
                k: t.shape[1],
            })
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
                vec![(lin(&group)?, 0)]
            } else if parts.len() == 1 {
                vec![(parts[0].0, 0)]
            } else {
                Vec::new()
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
            let [(route, _), (shared, _)] = moe_runs(p);
            let proj = if weights.tensors.contains_key(&route) {
                // Router and shared-expert gate logit in one run (columns 0..513), the shared expert's
                // gate and up projections in another (from column 640); member columns as listed.
                let x = |s: &str| lin(&format!("{m}.{s}.weight"));
                let (router, sg) = (x("gate")?, x("shared_expert_gate")?);
                let (gate, up) = (x("shared_expert.gate_proj")?, x("shared_expert.up_proj")?);
                let at = MOE_SHARED_RUN_COL;
                ensure!(router.n + sg.n <= at, "MoE router run overlaps the shared-expert run");
                Proj {
                    parts: vec![(router, 0), (gate, at), (up, at + gate.n), (sg, router.n)],
                    fused: vec![(lin(&route)?, 0), (lin(&shared)?, at)],
                    n: at + gate.n + up.n,
                }
            } else {
                proj(moe_group(p))?
            };
            Ok(Moe {
                proj,
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
            let tables = ExpertTables::bind(ckpt, &weights, &p, stream)?;

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
            adapter: None,
            dflash: None,
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
    (
        format!("{g}.quench6k.in_proj"),
        ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"]
            .iter()
            .map(|m| format!("{g}.{m}.weight"))
            .collect(),
    )
}

fn qsa_group(layer: &str) -> (String, Vec<String>) {
    let a = format!("{layer}.self_attn");
    (
        format!("{a}.quench6k.in_proj"),
        ["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"]
            .iter()
            .map(|m| format!("{a}.{m}.weight"))
            .collect(),
    )
}

fn moe_group(layer: &str) -> (String, Vec<String>) {
    let m = format!("{layer}.mlp");
    (
        format!("{m}.quench6k.in_proj"),
        [
            "gate",
            "shared_expert.gate_proj",
            "shared_expert.up_proj",
            "shared_expert_gate",
        ]
        .iter()
        .map(|x| format!("{m}.{x}.weight"))
        .collect(),
    )
}

/// Output column of the shared expert's gate/up run when the MoE input projection is computed as two runs.
pub const MOE_SHARED_RUN_COL: usize = 640;

/// The MoE input projection as two runs of one format each, for checkpoints that store the shared
/// expert's gate and up projections as MXFP8 but the router and shared-expert gate in BF16: (router,
/// shared-expert gate logit) and (shared gate, shared up).
pub fn moe_runs(layer: &str) -> [(String, Vec<String>); 2] {
    let m = format!("{layer}.mlp");
    let w = |x: &str| format!("{m}.{x}.weight");
    [
        (format!("{m}.quench6k.route"), vec![w("gate"), w("shared_expert_gate")]),
        (
            format!("{m}.quench6k.shared"),
            vec![w("shared_expert.gate_proj"), w("shared_expert.up_proj")],
        ),
    ]
}

/// Every multi-member projection group of the text model; quantized profiles store each contiguously.
pub fn projection_groups(mtp: bool, layers: usize) -> Vec<(String, Vec<String>)> {
    let mut groups = Vec::new();
    for i in 0..layers {
        let p = format!("model.language_model.layers.{i}");
        groups.push(hc_group(&format!("{p}.attn_hyper_connection"), true));
        groups.push(hc_group(&format!("{p}.mlp_hyper_connection"), true));
        groups.push(if is_attention(i) {
            qsa_group(&p)
        } else {
            gdn_group(&p)
        });
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
fn load_draft_head(
    ckpt: &Checkpoint,
    opts: crate::loader::LoadOptions,
    stream: &Stream,
) -> Result<((u64, u64, u64), u64, Vec<DeviceBuffer>)> {
    let (staged, _) = crate::loader::upload(ckpt, |t| t == "lm_head.weight", opts)?;
    let src = staged.get("lm_head.weight")?;
    let codes = DeviceBuffer::new(VOCAB * D / 2)?;
    let scales = DeviceBuffer::new(VOCAB * D / 16)?;
    let s2 = DeviceBuffer::new(4)?;
    let amax = DeviceBuffer::new(4)?;
    quench6k_cuda::kernels::quantize_nvfp4(
        stream,
        src.addr,
        1,
        VOCAB,
        D,
        codes.addr(),
        scales.addr(),
        s2.addr(),
        amax.addr(),
    )?;
    let sketch = DeviceBuffer::new(quench6k_cuda::kernels::draft_sketch_bytes(VOCAB))?;
    quench6k_cuda::kernels::build_draft_sketch(stream, src.addr, sketch.addr(), VOCAB)?;
    stream.synchronize()?;
    Ok((
        (codes.addr(), scales.addr(), s2.addr()),
        sketch.addr(),
        vec![codes, scales, s2, sketch],
    ))
}

/// Uploads the MTP layer's BF16 experts one tensor at a time and NVFP4-quantizes them into the layout
/// of the main model's experts (gate_up [512, 1280, 2560] holds gate rows 0..640 then up rows per expert).
fn load_mtp_experts(
    ckpt: &Checkpoint,
    opts: crate::loader::LoadOptions,
    stream: &Stream,
) -> Result<ExpertTables> {
    use quench6k_cuda::kernels as k;
    let mut addrs = [[0u64; EXPERTS]; 6];
    let mut scale2 = [[0f32; EXPERTS]; 3];
    let mut buffers = Vec::new();
    // (tensor, matrices per expert, rows, cols, first projection index)
    for (name, per_expert, n, kk, proj) in [
        (
            "mtp.layers.0.mlp.experts.gate_up_proj",
            2,
            EXPERT_WIDTH,
            D,
            0,
        ),
        ("mtp.layers.0.mlp.experts.down_proj", 1, D, EXPERT_WIDTH, 2),
    ] {
        let (staged, _) = crate::loader::upload(ckpt, |t| t == name, opts)?;
        let src = staged.get(name)?;
        let count = EXPERTS * per_expert;
        let codes = DeviceBuffer::new(count * n * kk / 2)?;
        let scales = DeviceBuffer::new(count * n * kk / 16)?;
        let s2 = DeviceBuffer::new(count * 4)?;
        let amax = DeviceBuffer::new(count * 4)?;
        k::quantize_nvfp4(
            stream,
            src.addr,
            count,
            n,
            kk,
            codes.addr(),
            scales.addr(),
            s2.addr(),
            amax.addr(),
        )?;
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
pub fn load_text(
    ckpt: &Checkpoint,
    precision: Precision,
    mtp: bool,
    opts: crate::loader::LoadOptions,
    stream: &Stream,
) -> Result<(Model, crate::loader::LoadStats)> {
    use crate::{loader, selection};
    let format = match precision {
        Precision::Bf16 => WeightFormat::Bf16,
        Precision::Int8Group => WeightFormat::Int8Group,
    };
    // A truncated model (opts.layers < LAYERS) leaves out the later layers' tensors.
    let in_layers = |n: &str| match n
        .strip_prefix("model.language_model.layers.")
        .and_then(|r| r.split('.').next())
        .and_then(|i| i.parse::<usize>().ok())
    {
        Some(i) => i < opts.layers,
        None => true,
    };
    // Checkpoints quantized for local inference (NVIDIA's FP4-Local) store the MTP experts as NVFP4 per
    // expert like the main layers' (instead of one BF16 tensor quantized at load), the attention, GDN and
    // shared-expert projections as MXFP8 (used as stored), and NVFP4 block scales in CSF form.
    let mtp_nvfp4 = ckpt
        .tensors
        .contains_key("mtp.layers.0.mlp.experts.0.gate_proj.weight");
    let native = |n: &str| loader::native_mxfp8(ckpt, n);
    let native_scale = |n: &str| {
        n.strip_suffix("_scale")
            .is_some_and(|w| selection::eight_bit(w) && native(w))
    };
    let in_model = |n: &str| {
        (selection::text(n) && in_layers(n))
            || (mtp && selection::mtp(n) && (mtp_nvfp4 || !n.contains(".experts.")))
    };
    let wanted = |n: &str| in_model(n) && !n.contains(".nvfp4_csf_") && !native_scale(n);
    let quantized = |n: &str| format != WeightFormat::Bf16 && selection::eight_bit(n);
    ensure!(
        format != WeightFormat::Bf16
            || !ckpt.tensors.keys().any(|n| selection::eight_bit(n) && native(n)),
        "the BF16 profile cannot load a checkpoint with MXFP8 projections"
    );
    let (mut weights, mut stats) = loader::upload(ckpt, |n| wanted(n) && !quantized(n), opts)?;
    // Norm weights stored in FP32 (NVIDIA's FP4-Local keeps the HyperConnection and PLE norms in FP32;
    // the kernels read BF16 norms): rounded to nearest-even BF16.
    let f32_norms: Vec<String> = weights
        .tensors
        .iter()
        .filter(|(n, t)| {
            t.dtype == DType::F32
                && t.shape.len() == 1
                && n.strip_suffix(".weight")
                    .and_then(|m| m.rsplit('.').next())
                    .is_some_and(|m| m.contains("norm"))
        })
        .map(|(n, _)| n.clone())
        .collect();
    for name in f32_norms {
        let raw = ckpt.read(&name)?;
        let bf: Vec<u16> = raw
            .chunks_exact(4)
            .map(|b| bf16_rne(f32::from_le_bytes(b.try_into().unwrap())))
            .collect();
        let buf = DeviceBuffer::new(bf.len() * 2)?;
        stream.upload(&buf, &bf)?;
        let t = weights.tensors.get_mut(&name).unwrap();
        t.addr = buf.addr();
        t.bytes = (bf.len() * 2) as u64;
        t.dtype = DType::BF16;
        weights.segments.push(buf);
    }
    stream.synchronize()?;
    let (scales, s) = loader::upload_csf_scales(ckpt, |n| in_model(n), 256 << 20, stream)?;
    stats.bytes += s.bytes;
    stats.seconds += s.seconds;
    stats.segments += s.segments;
    weights.merge(scales);
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
                "shared" => {
                    n.contains(".shared_expert.") && !n.ends_with("shared_expert_gate.weight")
                }
                "hc" => n.contains("hyper_connection."),
                "router" => n.ends_with(".mlp.gate.weight"),
                _ => false,
            }) && (set.split(',').any(|c| c == "mtp") || !n.starts_with("mtp."))
        })
    };
    let fold = |name: &str, addr: u64, n: usize, k: usize| -> Result<()> {
        let prefix = name
            .strip_suffix(".input_mix_weight_down.weight")
            .or_else(|| name.strip_suffix(".block_inject_weight.weight"));
        if let Some(prefix) = prefix {
            quench6k_cuda::kernels::fold_norm(
                stream,
                addr,
                weights.get(&format!("{prefix}.hc_norm.weight"))?.addr,
                n,
                k,
            )?;
        }
        if fp4_class(name) {
            // QUENCH6K_FAKE_BITS=b (2..8): b-bit integers with a scale per 64 instead of NVFP4.
            match std::env::var("QUENCH6K_FAKE_BITS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
            {
                Some(bits) => quench6k_cuda::kernels::fake_int(stream, addr, n, k, 64, bits)?,
                None => quench6k_cuda::kernels::fake_fp4_act(stream, addr, n, k)?,
            }
        }
        Ok(())
    };
    if format == WeightFormat::Bf16 {
        let names: Vec<String> = weights
            .tensors
            .keys()
            .filter(|n| {
                n.ends_with(".input_mix_weight_down.weight")
                    || n.ends_with(".block_inject_weight.weight")
            })
            .cloned()
            .collect();
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
            let names: Vec<String> = weights
                .tensors
                .keys()
                .filter(|n| selection::eight_bit(n) && in_set(n))
                .cloned()
                .collect();
            for name in names {
                let t = weights.get(&name)?;
                if t.shape.len() == 2 && t.shape[1] % 32 == 0 {
                    quench6k_cuda::kernels::fake_mxfp8(
                        stream, t.addr, t.shape[0], t.shape[1], t.shape[1],
                    )?;
                }
            }
        }
        stream.synchronize()?;
    } else {
        // A group fuses into one weight only if its members share a format: the MoE input group of an
        // MXFP8 checkpoint (BF16 router and shared-expert gate, MXFP8 shared gate/up) stays unfused.
        // Such a group is computed as runs of one format each where a layout exists (the MoE input).
        let mut groups = Vec::new();
        for (name, members) in projection_groups(mtp, opts.layers) {
            let k = members.iter().filter(|x| native(x)).count();
            if k == 0 || k == members.len() {
                groups.push((name, members));
            } else if let Some(layer) = name.strip_suffix(".mlp.quench6k.in_proj") {
                let runs = moe_runs(layer);
                if runs.iter().all(|(_, m)| {
                    let k = m.iter().filter(|x| native(x)).count();
                    k == 0 || k == m.len()
                }) {
                    groups.extend(runs);
                }
            }
        }
        let (converted, s) = loader::upload_quantized(
            ckpt,
            |n| wanted(n) && quantized(n),
            format,
            &groups,
            &fold,
            opts,
            stream,
        )?;
        stats.bytes += s.bytes;
        stats.seconds += s.seconds;
        stats.segments += s.segments;
        weights.merge(converted);
    }
    let mtp_experts = if mtp {
        let (head, sketch, bufs) = load_draft_head(ckpt, opts, stream)?;
        let tables = if mtp_nvfp4 {
            ExpertTables::bind(ckpt, &weights, "mtp.layers.0", stream)?
        } else {
            load_mtp_experts(ckpt, opts, stream)?
        };
        Some((tables, head, sketch, bufs))
    } else {
        None
    };
    Ok((
        Model::bind(ckpt, weights, mtp_experts, opts.layers, stream)?,
        stats,
    ))
}

#[cfg(test)]
mod tests {
    use super::{bf16_rne, e4m3, nvfp4_row};

    #[test]
    fn nvfp4_rows() {
        // Codes (1, 7) and (9, 15) under block scale 2.0 (0x40), then (2, 0) under 0.5 (0x30); scale2 0.25.
        let mut codes = vec![0x71, 0xf9];
        codes.extend(std::iter::repeat_n(0, 6));
        codes.push(0x02);
        codes.extend(std::iter::repeat_n(0, 7));
        let row: Vec<f32> = nvfp4_row(&codes, &[0x40, 0x30], 0.25)
            .map(|b| f32::from_bits((b as u32) << 16))
            .collect();
        assert_eq!(row.len(), 32);
        assert_eq!(&row[..4], &[0.25, 3.0, -0.25, -3.0]);
        assert_eq!(&row[16..18], &[0.125, 0.0]);
    }

    #[test]
    fn e4m3_values() {
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0x7e), 448.0);
        assert_eq!(e4m3(0xfe), -448.0);
        assert_eq!(e4m3(0x01), 2f32.powi(-9));
        assert_eq!(e4m3(0x08), 2f32.powi(-6));
        assert_eq!(e4m3(0x3c), 1.5);
        assert!(e4m3(0x7f).is_nan() && e4m3(0xff).is_nan());
        // Monotone over the positive codes.
        assert!((1..0x7f).all(|c| e4m3(c) > e4m3(c - 1)));
    }

    #[test]
    fn bf16_rounding() {
        assert_eq!(bf16_rne(1.0), 0x3f80);
        assert_eq!(bf16_rne(f32::from_bits(0x3f80_8000)), 0x3f80); // tie, even stays
        assert_eq!(bf16_rne(f32::from_bits(0x3f81_8000)), 0x3f82); // tie, odd rounds up
        assert_eq!(bf16_rne(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(bf16_rne(-2.5), 0xc020);
    }
}
