//! Logit attribution: why the model gives a token its score. The residual a step ends with is a sum
//! of parts: the token embedding (in all four streams), the PLE injection, and each block's and MoE's
//! output times its injection weights per stream (SPEC §3). The final mixer is linear in the residual
//! once its gates and the streams' RMS factors are held at the values this step computed, so the
//! target token's logit splits exactly into one term per part, per routed expert and per neuron (an
//! expert's intermediate unit). These are direct effects: a part's contribution through later layers
//! is credited to those layers.

use anyhow::{Result, bail, ensure};
use quench6k_cuda::{DeviceBuffer, Stream, kernels as k};
use serde::Serialize;

use crate::forward::{CaptureLayout as L, HeadWorkspace, head_logits_rows};
use crate::model::{D, EXPERT_WIDTH, H, LAYERS, Linear, Model, S, TOP_K, VOCAB, is_attention};

/// What a step left for inspection (see `Session::inspect`).
pub struct Raw {
    /// Capture records of the inspected token, [`CaptureLayout::BYTES`](L::BYTES) per layer.
    pub records: Vec<u8>,
    /// The residual after the last layer, F32 [4, 2560].
    pub h: Vec<f32>,
    /// The final mixer's gate pre-activations, F32 [4, 2560].
    pub u: Vec<f32>,
    /// The mixer output (BF16 bits [2560]) and the logits it gave.
    pub out: Vec<u16>,
    pub logits: Vec<f32>,
}

#[derive(Serialize, Clone, Copy)]
pub struct TokenScore {
    pub id: i32,
    pub value: f32,
}

#[derive(Serialize)]
pub struct Neuron {
    pub layer: usize,
    /// Routed expert, or None for the shared expert.
    pub expert: Option<usize>,
    pub unit: usize,
    /// The unit's activation (silu(gate) times up) and its contribution to the target logit.
    pub activation: f32,
    pub contribution: f32,
}

#[derive(Serialize)]
pub struct Expert {
    pub expert: usize,
    /// Routing weight (renormalized over the 10 chosen).
    pub weight: f32,
    pub contribution: f32,
    /// Tokens the expert's output pushes up and down most (direct effect), for the top experts only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pushes: Option<Pushes>,
}

#[derive(Serialize, Clone)]
pub struct Pushes {
    pub up: Vec<TokenScore>,
    pub down: Vec<TokenScore>,
}

#[derive(Serialize)]
pub struct Layer {
    pub layer: usize,
    /// "gdn" or "attention".
    pub mixer: &'static str,
    pub mixer_contribution: f32,
    pub moe_contribution: f32,
    pub shared_contribution: f32,
    /// The block output's injection weights per stream, then the MoE output's.
    pub inject_mixer: [f32; S],
    pub inject_moe: [f32; S],
    pub experts: Vec<Expert>,
    /// Experts the router ranked next after the chosen ten (probability before renormalization).
    pub runners_up: Vec<TokenScore>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mixer_pushes: Option<Pushes>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moe_pushes: Option<Pushes>,
}

#[derive(Serialize)]
pub struct Report {
    pub target: i32,
    /// The target's logit and log-probability as the model computed them.
    pub logit: f32,
    pub logprob: f32,
    /// The model's most likely next tokens (log-probabilities).
    pub top: Vec<TokenScore>,
    /// The sum of all contributions, and the logit of the linearized mixer output (they agree up to
    /// summation order); the mixer output rounds to BF16 before the LM head, hence a small gap to `logit`.
    pub sum: f32,
    pub linear_logit: f32,
    /// Largest difference between the residual rebuilt from its parts and the step's own.
    pub residual_error: f32,
    /// Largest difference between the linearized mixer output and the step's (BF16) output.
    pub mixer_error: f32,
    pub embedding: f32,
    pub ple: f32,
    pub layers: Vec<Layer>,
    /// Neurons with the largest contributions (either sign), routed and shared.
    pub neurons: Vec<Neuron>,
    /// How much of the routed experts' contributions their neurons account for (1: exact).
    pub neuron_coverage: f32,
    /// The same for the shared experts (their activations are recomputed from the gate and up
    /// projections, which the decoding kernel rounds differently).
    pub shared_coverage: f32,
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn i32s(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| bf16(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn e4m3(v: u8) -> f32 {
    let (sign, exp, mant) = ((v >> 7) & 1, ((v >> 3) & 0xf) as i32, (v & 7) as f32);
    let mag = if exp == 0 {
        mant / 8.0 * (-6f32).exp2()
    } else if exp == 15 && mant == 7.0 {
        f32::NAN
    } else {
        (1.0 + mant / 8.0) * ((exp - 7) as f32).exp2()
    };
    if sign == 1 { -mag } else { mag }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum::<f64>() as f32
}

/// Rows `rows` of a projection weight as F32 [rows.len(), k].
fn weight_rows(s: &Stream, w: &Linear, rows: std::ops::Range<usize>) -> Result<Vec<f32>> {
    let (n, kk) = (rows.len(), w.k);
    Ok(match w.fmt {
        k::WeightFormat::Bf16 => {
            let raw: Vec<u16> = s.download(w.addr + (rows.start * kk * 2) as u64, n * kk)?;
            raw.into_iter().map(bf16).collect()
        }
        k::WeightFormat::Int8Group => {
            let g = k::int8_group();
            let codes: Vec<i8> = s.download(w.addr + (rows.start * kk) as u64, n * kk)?;
            let scales: Vec<f32> =
                s.download(w.scale + (rows.start * kk / g * 4) as u64, n * kk / g)?;
            codes
                .iter()
                .enumerate()
                .map(|(i, &c)| c as f32 * scales[i / g])
                .collect()
        }
        k::WeightFormat::Mxfp8 => {
            let codes: Vec<u8> = s.download(w.addr + (rows.start * kk) as u64, n * kk)?;
            let scales: Vec<u8> =
                s.download(w.scale + (rows.start * kk / 32) as u64, n * kk / 32)?;
            codes
                .iter()
                .enumerate()
                .map(|(i, &c)| e4m3(c) * f32::from_bits((scales[i / 32] as u32) << 23))
                .collect()
        }
    })
}

/// Tokens a direction in the mixer output space pushes up and down most (LM head over the rows).
fn pushes(
    model: &Model,
    s: &Stream,
    hw: &HeadWorkspace,
    vectors: &[Vec<f32>],
) -> Result<Vec<Pushes>> {
    const N: usize = 8;
    let rows = hw.rows();
    let buf = DeviceBuffer::new(rows * D * 2)?;
    let mut out = Vec::new();
    for chunk in vectors.chunks(rows) {
        let bits: Vec<u16> = chunk
            .iter()
            .flat_map(|v| v.iter().map(|x| (x.to_bits() >> 16) as u16))
            .collect();
        s.upload_to(buf.addr(), &bits)?;
        head_logits_rows(model, hw, s, buf.addr(), chunk.len())?;
        let logits: Vec<f32> = s.download(hw.logits_addr(), chunk.len() * VOCAB)?;
        for row in logits.chunks_exact(VOCAB) {
            let mut idx: Vec<u32> = (0..VOCAB as u32).collect();
            let by = |a: &u32, b: &u32| row[*b as usize].total_cmp(&row[*a as usize]);
            idx.select_nth_unstable_by(N, by);
            let mut up: Vec<u32> = idx[..N].to_vec();
            up.sort_by(by);
            idx.select_nth_unstable_by(VOCAB - N, by);
            let mut down: Vec<u32> = idx[VOCAB - N..].to_vec();
            down.sort_by(|a, b| by(b, a));
            let score = |i: &u32| TokenScore {
                id: *i as i32,
                value: row[*i as usize],
            };
            out.push(Pushes {
                up: up.iter().map(score).collect(),
                down: down.iter().map(score).collect(),
            });
        }
    }
    Ok(out)
}

/// Splits the logit of `target` (default: the most likely token) into its parts.
pub fn analyze(
    model: &Model,
    s: &Stream,
    hw: &HeadWorkspace,
    raw: &Raw,
    target: Option<i32>,
) -> Result<Report> {
    ensure!(raw.records.len() == LAYERS * L::BYTES, "capture records");
    let lse = {
        let m = raw.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        m + raw
            .logits
            .iter()
            .map(|x| ((x - m) as f64).exp())
            .sum::<f64>()
            .ln() as f32
    };
    let mut order: Vec<usize> = (0..VOCAB).collect();
    order.select_nth_unstable_by(20, |a, b| raw.logits[*b].total_cmp(&raw.logits[*a]));
    order.truncate(20);
    order.sort_by(|a, b| raw.logits[*b].total_cmp(&raw.logits[*a]));
    let target = target.unwrap_or(order[0] as i32);
    if !(0..VOCAB as i32).contains(&target) {
        bail!("token {target} is outside the vocabulary");
    }

    // The mixer output is sum over streams of c[s, d] h[s, d]; ρ folds in the target's LM-head row.
    let norm: Vec<u16> = s.download(model.final_hc.norm, H)?;
    let lm = weight_rows(s, &model.lm_head, target as usize..target as usize + 1)?;
    let mut c = vec![0f32; H];
    for st in 0..S {
        let x = &raw.h[st * D..(st + 1) * D];
        let r = 1.0 / ((x.iter().map(|v| v * v).sum::<f32>() / D as f32) + 1e-6).sqrt();
        for d in 0..D {
            let i = st * D + d;
            c[i] = sigmoid(raw.u[i]) * (1.0 + bf16(norm[i])) * r / S as f32;
        }
    }
    let rho: Vec<f32> = (0..H).map(|i| c[i] * lm[i % D]).collect();
    // Readout of a part written as y ⊗ inj: R = sum over streams of inj[s] ρ[s]; its mixer-output
    // direction y ⊙ (sum of inj[s] c[s]).
    let readout = |inj: &[f32], w: &[f32]| -> Vec<f32> {
        (0..D)
            .map(|d| (0..S).map(|st| inj[st] * w[st * D + d]).sum())
            .collect()
    };
    let mixer_out = |y: &[f32], inj: &[f32]| -> Vec<f32> {
        let g = readout(inj, &c);
        y.iter().zip(&g).map(|(a, b)| a * b).collect()
    };
    let linear_logit = dot(&rho, &raw.h);
    let mixer_error = (0..D)
        .map(|d| {
            let v: f32 = (0..S).map(|st| c[st * D + d] * raw.h[st * D + d]).sum();
            (v - bf16(raw.out[d])).abs()
        })
        .fold(0f32, f32::max);

    let rec = |l: usize, off: usize, len: usize| {
        &raw.records[l * L::BYTES + off..l * L::BYTES + off + len]
    };
    let premix0 = f32s(rec(0, L::PREMIX, H * 4));
    let embedding = dot(&rho, &premix0);
    let mut rebuilt = premix0.clone();

    struct Part {
        y: Vec<f32>,
        inj: Vec<f32>,
    }
    let mut mixers = Vec::new();
    let mut moes = Vec::new();
    let mut layers = Vec::new();
    // Routed (layer, slot, expert, readout) pairs for the per-neuron readouts on the device.
    let mut pairs: Vec<(usize, usize, i32)> = Vec::new();
    let mut readouts: Vec<f32> = Vec::new();
    let mut ple = 0.0;
    for l in 0..LAYERS {
        if l == 1 {
            // PREMIX of layer 1 includes the PLE injection; the rest of it is rebuilt.
            let premix1 = f32s(rec(1, L::PREMIX, H * 4));
            let p: Vec<f32> = premix1.iter().zip(&rebuilt).map(|(a, b)| a - b).collect();
            ple = dot(&rho, &p);
            rebuilt = premix1;
        }
        let mixer = Part {
            y: f32s(rec(l, L::BLOCK, D * 4)),
            inj: f32s(rec(l, L::INJ_ATTN, S * 4)),
        };
        let moe = Part {
            y: f32s(rec(l, L::MOE, D * 4)),
            inj: f32s(rec(l, L::INJ_MLP, S * 4)),
        };
        for part in [&mixer, &moe] {
            for st in 0..S {
                for d in 0..D {
                    rebuilt[st * D + d] += part.y[d] * part.inj[st];
                }
            }
        }
        let r_mix = readout(&mixer.inj, &rho);
        let r_moe = readout(&moe.inj, &rho);
        let top_i = i32s(rec(l, L::TOP_I, TOP_K * 4));
        let top_w = f32s(rec(l, L::TOP_W, TOP_K * 4));
        let yd = f32s(rec(l, L::YD, TOP_K * D * 4));
        let mut experts = Vec::new();
        let mut routed = 0.0;
        for slot in 0..TOP_K {
            let v = top_w[slot] * dot(&yd[slot * D..(slot + 1) * D], &r_moe);
            routed += v;
            experts.push(Expert {
                expert: top_i[slot] as usize,
                weight: top_w[slot],
                contribution: v,
                pushes: None,
            });
            pairs.push((l, slot, top_i[slot]));
            readouts.extend_from_slice(&r_moe);
        }
        let moe_contribution = dot(&moe.y, &r_moe);
        let router = f32s(rec(l, L::ROUTER, crate::model::EXPERTS * 4));
        let m = router.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = router.iter().map(|x| (x - m).exp()).sum();
        let mut rank: Vec<usize> = (0..router.len()).collect();
        rank.sort_by(|a, b| router[*b].total_cmp(&router[*a]));
        let runners_up = rank
            .iter()
            .filter(|e| !top_i.contains(&(**e as i32)))
            .take(5)
            .map(|&e| TokenScore {
                id: e as i32,
                value: (router[e] - m).exp() / z,
            })
            .collect();
        layers.push(Layer {
            layer: l,
            mixer: if is_attention(l) { "attention" } else { "gdn" },
            mixer_contribution: dot(&mixer.y, &r_mix),
            moe_contribution,
            shared_contribution: moe_contribution - routed,
            inject_mixer: mixer.inj.clone().try_into().unwrap(),
            inject_moe: moe.inj.clone().try_into().unwrap(),
            experts,
            runners_up,
            mixer_pushes: None,
            moe_pushes: None,
        });
        mixers.push(mixer);
        moes.push((moe, r_moe));
    }
    let residual_error = rebuilt
        .iter()
        .zip(&raw.h)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    // Routed neurons: activation times weight times the unit's down-projection column read out.
    let n = pairs.len();
    let experts_dev = DeviceBuffer::new(n * 4)?;
    let r_dev = DeviceBuffer::new(n * D * 4)?;
    let out_dev = DeviceBuffer::new(n * EXPERT_WIDTH * 4)?;
    let ids: Vec<i32> = pairs.iter().map(|p| p.2).collect();
    s.upload_to(experts_dev.addr(), &ids)?;
    s.upload_to(r_dev.addr(), &readouts)?;
    let mut neurons = Vec::new();
    let (mut covered, mut routed_total) = (0f64, 0f64);
    for l in 0..LAYERS {
        let down = &model.layers[l].moe.down;
        let at = l * TOP_K;
        k::nvfp4_column_dots(
            s,
            (down.weights, down.scales, down.scale2),
            experts_dev.addr() + (at * 4) as u64,
            TOP_K,
            r_dev.addr() + (at * D * 4) as u64,
            out_dev.addr() + (at * EXPERT_WIDTH * 4) as u64,
        )?;
    }
    let cols: Vec<f32> = s.download(out_dev.addr(), n * EXPERT_WIDTH)?;
    for (p, &(l, slot, e)) in pairs.iter().enumerate() {
        let act = bf16s(rec(l, L::ACT + slot * EXPERT_WIDTH * 2, EXPERT_WIDTH * 2));
        let w = layers[l].experts[slot].weight;
        let mut sum = 0f64;
        for j in 0..EXPERT_WIDTH {
            let v = w * act[j] * cols[p * EXPERT_WIDTH + j];
            sum += v as f64;
            neurons.push(Neuron {
                layer: l,
                expert: Some(e as usize),
                unit: j,
                activation: act[j],
                contribution: v,
            });
        }
        covered += sum;
        routed_total += layers[l].experts[slot].contribution as f64;
    }
    // Shared expert neurons: sigmoid(gate logit) silu(gate) up, read out through its down projection.
    let (mut shared_sum, mut shared_total) = (0f64, 0f64);
    for l in 0..LAYERS {
        shared_total += layers[l].shared_contribution as f64;
        let sh = f32s(rec(l, L::SHARED, (2 * EXPERT_WIDTH + 1) * 4));
        let gate = sigmoid(sh[2 * EXPERT_WIDTH]);
        let w = weight_rows(s, &model.layers[l].moe.shared_down, 0..D)?;
        let r = &moes[l].1;
        for j in 0..EXPERT_WIDTH {
            let (g, u) = (sh[j], sh[EXPERT_WIDTH + j]);
            let act = g * sigmoid(g) * u;
            let col: f32 = (0..D).map(|d| w[d * EXPERT_WIDTH + j] * r[d]).sum();
            shared_sum += (gate * act * col) as f64;
            neurons.push(Neuron {
                layer: l,
                expert: None,
                unit: j,
                activation: act,
                contribution: gate * act * col,
            });
        }
    }
    neurons.sort_by(|a, b| b.contribution.abs().total_cmp(&a.contribution.abs()));
    neurons.truncate(200);

    // What the largest parts push toward: blocks, MoE outputs and experts by |contribution|.
    let mut parts: Vec<(f32, usize, u8, usize)> = Vec::new();
    for (l, layer) in layers.iter().enumerate() {
        parts.push((layer.mixer_contribution.abs(), l, 0, 0));
        parts.push((layer.moe_contribution.abs(), l, 1, 0));
        for (slot, e) in layer.experts.iter().enumerate() {
            parts.push((e.contribution.abs(), l, 2, slot));
        }
    }
    parts.sort_by(|a, b| b.0.total_cmp(&a.0));
    parts.truncate(24);
    let vectors: Vec<Vec<f32>> = parts
        .iter()
        .map(|&(_, l, kind, slot)| match kind {
            0 => mixer_out(&mixers[l].y, &mixers[l].inj),
            1 => mixer_out(&moes[l].0.y, &moes[l].0.inj),
            _ => {
                let yd = f32s(rec(l, L::YD + slot * D * 4, D * 4));
                let w = layers[l].experts[slot].weight;
                let y: Vec<f32> = yd.iter().map(|v| v * w).collect();
                mixer_out(&y, &moes[l].0.inj)
            }
        })
        .collect();
    for ((_, l, kind, slot), p) in parts.iter().zip(pushes(model, s, hw, &vectors)?) {
        match kind {
            0 => layers[*l].mixer_pushes = Some(p),
            1 => layers[*l].moe_pushes = Some(p),
            _ => layers[*l].experts[*slot].pushes = Some(p),
        }
    }

    let sum = embedding
        + ple
        + layers
            .iter()
            .map(|l| l.mixer_contribution + l.moe_contribution)
            .sum::<f32>();
    Ok(Report {
        target,
        logit: raw.logits[target as usize],
        logprob: raw.logits[target as usize] - lse,
        top: order
            .iter()
            .map(|&i| TokenScore {
                id: i as i32,
                value: raw.logits[i] - lse,
            })
            .collect(),
        sum,
        linear_logit,
        residual_error,
        mixer_error,
        embedding,
        ple,
        layers,
        neurons,
        neuron_coverage: (covered / routed_total) as f32,
        shared_coverage: (shared_sum / shared_total) as f32,
    })
}

/// What neurons write, independent of context: each neuron's down-projection column through the LM
/// head (a logit lens), scaled per feature by the final mixer's mean norm weight (its gates and the
/// streams' RMS factors depend on the context and are left out).
pub fn lens(
    model: &Model,
    s: &Stream,
    hw: &HeadWorkspace,
    neurons: &[crate::atlas::NeuronId],
) -> Result<Vec<Pushes>> {
    let norm: Vec<u16> = s.download(model.final_hc.norm, H)?;
    let weight: Vec<f32> = (0..D)
        .map(|d| (0..S).map(|st| 1.0 + bf16(norm[st * D + d])).sum::<f32>() / S as f32)
        .collect();
    let mut vectors = Vec::new();
    for &(layer, expert, unit) in neurons {
        ensure!(
            layer < LAYERS && unit < EXPERT_WIDTH,
            "no neuron L{layer}.N{unit}"
        );
        let col: Vec<f32> = match expert {
            Some(e) => {
                ensure!(e < crate::model::EXPERTS, "no expert {e}");
                let ids = DeviceBuffer::new(8)?;
                s.upload_to(ids.addr(), &[e as i32, unit as i32])?;
                let out = DeviceBuffer::new(D * 4)?;
                let down = &model.layers[layer].moe.down;
                k::nvfp4_columns(
                    s,
                    (down.weights, down.scales, down.scale2),
                    ids.addr(),
                    ids.addr() + 4,
                    1,
                    out.addr(),
                )?;
                s.download(out.addr(), D)?
            }
            None => {
                let w = weight_rows(s, &model.layers[layer].moe.shared_down, 0..D)?;
                (0..D).map(|d| w[d * EXPERT_WIDTH + unit]).collect()
            }
        };
        // Unit length (BF16 range), then the mixer's per-feature weight.
        let len = col.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        vectors.push(col.iter().zip(&weight).map(|(v, w)| v / len * w).collect());
    }
    pushes(model, s, hw, &vectors)
}
