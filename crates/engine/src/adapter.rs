//! Rank-1 weight adapters (a PEFT LoRA directory of rank 1: `adapter_config.json` and
//! `adapter_model.safetensors`) on the blocks' output projections (`linear_attn.out_proj`,
//! `self_attn.o_proj`). The forward pass adds `scale * b (a . x)` to a projection's output with a
//! per-sequence scale, so one loaded model serves the base and the adapted variant side by side.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use quench6k_cuda::{DeviceBuffer, Stream};

use crate::model::{Block, D, LAYERS, Model};

/// Input width of the output projections (GDN value heads, QSA heads).
pub const K: usize = 6144;

pub struct Adapter {
    /// Per layer: (a [K] F32, b [2560] F32 with lora_alpha / r folded in).
    pub layers: Vec<Option<(DeviceBuffer, DeviceBuffer)>>,
}

fn f16(bits: u16) -> f32 {
    let (sign, exp, mant) = (
        (bits >> 15) as u32,
        ((bits >> 10) & 0x1f) as i32,
        (bits & 0x3ff) as u32,
    );
    let v = match exp {
        0 => mant as f32 * (-24f32).exp2(),
        31 => {
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => f32::from_bits(((exp + 112) as u32) << 23 | mant << 13),
    };
    if sign == 1 { -v } else { v }
}

fn floats(dtype: &str, data: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        "F16" => data
            .chunks_exact(2)
            .map(|b| f16(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        "BF16" => data
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect(),
        "F32" => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        other => bail!("adapter tensor dtype {other}"),
    })
}

impl Adapter {
    pub fn load(dir: &Path, model: &Model, stream: &Stream) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("adapter_config.json"))
                .with_context(|| format!("reading {}/adapter_config.json", dir.display()))?,
        )?;
        let r = config["r"].as_u64().context("adapter_config.json: r")?;
        ensure!(r == 1, "only rank-1 adapters are supported (r = {r})");
        let scaling = config["lora_alpha"]
            .as_f64()
            .context("adapter_config.json: lora_alpha")? as f32
            / r as f32;
        let bytes = std::fs::read(dir.join("adapter_model.safetensors"))
            .with_context(|| format!("reading {}/adapter_model.safetensors", dir.display()))?;
        ensure!(bytes.len() >= 8, "adapter file too short");
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        ensure!(
            8 + header_len <= bytes.len(),
            "adapter header exceeds the file"
        );
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..8 + header_len])?;
        let data = &bytes[8 + header_len..];
        let tensor = |name: &str, want: [usize; 2]| -> Result<Vec<f32>> {
            let t = header
                .get(name)
                .with_context(|| format!("adapter has no {name}"))?;
            let shape: Vec<usize> = serde_json::from_value(t["shape"].clone())?;
            ensure!(shape == want, "{name}: shape {shape:?}, want {want:?}");
            let [a, b]: [usize; 2] = serde_json::from_value(t["data_offsets"].clone())?;
            ensure!(
                a <= b && b <= data.len(),
                "{name}: data range outside the file"
            );
            floats(t["dtype"].as_str().unwrap_or(""), &data[a..b])
        };
        let mut layers: Vec<Option<(DeviceBuffer, DeviceBuffer)>> =
            (0..LAYERS).map(|_| None).collect();
        let mut found = 0;
        for name in header.keys().filter(|n| n.ends_with(".lora_A.weight")) {
            let module = name.trim_end_matches(".lora_A.weight");
            let rest = module
                .split(".layers.")
                .nth(1)
                .with_context(|| format!("{name}: no layer index"))?;
            let (layer, proj) = rest
                .split_once('.')
                .with_context(|| format!("{name}: no module"))?;
            let layer: usize = layer
                .parse()
                .with_context(|| format!("{name}: layer index"))?;
            ensure!(
                layer < LAYERS && module.contains("model.language_model.layers."),
                "{name}: not a text-model layer"
            );
            match (proj, &model.layers[layer].block) {
                ("linear_attn.out_proj", Block::Gdn(_)) | ("self_attn.o_proj", Block::Qsa(_)) => {}
                _ => bail!(
                    "{name}: unsupported module (adapters cover linear_attn.out_proj and self_attn.o_proj)"
                ),
            }
            let a = tensor(name, [1, K])?;
            let b: Vec<f32> = tensor(&format!("{module}.lora_B.weight"), [D, 1])?
                .iter()
                .map(|v| v * scaling)
                .collect();
            let (da, db) = (DeviceBuffer::zeroed(K * 4)?, DeviceBuffer::zeroed(D * 4)?);
            stream.upload(&da, &a)?;
            stream.upload(&db, &b)?;
            ensure!(
                layers[layer].replace((da, db)).is_none(),
                "{name}: two adapters on layer {layer}"
            );
            found += 1;
        }
        ensure!(found > 0, "no lora_A tensors in {}", dir.display());
        stream.synchronize()?;
        Ok(Self { layers })
    }

    /// Layers the adapter changes.
    pub fn layer_indices(&self) -> Vec<usize> {
        (0..LAYERS).filter(|&l| self.layers[l].is_some()).collect()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn f16_values() {
        use super::f16;
        assert_eq!(f16(0x3c00), 1.0);
        assert_eq!(f16(0xc000), -2.0);
        assert_eq!(f16(0x0001), 2f32.powi(-24));
        assert_eq!(f16(0x7bff), 65504.0);
        assert!(f16(0x7c00).is_infinite());
    }
}
