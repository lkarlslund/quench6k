//! Direction-removal steering packs (`ninfer.steering/1`, safetensors): per steered layer L a tensor
//! `layers.{L}.directions` [k, 4, 2560] F32 of mutually orthogonal directions, one 2560 vector per
//! residual lane, and JSON metadata under `__metadata__["ninfer.steering"]` (lane mask per layer,
//! norm preservation, default strength, provenance). The forward pass applies the pack at every
//! layer's attention-input pre-mix point (MTP excluded) with a per-sequence strength.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use quench6k_cuda::{DeviceBuffer, Stream};

use crate::model::{D, LAYERS, S};

pub struct Steering {
    /// [LAYERS][MAX_RANK][4][2560] F32, zero where unused.
    pub directions: DeviceBuffer,
    /// [LAYERS] i32: directions per layer (0: not steered).
    pub ranks: DeviceBuffer,
    /// [LAYERS] i32: bit s set when lane s is steered.
    pub masks: DeviceBuffer,
    pub norm_preserve: bool,
    pub default_strength: f32,
    /// (layer, rank, lane mask) of every steered layer.
    pub layers: Vec<(usize, usize, u32)>,
    pub model_sha: Option<String>,
}

impl Steering {
    pub fn load(path: &Path, stream: &Stream) -> Result<Self> {
        let max_rank = quench6k_cuda::kernels::steer_max_rank();
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading steering pack {}", path.display()))?;
        ensure!(bytes.len() >= 8, "steering pack too short");
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        ensure!(
            8 + header_len <= bytes.len(),
            "steering pack header exceeds the file"
        );
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[8..8 + header_len]).context("steering pack header")?;
        let data = &bytes[8 + header_len..];
        let meta_text = header["__metadata__"]["ninfer.steering"]
            .as_str()
            .context("pack metadata ninfer.steering")?;
        let meta: serde_json::Value =
            serde_json::from_str(meta_text).context("pack metadata JSON")?;
        let schema = meta["schema"].as_str().unwrap_or("");
        ensure!(
            schema == "ninfer.steering/1",
            "unsupported steering pack schema {schema:?}"
        );
        let pack_max = meta["max_rank"].as_u64().unwrap_or(0) as usize;
        ensure!(
            pack_max <= max_rank,
            "pack rank {pack_max} exceeds the supported maximum {max_rank}"
        );
        let norm_preserve = meta["norm_preserve"].as_bool().unwrap_or(false);
        let default_strength = meta["default_steering_strength"].as_f64().unwrap_or(0.0) as f32;
        let layer_meta: HashMap<String, serde_json::Value> = meta["layers"]
            .as_object()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        let mut host = vec![0f32; LAYERS * max_rank * S * D];
        let mut ranks = vec![0i32; LAYERS];
        let mut masks = vec![0i32; LAYERS];
        let mut layers = Vec::new();
        for (name, info) in header.as_object().context("pack header is not an object")? {
            if name == "__metadata__" {
                continue;
            }
            let Some(layer) = name
                .strip_prefix("layers.")
                .and_then(|r| r.strip_suffix(".directions"))
                .and_then(|l| l.parse::<usize>().ok())
            else {
                bail!("unexpected tensor {name:?} in steering pack");
            };
            ensure!(layer < LAYERS, "{name}: layer {layer} out of range");
            ensure!(info["dtype"] == "F32", "{name}: dtype must be F32");
            let shape: Vec<usize> = info["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            ensure!(
                shape.len() == 3 && shape[1] == S && shape[2] == D,
                "{name}: shape {shape:?}, expected [k, 4, 2560]"
            );
            let k = shape[0];
            ensure!(
                (1..=max_rank).contains(&k),
                "{name}: rank {k} out of 1..={max_rank}"
            );
            let off = info["data_offsets"].as_array().context("data_offsets")?;
            let (a, b) = (
                off[0].as_u64().unwrap_or(0) as usize,
                off[1].as_u64().unwrap_or(0) as usize,
            );
            ensure!(
                b - a == k * S * D * 4 && b <= data.len(),
                "{name}: data range"
            );
            let v: Vec<f32> = data[a..b]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            // Directions of one layer must be mutually orthogonal (per lane), or multi-rank removal is
            // order-dependent.
            for s in 0..S {
                for i in 0..k {
                    for j in 0..i {
                        let (di, dj) = (&v[(i * S + s) * D..][..D], &v[(j * S + s) * D..][..D]);
                        let dot: f64 = di.iter().zip(dj).map(|(x, y)| *x as f64 * *y as f64).sum();
                        let ni: f64 = di.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                        let nj: f64 = dj.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                        ensure!(
                            dot.abs() <= 1e-3 * ni * nj + 1e-12,
                            "{name}: directions {i} and {j} of lane {s} are not orthogonal"
                        );
                    }
                }
            }
            for i in 0..k {
                let dst = ((layer * max_rank + i) * S) * D;
                host[dst..dst + S * D].copy_from_slice(&v[i * S * D..(i + 1) * S * D]);
            }
            let lanes = layer_meta
                .get(&layer.to_string())
                .and_then(|m| m["lanes"].as_u64())
                .unwrap_or(0b1111) as u32
                & 0b1111;
            ranks[layer] = k as i32;
            masks[layer] = lanes as i32;
            layers.push((layer, k, lanes));
        }
        layers.sort();
        let directions = DeviceBuffer::new(host.len() * 4)?;
        stream.upload(&directions, &host)?;
        let rk = DeviceBuffer::new(LAYERS * 4)?;
        stream.upload(&rk, &ranks)?;
        let mk = DeviceBuffer::new(LAYERS * 4)?;
        stream.upload(&mk, &masks)?;
        stream.synchronize()?;
        Ok(Self {
            directions,
            ranks: rk,
            masks: mk,
            norm_preserve,
            default_strength,
            layers,
            model_sha: meta["model_sha"].as_str().map(str::to_owned),
        })
    }
}
