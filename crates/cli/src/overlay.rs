//! `quench6k overlay`: compares a variant checkpoint (for example a weight edit of the model, in BF16)
//! with the checkpoint we run, tensor by tensor, and writes the tensors it changes as an overlay
//! directory (checkpoint::OVERLAY) that loads as the base checkpoint with those tensors replaced.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use quench6k_engine::checkpoint::{Checkpoint, DType, OVERLAY};
use quench6k_engine::fileio::read_exact_at;

const CHUNK: usize = 64 << 20;

/// A differing tensor: elements that differ, and the largest absolute difference (float dtypes).
struct Diff {
    name: String,
    elements: usize,
    differing: usize,
    max_abs: f64,
}

fn value(dtype: DType, b: &[u8]) -> Option<f64> {
    Some(match dtype {
        DType::BF16 => f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16) as f64,
        DType::F32 => f32::from_le_bytes(b.try_into().ok()?) as f64,
        _ => return None,
    })
}

/// Compares one tensor present in both checkpoints with the same dtype and shape, in chunks.
fn compare(base: &Checkpoint, variant: &Checkpoint, name: &str) -> Result<Option<Diff>> {
    let (a, b) = (base.get(name)?, variant.get(name)?);
    let size = a.dtype.size();
    let (mut buf_a, mut buf_b) = (vec![0u8; CHUNK], vec![0u8; CHUNK]);
    let mut diff = Diff {
        name: name.into(),
        elements: a.bytes as usize / size,
        differing: 0,
        max_abs: 0.0,
    };
    let mut at = 0u64;
    while at < a.bytes {
        let n = CHUNK.min((a.bytes - at) as usize);
        read_exact_at(&base.files[a.file].file, &mut buf_a[..n], a.offset + at)?;
        read_exact_at(&variant.files[b.file].file, &mut buf_b[..n], b.offset + at)?;
        if buf_a[..n] != buf_b[..n] {
            for (x, y) in buf_a[..n]
                .chunks_exact(size)
                .zip(buf_b[..n].chunks_exact(size))
            {
                if x != y {
                    diff.differing += 1;
                    if let (Some(x), Some(y)) = (value(a.dtype, x), value(a.dtype, y)) {
                        diff.max_abs = diff.max_abs.max((x - y).abs());
                    }
                }
            }
        }
        at += n as u64;
    }
    Ok((diff.differing > 0).then_some(diff))
}

/// "layers.N.mlp.experts.N.up_proj.weight" for a tensor name: layer and expert numbers folded.
fn pattern(name: &str) -> String {
    let mut out = String::new();
    for part in name.split('.') {
        out.push_str(if part.chars().all(|c| c.is_ascii_digit()) {
            "N"
        } else {
            part
        });
        out.push('.');
    }
    out.pop();
    out
}

fn summarize(what: &str, names: &[&String]) {
    let mut by: BTreeMap<String, usize> = BTreeMap::new();
    for n in names {
        *by.entry(pattern(n)).or_default() += 1;
    }
    println!("{what}: {} tensors", names.len());
    for (p, n) in by {
        println!("  {n:>6}  {p}");
    }
}

/// Safetensors dtype names of the checkpoint dtypes an overlay can hold.
fn dtype_name(d: DType) -> Result<&'static str> {
    Ok(match d {
        DType::BF16 => "BF16",
        DType::F16 => "F16",
        DType::F32 => "F32",
        DType::F8E4M3 => "F8_E4M3",
        DType::U8 => "U8",
        DType::I32 => "I32",
        DType::I64 => "I64",
        other => anyhow::bail!("{other:?} is not a checkpoint dtype"),
    })
}

fn write_overlay(
    base_dir: &Path,
    variant: &Checkpoint,
    names: &[String],
    out: &Path,
) -> Result<()> {
    ensure!(
        !out.exists() || std::fs::read_dir(out)?.next().is_none(),
        "{} exists and is not empty",
        out.display()
    );
    std::fs::create_dir_all(out)?;
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for n in names {
        let t = variant.get(n)?;
        header.insert(n.clone(), serde_json::json!({"dtype": dtype_name(t.dtype)?, "shape": t.shape, "data_offsets": [offset, offset + t.bytes]}));
        offset += t.bytes;
    }
    let mut header = serde_json::to_vec(&header)?;
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut f = std::io::BufWriter::new(std::fs::File::create(out.join("overlay.safetensors"))?);
    f.write_all(&(header.len() as u64).to_le_bytes())?;
    f.write_all(&header)?;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for n in names {
        let bytes = variant.read(n)?;
        hash = bytes
            .iter()
            .fold(hash, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
        f.write_all(&bytes)?;
    }
    f.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    // The manifest names the base and identifies the content (stored contexts are keyed on it).
    let base_dir = std::fs::canonicalize(base_dir)?;
    let manifest = serde_json::json!({"base": base_dir, "source": std::fs::canonicalize(&variant.dir)?, "tensors": names, "fnv64": format!("{hash:016x}")});
    std::fs::write(
        out.join(OVERLAY),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    // Everything else (config, tokenizer, template, preprocessors) is the base's.
    for entry in std::fs::read_dir(&base_dir)? {
        let p = entry?.path();
        let name = p
            .file_name()
            .context("file name")?
            .to_string_lossy()
            .into_owned();
        if p.is_file() && !name.ends_with(".safetensors") && name != "model.safetensors.index.json"
        {
            std::os::unix::fs::symlink(&p, out.join(&name))?;
        }
    }
    Ok(())
}

pub fn run(base_dir: &Path, variant_dir: &Path, out: Option<&Path>) -> Result<()> {
    let base = Checkpoint::open(base_dir)?;
    let variant = Checkpoint::open(variant_dir)?;
    let mut common: Vec<&String> = base
        .tensors
        .keys()
        .filter(|n| variant.tensors.contains_key(*n))
        .collect();
    common.sort();
    let only_base: Vec<&String> = base
        .tensors
        .keys()
        .filter(|n| !variant.tensors.contains_key(*n))
        .collect();
    let only_variant: Vec<&String> = variant
        .tensors
        .keys()
        .filter(|n| !base.tensors.contains_key(*n))
        .collect();
    let (mut same, mut same_bytes, mut diffs, mut format) = (0usize, 0u64, Vec::new(), Vec::new());
    for (i, n) in common.iter().enumerate() {
        let (a, b) = (base.get(n)?, variant.get(n)?);
        if a.dtype != b.dtype || a.shape != b.shape {
            format.push(*n);
            continue;
        }
        match compare(&base, &variant, n)? {
            Some(d) => diffs.push(d),
            None => {
                same += 1;
                same_bytes += a.bytes;
            }
        }
        if (i + 1) % 100 == 0 {
            eprintln!("compared {} of {} tensors", i + 1, common.len());
        }
    }
    println!(
        "in both, identical: {same} tensors, {:.2} GiB",
        same_bytes as f64 / (1u64 << 30) as f64
    );
    println!("in both, different values: {} tensors", diffs.len());
    for d in &diffs {
        println!(
            "  {}: {} of {} elements, max |diff| {:.3e}",
            d.name, d.differing, d.elements, d.max_abs
        );
    }
    summarize("in both, different dtype or shape (not compared)", &format);
    summarize("only in the base", &only_base);
    summarize("only in the variant", &only_variant);
    // The other files of the model directory.
    for entry in std::fs::read_dir(base_dir)? {
        let p = entry?.path();
        let name = p
            .file_name()
            .context("file name")?
            .to_string_lossy()
            .into_owned();
        if !p.is_file() || name.ends_with(".safetensors") || name.ends_with(".index.json") {
            continue;
        }
        if let Ok(v) = std::fs::read(variant_dir.join(&name)) {
            println!(
                "{name}: {}",
                if v == std::fs::read(&p)? {
                    "identical"
                } else {
                    "differs"
                }
            );
        }
    }
    if let Some(out) = out {
        let names: Vec<String> = diffs.iter().map(|d| d.name.clone()).collect();
        ensure!(
            !names.is_empty(),
            "the variant changes no tensor the base holds in the same format"
        );
        write_overlay(base_dir, &variant, &names, out)?;
        println!("wrote {} ({} tensors)", out.display(), names.len());
    }
    Ok(())
}
