//! Index of a safetensors checkpoint directory: every tensor's file, byte range, dtype and shape.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    BF16,
    F16,
    F32,
    F8E4M3,
    U8,
    /// Not a checkpoint dtype: INT8 weights produced at load, with a scale per 64 weights.
    I8G,
    /// Not a checkpoint dtype: MXFP8 weights produced at load (E4M3, a UE8M0 scale per 32 weights).
    MX8,
    I32,
    I64,
}

impl DType {
    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "BF16" => Self::BF16,
            "F16" => Self::F16,
            "F32" => Self::F32,
            "F8_E4M3" => Self::F8E4M3,
            "U8" => Self::U8,
            "I32" => Self::I32,
            "I64" => Self::I64,
            other => bail!("unsupported safetensors dtype {other}"),
        })
    }

    pub fn size(self) -> usize {
        match self {
            Self::F8E4M3 | Self::U8 | Self::I8G | Self::MX8 => 1,
            Self::BF16 | Self::F16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub file: usize,
    /// Absolute byte offset of the tensor data within its file.
    pub offset: u64,
    pub bytes: u64,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

pub struct SafetensorsFile {
    pub path: PathBuf,
    pub file: File,
    /// Absolute offset of the data section (8 + header length).
    pub data_start: u64,
    pub len: u64,
}

pub struct Checkpoint {
    pub dir: PathBuf,
    pub files: Vec<SafetensorsFile>,
    pub tensors: HashMap<String, TensorInfo>,
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Manifest of an overlay directory: a variant of a checkpoint that changes some of its tensors. The
/// directory's own safetensors files replace the base checkpoint's tensors of the same name, dtype and
/// shape; its other files (config, tokenizer, template) are those of the base.
pub const OVERLAY: &str = "overlay.json";

#[derive(Deserialize)]
struct Overlay {
    /// The base checkpoint directory, absolute or relative to the overlay directory.
    base: PathBuf,
}

/// Bytes that identify a checkpoint's numerics: its config.json, plus the overlay manifest of an
/// overlay directory (whose config.json is the base's).
pub fn identity(dir: &Path) -> Vec<u8> {
    let mut id = std::fs::read(dir.join("config.json")).unwrap_or_default();
    if let Ok(overlay) = std::fs::read(dir.join(OVERLAY)) {
        id.extend_from_slice(b"\n");
        id.extend(overlay);
    }
    id
}

impl Checkpoint {
    /// Reads the header of every `*.safetensors` file in `dir`, or, for an overlay directory, of its
    /// base checkpoint's files and its own.
    pub fn open(dir: &Path) -> Result<Self> {
        let manifest = dir.join(OVERLAY);
        if !manifest.exists() {
            return Self::read_dir(dir);
        }
        let overlay: Overlay = serde_json::from_slice(&std::fs::read(&manifest)?)
            .with_context(|| format!("parsing {}", manifest.display()))?;
        let mut ckpt = Self::open(&dir.join(&overlay.base))
            .context("opening the overlay's base checkpoint")?;
        let own = Self::read_dir(dir)?;
        let first = ckpt.files.len();
        for (name, mut info) in own.tensors {
            let base = ckpt
                .tensors
                .get(&name)
                .with_context(|| format!("overlay tensor {name} is not in the base checkpoint"))?;
            ensure!(
                base.dtype == info.dtype && base.shape == info.shape,
                "overlay tensor {name}: {:?} {:?}, base {:?} {:?}",
                info.dtype,
                info.shape,
                base.dtype,
                base.shape
            );
            info.file += first;
            ckpt.tensors.insert(name, info);
        }
        ckpt.files.extend(own.files);
        ckpt.dir = dir.to_path_buf();
        Ok(ckpt)
    }

    fn read_dir(dir: &Path) -> Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<_>>()?;
        paths.retain(|p| p.extension().is_some_and(|e| e == "safetensors"));
        paths.sort();
        ensure!(
            !paths.is_empty(),
            "no .safetensors files in {}",
            dir.display()
        );

        let mut files = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::new();
        for path in paths {
            let mut file =
                File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            let len = file.metadata()?.len();
            let mut header_len = [0u8; 8];
            file.read_exact(&mut header_len)?;
            let header_len = u64::from_le_bytes(header_len);
            ensure!(
                header_len < len,
                "{}: header length {header_len} exceeds file",
                path.display()
            );
            let mut header = vec![0u8; header_len as usize];
            file.read_exact(&mut header)?;
            let data_start = 8 + header_len;

            let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(&header)
                .with_context(|| format!("parsing header of {}", path.display()))?;
            let index = files.len();
            for (name, value) in raw {
                if name == "__metadata__" {
                    continue;
                }
                let t: RawTensor =
                    serde_json::from_value(value).with_context(|| format!("tensor {name}"))?;
                let dtype = DType::parse(&t.dtype)?;
                let [begin, end] = t.data_offsets;
                let elements: usize = t.shape.iter().product();
                ensure!(
                    end - begin == (elements * dtype.size()) as u64 && data_start + end <= len,
                    "{name}: inconsistent byte range in {}",
                    path.display()
                );
                let info = TensorInfo {
                    file: index,
                    offset: data_start + begin,
                    bytes: end - begin,
                    dtype,
                    shape: t.shape,
                };
                ensure!(
                    tensors.insert(name.clone(), info).is_none(),
                    "duplicate tensor {name}"
                );
            }
            files.push(SafetensorsFile {
                path,
                file,
                data_start,
                len,
            });
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            files,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("checkpoint has no tensor {name}"))
    }

    /// Reads a tensor's raw bytes from disk.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let info = self.get(name)?;
        let mut bytes = vec![0u8; info.bytes as usize];
        crate::fileio::read_exact_at(&self.files[info.file].file, &mut bytes, info.offset)?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, tensors: &[(&str, &[usize], &[u8])]) {
        let mut header = serde_json::Map::new();
        let mut at = 0;
        for (name, shape, data) in tensors {
            header.insert(name.to_string(), serde_json::json!({"dtype": "U8", "shape": shape, "data_offsets": [at, at + data.len()]}));
            at += data.len();
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        tensors.iter().for_each(|t| bytes.extend_from_slice(t.2));
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn overlay_replaces_tensors() {
        let dir =
            std::env::temp_dir().join(format!("quench6k-overlay-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (base, over) = (dir.join("base"), dir.join("over"));
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&over).unwrap();
        std::fs::write(base.join("config.json"), "{}").unwrap();
        write(
            &base.join("a.safetensors"),
            &[("x", &[2], &[1, 2]), ("y", &[2], &[3, 4])],
        );
        write(&over.join("overlay.safetensors"), &[("y", &[2], &[5, 6])]);
        std::fs::write(over.join(OVERLAY), r#"{"base": "../base"}"#).unwrap();
        let ckpt = Checkpoint::open(&over).unwrap();
        assert_eq!(
            (ckpt.read("x").unwrap(), ckpt.read("y").unwrap()),
            (vec![1, 2], vec![5, 6])
        );
        assert_ne!(identity(&over), identity(&base));
        // A tensor the base lacks, or one of another shape, is an error.
        write(&over.join("overlay.safetensors"), &[("y", &[1], &[5])]);
        assert!(Checkpoint::open(&over).is_err());
        write(&over.join("overlay.safetensors"), &[("z", &[1], &[5])]);
        assert!(Checkpoint::open(&over).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
