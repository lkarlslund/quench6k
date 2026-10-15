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
            Self::F8E4M3 | Self::U8 => 1,
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

impl Checkpoint {
    /// Reads the header of every `*.safetensors` file in `dir`.
    pub fn open(dir: &Path) -> Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<_>>()?;
        paths.retain(|p| p.extension().is_some_and(|e| e == "safetensors"));
        paths.sort();
        ensure!(!paths.is_empty(), "no .safetensors files in {}", dir.display());

        let mut files = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::new();
        for path in paths {
            let mut file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            let len = file.metadata()?.len();
            let mut header_len = [0u8; 8];
            file.read_exact(&mut header_len)?;
            let header_len = u64::from_le_bytes(header_len);
            ensure!(header_len < len, "{}: header length {header_len} exceeds file", path.display());
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
                let t: RawTensor = serde_json::from_value(value).with_context(|| format!("tensor {name}"))?;
                let dtype = DType::parse(&t.dtype)?;
                let [begin, end] = t.data_offsets;
                let elements: usize = t.shape.iter().product();
                ensure!(
                    end - begin == (elements * dtype.size()) as u64 && data_start + end <= len,
                    "{name}: inconsistent byte range in {}",
                    path.display()
                );
                let info = TensorInfo { file: index, offset: data_start + begin, bytes: end - begin, dtype, shape: t.shape };
                ensure!(tensors.insert(name.clone(), info).is_none(), "duplicate tensor {name}");
            }
            files.push(SafetensorsFile { path, file, data_start, len });
        }
        Ok(Self { dir: dir.to_path_buf(), files, tensors })
    }

    pub fn get(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("checkpoint has no tensor {name}"))
    }

    /// Reads a tensor's raw bytes from disk.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        use std::os::unix::fs::FileExt;
        let info = self.get(name)?;
        let mut bytes = vec![0u8; info.bytes as usize];
        self.files[info.file].file.read_exact_at(&mut bytes, info.offset)?;
        Ok(bytes)
    }
}
