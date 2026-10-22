//! Minimal GGUF (v3) reader: metadata and tensor directory of a memory-mapped file.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(v) => Some(*v),
            Value::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// GGML tensor types this engine reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    Bf16,
    /// 64-value blocks: 4 UE4M3 scales and 32 code bytes.
    Nvfp4,
    Other(u32),
}

impl GgmlType {
    fn from(code: u32) -> Self {
        match code {
            0 => Self::F32,
            30 => Self::Bf16,
            40 => Self::Nvfp4,
            c => Self::Other(c),
        }
    }

    pub fn bytes(self, elements: usize) -> Option<usize> {
        match self {
            Self::F32 => Some(elements * 4),
            Self::Bf16 => Some(elements * 2),
            Self::Nvfp4 => Some(elements / 64 * 36),
            Self::Other(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// ggml order: dims[0] is the contiguous (input) dimension.
    pub dims: Vec<usize>,
    pub ty: GgmlType,
    /// Absolute file offset.
    pub offset: usize,
    pub bytes: usize,
}

impl TensorInfo {
    pub fn elements(&self) -> usize {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    pub map: Mmap,
    pub meta: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        ensure!(self.at + n <= self.b.len(), "GGUF header truncated");
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::Int(self.take(1)?[0] as i64),
            1 => Value::Int(self.take(1)?[0] as i8 as i64),
            2 => Value::Int(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            3 => Value::Int(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::Int(self.u32()? as i64),
            5 => Value::Int(self.u32()? as i32 as i64),
            6 => Value::Float(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::Array(v)
            }
            10 => Value::Int(self.u64()? as i64),
            11 => Value::Int(self.u64()? as i64),
            12 => Value::Float(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            t => bail!("unknown GGUF value type {t}"),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let map = unsafe { Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))?;
        let mut r = Reader { b: &map, at: 0 };
        ensure!(r.take(4)? == b"GGUF", "{}: not a GGUF file", path.display());
        let version = r.u32()?;
        ensure!(version == 3, "GGUF version {version} is not supported");
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;
        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            meta.insert(key, r.value(ty)?);
        }
        let mut dir = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let nd = r.u32()? as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(r.u64()? as usize);
            }
            let ty = GgmlType::from(r.u32()?);
            let offset = r.u64()? as usize;
            dir.push((name, dims, ty, offset));
        }
        let align = meta.get("general.alignment").and_then(Value::as_i64).unwrap_or(32) as usize;
        let data = r.at.div_ceil(align) * align;
        let mut tensors = HashMap::new();
        for (name, dims, ty, offset) in dir {
            let elements: usize = dims.iter().product();
            let bytes = ty.bytes(elements).with_context(|| format!("{name}: unsupported tensor type {ty:?}"))?;
            ensure!(data + offset + bytes <= map.len(), "{name}: data beyond the end of the file");
            tensors.insert(name, TensorInfo { dims, ty, offset: data + offset, bytes });
        }
        Ok(Self { map, meta, tensors })
    }

    pub fn get(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("tensor {name} missing"))
    }

    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        &self.map[t.offset..t.offset + t.bytes]
    }

    pub fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.get(name)?;
        ensure!(t.ty == GgmlType::F32, "{name}: expected F32");
        Ok(self.data(t).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    pub fn meta_i64(&self, key: &str) -> Result<i64> {
        self.meta.get(key).and_then(Value::as_i64).with_context(|| format!("metadata {key} missing"))
    }
}
