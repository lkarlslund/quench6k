//! Model edits applied per request: neurons (an expert's intermediate units) scaled, flipped, switched
//! off or pushed, and router biases that ban or favour experts. A request with edits decodes alone and
//! eagerly (no speculation, no batched steps or graphs), so other requests never see them; its cached
//! states are tagged with the edit set and only reused by requests with the same edits.

use anyhow::{Result, ensure};
use quench6k_cuda::{DeviceBuffer, Stream};

use crate::model::{EXPERT_WIDTH, EXPERTS, LAYERS};

#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    /// activation = activation * scale + add, for unit `unit` of routed `expert` (None: the shared
    /// expert) in `layer`; a routed expert's edit applies to tokens that route to it.
    Neuron {
        layer: usize,
        expert: Option<usize>,
        unit: usize,
        scale: f32,
        add: f32,
    },
    /// Adds `bias` to the router logit of `expert` in `layer` (-inf: never routed).
    Router {
        layer: usize,
        expert: usize,
        bias: f32,
    },
}

/// Device arrays of one kind of edit in one layer: `n` entries, each array `n` long.
pub struct Arrays {
    pub buf: DeviceBuffer,
    pub n: usize,
}

#[derive(Default)]
pub struct LayerEdits {
    /// Routed neurons: experts I32, units I32, scale F32, add F32.
    pub routed: Option<Arrays>,
    /// Shared-expert neurons, same layout (experts unused).
    pub shared: Option<Arrays>,
    /// Router biases: experts I32, bias F32.
    pub router: Option<Arrays>,
}

pub struct Edits {
    pub list: Vec<Edit>,
    /// Identifies the edit set (cached states are tagged with it); never 0.
    pub id: u64,
    /// Per layer, uploaded by [`upload`](Self::upload) on the inference thread.
    device: std::sync::OnceLock<Vec<LayerEdits>>,
}

impl std::fmt::Debug for Edits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Edits({:016x}, {:?})", self.id, self.list)
    }
}

fn upload(s: &Stream, words: &[[u32; 4]], cols: usize) -> Result<Option<Arrays>> {
    if words.is_empty() {
        return Ok(None);
    }
    let n = words.len();
    let flat: Vec<u32> = (0..cols)
        .flat_map(|c| words.iter().map(move |w| w[c]))
        .collect();
    let buf = DeviceBuffer::new(flat.len() * 4)?;
    s.upload_to(buf.addr(), &flat)?;
    Ok(Some(Arrays { buf, n }))
}

impl Edits {
    /// Checks the edits; the device arrays are uploaded on first use.
    pub fn new(list: Vec<Edit>) -> Result<Self> {
        ensure!(!list.is_empty(), "no edits");
        for e in &list {
            match *e {
                Edit::Neuron {
                    layer,
                    expert,
                    unit,
                    scale,
                    add,
                } => {
                    ensure!(layer < LAYERS, "layer {layer} out of range");
                    ensure!(unit < EXPERT_WIDTH, "unit {unit} out of range");
                    ensure!(expert.is_none_or(|e| e < EXPERTS), "expert out of range");
                    ensure!(
                        scale.is_finite() && add.is_finite(),
                        "scale and add must be finite"
                    );
                }
                Edit::Router {
                    layer,
                    expert,
                    bias,
                } => {
                    ensure!(layer < LAYERS, "layer {layer} out of range");
                    ensure!(expert < EXPERTS, "expert {expert} out of range");
                    ensure!(!bias.is_nan(), "router bias must be a number");
                }
            }
        }
        // FNV-1a over the edits' description.
        let id = format!("{list:?}")
            .bytes()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
            })
            .max(1);
        Ok(Self {
            list,
            id,
            device: std::sync::OnceLock::new(),
        })
    }

    /// The edits of `layer` (after [`upload`](Self::upload)).
    pub fn layer(&self, layer: usize) -> Option<&LayerEdits> {
        self.device.get().and_then(|d| d.get(layer))
    }

    pub fn upload(&self, s: &Stream) -> Result<()> {
        if self.device.get().is_some() {
            return Ok(());
        }
        let mut per: Vec<[Vec<[u32; 4]>; 3]> = (0..LAYERS).map(|_| Default::default()).collect();
        for e in &self.list {
            match *e {
                Edit::Neuron {
                    layer,
                    expert,
                    unit,
                    scale,
                    add,
                } => {
                    let w = [
                        expert.map_or(u32::MAX, |e| e as u32),
                        unit as u32,
                        scale.to_bits(),
                        add.to_bits(),
                    ];
                    per[layer][if expert.is_some() { 0 } else { 1 }].push(w);
                }
                Edit::Router {
                    layer,
                    expert,
                    bias,
                } => {
                    per[layer][2].push([expert as u32, bias.to_bits(), 0, 0]);
                }
            }
        }
        let layers = per
            .iter()
            .map(|[routed, shared, router]| {
                Ok(LayerEdits {
                    routed: upload(s, routed, 4)?,
                    shared: upload(s, shared, 4)?,
                    router: upload(s, router, 2)?,
                })
            })
            .collect::<Result<_>>()?;
        s.synchronize()?;
        let _ = self.device.set(layers);
        Ok(())
    }
}
