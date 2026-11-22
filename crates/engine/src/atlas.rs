//! Neuron atlas: activation statistics of every expert neuron over a corpus. While a corpus is
//! prefilled with a [`Recorder`] attached, each layer's expert activations add into per-neuron sums of
//! |activation|, the strongest activation (with its sign and corpus token), and per-expert routing
//! counts, on the device; [`Recorder::download`] turns them into [`Stats`].

use anyhow::Result;
use quench6k_cuda::{DeviceBuffer, Stream, kernels as k};

use crate::model::{EXPERT_WIDTH, EXPERTS, LAYERS};

const ROUTED: usize = LAYERS * EXPERTS * EXPERT_WIDTH;
const SHARED: usize = LAYERS * EXPERT_WIDTH;

pub struct Recorder {
    sum: DeviceBuffer,
    top: DeviceBuffer,
    counts: DeviceBuffer,
    shared_sum: DeviceBuffer,
    shared_top: DeviceBuffer,
}

impl std::fmt::Debug for Recorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Recorder")
    }
}

impl Recorder {
    pub fn new() -> Result<Self> {
        Ok(Self {
            sum: DeviceBuffer::zeroed(ROUTED * 4)?,
            top: DeviceBuffer::zeroed(ROUTED * 8)?,
            counts: DeviceBuffer::zeroed(LAYERS * EXPERTS * 4)?,
            shared_sum: DeviceBuffer::zeroed(SHARED * 4)?,
            shared_top: DeviceBuffer::zeroed(SHARED * 8)?,
        })
    }

    /// Records layer `layer`'s routed activations (`act` rows, `top_i`/`pair_row` over `pairs`
    /// token-major pairs; the first pair's token is corpus token `token0`).
    pub fn routed(
        &self,
        s: &Stream,
        layer: usize,
        act: u64,
        top_i: u64,
        pair_row: u64,
        pairs: usize,
        token0: usize,
    ) -> Result<()> {
        let n = layer * EXPERTS * EXPERT_WIDTH;
        k::atlas_record(
            s,
            act,
            top_i,
            pair_row,
            pairs,
            token0,
            self.sum.addr() + (n * 4) as u64,
            self.top.addr() + (n * 8) as u64,
            self.counts.addr() + (layer * EXPERTS * 4) as u64,
        )
    }

    /// Records layer `layer`'s shared-expert activations (`rows` tokens from corpus token `token0`).
    pub fn shared(
        &self,
        s: &Stream,
        layer: usize,
        act: u64,
        rows: usize,
        token0: usize,
    ) -> Result<()> {
        let n = layer * EXPERT_WIDTH;
        k::atlas_record(
            s,
            act,
            0,
            0,
            rows,
            token0,
            self.shared_sum.addr() + (n * 4) as u64,
            self.shared_top.addr() + (n * 8) as u64,
            0,
        )
    }

    pub fn download(&self, s: &Stream) -> Result<Stats> {
        Ok(Stats {
            sum: s.download(self.sum.addr(), ROUTED)?,
            top: s.download(self.top.addr(), ROUTED)?,
            counts: s.download(self.counts.addr(), LAYERS * EXPERTS)?,
            shared_sum: s.download(self.shared_sum.addr(), SHARED)?,
            shared_top: s.download(self.shared_top.addr(), SHARED)?,
        })
    }
}

/// A neuron: (layer, routed expert or None for the shared expert, unit).
pub type NeuronId = (usize, Option<usize>, usize);

/// The strongest activation of a neuron: its value (signed) and corpus token.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Peak {
    pub value: f32,
    pub token: usize,
}

pub struct Stats {
    /// Per neuron (layer, expert, unit): the sum of |activation| and the packed strongest activation.
    pub sum: Vec<f32>,
    pub top: Vec<u64>,
    /// Per (layer, expert): tokens routed to the expert.
    pub counts: Vec<u32>,
    pub shared_sum: Vec<f32>,
    pub shared_top: Vec<u64>,
}

impl Stats {
    fn index((layer, expert, unit): NeuronId) -> (bool, usize) {
        match expert {
            Some(e) => (true, (layer * EXPERTS + e) * EXPERT_WIDTH + unit),
            None => (false, layer * EXPERT_WIDTH + unit),
        }
    }

    /// Sum of |activation| over the corpus.
    pub fn sum(&self, n: NeuronId) -> f32 {
        match Self::index(n) {
            (true, i) => self.sum[i],
            (false, i) => self.shared_sum[i],
        }
    }

    pub fn peak(&self, n: NeuronId) -> Option<Peak> {
        let packed = match Self::index(n) {
            (true, i) => self.top[i],
            (false, i) => self.shared_top[i],
        };
        (packed != 0).then(|| {
            let value = f32::from_bits((packed >> 32) as u32);
            Peak {
                value: if packed & (1 << 31) != 0 {
                    -value
                } else {
                    value
                },
                token: (packed & 0x7fff_ffff) as usize,
            }
        })
    }

    /// Tokens routed to `expert` in `layer` (every token for the shared expert).
    pub fn routed(&self, layer: usize, expert: usize) -> u32 {
        self.counts[layer * EXPERTS + expert]
    }

    /// Every neuron, in index order.
    pub fn neurons() -> impl Iterator<Item = NeuronId> {
        (0..LAYERS).flat_map(|l| {
            (0..EXPERTS)
                .flat_map(move |e| (0..EXPERT_WIDTH).map(move |u| (l, Some(e), u)))
                .chain((0..EXPERT_WIDTH).map(move |u| (l, None, u)))
        })
    }
}
