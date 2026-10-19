pub mod checkpoint;
pub mod context_store;
pub mod dflash;
pub mod fileio;
pub mod forward;
pub mod loader;
pub mod model;
pub mod oracle;
pub mod profile;
pub mod session;
pub mod spec;
pub mod steering;
pub mod vision;

/// Which checkpoint tensors belong on the GPU.
pub mod selection {
    /// Text-model weights: everything under `model.language_model` except the host-resident PLE table.
    pub fn text(name: &str) -> bool {
        (name.starts_with("model.language_model.") && !name.contains(".ngram_embedding.")) || name == "lm_head.weight"
    }

    pub fn mtp(name: &str) -> bool {
        name.starts_with("mtp.")
    }

    pub fn vision(name: &str) -> bool {
        name.starts_with("model.visual.")
    }

    /// Projections that the 8-bit profiles quantize. Everything else (embedding, norms, convolutions,
    /// gating vectors and the NVFP4 experts) keeps its checkpoint format.
    pub fn eight_bit(name: &str) -> bool {
        if name.contains(".experts.") || name.contains(".ngram_embedding.") {
            return false;
        }
        const SUFFIXES: [&str; 22] = [
            "mtp.fc_embedding.weight",
            "mtp.fc_hidden.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "block_inject_weight.weight",
            "indexer.index_qk_proj.weight",
            "mlp.shared_expert_gate.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.out_proj.weight",
            "input_mix_weight_down.weight",
            "input_mix_weight_up.weight",
            "mlp.gate.weight",
            "shared_expert.gate_proj.weight",
            "shared_expert.up_proj.weight",
            "shared_expert.down_proj.weight",
            "ple.key_proj.weight",
            "ple.value_proj.weight",
        ];
        name == "lm_head.weight" || SUFFIXES.iter().any(|s| name.ends_with(s))
    }
}
