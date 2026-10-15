pub mod checkpoint;
pub mod forward;
pub mod loader;
pub mod model;
pub mod oracle;

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
}
