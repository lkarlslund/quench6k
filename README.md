# squash

A from-scratch inference engine for Qwen3.8 Flash-Next 125B-A6B on a single NVIDIA RTX PRO 6000
Blackwell GPU: Rust host, hand-written CUDA kernels for `sm_120a`.

Scope: text and vision, MTP speculative decoding, a small number of concurrent requests, and
direction-removal activation steering. Weights load directly from the ModelOpt NVFP4 safetensors
checkpoint.

Status: design. No license has been chosen yet; all rights reserved until one is.
