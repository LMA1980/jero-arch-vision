---
name: jero-arch-vision
description: Multimodal Vision Transformer recipes (CLIP, SigLIP) and VLM soft-prompt mmproj bridge (GGUF and SafeTensors) in jero-arch-vision.
---

# jero-arch-vision

Flat multimodal vision recipe crate for the SpacemiT K3 runtime (`ClipVisionRecipe`, `SiglipVisionRecipe`, and `VlmMmprojBridge`).

## Public Types & Entry Points

- **`ClipVisionRecipe`** (`architecture_id() == "clip"`):
  - Bilinear RGB preprocessing (`preprocess_rgb_hwc`) with CLIP channel statistics.
  - `Conv2D` patch stem (`conv2d_patch_embed_f32`), `[CLS]` token prepending (`v.class_embd`), and dynamic 2D bicubic/bilinear positional interpolation (`interpolate_2d_pos_embed_f32`).
  - Bidirectional pre-LayerNorm encoder blocks (`bidirectional_attention_f32` + `QuickGELU` or `GELU_tanh` MLP) and `v.post_ln`.
  - Standalone `ArchRecipe` execution reading `ExecutionContext::vision_input()` and `vision_dims()`, writing L2-normalized pooled (`Cls` / `Mean`) + optional `visual_projection` vectors to `ExecutionContext::embedding_output_mut()`.
- **`SiglipVisionRecipe`** (`architecture_id() == "siglip"`):
  - Bilinear RGB preprocessing (`preprocess_rgb_hwc`) with `0.5 / 0.5` channel statistics.
  - Spatial-only `Conv2D` patch stem (`has_cls_token = false`), 2D interpolated positional embeddings, bidirectional `GELU_tanh` encoder blocks, and Multi-head Attention Pooling (`MAP` head probe + `head.attention` + `head.mlp`) with `Mean` pooling fallback when `use_map_head = false`.
- **`VlmMmprojBridge`** & **`VisionEncoderConfig`**:
  - Converts `[channels, height, width]` CHW inputs (or precomputed `VisionTokens`) through `CLIP`/`SigLIP`, optional pre-merge norm (`mm.input_norm` / `merger.ln_q`), optional `2x2` spatial merging (`pixel_shuffle_merge_2x2_f32` for Gemma3 / Qwen2-VL), and a 2-layer `GELU` MLP (`mm.0` -> `mm.2`) or single-layer linear projector (`mm.weight` / `mm.input_projection_weight`).
  - `encode_image_to_soft_prompt(&self, weights, vision_pixels, channels, height, width)` returns flattened `[num_tokens, llm_embed_dim]` soft-prompt embeddings.
  - `encode_image_to_soft_prompt_with_dims(...)` returns `(embeds, num_tokens, llm_embed_dim)`.
  - `encode_image_to_soft_prompt_split(vision_weights, mmproj_weights, ...)` supports split backbone + `mmproj-*.gguf` files.

## Tensor Schemas & Fail-Closed Invariants

- Accepts both `GGUF` (`clip.vision.*`, `v.patch_embd.*`, `v.blk.*`, `mm.0.*`/`mm.2.*`) and `SafeTensors` (`vision_model.*`, `multi_modal_projector.*`, `merger.*`) conventions across `F32`, `F16`, `BF16`, and K-quant / legacy-quant weights (with `with-kernels`).
- Fails closed (`K3Error::TensorBoundsError` / `K3Error::CapabilityMismatch`) on missing required tensors, zero or non-divisible geometry, odd spatial grids under `2x2` pixel-shuffle, non-finite inputs/weights/activations, `fc1` without `fc2`, `llm_embed_dim` mismatch, or autoregressive `decode_step` calls.
- Does not depend on `jero-engine` and contains `#![forbid(unsafe_code)]`.

## Verification Commands

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo clippy --no-default-features --all-targets -- -D warnings
cargo check --target riscv64gc-unknown-linux-gnu --all-targets
cargo test --lib --tests
cargo test --no-default-features --lib --tests
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
cargo agents plugin validate Symposium.toml --no-check-crates
```
