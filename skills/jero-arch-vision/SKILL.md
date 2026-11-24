---
name: jero-arch-vision
description: Multimodal Vision Transformer recipes (`CLIP`, `SigLIP`) and VLM soft-prompt `mmproj` bridge (`ClipVisionRecipe`, `SiglipVisionRecipe`, `VlmMmprojBridge`) for the SpacemiT K3 platform.
---

# `jero-arch-vision` Architecture Crate

Provides Multimodal Vision Transformer recipes (`ClipVisionRecipe`, `SiglipVisionRecipe`) and `VlmMmprojBridge` for standalone image embeddings and VLM soft-prompt projection on K3.

## Crate Contract & Dependencies

- **License**: `LGPL-3.0-only WITH LGPL-3.0-linking-exception`
- **Upstream Dependencies**: `jero-core` (zero-copy `ModelWeights`, `ExecutionContext`, `ArchRecipe`, `K3Error`) and optional `jero-kernels-rvv` (enabled by default via `with-kernels`).
- **Safety & Lints**: `#![deny(unsafe_op_in_unsafe_fn)]`, `clippy::pedantic`, and `clippy::undocumented_unsafe_blocks`.

## Local Verification Gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --lib --tests
cargo test --no-default-features --lib --tests
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
```
