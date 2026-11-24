# jero-arch-vision

Multimodal Vision Transformer (`CLIP`, `SigLIP`) architecture recipes and VLM soft-prompt `mmproj` projection bridge for the SpacemiT K3 runtime.

## Implemented Components

- **`ClipVisionRecipe`** (`"clip"`): `ArchRecipe` implementation executing RGB bilinear preprocessing, `Conv2D` patch embedding (`conv2d_patch_embed_f32`), optional `[CLS]` token prepending (`v.class_embd`), dynamic 2D spatial positional interpolation (`interpolate_2d_pos_embed_f32`), pre-LayerNorm bidirectional self-attention (`bidirectional_attention_f32`) + `QuickGELU`/`GELU_tanh` MLP blocks, `v.post_ln`, and standalone L2-normalized image embedding projection (`visual_projection`).
- **`SiglipVisionRecipe`** (`"siglip"`): `ArchRecipe` implementation executing `Conv2D` patch embedding without `[CLS]`, dynamic 2D spatial positional interpolation, bidirectional `GELU_tanh` encoder blocks, `v.post_ln`, and Multi-head Attention Pooling (`MAP` head probe) or `Mean` pooling for standalone vision embeddings.
- **`VlmMmprojBridge` & `VisionEncoderConfig`**: Multimodal vision-to-language bridge combining `CLIP`/`SigLIP` encoding, optional pre-merge `LayerNorm` (`mm.input_norm` / `merger.ln_q`), optional `2x2` spatial merging (`pixel_shuffle_merge_2x2_f32` for Gemma3 and Qwen2-VL), and 2-layer `GELU` MLP (`mm.0` -> `mm.2`) or single-layer linear projection across both `GGUF` (`mmproj-*.gguf`) and `SafeTensors` weight containers.

## Feature Flags & Dependencies

- Default feature `with-kernels` dispatches kernels through `jero-kernels-rvv`.
- `--no-default-features` compiles and runs using portable scalar equivalents while enforcing identical bounds, finiteness, and fail-closed validation invariants.
- Remains a flat peer crate depending strictly on `jero-core` (and optional `jero-kernels-rvv`), with no dependency on `jero-engine` and `#![forbid(unsafe_code)]`.

## Verification

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo clippy --no-default-features --all-targets -- -D warnings
cargo check --target riscv64gc-unknown-linux-gnu --all-targets
cargo test --lib --tests
cargo test --no-default-features --lib --tests
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
cargo agents plugin validate Symposium.toml --no-check-crates
```

## License

Licensed under `LGPL-3.0-only WITH LGPL-3.0-linking-exception`. See `LICENSE` for details.
