//! `jero-arch-vision`: Multimodal Vision Transformer recipes (`CLIP`, `SigLIP`) and VLM
//! soft-prompt `mmproj` bridge for the `SpacemiT` K3 platform.
//!
//! Provides flat, kernel-dispatched implementations of:
//! - [`ClipVisionRecipe`] (`"clip"`): `Conv2D` patch stem + `[CLS]` token + dynamic 2D positional
//!   interpolation + bidirectional `QuickGELU` / `GELU_tanh` transformer blocks + `visual_projection`.
//! - [`SiglipVisionRecipe`] (`"siglip"`): `Conv2D` patch stem + dynamic 2D positional interpolation
//!   + bidirectional `GELU_tanh` transformer blocks + `MAP` attention pooling probe.
//! - [`VlmMmprojBridge`]: Vision backbone + optional `2x2` spatial pixel-shuffle merge + 2-layer
//!   MLP or linear multimodal projector supporting both `GGUF` (`mmproj-*.gguf`) and `SafeTensors`
//!   weights.

#![deny(unsafe_op_in_unsafe_fn)]
#![forbid(unsafe_code)]

mod kernels;
pub mod mmproj;
pub mod vit;

pub use mmproj::{VisionEncoderConfig, VlmMmprojBridge};
pub use vit::{ClipVisionRecipe, SiglipVisionRecipe, VisionTokens};

/// Canonical crate identifier for this architecture recipe package.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// Canonical SPDX license identifier carried by `jero-arch-vision`.
pub const CRATE_LICENSE: &str = env!("CARGO_PKG_LICENSE");

/// Returns the crate metadata description.
#[must_use]
pub const fn crate_description() -> &'static str {
    env!("CARGO_PKG_DESCRIPTION")
}

#[cfg(test)]
mod tests {
    use jero_core::safetensors::test_helpers::create_synthetic_safetensors;
    use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
    use jero_core::{
        ArchRecipe, EmbeddingPooling, ExecutionContext, GgufFile, GgufValue, K3Error, ModelWeights,
        QuantType, SafeTensorsFile,
    };

    use super::{
        CRATE_LICENSE, CRATE_NAME, ClipVisionRecipe, SiglipVisionRecipe, VisionEncoderConfig,
        VlmMmprojBridge, crate_description,
    };

    fn f32_to_le_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(size_of_val(vals));
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn pattern_weights(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|idx| {
                let rem = i16::try_from((idx * 17 + seed * 31) % 23).expect("rem fits i16");
                (f32::from(rem) - 11.0) * 0.035_f32
            })
            .collect()
    }

    fn push_f32(
        owned: &mut Vec<(&'static str, Vec<usize>, Vec<u8>)>,
        name: &'static str,
        shape: Vec<usize>,
        vals: &[f32],
    ) {
        owned.push((name, shape, f32_to_le_bytes(vals)));
    }

    fn push_clip_layer0(
        owned: &mut Vec<(&'static str, Vec<usize>, Vec<u8>)>,
        hidden: usize,
        ffn: usize,
    ) {
        let ones = vec![1.0_f32; hidden];
        let zeros = vec![0.0_f32; hidden];
        push_f32(
            owned,
            "v.blk.0.attn_q.weight",
            vec![hidden, hidden],
            &pattern_weights(hidden * hidden, 5),
        );
        push_f32(
            owned,
            "v.blk.0.attn_q.bias",
            vec![hidden],
            &pattern_weights(hidden, 6),
        );
        push_f32(
            owned,
            "v.blk.0.attn_k.weight",
            vec![hidden, hidden],
            &pattern_weights(hidden * hidden, 7),
        );
        push_f32(
            owned,
            "v.blk.0.attn_k.bias",
            vec![hidden],
            &pattern_weights(hidden, 8),
        );
        push_f32(
            owned,
            "v.blk.0.attn_v.weight",
            vec![hidden, hidden],
            &pattern_weights(hidden * hidden, 9),
        );
        push_f32(
            owned,
            "v.blk.0.attn_v.bias",
            vec![hidden],
            &pattern_weights(hidden, 10),
        );
        push_f32(
            owned,
            "v.blk.0.attn_out.weight",
            vec![hidden, hidden],
            &pattern_weights(hidden * hidden, 11),
        );
        push_f32(
            owned,
            "v.blk.0.attn_out.bias",
            vec![hidden],
            &pattern_weights(hidden, 12),
        );
        push_f32(owned, "v.blk.0.ln1.weight", vec![hidden], &ones);
        push_f32(owned, "v.blk.0.ln1.bias", vec![hidden], &zeros);
        push_f32(
            owned,
            "v.blk.0.ffn_up.weight",
            vec![ffn, hidden],
            &pattern_weights(ffn * hidden, 13),
        );
        push_f32(
            owned,
            "v.blk.0.ffn_up.bias",
            vec![ffn],
            &pattern_weights(ffn, 14),
        );
        push_f32(
            owned,
            "v.blk.0.ffn_down.weight",
            vec![hidden, ffn],
            &pattern_weights(hidden * ffn, 15),
        );
        push_f32(
            owned,
            "v.blk.0.ffn_down.bias",
            vec![hidden],
            &pattern_weights(hidden, 16),
        );
        push_f32(owned, "v.blk.0.ln2.weight", vec![hidden], &ones);
        push_f32(owned, "v.blk.0.ln2.bias", vec![hidden], &zeros);
    }

    fn build_clip_safetensors_bytes(
        hidden: usize,
        ffn: usize,
        patch: usize,
        base_img: usize,
        proj_dim: usize,
        with_mmproj: bool,
        llm_dim: usize,
    ) -> Vec<u8> {
        let grid = base_img / patch;
        let num_pos = grid * grid + 1;
        let ones = vec![1.0_f32; hidden];
        let zeros = vec![0.0_f32; hidden];
        let mut owned = Vec::new();
        push_f32(
            &mut owned,
            "v.patch_embd.weight",
            vec![hidden, 3, patch, patch],
            &pattern_weights(hidden * 3 * patch * patch, 1),
        );
        push_f32(
            &mut owned,
            "v.patch_embd.bias",
            vec![hidden],
            &pattern_weights(hidden, 2),
        );
        push_f32(
            &mut owned,
            "v.class_embd",
            vec![hidden],
            &pattern_weights(hidden, 3),
        );
        push_f32(
            &mut owned,
            "v.position_embd.weight",
            vec![num_pos, hidden],
            &pattern_weights(num_pos * hidden, 4),
        );
        push_f32(&mut owned, "v.pre_ln.weight", vec![hidden], &ones);
        push_f32(&mut owned, "v.pre_ln.bias", vec![hidden], &zeros);
        push_clip_layer0(&mut owned, hidden, ffn);
        push_f32(&mut owned, "v.post_ln.weight", vec![hidden], &ones);
        push_f32(&mut owned, "v.post_ln.bias", vec![hidden], &zeros);
        push_f32(
            &mut owned,
            "visual_projection.weight",
            vec![proj_dim, hidden],
            &pattern_weights(proj_dim * hidden, 17),
        );
        if with_mmproj {
            push_f32(
                &mut owned,
                "mm.0.weight",
                vec![llm_dim, hidden],
                &pattern_weights(llm_dim * hidden, 18),
            );
            push_f32(
                &mut owned,
                "mm.0.bias",
                vec![llm_dim],
                &pattern_weights(llm_dim, 19),
            );
            push_f32(
                &mut owned,
                "mm.2.weight",
                vec![llm_dim, llm_dim],
                &pattern_weights(llm_dim * llm_dim, 20),
            );
            push_f32(
                &mut owned,
                "mm.2.bias",
                vec![llm_dim],
                &pattern_weights(llm_dim, 21),
            );
        }

        let refs: Vec<(&str, QuantType, &[usize], &[u8])> = owned
            .iter()
            .map(|(n, s, b)| (*n, QuantType::F32, s.as_slice(), b.as_slice()))
            .collect();
        let img_str = base_img.to_string();
        let patch_str = patch.to_string();
        let hidden_str = hidden.to_string();
        create_synthetic_safetensors(
            &[
                ("general.architecture", "clip"),
                ("clip.vision.image_size", &img_str),
                ("clip.vision.patch_size", &patch_str),
                ("clip.vision.embedding_length", &hidden_str),
            ],
            &refs,
        )
    }

    fn push_siglip_map_head(
        owned: &mut Vec<(&'static str, Vec<usize>, Vec<u8>)>,
        hidden: usize,
        ffn: usize,
    ) {
        let ones = vec![1.0_f32; hidden];
        let zeros = vec![0.0_f32; hidden];
        push_f32(
            owned,
            "vision_model.head.probe",
            vec![1, 1, hidden],
            &pattern_weights(hidden, 42),
        );
        push_f32(
            owned,
            "vision_model.head.attention.in_proj_weight",
            vec![3 * hidden, hidden],
            &pattern_weights(3 * hidden * hidden, 43),
        );
        push_f32(
            owned,
            "vision_model.head.attention.in_proj_bias",
            vec![3 * hidden],
            &pattern_weights(3 * hidden, 44),
        );
        push_f32(
            owned,
            "vision_model.head.attention.out_proj.weight",
            vec![hidden, hidden],
            &pattern_weights(hidden * hidden, 45),
        );
        push_f32(
            owned,
            "vision_model.head.attention.out_proj.bias",
            vec![hidden],
            &pattern_weights(hidden, 46),
        );
        push_f32(
            owned,
            "vision_model.head.layernorm.weight",
            vec![hidden],
            &ones,
        );
        push_f32(
            owned,
            "vision_model.head.layernorm.bias",
            vec![hidden],
            &zeros,
        );
        push_f32(
            owned,
            "vision_model.head.mlp.fc1.weight",
            vec![ffn, hidden],
            &pattern_weights(ffn * hidden, 47),
        );
        push_f32(
            owned,
            "vision_model.head.mlp.fc1.bias",
            vec![ffn],
            &pattern_weights(ffn, 48),
        );
        push_f32(
            owned,
            "vision_model.head.mlp.fc2.weight",
            vec![hidden, ffn],
            &pattern_weights(hidden * ffn, 49),
        );
        push_f32(
            owned,
            "vision_model.head.mlp.fc2.bias",
            vec![hidden],
            &pattern_weights(hidden, 50),
        );
    }

    fn push_siglip_layer0(
        owned: &mut Vec<(&'static str, Vec<usize>, Vec<u8>)>,
        hidden: usize,
        ffn: usize,
    ) {
        let ones = vec![1.0_f32; hidden];
        let zeros = vec![0.0_f32; hidden];
        push_f32(
            owned,
            "vision_model.encoder.layers.0.self_attn.qkv.weight",
            vec![3 * hidden, hidden],
            &pattern_weights(3 * hidden * hidden, 34),
        );
        push_f32(
            owned,
            "vision_model.encoder.layers.0.self_attn.qkv.bias",
            vec![3 * hidden],
            &pattern_weights(3 * hidden, 35),
        );
        push_f32(
            owned,
            "vision_model.encoder.layers.0.self_attn.out_proj.weight",
            vec![hidden, hidden],
            &pattern_weights(hidden * hidden, 36),
        );
        push_f32(
            owned,
            "vision_model.encoder.layers.0.self_attn.out_proj.bias",
            vec![hidden],
            &pattern_weights(hidden, 37),
        );
        push_f32(
            owned,
            "vision_model.encoder.layers.0.layer_norm1.weight",
            vec![hidden],
            &ones,
        );
        push_f32(
            owned,
            "vision_model.encoder.layers.0.layer_norm1.bias",
            vec![hidden],
            &zeros,
        );
        push_f32(
            owned,
            "vision_model.encoder.layers.0.mlp.fc1.weight",
            vec![ffn, hidden],
            &pattern_weights(ffn * hidden, 38),
        );
        push_f32(
            owned,
            "vision_model.encoder.layers.0.mlp.fc1.bias",
            vec![ffn],
            &pattern_weights(ffn, 39),
        );
        push_f32(
            owned,
            "vision_model.encoder.layers.0.mlp.fc2.weight",
            vec![hidden, ffn],
            &pattern_weights(hidden * ffn, 40),
        );
        push_f32(
            owned,
            "vision_model.encoder.layers.0.mlp.fc2.bias",
            vec![hidden],
            &pattern_weights(hidden, 41),
        );
        push_f32(
            owned,
            "vision_model.encoder.layers.0.layer_norm2.weight",
            vec![hidden],
            &ones,
        );
        push_f32(
            owned,
            "vision_model.encoder.layers.0.layer_norm2.bias",
            vec![hidden],
            &zeros,
        );
    }

    fn build_siglip_safetensors_bytes(
        hidden: usize,
        ffn: usize,
        patch: usize,
        base_img: usize,
        with_map_head: bool,
        with_pixel_shuffle_mmproj: bool,
        llm_dim: usize,
    ) -> Vec<u8> {
        let grid = base_img / patch;
        let num_pos = grid * grid;
        let ones = vec![1.0_f32; hidden];
        let zeros = vec![0.0_f32; hidden];
        let mut owned = Vec::new();
        push_f32(
            &mut owned,
            "vision_model.embeddings.patch_embedding.weight",
            vec![hidden, 3, patch, patch],
            &pattern_weights(hidden * 3 * patch * patch, 31),
        );
        push_f32(
            &mut owned,
            "vision_model.embeddings.patch_embedding.bias",
            vec![hidden],
            &pattern_weights(hidden, 32),
        );
        push_f32(
            &mut owned,
            "vision_model.embeddings.position_embedding.weight",
            vec![num_pos, hidden],
            &pattern_weights(num_pos * hidden, 33),
        );
        push_siglip_layer0(&mut owned, hidden, ffn);
        push_f32(
            &mut owned,
            "vision_model.post_layernorm.weight",
            vec![hidden],
            &ones,
        );
        push_f32(
            &mut owned,
            "vision_model.post_layernorm.bias",
            vec![hidden],
            &zeros,
        );

        if with_map_head {
            push_siglip_map_head(&mut owned, hidden, ffn);
        }
        if with_pixel_shuffle_mmproj {
            let merged_in = hidden * 4;
            let mlp_hidden = llm_dim * 2;
            push_f32(&mut owned, "merger.ln_q.weight", vec![hidden], &ones);
            push_f32(
                &mut owned,
                "multi_modal_projector.linear_1.weight",
                vec![mlp_hidden, merged_in],
                &pattern_weights(mlp_hidden * merged_in, 51),
            );
            push_f32(
                &mut owned,
                "multi_modal_projector.linear_1.bias",
                vec![mlp_hidden],
                &pattern_weights(mlp_hidden, 52),
            );
            push_f32(
                &mut owned,
                "multi_modal_projector.linear_2.weight",
                vec![llm_dim, mlp_hidden],
                &pattern_weights(llm_dim * mlp_hidden, 53),
            );
            push_f32(
                &mut owned,
                "multi_modal_projector.linear_2.bias",
                vec![llm_dim],
                &pattern_weights(llm_dim, 54),
            );
        }

        let refs: Vec<(&str, QuantType, &[usize], &[u8])> = owned
            .iter()
            .map(|(n, s, b)| (*n, QuantType::F32, s.as_slice(), b.as_slice()))
            .collect();
        let img_str = base_img.to_string();
        let patch_str = patch.to_string();
        let hidden_str = hidden.to_string();
        create_synthetic_safetensors(
            &[
                ("general.architecture", "siglip"),
                ("clip.vision.image_size", &img_str),
                ("clip.vision.patch_size", &patch_str),
                ("clip.vision.embedding_length", &hidden_str),
            ],
            &refs,
        )
    }

    struct FilteredWeights<'a>(&'a SafeTensorsFile<'a>, &'static str);
    impl ModelWeights for FilteredWeights<'_> {
        fn tensor(&self, name: &str) -> Result<jero_core::TensorView<'_>, K3Error> {
            if name == self.1 {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!("injected missing tensor '{name}'"),
                });
            }
            self.0.tensor(name)
        }

        fn tensor_count(&self) -> usize {
            self.0.tensor_count()
        }

        fn total_weight_bytes(&self) -> usize {
            self.0.total_weight_bytes()
        }
    }

    #[test]
    fn crate_exports_and_metadata_match() {
        assert_eq!(CRATE_NAME, "jero-arch-vision");
        assert_eq!(
            CRATE_LICENSE,
            "LGPL-3.0-only WITH LGPL-3.0-linking-exception"
        );
        assert_ne!(crate_description(), "");

        let clip = ClipVisionRecipe::new();
        assert_eq!(clip.architecture_id(), "clip");
        assert!(!clip.requires_kv_cache());
        assert!(clip.supports_quant(QuantType::F32));

        let siglip = SiglipVisionRecipe::new();
        assert_eq!(siglip.architecture_id(), "siglip");
        assert!(!siglip.requires_kv_cache());
    }

    #[test]
    fn clip_standalone_vision_embedding_and_pos_interp_succeed() {
        let bytes = build_clip_safetensors_bytes(8, 16, 2, 4, 6, false, 10);
        let st = SafeTensorsFile::from_bytes(&bytes).expect("parse clip safetensors");
        let recipe = ClipVisionRecipe::from_safetensors(&st)
            .with_dims(4, 2, 1, 8, 16, 2)
            .with_projection_dim(6);
        assert_eq!(recipe.image_size, 4);
        assert_eq!(recipe.patch_size, 2);

        // Evaluate at 8x8 (4x4 patches != 2x2 trained pos_embd grid -> exercises interpolate_2d_pos_embed_f32).
        let pixels = pattern_weights(3 * 8 * 8, 99);
        let mut logits = [0.0_f32; 8];
        let mut kv_cache = [];
        let mut emb_out = [0.0_f32; 6];
        let mut ctx = ExecutionContext::new(&[], 64, &mut logits, &mut kv_cache)
            .expect("execution context")
            .with_embedding_output(&mut emb_out, EmbeddingPooling::Cls, None)
            .expect("embedding output")
            .with_vision_input(&pixels, 3, 8, 8)
            .expect("vision input");

        recipe
            .forward_classify(&mut ctx, &st)
            .expect("clip forward_classify");
        assert_eq!(
            ctx.position(),
            0,
            "ViT must not consume autoregressive KV positions"
        );

        let norm_sq: f32 = emb_out.iter().map(|&v| v * v).sum();
        assert!(
            (norm_sq - 1.0_f32).abs() < 1e-4,
            "expected L2-normalized CLIP embed, got norm_sq={norm_sq}"
        );
        assert!(
            logits[..6]
                .iter()
                .zip(emb_out.iter())
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
    }

    #[test]
    fn siglip_map_head_and_mean_pool_embeddings_succeed() {
        let bytes = build_siglip_safetensors_bytes(8, 16, 2, 4, true, false, 10);
        let st = SafeTensorsFile::from_bytes(&bytes).expect("parse siglip safetensors");
        let recipe = SiglipVisionRecipe::from_safetensors(&st).with_dims(4, 2, 1, 8, 16, 2);

        let pixels = pattern_weights(3 * 4 * 4, 71);
        let mut emb_map = [0.0_f32; 8];
        let mut logits = [];
        let mut kv_cache = [];
        let mut ctx_map = ExecutionContext::new(&[], 64, &mut logits, &mut kv_cache)
            .expect("execution context")
            .with_embedding_output(&mut emb_map, EmbeddingPooling::Mean, None)
            .expect("embedding output")
            .with_vision_input(&pixels, 3, 4, 4)
            .expect("vision input");
        recipe
            .prefill_step(&mut ctx_map, &st)
            .expect("siglip MAP pool");
        let map_norm_sq: f32 = emb_map.iter().map(|&v| v * v).sum();
        assert!((map_norm_sq - 1.0_f32).abs() < 1e-4);

        let recipe_mean = recipe.clone().with_map_head(false);
        let mut emb_mean = [0.0_f32; 8];
        let mut ctx_mean = ExecutionContext::new(&[], 64, &mut logits, &mut kv_cache)
            .expect("execution context")
            .with_embedding_output(&mut emb_mean, EmbeddingPooling::Mean, Some(4))
            .expect("embedding output")
            .with_vision_input(&pixels, 3, 4, 4)
            .expect("vision input");
        recipe_mean
            .forward_classify(&mut ctx_mean, &st)
            .expect("siglip mean Matryoshka pool");
        let mrl_norm_sq: f32 = emb_mean[..4].iter().map(|&v| v * v).sum();
        assert!((mrl_norm_sq - 1.0_f32).abs() < 1e-4);
        assert!(emb_mean[4..].iter().all(|&v| v == 0.0_f32));
    }

    #[test]
    fn mmproj_bridge_clip_mlp_and_siglip_pixel_shuffle_2x2_succeed() {
        // 1. CLIP + 2-layer MLP mmproj (no spatial merge)
        let clip_bytes = build_clip_safetensors_bytes(8, 16, 2, 4, 6, true, 12);
        let clip_st = SafeTensorsFile::from_bytes(&clip_bytes).expect("parse clip mmproj");
        let clip_bridge = VlmMmprojBridge::new_llava_clip(
            ClipVisionRecipe::from_safetensors(&clip_st).with_dims(4, 2, 1, 8, 16, 2),
            12,
        );
        let pixels = pattern_weights(3 * 4 * 4, 15);
        let (soft_clip, tokens_clip, dim_clip) = clip_bridge
            .encode_image_to_soft_prompt_with_dims(&clip_st, &pixels, 3, 4, 4)
            .expect("clip mmproj soft prompt");
        assert_eq!(tokens_clip, 4);
        assert_eq!(dim_clip, 12);
        assert_eq!(soft_clip.len(), 48);
        assert!(soft_clip.iter().all(|v| v.is_finite()));

        let soft_clip_flat = clip_bridge
            .encode_image_to_soft_prompt(&clip_st, &pixels, 3, 4, 4)
            .expect("flat helper matches");
        assert_eq!(soft_clip_flat.len(), soft_clip.len());
        assert!(
            soft_clip_flat
                .iter()
                .zip(soft_clip.iter())
                .all(|(a, b)| (a - b).abs() < 1e-5)
        );

        // 2. SigLIP + 2x2 pixel-shuffle + 2-layer MLP (4x4 spatial grid -> 2x2 = 4 merged tokens)
        let sig_bytes = build_siglip_safetensors_bytes(8, 16, 2, 8, true, true, 10);
        let sig_st = SafeTensorsFile::from_bytes(&sig_bytes).expect("parse siglip mmproj");
        let sig_bridge = VlmMmprojBridge::new_gemma3_siglip(
            SiglipVisionRecipe::from_safetensors(&sig_st).with_dims(8, 2, 1, 8, 16, 2),
            10,
        );
        let rgb_hwc = vec![0.4_f32; 6 * 6 * 3];
        let (soft_sig, tokens_sig, dim_sig) = sig_bridge
            .encode_rgb_hwc_to_soft_prompt(&sig_st, &rgb_hwc, 6, 6, 8, 8)
            .expect("siglip 2x2 pixel-shuffle soft prompt");
        assert_eq!(tokens_sig, 4);
        assert_eq!(dim_sig, 10);
        assert_eq!(soft_sig.len(), 40);
        assert!(soft_sig.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gguf_hydration_and_mmproj_execution_succeed() {
        let kvs = [
            ("clip.vision.image_size", GgufValue::Uint32(4)),
            ("clip.vision.patch_size", GgufValue::Uint32(2)),
            ("clip.vision.embedding_length", GgufValue::Uint32(8)),
            ("clip.vision.feed_forward_length", GgufValue::Uint32(16)),
            ("clip.vision.block_count", GgufValue::Uint32(1)),
            ("clip.vision.attention.head_count", GgufValue::Uint32(2)),
            ("clip.projection_dim", GgufValue::Uint32(10)),
            ("clip.projector_type", GgufValue::String("mlp")),
            ("clip.vision.use_gelu", GgufValue::Bool(false)),
        ];
        let specs: [(&str, QuantType, &[usize]); 21] = [
            ("v.patch_embd.weight", QuantType::F32, &[2, 2, 3, 8]),
            ("v.patch_embd.bias", QuantType::F32, &[8]),
            ("v.class_embd", QuantType::F32, &[8]),
            ("v.position_embd.weight", QuantType::F32, &[8, 5]),
            ("v.blk.0.attn_q.weight", QuantType::F32, &[8, 8]),
            ("v.blk.0.attn_k.weight", QuantType::F32, &[8, 8]),
            ("v.blk.0.attn_v.weight", QuantType::F32, &[8, 8]),
            ("v.blk.0.attn_out.weight", QuantType::F32, &[8, 8]),
            ("v.blk.0.ln1.weight", QuantType::F32, &[8]),
            ("v.blk.0.ffn_up.weight", QuantType::F32, &[8, 16]),
            ("v.blk.0.ffn_down.weight", QuantType::F32, &[16, 8]),
            ("v.blk.0.ln2.weight", QuantType::F32, &[8]),
            ("v.post_ln.weight", QuantType::F32, &[8]),
            ("mm.0.weight", QuantType::F32, &[8, 16]),
            ("mm.0.bias", QuantType::F32, &[16]),
            ("mm.2.weight", QuantType::F32, &[16, 10]),
            ("mm.2.bias", QuantType::F32, &[10]),
            ("v.blk.0.ln1.bias", QuantType::F32, &[8]),
            ("v.blk.0.ln2.bias", QuantType::F32, &[8]),
            ("v.post_ln.bias", QuantType::F32, &[8]),
            ("v.pre_ln.weight", QuantType::F32, &[8]),
        ];
        let mut gguf_bytes = create_synthetic_gguf_v3_with_kvs("clip", 1, &kvs, &specs);
        let patch_ranges: Vec<(usize, usize, bool)> = {
            let parsed = GgufFile::parse(&gguf_bytes).expect("parse layout");
            specs
                .iter()
                .map(|&(name, _, _)| {
                    let tv = parsed.tensor(name).expect("tensor present");
                    let is_norm_w = name.ends_with(".weight") && name.contains("ln");
                    (tv.offset(), tv.byte_len(), is_norm_w)
                })
                .collect()
        };
        for (idx, (off, len, is_norm_w)) in patch_ranges.into_iter().enumerate() {
            let elems = len / size_of::<f32>();
            let vals = if is_norm_w {
                vec![1.0_f32; elems]
            } else {
                pattern_weights(elems, idx + 1)
            };
            gguf_bytes[off..off + len].copy_from_slice(&f32_to_le_bytes(&vals));
        }
        let gguf = GgufFile::parse(&gguf_bytes).expect("parse clip gguf");

        let bridge = VlmMmprojBridge::from_gguf(&gguf);
        assert!(matches!(bridge.encoder, VisionEncoderConfig::Clip(_)));
        assert_eq!(bridge.llm_embed_dim, Some(10));

        let pixels = pattern_weights(3 * 4 * 4, 23);
        let (embeds, n_tok, dim) = bridge
            .encode_image_to_soft_prompt_with_dims(&gguf, &pixels, 3, 4, 4)
            .expect("gguf clip mmproj");
        assert_eq!(n_tok, 4);
        assert_eq!(dim, 10);
        assert_eq!(embeds.len(), 40);
        assert!(embeds.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fail_closed_negative_paths_reject_invalid_inputs_and_missing_tensors() {
        let clip_bytes = build_clip_safetensors_bytes(8, 16, 2, 4, 6, true, 12);
        let clip_st = SafeTensorsFile::from_bytes(&clip_bytes).expect("parse clip safetensors");
        let recipe = ClipVisionRecipe::from_safetensors(&clip_st).with_dims(4, 2, 1, 8, 16, 2);
        let valid_pixels = pattern_weights(3 * 4 * 4, 5);

        // 1. Missing vision_input on ExecutionContext
        let mut logits = [0.0_f32; 8];
        let mut kv_cache = [];
        let mut emb = [0.0_f32; 8];
        let mut missing_vis_ctx = ExecutionContext::new(&[], 64, &mut logits, &mut kv_cache)
            .expect("execution context")
            .with_embedding_output(&mut emb, EmbeddingPooling::Cls, None)
            .expect("embedding output");
        assert!(matches!(
            recipe.forward_classify(&mut missing_vis_ctx, &clip_st),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 2. Non-finite pixel input
        let mut bad_pixels = valid_pixels.clone();
        bad_pixels[3] = f32::NAN;
        assert!(matches!(
            recipe.encode_patch_tokens(&clip_st, &bad_pixels, 3, 4, 4),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 3. Odd spatial grid rejected by 2x2 pixel-shuffle bridge (6x6 image with patch 2 -> 3x3 patches)
        let sig_bytes = build_siglip_safetensors_bytes(8, 16, 2, 6, false, true, 10);
        let sig_st = SafeTensorsFile::from_bytes(&sig_bytes).expect("parse siglip");
        let odd_bridge = VlmMmprojBridge::new_qwen2_vl(
            SiglipVisionRecipe::from_safetensors(&sig_st).with_dims(6, 2, 1, 8, 16, 2),
            10,
        );
        let odd_pixels = pattern_weights(3 * 6 * 6, 9);
        assert!(matches!(
            odd_bridge.encode_image_to_soft_prompt(&sig_st, &odd_pixels, 3, 6, 6),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 4. LLM embedding dim mismatch fails closed
        let bad_dim_bridge = VlmMmprojBridge::new_llava_clip(recipe.clone(), 99);
        assert!(matches!(
            bad_dim_bridge.encode_image_to_soft_prompt(&clip_st, &valid_pixels, 3, 4, 4),
            Err(K3Error::TensorBoundsError { .. } | K3Error::CapabilityMismatch { .. })
        ));

        // 5. Autoregressive decode_step must fail closed on non-KV ViT recipe
        let mut ok_ctx = ExecutionContext::new(&[], 64, &mut logits, &mut kv_cache)
            .expect("execution context")
            .with_embedding_output(&mut emb, EmbeddingPooling::Cls, None)
            .expect("embedding output")
            .with_vision_input(&valid_pixels, 3, 4, 4)
            .expect("vision input");
        assert!(matches!(
            recipe.decode_step(&mut ok_ctx, &clip_st),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // 6. Missing projector weights fails closed
        let no_mm_bytes = build_clip_safetensors_bytes(8, 16, 2, 4, 6, false, 12);
        let no_mm_st = SafeTensorsFile::from_bytes(&no_mm_bytes).expect("parse no-mm clip");
        let bridge_no_mm = VlmMmprojBridge::new_clip(recipe);
        assert!(matches!(
            bridge_no_mm.encode_image_to_soft_prompt(&no_mm_st, &valid_pixels, 3, 4, 4),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 7. fc1 without fc2 must fail closed (never fall back silently)
        assert!(matches!(
            bridge_no_mm.encode_image_to_soft_prompt(
                &FilteredWeights(&clip_st, "mm.2.weight"),
                &valid_pixels,
                3,
                4,
                4
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn validate_model_enforces_adr045_checks_for_clip_and_siglip() {
        let clip_bytes = build_clip_safetensors_bytes(8, 16, 2, 4, 6, false, 12);
        let clip_st = SafeTensorsFile::from_bytes(&clip_bytes).expect("parse clip st");
        let clip = ClipVisionRecipe::from_safetensors(&clip_st)
            .with_dims(4, 2, 1, 8, 16, 2)
            .with_projection_dim(6);
        clip.validate_model(clip_st.metadata(), &clip_st)
            .expect("valid CLIP passes validate_model");

        let sig_bytes = build_siglip_safetensors_bytes(8, 16, 2, 4, true, false, 10);
        let sig_st = SafeTensorsFile::from_bytes(&sig_bytes).expect("parse siglip st");
        let sig = SiglipVisionRecipe::from_safetensors(&sig_st).with_dims(4, 2, 1, 8, 16, 2);
        sig.validate_model(sig_st.metadata(), &sig_st)
            .expect("valid SigLIP passes validate_model");

        // Unrelated LLM / tabular metadata fails closed on both recipes.
        let llama_gguf_bytes = create_synthetic_gguf_v3_with_kvs("llama", 1, &[], &[]);
        let llama_gguf = GgufFile::parse(&llama_gguf_bytes).expect("parse llama gguf");
        assert!(matches!(
            clip.validate_model(llama_gguf.metadata(), &clip_st),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        assert!(matches!(
            sig.validate_model(llama_gguf.metadata(), &sig_st),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // SigLIP accepts GGUF `general.architecture = "clip"` metadata (mmproj-*.gguf compatibility),
        // while CLIP rejects `siglip` metadata.
        sig.validate_model(clip_st.metadata(), &sig_st)
            .expect("SigLIP accepts clip metadata for mmproj compatibility");
        assert!(matches!(
            clip.validate_model(sig_st.metadata(), &clip_st),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // Invalid recipe hyperparameters fail closed.
        let mut bad_clip = clip.clone();
        bad_clip.hidden_dim = 0;
        assert!(matches!(
            bad_clip.validate_model(clip_st.metadata(), &clip_st),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        assert!(matches!(
            clip.clone()
                .with_projection_dim(0)
                .validate_model(clip_st.metadata(), &clip_st),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // Wrong visual_projection dim fails closed.
        assert!(matches!(
            clip.clone()
                .with_projection_dim(12)
                .validate_model(clip_st.metadata(), &clip_st),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // Corrupted rms_norm_eps (negative / NaN) does not poison hydrated norm_eps.
        let bad_eps_gguf_bytes = create_synthetic_gguf_v3_with_kvs(
            "clip",
            1,
            &[(
                "clip.attention.layer_norm_rms_epsilon",
                GgufValue::Float32(-1.0),
            )],
            &[],
        );
        let bad_eps_gguf = GgufFile::parse(&bad_eps_gguf_bytes).expect("parse bad eps gguf");
        let hydrated = ClipVisionRecipe::from_metadata(bad_eps_gguf.metadata());
        assert!(hydrated.norm_eps.is_finite() && hydrated.norm_eps > 0.0);

        // Missing required tensors when `weights.tensor_count() > 0` fail closed.
        for missing_name in [
            "v.patch_embd.weight",
            "v.class_embd",
            "v.position_embd.weight",
            "v.blk.0.attn_q.weight",
            "v.post_ln.weight",
        ] {
            let filtered = FilteredWeights(&clip_st, missing_name);
            assert!(
                matches!(
                    clip.validate_model(clip_st.metadata(), &filtered),
                    Err(K3Error::TensorBoundsError { .. })
                ),
                "expected validate_model to fail when '{missing_name}' is missing"
            );
        }
        assert!(matches!(
            sig.validate_model(
                sig_st.metadata(),
                &FilteredWeights(&sig_st, "vision_model.post_layernorm.weight")
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }
}
