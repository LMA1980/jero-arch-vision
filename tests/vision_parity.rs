//! End-to-end integration parity and fail-closed tests for `jero-arch-vision`.

use jero_arch_vision::{
    ClipVisionRecipe, SiglipVisionRecipe, VisionEncoderConfig, VlmMmprojBridge,
};
use jero_core::safetensors::test_helpers::create_synthetic_safetensors;
use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use jero_core::{
    ArchRecipe, EmbeddingPooling, ExecutionContext, GgufFile, GgufValue, K3Error, ModelWeights,
    QuantType, SafeTensorsFile,
};

fn f32_to_le_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(size_of_val(vals));
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn pattern_vec(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let rem = i16::try_from((idx * 13 + seed * 29) % 19).expect("rem fits i16");
            (f32::from(rem) - 9.0) * 0.04_f32
        })
        .collect()
}

fn push_st_f32(
    entries: &mut Vec<(&'static str, Vec<usize>, Vec<u8>)>,
    name: &'static str,
    shape: Vec<usize>,
    vals: &[f32],
) {
    entries.push((name, shape, f32_to_le_bytes(vals)));
}

fn build_gguf_with_payloads(
    arch: &str,
    n_layers: usize,
    kvs: &[(&str, GgufValue<'_>)],
    tensors: &[(&'static str, Vec<usize>, &[f32])],
) -> Vec<u8> {
    let specs: Vec<(&str, QuantType, &[usize])> = tensors
        .iter()
        .map(|(name, shape, _)| (*name, QuantType::F32, shape.as_slice()))
        .collect();
    let mut buf = create_synthetic_gguf_v3_with_kvs(arch, n_layers, kvs, &specs);
    let ranges: Vec<(usize, usize)> = {
        let parsed = GgufFile::parse(&buf).expect("parse empty synthetic gguf");
        tensors
            .iter()
            .map(|(name, _, _)| {
                let tv = parsed.tensor(name).expect("synthetic tensor exists");
                (tv.offset(), tv.byte_len())
            })
            .collect()
    };
    for ((off, len), (_, _, vals)) in ranges.into_iter().zip(tensors.iter()) {
        let bytes = f32_to_le_bytes(vals);
        assert_eq!(bytes.len(), len);
        buf[off..off + len].copy_from_slice(&bytes);
    }
    buf
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "slice length mismatch in parity comparison"
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[derive(Clone)]
struct SharedClipWeights {
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    cls_w: Vec<f32>,
    pos_w: Vec<f32>,
    ones: Vec<f32>,
    zeros: Vec<f32>,
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    o_w: Vec<f32>,
    o_b: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    vis_proj_w: Vec<f32>,
    mm0_w: Vec<f32>,
    mm0_b: Vec<f32>,
    mm2_w: Vec<f32>,
    mm2_b: Vec<f32>,
}

impl SharedClipWeights {
    fn new(
        hidden: usize,
        ffn: usize,
        patch: usize,
        grid: usize,
        vis_proj: usize,
        llm_dim: usize,
    ) -> Self {
        let num_pos = grid * grid + 1;
        Self {
            patch_w: pattern_vec(hidden * 3 * patch * patch, 1),
            patch_b: pattern_vec(hidden, 2),
            cls_w: pattern_vec(hidden, 3),
            pos_w: pattern_vec(num_pos * hidden, 4),
            ones: vec![1.0_f32; hidden],
            zeros: vec![0.0_f32; hidden],
            q_w: pattern_vec(hidden * hidden, 5),
            q_b: pattern_vec(hidden, 6),
            k_w: pattern_vec(hidden * hidden, 7),
            k_b: pattern_vec(hidden, 8),
            v_w: pattern_vec(hidden * hidden, 9),
            v_b: pattern_vec(hidden, 10),
            o_w: pattern_vec(hidden * hidden, 11),
            o_b: pattern_vec(hidden, 12),
            fc1_w: pattern_vec(ffn * hidden, 13),
            fc1_b: pattern_vec(ffn, 14),
            fc2_w: pattern_vec(hidden * ffn, 15),
            fc2_b: pattern_vec(hidden, 16),
            vis_proj_w: pattern_vec(vis_proj * hidden, 17),
            mm0_w: pattern_vec(llm_dim * hidden, 18),
            mm0_b: pattern_vec(llm_dim, 19),
            mm2_w: pattern_vec(llm_dim * llm_dim, 20),
            mm2_b: pattern_vec(llm_dim, 21),
        }
    }

    fn to_gguf_bytes(
        &self,
        hidden: usize,
        ffn: usize,
        patch: usize,
        base_img: usize,
        vis_proj: usize,
        llm_dim: usize,
    ) -> Vec<u8> {
        self.to_gguf_bytes_with_heads(hidden, ffn, patch, base_img, vis_proj, llm_dim, 2)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "test builder takes vision geometry, projection widths, and head count"
    )]
    fn to_gguf_bytes_with_heads(
        &self,
        hidden: usize,
        ffn: usize,
        patch: usize,
        base_img: usize,
        vis_proj: usize,
        llm_dim: usize,
        n_heads: usize,
    ) -> Vec<u8> {
        let num_pos = (base_img / patch) * (base_img / patch) + 1;
        let kvs = [
            (
                "clip.vision.image_size",
                GgufValue::Uint32(u32::try_from(base_img).expect("u32")),
            ),
            (
                "clip.vision.patch_size",
                GgufValue::Uint32(u32::try_from(patch).expect("u32")),
            ),
            (
                "clip.vision.embedding_length",
                GgufValue::Uint32(u32::try_from(hidden).expect("u32")),
            ),
            (
                "clip.vision.feed_forward_length",
                GgufValue::Uint32(u32::try_from(ffn).expect("u32")),
            ),
            ("clip.vision.block_count", GgufValue::Uint32(1)),
            (
                "clip.vision.attention.head_count",
                GgufValue::Uint32(u32::try_from(n_heads).expect("u32")),
            ),
            (
                "clip.vision.projection_dim",
                GgufValue::Uint32(u32::try_from(vis_proj).expect("u32")),
            ),
            (
                "clip.projection_dim",
                GgufValue::Uint32(u32::try_from(llm_dim).expect("u32")),
            ),
            ("clip.projector_type", GgufValue::String("mlp")),
            ("clip.vision.use_gelu", GgufValue::Bool(false)),
        ];
        let tensors: Vec<(&'static str, Vec<usize>, &[f32])> = vec![
            (
                "v.patch_embd.weight",
                vec![patch, patch, 3, hidden],
                &self.patch_w,
            ),
            ("v.patch_embd.bias", vec![hidden], &self.patch_b),
            ("v.class_embd", vec![hidden], &self.cls_w),
            ("v.position_embd.weight", vec![hidden, num_pos], &self.pos_w),
            ("v.pre_ln.weight", vec![hidden], &self.ones),
            ("v.pre_ln.bias", vec![hidden], &self.zeros),
            ("v.blk.0.attn_q.weight", vec![hidden, hidden], &self.q_w),
            ("v.blk.0.attn_q.bias", vec![hidden], &self.q_b),
            ("v.blk.0.attn_k.weight", vec![hidden, hidden], &self.k_w),
            ("v.blk.0.attn_k.bias", vec![hidden], &self.k_b),
            ("v.blk.0.attn_v.weight", vec![hidden, hidden], &self.v_w),
            ("v.blk.0.attn_v.bias", vec![hidden], &self.v_b),
            ("v.blk.0.attn_out.weight", vec![hidden, hidden], &self.o_w),
            ("v.blk.0.attn_out.bias", vec![hidden], &self.o_b),
            ("v.blk.0.ln1.weight", vec![hidden], &self.ones),
            ("v.blk.0.ln1.bias", vec![hidden], &self.zeros),
            ("v.blk.0.ffn_up.weight", vec![hidden, ffn], &self.fc1_w),
            ("v.blk.0.ffn_up.bias", vec![ffn], &self.fc1_b),
            ("v.blk.0.ffn_down.weight", vec![ffn, hidden], &self.fc2_w),
            ("v.blk.0.ffn_down.bias", vec![hidden], &self.fc2_b),
            ("v.blk.0.ln2.weight", vec![hidden], &self.ones),
            ("v.blk.0.ln2.bias", vec![hidden], &self.zeros),
            ("v.post_ln.weight", vec![hidden], &self.ones),
            ("v.post_ln.bias", vec![hidden], &self.zeros),
            (
                "visual_projection.weight",
                vec![hidden, vis_proj],
                &self.vis_proj_w,
            ),
            ("mm.0.weight", vec![hidden, llm_dim], &self.mm0_w),
            ("mm.0.bias", vec![llm_dim], &self.mm0_b),
            ("mm.2.weight", vec![llm_dim, llm_dim], &self.mm2_w),
            ("mm.2.bias", vec![llm_dim], &self.mm2_b),
        ];
        build_gguf_with_payloads("clip", 1, &kvs, &tensors)
    }

    fn push_st_layer0(
        &self,
        owned: &mut Vec<(&'static str, Vec<usize>, Vec<u8>)>,
        hidden: usize,
        ffn: usize,
    ) {
        for (name, shape, vals) in [
            (
                "vision_model.encoder.layers.0.self_attn.q_proj.weight",
                vec![hidden, hidden],
                &self.q_w,
            ),
            (
                "vision_model.encoder.layers.0.self_attn.q_proj.bias",
                vec![hidden],
                &self.q_b,
            ),
            (
                "vision_model.encoder.layers.0.self_attn.k_proj.weight",
                vec![hidden, hidden],
                &self.k_w,
            ),
            (
                "vision_model.encoder.layers.0.self_attn.k_proj.bias",
                vec![hidden],
                &self.k_b,
            ),
            (
                "vision_model.encoder.layers.0.self_attn.v_proj.weight",
                vec![hidden, hidden],
                &self.v_w,
            ),
            (
                "vision_model.encoder.layers.0.self_attn.v_proj.bias",
                vec![hidden],
                &self.v_b,
            ),
            (
                "vision_model.encoder.layers.0.self_attn.out_proj.weight",
                vec![hidden, hidden],
                &self.o_w,
            ),
            (
                "vision_model.encoder.layers.0.self_attn.out_proj.bias",
                vec![hidden],
                &self.o_b,
            ),
            (
                "vision_model.encoder.layers.0.layer_norm1.weight",
                vec![hidden],
                &self.ones,
            ),
            (
                "vision_model.encoder.layers.0.layer_norm1.bias",
                vec![hidden],
                &self.zeros,
            ),
            (
                "vision_model.encoder.layers.0.mlp.fc1.weight",
                vec![ffn, hidden],
                &self.fc1_w,
            ),
            (
                "vision_model.encoder.layers.0.mlp.fc1.bias",
                vec![ffn],
                &self.fc1_b,
            ),
            (
                "vision_model.encoder.layers.0.mlp.fc2.weight",
                vec![hidden, ffn],
                &self.fc2_w,
            ),
            (
                "vision_model.encoder.layers.0.mlp.fc2.bias",
                vec![hidden],
                &self.fc2_b,
            ),
            (
                "vision_model.encoder.layers.0.layer_norm2.weight",
                vec![hidden],
                &self.ones,
            ),
            (
                "vision_model.encoder.layers.0.layer_norm2.bias",
                vec![hidden],
                &self.zeros,
            ),
        ] {
            push_st_f32(owned, name, shape, vals);
        }
    }

    fn to_safetensors_bytes(
        &self,
        hidden: usize,
        ffn: usize,
        patch: usize,
        base_img: usize,
        vis_proj: usize,
        llm_dim: usize,
    ) -> Vec<u8> {
        let num_pos = (base_img / patch) * (base_img / patch) + 1;
        let mut owned = Vec::new();
        for (name, shape, vals) in [
            (
                "vision_model.embeddings.patch_embedding.weight",
                vec![hidden, 3, patch, patch],
                &self.patch_w,
            ),
            (
                "vision_model.embeddings.patch_embedding.bias",
                vec![hidden],
                &self.patch_b,
            ),
            (
                "vision_model.embeddings.class_embedding",
                vec![hidden],
                &self.cls_w,
            ),
            (
                "vision_model.embeddings.position_embedding.weight",
                vec![num_pos, hidden],
                &self.pos_w,
            ),
            ("vision_model.pre_layrnorm.weight", vec![hidden], &self.ones),
            ("vision_model.pre_layrnorm.bias", vec![hidden], &self.zeros),
        ] {
            push_st_f32(&mut owned, name, shape, vals);
        }
        self.push_st_layer0(&mut owned, hidden, ffn);
        for (name, shape, vals) in [
            (
                "vision_model.post_layernorm.weight",
                vec![hidden],
                &self.ones,
            ),
            (
                "vision_model.post_layernorm.bias",
                vec![hidden],
                &self.zeros,
            ),
            (
                "visual_projection.weight",
                vec![vis_proj, hidden],
                &self.vis_proj_w,
            ),
            (
                "multi_modal_projector.linear_1.weight",
                vec![llm_dim, hidden],
                &self.mm0_w,
            ),
            (
                "multi_modal_projector.linear_1.bias",
                vec![llm_dim],
                &self.mm0_b,
            ),
            (
                "multi_modal_projector.linear_2.weight",
                vec![llm_dim, llm_dim],
                &self.mm2_w,
            ),
            (
                "multi_modal_projector.linear_2.bias",
                vec![llm_dim],
                &self.mm2_b,
            ),
        ] {
            push_st_f32(&mut owned, name, shape, vals);
        }

        let refs: Vec<(&str, QuantType, &[usize], &[u8])> = owned
            .iter()
            .map(|(n, s, b)| (*n, QuantType::F32, s.as_slice(), b.as_slice()))
            .collect();
        let img_str = base_img.to_string();
        let patch_str = patch.to_string();
        let hidden_str = hidden.to_string();
        let llm_str = llm_dim.to_string();
        let vis_str = vis_proj.to_string();
        create_synthetic_safetensors(
            &[
                ("general.architecture", "clip"),
                ("image_size", &img_str),
                ("patch_size", &patch_str),
                ("hidden_size", &hidden_str),
                ("num_hidden_layers", "1"),
                ("num_attention_heads", "2"),
                ("projection_dim", &vis_str),
                ("text_config.hidden_size", &llm_str),
                ("hidden_act", "quick_gelu"),
            ],
            &refs,
        )
    }
}

struct SharedSiglipWeights {
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    pos_w: Vec<f32>,
    ones: Vec<f32>,
    zeros: Vec<f32>,
    qkv_w: Vec<f32>,
    qkv_b: Vec<f32>,
    o_w: Vec<f32>,
    o_b: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    probe_w: Vec<f32>,
    head_in_w: Vec<f32>,
    head_in_b: Vec<f32>,
    head_o_w: Vec<f32>,
    head_o_b: Vec<f32>,
    head_fc1_w: Vec<f32>,
    head_fc1_b: Vec<f32>,
    head_fc2_w: Vec<f32>,
    head_fc2_b: Vec<f32>,
    mm_norm_w: Vec<f32>,
    mm0_w: Vec<f32>,
    mm0_b: Vec<f32>,
    mm2_w: Vec<f32>,
    mm2_b: Vec<f32>,
}

impl SharedSiglipWeights {
    fn new(
        hidden: usize,
        ffn: usize,
        patch: usize,
        grid: usize,
        mlp_hidden: usize,
        llm_dim: usize,
    ) -> Self {
        let num_pos = grid * grid;
        let merged_in = hidden * 4;
        Self {
            patch_w: pattern_vec(hidden * 3 * patch * patch, 101),
            patch_b: pattern_vec(hidden, 102),
            pos_w: pattern_vec(num_pos * hidden, 103),
            ones: vec![1.0_f32; hidden],
            zeros: vec![0.0_f32; hidden],
            qkv_w: pattern_vec(3 * hidden * hidden, 104),
            qkv_b: pattern_vec(3 * hidden, 105),
            o_w: pattern_vec(hidden * hidden, 106),
            o_b: pattern_vec(hidden, 107),
            fc1_w: pattern_vec(ffn * hidden, 108),
            fc1_b: pattern_vec(ffn, 109),
            fc2_w: pattern_vec(hidden * ffn, 110),
            fc2_b: pattern_vec(hidden, 111),
            probe_w: pattern_vec(hidden, 112),
            head_in_w: pattern_vec(3 * hidden * hidden, 113),
            head_in_b: pattern_vec(3 * hidden, 114),
            head_o_w: pattern_vec(hidden * hidden, 115),
            head_o_b: pattern_vec(hidden, 116),
            head_fc1_w: pattern_vec(ffn * hidden, 117),
            head_fc1_b: pattern_vec(ffn, 118),
            head_fc2_w: pattern_vec(hidden * ffn, 119),
            head_fc2_b: pattern_vec(hidden, 120),
            mm_norm_w: pattern_vec(hidden, 121)
                .into_iter()
                .map(|v| v.abs() + 0.5)
                .collect(),
            mm0_w: pattern_vec(mlp_hidden * merged_in, 122),
            mm0_b: pattern_vec(mlp_hidden, 123),
            mm2_w: pattern_vec(llm_dim * mlp_hidden, 124),
            mm2_b: pattern_vec(llm_dim, 125),
        }
    }

    fn build_gguf_bytes(
        &self,
        hidden: usize,
        ffn: usize,
        patch: usize,
        grid: usize,
        mlp_hidden: usize,
        llm_dim: usize,
    ) -> Vec<u8> {
        let num_pos = grid * grid;
        let merged_in = hidden * 4;
        let gguf_kvs = [
            ("clip.vision.image_size", GgufValue::Uint32(8)),
            ("clip.vision.patch_size", GgufValue::Uint32(2)),
            ("clip.vision.embedding_length", GgufValue::Uint32(8)),
            ("clip.vision.feed_forward_length", GgufValue::Uint32(16)),
            ("clip.vision.block_count", GgufValue::Uint32(1)),
            ("clip.vision.attention.head_count", GgufValue::Uint32(2)),
            ("clip.projector_type", GgufValue::String("gemma3")),
            ("clip.vision.spatial_merge_size", GgufValue::Uint32(2)),
            ("clip.projection_dim", GgufValue::Uint32(10)),
        ];
        let gguf_tensors: Vec<(&'static str, Vec<usize>, &[f32])> = vec![
            (
                "v.patch_embd.weight",
                vec![patch, patch, 3, hidden],
                &self.patch_w,
            ),
            ("v.patch_embd.bias", vec![hidden], &self.patch_b),
            ("v.position_embd.weight", vec![hidden, num_pos], &self.pos_w),
            (
                "v.blk.0.attn_qkv.weight",
                vec![hidden, 3 * hidden],
                &self.qkv_w,
            ),
            ("v.blk.0.attn_qkv.bias", vec![3 * hidden], &self.qkv_b),
            ("v.blk.0.attn_out.weight", vec![hidden, hidden], &self.o_w),
            ("v.blk.0.attn_out.bias", vec![hidden], &self.o_b),
            ("v.blk.0.ln1.weight", vec![hidden], &self.ones),
            ("v.blk.0.ln1.bias", vec![hidden], &self.zeros),
            ("v.blk.0.ffn_up.weight", vec![hidden, ffn], &self.fc1_w),
            ("v.blk.0.ffn_up.bias", vec![ffn], &self.fc1_b),
            ("v.blk.0.ffn_down.weight", vec![ffn, hidden], &self.fc2_w),
            ("v.blk.0.ffn_down.bias", vec![hidden], &self.fc2_b),
            ("v.blk.0.ln2.weight", vec![hidden], &self.ones),
            ("v.blk.0.ln2.bias", vec![hidden], &self.zeros),
            ("v.post_ln.weight", vec![hidden], &self.ones),
            ("v.post_ln.bias", vec![hidden], &self.zeros),
            ("v.head.probe", vec![hidden, 1, 1], &self.probe_w),
            (
                "v.head.attention.in_proj_weight",
                vec![hidden, 3 * hidden],
                &self.head_in_w,
            ),
            (
                "v.head.attention.in_proj_bias",
                vec![3 * hidden],
                &self.head_in_b,
            ),
            (
                "v.head.attention.out_proj.weight",
                vec![hidden, hidden],
                &self.head_o_w,
            ),
            (
                "v.head.attention.out_proj.bias",
                vec![hidden],
                &self.head_o_b,
            ),
            ("v.head.layernorm.weight", vec![hidden], &self.ones),
            ("v.head.layernorm.bias", vec![hidden], &self.zeros),
            ("v.head.mlp.fc1.weight", vec![hidden, ffn], &self.head_fc1_w),
            ("v.head.mlp.fc1.bias", vec![ffn], &self.head_fc1_b),
            ("v.head.mlp.fc2.weight", vec![ffn, hidden], &self.head_fc2_w),
            ("v.head.mlp.fc2.bias", vec![hidden], &self.head_fc2_b),
            ("mm.input_norm.weight", vec![hidden], &self.mm_norm_w),
            ("mm.0.weight", vec![merged_in, mlp_hidden], &self.mm0_w),
            ("mm.0.bias", vec![mlp_hidden], &self.mm0_b),
            ("mm.2.weight", vec![mlp_hidden, llm_dim], &self.mm2_w),
            ("mm.2.bias", vec![llm_dim], &self.mm2_b),
        ];
        build_gguf_with_payloads("siglip", 1, &gguf_kvs, &gguf_tensors)
    }

    fn push_st_head_and_merger(
        &self,
        st_owned: &mut Vec<(&'static str, Vec<usize>, Vec<u8>)>,
        hidden: usize,
        ffn: usize,
        mlp_hidden: usize,
        llm_dim: usize,
    ) {
        let merged_in = hidden * 4;
        for (name, shape, vals) in [
            ("vision_model.head.probe", vec![1, 1, hidden], &self.probe_w),
            (
                "vision_model.head.attention.in_proj_weight",
                vec![3 * hidden, hidden],
                &self.head_in_w,
            ),
            (
                "vision_model.head.attention.in_proj_bias",
                vec![3 * hidden],
                &self.head_in_b,
            ),
            (
                "vision_model.head.attention.out_proj.weight",
                vec![hidden, hidden],
                &self.head_o_w,
            ),
            (
                "vision_model.head.attention.out_proj.bias",
                vec![hidden],
                &self.head_o_b,
            ),
            (
                "vision_model.head.layernorm.weight",
                vec![hidden],
                &self.ones,
            ),
            (
                "vision_model.head.layernorm.bias",
                vec![hidden],
                &self.zeros,
            ),
            (
                "vision_model.head.mlp.fc1.weight",
                vec![ffn, hidden],
                &self.head_fc1_w,
            ),
            (
                "vision_model.head.mlp.fc1.bias",
                vec![ffn],
                &self.head_fc1_b,
            ),
            (
                "vision_model.head.mlp.fc2.weight",
                vec![hidden, ffn],
                &self.head_fc2_w,
            ),
            (
                "vision_model.head.mlp.fc2.bias",
                vec![hidden],
                &self.head_fc2_b,
            ),
            ("merger.ln_q.weight", vec![hidden], &self.mm_norm_w),
            (
                "merger.mlp.0.weight",
                vec![mlp_hidden, merged_in],
                &self.mm0_w,
            ),
            ("merger.mlp.0.bias", vec![mlp_hidden], &self.mm0_b),
            (
                "merger.mlp.2.weight",
                vec![llm_dim, mlp_hidden],
                &self.mm2_w,
            ),
            ("merger.mlp.2.bias", vec![llm_dim], &self.mm2_b),
        ] {
            push_st_f32(st_owned, name, shape, vals);
        }
    }

    fn push_st_layer0(
        &self,
        st_owned: &mut Vec<(&'static str, Vec<usize>, Vec<u8>)>,
        hidden: usize,
        ffn: usize,
    ) {
        for (name, shape, vals) in [
            (
                "vision_model.encoder.layers.0.self_attn.qkv.weight",
                vec![3 * hidden, hidden],
                &self.qkv_w,
            ),
            (
                "vision_model.encoder.layers.0.self_attn.qkv.bias",
                vec![3 * hidden],
                &self.qkv_b,
            ),
            (
                "vision_model.encoder.layers.0.self_attn.out_proj.weight",
                vec![hidden, hidden],
                &self.o_w,
            ),
            (
                "vision_model.encoder.layers.0.self_attn.out_proj.bias",
                vec![hidden],
                &self.o_b,
            ),
            (
                "vision_model.encoder.layers.0.layer_norm1.weight",
                vec![hidden],
                &self.ones,
            ),
            (
                "vision_model.encoder.layers.0.layer_norm1.bias",
                vec![hidden],
                &self.zeros,
            ),
            (
                "vision_model.encoder.layers.0.mlp.fc1.weight",
                vec![ffn, hidden],
                &self.fc1_w,
            ),
            (
                "vision_model.encoder.layers.0.mlp.fc1.bias",
                vec![ffn],
                &self.fc1_b,
            ),
            (
                "vision_model.encoder.layers.0.mlp.fc2.weight",
                vec![hidden, ffn],
                &self.fc2_w,
            ),
            (
                "vision_model.encoder.layers.0.mlp.fc2.bias",
                vec![hidden],
                &self.fc2_b,
            ),
            (
                "vision_model.encoder.layers.0.layer_norm2.weight",
                vec![hidden],
                &self.ones,
            ),
            (
                "vision_model.encoder.layers.0.layer_norm2.bias",
                vec![hidden],
                &self.zeros,
            ),
        ] {
            push_st_f32(st_owned, name, shape, vals);
        }
    }

    fn build_st_bytes(
        &self,
        hidden: usize,
        ffn: usize,
        patch: usize,
        grid: usize,
        mlp_hidden: usize,
        llm_dim: usize,
    ) -> Vec<u8> {
        let num_pos = grid * grid;
        let mut st_owned = Vec::new();
        for (name, shape, vals) in [
            (
                "vision_model.embeddings.patch_embedding.weight",
                vec![hidden, 3, patch, patch],
                &self.patch_w,
            ),
            (
                "vision_model.embeddings.patch_embedding.bias",
                vec![hidden],
                &self.patch_b,
            ),
            (
                "vision_model.embeddings.position_embedding.weight",
                vec![num_pos, hidden],
                &self.pos_w,
            ),
            (
                "vision_model.post_layernorm.weight",
                vec![hidden],
                &self.ones,
            ),
            (
                "vision_model.post_layernorm.bias",
                vec![hidden],
                &self.zeros,
            ),
        ] {
            push_st_f32(&mut st_owned, name, shape, vals);
        }
        self.push_st_layer0(&mut st_owned, hidden, ffn);
        self.push_st_head_and_merger(&mut st_owned, hidden, ffn, mlp_hidden, llm_dim);
        let st_refs: Vec<(&str, QuantType, &[usize], &[u8])> = st_owned
            .iter()
            .map(|(n, s, b)| (*n, QuantType::F32, s.as_slice(), b.as_slice()))
            .collect();
        create_synthetic_safetensors(
            &[
                ("general.architecture", "siglip"),
                ("projector_type", "qwen2_vl"),
                ("image_size", "8"),
                ("patch_size", "2"),
                ("hidden_size", "8"),
                ("num_hidden_layers", "1"),
                ("num_attention_heads", "2"),
                ("text_config.hidden_size", "10"),
            ],
            &st_refs,
        )
    }

    fn build_pair(
        &self,
        hidden: usize,
        ffn: usize,
        patch: usize,
        grid: usize,
        mlp_hidden: usize,
        llm_dim: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        (
            self.build_gguf_bytes(hidden, ffn, patch, grid, mlp_hidden, llm_dim),
            self.build_st_bytes(hidden, ffn, patch, grid, mlp_hidden, llm_dim),
        )
    }
}

#[test]
fn clip_gguf_vs_safetensors_forward_and_mmproj_parity() {
    let weights = SharedClipWeights::new(8, 16, 2, 2, 6, 12);
    let gguf_bytes = weights.to_gguf_bytes(8, 16, 2, 4, 6, 12);
    let st_bytes = weights.to_safetensors_bytes(8, 16, 2, 4, 6, 12);

    let gguf = GgufFile::parse(&gguf_bytes).expect("parse clip gguf");
    let st = SafeTensorsFile::from_bytes(&st_bytes).expect("parse clip safetensors");

    let clip_gguf = ClipVisionRecipe::from_gguf(&gguf);
    let clip_st = ClipVisionRecipe::from_safetensors(&st);
    assert_eq!(clip_gguf.hidden_dim, 8);
    assert_eq!(clip_st.hidden_dim, 8);
    assert!(clip_gguf.use_quick_gelu);
    assert!(clip_st.use_quick_gelu);

    for (img_side, mrl) in [(4usize, None), (8usize, Some(4usize))] {
        let pixels = pattern_vec(3 * img_side * img_side, 77 + img_side);
        let mut out_gguf = [0.0_f32; 6];
        let mut out_st = [0.0_f32; 6];
        let mut logits_g = [];
        let mut logits_s = [];
        let mut kv_g = [];
        let mut kv_s = [];

        let mut ctx_g = ExecutionContext::new(&[], 8, &mut logits_g, &mut kv_g)
            .expect("ctx gguf")
            .with_embedding_output(&mut out_gguf, EmbeddingPooling::Cls, mrl)
            .expect("emb gguf")
            .with_vision_input(&pixels, 3, img_side, img_side)
            .expect("vis gguf");
        let mut ctx_s = ExecutionContext::new(&[], 8, &mut logits_s, &mut kv_s)
            .expect("ctx st")
            .with_embedding_output(&mut out_st, EmbeddingPooling::Cls, mrl)
            .expect("emb st")
            .with_vision_input(&pixels, 3, img_side, img_side)
            .expect("vis st");

        clip_gguf
            .forward_classify(&mut ctx_g, &gguf)
            .expect("clip gguf forward");
        clip_st
            .forward_classify(&mut ctx_s, &st)
            .expect("clip st forward");
        assert!(
            max_abs_diff(&out_gguf, &out_st) < 1e-5,
            "CLIP GGUF vs SafeTensors embedding diverged at {img_side}x{img_side}"
        );
    }

    let bridge_gguf = VlmMmprojBridge::from_gguf(&gguf);
    let bridge_st = VlmMmprojBridge::from_safetensors(&st);
    assert!(matches!(bridge_gguf.encoder, VisionEncoderConfig::Clip(_)));
    assert!(matches!(bridge_st.encoder, VisionEncoderConfig::Clip(_)));

    let pixels_8x8 = pattern_vec(3 * 8 * 8, 91);
    let (sp_g, n_g, d_g) = bridge_gguf
        .encode_image_to_soft_prompt_with_dims(&gguf, &pixels_8x8, 3, 8, 8)
        .expect("clip mmproj gguf");
    let (sp_s, n_s, d_s) = bridge_st
        .encode_image_to_soft_prompt_with_dims(&st, &pixels_8x8, 3, 8, 8)
        .expect("clip mmproj st");
    assert_eq!((n_g, d_g), (16, 12));
    assert_eq!((n_s, d_s), (16, 12));
    assert!(max_abs_diff(&sp_g, &sp_s) < 1e-5);
}

#[test]
fn siglip_gguf_vs_safetensors_map_and_2x2_pixel_shuffle_parity() {
    let shared = SharedSiglipWeights::new(8, 16, 2, 4, 14, 10);
    let (gguf_bytes, st_bytes) = shared.build_pair(8, 16, 2, 4, 14, 10);

    let gguf = GgufFile::parse(&gguf_bytes).expect("parse siglip gguf");
    let st = SafeTensorsFile::from_bytes(&st_bytes).expect("parse siglip safetensors");

    let sig_g = SiglipVisionRecipe::from_gguf(&gguf);
    let sig_s = SiglipVisionRecipe::from_safetensors(&st);
    let pixels = pattern_vec(3 * 8 * 8, 55);

    let emb_g = sig_g
        .encode_image_embedding(&gguf, &pixels, 3, 8, 8, EmbeddingPooling::Mean, None)
        .expect("siglip gguf MAP emb");
    let emb_s = sig_s
        .encode_image_embedding(&st, &pixels, 3, 8, 8, EmbeddingPooling::Mean, None)
        .expect("siglip st MAP emb");
    assert!(max_abs_diff(&emb_g, &emb_s) < 1e-5);

    let bridge_g = VlmMmprojBridge::from_gguf(&gguf);
    let bridge_s = VlmMmprojBridge::from_safetensors(&st);
    assert!(bridge_g.merge_2x2_pixel_shuffle);
    assert!(bridge_s.merge_2x2_pixel_shuffle);

    let (sp_g, n_g, d_g) = bridge_g
        .encode_image_to_soft_prompt_with_dims(&gguf, &pixels, 3, 8, 8)
        .expect("gguf pixel shuffle mmproj");
    let (sp_s, n_s, d_s) = bridge_s
        .encode_image_to_soft_prompt_with_dims(&st, &pixels, 3, 8, 8)
        .expect("st pixel shuffle mmproj");
    let (sp_split, n_split, d_split) = bridge_s
        .encode_image_to_soft_prompt_split(&st, &gguf, &pixels, 3, 8, 8)
        .expect("split st backbone + gguf mmproj");
    assert_eq!((n_g, d_g), (4, 10));
    assert_eq!((n_s, d_s), (4, 10));
    assert_eq!((n_split, d_split), (4, 10));
    assert!(max_abs_diff(&sp_g, &sp_s) < 1e-5);
    assert!(max_abs_diff(&sp_g, &sp_split) < 1e-5);
}

struct NanProj<'a>(&'a SafeTensorsFile<'a>, Vec<u8>);

impl ModelWeights for NanProj<'_> {
    fn tensor(&self, name: &str) -> Result<jero_core::TensorView<'_>, K3Error> {
        if name == "multi_modal_projector.mm_input_projection_weight" {
            return jero_core::TensorView::from_bytes(
                "multi_modal_projector.mm_input_projection_weight",
                QuantType::F32,
                &[6, 16],
                0,
                &self.1,
            );
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
fn single_layer_linear_projector_and_corrupted_weight_rejection_work() {
    let mut owned = Vec::new();
    let hidden = 4usize;
    let ffn = 8usize;
    let patch = 2usize;
    let ones = vec![1.0_f32; hidden];
    push_st_f32(
        &mut owned,
        "v.patch_embd.weight",
        vec![hidden, 3, patch, patch],
        &pattern_vec(hidden * 3 * patch * patch, 1),
    );
    push_st_f32(
        &mut owned,
        "v.position_embd.weight",
        vec![4, hidden],
        &pattern_vec(4 * hidden, 2),
    );
    push_st_f32(
        &mut owned,
        "v.blk.0.attn_qkv.weight",
        vec![3 * hidden, hidden],
        &pattern_vec(3 * hidden * hidden, 3),
    );
    push_st_f32(
        &mut owned,
        "v.blk.0.attn_out.weight",
        vec![hidden, hidden],
        &pattern_vec(hidden * hidden, 4),
    );
    push_st_f32(&mut owned, "v.blk.0.ln1.weight", vec![hidden], &ones);
    push_st_f32(
        &mut owned,
        "v.blk.0.ffn_up.weight",
        vec![ffn, hidden],
        &pattern_vec(ffn * hidden, 5),
    );
    push_st_f32(
        &mut owned,
        "v.blk.0.ffn_down.weight",
        vec![hidden, ffn],
        &pattern_vec(hidden * ffn, 6),
    );
    push_st_f32(&mut owned, "v.blk.0.ln2.weight", vec![hidden], &ones);
    push_st_f32(&mut owned, "v.post_ln.weight", vec![hidden], &ones);
    push_st_f32(
        &mut owned,
        "multi_modal_projector.mm_input_projection_weight",
        vec![6, hidden * 4],
        &pattern_vec(6 * hidden * 4, 7),
    );

    let refs: Vec<(&str, QuantType, &[usize], &[u8])> = owned
        .iter()
        .map(|(n, s, b)| (*n, QuantType::F32, s.as_slice(), b.as_slice()))
        .collect();
    let st_bytes = create_synthetic_safetensors(&[("general.architecture", "gemma3")], &refs);
    let st = SafeTensorsFile::from_bytes(&st_bytes).expect("parse linear mmproj");

    let sig = SiglipVisionRecipe::new().with_dims(4, 2, 1, 4, 8, 2);
    let bridge = VlmMmprojBridge::new_gemma3_siglip(sig, 6);
    let pixels = pattern_vec(3 * 4 * 4, 9);
    let (embeds, n_tok, llm_dim) = bridge
        .encode_image_to_soft_prompt_with_dims(&st, &pixels, 3, 4, 4)
        .expect("linear mmproj succeeds");
    assert_eq!(n_tok, 1);
    assert_eq!(llm_dim, 6);
    assert_eq!(embeds.len(), 6);
    assert!(embeds.iter().all(|v| v.is_finite()));

    // Verify NaN weight tensor triggers fail-closed TensorBoundsError.
    let mut bad_proj = vec![0.0_f32; 6 * 16];
    bad_proj[7] = f32::NAN;
    let nan_weights = NanProj(&st, f32_to_le_bytes(&bad_proj));
    assert!(matches!(
        bridge.encode_image_to_soft_prompt(&nan_weights, &pixels, 3, 4, 4),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = i32::try_from((bits >> 23) & 0xFF).expect("8-bit exp fits i32");
    let frac = bits & 0x007F_FFFF;
    if exp == 0 {
        return sign;
    }
    let unbiased = exp - 127;
    if unbiased < -14 {
        return sign;
    }
    if unbiased > 15 {
        return sign | 0x7C00;
    }
    let h_exp = u16::try_from(unbiased + 15).expect("f16 exp fits u16");
    let h_frac = u16::try_from(frac >> 13).expect("f16 frac fits u16");
    sign | (h_exp << 10) | h_frac
}

fn f32_to_f16_le_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
    }
    out
}

fn quantize_roundtrip_f16(vals: &[f32]) -> Vec<f32> {
    vals.iter()
        .map(|&v| {
            let bits = f32_to_f16_bits(v);
            let sign = u32::from((bits >> 15) & 1);
            let exp = u32::from((bits >> 10) & 0x1F);
            let mant = u32::from(bits & 0x03FF);
            if exp == 0 {
                f32::from_bits(sign << 31)
            } else {
                f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
            }
        })
        .collect()
}

#[expect(
    clippy::too_many_lines,
    reason = "parity test compares full upstream Gemma-3 GGUF (F16 + swapped FFN) against HF SafeTensors"
)]
#[test]
fn gemma3_gguf_mmproj_swapped_ffn_and_avg_pool_parity_with_safetensors() {
    let hidden = 8usize;
    let ffn = 16usize;
    let patch = 2usize;
    let img_side = 16usize;
    let grid = img_side / patch; // 8x8 -> 4x4 avg-pool -> 2x2 = 4 tokens
    let num_pos = grid * grid;
    let llm_dim = 12usize;

    let patch_w = pattern_vec(hidden * 3 * patch * patch, 201);
    let patch_b = pattern_vec(hidden, 202);
    let pos_w = pattern_vec(num_pos * hidden, 203);
    let ones = vec![1.0_f32; hidden];
    let zeros = vec![0.0_f32; hidden];
    let q_w = quantize_roundtrip_f16(&pattern_vec(hidden * hidden, 204));
    let q_b = pattern_vec(hidden, 205);
    let k_w = quantize_roundtrip_f16(&pattern_vec(hidden * hidden, 206));
    let k_b = pattern_vec(hidden, 207);
    let v_w = quantize_roundtrip_f16(&pattern_vec(hidden * hidden, 208));
    let v_b = pattern_vec(hidden, 209);
    let o_w = quantize_roundtrip_f16(&pattern_vec(hidden * hidden, 210));
    let o_b = pattern_vec(hidden, 211);
    let fc1_w = quantize_roundtrip_f16(&pattern_vec(ffn * hidden, 212));
    let fc1_b = pattern_vec(ffn, 213);
    let fc2_w = quantize_roundtrip_f16(&pattern_vec(hidden * ffn, 214));
    let fc2_b = pattern_vec(hidden, 215);
    let soft_norm_w: Vec<f32> = pattern_vec(hidden, 216)
        .into_iter()
        .map(|v| 1.0 + v * 0.5)
        .collect();
    // Transposed [in=hidden, out=llm_dim] row-major weight matrix
    let mm_proj_w = quantize_roundtrip_f16(&pattern_vec(hidden * llm_dim, 217));

    // 1. Upstream llama.cpp Gemma-3 GGUF (F16 linear/proj weights, swapped ffn_up/ffn_down,
    //    mm.soft_emb_norm.weight, transposed mm.input_projection.weight [llm_dim, hidden], no spatial_merge_size KV)
    let kvs = [
        ("clip.vision.image_size", GgufValue::Uint32(16)),
        ("clip.vision.patch_size", GgufValue::Uint32(2)),
        ("clip.vision.embedding_length", GgufValue::Uint32(8)),
        ("clip.vision.feed_forward_length", GgufValue::Uint32(16)),
        ("clip.vision.projection_dim", GgufValue::Uint32(12)),
        ("clip.vision.block_count", GgufValue::Uint32(1)),
        ("clip.vision.attention.head_count", GgufValue::Uint32(2)),
        (
            "clip.vision.attention.layer_norm_epsilon",
            GgufValue::Float32(1e-6),
        ),
        ("clip.projector_type", GgufValue::String("gemma3")),
        ("clip.use_gelu", GgufValue::Bool(true)),
    ];
    let gguf_specs: Vec<(&str, QuantType, Vec<usize>, Vec<u8>)> = vec![
        (
            "v.patch_embd.weight",
            QuantType::F32,
            vec![patch, patch, 3, hidden],
            f32_to_le_bytes(&patch_w),
        ),
        (
            "v.patch_embd.bias",
            QuantType::F32,
            vec![hidden],
            f32_to_le_bytes(&patch_b),
        ),
        (
            "v.position_embd.weight",
            QuantType::F32,
            vec![hidden, num_pos],
            f32_to_le_bytes(&pos_w),
        ),
        (
            "v.blk.0.attn_q.weight",
            QuantType::F16,
            vec![hidden, hidden],
            f32_to_f16_le_bytes(&q_w),
        ),
        (
            "v.blk.0.attn_q.bias",
            QuantType::F32,
            vec![hidden],
            f32_to_le_bytes(&q_b),
        ),
        (
            "v.blk.0.attn_k.weight",
            QuantType::F16,
            vec![hidden, hidden],
            f32_to_f16_le_bytes(&k_w),
        ),
        (
            "v.blk.0.attn_k.bias",
            QuantType::F32,
            vec![hidden],
            f32_to_le_bytes(&k_b),
        ),
        (
            "v.blk.0.attn_v.weight",
            QuantType::F16,
            vec![hidden, hidden],
            f32_to_f16_le_bytes(&v_w),
        ),
        (
            "v.blk.0.attn_v.bias",
            QuantType::F32,
            vec![hidden],
            f32_to_le_bytes(&v_b),
        ),
        (
            "v.blk.0.attn_out.weight",
            QuantType::F16,
            vec![hidden, hidden],
            f32_to_f16_le_bytes(&o_w),
        ),
        (
            "v.blk.0.attn_out.bias",
            QuantType::F32,
            vec![hidden],
            f32_to_le_bytes(&o_b),
        ),
        (
            "v.blk.0.ln1.weight",
            QuantType::F32,
            vec![hidden],
            f32_to_le_bytes(&ones),
        ),
        (
            "v.blk.0.ln1.bias",
            QuantType::F32,
            vec![hidden],
            f32_to_le_bytes(&zeros),
        ),
        // Swapped FFN per upstream clip.cpp Gemma-3 export:
        // ffn_up holds [ffn, hidden] (fc2) with [hidden] bias; ffn_down holds [hidden, ffn] (fc1) with [ffn] bias
        (
            "v.blk.0.ffn_up.weight",
            QuantType::F16,
            vec![ffn, hidden],
            f32_to_f16_le_bytes(&fc2_w),
        ),
        (
            "v.blk.0.ffn_up.bias",
            QuantType::F32,
            vec![hidden],
            f32_to_le_bytes(&fc2_b),
        ),
        (
            "v.blk.0.ffn_down.weight",
            QuantType::F16,
            vec![hidden, ffn],
            f32_to_f16_le_bytes(&fc1_w),
        ),
        (
            "v.blk.0.ffn_down.bias",
            QuantType::F32,
            vec![ffn],
            f32_to_le_bytes(&fc1_b),
        ),
        (
            "v.blk.0.ln2.weight",
            QuantType::F32,
            vec![hidden],
            f32_to_le_bytes(&ones),
        ),
        (
            "v.blk.0.ln2.bias",
            QuantType::F32,
            vec![hidden],
            f32_to_le_bytes(&zeros),
        ),
        (
            "v.post_ln.weight",
            QuantType::F32,
            vec![hidden],
            f32_to_le_bytes(&ones),
        ),
        (
            "v.post_ln.bias",
            QuantType::F32,
            vec![hidden],
            f32_to_le_bytes(&zeros),
        ),
        (
            "mm.soft_emb_norm.weight",
            QuantType::F32,
            vec![hidden],
            f32_to_le_bytes(&soft_norm_w),
        ),
        (
            "mm.input_projection.weight",
            QuantType::F16,
            vec![llm_dim, hidden],
            f32_to_f16_le_bytes(&mm_proj_w),
        ),
    ];
    let gguf_schema: Vec<(&str, QuantType, &[usize])> = gguf_specs
        .iter()
        .map(|(n, q, s, _)| (*n, *q, s.as_slice()))
        .collect();
    let mut gguf_bytes = create_synthetic_gguf_v3_with_kvs("clip", 1, &kvs, &gguf_schema);
    let gguf_ranges: Vec<(usize, usize)> = {
        let parsed = GgufFile::parse(&gguf_bytes).expect("parse layout");
        gguf_specs
            .iter()
            .map(|(n, _, _, _)| {
                let tv = parsed.tensor(n).expect("tensor present");
                (tv.offset(), tv.byte_len())
            })
            .collect()
    };
    for ((off, len), (_, _, _, payload)) in gguf_ranges.into_iter().zip(gguf_specs.iter()) {
        assert_eq!(payload.len(), len);
        gguf_bytes[off..off + len].copy_from_slice(payload);
    }
    let gguf = GgufFile::parse(&gguf_bytes).expect("parse gemma3 gguf");

    // 2. Reference HF SafeTensors Gemma-3 layout (unswapped mlp.fc1/fc2, mm_soft_emb_norm, mm_input_projection_weight [hidden, llm_dim])
    let mut st_owned = Vec::new();
    for (name, shape, vals) in [
        (
            "vision_model.embeddings.patch_embedding.weight",
            vec![hidden, 3, patch, patch],
            &patch_w,
        ),
        (
            "vision_model.embeddings.patch_embedding.bias",
            vec![hidden],
            &patch_b,
        ),
        (
            "vision_model.embeddings.position_embedding.weight",
            vec![num_pos, hidden],
            &pos_w,
        ),
        (
            "vision_model.encoder.layers.0.self_attn.q_proj.weight",
            vec![hidden, hidden],
            &q_w,
        ),
        (
            "vision_model.encoder.layers.0.self_attn.q_proj.bias",
            vec![hidden],
            &q_b,
        ),
        (
            "vision_model.encoder.layers.0.self_attn.k_proj.weight",
            vec![hidden, hidden],
            &k_w,
        ),
        (
            "vision_model.encoder.layers.0.self_attn.k_proj.bias",
            vec![hidden],
            &k_b,
        ),
        (
            "vision_model.encoder.layers.0.self_attn.v_proj.weight",
            vec![hidden, hidden],
            &v_w,
        ),
        (
            "vision_model.encoder.layers.0.self_attn.v_proj.bias",
            vec![hidden],
            &v_b,
        ),
        (
            "vision_model.encoder.layers.0.self_attn.out_proj.weight",
            vec![hidden, hidden],
            &o_w,
        ),
        (
            "vision_model.encoder.layers.0.self_attn.out_proj.bias",
            vec![hidden],
            &o_b,
        ),
        (
            "vision_model.encoder.layers.0.layer_norm1.weight",
            vec![hidden],
            &ones,
        ),
        (
            "vision_model.encoder.layers.0.layer_norm1.bias",
            vec![hidden],
            &zeros,
        ),
        (
            "vision_model.encoder.layers.0.mlp.fc1.weight",
            vec![ffn, hidden],
            &fc1_w,
        ),
        (
            "vision_model.encoder.layers.0.mlp.fc1.bias",
            vec![ffn],
            &fc1_b,
        ),
        (
            "vision_model.encoder.layers.0.mlp.fc2.weight",
            vec![hidden, ffn],
            &fc2_w,
        ),
        (
            "vision_model.encoder.layers.0.mlp.fc2.bias",
            vec![hidden],
            &fc2_b,
        ),
        (
            "vision_model.encoder.layers.0.layer_norm2.weight",
            vec![hidden],
            &ones,
        ),
        (
            "vision_model.encoder.layers.0.layer_norm2.bias",
            vec![hidden],
            &zeros,
        ),
        ("vision_model.post_layernorm.weight", vec![hidden], &ones),
        ("vision_model.post_layernorm.bias", vec![hidden], &zeros),
        (
            "multi_modal_projector.mm_soft_emb_norm.weight",
            vec![hidden],
            &soft_norm_w,
        ),
        (
            "multi_modal_projector.mm_input_projection_weight",
            vec![hidden, llm_dim],
            &mm_proj_w,
        ),
    ] {
        push_st_f32(&mut st_owned, name, shape, vals);
    }
    let st_refs: Vec<(&str, QuantType, &[usize], &[u8])> = st_owned
        .iter()
        .map(|(n, s, b)| (*n, QuantType::F32, s.as_slice(), b.as_slice()))
        .collect();
    let st_bytes = create_synthetic_safetensors(
        &[
            ("general.architecture", "gemma3"),
            ("num_attention_heads", "2"),
            ("text_config.hidden_size", "12"),
        ],
        &st_refs,
    );
    let st = SafeTensorsFile::from_bytes(&st_bytes).expect("parse gemma3 st");

    let bridge_g = VlmMmprojBridge::from_gguf(&gguf);
    let bridge_s = VlmMmprojBridge::from_safetensors(&st);
    assert_eq!(bridge_g.spatial_avg_pool, Some(4));
    assert_eq!(bridge_s.spatial_avg_pool, Some(4));
    assert!(!bridge_g.merge_2x2_pixel_shuffle);
    assert!(!bridge_s.merge_2x2_pixel_shuffle);
    assert_eq!(bridge_g.llm_embed_dim, Some(12));
    assert_eq!(bridge_s.llm_embed_dim, Some(12));

    let src_hwc = pattern_vec(16 * 16 * 3, 218);
    let (sp_g, n_g, d_g) = bridge_g
        .encode_rgb_hwc_to_soft_prompt(&gguf, &src_hwc, 16, 16, 16, 16)
        .expect("gemma3 gguf forward");
    let (sp_s, n_s, d_s) = bridge_s
        .encode_rgb_hwc_to_soft_prompt(&st, &src_hwc, 16, 16, 16, 16)
        .expect("gemma3 st forward");
    assert_eq!((n_g, d_g), (4, 12));
    assert_eq!((n_s, d_s), (4, 12));
    assert!(
        max_abs_diff(&sp_g, &sp_s) < 1e-4,
        "Gemma-3 GGUF vs SafeTensors soft-prompt diverged: max_diff={}",
        max_abs_diff(&sp_g, &sp_s)
    );
}

fn build_hf_clip_safetensors(
    shared: &SharedClipWeights,
    hidden: usize,
    ffn: usize,
    patch: usize,
    grid: usize,
    vis_proj: usize,
) -> Vec<u8> {
    let mut st_owned = Vec::new();
    let num_pos = grid * grid + 1;
    for (name, shape, vals) in [
        (
            "vision_model.embeddings.patch_embedding.weight",
            vec![hidden, 3, patch, patch],
            &shared.patch_w,
        ),
        (
            "vision_model.embeddings.patch_embedding.bias",
            vec![hidden],
            &shared.patch_b,
        ),
        (
            "vision_model.embeddings.class_embedding",
            vec![hidden],
            &shared.cls_w,
        ),
        (
            "vision_model.embeddings.position_embedding.weight",
            vec![num_pos, hidden],
            &shared.pos_w,
        ),
        (
            "vision_model.pre_layrnorm.weight",
            vec![hidden],
            &shared.ones,
        ),
        (
            "vision_model.pre_layrnorm.bias",
            vec![hidden],
            &shared.zeros,
        ),
    ] {
        push_st_f32(&mut st_owned, name, shape, vals);
    }
    shared.push_st_layer0(&mut st_owned, hidden, ffn);
    for (name, shape, vals) in [
        (
            "vision_model.post_layernorm.weight",
            vec![hidden],
            &shared.ones,
        ),
        (
            "vision_model.post_layernorm.bias",
            vec![hidden],
            &shared.zeros,
        ),
        (
            "visual_projection.weight",
            vec![vis_proj, hidden],
            &shared.vis_proj_w,
        ),
        (
            "text_model.embeddings.token_embedding.weight",
            vec![32, 16],
            &pattern_vec(32 * 16, 301),
        ),
        (
            "text_model.encoder.layers.0.mlp.fc1.weight",
            vec![64, 16],
            &pattern_vec(64 * 16, 302),
        ),
        (
            "text_model.encoder.layers.1.mlp.fc1.weight",
            vec![64, 16],
            &pattern_vec(64 * 16, 303),
        ),
    ] {
        push_st_f32(&mut st_owned, name, shape, vals);
    }
    let st_refs: Vec<(&str, QuantType, &[usize], &[u8])> = st_owned
        .iter()
        .map(|(n, s, b)| (*n, QuantType::F32, s.as_slice(), b.as_slice()))
        .collect();
    create_synthetic_safetensors(&[("format", "pt")], &st_refs)
}

#[test]
fn clip_standalone_hf_safetensors_and_gguf_hydration_and_parity() {
    // Realistic HF CLIPModel SafeTensors: __metadata__ has ONLY format="pt" (no vision dims)
    // and includes decoy text_model.* tensors with a different hidden_size=16 and 2 layers.
    let (hidden, ffn, patch, grid, vis_proj) = (8usize, 16usize, 2usize, 2usize, 6usize);
    let shared = SharedClipWeights::new(hidden, ffn, patch, grid, vis_proj, 12);
    let hf_st_bytes = build_hf_clip_safetensors(&shared, hidden, ffn, patch, grid, vis_proj);
    let hf_st = SafeTensorsFile::from_bytes(&hf_st_bytes).expect("parse HF CLIP SafeTensors");
    let gguf_bytes = shared.to_gguf_bytes_with_heads(hidden, ffn, patch, 4, vis_proj, 12, 4);
    let gguf = GgufFile::parse(&gguf_bytes).expect("parse CLIP GGUF");

    let clip_st = ClipVisionRecipe::from_safetensors(&hf_st);
    let clip_gguf = ClipVisionRecipe::from_gguf(&gguf);
    for (label, recipe) in [("st", &clip_st), ("gguf", &clip_gguf)] {
        assert_eq!(recipe.image_size, 4, "{label} image_size");
        assert_eq!(recipe.patch_size, 2, "{label} patch_size");
        assert_eq!(recipe.num_layers, 1, "{label} num_layers");
        assert_eq!(recipe.hidden_dim, 8, "{label} hidden_dim");
        assert_eq!(recipe.ffn_dim, 16, "{label} ffn_dim");
        assert_eq!(recipe.n_heads, 4, "{label} n_heads");
        assert_eq!(recipe.projection_dim, Some(6), "{label} projection_dim");
        assert!(recipe.use_quick_gelu, "{label} quick_gelu");
    }
    clip_st
        .validate_model(hf_st.metadata(), &hf_st)
        .expect("HF CLIP SafeTensors validate_model");
    clip_gguf
        .validate_model(gguf.metadata(), &gguf)
        .expect("CLIP GGUF validate_model");

    for side in [4usize, 8usize] {
        let pixels = pattern_vec(3 * side * side, 310 + side);
        let emb_st = clip_st
            .encode_image_embedding(&hf_st, &pixels, 3, side, side, EmbeddingPooling::Cls, None)
            .expect("HF CLIP SafeTensors embed");
        let emb_g = clip_gguf
            .encode_image_embedding(&gguf, &pixels, 3, side, side, EmbeddingPooling::Cls, None)
            .expect("CLIP GGUF embed");
        assert_eq!(emb_st.len(), vis_proj);
        assert!(
            max_abs_diff(&emb_st, &emb_g) < 1e-5,
            "CLIP HF SafeTensors vs GGUF diverged at {side}x{side}"
        );
    }
}
