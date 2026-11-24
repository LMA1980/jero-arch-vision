//! VLM multimodal projection (`mmproj`) bridge for turning vision patch embeddings into LLM
//! soft-prompt embeddings.
//!
//! Supports both 2-layer MLP (`mm.0` -> `GELU` -> `mm.2`) and single-layer linear projectors,
//! optional `2x2` spatial pixel-shuffle merging (`pixel_shuffle_merge_2x2_f32` for `Gemma3` /
//! `Qwen2-VL`), optional pre/post normalization, and both `GGUF` (`mmproj-*.gguf`) and
//! `SafeTensors` weight layouts.

use jero_core::{GgufFile, K3Error, ModelMetadata, ModelWeights, SafeTensorsFile, TensorView};

use crate::kernels::{
    MatrixSlice, apply_layernorm_seq, bounds_err, checked_mul, decode_1d_view, decode_optional_1d,
    gelu_inplace_f32, infer_2d_other_dim, linear_forward_seq, lookup_first_tensor,
    lookup_optional_tensor, mismatch_err, pixel_shuffle_merge_2x2_f32, validate_2d_linear_shape,
    validate_finite_slice,
};
use crate::vit::{ClipVisionRecipe, SiglipVisionRecipe, VisionTokens};

const PRE_NORM_WEIGHT_ALIASES: &[&str] = &[
    "mm.input_norm.weight",
    "mmproj.input_norm.weight",
    "merger.ln_q.weight",
    "visual.merger.ln_q.weight",
    "multi_modal_projector.mm_input_norm.weight",
];
const PRE_NORM_BIAS_ALIASES: &[&str] = &[
    "mm.input_norm.bias",
    "mmproj.input_norm.bias",
    "merger.ln_q.bias",
    "visual.merger.ln_q.bias",
    "multi_modal_projector.mm_input_norm.bias",
];

const MLP_FC1_WEIGHT_ALIASES: &[&str] = &[
    "mm.0.weight",
    "mmproj.0.weight",
    "mmproj.fc1.weight",
    "v.mm.0.weight",
    "multi_modal_projector.linear_1.weight",
    "model.mm_projector.0.weight",
    "merger.mlp.0.weight",
    "visual.merger.mlp.0.weight",
];
const MLP_FC1_BIAS_ALIASES: &[&str] = &[
    "mm.0.bias",
    "mmproj.0.bias",
    "mmproj.fc1.bias",
    "v.mm.0.bias",
    "multi_modal_projector.linear_1.bias",
    "model.mm_projector.0.bias",
    "merger.mlp.0.bias",
    "visual.merger.mlp.0.bias",
];

const MLP_FC2_WEIGHT_ALIASES: &[&str] = &[
    "mm.2.weight",
    "mmproj.2.weight",
    "mmproj.fc2.weight",
    "v.mm.2.weight",
    "multi_modal_projector.linear_2.weight",
    "model.mm_projector.2.weight",
    "merger.mlp.2.weight",
    "visual.merger.mlp.2.weight",
];
const MLP_FC2_BIAS_ALIASES: &[&str] = &[
    "mm.2.bias",
    "mmproj.2.bias",
    "mmproj.fc2.bias",
    "v.mm.2.bias",
    "multi_modal_projector.linear_2.bias",
    "model.mm_projector.2.bias",
    "merger.mlp.2.bias",
    "visual.merger.mlp.2.bias",
];

const LINEAR_PROJ_WEIGHT_ALIASES: &[&str] = &[
    "mm.weight",
    "mmproj.weight",
    "mm.input_projection.weight",
    "mm.input_projection_weight",
    "multi_modal_projector.mm_input_projection_weight",
];
const LINEAR_PROJ_BIAS_ALIASES: &[&str] = &["mm.bias", "mmproj.bias", "mm.input_projection.bias"];

const POST_NORM_WEIGHT_ALIASES: &[&str] = &[
    "mm.norm.weight",
    "mmproj.norm.weight",
    "mm.post_norm.weight",
    "multi_modal_projector.norm.weight",
];
const POST_NORM_BIAS_ALIASES: &[&str] = &[
    "mm.norm.bias",
    "mmproj.norm.bias",
    "mm.post_norm.bias",
    "multi_modal_projector.norm.bias",
];

/// Vision backbone recipe variant consumed by [`VlmMmprojBridge`].
#[derive(Debug, Clone, PartialEq)]
pub enum VisionEncoderConfig {
    /// `CLIP` Vision Transformer backbone (`[CLS]` + `QuickGELU` or `GELU_tanh`).
    Clip(ClipVisionRecipe),
    /// `SigLIP` Vision Transformer backbone (`GELU_tanh`, no `[CLS]`, optional `MAP` probe).
    Siglip(SiglipVisionRecipe),
}

impl VisionEncoderConfig {
    /// Returns the canonical architecture label (`"clip"` or `"siglip"`).
    #[must_use]
    pub const fn architecture(&self) -> &'static str {
        match self {
            Self::Clip(cfg) => cfg.architecture,
            Self::Siglip(cfg) => cfg.architecture,
        }
    }

    /// Returns the nominal square image input resolution.
    #[must_use]
    pub const fn image_size(&self) -> usize {
        match self {
            Self::Clip(cfg) => cfg.image_size,
            Self::Siglip(cfg) => cfg.image_size,
        }
    }

    /// Returns the square convolution patch size.
    #[must_use]
    pub const fn patch_size(&self) -> usize {
        match self {
            Self::Clip(cfg) => cfg.patch_size,
            Self::Siglip(cfg) => cfg.patch_size,
        }
    }

    /// Returns the expected input color channel count.
    #[must_use]
    pub const fn in_channels(&self) -> usize {
        match self {
            Self::Clip(cfg) => cfg.in_channels,
            Self::Siglip(cfg) => cfg.in_channels,
        }
    }

    /// Returns the hidden width per spatial patch token produced by the vision encoder.
    #[must_use]
    pub const fn hidden_dim(&self) -> usize {
        match self {
            Self::Clip(cfg) => cfg.hidden_dim,
            Self::Siglip(cfg) => cfg.hidden_dim,
        }
    }

    /// Bilinearly resizes and channel-normalizes an interleaved `[src_h, src_w, 3]` RGB image into
    /// `[3, dst_h, dst_w]` `CHW` layout using this encoder's normalization statistics.
    ///
    /// # Errors
    /// Returns [`K3Error`] if dimensions are invalid or any pixel value is non-finite.
    pub fn preprocess_rgb_hwc(
        &self,
        src_hwc: &[f32],
        src_h: usize,
        src_w: usize,
        dst_h: usize,
        dst_w: usize,
    ) -> Result<Vec<f32>, K3Error> {
        match self {
            Self::Clip(cfg) => cfg.preprocess_rgb_hwc(src_hwc, src_h, src_w, dst_h, dst_w),
            Self::Siglip(cfg) => cfg.preprocess_rgb_hwc(src_hwc, src_h, src_w, dst_h, dst_w),
        }
    }

    /// Encodes `CHW` pixel input through the configured vision encoder into [`VisionTokens`].
    ///
    /// # Errors
    /// Returns [`K3Error`] if inputs, geometry, or vision weights are invalid.
    pub fn encode_patch_tokens(
        &self,
        weights: &dyn ModelWeights,
        vision_pixels: &[f32],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<VisionTokens, K3Error> {
        match self {
            Self::Clip(cfg) => {
                cfg.encode_patch_tokens(weights, vision_pixels, channels, height, width)
            }
            Self::Siglip(cfg) => {
                cfg.encode_patch_tokens(weights, vision_pixels, channels, height, width)
            }
        }
    }
}

impl From<ClipVisionRecipe> for VisionEncoderConfig {
    fn from(value: ClipVisionRecipe) -> Self {
        Self::Clip(value)
    }
}

impl From<SiglipVisionRecipe> for VisionEncoderConfig {
    fn from(value: SiglipVisionRecipe) -> Self {
        Self::Siglip(value)
    }
}

/// Bridge converting raw image tensors through a Vision Transformer (`CLIP` or `SigLIP`),
/// optional `2x2` spatial pixel-shuffle merging, and a multimodal MLP/linear projector into
/// LLM soft-prompt embeddings `[num_tokens, llm_embed_dim]`.
#[derive(Debug, Clone, PartialEq)]
pub struct VlmMmprojBridge {
    /// Configured `CLIP` or `SigLIP` vision encoder backbone.
    pub encoder: VisionEncoderConfig,
    /// Optional explicit LLM token embedding width verified against projector output weights.
    pub llm_embed_dim: Option<usize>,
    /// Enables `2x2` spatial token merging (`4x` width, `0.25x` tokens) before projection.
    pub merge_2x2_pixel_shuffle: bool,
    /// Strips a leading `[CLS]` token before spatial merging / projection when present.
    pub drop_cls_token: bool,
    /// Normalization stability epsilon for optional `mm.input_norm` / `mm.norm` layers.
    pub norm_eps: f32,
}

impl VlmMmprojBridge {
    /// Creates a bridge wrapping any [`VisionEncoderConfig`].
    #[must_use]
    pub const fn new(encoder: VisionEncoderConfig) -> Self {
        Self {
            encoder,
            llm_embed_dim: None,
            merge_2x2_pixel_shuffle: false,
            drop_cls_token: true,
            norm_eps: 1e-6,
        }
    }

    /// Creates a `CLIP`-backed `mmproj` bridge (e.g., `LLaVA-1.5` / `LLaVA-1.6`).
    #[must_use]
    pub const fn new_clip(vision: ClipVisionRecipe) -> Self {
        Self::new(VisionEncoderConfig::Clip(vision))
    }

    /// Creates a `SigLIP`-backed `mmproj` bridge.
    #[must_use]
    pub const fn new_siglip(vision: SiglipVisionRecipe) -> Self {
        Self::new(VisionEncoderConfig::Siglip(vision))
    }

    /// Creates a `LLaVA`-style `CLIP` + 2-layer MLP bridge enforcing `llm_embed_dim`.
    #[must_use]
    pub const fn new_llava_clip(vision: ClipVisionRecipe, llm_embed_dim: usize) -> Self {
        Self::new_clip(vision).with_llm_embed_dim(llm_embed_dim)
    }

    /// Creates a `Gemma3`-style `SigLIP` + `2x2` pixel-shuffle merge bridge enforcing `llm_embed_dim`.
    #[must_use]
    pub const fn new_gemma3_siglip(vision: SiglipVisionRecipe, llm_embed_dim: usize) -> Self {
        Self::new_siglip(vision)
            .with_llm_embed_dim(llm_embed_dim)
            .with_pixel_shuffle_2x2(true)
    }

    /// Creates a `Qwen2-VL`-style `SigLIP` + `2x2` pixel-shuffle merge bridge enforcing `llm_embed_dim`.
    #[must_use]
    pub const fn new_qwen2_vl(vision: SiglipVisionRecipe, llm_embed_dim: usize) -> Self {
        Self::new_siglip(vision)
            .with_llm_embed_dim(llm_embed_dim)
            .with_pixel_shuffle_2x2(true)
    }

    /// Sets the expected LLM hidden embedding width enforced against projector outputs.
    #[must_use]
    pub const fn with_llm_embed_dim(mut self, llm_embed_dim: usize) -> Self {
        self.llm_embed_dim = Some(llm_embed_dim);
        self
    }

    /// Enables or disables `2x2` spatial pixel-shuffle merging (`grid_h/2 * grid_w/2`, `4 * hidden_dim`).
    #[must_use]
    pub const fn with_pixel_shuffle_2x2(mut self, enabled: bool) -> Self {
        self.merge_2x2_pixel_shuffle = enabled;
        self
    }

    /// Controls whether a leading `[CLS]` token is stripped prior to projector execution.
    #[must_use]
    pub const fn with_drop_cls_token(mut self, drop_cls_token: bool) -> Self {
        self.drop_cls_token = drop_cls_token;
        self
    }

    /// Overrides the epsilon used by optional projector `LayerNorm` layers.
    #[must_use]
    pub const fn with_norm_eps(mut self, eps: f32) -> Self {
        self.norm_eps = eps;
        self
    }

    /// Hydrates a [`VlmMmprojBridge`] from parsed [`ModelMetadata`] (GGUF or `SafeTensors`).
    #[must_use]
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        let arch = metadata.architecture().to_ascii_lowercase();
        let use_siglip = prefers_siglip(&arch);
        let merge_2x2 = prefers_pixel_shuffle(&arch);
        let encoder = if use_siglip {
            VisionEncoderConfig::Siglip(SiglipVisionRecipe::from_metadata(metadata))
        } else {
            VisionEncoderConfig::Clip(ClipVisionRecipe::from_metadata(metadata))
        };
        Self {
            encoder,
            llm_embed_dim: None,
            merge_2x2_pixel_shuffle: merge_2x2,
            drop_cls_token: true,
            norm_eps: 1e-6,
        }
    }

    /// Hydrates a [`VlmMmprojBridge`] from a parsed [`GgufFile`], inspecting `clip.projector_type`,
    /// `spatial_merge_size`, projection dimensions, and weight tensor layout.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        let arch = gguf.metadata().architecture().to_ascii_lowercase();
        let proj_type = gguf_str(
            gguf,
            &[
                "clip.projector_type",
                "clip.vision.projector_type",
                "projector_type",
            ],
        )
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();

        let has_cls_tensor = lookup_optional_tensor(
            gguf,
            &[
                "v.class_embd",
                "v.class_embd.weight",
                "class_embd",
                "vision_model.embeddings.class_embedding",
            ],
        )
        .ok()
        .flatten()
        .is_some();
        let has_merge_norm = lookup_optional_tensor(
            gguf,
            &[
                "mm.input_norm.weight",
                "merger.ln_q.weight",
                "visual.merger.ln_q.weight",
            ],
        )
        .ok()
        .flatten()
        .is_some();

        let use_siglip = prefers_siglip(&arch)
            || prefers_siglip(&proj_type)
            || (!has_cls_tensor && (has_merge_norm || arch.is_empty()));

        let mut merge_2x2 = prefers_pixel_shuffle(&arch) || prefers_pixel_shuffle(&proj_type);
        if let Some(merge_size) = gguf_usize(
            gguf,
            &[
                "clip.vision.spatial_merge_size",
                "spatial_merge_size",
                "vision.spatial_merge_size",
            ],
        ) {
            merge_2x2 = merge_size == 2;
        }

        let llm_embed_dim = gguf_usize(
            gguf,
            &[
                "clip.projection_dim",
                "clip.vision.projection_dim",
                "mm.output_dim",
                "llm.embedding_length",
                "llama.embedding_length",
                "gemma3.embedding_length",
                "qwen2vl.embedding_length",
            ],
        );

        let encoder = if use_siglip {
            VisionEncoderConfig::Siglip(SiglipVisionRecipe::from_gguf(gguf))
        } else {
            VisionEncoderConfig::Clip(ClipVisionRecipe::from_gguf(gguf))
        };

        Self {
            encoder,
            llm_embed_dim,
            merge_2x2_pixel_shuffle: merge_2x2,
            drop_cls_token: true,
            norm_eps: 1e-6,
        }
    }

    /// Hydrates a [`VlmMmprojBridge`] from a parsed [`SafeTensorsFile`] including `__metadata__`
    /// and tensor key conventions.
    #[must_use]
    pub fn from_safetensors(st: &SafeTensorsFile<'_>) -> Self {
        let arch = st.metadata().architecture().to_ascii_lowercase();
        let proj_type = st_str(st, &["projector_type", "clip.projector_type", "model_type"])
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();

        let has_cls_tensor = lookup_optional_tensor(
            st,
            &[
                "v.class_embd",
                "vision_model.embeddings.class_embedding",
                "vision_model.embeddings.class_embedding.weight",
                "embeddings.class_embedding",
            ],
        )
        .ok()
        .flatten()
        .is_some();
        let has_qwen_merger =
            lookup_optional_tensor(st, &["merger.ln_q.weight", "visual.merger.ln_q.weight"])
                .ok()
                .flatten()
                .is_some();

        let use_siglip = prefers_siglip(&arch)
            || prefers_siglip(&proj_type)
            || (!has_cls_tensor && has_qwen_merger);
        let mut merge_2x2 =
            prefers_pixel_shuffle(&arch) || prefers_pixel_shuffle(&proj_type) || has_qwen_merger;
        if let Some(merge_size) = st_usize(
            st,
            &["spatial_merge_size", "vision_config.spatial_merge_size"],
        ) {
            merge_2x2 = merge_size == 2;
        }

        let llm_embed_dim = st_usize(
            st,
            &[
                "text_config.hidden_size",
                "llm.embedding_length",
                "mm.output_dim",
                "projection_dim",
            ],
        );

        let encoder = if use_siglip {
            VisionEncoderConfig::Siglip(SiglipVisionRecipe::from_safetensors(st))
        } else {
            VisionEncoderConfig::Clip(ClipVisionRecipe::from_safetensors(st))
        };

        Self {
            encoder,
            llm_embed_dim,
            merge_2x2_pixel_shuffle: merge_2x2,
            drop_cls_token: true,
            norm_eps: 1e-6,
        }
    }

    /// Encodes a `CHW` image `[channels, height, width]` through the vision backbone and multimodal
    /// projector into flattened `[num_tokens, llm_embed_dim]` soft-prompt embeddings.
    ///
    /// # Errors
    /// Returns [`K3Error`] if image dimensions/pixels are invalid, required vision or projector
    /// weights are missing/corrupted, spatial dimensions are incompatible with `2x2` merging, or
    /// any output value is non-finite.
    pub fn encode_image_to_soft_prompt(
        &self,
        weights: &dyn ModelWeights,
        vision_pixels: &[f32],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, K3Error> {
        let (embeds, _, _) = self.encode_image_to_soft_prompt_with_dims(
            weights,
            vision_pixels,
            channels,
            height,
            width,
        )?;
        Ok(embeds)
    }

    /// Encodes a `CHW` image `[channels, height, width]` and returns
    /// `(embeds, num_tokens, llm_embed_dim)`.
    ///
    /// # Errors
    /// Returns [`K3Error`] on invalid inputs, missing/malformed weights, or projection mismatches.
    pub fn encode_image_to_soft_prompt_with_dims(
        &self,
        weights: &dyn ModelWeights,
        vision_pixels: &[f32],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<(Vec<f32>, usize, usize), K3Error> {
        self.encode_image_to_soft_prompt_split(
            weights,
            weights,
            vision_pixels,
            channels,
            height,
            width,
        )
    }

    /// Encodes a `CHW` image using separate `vision_weights` (backbone) and `mmproj_weights`
    /// (multimodal projector), returning `(embeds, num_tokens, llm_embed_dim)`.
    ///
    /// # Errors
    /// Returns [`K3Error`] on invalid inputs, missing/malformed weights, or projection mismatches.
    pub fn encode_image_to_soft_prompt_split(
        &self,
        vision_weights: &dyn ModelWeights,
        mmproj_weights: &dyn ModelWeights,
        vision_pixels: &[f32],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<(Vec<f32>, usize, usize), K3Error> {
        self.validate_config()?;
        let tokens = self.encoder.encode_patch_tokens(
            vision_weights,
            vision_pixels,
            channels,
            height,
            width,
        )?;
        self.project_tokens(mmproj_weights, &tokens)
    }

    /// Bilinearly resizes and normalizes an interleaved `[src_h, src_w, 3]` RGB image to
    /// `[3, dst_h, dst_w]` and encodes it into `(embeds, num_tokens, llm_embed_dim)`.
    ///
    /// # Errors
    /// Returns [`K3Error`] on invalid dimensions, non-finite pixels, or projection failures.
    pub fn encode_rgb_hwc_to_soft_prompt(
        &self,
        weights: &dyn ModelWeights,
        src_hwc: &[f32],
        src_h: usize,
        src_w: usize,
        dst_h: usize,
        dst_w: usize,
    ) -> Result<(Vec<f32>, usize, usize), K3Error> {
        let chw = self
            .encoder
            .preprocess_rgb_hwc(src_hwc, src_h, src_w, dst_h, dst_w)?;
        self.encode_image_to_soft_prompt_with_dims(
            weights,
            &chw,
            self.encoder.in_channels(),
            dst_h,
            dst_w,
        )
    }

    /// Projects pre-computed [`VisionTokens`] through optional spatial `2x2` pixel-shuffle
    /// merging and the multimodal MLP/linear projector into `(embeds, num_tokens, llm_embed_dim)`.
    ///
    /// # Errors
    /// Returns [`K3Error`] if token geometry is invalid, projector weights are missing or
    /// mismatched, or non-finite values are encountered.
    pub fn project_tokens(
        &self,
        mmproj_weights: &dyn ModelWeights,
        vision_tokens: &VisionTokens,
    ) -> Result<(Vec<f32>, usize, usize), K3Error> {
        self.validate_config()?;
        validate_vision_tokens(vision_tokens)?;

        let arch = self.encoder.architecture();
        let hidden_dim = vision_tokens.hidden_dim;
        if hidden_dim != self.encoder.hidden_dim() {
            return Err(mismatch_err(
                arch,
                format!(
                    "vision token hidden_dim {} does not match configured encoder hidden_dim {}",
                    hidden_dim,
                    self.encoder.hidden_dim()
                ),
            ));
        }

        let (raw_tokens, token_count) = if self.drop_cls_token && vision_tokens.has_cls_token {
            (
                vision_tokens.spatial_tokens(),
                vision_tokens.spatial_token_count(),
            )
        } else {
            (vision_tokens.tokens.as_slice(), vision_tokens.seq_len)
        };

        if self.merge_2x2_pixel_shuffle && !self.drop_cls_token && vision_tokens.has_cls_token {
            return Err(mismatch_err(
                arch,
                "2x2 pixel-shuffle merging requires dropping [CLS] token (drop_cls_token=true)",
            ));
        }

        // Optional pre-merge normalization (e.g., Qwen2-VL merger.ln_q / mm.input_norm).
        let normed_storage = apply_optional_seq_norm(
            mmproj_weights,
            (PRE_NORM_WEIGHT_ALIASES, PRE_NORM_BIAS_ALIASES),
            raw_tokens,
            token_count,
            hidden_dim,
            self.norm_eps,
            arch,
        )?;
        let pre_merge_tokens = normed_storage.as_deref().unwrap_or(raw_tokens);

        // Optional 2x2 spatial pixel-shuffle merge.
        let (proj_in_buf, proj_tokens, proj_in_dim) = if self.merge_2x2_pixel_shuffle {
            let grid_h = vision_tokens.grid_h;
            let grid_w = vision_tokens.grid_w;
            if grid_h < 2 || grid_w < 2 || grid_h % 2 != 0 || grid_w % 2 != 0 {
                return Err(bounds_err(
                    "mmproj.pixel_shuffle_2x2",
                    checked_mul("mmproj.pixel_shuffle_2x2", grid_h * grid_w, hidden_dim)?,
                    pre_merge_tokens.len(),
                    format!(
                        "2x2 pixel shuffle requires even patch grid dimensions >= 2, got {grid_h}x{grid_w}"
                    ),
                ));
            }
            let merged_tokens = (grid_h / 2) * (grid_w / 2);
            let merged_dim = checked_mul("mmproj.pixel_shuffle_2x2", hidden_dim, 4)?;
            let mut merged =
                vec![0.0_f32; checked_mul("mmproj.pixel_shuffle_2x2", merged_tokens, merged_dim)?];
            pixel_shuffle_merge_2x2_f32(&mut merged, pre_merge_tokens, grid_h, grid_w, hidden_dim)?;
            (Some(merged), merged_tokens, merged_dim)
        } else {
            (None, token_count, hidden_dim)
        };
        let proj_input = proj_in_buf.as_deref().unwrap_or(pre_merge_tokens);

        let (mut projected, out_dim) =
            self.run_projector_layers(mmproj_weights, proj_input, proj_tokens, proj_in_dim)?;

        if let Some(expected_dim) = self.llm_embed_dim
            && out_dim != expected_dim
        {
            return Err(mismatch_err(
                arch,
                format!(
                    "mmproj output dimension {out_dim} != expected llm_embed_dim {expected_dim}"
                ),
            ));
        }

        // Optional post-projector normalization.
        if let Some(post_normed) = apply_optional_seq_norm(
            mmproj_weights,
            (POST_NORM_WEIGHT_ALIASES, POST_NORM_BIAS_ALIASES),
            &projected,
            proj_tokens,
            out_dim,
            self.norm_eps,
            arch,
        )? {
            projected = post_normed;
        }

        let expected_len = checked_mul("mmproj.soft_prompt", proj_tokens, out_dim)?;
        if projected.len() != expected_len {
            return Err(bounds_err(
                "mmproj.soft_prompt",
                expected_len,
                projected.len(),
                "soft-prompt output buffer length mismatch",
            ));
        }
        validate_finite_slice("mmproj.soft_prompt", &projected)?;
        Ok((projected, proj_tokens, out_dim))
    }

    fn run_projector_layers(
        &self,
        weights: &dyn ModelWeights,
        input: &[f32],
        num_tokens: usize,
        in_dim: usize,
    ) -> Result<(Vec<f32>, usize), K3Error> {
        let arch = self.encoder.architecture();
        if let Some((fc1_idx, fc1_view)) = lookup_optional_indexed(weights, MLP_FC1_WEIGHT_ALIASES)
        {
            let mlp_hidden = infer_2d_other_dim(&fc1_view, in_dim, "mmproj.fc1")?;
            let fc1_mat = MatrixSlice::from_view(&fc1_view, mlp_hidden, in_dim)?;
            let fc1_bias = decode_matching_or_any_bias(
                weights,
                fc1_idx,
                MLP_FC1_BIAS_ALIASES,
                mlp_hidden,
                arch,
            )?;

            // Fail closed if fc1 is present without fc2.
            let (fc2_idx, fc2_view) =
                lookup_required_indexed(weights, MLP_FC2_WEIGHT_ALIASES, "mmproj.fc2")?;
            let out_dim = if let Some(expected) = self.llm_embed_dim {
                validate_2d_linear_shape(&fc2_view, expected, mlp_hidden)?;
                expected
            } else {
                infer_2d_other_dim(&fc2_view, mlp_hidden, "mmproj.fc2")?
            };
            let fc2_mat = MatrixSlice::from_view(&fc2_view, out_dim, mlp_hidden)?;
            let fc2_bias =
                decode_matching_or_any_bias(weights, fc2_idx, MLP_FC2_BIAS_ALIASES, out_dim, arch)?;

            let mut hidden = vec![0.0_f32; checked_mul("mmproj.fc1.out", num_tokens, mlp_hidden)?];
            linear_forward_seq(
                &mut hidden,
                &fc1_mat,
                fc1_bias.as_deref(),
                input,
                num_tokens,
                in_dim,
                mlp_hidden,
                arch,
            )?;
            gelu_inplace_f32(&mut hidden)?;

            let mut out = vec![0.0_f32; checked_mul("mmproj.fc2.out", num_tokens, out_dim)?];
            linear_forward_seq(
                &mut out,
                &fc2_mat,
                fc2_bias.as_deref(),
                &hidden,
                num_tokens,
                mlp_hidden,
                out_dim,
                arch,
            )?;
            return Ok((out, out_dim));
        }

        // Fallback to single-layer linear projector only when no 2-layer fc1 exists.
        let lin_view = lookup_first_tensor(weights, LINEAR_PROJ_WEIGHT_ALIASES, "mmproj.linear")?;
        let out_dim = if let Some(expected) = self.llm_embed_dim {
            validate_2d_linear_shape(&lin_view, expected, in_dim)?;
            expected
        } else {
            infer_2d_other_dim(&lin_view, in_dim, "mmproj.linear")?
        };
        let lin_mat = MatrixSlice::from_view(&lin_view, out_dim, in_dim)?;
        let lin_bias = decode_optional_1d(weights, LINEAR_PROJ_BIAS_ALIASES, out_dim, arch)?;
        let mut out = vec![0.0_f32; checked_mul("mmproj.linear.out", num_tokens, out_dim)?];
        linear_forward_seq(
            &mut out,
            &lin_mat,
            lin_bias.as_deref(),
            input,
            num_tokens,
            in_dim,
            out_dim,
            arch,
        )?;
        Ok((out, out_dim))
    }

    fn validate_config(&self) -> Result<(), K3Error> {
        let arch = self.encoder.architecture();
        if self.llm_embed_dim == Some(0) {
            return Err(mismatch_err(
                arch,
                "mmproj llm_embed_dim cannot be zero when specified",
            ));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(mismatch_err(
                arch,
                format!(
                    "mmproj norm_eps ({}) must be finite and positive",
                    self.norm_eps
                ),
            ));
        }
        Ok(())
    }
}

fn apply_optional_seq_norm(
    weights: &dyn ModelWeights,
    aliases: (&[&str], &[&str]),
    input: &[f32],
    token_count: usize,
    hidden_dim: usize,
    norm_eps: f32,
    arch: &str,
) -> Result<Option<Vec<f32>>, K3Error> {
    let (weight_aliases, bias_aliases) = aliases;
    let Some(norm_w_view) = lookup_optional_tensor(weights, weight_aliases)? else {
        return Ok(None);
    };
    let norm_w = decode_1d_view(&norm_w_view, hidden_dim, arch)?;
    let norm_b = decode_optional_1d(weights, bias_aliases, hidden_dim, arch)?;
    let mut out = vec![0.0_f32; input.len()];
    apply_layernorm_seq(
        &mut out,
        input,
        token_count,
        hidden_dim,
        &norm_w,
        norm_b.as_deref(),
        norm_eps,
    )?;
    Ok(Some(out))
}

fn validate_vision_tokens(tokens: &VisionTokens) -> Result<(), K3Error> {
    if tokens.grid_h == 0 || tokens.grid_w == 0 || tokens.hidden_dim == 0 || tokens.seq_len == 0 {
        return Err(bounds_err(
            "vision_tokens",
            1,
            0,
            format!(
                "invalid empty VisionTokens geometry: grid={}x{}, seq_len={}, hidden_dim={}",
                tokens.grid_h, tokens.grid_w, tokens.seq_len, tokens.hidden_dim
            ),
        ));
    }
    let spatial = checked_mul("vision_tokens.grid", tokens.grid_h, tokens.grid_w)?;
    let expected_seq = spatial + usize::from(tokens.has_cls_token);
    if tokens.seq_len != expected_seq {
        return Err(bounds_err(
            "vision_tokens.seq_len",
            expected_seq,
            tokens.seq_len,
            format!(
                "VisionTokens seq_len ({}) != expected spatial ({spatial}) + cls ({})",
                tokens.seq_len,
                usize::from(tokens.has_cls_token)
            ),
        ));
    }
    let expected_len = checked_mul("vision_tokens", tokens.seq_len, tokens.hidden_dim)?;
    if tokens.tokens.len() != expected_len {
        return Err(bounds_err(
            "vision_tokens",
            expected_len,
            tokens.tokens.len(),
            format!(
                "VisionTokens buffer length ({}) != seq_len * hidden_dim ({expected_len})",
                tokens.tokens.len()
            ),
        ));
    }
    validate_finite_slice("vision_tokens", &tokens.tokens)
}

fn lookup_optional_indexed<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[&str],
) -> Option<(usize, TensorView<'w>)> {
    for (idx, name) in aliases.iter().enumerate() {
        if let Ok(view) = weights.tensor(name) {
            return Some((idx, view));
        }
    }
    None
}

fn lookup_required_indexed<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[&str],
    role: &str,
) -> Result<(usize, TensorView<'w>), K3Error> {
    lookup_optional_indexed(weights, aliases).ok_or_else(|| {
        bounds_err(
            aliases.first().copied().unwrap_or(role),
            1,
            0,
            format!("missing required projector tensor for {role}; tried aliases: {aliases:?}"),
        )
    })
}

fn decode_matching_or_any_bias(
    weights: &dyn ModelWeights,
    preferred_idx: usize,
    bias_aliases: &[&str],
    expected_len: usize,
    arch: &str,
) -> Result<Option<Vec<f32>>, K3Error> {
    if let Some(name) = bias_aliases.get(preferred_idx)
        && let Ok(view) = weights.tensor(name)
    {
        return decode_1d_view(&view, expected_len, arch).map(Some);
    }
    decode_optional_1d(weights, bias_aliases, expected_len, arch)
}

fn prefers_siglip(label: &str) -> bool {
    label.contains("siglip")
        || label.contains("gemma3")
        || label.contains("qwen2_vl")
        || label.contains("qwen2vl")
        || label.contains("qwen2_5_vl")
        || label.contains("paligemma")
}

fn prefers_pixel_shuffle(label: &str) -> bool {
    label.contains("gemma3")
        || label.contains("qwen2_vl")
        || label.contains("qwen2vl")
        || label.contains("qwen2_5_vl")
        || label.contains("pixel_shuffle")
}

fn gguf_str<'a>(gguf: &'a GgufFile<'a>, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| gguf.metadata_value(key)?.as_str())
}

fn gguf_usize(gguf: &GgufFile<'_>, keys: &[&str]) -> Option<usize> {
    keys.iter()
        .find_map(|key| gguf.metadata_value(key)?.as_usize())
        .filter(|&v| v > 0)
}

fn st_str<'a>(st: &'a SafeTensorsFile<'a>, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|key| st.metadata_value(key))
}

fn st_usize(st: &SafeTensorsFile<'_>, keys: &[&str]) -> Option<usize> {
    keys.iter()
        .find_map(|key| st.metadata_value(key)?.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
}
