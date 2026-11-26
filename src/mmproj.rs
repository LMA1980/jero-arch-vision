//! VLM multimodal projection (`mmproj`) bridge for turning vision patch embeddings into LLM
//! soft-prompt embeddings.
//!
//! Supports both 2-layer MLP (`mm.0` -> `GELU` -> `mm.2`) and single-layer linear projectors,
//! optional `2x2` spatial pixel-shuffle merging (`pixel_shuffle_merge_2x2_f32` for `Gemma3` /
//! `Qwen2-VL`), optional pre/post normalization, and both `GGUF` (`mmproj-*.gguf`) and
//! `SafeTensors` weight layouts.

use jero_core::{GgufFile, K3Error, ModelMetadata, ModelWeights, SafeTensorsFile, TensorView};

use crate::kernels::{
    MatrixSlice, apply_layernorm_seq, apply_rmsnorm_seq, bounds_err, checked_mul, decode_1d_view,
    decode_optional_1d, gelu_inplace_f32, infer_2d_other_dim, linear_forward_seq,
    linear_forward_seq_transposed, lookup_first_tensor, lookup_optional_tensor, mismatch_err,
    pixel_shuffle_merge_2x2_f32, spatial_avg_pool_2d_f32, validate_2d_linear_shape,
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
    "mm.input_projection",
    "mm.input_projection_weight",
    "mm_input_projection_w",
    "mm_input_projection.weight",
    "mm_input_proj.weight",
    "multi_modal_projector.mm_input_projection_weight",
    "multi_modal_projector.mm_input_projection.weight",
];
const LINEAR_PROJ_BIAS_ALIASES: &[&str] = &["mm.bias", "mmproj.bias", "mm.input_projection.bias"];

const SOFT_EMB_NORM_ALIASES: &[&str] = &[
    "mm.soft_emb_norm.weight",
    "mm.soft_emb_norm",
    "mm_soft_emb_norm.weight",
    "mm_soft_emb_norm",
    "multi_modal_projector.mm_soft_emb_norm.weight",
];

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
/// optional spatial merging (`2x2` pixel-shuffle or `k x k` Gemma-3 average pooling), optional
/// normalization (`mm.input_norm` `LayerNorm` / `mm.soft_emb_norm` `RMSNorm`), and a multimodal
/// MLP/linear projector into LLM soft-prompt embeddings `[num_tokens, llm_embed_dim]`.
#[derive(Debug, Clone, PartialEq)]
pub struct VlmMmprojBridge {
    /// Configured `CLIP` or `SigLIP` vision encoder backbone.
    pub encoder: VisionEncoderConfig,
    /// Optional explicit LLM token embedding width verified against projector output weights.
    pub llm_embed_dim: Option<usize>,
    /// Enables `2x2` spatial token merging (`4x` width, `0.25x` tokens) before projection.
    pub merge_2x2_pixel_shuffle: bool,
    /// Optional `k x k` 2D spatial average pooling kernel (`Gemma-3` `mmproj` convention).
    pub spatial_avg_pool: Option<usize>,
    /// Strips a leading `[CLS]` token before spatial merging / projection when present.
    pub drop_cls_token: bool,
    /// Normalization stability epsilon for optional `mm.input_norm` / `mm.soft_emb_norm` / `mm.norm`.
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
            spatial_avg_pool: None,
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

    /// Creates an upstream `Gemma3` `SigLIP` + `k x k` average-pooling bridge enforcing `llm_embed_dim`.
    #[must_use]
    pub const fn new_gemma3_avg_pool(
        vision: SiglipVisionRecipe,
        llm_embed_dim: usize,
        pool_kernel: usize,
    ) -> Self {
        Self::new_siglip(vision)
            .with_llm_embed_dim(llm_embed_dim)
            .with_spatial_avg_pool(Some(pool_kernel))
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

    /// Configures optional `k x k` 2D spatial average pooling before projection (`Gemma3` `mmproj`).
    #[must_use]
    pub const fn with_spatial_avg_pool(mut self, kernel_size: Option<usize>) -> Self {
        self.spatial_avg_pool = kernel_size;
        self
    }

    /// Controls whether a leading `[CLS]` token is stripped prior to projector execution.
    #[must_use]
    pub const fn with_drop_cls_token(mut self, drop_cls_token: bool) -> Self {
        self.drop_cls_token = drop_cls_token;
        self
    }

    /// Overrides the epsilon used by optional projector `LayerNorm` / `RMSNorm` layers.
    #[must_use]
    pub const fn with_norm_eps(mut self, eps: f32) -> Self {
        self.norm_eps = eps;
        self
    }

    /// Hydrates a [`VlmMmprojBridge`] from parsed [`ModelMetadata`] (GGUF or `SafeTensors`).
    #[must_use]
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        let arch = metadata.architecture().to_ascii_lowercase();
        let has_soft_emb_norm = metadata
            .tensors()
            .iter()
            .any(|d| SOFT_EMB_NORM_ALIASES.contains(&d.name()));
        let use_siglip = prefers_siglip(&arch) || has_soft_emb_norm;
        let encoder = if use_siglip {
            VisionEncoderConfig::Siglip(SiglipVisionRecipe::from_metadata(metadata))
        } else {
            VisionEncoderConfig::Clip(ClipVisionRecipe::from_metadata(metadata))
        };
        let (merge_2x2, spatial_avg_pool) = if has_soft_emb_norm {
            (false, Some(4))
        } else {
            (prefers_pixel_shuffle(&arch), None)
        };
        Self {
            encoder,
            llm_embed_dim: None,
            merge_2x2_pixel_shuffle: merge_2x2,
            spatial_avg_pool,
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
        let has_soft_emb_norm = lookup_optional_tensor(gguf, SOFT_EMB_NORM_ALIASES)
            .ok()
            .flatten()
            .is_some();

        let use_siglip = prefers_siglip(&arch)
            || prefers_siglip(&proj_type)
            || has_soft_emb_norm
            || (!has_cls_tensor && (has_merge_norm || arch.is_empty()));

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
        let merge_size = gguf_usize(
            gguf,
            &[
                "clip.vision.spatial_merge_size",
                "clip.vision.proj_scale_factor",
                "clip.vision.pooling_kernel_size",
                "spatial_merge_size",
                "vision.spatial_merge_size",
            ],
        );
        let (merge_2x2, spatial_avg_pool) = resolve_spatial_merge_mode(
            gguf,
            &arch,
            &proj_type,
            encoder.hidden_dim(),
            merge_size,
            has_soft_emb_norm,
            false,
        );

        Self {
            encoder,
            llm_embed_dim,
            merge_2x2_pixel_shuffle: merge_2x2,
            spatial_avg_pool,
            drop_cls_token: true,
            norm_eps: gguf_norm_eps(gguf),
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
        let has_soft_emb_norm = lookup_optional_tensor(st, SOFT_EMB_NORM_ALIASES)
            .ok()
            .flatten()
            .is_some();

        let use_siglip = prefers_siglip(&arch)
            || prefers_siglip(&proj_type)
            || has_soft_emb_norm
            || (!has_cls_tensor && has_qwen_merger);

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
        let merge_size = st_usize(
            st,
            &[
                "spatial_merge_size",
                "vision_config.spatial_merge_size",
                "pooling_kernel_size",
            ],
        );
        let (merge_2x2, spatial_avg_pool) = resolve_spatial_merge_mode(
            st,
            &arch,
            &proj_type,
            encoder.hidden_dim(),
            merge_size,
            has_soft_emb_norm,
            has_qwen_merger,
        );

        Self {
            encoder,
            llm_embed_dim,
            merge_2x2_pixel_shuffle: merge_2x2,
            spatial_avg_pool,
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

        if (self.merge_2x2_pixel_shuffle || self.spatial_avg_pool.is_some_and(|k| k > 1))
            && !self.drop_cls_token
            && vision_tokens.has_cls_token
        {
            return Err(mismatch_err(
                arch,
                "spatial token pooling/merging requires dropping [CLS] token (drop_cls_token=true)",
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

        // Optional 2x2 spatial pixel-shuffle merge or k x k spatial average pooling (Gemma-3).
        let (proj_in_buf, proj_tokens, proj_in_dim) = self.apply_spatial_pooling(
            pre_merge_tokens,
            vision_tokens.grid_h,
            vision_tokens.grid_w,
            token_count,
            hidden_dim,
        )?;
        let pre_proj_tokens = proj_in_buf.as_deref().unwrap_or(pre_merge_tokens);

        // Optional Gemma-3 soft-embedding RMSNorm (applied after spatial pooling, before linear projection).
        let soft_norm_storage = apply_optional_seq_rmsnorm(
            mmproj_weights,
            SOFT_EMB_NORM_ALIASES,
            pre_proj_tokens,
            proj_tokens,
            proj_in_dim,
            self.norm_eps,
            arch,
        )?;
        let proj_input = soft_norm_storage.as_deref().unwrap_or(pre_proj_tokens);

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

    fn apply_spatial_pooling(
        &self,
        tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        token_count: usize,
        hidden_dim: usize,
    ) -> Result<(Option<Vec<f32>>, usize, usize), K3Error> {
        if self.merge_2x2_pixel_shuffle {
            if grid_h < 2 || grid_w < 2 || grid_h % 2 != 0 || grid_w % 2 != 0 {
                return Err(bounds_err(
                    "mmproj.pixel_shuffle_2x2",
                    checked_mul("mmproj.pixel_shuffle_2x2", grid_h * grid_w, hidden_dim)?,
                    tokens.len(),
                    format!(
                        "2x2 pixel shuffle requires even patch grid dimensions >= 2, got {grid_h}x{grid_w}"
                    ),
                ));
            }
            let merged_tokens = (grid_h / 2) * (grid_w / 2);
            let merged_dim = checked_mul("mmproj.pixel_shuffle_2x2", hidden_dim, 4)?;
            let mut merged =
                vec![0.0_f32; checked_mul("mmproj.pixel_shuffle_2x2", merged_tokens, merged_dim)?];
            pixel_shuffle_merge_2x2_f32(&mut merged, tokens, grid_h, grid_w, hidden_dim)?;
            Ok((Some(merged), merged_tokens, merged_dim))
        } else if let Some(pool_k) = self.spatial_avg_pool {
            if grid_h < pool_k || grid_w < pool_k || grid_h % pool_k != 0 || grid_w % pool_k != 0 {
                return Err(bounds_err(
                    "mmproj.spatial_avg_pool",
                    checked_mul("mmproj.spatial_avg_pool", grid_h * grid_w, hidden_dim)?,
                    tokens.len(),
                    format!(
                        "{pool_k}x{pool_k} spatial average pooling requires patch grid dimensions divisible by {pool_k}, got {grid_h}x{grid_w}"
                    ),
                ));
            }
            let pooled_tokens = (grid_h / pool_k) * (grid_w / pool_k);
            let mut pooled =
                vec![0.0_f32; checked_mul("mmproj.spatial_avg_pool", pooled_tokens, hidden_dim)?];
            spatial_avg_pool_2d_f32(&mut pooled, tokens, grid_h, grid_w, hidden_dim, pool_k)?;
            Ok((Some(pooled), pooled_tokens, hidden_dim))
        } else {
            Ok((None, token_count, hidden_dim))
        }
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
        let is_transposed = linear_proj_is_transposed(weights, &lin_view, in_dim, out_dim);
        let lin_bias = decode_optional_1d(weights, LINEAR_PROJ_BIAS_ALIASES, out_dim, arch)?;
        let mut out = vec![0.0_f32; checked_mul("mmproj.linear.out", num_tokens, out_dim)?];
        if is_transposed {
            let lin_mat = MatrixSlice::from_view(&lin_view, in_dim, out_dim)?;
            linear_forward_seq_transposed(
                &mut out,
                &lin_mat,
                lin_bias.as_deref(),
                input,
                num_tokens,
                in_dim,
                out_dim,
                arch,
            )?;
        } else {
            let lin_mat = MatrixSlice::from_view(&lin_view, out_dim, in_dim)?;
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
        }
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
        if self.spatial_avg_pool == Some(0) {
            return Err(mismatch_err(
                arch,
                "mmproj spatial_avg_pool cannot be zero when specified",
            ));
        }
        if self.merge_2x2_pixel_shuffle && self.spatial_avg_pool.is_some_and(|k| k > 1) {
            return Err(mismatch_err(
                arch,
                "mmproj cannot enable both merge_2x2_pixel_shuffle and spatial_avg_pool",
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

fn apply_optional_seq_rmsnorm(
    weights: &dyn ModelWeights,
    aliases: &[&str],
    input: &[f32],
    token_count: usize,
    hidden_dim: usize,
    norm_eps: f32,
    arch: &str,
) -> Result<Option<Vec<f32>>, K3Error> {
    let Some(norm_w_view) = lookup_optional_tensor(weights, aliases)? else {
        return Ok(None);
    };
    let norm_w = decode_1d_view(&norm_w_view, hidden_dim, arch)?;
    let mut out = vec![0.0_f32; input.len()];
    apply_rmsnorm_seq(&mut out, input, token_count, hidden_dim, &norm_w, norm_eps)?;
    Ok(Some(out))
}

fn linear_proj_is_transposed(
    weights: &dyn ModelWeights,
    view: &TensorView<'_>,
    in_dim: usize,
    out_dim: usize,
) -> bool {
    let shape = view.shape();
    if shape.len() != 2 {
        return false;
    }
    let is_gguf_layout = if let Ok(Some(patch)) = lookup_optional_tensor(
        weights,
        &[
            "v.patch_embd.weight",
            "vision_model.embeddings.patch_embedding.weight",
        ],
    ) {
        let s = patch.shape();
        if s.len() == 4 && s[0] == s[1] && s[2] <= 4 && s[3] > 4 {
            true
        } else if s.len() == 4 && s[2] == s[3] && s[1] <= 4 && s[0] > 4 {
            false
        } else {
            view.name().starts_with("mm.") || view.name().starts_with("mm_input_proj")
        }
    } else if let Ok(Some(pos)) = lookup_optional_tensor(
        weights,
        &[
            "v.position_embd.weight",
            "vision_model.embeddings.position_embedding.weight",
        ],
    ) {
        let s = pos.shape();
        if s.len() == 2 && s[0] < s[1] && view.name().starts_with("mm.") {
            true
        } else {
            view.name().starts_with("mm.") || view.name().starts_with("mm_input_proj")
        }
    } else {
        view.name().starts_with("mm.") || view.name().starts_with("mm_input_proj")
    };

    if is_gguf_layout {
        if in_dim == out_dim {
            view.name().contains("input_projection") || view.name().contains("input_proj")
        } else {
            shape[0] == out_dim && shape[1] == in_dim
        }
    } else if in_dim != out_dim {
        shape[0] == in_dim && shape[1] == out_dim
    } else {
        false
    }
}

fn resolve_spatial_merge_mode(
    weights: &dyn ModelWeights,
    arch: &str,
    proj_type: &str,
    vision_hidden: usize,
    merge_size: Option<usize>,
    has_soft_emb_norm: bool,
    has_qwen_merger: bool,
) -> (bool, Option<usize>) {
    let has_mlp_fc1 = lookup_optional_tensor(weights, MLP_FC1_WEIGHT_ALIASES)
        .ok()
        .flatten()
        .is_some();
    let lin_expects_unmerged = lookup_optional_tensor(weights, LINEAR_PROJ_WEIGHT_ALIASES)
        .ok()
        .flatten()
        .is_some_and(|v| {
            v.shape().contains(&vision_hidden) && !v.shape().contains(&(vision_hidden * 4))
        });

    if has_soft_emb_norm
        || (!has_mlp_fc1
            && lin_expects_unmerged
            && (proj_type.contains("gemma3") || arch.contains("gemma3")))
    {
        (false, Some(merge_size.unwrap_or(4)))
    } else {
        let mut merge_2x2 =
            prefers_pixel_shuffle(arch) || prefers_pixel_shuffle(proj_type) || has_qwen_merger;
        if let Some(ms) = merge_size {
            merge_2x2 = ms == 2;
        }
        (merge_2x2, None)
    }
}

fn gguf_norm_eps(gguf: &GgufFile<'_>) -> f32 {
    gguf.metadata_value("clip.vision.attention.layer_norm_epsilon")
        .and_then(jero_core::GgufValue::as_f32)
        .filter(|v| v.is_finite() && *v > 0.0_f32)
        .unwrap_or(1e-6_f32)
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

#[cfg(test)]
mod tests {
    use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
    use jero_core::{ArchRecipe, GgufFile, GgufValue, K3Error, ModelWeights, QuantType};

    use super::{VisionEncoderConfig, VlmMmprojBridge};

    fn pattern_f32(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|idx| {
                let rem = i16::try_from((idx * 13 + seed * 29) % 19).expect("rem fits i16");
                (f32::from(rem) - 9.0) * 0.04_f32
            })
            .collect()
    }

    fn f32_to_le_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(size_of_val(vals));
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[derive(Clone, Copy)]
    enum FfnCase {
        Swapped,
        DuplicateShape,
        SwappedBiasWithNormalWeights,
    }

    fn gemma3_layer0_specs(
        hidden: usize,
        ffn: usize,
        ffn_case: FfnCase,
    ) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
        let (up_shape, down_shape, up_b_len, down_b_len) = match ffn_case {
            FfnCase::Swapped => (vec![ffn, hidden], vec![hidden, ffn], hidden, ffn),
            FfnCase::DuplicateShape => (vec![ffn, hidden], vec![ffn, hidden], hidden, ffn),
            FfnCase::SwappedBiasWithNormalWeights => {
                (vec![hidden, ffn], vec![ffn, hidden], hidden, ffn)
            }
        };
        vec![
            (
                "v.blk.0.attn_q.weight",
                vec![hidden, hidden],
                pattern_f32(hidden * hidden, 4),
            ),
            ("v.blk.0.attn_q.bias", vec![hidden], pattern_f32(hidden, 5)),
            (
                "v.blk.0.attn_k.weight",
                vec![hidden, hidden],
                pattern_f32(hidden * hidden, 6),
            ),
            ("v.blk.0.attn_k.bias", vec![hidden], pattern_f32(hidden, 7)),
            (
                "v.blk.0.attn_v.weight",
                vec![hidden, hidden],
                pattern_f32(hidden * hidden, 8),
            ),
            ("v.blk.0.attn_v.bias", vec![hidden], pattern_f32(hidden, 9)),
            (
                "v.blk.0.attn_out.weight",
                vec![hidden, hidden],
                pattern_f32(hidden * hidden, 10),
            ),
            (
                "v.blk.0.attn_out.bias",
                vec![hidden],
                pattern_f32(hidden, 11),
            ),
            ("v.blk.0.ln1.weight", vec![hidden], vec![1.0_f32; hidden]),
            ("v.blk.0.ln1.bias", vec![hidden], vec![0.0_f32; hidden]),
            (
                "v.blk.0.ffn_up.weight",
                up_shape,
                pattern_f32(ffn * hidden, 12),
            ),
            (
                "v.blk.0.ffn_up.bias",
                vec![up_b_len],
                pattern_f32(up_b_len, 13),
            ),
            (
                "v.blk.0.ffn_down.weight",
                down_shape,
                pattern_f32(hidden * ffn, 14),
            ),
            (
                "v.blk.0.ffn_down.bias",
                vec![down_b_len],
                pattern_f32(down_b_len, 15),
            ),
            ("v.blk.0.ln2.weight", vec![hidden], vec![1.0_f32; hidden]),
            ("v.blk.0.ln2.bias", vec![hidden], vec![0.0_f32; hidden]),
        ]
    }

    fn build_gemma3_gguf(
        hidden: usize,
        ffn: usize,
        patch: usize,
        img_side: usize,
        llm_dim: usize,
        ffn_case: FfnCase,
        soft_norm_len: usize,
    ) -> Vec<u8> {
        let grid = img_side / patch;
        let num_pos = grid * grid;
        let kvs = [
            (
                "clip.vision.image_size",
                GgufValue::Uint32(u32::try_from(img_side).expect("u32")),
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
            (
                "clip.vision.projection_dim",
                GgufValue::Uint32(u32::try_from(llm_dim).expect("u32")),
            ),
            ("clip.vision.block_count", GgufValue::Uint32(1)),
            ("clip.vision.attention.head_count", GgufValue::Uint32(2)),
            (
                "clip.vision.attention.layer_norm_epsilon",
                GgufValue::Float32(1e-6),
            ),
            ("clip.projector_type", GgufValue::String("gemma3")),
            ("clip.use_gelu", GgufValue::Bool(true)),
        ];

        let mut specs_owned: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
            (
                "v.patch_embd.weight",
                vec![patch, patch, 3, hidden],
                pattern_f32(hidden * 3 * patch * patch, 1),
            ),
            ("v.patch_embd.bias", vec![hidden], pattern_f32(hidden, 2)),
            (
                "v.position_embd.weight",
                vec![hidden, num_pos],
                pattern_f32(hidden * num_pos, 3),
            ),
        ];
        specs_owned.extend(gemma3_layer0_specs(hidden, ffn, ffn_case));
        specs_owned.extend([
            ("v.post_ln.weight", vec![hidden], vec![1.0_f32; hidden]),
            ("v.post_ln.bias", vec![hidden], vec![0.0_f32; hidden]),
            (
                "mm.soft_emb_norm.weight",
                vec![soft_norm_len],
                vec![1.1_f32; soft_norm_len],
            ),
            (
                "mm.input_projection.weight",
                vec![llm_dim, hidden],
                pattern_f32(hidden * llm_dim, 16),
            ),
        ]);

        let tensor_specs: Vec<(&str, QuantType, &[usize])> = specs_owned
            .iter()
            .map(|(n, s, _)| (*n, QuantType::F32, s.as_slice()))
            .collect();
        let mut bytes = create_synthetic_gguf_v3_with_kvs("clip", 1, &kvs, &tensor_specs);
        let ranges: Vec<(usize, usize)> = {
            let parsed = GgufFile::parse(&bytes).expect("parse layout");
            specs_owned
                .iter()
                .map(|(n, _, _)| {
                    let tv = parsed.tensor(n).expect("tensor present");
                    (tv.offset(), tv.byte_len())
                })
                .collect()
        };
        for ((off, len), (_, _, vals)) in ranges.into_iter().zip(specs_owned.iter()) {
            let payload = f32_to_le_bytes(vals);
            assert_eq!(payload.len(), len);
            bytes[off..off + len].copy_from_slice(&payload);
        }
        bytes
    }

    #[test]
    fn gemma3_gguf_swapped_ffn_avg_pool_and_soft_emb_norm_hydrate_and_project() {
        let gguf_bytes = build_gemma3_gguf(8, 16, 2, 8, 12, FfnCase::Swapped, 8);
        let gguf = GgufFile::parse(&gguf_bytes).expect("parse gemma3 gguf");

        let bridge = VlmMmprojBridge::from_gguf(&gguf);
        assert!(matches!(bridge.encoder, VisionEncoderConfig::Siglip(_)));
        assert!(!bridge.merge_2x2_pixel_shuffle);
        assert_eq!(bridge.spatial_avg_pool, Some(4));
        assert_eq!(bridge.llm_embed_dim, Some(12));

        if let VisionEncoderConfig::Siglip(ref sig) = bridge.encoder {
            sig.validate_model(gguf.metadata(), &gguf)
                .expect("swapped gemma3 FFN validates cleanly");
        }

        let src_hwc = pattern_f32(8 * 8 * 3, 25);
        let (embeds, n_tok, dim) = bridge
            .encode_rgb_hwc_to_soft_prompt(&gguf, &src_hwc, 8, 8, 8, 8)
            .expect("gemma3 mmproj forward succeeds");
        // 8x8 image with patch_size=2 -> 4x4 grid, 4x4 avg-pool -> 1 token of dim 12
        assert_eq!(n_tok, 1);
        assert_eq!(dim, 12);
        assert_eq!(embeds.len(), 12);
        assert!(embeds.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gemma3_malformed_ffn_and_projector_shapes_fail_closed() {
        // 1. Both ffn_up and ffn_down sharing [ffn, hidden] shape fails closed
        let dup_bytes = build_gemma3_gguf(8, 16, 2, 8, 12, FfnCase::DuplicateShape, 8);
        let dup_gguf = GgufFile::parse(&dup_bytes).expect("parse dup gguf");
        let dup_bridge = VlmMmprojBridge::from_gguf(&dup_gguf);
        if let VisionEncoderConfig::Siglip(ref sig) = dup_bridge.encoder {
            assert!(matches!(
                sig.validate_model(dup_gguf.metadata(), &dup_gguf),
                Err(K3Error::TensorBoundsError { .. })
            ));
        }
        let pixels = pattern_f32(3 * 8 * 8, 31);
        assert!(matches!(
            dup_bridge.encode_image_to_soft_prompt(&dup_gguf, &pixels, 3, 8, 8),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 2. Swapped biases on unswapped weights fails closed
        let bias_bytes =
            build_gemma3_gguf(8, 16, 2, 8, 12, FfnCase::SwappedBiasWithNormalWeights, 8);
        let bias_gguf = GgufFile::parse(&bias_bytes).expect("parse bias mismatch gguf");
        let bias_bridge = VlmMmprojBridge::from_gguf(&bias_gguf);
        assert!(matches!(
            bias_bridge.encode_image_to_soft_prompt(&bias_gguf, &pixels, 3, 8, 8),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 3. Bad mm.soft_emb_norm length (12 != hidden_dim 8) fails closed
        let bad_norm_bytes = build_gemma3_gguf(8, 16, 2, 8, 12, FfnCase::Swapped, 12);
        let bad_norm_gguf = GgufFile::parse(&bad_norm_bytes).expect("parse bad norm gguf");
        let bad_norm_bridge = VlmMmprojBridge::from_gguf(&bad_norm_gguf);
        assert!(matches!(
            bad_norm_bridge.encode_image_to_soft_prompt(&bad_norm_gguf, &pixels, 3, 8, 8),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 4. Patch grid (2x2 for 4x4 image) not divisible by 4x4 avg-pool fails closed
        let ok_bytes = build_gemma3_gguf(8, 16, 2, 8, 12, FfnCase::Swapped, 8);
        let ok_gguf = GgufFile::parse(&ok_bytes).expect("parse ok gguf");
        let ok_bridge = VlmMmprojBridge::from_gguf(&ok_gguf);
        let small_pixels = pattern_f32(3 * 4 * 4, 41);
        assert!(matches!(
            ok_bridge.encode_image_to_soft_prompt(&ok_gguf, &small_pixels, 3, 4, 4),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 5. Enabling both 2x2 pixel-shuffle and 4x4 spatial avg-pool fails closed
        let conflicting = ok_bridge.with_pixel_shuffle_2x2(true);
        assert!(matches!(
            conflicting.encode_image_to_soft_prompt(&ok_gguf, &pixels, 3, 8, 8),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }
}
