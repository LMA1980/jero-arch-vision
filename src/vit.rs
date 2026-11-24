//! Vision Transformer (`ViT`) architecture recipes for `CLIP` and `SigLIP`.
//!
//! Implements [`ArchRecipe`] for [`ClipVisionRecipe`] (`"clip"`) and [`SiglipVisionRecipe`]
//! (`"siglip"`), executing:
//! 1. `Conv2D` patch embedding (`conv2d_patch_embed_f32`),
//! 2. Optional `[CLS]` token prepending (`v.class_embd` for `CLIP`),
//! 3. Dynamic 2D spatial positional interpolation (`interpolate_2d_pos_embed_f32`),
//! 4. Optional `v.pre_ln` and `0..num_layers` pre-`LayerNorm` bidirectional self-attention
//!    (`bidirectional_attention_f32`) + MLP blocks (`QuickGELU` or `GELU_tanh`) + `v.post_ln`,
//! 5. Standalone L2-normalized image embeddings (`[CLS]` + `visual_projection` for `CLIP`; `MAP`
//!    attention pooling or `Mean` pooling for `SigLIP`) written to
//!    [`ExecutionContext::embedding_output_mut`] (and summary logits when non-empty).

use jero_core::{
    ArchRecipe, EmbeddingPooling, ExecutionContext, GgufFile, K3Error, ModelMetadata, ModelWeights,
    QuantType, SafeTensorsFile, TensorDescriptor, TensorView,
};

use crate::kernels::{
    EMBED_POOL_CLS, EMBED_POOL_LAST_TOKEN, EMBED_POOL_MEAN, MatrixSlice, apply_layernorm_seq,
    bidirectional_attention_f32, bounds_err, checked_mul, conv2d_patch_embed_f32, decode_1d_view,
    decode_all_finite, decode_optional_1d, decode_required_1d, gelu_inplace_f32,
    infer_2d_other_dim, interpolate_2d_pos_embed_f32, is_supported_vision_quant, layernorm_f32,
    linear_forward_seq, lookup_first_tensor, lookup_optional_tensor, matvec_slice, mismatch_err,
    pool_and_l2_normalize_f32, quick_gelu_inplace, resize_rgb_bilinear_norm_f32,
    validate_finite_slice, validate_float_view_finite,
};

const CLIP_IMAGE_MEAN: [f32; 3] = [0.481_454_66_f32, 0.457_827_5_f32, 0.408_210_73_f32];
const CLIP_IMAGE_STD: [f32; 3] = [0.268_629_54_f32, 0.261_302_6_f32, 0.275_777_1_f32];
const SIGLIP_IMAGE_MEAN: [f32; 3] = [0.5_f32, 0.5_f32, 0.5_f32];
const SIGLIP_IMAGE_STD: [f32; 3] = [0.5_f32, 0.5_f32, 0.5_f32];

const PATCH_WEIGHT_ALIASES: &[&str] = &[
    "v.patch_embd.weight",
    "patch_embd.weight",
    "vision_model.embeddings.patch_embedding.weight",
    "visual.patch_embed.proj.weight",
    "embeddings.patch_embedding.weight",
    "patch_embedding.weight",
];
const PATCH_BIAS_ALIASES: &[&str] = &[
    "v.patch_embd.bias",
    "patch_embd.bias",
    "vision_model.embeddings.patch_embedding.bias",
    "embeddings.patch_embedding.bias",
    "patch_embedding.bias",
];
const CLS_EMBED_ALIASES: &[&str] = &[
    "v.class_embd",
    "v.class_embd.weight",
    "class_embd",
    "vision_model.embeddings.class_embedding",
    "vision_model.embeddings.class_embedding.weight",
    "embeddings.class_embedding",
];
const POS_EMBED_ALIASES: &[&str] = &[
    "v.position_embd.weight",
    "v.position_embd",
    "position_embd.weight",
    "vision_model.embeddings.position_embedding.weight",
    "embeddings.position_embedding.weight",
    "position_embedding.weight",
];
const PRE_LN_WEIGHT_ALIASES: &[&str] = &[
    "v.pre_ln.weight",
    "pre_ln.weight",
    "vision_model.pre_layrnorm.weight",
    "vision_model.pre_layernorm.weight",
];
const PRE_LN_BIAS_ALIASES: &[&str] = &[
    "v.pre_ln.bias",
    "pre_ln.bias",
    "vision_model.pre_layrnorm.bias",
    "vision_model.pre_layernorm.bias",
];
const POST_LN_WEIGHT_ALIASES: &[&str] = &[
    "v.post_ln.weight",
    "post_ln.weight",
    "v.post_layernorm.weight",
    "vision_model.post_layernorm.weight",
    "post_layernorm.weight",
];
const POST_LN_BIAS_ALIASES: &[&str] = &[
    "v.post_ln.bias",
    "post_ln.bias",
    "v.post_layernorm.bias",
    "vision_model.post_layernorm.bias",
    "post_layernorm.bias",
];
const VISUAL_PROJ_WEIGHT_ALIASES: &[&str] = &[
    "visual_projection.weight",
    "v.proj.weight",
    "v.visual_projection.weight",
    "visual.proj",
    "vision_model.visual_projection.weight",
];
const VISUAL_PROJ_BIAS_ALIASES: &[&str] = &[
    "visual_projection.bias",
    "v.proj.bias",
    "v.visual_projection.bias",
    "vision_model.visual_projection.bias",
];
const MAP_PROBE_ALIASES: &[&str] = &["v.head.probe", "head.probe", "vision_model.head.probe"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VitActivation {
    QuickGelu,
    GeluTanh,
}

/// Encoded Vision Transformer output sequence from [`ClipVisionRecipe::encode_patch_tokens`] or
/// [`SiglipVisionRecipe::encode_patch_tokens`].
#[derive(Debug, Clone, PartialEq)]
pub struct VisionTokens {
    /// Flattened `[seq_len, hidden_dim]` encoder hidden states after `v.post_ln`.
    pub tokens: Vec<f32>,
    /// Total sequence length (`grid_h * grid_w + usize::from(has_cls_token)`).
    pub seq_len: usize,
    /// Spatial patch grid height (`height / patch_size`).
    pub grid_h: usize,
    /// Spatial patch grid width (`width / patch_size`).
    pub grid_w: usize,
    /// Vision hidden embedding dimension per token.
    pub hidden_dim: usize,
    /// Whether row `0` holds the prepended `[CLS]` token (`true` for `CLIP`, `false` for `SigLIP`).
    pub has_cls_token: bool,
}

impl VisionTokens {
    /// Returns the spatial patch tokens `[grid_h * grid_w, hidden_dim]`, stripping the leading
    /// `[CLS]` token when `self.has_cls_token` is `true`.
    #[must_use]
    pub fn spatial_tokens(&self) -> &[f32] {
        if self.has_cls_token && self.tokens.len() >= self.hidden_dim {
            &self.tokens[self.hidden_dim..]
        } else {
            &self.tokens
        }
    }

    /// Returns the number of spatial patch tokens (`grid_h * grid_w`).
    #[must_use]
    pub const fn spatial_token_count(&self) -> usize {
        self.grid_h * self.grid_w
    }
}

/// `CLIP` Vision Transformer recipe (`"clip"`) for standalone image embeddings and VLM backbones.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipVisionRecipe {
    /// Canonical architecture identifier (`"clip"`).
    pub architecture: &'static str,
    /// Nominal training input square image resolution (`224` or `336`).
    pub image_size: usize,
    /// Square convolution patch stride and kernel size (`14`, `16`, or `32`).
    pub patch_size: usize,
    /// Input color channel count (`3` for RGB).
    pub in_channels: usize,
    /// Number of bidirectional `ViT` transformer encoder blocks.
    pub num_layers: usize,
    /// Hidden embedding width per patch token.
    pub hidden_dim: usize,
    /// Feed-forward MLP intermediate width.
    pub ffn_dim: usize,
    /// Number of multi-head bidirectional self-attention heads.
    pub n_heads: usize,
    /// Scalar dimension per attention head (`hidden_dim / n_heads`).
    pub head_dim: usize,
    /// Optional explicit output projection dimension (`visual_projection`), inferred if `None`.
    pub projection_dim: Option<usize>,
    /// `LayerNorm` stability epsilon (`1e-5`).
    pub norm_eps: f32,
    /// Uses `QuickGELU` MLP activation when `true`, or `GELU_tanh` when `false`.
    pub use_quick_gelu: bool,
    /// Per-channel RGB normalization mean used by [`Self::preprocess_rgb_hwc`].
    pub image_mean: [f32; 3],
    /// Per-channel RGB normalization standard deviation used by [`Self::preprocess_rgb_hwc`].
    pub image_std: [f32; 3],
}

impl Default for ClipVisionRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl ClipVisionRecipe {
    /// Creates a default `CLIP ViT-L/14` recipe (`224x224`, `patch_size = 14`, `QuickGELU`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            architecture: "clip",
            image_size: 224,
            patch_size: 14,
            in_channels: 3,
            num_layers: 12,
            hidden_dim: 768,
            ffn_dim: 3072,
            n_heads: 12,
            head_dim: 64,
            projection_dim: None,
            norm_eps: 1e-5,
            use_quick_gelu: true,
            image_mean: CLIP_IMAGE_MEAN,
            image_std: CLIP_IMAGE_STD,
        }
    }

    /// Configures core geometric and transformer dimensions cleanly.
    #[must_use]
    pub const fn with_dims(
        mut self,
        image_size: usize,
        patch_size: usize,
        num_layers: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        n_heads: usize,
    ) -> Self {
        self.image_size = image_size;
        self.patch_size = patch_size;
        self.num_layers = num_layers;
        self.hidden_dim = hidden_dim;
        self.ffn_dim = ffn_dim;
        self.n_heads = n_heads;
        self.head_dim = if let Some(ratio) = hidden_dim.checked_div(n_heads) {
            if ratio > 0 { ratio } else { 1 }
        } else {
            1
        };
        self
    }

    /// Sets an explicit expected `visual_projection` output dimension.
    #[must_use]
    pub const fn with_projection_dim(mut self, projection_dim: usize) -> Self {
        self.projection_dim = Some(projection_dim);
        self
    }

    /// Selects `QuickGELU` (`true`) or `GELU_tanh` (`false`) MLP activation.
    #[must_use]
    pub const fn with_quick_gelu(mut self, enabled: bool) -> Self {
        self.use_quick_gelu = enabled;
        self
    }

    /// Hydrates a [`ClipVisionRecipe`] from parsed [`ModelMetadata`] (GGUF or `SafeTensors`).
    #[must_use]
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        let mut recipe = Self::new();
        let hydrated = hydrate_vit_fields(
            metadata,
            true,
            VitFields {
                image_size: recipe.image_size,
                patch_size: recipe.patch_size,
                in_channels: recipe.in_channels,
                num_layers: recipe.num_layers,
                hidden_dim: recipe.hidden_dim,
                ffn_dim: recipe.ffn_dim,
                n_heads: recipe.n_heads,
                head_dim: recipe.head_dim,
                projection_dim: recipe.projection_dim,
                norm_eps: recipe.norm_eps,
            },
        );
        recipe.image_size = hydrated.image_size;
        recipe.patch_size = hydrated.patch_size;
        recipe.in_channels = hydrated.in_channels;
        recipe.num_layers = hydrated.num_layers;
        recipe.hidden_dim = hydrated.hidden_dim;
        recipe.ffn_dim = hydrated.ffn_dim;
        recipe.n_heads = hydrated.n_heads;
        recipe.head_dim = hydrated.head_dim;
        recipe.projection_dim = hydrated.projection_dim;
        recipe.norm_eps = hydrated.norm_eps;
        recipe
    }

    /// Hydrates a [`ClipVisionRecipe`] from a parsed [`GgufFile`] including `clip.vision.*` keys.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(gguf.metadata());
        let mut fields = VitFields {
            image_size: recipe.image_size,
            patch_size: recipe.patch_size,
            in_channels: recipe.in_channels,
            num_layers: recipe.num_layers,
            hidden_dim: recipe.hidden_dim,
            ffn_dim: recipe.ffn_dim,
            n_heads: recipe.n_heads,
            head_dim: recipe.head_dim,
            projection_dim: recipe.projection_dim,
            norm_eps: recipe.norm_eps,
        };
        apply_gguf_vision_kvs(gguf, &mut fields, &mut recipe.use_quick_gelu);
        recipe.image_size = fields.image_size;
        recipe.patch_size = fields.patch_size;
        recipe.in_channels = fields.in_channels;
        recipe.num_layers = fields.num_layers;
        recipe.hidden_dim = fields.hidden_dim;
        recipe.ffn_dim = fields.ffn_dim;
        recipe.n_heads = fields.n_heads;
        recipe.head_dim = fields.head_dim;
        recipe.projection_dim = fields.projection_dim;
        recipe.norm_eps = fields.norm_eps;
        recipe
    }

    /// Hydrates a [`ClipVisionRecipe`] from a parsed [`SafeTensorsFile`] including `__metadata__`.
    #[must_use]
    pub fn from_safetensors(st: &SafeTensorsFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(st.metadata());
        let mut fields = VitFields {
            image_size: recipe.image_size,
            patch_size: recipe.patch_size,
            in_channels: recipe.in_channels,
            num_layers: recipe.num_layers,
            hidden_dim: recipe.hidden_dim,
            ffn_dim: recipe.ffn_dim,
            n_heads: recipe.n_heads,
            head_dim: recipe.head_dim,
            projection_dim: recipe.projection_dim,
            norm_eps: recipe.norm_eps,
        };
        apply_safetensors_vision_meta(st, &mut fields, Some(&mut recipe.use_quick_gelu));
        recipe.image_size = fields.image_size;
        recipe.patch_size = fields.patch_size;
        recipe.in_channels = fields.in_channels;
        recipe.num_layers = fields.num_layers;
        recipe.hidden_dim = fields.hidden_dim;
        recipe.ffn_dim = fields.ffn_dim;
        recipe.n_heads = fields.n_heads;
        recipe.head_dim = fields.head_dim;
        recipe.projection_dim = fields.projection_dim;
        recipe.norm_eps = fields.norm_eps;
        recipe
    }

    const fn as_vit_config(&self) -> VitEncoderConfig<'_> {
        VitEncoderConfig {
            architecture: self.architecture,
            image_size: self.image_size,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
            num_layers: self.num_layers,
            hidden_dim: self.hidden_dim,
            ffn_dim: self.ffn_dim,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            projection_dim: self.projection_dim,
            norm_eps: self.norm_eps,
            activation: if self.use_quick_gelu {
                VitActivation::QuickGelu
            } else {
                VitActivation::GeluTanh
            },
            use_cls_token: true,
            use_map_head: false,
        }
    }

    /// Bilinearly resizes and channel-normalizes an interleaved `[src_h, src_w, 3]` RGB image
    /// into `[3, dst_h, dst_w]` `CHW` layout suitable for [`ExecutionContext::with_vision_input`].
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if dimensions are zero, overflow `usize`, lengths
    /// mismatch, or input pixels are non-finite.
    pub fn preprocess_rgb_hwc(
        &self,
        src_hwc: &[f32],
        src_h: usize,
        src_w: usize,
        dst_h: usize,
        dst_w: usize,
    ) -> Result<Vec<f32>, K3Error> {
        preprocess_rgb_hwc_impl(
            src_hwc,
            src_h,
            src_w,
            dst_h,
            dst_w,
            self.image_mean,
            self.image_std,
        )
    }

    /// Encodes a validated `CHW` image `[channels, height, width]` through the `CLIP` `ViT`
    /// encoder and returns all post-`LayerNorm` token embeddings (`[CLS]` + spatial patches).
    ///
    /// # Errors
    /// Returns [`K3Error`] if dimensions are invalid, pixels are non-finite, or any required
    /// weight tensor is missing, malformed, or uses an unsupported quantization format.
    pub fn encode_patch_tokens(
        &self,
        weights: &dyn ModelWeights,
        vision_pixels: &[f32],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<VisionTokens, K3Error> {
        run_vit_encoder(
            &self.as_vit_config(),
            weights,
            vision_pixels,
            channels,
            height,
            width,
        )
    }

    /// Encodes a validated `CHW` image into a standalone L2-normalized embedding vector.
    ///
    /// # Errors
    /// Returns [`K3Error`] if vision encoding, projection, pooling, or bounds checks fail.
    #[expect(
        clippy::too_many_arguments,
        reason = "accepts explicit vision tensor geometry and standalone pooling options cleanly"
    )]
    pub fn encode_image_embedding(
        &self,
        weights: &dyn ModelWeights,
        vision_pixels: &[f32],
        channels: usize,
        height: usize,
        width: usize,
        pooling: EmbeddingPooling,
        matryoshka_dim: Option<usize>,
    ) -> Result<Vec<f32>, K3Error> {
        encode_standalone_embedding(
            &self.as_vit_config(),
            weights,
            vision_pixels,
            channels,
            height,
            width,
            pooling,
            matryoshka_dim,
        )
    }
}

impl ArchRecipe for ClipVisionRecipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn requires_kv_cache(&self) -> bool {
        false
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_vision_quant(quant)
    }

    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_standalone_vision_forward(&self.as_vit_config(), ctx, weights)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_standalone_vision_forward(&self.as_vit_config(), ctx, weights)
    }

    fn decode_step(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        Err(mismatch_err(
            self.architecture,
            "vision encoder 'clip' is single-pass and does not support autoregressive decode_step",
        ))
    }
}

/// `SigLIP` Vision Transformer recipe (`"siglip"`) with `GELU_tanh` MLP and `MAP` attention pooling.
#[derive(Debug, Clone, PartialEq)]
pub struct SiglipVisionRecipe {
    /// Canonical architecture identifier (`"siglip"`).
    pub architecture: &'static str,
    /// Nominal training input square image resolution (`384` or `224`).
    pub image_size: usize,
    /// Square convolution patch stride and kernel size (`14` or `16`).
    pub patch_size: usize,
    /// Input color channel count (`3` for RGB).
    pub in_channels: usize,
    /// Number of bidirectional `ViT` transformer encoder blocks.
    pub num_layers: usize,
    /// Hidden embedding width per patch token.
    pub hidden_dim: usize,
    /// Feed-forward MLP intermediate width.
    pub ffn_dim: usize,
    /// Number of multi-head bidirectional self-attention heads.
    pub n_heads: usize,
    /// Scalar dimension per attention head (`hidden_dim / n_heads`).
    pub head_dim: usize,
    /// Optional explicit output projection dimension (`visual_projection`), inferred if `None`.
    pub projection_dim: Option<usize>,
    /// `LayerNorm` stability epsilon (`1e-6`).
    pub norm_eps: f32,
    /// Enables Multi-head Attention Pooling (`MAP` probe) when `head.probe` weights exist.
    pub use_map_head: bool,
    /// Per-channel RGB normalization mean used by [`Self::preprocess_rgb_hwc`].
    pub image_mean: [f32; 3],
    /// Per-channel RGB normalization standard deviation used by [`Self::preprocess_rgb_hwc`].
    pub image_std: [f32; 3],
}

impl Default for SiglipVisionRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl SiglipVisionRecipe {
    /// Creates a default `SigLIP` recipe (`384x384`, `patch_size = 14`, `GELU_tanh`, `MAP` head).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            architecture: "siglip",
            image_size: 384,
            patch_size: 14,
            in_channels: 3,
            num_layers: 12,
            hidden_dim: 768,
            ffn_dim: 3072,
            n_heads: 12,
            head_dim: 64,
            projection_dim: None,
            norm_eps: 1e-6,
            use_map_head: true,
            image_mean: SIGLIP_IMAGE_MEAN,
            image_std: SIGLIP_IMAGE_STD,
        }
    }

    /// Configures core geometric and transformer dimensions cleanly.
    #[must_use]
    pub const fn with_dims(
        mut self,
        image_size: usize,
        patch_size: usize,
        num_layers: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        n_heads: usize,
    ) -> Self {
        self.image_size = image_size;
        self.patch_size = patch_size;
        self.num_layers = num_layers;
        self.hidden_dim = hidden_dim;
        self.ffn_dim = ffn_dim;
        self.n_heads = n_heads;
        self.head_dim = if let Some(ratio) = hidden_dim.checked_div(n_heads) {
            if ratio > 0 { ratio } else { 1 }
        } else {
            1
        };
        self
    }

    /// Sets an explicit expected `visual_projection` output dimension.
    #[must_use]
    pub const fn with_projection_dim(mut self, projection_dim: usize) -> Self {
        self.projection_dim = Some(projection_dim);
        self
    }

    /// Controls whether the `MAP` attention pooling probe is invoked when present in weights.
    #[must_use]
    pub const fn with_map_head(mut self, enabled: bool) -> Self {
        self.use_map_head = enabled;
        self
    }

    /// Hydrates a [`SiglipVisionRecipe`] from parsed [`ModelMetadata`] (GGUF or `SafeTensors`).
    #[must_use]
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        let mut recipe = Self::new();
        let hydrated = hydrate_vit_fields(
            metadata,
            false,
            VitFields {
                image_size: recipe.image_size,
                patch_size: recipe.patch_size,
                in_channels: recipe.in_channels,
                num_layers: recipe.num_layers,
                hidden_dim: recipe.hidden_dim,
                ffn_dim: recipe.ffn_dim,
                n_heads: recipe.n_heads,
                head_dim: recipe.head_dim,
                projection_dim: recipe.projection_dim,
                norm_eps: recipe.norm_eps,
            },
        );
        recipe.image_size = hydrated.image_size;
        recipe.patch_size = hydrated.patch_size;
        recipe.in_channels = hydrated.in_channels;
        recipe.num_layers = hydrated.num_layers;
        recipe.hidden_dim = hydrated.hidden_dim;
        recipe.ffn_dim = hydrated.ffn_dim;
        recipe.n_heads = hydrated.n_heads;
        recipe.head_dim = hydrated.head_dim;
        recipe.projection_dim = hydrated.projection_dim;
        recipe.norm_eps = hydrated.norm_eps;
        recipe
    }

    /// Hydrates a [`SiglipVisionRecipe`] from a parsed [`GgufFile`] including `clip.vision.*` keys.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(gguf.metadata());
        let mut fields = VitFields {
            image_size: recipe.image_size,
            patch_size: recipe.patch_size,
            in_channels: recipe.in_channels,
            num_layers: recipe.num_layers,
            hidden_dim: recipe.hidden_dim,
            ffn_dim: recipe.ffn_dim,
            n_heads: recipe.n_heads,
            head_dim: recipe.head_dim,
            projection_dim: recipe.projection_dim,
            norm_eps: recipe.norm_eps,
        };
        let mut unused_gelu = false;
        apply_gguf_vision_kvs(gguf, &mut fields, &mut unused_gelu);
        recipe.image_size = fields.image_size;
        recipe.patch_size = fields.patch_size;
        recipe.in_channels = fields.in_channels;
        recipe.num_layers = fields.num_layers;
        recipe.hidden_dim = fields.hidden_dim;
        recipe.ffn_dim = fields.ffn_dim;
        recipe.n_heads = fields.n_heads;
        recipe.head_dim = fields.head_dim;
        recipe.projection_dim = fields.projection_dim;
        recipe.norm_eps = fields.norm_eps;
        recipe
    }

    /// Hydrates a [`SiglipVisionRecipe`] from a parsed [`SafeTensorsFile`] including `__metadata__`.
    #[must_use]
    pub fn from_safetensors(st: &SafeTensorsFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(st.metadata());
        let mut fields = VitFields {
            image_size: recipe.image_size,
            patch_size: recipe.patch_size,
            in_channels: recipe.in_channels,
            num_layers: recipe.num_layers,
            hidden_dim: recipe.hidden_dim,
            ffn_dim: recipe.ffn_dim,
            n_heads: recipe.n_heads,
            head_dim: recipe.head_dim,
            projection_dim: recipe.projection_dim,
            norm_eps: recipe.norm_eps,
        };
        apply_safetensors_vision_meta(st, &mut fields, None);
        recipe.image_size = fields.image_size;
        recipe.patch_size = fields.patch_size;
        recipe.in_channels = fields.in_channels;
        recipe.num_layers = fields.num_layers;
        recipe.hidden_dim = fields.hidden_dim;
        recipe.ffn_dim = fields.ffn_dim;
        recipe.n_heads = fields.n_heads;
        recipe.head_dim = fields.head_dim;
        recipe.projection_dim = fields.projection_dim;
        recipe.norm_eps = fields.norm_eps;
        recipe
    }

    const fn as_vit_config(&self) -> VitEncoderConfig<'_> {
        VitEncoderConfig {
            architecture: self.architecture,
            image_size: self.image_size,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
            num_layers: self.num_layers,
            hidden_dim: self.hidden_dim,
            ffn_dim: self.ffn_dim,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            projection_dim: self.projection_dim,
            norm_eps: self.norm_eps,
            activation: VitActivation::GeluTanh,
            use_cls_token: false,
            use_map_head: self.use_map_head,
        }
    }

    /// Bilinearly resizes and channel-normalizes an interleaved `[src_h, src_w, 3]` RGB image
    /// into `[3, dst_h, dst_w]` `CHW` layout suitable for [`ExecutionContext::with_vision_input`].
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if dimensions are zero, overflow `usize`, lengths
    /// mismatch, or input pixels are non-finite.
    pub fn preprocess_rgb_hwc(
        &self,
        src_hwc: &[f32],
        src_h: usize,
        src_w: usize,
        dst_h: usize,
        dst_w: usize,
    ) -> Result<Vec<f32>, K3Error> {
        preprocess_rgb_hwc_impl(
            src_hwc,
            src_h,
            src_w,
            dst_h,
            dst_w,
            self.image_mean,
            self.image_std,
        )
    }

    /// Encodes a validated `CHW` image `[channels, height, width]` through the `SigLIP` `ViT`
    /// encoder and returns all post-`LayerNorm` patch embeddings (`[grid_h * grid_w, hidden_dim]`).
    ///
    /// # Errors
    /// Returns [`K3Error`] if dimensions are invalid, pixels are non-finite, or any required
    /// weight tensor is missing, malformed, or uses an unsupported quantization format.
    pub fn encode_patch_tokens(
        &self,
        weights: &dyn ModelWeights,
        vision_pixels: &[f32],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<VisionTokens, K3Error> {
        run_vit_encoder(
            &self.as_vit_config(),
            weights,
            vision_pixels,
            channels,
            height,
            width,
        )
    }

    /// Encodes a validated `CHW` image into a standalone L2-normalized embedding vector.
    ///
    /// # Errors
    /// Returns [`K3Error`] if vision encoding, pooling, projection, or bounds checks fail.
    #[expect(
        clippy::too_many_arguments,
        reason = "accepts explicit vision tensor geometry and standalone pooling options cleanly"
    )]
    pub fn encode_image_embedding(
        &self,
        weights: &dyn ModelWeights,
        vision_pixels: &[f32],
        channels: usize,
        height: usize,
        width: usize,
        pooling: EmbeddingPooling,
        matryoshka_dim: Option<usize>,
    ) -> Result<Vec<f32>, K3Error> {
        encode_standalone_embedding(
            &self.as_vit_config(),
            weights,
            vision_pixels,
            channels,
            height,
            width,
            pooling,
            matryoshka_dim,
        )
    }
}

impl ArchRecipe for SiglipVisionRecipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn requires_kv_cache(&self) -> bool {
        false
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_vision_quant(quant)
    }

    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_standalone_vision_forward(&self.as_vit_config(), ctx, weights)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_standalone_vision_forward(&self.as_vit_config(), ctx, weights)
    }

    fn decode_step(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        Err(mismatch_err(
            self.architecture,
            "vision encoder 'siglip' is single-pass and does not support autoregressive decode_step",
        ))
    }
}

// ============================================================================
// Metadata Hydration Helpers
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct VitFields {
    image_size: usize,
    patch_size: usize,
    in_channels: usize,
    num_layers: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    n_heads: usize,
    head_dim: usize,
    projection_dim: Option<usize>,
    norm_eps: f32,
}

fn find_desc<'a>(
    tensors: &'a [TensorDescriptor],
    aliases: &[&str],
) -> Option<&'a TensorDescriptor> {
    aliases
        .iter()
        .find_map(|&name| tensors.iter().find(|d| d.name() == name))
}

fn extract_vision_layer_count(tensors: &[TensorDescriptor]) -> usize {
    const PREFIXES: &[&str] = &[
        "v.blk.",
        "blk.",
        "vision_model.encoder.layers.",
        "encoder.layers.",
        "blocks.",
    ];
    let mut max_idx: Option<usize> = None;
    for desc in tensors {
        for prefix in PREFIXES {
            if let Some(rest) = desc.name().strip_prefix(prefix)
                && let Some((idx_str, _)) = rest.split_once('.')
                && let Ok(idx) = idx_str.parse::<usize>()
            {
                max_idx = Some(max_idx.map_or(idx, |cur| cur.max(idx)));
            }
        }
    }
    max_idx.map_or(0, |idx| idx.saturating_add(1))
}

fn hydrate_vit_fields(
    metadata: &ModelMetadata,
    use_cls_token: bool,
    mut fields: VitFields,
) -> VitFields {
    let tensors = metadata.tensors();

    let inferred_layers = extract_vision_layer_count(tensors);
    if inferred_layers > 0 {
        fields.num_layers = inferred_layers;
    } else if metadata.layer_count() > 0 {
        fields.num_layers = metadata.layer_count();
    }

    if let Some(patch_desc) = find_desc(tensors, PATCH_WEIGHT_ALIASES) {
        let shape = patch_desc.shape();
        if shape.len() == 4 {
            // SafeTensors [O, C, kH, kW] vs GGUF [kW, kH, C, O]
            if shape[0] == shape[1] && shape[2] <= 4 && shape[3] > 0 {
                fields.patch_size = shape[0];
                fields.in_channels = shape[2];
                fields.hidden_dim = shape[3];
            } else if shape[2] == shape[3] && shape[1] <= 4 && shape[0] > 0 {
                fields.hidden_dim = shape[0];
                fields.in_channels = shape[1];
                fields.patch_size = shape[2];
            }
        }
    }

    if let Some(post_desc) = find_desc(tensors, POST_LN_WEIGHT_ALIASES)
        && post_desc.shape().len() == 1
        && post_desc.shape()[0] > 0
    {
        fields.hidden_dim = post_desc.shape()[0];
    } else if metadata.embedding_dim() > 0 {
        fields.hidden_dim = metadata.embedding_dim();
    }

    if let Some(up_desc) = find_desc(
        tensors,
        &[
            "v.blk.0.ffn_up.weight",
            "blk.0.ffn_up.weight",
            "v.blk.0.ffn_fc1.weight",
            "vision_model.encoder.layers.0.mlp.fc1.weight",
            "encoder.layers.0.mlp.fc1.weight",
            "blocks.0.mlp.fc1.weight",
        ],
    ) && let Some(ffn) = infer_other_dim_from_shape(up_desc.shape(), fields.hidden_dim)
    {
        fields.ffn_dim = ffn;
    } else if let Some(ffn) = metadata.feed_forward_length().filter(|&v| v > 0) {
        fields.ffn_dim = ffn;
    } else {
        fields.ffn_dim = fields.hidden_dim.saturating_mul(4);
    }

    if metadata.head_count() > 0 && fields.hidden_dim % metadata.head_count() == 0 {
        fields.n_heads = metadata.head_count();
    } else if fields.hidden_dim % fields.n_heads != 0 {
        fields.n_heads = if fields.hidden_dim % 64 == 0 {
            fields.hidden_dim / 64
        } else if fields.hidden_dim % 4 == 0 {
            4
        } else if fields.hidden_dim % 2 == 0 {
            2
        } else {
            1
        };
    }
    fields.head_dim = (fields.hidden_dim / fields.n_heads.max(1)).max(1);

    if let Some(pos_desc) = find_desc(tensors, POS_EMBED_ALIASES)
        && fields.hidden_dim > 0
        && pos_desc.num_elements() % fields.hidden_dim == 0
    {
        let total_pos = pos_desc.num_elements() / fields.hidden_dim;
        let spatial = if use_cls_token {
            total_pos.saturating_sub(1)
        } else {
            total_pos
        };
        if let Some(side) = exact_isqrt(spatial) {
            fields.image_size = side.saturating_mul(fields.patch_size);
        }
    }

    if let Some(proj_desc) = find_desc(tensors, VISUAL_PROJ_WEIGHT_ALIASES)
        && let Some(proj_dim) = infer_other_dim_from_shape(proj_desc.shape(), fields.hidden_dim)
    {
        fields.projection_dim = Some(proj_dim);
    }

    if let Some(eps) = metadata.rms_norm_eps() {
        fields.norm_eps = eps;
    }

    fields
}

fn infer_other_dim_from_shape(shape: &[usize], known: usize) -> Option<usize> {
    if shape.len() != 2 || known == 0 {
        return None;
    }
    if shape[0] == known && shape[1] > 0 {
        Some(shape[1])
    } else if shape[1] == known && shape[0] > 0 {
        Some(shape[0])
    } else {
        None
    }
}

fn apply_gguf_vision_kvs(gguf: &GgufFile<'_>, fields: &mut VitFields, use_quick_gelu: &mut bool) {
    if let Some(v) = gguf
        .metadata_value("clip.vision.image_size")
        .and_then(jero_core::GgufValue::as_usize)
        && v > 0
    {
        fields.image_size = v;
    }
    if let Some(v) = gguf
        .metadata_value("clip.vision.patch_size")
        .and_then(jero_core::GgufValue::as_usize)
        && v > 0
    {
        fields.patch_size = v;
    }
    if let Some(v) = gguf
        .metadata_value("clip.vision.embedding_length")
        .and_then(jero_core::GgufValue::as_usize)
        && v > 0
    {
        fields.hidden_dim = v;
    }
    if let Some(v) = gguf
        .metadata_value("clip.vision.feed_forward_length")
        .and_then(jero_core::GgufValue::as_usize)
        && v > 0
    {
        fields.ffn_dim = v;
    }
    if let Some(v) = gguf
        .metadata_value("clip.vision.block_count")
        .and_then(jero_core::GgufValue::as_usize)
        && v > 0
    {
        fields.num_layers = v;
    }
    if let Some(v) = gguf
        .metadata_value("clip.vision.attention.head_count")
        .and_then(jero_core::GgufValue::as_usize)
        && v > 0
        && fields.hidden_dim % v == 0
    {
        fields.n_heads = v;
        fields.head_dim = fields.hidden_dim / v;
    }
    if let Some(v) = gguf
        .metadata_value("clip.vision.attention.layer_norm_epsilon")
        .and_then(jero_core::GgufValue::as_f32)
        && v.is_finite()
        && v > 0.0
    {
        fields.norm_eps = v;
    }
    if let Some(v) = gguf
        .metadata_value("clip.vision.projection_dim")
        .and_then(jero_core::GgufValue::as_usize)
        && v > 0
    {
        fields.projection_dim = Some(v);
    }
    if let Some(use_gelu) = gguf
        .metadata_value("clip.vision.use_gelu")
        .or_else(|| gguf.metadata_value("clip.use_gelu"))
        .and_then(jero_core::GgufValue::as_bool)
    {
        *use_quick_gelu = !use_gelu;
    }
}

fn apply_safetensors_vision_meta(
    st: &SafeTensorsFile<'_>,
    fields: &mut VitFields,
    use_quick_gelu: Option<&mut bool>,
) {
    let meta_usize = |keys: &[&str]| -> Option<usize> {
        keys.iter().find_map(|k| {
            st.metadata_value(k)
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
        })
    };
    if let Some(v) = meta_usize(&["image_size", "vision.image_size"]) {
        fields.image_size = v;
    }
    if let Some(v) = meta_usize(&["patch_size", "vision.patch_size"]) {
        fields.patch_size = v;
    }
    if let Some(v) = meta_usize(&["hidden_size", "vision.hidden_size"]) {
        fields.hidden_dim = v;
    }
    if let Some(v) = meta_usize(&["intermediate_size", "vision.intermediate_size"]) {
        fields.ffn_dim = v;
    }
    if let Some(v) = meta_usize(&["num_hidden_layers", "vision.num_hidden_layers"]) {
        fields.num_layers = v;
    }
    if let Some(v) = meta_usize(&["num_attention_heads", "vision.num_attention_heads"])
        && fields.hidden_dim % v == 0
    {
        fields.n_heads = v;
        fields.head_dim = fields.hidden_dim / v;
    }
    if let Some(v) = meta_usize(&["projection_dim", "vision.projection_dim"]) {
        fields.projection_dim = Some(v);
    }
    if let Some(eps) = st
        .metadata_value("layer_norm_eps")
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
    {
        fields.norm_eps = eps;
    }
    if let Some(flag) = use_quick_gelu
        && let Some(act) = st.metadata_value("hidden_act")
    {
        if act.eq_ignore_ascii_case("quick_gelu") {
            *flag = true;
        } else if act.eq_ignore_ascii_case("gelu") || act.eq_ignore_ascii_case("gelu_pytorch_tanh")
        {
            *flag = false;
        }
    }
}

#[must_use]
fn exact_isqrt(n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let root = n.isqrt();
    if root * root == n { Some(root) } else { None }
}

// ============================================================================
// Shared ViT Encoder & Standalone Embedding Engine
// ============================================================================

struct VitEncoderConfig<'a> {
    architecture: &'a str,
    image_size: usize,
    patch_size: usize,
    in_channels: usize,
    num_layers: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    n_heads: usize,
    head_dim: usize,
    projection_dim: Option<usize>,
    norm_eps: f32,
    activation: VitActivation,
    use_cls_token: bool,
    use_map_head: bool,
}

fn preprocess_rgb_hwc_impl(
    src_hwc: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Vec<f32>, K3Error> {
    let dst_pixels = checked_mul("preprocess_rgb_dst", dst_h, dst_w)?;
    let dst_elems = checked_mul("preprocess_rgb_dst", 3, dst_pixels)?;
    let mut out_chw = vec![0.0_f32; dst_elems];
    resize_rgb_bilinear_norm_f32(&mut out_chw, src_hwc, src_h, src_w, dst_h, dst_w, mean, std)?;
    Ok(out_chw)
}

fn validate_encoder_config(cfg: &VitEncoderConfig<'_>) -> Result<usize, K3Error> {
    if cfg.image_size == 0
        || cfg.patch_size == 0
        || cfg.in_channels == 0
        || cfg.num_layers == 0
        || cfg.hidden_dim == 0
        || cfg.ffn_dim == 0
        || cfg.n_heads == 0
        || cfg.head_dim == 0
    {
        return Err(mismatch_err(
            cfg.architecture,
            "vision recipe hyperparameters must all be non-zero",
        ));
    }
    if !cfg.norm_eps.is_finite() || cfg.norm_eps <= 0.0_f32 {
        return Err(mismatch_err(
            cfg.architecture,
            format!("vision norm_eps ({}) must be finite and > 0", cfg.norm_eps),
        ));
    }
    checked_mul(cfg.architecture, cfg.n_heads, cfg.head_dim)
}

fn validate_vision_input_pixels(
    cfg: &VitEncoderConfig<'_>,
    vision_pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<(usize, usize, usize), K3Error> {
    if channels != cfg.in_channels || height == 0 || width == 0 {
        return Err(bounds_err(
            "vision_input",
            cfg.in_channels,
            channels,
            format!(
                "invalid vision input dims ({channels}, {height}, {width}); expected channels={} and non-zero height/width",
                cfg.in_channels
            ),
        ));
    }
    if height % cfg.patch_size != 0 || width % cfg.patch_size != 0 {
        return Err(bounds_err(
            "vision_input",
            0,
            0,
            format!(
                "vision image ({height}x{width}) must be divisible by patch_size ({})",
                cfg.patch_size
            ),
        ));
    }
    let hw = checked_mul("vision_input", height, width)?;
    let expected_elems = checked_mul("vision_input", channels, hw)?;
    if vision_pixels.len() != expected_elems {
        return Err(bounds_err(
            "vision_input",
            expected_elems.saturating_mul(size_of::<f32>()),
            size_of_val(vision_pixels),
            format!(
                "vision_input length {} != expected {expected_elems} ({channels}x{height}x{width})",
                vision_pixels.len()
            ),
        ));
    }
    validate_finite_slice("vision_input", vision_pixels)?;
    let grid_h = height / cfg.patch_size;
    let grid_w = width / cfg.patch_size;
    let n_patches = checked_mul("vision_grid", grid_h, grid_w)?;
    Ok((grid_h, grid_w, n_patches))
}

fn validate_patch_kernel_shape(
    cfg: &VitEncoderConfig<'_>,
    view: &TensorView<'_>,
) -> Result<(), K3Error> {
    let p = cfg.patch_size;
    let c = cfg.in_channels;
    let d = cfg.hidden_dim;
    let patch_flat = checked_mul(
        "v.patch_embd.weight",
        c,
        checked_mul("v.patch_embd.weight", p, p)?,
    )?;
    let expected_elems = checked_mul("v.patch_embd.weight", d, patch_flat)?;

    let shape = view.shape();
    let valid_shape = match shape.len() {
        4 => {
            (shape[0] == d && shape[1] == c && shape[2] == p && shape[3] == p)
                || (shape[0] == p && shape[1] == p && shape[2] == c && shape[3] == d)
        }
        2 => (shape[0] == d && shape[1] == patch_flat) || (shape[0] == patch_flat && shape[1] == d),
        _ => false,
    };
    if !valid_shape || view.num_elements() != expected_elems {
        return Err(bounds_err(
            view.name(),
            expected_elems.saturating_mul(size_of::<f32>()),
            view.byte_len(),
            format!(
                "patch embedding '{}' shape {shape:?} does not match [hidden={d}, channels={c}, patch={p}x{p}]",
                view.name()
            ),
        ));
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "patch embedding helper accepts image slices and pre-computed patch grid geometry"
)]
fn embed_patches_and_positions(
    cfg: &VitEncoderConfig<'_>,
    weights: &dyn ModelWeights,
    vision_pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    grid_h: usize,
    grid_w: usize,
    n_patches: usize,
) -> Result<(Vec<f32>, usize), K3Error> {
    let hidden_dim = cfg.hidden_dim;
    let patch_view = lookup_first_tensor(weights, PATCH_WEIGHT_ALIASES, "v.patch_embd.weight")?;
    validate_patch_kernel_shape(cfg, &patch_view)?;
    let patch_kernel = decode_all_finite(&patch_view, cfg.architecture)?;
    let patch_bias = decode_optional_1d(weights, PATCH_BIAS_ALIASES, hidden_dim, cfg.architecture)?;

    let seq_len = if cfg.use_cls_token {
        n_patches
            .checked_add(1)
            .ok_or_else(|| bounds_err("vision_seq_len", usize::MAX, 0, "seq_len overflow"))?
    } else {
        n_patches
    };
    let seq_elems = checked_mul("vision_seq", seq_len, hidden_dim)?;
    let mut seq_hidden = vec![0.0_f32; seq_elems];

    let patch_offset = if cfg.use_cls_token {
        let cls_vec = decode_required_1d(
            weights,
            CLS_EMBED_ALIASES,
            hidden_dim,
            "v.class_embd",
            cfg.architecture,
        )?;
        seq_hidden[..hidden_dim].copy_from_slice(&cls_vec);
        hidden_dim
    } else {
        0
    };

    conv2d_patch_embed_f32(
        &mut seq_hidden[patch_offset..],
        vision_pixels,
        channels,
        height,
        width,
        cfg.patch_size,
        &patch_kernel,
        patch_bias.as_deref(),
        hidden_dim,
    )?;

    add_interpolated_positions(
        cfg,
        weights,
        &mut seq_hidden,
        patch_offset,
        grid_h,
        grid_w,
        n_patches,
    )?;

    if let Some(pre_ln_w) = lookup_optional_tensor(weights, PRE_LN_WEIGHT_ALIASES)? {
        let w = decode_1d_view(&pre_ln_w, hidden_dim, cfg.architecture)?;
        let b = decode_optional_1d(weights, PRE_LN_BIAS_ALIASES, hidden_dim, cfg.architecture)?;
        let mut normed = vec![0.0_f32; seq_elems];
        apply_layernorm_seq(
            &mut normed,
            &seq_hidden,
            seq_len,
            hidden_dim,
            &w,
            b.as_deref(),
            cfg.norm_eps,
        )?;
        seq_hidden = normed;
    }

    Ok((seq_hidden, seq_len))
}

fn add_interpolated_positions(
    cfg: &VitEncoderConfig<'_>,
    weights: &dyn ModelWeights,
    seq_hidden: &mut [f32],
    patch_offset: usize,
    dst_grid_h: usize,
    dst_grid_w: usize,
    n_patches: usize,
) -> Result<(), K3Error> {
    let hidden_dim = cfg.hidden_dim;
    let pos_view = lookup_first_tensor(weights, POS_EMBED_ALIASES, "v.position_embd.weight")?;
    if pos_view.num_elements() == 0 || pos_view.num_elements() % hidden_dim != 0 {
        return Err(bounds_err(
            pos_view.name(),
            hidden_dim.saturating_mul(size_of::<f32>()),
            pos_view.byte_len(),
            format!(
                "position embedding '{}' element count {} is not a positive multiple of hidden_dim {hidden_dim}",
                pos_view.name(),
                pos_view.num_elements()
            ),
        ));
    }
    let num_positions = pos_view.num_elements() / hidden_dim;
    let pos_shape = pos_view.shape();
    let valid_shape = match pos_shape.len() {
        2 => {
            (pos_shape[0] == hidden_dim && pos_shape[1] == num_positions)
                || (pos_shape[0] == num_positions && pos_shape[1] == hidden_dim)
        }
        3 => pos_shape[0] == 1 && pos_shape[1] == num_positions && pos_shape[2] == hidden_dim,
        _ => false,
    };
    if !valid_shape {
        return Err(bounds_err(
            pos_view.name(),
            pos_view.byte_len(),
            pos_view.byte_len(),
            format!(
                "position embedding '{}' shape {pos_shape:?} incompatible with [{num_positions}, {hidden_dim}]",
                pos_view.name()
            ),
        ));
    }

    let pos_data = decode_all_finite(&pos_view, cfg.architecture)?;
    let spatial_pos = if cfg.use_cls_token {
        if num_positions < 2 {
            return Err(bounds_err(
                pos_view.name(),
                2 * hidden_dim * size_of::<f32>(),
                pos_view.byte_len(),
                "CLIP position embedding requires at least [CLS] + 1 spatial patch row",
            ));
        }
        for (dst, &p) in seq_hidden[..hidden_dim]
            .iter_mut()
            .zip(pos_data[..hidden_dim].iter())
        {
            *dst += p;
        }
        &pos_data[hidden_dim..]
    } else {
        &pos_data[..]
    };

    let spatial_count = spatial_pos.len() / hidden_dim;
    let cfg_side = cfg.image_size / cfg.patch_size;
    let (base_grid_h, base_grid_w) = if cfg_side > 0 && cfg_side * cfg_side == spatial_count {
        (cfg_side, cfg_side)
    } else if let Some(side) = exact_isqrt(spatial_count) {
        (side, side)
    } else if spatial_count == n_patches {
        (dst_grid_h, dst_grid_w)
    } else {
        return Err(bounds_err(
            pos_view.name(),
            n_patches * hidden_dim * size_of::<f32>(),
            size_of_val(spatial_pos),
            format!(
                "cannot infer 2D spatial grid from {spatial_count} position rows (dst {dst_grid_h}x{dst_grid_w})"
            ),
        ));
    };

    let mut interp_pos = vec![0.0_f32; n_patches * hidden_dim];
    interpolate_2d_pos_embed_f32(
        &mut interp_pos,
        spatial_pos,
        base_grid_h,
        base_grid_w,
        dst_grid_h,
        dst_grid_w,
        hidden_dim,
    )?;

    for (dst, &p) in seq_hidden[patch_offset..].iter_mut().zip(interp_pos.iter()) {
        *dst += p;
    }
    Ok(())
}

struct EncoderScratch {
    normed_seq: Vec<f32>,
    queries: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    attn_output: Vec<f32>,
    residual_proj: Vec<f32>,
    mlp_hidden: Vec<f32>,
}

impl EncoderScratch {
    fn new(
        seq_len: usize,
        hidden_dim: usize,
        attn_dim: usize,
        ffn_dim: usize,
    ) -> Result<Self, K3Error> {
        let seq_hidden = checked_mul("encoder_scratch_seq", seq_len, hidden_dim)?;
        let seq_attn = checked_mul("encoder_scratch_attn", seq_len, attn_dim)?;
        let seq_ffn = checked_mul("encoder_scratch_ffn", seq_len, ffn_dim)?;
        Ok(Self {
            normed_seq: vec![0.0_f32; seq_hidden],
            queries: vec![0.0_f32; seq_attn],
            keys: vec![0.0_f32; seq_attn],
            values: vec![0.0_f32; seq_attn],
            attn_output: vec![0.0_f32; seq_attn],
            residual_proj: vec![0.0_f32; seq_hidden],
            mlp_hidden: vec![0.0_f32; seq_ffn],
        })
    }
}

fn layer_aliases(layer: usize, suffixes: &[&str]) -> Vec<String> {
    let prefixes = [
        format!("v.blk.{layer}"),
        format!("blk.{layer}"),
        format!("vision_model.encoder.layers.{layer}"),
        format!("encoder.layers.{layer}"),
        format!("blocks.{layer}"),
    ];
    let mut out = Vec::with_capacity(prefixes.len() * suffixes.len());
    for prefix in &prefixes {
        for suffix in suffixes {
            out.push(format!("{prefix}.{suffix}"));
        }
    }
    out
}

fn lookup_layer_tensor<'w>(
    weights: &'w dyn ModelWeights,
    layer: usize,
    suffixes: &[&str],
    logical_name: &str,
) -> Result<TensorView<'w>, K3Error> {
    let candidates = layer_aliases(layer, suffixes);
    let refs: Vec<&str> = candidates.iter().map(String::as_str).collect();
    lookup_first_tensor(weights, &refs, logical_name)
}

fn lookup_optional_layer_1d(
    weights: &dyn ModelWeights,
    layer: usize,
    suffixes: &[&str],
    expected_len: usize,
    arch: &str,
) -> Result<Option<Vec<f32>>, K3Error> {
    let candidates = layer_aliases(layer, suffixes);
    let refs: Vec<&str> = candidates.iter().map(String::as_str).collect();
    decode_optional_1d(weights, &refs, expected_len, arch)
}

#[expect(
    clippy::too_many_lines,
    reason = "single-layer ViT transformer block keeps pre-LN attention and pre-LN MLP steps readable"
)]
fn run_vit_layer(
    cfg: &VitEncoderConfig<'_>,
    weights: &dyn ModelWeights,
    layer: usize,
    seq_len: usize,
    attn_dim: usize,
    seq_hidden: &mut [f32],
    scratch: &mut EncoderScratch,
) -> Result<(), K3Error> {
    let hidden_dim = cfg.hidden_dim;
    let arch = cfg.architecture;

    // 1. Pre-Attention LayerNorm
    let ln1_w_view = lookup_layer_tensor(
        weights,
        layer,
        &[
            "ln1.weight",
            "attn_norm.weight",
            "layer_norm1.weight",
            "norm1.weight",
        ],
        &format!("v.blk.{layer}.ln1.weight"),
    )?;
    let ln1_w = decode_1d_view(&ln1_w_view, hidden_dim, arch)?;
    let ln1_b = lookup_optional_layer_1d(
        weights,
        layer,
        &[
            "ln1.bias",
            "attn_norm.bias",
            "layer_norm1.bias",
            "norm1.bias",
        ],
        hidden_dim,
        arch,
    )?;
    apply_layernorm_seq(
        &mut scratch.normed_seq,
        seq_hidden,
        seq_len,
        hidden_dim,
        &ln1_w,
        ln1_b.as_deref(),
        cfg.norm_eps,
    )?;

    // 2. Q / K / V projections (separate or fused QKV)
    let q_candidates = layer_aliases(layer, &["attn_q.weight", "self_attn.q_proj.weight"]);
    let q_refs: Vec<&str> = q_candidates.iter().map(String::as_str).collect();
    if let Some(q_view) = lookup_optional_tensor(weights, &q_refs)? {
        let k_view = lookup_layer_tensor(
            weights,
            layer,
            &["attn_k.weight", "self_attn.k_proj.weight"],
            &format!("v.blk.{layer}.attn_k.weight"),
        )?;
        let v_view = lookup_layer_tensor(
            weights,
            layer,
            &["attn_v.weight", "self_attn.v_proj.weight"],
            &format!("v.blk.{layer}.attn_v.weight"),
        )?;
        let q_mat = MatrixSlice::from_view(&q_view, attn_dim, hidden_dim)?;
        let k_mat = MatrixSlice::from_view(&k_view, attn_dim, hidden_dim)?;
        let v_mat = MatrixSlice::from_view(&v_view, attn_dim, hidden_dim)?;

        let q_bias = lookup_optional_layer_1d(
            weights,
            layer,
            &["attn_q.bias", "self_attn.q_proj.bias"],
            attn_dim,
            arch,
        )?;
        let k_bias = lookup_optional_layer_1d(
            weights,
            layer,
            &["attn_k.bias", "self_attn.k_proj.bias"],
            attn_dim,
            arch,
        )?;
        let v_bias = lookup_optional_layer_1d(
            weights,
            layer,
            &["attn_v.bias", "self_attn.v_proj.bias"],
            attn_dim,
            arch,
        )?;

        linear_forward_seq(
            &mut scratch.queries,
            &q_mat,
            q_bias.as_deref(),
            &scratch.normed_seq,
            seq_len,
            hidden_dim,
            attn_dim,
            arch,
        )?;
        linear_forward_seq(
            &mut scratch.keys,
            &k_mat,
            k_bias.as_deref(),
            &scratch.normed_seq,
            seq_len,
            hidden_dim,
            attn_dim,
            arch,
        )?;
        linear_forward_seq(
            &mut scratch.values,
            &v_mat,
            v_bias.as_deref(),
            &scratch.normed_seq,
            seq_len,
            hidden_dim,
            attn_dim,
            arch,
        )?;
    } else {
        let qkv_view = lookup_layer_tensor(
            weights,
            layer,
            &[
                "attn_qkv.weight",
                "attn.qkv.weight",
                "self_attn.qkv.weight",
                "self_attn.in_proj_weight",
                "self_attn.in_proj.weight",
            ],
            &format!("v.blk.{layer}.attn_q.weight"),
        )?;
        let [q_mat, k_mat, v_mat] = MatrixSlice::split_fused_3way(&qkv_view, attn_dim, hidden_dim)?;
        let qkv_bias = lookup_optional_layer_1d(
            weights,
            layer,
            &[
                "attn_qkv.bias",
                "attn.qkv.bias",
                "self_attn.qkv.bias",
                "self_attn.in_proj_bias",
                "self_attn.in_proj.bias",
            ],
            3 * attn_dim,
            arch,
        )?;
        let (qb, kb, vb) = match qkv_bias.as_deref() {
            Some(b) => (
                Some(&b[..attn_dim]),
                Some(&b[attn_dim..2 * attn_dim]),
                Some(&b[2 * attn_dim..]),
            ),
            None => (None, None, None),
        };
        linear_forward_seq(
            &mut scratch.queries,
            &q_mat,
            qb,
            &scratch.normed_seq,
            seq_len,
            hidden_dim,
            attn_dim,
            arch,
        )?;
        linear_forward_seq(
            &mut scratch.keys,
            &k_mat,
            kb,
            &scratch.normed_seq,
            seq_len,
            hidden_dim,
            attn_dim,
            arch,
        )?;
        linear_forward_seq(
            &mut scratch.values,
            &v_mat,
            vb,
            &scratch.normed_seq,
            seq_len,
            hidden_dim,
            attn_dim,
            arch,
        )?;
    }

    // 3. Bidirectional Multi-Head Self-Attention
    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim fits comfortably in f32"
    )]
    let scale = 1.0_f32 / (cfg.head_dim as f32).sqrt();
    bidirectional_attention_f32(
        &mut scratch.attn_output,
        &scratch.queries,
        &scratch.keys,
        &scratch.values,
        seq_len,
        cfg.n_heads,
        cfg.head_dim,
        None,
        scale,
    )?;

    // 4. Attention Output Projection + Residual
    let attn_out_view = lookup_layer_tensor(
        weights,
        layer,
        &[
            "attn_out.weight",
            "attn_output.weight",
            "self_attn.out_proj.weight",
            "attn.proj.weight",
        ],
        &format!("v.blk.{layer}.attn_out.weight"),
    )?;
    let attn_out_mat = MatrixSlice::from_view(&attn_out_view, hidden_dim, attn_dim)?;
    let attn_out_bias = lookup_optional_layer_1d(
        weights,
        layer,
        &[
            "attn_out.bias",
            "attn_output.bias",
            "self_attn.out_proj.bias",
            "attn.proj.bias",
        ],
        hidden_dim,
        arch,
    )?;
    linear_forward_seq(
        &mut scratch.residual_proj,
        &attn_out_mat,
        attn_out_bias.as_deref(),
        &scratch.attn_output,
        seq_len,
        attn_dim,
        hidden_dim,
        arch,
    )?;
    for (dst, &delta) in seq_hidden.iter_mut().zip(scratch.residual_proj.iter()) {
        *dst += delta;
    }

    // 5. Pre-FFN LayerNorm
    let ln2_w_view = lookup_layer_tensor(
        weights,
        layer,
        &[
            "ln2.weight",
            "ffn_norm.weight",
            "layer_norm2.weight",
            "norm2.weight",
        ],
        &format!("v.blk.{layer}.ln2.weight"),
    )?;
    let ln2_w = decode_1d_view(&ln2_w_view, hidden_dim, arch)?;
    let ln2_b = lookup_optional_layer_1d(
        weights,
        layer,
        &[
            "ln2.bias",
            "ffn_norm.bias",
            "layer_norm2.bias",
            "norm2.bias",
        ],
        hidden_dim,
        arch,
    )?;
    apply_layernorm_seq(
        &mut scratch.normed_seq,
        seq_hidden,
        seq_len,
        hidden_dim,
        &ln2_w,
        ln2_b.as_deref(),
        cfg.norm_eps,
    )?;

    // 6. MLP (`ffn_up` -> QuickGELU / GELU_tanh -> `ffn_down`) + Residual
    let ffn_up_view = lookup_layer_tensor(
        weights,
        layer,
        &["ffn_up.weight", "ffn_fc1.weight", "mlp.fc1.weight"],
        &format!("v.blk.{layer}.ffn_up.weight"),
    )?;
    let ffn_up_mat = MatrixSlice::from_view(&ffn_up_view, cfg.ffn_dim, hidden_dim)?;
    let ffn_up_bias = lookup_optional_layer_1d(
        weights,
        layer,
        &["ffn_up.bias", "ffn_fc1.bias", "mlp.fc1.bias"],
        cfg.ffn_dim,
        arch,
    )?;
    linear_forward_seq(
        &mut scratch.mlp_hidden,
        &ffn_up_mat,
        ffn_up_bias.as_deref(),
        &scratch.normed_seq,
        seq_len,
        hidden_dim,
        cfg.ffn_dim,
        arch,
    )?;

    match cfg.activation {
        VitActivation::QuickGelu => quick_gelu_inplace(&mut scratch.mlp_hidden)?,
        VitActivation::GeluTanh => gelu_inplace_f32(&mut scratch.mlp_hidden)?,
    }

    let ffn_down_view = lookup_layer_tensor(
        weights,
        layer,
        &["ffn_down.weight", "ffn_fc2.weight", "mlp.fc2.weight"],
        &format!("v.blk.{layer}.ffn_down.weight"),
    )?;
    let ffn_down_mat = MatrixSlice::from_view(&ffn_down_view, hidden_dim, cfg.ffn_dim)?;
    let ffn_down_bias = lookup_optional_layer_1d(
        weights,
        layer,
        &["ffn_down.bias", "ffn_fc2.bias", "mlp.fc2.bias"],
        hidden_dim,
        arch,
    )?;
    linear_forward_seq(
        &mut scratch.residual_proj,
        &ffn_down_mat,
        ffn_down_bias.as_deref(),
        &scratch.mlp_hidden,
        seq_len,
        cfg.ffn_dim,
        hidden_dim,
        arch,
    )?;
    for (dst, &delta) in seq_hidden.iter_mut().zip(scratch.residual_proj.iter()) {
        *dst += delta;
    }

    Ok(())
}

fn run_vit_encoder(
    cfg: &VitEncoderConfig<'_>,
    weights: &dyn ModelWeights,
    vision_pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<VisionTokens, K3Error> {
    let attn_dim = validate_encoder_config(cfg)?;
    let (grid_h, grid_w, n_patches) =
        validate_vision_input_pixels(cfg, vision_pixels, channels, height, width)?;

    let (mut seq_hidden, seq_len) = embed_patches_and_positions(
        cfg,
        weights,
        vision_pixels,
        channels,
        height,
        width,
        grid_h,
        grid_w,
        n_patches,
    )?;

    let mut scratch = EncoderScratch::new(seq_len, cfg.hidden_dim, attn_dim, cfg.ffn_dim)?;
    for layer in 0..cfg.num_layers {
        run_vit_layer(
            cfg,
            weights,
            layer,
            seq_len,
            attn_dim,
            &mut seq_hidden,
            &mut scratch,
        )?;
    }

    let post_ln_view = lookup_first_tensor(weights, POST_LN_WEIGHT_ALIASES, "v.post_ln.weight")?;
    let post_ln_w = decode_1d_view(&post_ln_view, cfg.hidden_dim, cfg.architecture)?;
    let post_ln_b = decode_optional_1d(
        weights,
        POST_LN_BIAS_ALIASES,
        cfg.hidden_dim,
        cfg.architecture,
    )?;
    apply_layernorm_seq(
        &mut scratch.normed_seq,
        &seq_hidden,
        seq_len,
        cfg.hidden_dim,
        &post_ln_w,
        post_ln_b.as_deref(),
        cfg.norm_eps,
    )?;
    seq_hidden.copy_from_slice(&scratch.normed_seq);
    validate_finite_slice("vit_encoder_output", &seq_hidden)?;

    Ok(VisionTokens {
        tokens: seq_hidden,
        seq_len,
        grid_h,
        grid_w,
        hidden_dim: cfg.hidden_dim,
        has_cls_token: cfg.use_cls_token,
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "MAP attention pooling coordinates Q/K/V cross-attention over probe + LayerNorm + GELU MLP"
)]
fn run_siglip_map_head(
    cfg: &VitEncoderConfig<'_>,
    weights: &dyn ModelWeights,
    tokens: &VisionTokens,
) -> Result<Vec<f32>, K3Error> {
    let hidden_dim = cfg.hidden_dim;
    let arch = cfg.architecture;
    let attn_dim = checked_mul(arch, cfg.n_heads, cfg.head_dim)?;

    let probe = decode_required_1d(weights, MAP_PROBE_ALIASES, hidden_dim, "v.head.probe", arch)?;

    let mut q = vec![0.0_f32; attn_dim];
    let mut k_seq = vec![0.0_f32; tokens.seq_len * attn_dim];
    let mut v_seq = vec![0.0_f32; tokens.seq_len * attn_dim];

    if let Some(in_proj_view) = lookup_optional_tensor(
        weights,
        &[
            "v.head.attn_qkv.weight",
            "v.head.attention.in_proj_weight",
            "head.attention.in_proj_weight",
            "vision_model.head.attention.in_proj_weight",
        ],
    )? {
        let [q_mat, k_mat, v_mat] =
            MatrixSlice::split_fused_3way(&in_proj_view, attn_dim, hidden_dim)?;
        let bias = decode_optional_1d(
            weights,
            &[
                "v.head.attn_qkv.bias",
                "v.head.attention.in_proj_bias",
                "head.attention.in_proj_bias",
                "vision_model.head.attention.in_proj_bias",
            ],
            3 * attn_dim,
            arch,
        )?;
        let (qb, kb, vb) = match bias.as_deref() {
            Some(b) => (
                Some(&b[..attn_dim]),
                Some(&b[attn_dim..2 * attn_dim]),
                Some(&b[2 * attn_dim..]),
            ),
            None => (None, None, None),
        };
        linear_forward_seq(&mut q, &q_mat, qb, &probe, 1, hidden_dim, attn_dim, arch)?;
        linear_forward_seq(
            &mut k_seq,
            &k_mat,
            kb,
            &tokens.tokens,
            tokens.seq_len,
            hidden_dim,
            attn_dim,
            arch,
        )?;
        linear_forward_seq(
            &mut v_seq,
            &v_mat,
            vb,
            &tokens.tokens,
            tokens.seq_len,
            hidden_dim,
            attn_dim,
            arch,
        )?;
    } else {
        let q_view = lookup_first_tensor(
            weights,
            &[
                "v.head.attn_q.weight",
                "head.attention.q_proj.weight",
                "vision_model.head.attention.q_proj.weight",
            ],
            "v.head.attn_q.weight",
        )?;
        let k_view = lookup_first_tensor(
            weights,
            &[
                "v.head.attn_k.weight",
                "head.attention.k_proj.weight",
                "vision_model.head.attention.k_proj.weight",
            ],
            "v.head.attn_k.weight",
        )?;
        let v_view = lookup_first_tensor(
            weights,
            &[
                "v.head.attn_v.weight",
                "head.attention.v_proj.weight",
                "vision_model.head.attention.v_proj.weight",
            ],
            "v.head.attn_v.weight",
        )?;
        let q_mat = MatrixSlice::from_view(&q_view, attn_dim, hidden_dim)?;
        let k_mat = MatrixSlice::from_view(&k_view, attn_dim, hidden_dim)?;
        let v_mat = MatrixSlice::from_view(&v_view, attn_dim, hidden_dim)?;
        let qb = decode_optional_1d(
            weights,
            &[
                "v.head.attn_q.bias",
                "head.attention.q_proj.bias",
                "vision_model.head.attention.q_proj.bias",
            ],
            attn_dim,
            arch,
        )?;
        let kb = decode_optional_1d(
            weights,
            &[
                "v.head.attn_k.bias",
                "head.attention.k_proj.bias",
                "vision_model.head.attention.k_proj.bias",
            ],
            attn_dim,
            arch,
        )?;
        let vb = decode_optional_1d(
            weights,
            &[
                "v.head.attn_v.bias",
                "head.attention.v_proj.bias",
                "vision_model.head.attention.v_proj.bias",
            ],
            attn_dim,
            arch,
        )?;
        linear_forward_seq(
            &mut q,
            &q_mat,
            qb.as_deref(),
            &probe,
            1,
            hidden_dim,
            attn_dim,
            arch,
        )?;
        linear_forward_seq(
            &mut k_seq,
            &k_mat,
            kb.as_deref(),
            &tokens.tokens,
            tokens.seq_len,
            hidden_dim,
            attn_dim,
            arch,
        )?;
        linear_forward_seq(
            &mut v_seq,
            &v_mat,
            vb.as_deref(),
            &tokens.tokens,
            tokens.seq_len,
            hidden_dim,
            attn_dim,
            arch,
        )?;
    }

    // Cross-attention: 1 probe query over all `seq_len` spatial patch keys/values
    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim fits comfortably in f32"
    )]
    let scale = 1.0_f32 / (cfg.head_dim as f32).sqrt();
    let mut attn_ctx = vec![0.0_f32; attn_dim];
    let mut scores = vec![0.0_f32; tokens.seq_len];

    for h in 0..cfg.n_heads {
        let q_h = &q[h * cfg.head_dim..(h + 1) * cfg.head_dim];
        let mut max_s = f32::NEG_INFINITY;
        for (t, score_slot) in scores.iter_mut().enumerate() {
            let k_h = &k_seq[t * attn_dim + h * cfg.head_dim..][..cfg.head_dim];
            let s = crate::kernels::dot_f32(q_h, k_h) * scale;
            *score_slot = s;
            if s > max_s {
                max_s = s;
            }
        }
        let mut sum_w = 0.0_f32;
        for s in &mut scores {
            let w = (*s - max_s).exp();
            *s = w;
            sum_w += w;
        }
        #[expect(clippy::cast_precision_loss, reason = "seq_len fits f32")]
        let inv = if sum_w > 0.0 {
            1.0_f32 / sum_w
        } else {
            1.0_f32 / (tokens.seq_len as f32)
        };
        let out_h = &mut attn_ctx[h * cfg.head_dim..(h + 1) * cfg.head_dim];
        for (t, &s) in scores.iter().enumerate() {
            let w = s * inv;
            let v_h = &v_seq[t * attn_dim + h * cfg.head_dim..][..cfg.head_dim];
            for (dst, &vv) in out_h.iter_mut().zip(v_h.iter()) {
                *dst += w * vv;
            }
        }
    }

    let out_proj_view = lookup_first_tensor(
        weights,
        &[
            "v.head.attn_out.weight",
            "v.head.attention.out_proj.weight",
            "head.attention.out_proj.weight",
            "vision_model.head.attention.out_proj.weight",
        ],
        "v.head.attn_out.weight",
    )?;
    let out_proj_mat = MatrixSlice::from_view(&out_proj_view, hidden_dim, attn_dim)?;
    let out_proj_bias = decode_optional_1d(
        weights,
        &[
            "v.head.attn_out.bias",
            "v.head.attention.out_proj.bias",
            "head.attention.out_proj.bias",
            "vision_model.head.attention.out_proj.bias",
        ],
        hidden_dim,
        arch,
    )?;
    let mut pooled = vec![0.0_f32; hidden_dim];
    linear_forward_seq(
        &mut pooled,
        &out_proj_mat,
        out_proj_bias.as_deref(),
        &attn_ctx,
        1,
        attn_dim,
        hidden_dim,
        arch,
    )?;

    let ln_w = decode_required_1d(
        weights,
        &[
            "v.head.ln.weight",
            "v.head.layernorm.weight",
            "head.layernorm.weight",
            "vision_model.head.layernorm.weight",
        ],
        hidden_dim,
        "v.head.ln.weight",
        arch,
    )?;
    let ln_b = decode_optional_1d(
        weights,
        &[
            "v.head.ln.bias",
            "v.head.layernorm.bias",
            "head.layernorm.bias",
            "vision_model.head.layernorm.bias",
        ],
        hidden_dim,
        arch,
    )?;
    let mut normed = vec![0.0_f32; hidden_dim];
    layernorm_f32(
        &mut normed,
        &pooled,
        Some(&ln_w),
        ln_b.as_deref(),
        cfg.norm_eps,
    )?;

    let fc1_view = lookup_first_tensor(
        weights,
        &[
            "v.head.mlp.fc1.weight",
            "v.head.ffn_up.weight",
            "head.mlp.fc1.weight",
            "vision_model.head.mlp.fc1.weight",
        ],
        "v.head.mlp.fc1.weight",
    )?;
    let head_ffn_dim = infer_2d_other_dim(&fc1_view, hidden_dim, "MAP head fc1")?;
    let fc1_mat = MatrixSlice::from_view(&fc1_view, head_ffn_dim, hidden_dim)?;
    let fc1_bias = decode_optional_1d(
        weights,
        &[
            "v.head.mlp.fc1.bias",
            "v.head.ffn_up.bias",
            "head.mlp.fc1.bias",
            "vision_model.head.mlp.fc1.bias",
        ],
        head_ffn_dim,
        arch,
    )?;
    let mut mlp_mid = vec![0.0_f32; head_ffn_dim];
    linear_forward_seq(
        &mut mlp_mid,
        &fc1_mat,
        fc1_bias.as_deref(),
        &normed,
        1,
        hidden_dim,
        head_ffn_dim,
        arch,
    )?;
    gelu_inplace_f32(&mut mlp_mid)?;

    let fc2_view = lookup_first_tensor(
        weights,
        &[
            "v.head.mlp.fc2.weight",
            "v.head.ffn_down.weight",
            "head.mlp.fc2.weight",
            "vision_model.head.mlp.fc2.weight",
        ],
        "v.head.mlp.fc2.weight",
    )?;
    let fc2_mat = MatrixSlice::from_view(&fc2_view, hidden_dim, head_ffn_dim)?;
    let fc2_bias = decode_optional_1d(
        weights,
        &[
            "v.head.mlp.fc2.bias",
            "v.head.ffn_down.bias",
            "head.mlp.fc2.bias",
            "vision_model.head.mlp.fc2.bias",
        ],
        hidden_dim,
        arch,
    )?;
    let mut mlp_out = vec![0.0_f32; hidden_dim];
    linear_forward_seq(
        &mut mlp_out,
        &fc2_mat,
        fc2_bias.as_deref(),
        &mlp_mid,
        1,
        head_ffn_dim,
        hidden_dim,
        arch,
    )?;

    for (dst, &delta) in pooled.iter_mut().zip(mlp_out.iter()) {
        *dst += delta;
    }
    validate_finite_slice("siglip_map_head", &pooled)?;
    Ok(pooled)
}

fn pool_and_project_embedding(
    cfg: &VitEncoderConfig<'_>,
    weights: &dyn ModelWeights,
    tokens: &VisionTokens,
    pooling: EmbeddingPooling,
    matryoshka_dim: Option<usize>,
) -> Result<Vec<f32>, K3Error> {
    let hidden_dim = cfg.hidden_dim;
    let arch = cfg.architecture;

    let use_map = cfg.use_map_head
        && !matches!(pooling, EmbeddingPooling::Cls | EmbeddingPooling::LastToken)
        && lookup_optional_tensor(weights, MAP_PROBE_ALIASES)?.is_some();

    let proj_view_opt = lookup_optional_tensor(weights, VISUAL_PROJ_WEIGHT_ALIASES)?;

    let (raw_seq, raw_seq_len, mode_code) = if use_map {
        let map_vec = run_siglip_map_head(cfg, weights, tokens)?;
        (Some(map_vec), 1, EMBED_POOL_CLS)
    } else {
        let mode = match pooling {
            EmbeddingPooling::None => {
                if cfg.use_cls_token {
                    EMBED_POOL_CLS
                } else {
                    EMBED_POOL_MEAN
                }
            }
            EmbeddingPooling::Mean => EMBED_POOL_MEAN,
            EmbeddingPooling::Cls => EMBED_POOL_CLS,
            EmbeddingPooling::LastToken => EMBED_POOL_LAST_TOKEN,
        };
        (None, tokens.seq_len, mode)
    };

    if let Some(proj_view) = proj_view_opt {
        validate_float_view_finite(&proj_view)?;
        let out_dim = match cfg.projection_dim {
            Some(p) => {
                let mat = MatrixSlice::from_view(&proj_view, p, hidden_dim)?;
                let _ = mat;
                p
            }
            None => infer_2d_other_dim(&proj_view, hidden_dim, "visual_projection")?,
        };
        let proj_mat = MatrixSlice::from_view(&proj_view, out_dim, hidden_dim)?;
        let proj_bias = decode_optional_1d(weights, VISUAL_PROJ_BIAS_ALIASES, out_dim, arch)?;

        let pooled_vec = if let Some(v) = raw_seq {
            v
        } else {
            extract_unnormalized_pool(&tokens.tokens, raw_seq_len, hidden_dim, mode_code)
        };
        let mut projected = vec![0.0_f32; out_dim];
        matvec_slice(&mut projected, &proj_mat, &pooled_vec, arch)?;
        if let Some(b) = proj_bias.as_deref() {
            for (dst, &bv) in projected.iter_mut().zip(b.iter()) {
                *dst += bv;
            }
        }
        validate_finite_slice(proj_view.name(), &projected)?;
        let eff_dim = resolve_matryoshka_dim(out_dim, matryoshka_dim)?;
        let mut final_embed = vec![0.0_f32; eff_dim];
        pool_and_l2_normalize_f32(
            &mut final_embed,
            &projected,
            1,
            out_dim,
            EMBED_POOL_CLS,
            matryoshka_dim,
        )?;
        Ok(final_embed)
    } else {
        if let Some(expected_proj) = cfg.projection_dim
            && expected_proj != hidden_dim
        {
            return Err(bounds_err(
                "visual_projection.weight",
                expected_proj * hidden_dim * size_of::<f32>(),
                0,
                format!(
                    "projection_dim ({expected_proj}) != hidden_dim ({hidden_dim}) requires visual_projection weights"
                ),
            ));
        }
        let eff_dim = resolve_matryoshka_dim(hidden_dim, matryoshka_dim)?;
        let mut final_embed = vec![0.0_f32; eff_dim];
        let src = raw_seq.as_deref().unwrap_or(&tokens.tokens);
        pool_and_l2_normalize_f32(
            &mut final_embed,
            src,
            raw_seq_len,
            hidden_dim,
            mode_code,
            matryoshka_dim,
        )?;
        Ok(final_embed)
    }
}

fn extract_unnormalized_pool(
    seq: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    mode_code: u8,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; hidden_dim];
    match mode_code {
        EMBED_POOL_CLS => out.copy_from_slice(&seq[..hidden_dim]),
        EMBED_POOL_LAST_TOKEN => {
            let start = (seq_len - 1) * hidden_dim;
            out.copy_from_slice(&seq[start..start + hidden_dim]);
        }
        EMBED_POOL_MEAN => {
            for s in 0..seq_len {
                let row = &seq[s * hidden_dim..(s + 1) * hidden_dim];
                for (d, &val) in out.iter_mut().zip(row.iter()) {
                    *d += val;
                }
            }
            #[expect(
                clippy::cast_precision_loss,
                reason = "seq_len fits comfortably in f32"
            )]
            let inv = 1.0_f32 / (seq_len as f32);
            for d in &mut out {
                *d *= inv;
            }
        }
        _ => {}
    }
    out
}

fn resolve_matryoshka_dim(
    full_dim: usize,
    matryoshka_dim: Option<usize>,
) -> Result<usize, K3Error> {
    match matryoshka_dim {
        None => Ok(full_dim),
        Some(d) if d == 0 || d > full_dim => Err(bounds_err(
            "matryoshka_dim",
            full_dim,
            d,
            format!("matryoshka_dim ({d}) must lie within 1..={full_dim}"),
        )),
        Some(d) => Ok(d),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "standalone image embedding API takes explicit image tensor, pooling, and Matryoshka options"
)]
fn encode_standalone_embedding(
    cfg: &VitEncoderConfig<'_>,
    weights: &dyn ModelWeights,
    vision_pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    pooling: EmbeddingPooling,
    matryoshka_dim: Option<usize>,
) -> Result<Vec<f32>, K3Error> {
    let tokens = run_vit_encoder(cfg, weights, vision_pixels, channels, height, width)?;
    pool_and_project_embedding(cfg, weights, &tokens, pooling, matryoshka_dim)
}

fn run_standalone_vision_forward(
    cfg: &VitEncoderConfig<'_>,
    ctx: &mut ExecutionContext<'_>,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let vision_pixels = ctx.vision_input().ok_or_else(|| {
        bounds_err(
            "vision_input",
            size_of::<f32>(),
            0,
            format!(
                "'{}' requires vision_input attached on ExecutionContext",
                cfg.architecture
            ),
        )
    })?;
    let (channels, height, width) = ctx.vision_dims().ok_or_else(|| {
        bounds_err(
            "vision_dims",
            size_of::<usize>() * 3,
            0,
            format!(
                "'{}' requires vision_dims attached on ExecutionContext",
                cfg.architecture
            ),
        )
    })?;

    let has_emb = ctx.embedding_output().is_some();
    let logits_len = ctx.logits().len();
    if !has_emb && logits_len == 0 {
        return Err(bounds_err(
            "embedding_output",
            cfg.hidden_dim.saturating_mul(size_of::<f32>()),
            0,
            format!(
                "'{}' requires either embedding_output or non-empty logits buffer",
                cfg.architecture
            ),
        ));
    }

    let pooling = ctx.embedding_pooling();
    let matryoshka_dim = ctx.matryoshka_dim();
    let embed = encode_standalone_embedding(
        cfg,
        weights,
        vision_pixels,
        channels,
        height,
        width,
        pooling,
        matryoshka_dim,
    )?;

    if let Some(emb_out) = ctx.embedding_output_mut() {
        if emb_out.len() < embed.len() {
            return Err(bounds_err(
                "embedding_output",
                embed.len().saturating_mul(size_of::<f32>()),
                size_of_val(emb_out),
                format!(
                    "embedding_output length {} < required vision embedding length {}",
                    emb_out.len(),
                    embed.len()
                ),
            ));
        }
        emb_out[..embed.len()].copy_from_slice(&embed);
        emb_out[embed.len()..].fill(0.0_f32);
    }

    if logits_len > 0 {
        let logits = ctx.logits_mut();
        logits.fill(0.0_f32);
        let copy_len = logits_len.min(embed.len());
        logits[..copy_len].copy_from_slice(&embed[..copy_len]);
    }

    Ok(())
}
