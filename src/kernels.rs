//! Kernel dispatch facade and fail-closed tensor validation helpers for `jero-arch-vision`.
//!
//! Dispatches compute primitives to `jero-kernels-rvv` when the default `with-kernels` feature is
//! enabled and provides portable scalar equivalents under `--no-default-features` so every test and
//! validation gate compiles and passes across both feature configurations.

use jero_core::{K3Error, ModelWeights, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    BaselineKernel, VectorKernel, bf16_to_f32_ieee, bidirectional_attention_f32 as rvv_bidir_attn,
    conv2d_patch_embed_f32 as rvv_conv2d_patch_embed, dequantize_token_row, f16_to_f32_ieee,
    gelu_inplace_f32 as rvv_gelu_inplace, interpolate_2d_pos_embed_f32 as rvv_interp_2d_pos,
    layernorm_f32 as rvv_layernorm, pixel_shuffle_merge_2x2_f32 as rvv_pixel_shuffle_2x2,
    pool_and_l2_normalize_f32 as rvv_pool_and_l2_norm,
    resize_rgb_bilinear_norm_f32 as rvv_resize_rgb, vec_dot_bf16, vec_dot_f16, vec_dot_q2_k,
    vec_dot_q3_k, vec_dot_q4_0, vec_dot_q4_1, vec_dot_q5_0, vec_dot_q5_1, vec_dot_q5_k,
    vec_dot_q6_k, vec_dot_q8_0,
};

#[cfg(feature = "with-kernels")]
pub(crate) use jero_kernels_rvv::{EMBED_POOL_CLS, EMBED_POOL_LAST_TOKEN, EMBED_POOL_MEAN};

#[cfg(not(feature = "with-kernels"))]
pub(crate) const EMBED_POOL_MEAN: u8 = 1;
#[cfg(not(feature = "with-kernels"))]
pub(crate) const EMBED_POOL_CLS: u8 = 2;
#[cfg(not(feature = "with-kernels"))]
pub(crate) const EMBED_POOL_LAST_TOKEN: u8 = 3;

#[inline]
pub(crate) fn bounds_err(
    tensor_name: &str,
    expected_bytes: usize,
    actual_bytes: usize,
    reason: impl Into<String>,
) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: tensor_name.to_owned(),
        offset: 0,
        byte_len: expected_bytes,
        buffer_len: actual_bytes,
        reason: reason.into(),
    }
}

#[inline]
pub(crate) fn mismatch_err(arch: &str, reason: impl Into<String>) -> K3Error {
    K3Error::CapabilityMismatch {
        architecture: arch.to_owned(),
        reason: reason.into(),
    }
}

pub(crate) fn checked_mul(name: &str, a: usize, b: usize) -> Result<usize, K3Error> {
    a.checked_mul(b).ok_or_else(|| {
        bounds_err(
            name,
            usize::MAX,
            0,
            format!("{name} dimensions ({a} * {b}) overflowed usize"),
        )
    })
}

pub(crate) fn validate_finite_slice(name: &str, data: &[f32]) -> Result<(), K3Error> {
    for (idx, &val) in data.iter().enumerate() {
        if !val.is_finite() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: idx.saturating_mul(size_of::<f32>()),
                byte_len: size_of::<f32>(),
                buffer_len: size_of_val(data),
                reason: format!("non-finite f32 value ({val}) in '{name}' at index {idx}"),
            });
        }
    }
    Ok(())
}

/// Returns `true` when `quant` is supported by this crate build.
#[must_use]
pub(crate) const fn is_supported_vision_quant(quant: QuantType) -> bool {
    #[cfg(feature = "with-kernels")]
    {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q4_0
                | QuantType::Q4_1
                | QuantType::Q5_0
                | QuantType::Q5_1
                | QuantType::Q8_0
                | QuantType::Q2_K
                | QuantType::Q3_K
                | QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q6_K
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16)
    }
}

#[cfg(not(feature = "with-kernels"))]
#[must_use]
fn f16_to_f32_ieee(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 0x0001);
    let exp = u32::from((bits >> 10) & 0x001F);
    let mant = u32::from(bits & 0x03FF);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e = 127_u32 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        f32::from_bits((sign << 31) | (e << 23) | (m << 13))
    } else if exp == 0x1F {
        let f32_bits = if mant == 0 {
            (sign << 31) | 0x7F80_0000
        } else {
            (sign << 31) | 0x7FC0_0000 | (mant << 13)
        };
        f32::from_bits(f32_bits)
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

#[cfg(not(feature = "with-kernels"))]
#[must_use]
const fn bf16_to_f32_ieee(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "with-kernels")]
    {
        BaselineKernel::new().dot_product_f32(a, b)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

/// Evaluates `QuickGELU` (`x * sigmoid(1.702 * x)`) in-place across finite activations.
pub(crate) fn quick_gelu_inplace(values: &mut [f32]) -> Result<(), K3Error> {
    if values.is_empty() {
        return Err(bounds_err(
            "quick_gelu",
            size_of::<f32>(),
            0,
            "quick_gelu input slice cannot be empty",
        ));
    }
    validate_finite_slice("quick_gelu", values)?;
    for val in values.iter_mut() {
        let x = *val;
        let z = 1.702_f32 * x;
        *val = if z >= 88.0_f32 {
            x
        } else if z <= -88.0_f32 {
            0.0_f32
        } else {
            x / (1.0_f32 + (-z).exp())
        };
    }
    Ok(())
}

/// Looks up the first matching tensor among `aliases`, failing closed if absent.
pub(crate) fn lookup_first_tensor<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[&str],
    logical_name: &str,
) -> Result<TensorView<'w>, K3Error> {
    for &name in aliases {
        if weights.contains_tensor(name) {
            return weights.tensor(name);
        }
    }
    Err(bounds_err(
        logical_name,
        0,
        0,
        format!("missing required tensor '{logical_name}' (tried {aliases:?})"),
    ))
}

/// Looks up the first matching optional tensor among `aliases`.
pub(crate) fn lookup_optional_tensor<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[&str],
) -> Result<Option<TensorView<'w>>, K3Error> {
    for &name in aliases {
        if weights.contains_tensor(name) {
            return weights.tensor(name).map(Some);
        }
    }
    Ok(None)
}

/// Verifies that every element in an `F32`, `F16`, or `BF16` view is finite (`NaN`/`±Inf` fail closed).
pub(crate) fn validate_float_view_finite(view: &TensorView<'_>) -> Result<(), K3Error> {
    let raw = view.as_bytes();
    match view.quant() {
        QuantType::F32 => {
            let expected_bytes = view.num_elements().saturating_mul(size_of::<f32>());
            if raw.len() < expected_bytes {
                return Err(bounds_err(
                    view.name(),
                    expected_bytes,
                    raw.len(),
                    format!("F32 tensor '{}' payload truncated", view.name()),
                ));
            }
            for (idx, chunk) in raw[..expected_bytes]
                .chunks_exact(size_of::<f32>())
                .enumerate()
            {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if !val.is_finite() {
                    return Err(bounds_err(
                        view.name(),
                        size_of::<f32>(),
                        raw.len(),
                        format!(
                            "non-finite F32 weight ({val}) in '{}' at element {idx}",
                            view.name()
                        ),
                    ));
                }
            }
        }
        QuantType::F16 => {
            let expected_bytes = view.num_elements().saturating_mul(size_of::<u16>());
            if raw.len() < expected_bytes {
                return Err(bounds_err(
                    view.name(),
                    expected_bytes,
                    raw.len(),
                    format!("F16 tensor '{}' payload truncated", view.name()),
                ));
            }
            for (idx, chunk) in raw[..expected_bytes]
                .chunks_exact(size_of::<u16>())
                .enumerate()
            {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let val = f16_to_f32_ieee(bits);
                if !val.is_finite() {
                    return Err(bounds_err(
                        view.name(),
                        size_of::<u16>(),
                        raw.len(),
                        format!(
                            "non-finite F16 weight ({val}) in '{}' at element {idx}",
                            view.name()
                        ),
                    ));
                }
            }
        }
        QuantType::BF16 => {
            let expected_bytes = view.num_elements().saturating_mul(size_of::<u16>());
            if raw.len() < expected_bytes {
                return Err(bounds_err(
                    view.name(),
                    expected_bytes,
                    raw.len(),
                    format!("BF16 tensor '{}' payload truncated", view.name()),
                ));
            }
            for (idx, chunk) in raw[..expected_bytes]
                .chunks_exact(size_of::<u16>())
                .enumerate()
            {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let val = bf16_to_f32_ieee(bits);
                if !val.is_finite() {
                    return Err(bounds_err(
                        view.name(),
                        size_of::<u16>(),
                        raw.len(),
                        format!(
                            "non-finite BF16 weight ({val}) in '{}' at element {idx}",
                            view.name()
                        ),
                    ));
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Decodes all elements of `view` into finite `Vec<f32>`, failing closed on truncation,
/// unsupported quantization, or `NaN`/`±Inf`.
pub(crate) fn decode_all_finite(view: &TensorView<'_>, arch: &str) -> Result<Vec<f32>, K3Error> {
    if !is_supported_vision_quant(view.quant()) {
        return Err(K3Error::unsupported_tensor_quant(
            arch,
            view.name(),
            view.quant(),
        ));
    }
    let total = view.num_elements();
    if total == 0 {
        return Err(bounds_err(
            view.name(),
            size_of::<f32>(),
            0,
            format!("tensor '{}' cannot be empty", view.name()),
        ));
    }

    let raw = view.as_bytes();
    let mut out = vec![0.0_f32; total];
    match view.quant() {
        QuantType::F32 => {
            let req = total.saturating_mul(size_of::<f32>());
            if raw.len() < req {
                return Err(bounds_err(
                    view.name(),
                    req,
                    raw.len(),
                    format!(
                        "F32 tensor '{}' byte length {} < {req}",
                        view.name(),
                        raw.len()
                    ),
                ));
            }
            for (dst, chunk) in out
                .iter_mut()
                .zip(raw[..req].chunks_exact(size_of::<f32>()))
            {
                *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        QuantType::F16 => {
            decode_u16_float_view(view, raw, &mut out, f16_to_f32_ieee)?;
        }
        QuantType::BF16 => {
            decode_u16_float_view(view, raw, &mut out, bf16_to_f32_ieee)?;
        }
        #[cfg(feature = "with-kernels")]
        quant => {
            let blk = quant.block_elements();
            let shape = view.shape();
            let row_len = if shape.len() == 2 && shape[0] % blk == 0 {
                shape[0]
            } else if shape.len() == 2 && shape[1] % blk == 0 {
                shape[1]
            } else if total % blk == 0 {
                total
            } else {
                return Err(bounds_err(
                    view.name(),
                    0,
                    raw.len(),
                    format!(
                        "tensor '{}' elements ({total}) not divisible by {} block size {blk}",
                        view.name(),
                        quant.as_str()
                    ),
                ));
            };
            let rows = total / row_len;
            for r in 0..rows {
                dequantize_token_row(view, r, &mut out[r * row_len..(r + 1) * row_len])?;
            }
        }
        #[cfg(not(feature = "with-kernels"))]
        other => {
            return Err(K3Error::unsupported_tensor_quant(arch, view.name(), other));
        }
    }

    validate_finite_slice(view.name(), &out)?;
    Ok(out)
}

fn decode_u16_float_view(
    view: &TensorView<'_>,
    raw: &[u8],
    out: &mut [f32],
    decode_fn: fn(u16) -> f32,
) -> Result<(), K3Error> {
    let req = out.len().saturating_mul(size_of::<u16>());
    if raw.len() < req {
        return Err(bounds_err(
            view.name(),
            req,
            raw.len(),
            format!(
                "{} tensor '{}' byte length {} < {req}",
                view.quant().as_str(),
                view.name(),
                raw.len()
            ),
        ));
    }
    for (dst, chunk) in out
        .iter_mut()
        .zip(raw[..req].chunks_exact(size_of::<u16>()))
    {
        *dst = decode_fn(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(())
}

/// Decodes a required 1D tensor matching `expected_len`.
pub(crate) fn decode_required_1d(
    weights: &dyn ModelWeights,
    aliases: &[&str],
    expected_len: usize,
    logical_name: &str,
    arch: &str,
) -> Result<Vec<f32>, K3Error> {
    let view = lookup_first_tensor(weights, aliases, logical_name)?;
    decode_1d_view(&view, expected_len, arch)
}

/// Decodes an optional 1D tensor matching `expected_len`.
pub(crate) fn decode_optional_1d(
    weights: &dyn ModelWeights,
    aliases: &[&str],
    expected_len: usize,
    arch: &str,
) -> Result<Option<Vec<f32>>, K3Error> {
    let Some(view) = lookup_optional_tensor(weights, aliases)? else {
        return Ok(None);
    };
    decode_1d_view(&view, expected_len, arch).map(Some)
}

pub(crate) fn validate_view_ready(view: &TensorView<'_>, arch: &str) -> Result<(), K3Error> {
    if !is_supported_vision_quant(view.quant()) {
        return Err(K3Error::unsupported_tensor_quant(
            arch,
            view.name(),
            view.quant(),
        ));
    }
    let req = view
        .quant()
        .storage_bytes(view.name(), view.num_elements())?;
    let actual = view.as_bytes().len();
    if actual < req {
        return Err(bounds_err(
            view.name(),
            req,
            actual,
            format!(
                "tensor '{}' payload truncated: {actual} bytes < required {req}",
                view.name()
            ),
        ));
    }
    Ok(())
}

pub(crate) fn validate_1d_view(
    view: &TensorView<'_>,
    expected_len: usize,
    arch: &str,
) -> Result<(), K3Error> {
    if expected_len == 0 {
        return Err(bounds_err(
            view.name(),
            0,
            view.byte_len(),
            "expected 1D tensor length must be non-zero",
        ));
    }
    if view.num_elements() != expected_len {
        return Err(bounds_err(
            view.name(),
            expected_len.saturating_mul(size_of::<f32>()),
            view.byte_len(),
            format!(
                "tensor '{}' expected {expected_len} elements, got {}",
                view.name(),
                view.num_elements()
            ),
        ));
    }
    let shape = view.shape();
    let valid_shape = shape.len() == 1
        || (shape.len() == 2 && (shape == [1, expected_len] || shape == [expected_len, 1]))
        || (shape.len() == 3 && (shape == [1, 1, expected_len] || shape == [expected_len, 1, 1]));
    if !valid_shape {
        return Err(bounds_err(
            view.name(),
            expected_len.saturating_mul(size_of::<f32>()),
            view.byte_len(),
            format!(
                "tensor '{}' shape {shape:?} incompatible with 1D length {expected_len}",
                view.name()
            ),
        ));
    }
    validate_view_ready(view, arch)
}

pub(crate) fn decode_1d_view(
    view: &TensorView<'_>,
    expected_len: usize,
    arch: &str,
) -> Result<Vec<f32>, K3Error> {
    validate_1d_view(view, expected_len, arch)?;
    decode_all_finite(view, arch)
}

/// Validates that a 2D weight matrix accepts either GGUF `[in_dim, out_dim]` or `SafeTensors`
/// `[out_dim, in_dim]` shape ordering.
pub(crate) fn validate_2d_linear_shape(
    view: &TensorView<'_>,
    expected_out: usize,
    expected_in: usize,
) -> Result<(), K3Error> {
    if expected_out == 0 || expected_in == 0 {
        return Err(bounds_err(
            view.name(),
            0,
            view.byte_len(),
            "linear projection dimensions must be non-zero",
        ));
    }
    let expected_elems = checked_mul(view.name(), expected_out, expected_in)?;
    let shape = view.shape();
    if shape.len() == 2 {
        let matches_gguf = shape[0] == expected_in && shape[1] == expected_out;
        let matches_st = shape[0] == expected_out && shape[1] == expected_in;
        if !matches_gguf && !matches_st {
            return Err(bounds_err(
                view.name(),
                expected_elems.saturating_mul(size_of::<f32>()),
                view.byte_len(),
                format!(
                    "tensor '{}' shape {shape:?} does not match expected [{expected_out}, {expected_in}] or [{expected_in}, {expected_out}]",
                    view.name()
                ),
            ));
        }
    }
    if view.num_elements() != expected_elems {
        return Err(bounds_err(
            view.name(),
            expected_elems.saturating_mul(size_of::<f32>()),
            view.byte_len(),
            format!(
                "tensor '{}' element count {} != expected {expected_out} * {expected_in} ({expected_elems})",
                view.name(),
                view.num_elements()
            ),
        ));
    }
    Ok(())
}

/// Infers `(out_dim)` for a 2D matrix whose known input dimension is `known_in`.
pub(crate) fn infer_2d_other_dim(
    view: &TensorView<'_>,
    known_dim: usize,
    role: &str,
) -> Result<usize, K3Error> {
    if known_dim == 0 {
        return Err(bounds_err(
            view.name(),
            0,
            view.byte_len(),
            format!("{role} known dimension cannot be zero"),
        ));
    }
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(bounds_err(
            view.name(),
            0,
            view.byte_len(),
            format!(
                "tensor '{}' ({role}) must be 2D, got shape {shape:?}",
                view.name()
            ),
        ));
    }
    if shape[0] == known_dim && shape[1] > 0 {
        Ok(shape[1])
    } else if shape[1] == known_dim && shape[0] > 0 {
        Ok(shape[0])
    } else {
        Err(bounds_err(
            view.name(),
            0,
            view.byte_len(),
            format!(
                "tensor '{}' ({role}) shape {shape:?} does not contain expected dimension {known_dim}",
                view.name()
            ),
        ))
    }
}

/// Borrowed 2D matrix slice supporting both full `TensorView` and contiguous fused sub-slices
/// without heap allocation or leaking shape slices.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MatrixSlice<'a> {
    name: &'a str,
    quant: QuantType,
    rows: usize,
    cols: usize,
    raw: &'a [u8],
}

impl<'a> MatrixSlice<'a> {
    pub(crate) fn from_view(
        view: &TensorView<'a>,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<Self, K3Error> {
        validate_2d_linear_shape(view, expected_rows, expected_cols)?;
        validate_float_view_finite(view)?;
        Ok(Self {
            name: view.name(),
            quant: view.quant(),
            rows: expected_rows,
            cols: expected_cols,
            raw: view.as_bytes(),
        })
    }

    pub(crate) fn split_fused_3way(
        view: &TensorView<'a>,
        attn_dim: usize,
        in_dim: usize,
    ) -> Result<[Self; 3], K3Error> {
        let total_out = checked_mul(view.name(), 3, attn_dim)?;
        validate_2d_linear_shape(view, total_out, in_dim)?;
        validate_float_view_finite(view)?;
        let quant = view.quant();
        let blk = quant.block_elements();
        if in_dim % blk != 0 {
            return Err(bounds_err(
                view.name(),
                0,
                view.byte_len(),
                format!(
                    "fused QKV input dim {in_dim} is not divisible by {} block size {blk}",
                    quant.as_str()
                ),
            ));
        }
        let row_bytes = (in_dim / blk) * quant.block_bytes();
        let part_bytes = checked_mul(view.name(), attn_dim, row_bytes)?;
        let total_bytes = checked_mul(view.name(), 3, part_bytes)?;
        let raw = view.as_bytes();
        if raw.len() < total_bytes {
            return Err(bounds_err(
                view.name(),
                total_bytes,
                raw.len(),
                format!(
                    "fused QKV payload length {} < required {total_bytes}",
                    raw.len()
                ),
            ));
        }
        Ok([
            Self {
                name: view.name(),
                quant,
                rows: attn_dim,
                cols: in_dim,
                raw: &raw[..part_bytes],
            },
            Self {
                name: view.name(),
                quant,
                rows: attn_dim,
                cols: in_dim,
                raw: &raw[part_bytes..2 * part_bytes],
            },
            Self {
                name: view.name(),
                quant,
                rows: attn_dim,
                cols: in_dim,
                raw: &raw[2 * part_bytes..3 * part_bytes],
            },
        ])
    }
}

fn matvec_f32_bytes(
    out: &mut [f32],
    name: &str,
    rows: usize,
    cols: usize,
    raw: &[u8],
    input: &[f32],
) -> Result<(), K3Error> {
    let expected_elems = checked_mul(name, rows, cols)?;
    let req_bytes = checked_mul(name, expected_elems, size_of::<f32>())?;
    if raw.len() < req_bytes {
        return Err(bounds_err(
            name,
            req_bytes,
            raw.len(),
            format!(
                "F32 matrix '{name}' buffer length {} < required {req_bytes}",
                raw.len()
            ),
        ));
    }
    let mut chunk_buf = [0.0_f32; 256];
    let stride_bytes = cols * size_of::<f32>();
    for (r, out_slot) in out.iter_mut().enumerate() {
        let row_bytes = &raw[r * stride_bytes..(r + 1) * stride_bytes];
        let mut acc = 0.0_f32;
        let mut col = 0_usize;
        while col < cols {
            let n = (cols - col).min(chunk_buf.len());
            for (i, dst) in chunk_buf[..n].iter_mut().enumerate() {
                let off = (col + i) * size_of::<f32>();
                *dst = f32::from_le_bytes([
                    row_bytes[off],
                    row_bytes[off + 1],
                    row_bytes[off + 2],
                    row_bytes[off + 3],
                ]);
            }
            acc += dot_f32(&chunk_buf[..n], &input[col..col + n]);
            col += n;
        }
        *out_slot = acc;
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn matvec_half_portable(
    out: &mut [f32],
    name: &str,
    quant: QuantType,
    rows: usize,
    cols: usize,
    raw: &[u8],
    input: &[f32],
) -> Result<(), K3Error> {
    let expected_elems = checked_mul(name, rows, cols)?;
    let req_bytes = checked_mul(name, expected_elems, size_of::<u16>())?;
    if raw.len() < req_bytes {
        return Err(bounds_err(
            name,
            req_bytes,
            raw.len(),
            format!(
                "half matrix '{name}' buffer length {} < required {req_bytes}",
                raw.len()
            ),
        ));
    }
    let stride_bytes = cols * size_of::<u16>();
    for (r, out_slot) in out.iter_mut().enumerate() {
        let row_bytes = &raw[r * stride_bytes..(r + 1) * stride_bytes];
        let mut acc = 0.0_f32;
        for (c, &x) in input.iter().enumerate() {
            let off = c * size_of::<u16>();
            let bits = u16::from_le_bytes([row_bytes[off], row_bytes[off + 1]]);
            let w = match quant {
                QuantType::F16 => f16_to_f32_ieee(bits),
                QuantType::BF16 => bf16_to_f32_ieee(bits),
                _ => 0.0_f32,
            };
            acc += w * x;
        }
        *out_slot = acc;
    }
    Ok(())
}

pub(crate) fn matvec_slice(
    out: &mut [f32],
    matrix: &MatrixSlice<'_>,
    input: &[f32],
    arch: &str,
) -> Result<(), K3Error> {
    if out.len() != matrix.rows || input.len() != matrix.cols {
        return Err(bounds_err(
            matrix.name,
            matrix.rows,
            out.len(),
            format!(
                "matvec dimension mismatch for '{}': out={} (expected {}), in={} (expected {})",
                matrix.name,
                out.len(),
                matrix.rows,
                input.len(),
                matrix.cols
            ),
        ));
    }
    match matrix.quant {
        QuantType::F32 => matvec_f32_bytes(
            out,
            matrix.name,
            matrix.rows,
            matrix.cols,
            matrix.raw,
            input,
        )?,
        #[cfg(not(feature = "with-kernels"))]
        QuantType::F16 | QuantType::BF16 => matvec_half_portable(
            out,
            matrix.name,
            matrix.quant,
            matrix.rows,
            matrix.cols,
            matrix.raw,
            input,
        )?,
        #[cfg(feature = "with-kernels")]
        quant => {
            let shape = [matrix.cols, matrix.rows];
            let tv = TensorView::from_bytes(matrix.name, quant, &shape, 0, matrix.raw)?;
            match quant {
                QuantType::F16 => vec_dot_f16(out, &tv, input)?,
                QuantType::BF16 => vec_dot_bf16(out, &tv, input)?,
                QuantType::Q4_0 => vec_dot_q4_0(out, &tv, input)?,
                QuantType::Q4_1 => vec_dot_q4_1(out, &tv, input)?,
                QuantType::Q5_0 => vec_dot_q5_0(out, &tv, input)?,
                QuantType::Q5_1 => vec_dot_q5_1(out, &tv, input)?,
                QuantType::Q8_0 => vec_dot_q8_0(out, &tv, input)?,
                QuantType::Q2_K => vec_dot_q2_k(out, &tv, input)?,
                QuantType::Q3_K => vec_dot_q3_k(out, &tv, input)?,
                QuantType::Q4_K_M => BaselineKernel::new().vec_dot_q4_k_m(out, &tv, input)?,
                QuantType::Q5_K => vec_dot_q5_k(out, &tv, input)?,
                QuantType::Q6_K => vec_dot_q6_k(out, &tv, input)?,
                other => {
                    return Err(K3Error::unsupported_tensor_quant(arch, matrix.name, other));
                }
            }
        }
        #[cfg(not(feature = "with-kernels"))]
        other => {
            return Err(K3Error::unsupported_tensor_quant(arch, matrix.name, other));
        }
    }
    validate_finite_slice(matrix.name, out)
}

/// Applies `out[t] = W * in[t] (+ bias)` over `[seq_len, in_dim] -> [seq_len, out_dim]`.
#[expect(
    clippy::too_many_arguments,
    reason = "sequential linear layer forward passes pre-validated matrix, bias, and buffers"
)]
pub(crate) fn linear_forward_seq(
    out: &mut [f32],
    matrix: &MatrixSlice<'_>,
    bias: Option<&[f32]>,
    input: &[f32],
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
    arch: &str,
) -> Result<(), K3Error> {
    if let Some(b) = bias
        && b.len() != out_dim
    {
        return Err(bounds_err(
            matrix.name,
            out_dim,
            b.len(),
            format!("bias length {} != expected out_dim {out_dim}", b.len()),
        ));
    }
    let expected_in = checked_mul(matrix.name, seq_len, in_dim)?;
    let expected_out = checked_mul(matrix.name, seq_len, out_dim)?;
    if input.len() != expected_in || out.len() != expected_out {
        return Err(bounds_err(
            matrix.name,
            expected_out,
            out.len(),
            format!(
                "linear sequence buffer mismatch: seq_len={seq_len}, in={} (expected {expected_in}), out={} (expected {expected_out})",
                input.len(),
                out.len()
            ),
        ));
    }
    for t in 0..seq_len {
        let in_row = &input[t * in_dim..(t + 1) * in_dim];
        let out_row = &mut out[t * out_dim..(t + 1) * out_dim];
        matvec_slice(out_row, matrix, in_row, arch)?;
        if let Some(b) = bias {
            for (dst, &bv) in out_row.iter_mut().zip(b.iter()) {
                *dst += bv;
            }
        }
    }
    validate_finite_slice(matrix.name, out)
}

/// Applies `layernorm_f32` per token across `[seq_len, hidden_dim]` into `out` using `norm_buf`.
pub(crate) fn apply_layernorm_seq(
    out: &mut [f32],
    input: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<(), K3Error> {
    let total = checked_mul("layernorm_seq", seq_len, hidden_dim)?;
    if out.len() != total || input.len() != total {
        return Err(bounds_err(
            "layernorm_seq",
            total,
            out.len(),
            "layernorm sequence slice length mismatch",
        ));
    }
    for t in 0..seq_len {
        let range = t * hidden_dim..(t + 1) * hidden_dim;
        layernorm_f32(
            &mut out[range.clone()],
            &input[range],
            Some(weight),
            bias,
            eps,
        )?;
    }
    Ok(())
}

/// Computes standard `RMSNorm` (`x / sqrt(mean(x^2) + eps) * weight`) over a single vector.
pub(crate) fn rmsnorm_f32(
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    let len = input.len();
    if len == 0 || out.len() != len || weight.len() != len || !eps.is_finite() || eps <= 0.0_f32 {
        return Err(bounds_err(
            "rmsnorm",
            len,
            out.len(),
            "invalid rmsnorm arguments or slice length mismatch",
        ));
    }
    validate_finite_slice("rmsnorm_input", input)?;
    validate_finite_slice("rmsnorm_weight", weight)?;
    #[cfg(feature = "with-kernels")]
    {
        BaselineKernel::new().rmsnorm(out, input, weight, eps)?;
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        #[expect(clippy::cast_precision_loss, reason = "vector length fits f32")]
        let mean_sq = dot_f32(input, input) / (len as f32);
        let inv_rms = 1.0_f32 / (mean_sq + eps).sqrt();
        for ((dst, &x), &w) in out.iter_mut().zip(input.iter()).zip(weight.iter()) {
            *dst = x * inv_rms * w;
        }
    }
    validate_finite_slice("rmsnorm_output", out)
}

/// Applies `rmsnorm_f32` per token across `[seq_len, hidden_dim]`.
pub(crate) fn apply_rmsnorm_seq(
    out: &mut [f32],
    input: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    let total = checked_mul("rmsnorm_seq", seq_len, hidden_dim)?;
    if out.len() != total || input.len() != total {
        return Err(bounds_err(
            "rmsnorm_seq",
            total,
            out.len(),
            "rmsnorm sequence slice length mismatch",
        ));
    }
    for t in 0..seq_len {
        let range = t * hidden_dim..(t + 1) * hidden_dim;
        rmsnorm_f32(&mut out[range.clone()], &input[range], weight, eps)?;
    }
    Ok(())
}

/// Computes non-overlapping 2D spatial average pooling over `[grid_h, grid_w, hidden_dim]`
/// into `[grid_h / kernel, grid_w / kernel, hidden_dim]` (Gemma-3 projector convention).
pub(crate) fn spatial_avg_pool_2d_f32(
    out_pooled: &mut [f32],
    patches: &[f32],
    grid_h: usize,
    grid_w: usize,
    hidden_dim: usize,
    kernel: usize,
) -> Result<(), K3Error> {
    if kernel == 0
        || grid_h < kernel
        || grid_w < kernel
        || hidden_dim == 0
        || grid_h % kernel != 0
        || grid_w % kernel != 0
    {
        return Err(bounds_err(
            "spatial_avg_pool_2d",
            0,
            patches.len(),
            format!(
                "spatial_avg_pool_2d requires grid {grid_h}x{grid_w} divisible by non-zero kernel {kernel}"
            ),
        ));
    }
    let in_tokens = checked_mul("spatial_avg_pool_in", grid_h, grid_w)?;
    let in_elems = checked_mul("spatial_avg_pool_in", in_tokens, hidden_dim)?;
    let out_h = grid_h / kernel;
    let out_w = grid_w / kernel;
    let out_tokens = checked_mul("spatial_avg_pool_out", out_h, out_w)?;
    let out_elems = checked_mul("spatial_avg_pool_out", out_tokens, hidden_dim)?;
    if patches.len() != in_elems || out_pooled.len() != out_elems {
        return Err(bounds_err(
            "spatial_avg_pool_2d",
            out_elems,
            out_pooled.len(),
            "spatial_avg_pool_2d buffer length mismatch",
        ));
    }
    validate_finite_slice("spatial_avg_pool_patches", patches)?;
    if kernel == 1 {
        out_pooled.copy_from_slice(patches);
        return Ok(());
    }
    let area = checked_mul("spatial_avg_pool_area", kernel, kernel)?;
    #[expect(clippy::cast_precision_loss, reason = "pooling kernel area fits f32")]
    let inv_area = 1.0_f32 / (area as f32);
    out_pooled.fill(0.0_f32);
    for oy in 0..out_h {
        for ox in 0..out_w {
            let dst = &mut out_pooled[(oy * out_w + ox) * hidden_dim..][..hidden_dim];
            for ky in 0..kernel {
                let sy = oy * kernel + ky;
                for kx in 0..kernel {
                    let sx = ox * kernel + kx;
                    let src = &patches[(sy * grid_w + sx) * hidden_dim..][..hidden_dim];
                    for (d_slot, &s_val) in dst.iter_mut().zip(src.iter()) {
                        *d_slot += s_val;
                    }
                }
            }
            for d_slot in dst.iter_mut() {
                *d_slot *= inv_area;
            }
        }
    }
    validate_finite_slice("spatial_avg_pool_out", out_pooled)
}

/// Applies `out[t] = input[t] @ W (+ bias)` where `matrix` stores `[in_dim, out_dim]` rows
/// (e.g. Gemma-3 `mm.input_projection.weight`).
#[expect(
    clippy::too_many_arguments,
    reason = "transposed linear forward accepts pre-validated matrix, optional bias, and sequence buffers"
)]
pub(crate) fn linear_forward_seq_transposed(
    out: &mut [f32],
    matrix: &MatrixSlice<'_>,
    bias: Option<&[f32]>,
    input: &[f32],
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
    arch: &str,
) -> Result<(), K3Error> {
    if matrix.rows != in_dim || matrix.cols != out_dim {
        return Err(bounds_err(
            matrix.name,
            in_dim.saturating_mul(out_dim),
            matrix.rows.saturating_mul(matrix.cols),
            format!(
                "transposed linear matrix '{}' expected [{in_dim}, {out_dim}], got [{}, {}]",
                matrix.name, matrix.rows, matrix.cols
            ),
        ));
    }
    if let Some(b) = bias
        && b.len() != out_dim
    {
        return Err(bounds_err(
            matrix.name,
            out_dim,
            b.len(),
            format!("bias length {} != expected out_dim {out_dim}", b.len()),
        ));
    }
    let expected_in = checked_mul(matrix.name, seq_len, in_dim)?;
    let expected_out = checked_mul(matrix.name, seq_len, out_dim)?;
    if input.len() != expected_in || out.len() != expected_out {
        return Err(bounds_err(
            matrix.name,
            expected_out,
            out.len(),
            format!(
                "transposed linear sequence buffer mismatch: seq_len={seq_len}, in={} (expected {expected_in}), out={} (expected {expected_out})",
                input.len(),
                out.len()
            ),
        ));
    }
    for t in 0..seq_len {
        let in_row = &input[t * in_dim..(t + 1) * in_dim];
        let out_row = &mut out[t * out_dim..(t + 1) * out_dim];
        matvec_transposed_slice(out_row, matrix, in_row, arch)?;
        if let Some(b) = bias {
            for (dst, &bv) in out_row.iter_mut().zip(b.iter()) {
                *dst += bv;
            }
        }
    }
    validate_finite_slice(matrix.name, out)
}

fn matvec_transposed_slice(
    out: &mut [f32],
    matrix: &MatrixSlice<'_>,
    input: &[f32],
    arch: &str,
) -> Result<(), K3Error> {
    let in_dim = matrix.rows;
    let out_dim = matrix.cols;
    if input.len() != in_dim || out.len() != out_dim {
        return Err(bounds_err(
            matrix.name,
            out_dim,
            out.len(),
            format!(
                "transposed matvec dimension mismatch for '{}': out={} (expected {out_dim}), in={} (expected {in_dim})",
                matrix.name,
                out.len(),
                input.len()
            ),
        ));
    }
    if !is_supported_vision_quant(matrix.quant) {
        return Err(K3Error::unsupported_tensor_quant(
            arch,
            matrix.name,
            matrix.quant,
        ));
    }
    let req_bytes = matrix
        .quant
        .storage_bytes(matrix.name, checked_mul(matrix.name, in_dim, out_dim)?)?;
    if matrix.raw.len() < req_bytes {
        return Err(bounds_err(
            matrix.name,
            req_bytes,
            matrix.raw.len(),
            format!(
                "transposed matrix '{}' buffer length {} < required {req_bytes}",
                matrix.name,
                matrix.raw.len()
            ),
        ));
    }

    out.fill(0.0_f32);
    match matrix.quant {
        QuantType::F32 => {
            let stride = out_dim * size_of::<f32>();
            for (r, &x) in input.iter().enumerate() {
                let row = &matrix.raw[r * stride..(r + 1) * stride];
                for (dst, chunk) in out.iter_mut().zip(row.chunks_exact(size_of::<f32>())) {
                    let w = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    *dst += x * w;
                }
            }
        }
        QuantType::F16 | QuantType::BF16 => {
            let stride = out_dim * size_of::<u16>();
            let is_bf16 = matches!(matrix.quant, QuantType::BF16);
            for (r, &x) in input.iter().enumerate() {
                let row = &matrix.raw[r * stride..(r + 1) * stride];
                for (dst, chunk) in out.iter_mut().zip(row.chunks_exact(size_of::<u16>())) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let w = if is_bf16 {
                        bf16_to_f32_ieee(bits)
                    } else {
                        f16_to_f32_ieee(bits)
                    };
                    *dst += x * w;
                }
            }
        }
        #[cfg(feature = "with-kernels")]
        quant => {
            let blk = quant.block_elements();
            if out_dim % blk != 0 {
                return Err(bounds_err(
                    matrix.name,
                    0,
                    matrix.raw.len(),
                    format!(
                        "transposed quantized matrix '{}' out_dim {out_dim} not divisible by {} block size {blk}",
                        matrix.name,
                        quant.as_str()
                    ),
                ));
            }
            let shape = [out_dim, in_dim];
            let tv = TensorView::from_bytes(matrix.name, quant, &shape, 0, matrix.raw)?;
            let mut row_buf = vec![0.0_f32; out_dim];
            for (r, &x) in input.iter().enumerate() {
                dequantize_token_row(&tv, r, &mut row_buf)?;
                for (dst, &w) in out.iter_mut().zip(row_buf.iter()) {
                    *dst += x * w;
                }
            }
        }
        #[cfg(not(feature = "with-kernels"))]
        other => {
            return Err(K3Error::unsupported_tensor_quant(arch, matrix.name, other));
        }
    }
    validate_finite_slice(matrix.name, out)
}

// ============================================================================
// Unified Vision / Attention / Norm / Embedding Kernel Wrappers
// ============================================================================

#[expect(
    clippy::too_many_arguments,
    reason = "vision resize+normalize takes explicit source/destination geometry and per-channel stats"
)]
pub(crate) fn resize_rgb_bilinear_norm_f32(
    out_chw: &mut [f32],
    src_hwc: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        rvv_resize_rgb(out_chw, src_hwc, src_h, src_w, dst_h, dst_w, mean, std)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        resize_rgb_bilinear_norm_portable(out_chw, src_hwc, src_h, src_w, dst_h, dst_w, mean, std)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Conv2D patch embedding accepts explicit CHW image, OIHW kernel, optional bias, and output slices"
)]
pub(crate) fn conv2d_patch_embed_f32(
    out_patches: &mut [f32],
    img_chw: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    kernel_oc_ic_kh_kw: &[f32],
    bias: Option<&[f32]>,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        rvv_conv2d_patch_embed(
            out_patches,
            img_chw,
            channels,
            height,
            width,
            patch_size,
            kernel_oc_ic_kh_kw,
            bias,
            hidden_dim,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        conv2d_patch_embed_portable(
            out_patches,
            img_chw,
            channels,
            height,
            width,
            patch_size,
            kernel_oc_ic_kh_kw,
            bias,
            hidden_dim,
        )
    }
}

pub(crate) fn interpolate_2d_pos_embed_f32(
    out_pos: &mut [f32],
    base_pos: &[f32],
    base_grid_h: usize,
    base_grid_w: usize,
    dst_grid_h: usize,
    dst_grid_w: usize,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        rvv_interp_2d_pos(
            out_pos,
            base_pos,
            base_grid_h,
            base_grid_w,
            dst_grid_h,
            dst_grid_w,
            hidden_dim,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        interpolate_2d_pos_embed_portable(
            out_pos,
            base_pos,
            base_grid_h,
            base_grid_w,
            dst_grid_h,
            dst_grid_w,
            hidden_dim,
        )
    }
}

pub(crate) fn pixel_shuffle_merge_2x2_f32(
    out_merged: &mut [f32],
    patches: &[f32],
    grid_h: usize,
    grid_w: usize,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        rvv_pixel_shuffle_2x2(out_merged, patches, grid_h, grid_w, hidden_dim)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        pixel_shuffle_merge_2x2_portable(out_merged, patches, grid_h, grid_w, hidden_dim)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "bidirectional attention accepts Q/K/V, geometry, optional window, and scale"
)]
pub(crate) fn bidirectional_attention_f32(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    window_size: Option<usize>,
    scale: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        rvv_bidir_attn(out, q, k, v, seq_len, n_heads, head_dim, window_size, scale)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        bidirectional_attention_portable(
            out,
            q,
            k,
            v,
            seq_len,
            n_heads,
            head_dim,
            window_size,
            scale,
        )
    }
}

pub(crate) fn layernorm_f32(
    out: &mut [f32],
    input: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        rvv_layernorm(out, input, weight, bias, eps)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        layernorm_portable(out, input, weight, bias, eps)
    }
}

pub(crate) fn gelu_inplace_f32(values: &mut [f32]) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        rvv_gelu_inplace(values)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        gelu_inplace_portable(values)
    }
}

pub(crate) fn pool_and_l2_normalize_f32(
    out: &mut [f32],
    hidden_seq: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    mode_code: u8,
    matryoshka_dim: Option<usize>,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        rvv_pool_and_l2_norm(
            out,
            hidden_seq,
            seq_len,
            hidden_dim,
            mode_code,
            matryoshka_dim,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        pool_and_l2_normalize_portable(
            out,
            hidden_seq,
            seq_len,
            hidden_dim,
            mode_code,
            matryoshka_dim,
        )
    }
}

// ============================================================================
// Portable Scalar Fallbacks (only compiled when `not(feature = "with-kernels")`)
// ============================================================================

#[cfg(not(feature = "with-kernels"))]
fn bilinear_coord(dst_idx: usize, src_len: usize, dst_len: usize) -> (usize, usize, f32) {
    if src_len == dst_len {
        return (dst_idx, dst_idx, 0.0_f32);
    }
    if src_len <= 1 {
        return (0, 0, 0.0_f32);
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "image grid fits exact f64 range"
    )]
    let pos = ((dst_idx as f64 + 0.5_f64) * (src_len as f64) / (dst_len as f64) - 0.5_f64)
        .clamp(0.0_f64, (src_len - 1) as f64);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped non-negative"
    )]
    let i0 = pos.floor() as usize;
    let i1 = (i0 + 1).min(src_len - 1);
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "fractional interpolation weight in [0, 1]"
    )]
    let w = (pos - (i0 as f64)) as f32;
    (i0, i1, w)
}

#[cfg(not(feature = "with-kernels"))]
#[expect(
    clippy::too_many_arguments,
    reason = "portable resize+normalize mirrors jero-kernels-rvv signature"
)]
fn resize_rgb_bilinear_norm_portable(
    out_chw: &mut [f32],
    src_hwc: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<(), K3Error> {
    if src_h == 0 || src_w == 0 || dst_h == 0 || dst_w == 0 {
        return Err(bounds_err(
            "resize_rgb",
            0,
            0,
            "dimensions must be non-zero",
        ));
    }
    let src_elems = checked_mul("resize_src", checked_mul("resize_src", src_h, src_w)?, 3)?;
    let dst_pixels = checked_mul("resize_dst", dst_h, dst_w)?;
    let dst_elems = checked_mul("resize_dst", dst_pixels, 3)?;
    if src_hwc.len() != src_elems || out_chw.len() != dst_elems {
        return Err(bounds_err(
            "resize_rgb",
            dst_elems,
            out_chw.len(),
            "slice length mismatch",
        ));
    }
    validate_finite_slice("resize_mean", &mean)?;
    for &s in &std {
        if !s.is_finite() || s <= 0.0_f32 {
            return Err(bounds_err(
                "resize_std",
                4,
                4,
                "std must be positive finite",
            ));
        }
    }
    validate_finite_slice("resize_src_hwc", src_hwc)?;

    let inv_std = [1.0_f32 / std[0], 1.0_f32 / std[1], 1.0_f32 / std[2]];
    for dy in 0..dst_h {
        let (y0, y1, wy) = bilinear_coord(dy, src_h, dst_h);
        for dx in 0..dst_w {
            let (x0, x1, wx) = bilinear_coord(dx, src_w, dst_w);
            let dst_idx = dy * dst_w + dx;
            for c in 0..3 {
                let p00 = src_hwc[(y0 * src_w + x0) * 3 + c];
                let p01 = src_hwc[(y0 * src_w + x1) * 3 + c];
                let p10 = src_hwc[(y1 * src_w + x0) * 3 + c];
                let p11 = src_hwc[(y1 * src_w + x1) * 3 + c];
                let top = p00 + wx * (p01 - p00);
                let bot = p10 + wx * (p11 - p10);
                let val = top + wy * (bot - top);
                out_chw[c * dst_pixels + dst_idx] = (val - mean[c]) * inv_std[c];
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
#[expect(
    clippy::too_many_arguments,
    reason = "portable Conv2D patch embed mirrors jero-kernels-rvv signature"
)]
fn conv2d_patch_embed_portable(
    out_patches: &mut [f32],
    img_chw: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    kernel_oc_ic_kh_kw: &[f32],
    bias: Option<&[f32]>,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    if channels == 0 || height == 0 || width == 0 || patch_size == 0 || hidden_dim == 0 {
        return Err(bounds_err(
            "conv2d_patch_embed",
            0,
            0,
            "dimensions must be non-zero",
        ));
    }
    if height % patch_size != 0 || width % patch_size != 0 {
        return Err(bounds_err(
            "conv2d_patch_embed",
            0,
            0,
            format!("image ({height}x{width}) not divisible by patch_size {patch_size}"),
        ));
    }
    let grid_h = height / patch_size;
    let grid_w = width / patch_size;
    let n_patches = checked_mul("patch_grid", grid_h, grid_w)?;
    let hw = checked_mul("patch_img", height, width)?;
    let img_elems = checked_mul("patch_img", channels, hw)?;
    let patch_area = checked_mul("patch_area", patch_size, patch_size)?;
    let patch_per_oc = checked_mul("patch_per_oc", channels, patch_area)?;
    let kern_elems = checked_mul("patch_kernel", hidden_dim, patch_per_oc)?;
    let out_elems = checked_mul("patch_out", n_patches, hidden_dim)?;
    if img_chw.len() != img_elems
        || kernel_oc_ic_kh_kw.len() != kern_elems
        || out_patches.len() != out_elems
    {
        return Err(bounds_err(
            "conv2d_patch_embed",
            out_elems,
            out_patches.len(),
            "length mismatch",
        ));
    }
    if let Some(b) = bias {
        if b.len() != hidden_dim {
            return Err(bounds_err(
                "patch_bias",
                hidden_dim,
                b.len(),
                "bias length mismatch",
            ));
        }
        validate_finite_slice("patch_bias", b)?;
    }
    validate_finite_slice("patch_img_chw", img_chw)?;
    validate_finite_slice("patch_kernel", kernel_oc_ic_kh_kw)?;

    for py in 0..grid_h {
        for px in 0..grid_w {
            let patch_idx = py * grid_w + px;
            let out_row = &mut out_patches[patch_idx * hidden_dim..(patch_idx + 1) * hidden_dim];
            for (oc, out_slot) in out_row.iter_mut().enumerate() {
                let mut acc = bias.map_or(0.0_f32, |b| b[oc]);
                let out_chan_offset = oc * patch_per_oc;
                for ic in 0..channels {
                    let img_ic = ic * hw;
                    let in_chan_offset = out_chan_offset + ic * patch_area;
                    for ky in 0..patch_size {
                        let iy = py * patch_size + ky;
                        let ix = px * patch_size;
                        let img_row = &img_chw[img_ic + iy * width + ix..][..patch_size];
                        let kern_row =
                            &kernel_oc_ic_kh_kw[in_chan_offset + ky * patch_size..][..patch_size];
                        acc += dot_f32(img_row, kern_row);
                    }
                }
                *out_slot = acc;
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn interpolate_2d_pos_embed_portable(
    out_pos: &mut [f32],
    base_pos: &[f32],
    base_grid_h: usize,
    base_grid_w: usize,
    dst_grid_h: usize,
    dst_grid_w: usize,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    if base_grid_h == 0 || base_grid_w == 0 || dst_grid_h == 0 || dst_grid_w == 0 || hidden_dim == 0
    {
        return Err(bounds_err(
            "interpolate_2d_pos_embed",
            0,
            0,
            "dimensions must be non-zero",
        ));
    }
    let base_elems = checked_mul(
        "pos_base",
        checked_mul("pos_base", base_grid_h, base_grid_w)?,
        hidden_dim,
    )?;
    let dst_elems = checked_mul(
        "pos_dst",
        checked_mul("pos_dst", dst_grid_h, dst_grid_w)?,
        hidden_dim,
    )?;
    if base_pos.len() != base_elems || out_pos.len() != dst_elems {
        return Err(bounds_err(
            "interpolate_2d_pos_embed",
            dst_elems,
            out_pos.len(),
            "length mismatch",
        ));
    }
    validate_finite_slice("pos_base", base_pos)?;
    if base_grid_h == dst_grid_h && base_grid_w == dst_grid_w {
        out_pos.copy_from_slice(base_pos);
        return Ok(());
    }
    for dy in 0..dst_grid_h {
        let (y0, y1, wy) = bilinear_coord(dy, base_grid_h, dst_grid_h);
        for dx in 0..dst_grid_w {
            let (x0, x1, wx) = bilinear_coord(dx, base_grid_w, dst_grid_w);
            let w00 = (1.0_f32 - wy) * (1.0_f32 - wx);
            let w01 = (1.0_f32 - wy) * wx;
            let w10 = wy * (1.0_f32 - wx);
            let w11 = wy * wx;
            let dst_off = (dy * dst_grid_w + dx) * hidden_dim;
            let s00 = &base_pos[(y0 * base_grid_w + x0) * hidden_dim..][..hidden_dim];
            let s01 = &base_pos[(y0 * base_grid_w + x1) * hidden_dim..][..hidden_dim];
            let s10 = &base_pos[(y1 * base_grid_w + x0) * hidden_dim..][..hidden_dim];
            let s11 = &base_pos[(y1 * base_grid_w + x1) * hidden_dim..][..hidden_dim];
            for d in 0..hidden_dim {
                out_pos[dst_off + d] = w00 * s00[d] + w01 * s01[d] + w10 * s10[d] + w11 * s11[d];
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn pixel_shuffle_merge_2x2_portable(
    out_merged: &mut [f32],
    patches: &[f32],
    grid_h: usize,
    grid_w: usize,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    if grid_h == 0 || grid_w == 0 || hidden_dim == 0 || grid_h % 2 != 0 || grid_w % 2 != 0 {
        return Err(bounds_err(
            "pixel_shuffle_merge_2x2",
            0,
            0,
            format!(
                "pixel_shuffle_merge_2x2 requires positive even grid_h={grid_h} and grid_w={grid_w}"
            ),
        ));
    }
    let in_elems = checked_mul(
        "pixel_shuffle_in",
        checked_mul("pixel_shuffle_in", grid_h, grid_w)?,
        hidden_dim,
    )?;
    let out_h = grid_h / 2;
    let out_w = grid_w / 2;
    let merged_dim = checked_mul("pixel_shuffle_out", 4, hidden_dim)?;
    let out_elems = checked_mul(
        "pixel_shuffle_out",
        checked_mul("pixel_shuffle_out", out_h, out_w)?,
        merged_dim,
    )?;
    if patches.len() != in_elems || out_merged.len() != out_elems {
        return Err(bounds_err(
            "pixel_shuffle_merge_2x2",
            out_elems,
            out_merged.len(),
            "length mismatch",
        ));
    }
    validate_finite_slice("pixel_shuffle_patches", patches)?;
    let pair_dim = 2 * hidden_dim;
    for my in 0..out_h {
        for mx in 0..out_w {
            let dst_off = (my * out_w + mx) * merged_dim;
            let top_off = ((2 * my) * grid_w + (2 * mx)) * hidden_dim;
            let bot_off = ((2 * my + 1) * grid_w + (2 * mx)) * hidden_dim;
            out_merged[dst_off..dst_off + pair_dim]
                .copy_from_slice(&patches[top_off..top_off + pair_dim]);
            out_merged[dst_off + pair_dim..dst_off + merged_dim]
                .copy_from_slice(&patches[bot_off..bot_off + pair_dim]);
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
#[expect(
    clippy::too_many_arguments,
    reason = "portable bidirectional attention mirrors jero-kernels-rvv signature"
)]
fn bidirectional_attention_portable(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    window_size: Option<usize>,
    scale: f32,
) -> Result<(), K3Error> {
    if seq_len == 0 || n_heads == 0 || head_dim == 0 {
        return Err(mismatch_err(
            "bidirectional_attention",
            "seq_len, n_heads, and head_dim must be non-zero",
        ));
    }
    let expected = checked_mul(
        "bidir_attn",
        checked_mul("bidir_attn", seq_len, n_heads)?,
        head_dim,
    )?;
    if out.len() != expected || q.len() != expected || k.len() != expected || v.len() != expected {
        return Err(bounds_err(
            "bidirectional_attention",
            expected,
            out.len(),
            "length mismatch",
        ));
    }
    let mut scores = vec![0.0_f32; seq_len];
    for i in 0..seq_len {
        let (ws, we) = match window_size {
            Some(w) => (i.saturating_sub(w), i.saturating_add(w).min(seq_len - 1)),
            None => (0, seq_len - 1),
        };
        let active_len = we - ws + 1;
        for h in 0..n_heads {
            let q_off = (i * n_heads + h) * head_dim;
            let q_h = &q[q_off..q_off + head_dim];
            let mut max_s = f32::NEG_INFINITY;
            for (rel, j) in (ws..=we).enumerate() {
                let k_off = (j * n_heads + h) * head_dim;
                let s = dot_f32(q_h, &k[k_off..k_off + head_dim]) * scale;
                scores[rel] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut sum_w = 0.0_f32;
            for s in &mut scores[..active_len] {
                let w = (*s - max_s).exp();
                *s = w;
                sum_w += w;
            }
            #[expect(clippy::cast_precision_loss, reason = "active_len fits f32")]
            let inv = if sum_w > 0.0 {
                1.0 / sum_w
            } else {
                1.0 / (active_len as f32)
            };
            let out_h = &mut out[q_off..q_off + head_dim];
            out_h.fill(0.0);
            for (rel, j) in (ws..=we).enumerate() {
                let w = scores[rel] * inv;
                let v_off = (j * n_heads + h) * head_dim;
                for (dst, &vv) in out_h.iter_mut().zip(v[v_off..v_off + head_dim].iter()) {
                    *dst += w * vv;
                }
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn layernorm_portable(
    out: &mut [f32],
    input: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<(), K3Error> {
    let len = input.len();
    if len == 0 || out.len() != len || !eps.is_finite() || eps <= 0.0_f32 {
        return Err(bounds_err(
            "layernorm",
            len,
            out.len(),
            "invalid layernorm args",
        ));
    }
    validate_finite_slice("layernorm_input", input)?;
    if let Some(w) = weight {
        if w.len() != len {
            return Err(bounds_err(
                "layernorm_weight",
                len,
                w.len(),
                "weight length mismatch",
            ));
        }
        validate_finite_slice("layernorm_weight", w)?;
    }
    if let Some(b) = bias {
        if b.len() != len {
            return Err(bounds_err(
                "layernorm_bias",
                len,
                b.len(),
                "bias length mismatch",
            ));
        }
        validate_finite_slice("layernorm_bias", b)?;
    }
    #[expect(clippy::cast_precision_loss, reason = "vector length fits f32")]
    let inv_n = 1.0_f32 / (len as f32);
    let mean = input.iter().copied().sum::<f32>() * inv_n;
    let var = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() * inv_n;
    let inv_std = 1.0_f32 / (var + eps).sqrt();
    for (i, &x) in input.iter().enumerate() {
        let mut n = (x - mean) * inv_std;
        if let Some(w) = weight {
            n *= w[i];
        }
        if let Some(b) = bias {
            n += b[i];
        }
        out[i] = n;
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn gelu_inplace_portable(values: &mut [f32]) -> Result<(), K3Error> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const CUBIC_COEF: f32 = 0.044_715;
    if values.is_empty() {
        return Err(bounds_err(
            "gelu_values",
            4,
            0,
            "gelu_values cannot be empty",
        ));
    }
    validate_finite_slice("gelu_values", values)?;
    for slot in values.iter_mut() {
        let x = *slot;
        *slot = if x <= -15.0_f32 {
            0.0_f32
        } else if x >= 15.0_f32 {
            x
        } else {
            let inner = SQRT_2_OVER_PI * (x + CUBIC_COEF * x * x * x);
            0.5_f32 * x * (1.0_f32 + inner.tanh())
        };
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn pool_and_l2_normalize_portable(
    out: &mut [f32],
    hidden_seq: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    mode_code: u8,
    matryoshka_dim: Option<usize>,
) -> Result<(), K3Error> {
    if !matches!(
        mode_code,
        EMBED_POOL_MEAN | EMBED_POOL_CLS | EMBED_POOL_LAST_TOKEN
    ) || seq_len == 0
        || hidden_dim == 0
    {
        return Err(bounds_err(
            "pool_and_l2_normalize",
            0,
            0,
            "invalid pooling arguments",
        ));
    }
    let total = checked_mul("embed_seq", seq_len, hidden_dim)?;
    let eff = match matryoshka_dim {
        None => hidden_dim,
        Some(d) if d == 0 || d > hidden_dim => {
            return Err(bounds_err(
                "embed_matryoshka_dim",
                hidden_dim,
                d,
                "invalid matryoshka_dim",
            ));
        }
        Some(d) => d,
    };
    if hidden_seq.len() != total || out.len() != eff {
        return Err(bounds_err(
            "embed_out",
            eff,
            out.len(),
            "slice length mismatch",
        ));
    }
    validate_finite_slice("embed_hidden_seq", hidden_seq)?;

    match mode_code {
        EMBED_POOL_MEAN => {
            out.copy_from_slice(&hidden_seq[..eff]);
            for s in 1..seq_len {
                let row = &hidden_seq[s * hidden_dim..s * hidden_dim + eff];
                for (d, &v) in out.iter_mut().zip(row.iter()) {
                    *d += v;
                }
            }
            #[expect(clippy::cast_precision_loss, reason = "seq_len fits f32")]
            let inv = 1.0_f32 / (seq_len as f32);
            for d in out.iter_mut() {
                *d *= inv;
            }
        }
        EMBED_POOL_CLS => out.copy_from_slice(&hidden_seq[..eff]),
        EMBED_POOL_LAST_TOKEN => {
            let start = (seq_len - 1) * hidden_dim;
            out.copy_from_slice(&hidden_seq[start..start + eff]);
        }
        _ => unreachable!(),
    }

    let max_abs = out.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
    if max_abs == 0.0_f32 || !max_abs.is_finite() {
        out.fill(0.0_f32);
        return Ok(());
    }
    let max_f64 = f64::from(max_abs);
    let sum_sq: f64 = out
        .iter()
        .map(|&v| {
            let s = f64::from(v) / max_f64;
            s * s
        })
        .sum();
    let norm = max_f64 * sum_sq.sqrt();
    if norm == 0.0_f64 || !norm.is_finite() {
        out.fill(0.0_f32);
        return Ok(());
    }
    let inv_norm = 1.0_f64 / norm;
    for slot in out.iter_mut() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "L2-normalized value lies in [-1.0, 1.0]"
        )]
        {
            *slot = (f64::from(*slot) * inv_norm) as f32;
        }
    }
    Ok(())
}
