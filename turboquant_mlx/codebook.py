"""Lloyd-Max codebook computation for optimal scalar quantization.

Precomputes centroids for quantizing coordinates/angles whose distributions
are known analytically (Gaussian marginals after random rotation, Beta-like
distributions for polar angles at each recursion level).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from scipy import integrate
from scipy.stats import norm


def _gaussian_density(x: np.ndarray) -> np.ndarray:
    """Standard normal density — marginal of rotated unit-norm vectors in high d."""
    return norm.pdf(x)


def _beta_angle_density(psi: np.ndarray, level: int) -> np.ndarray:
    """Density of polar angle at a given recursion level.

    Level 1: uniform on [0, 2*pi)
    Level l>=2: proportional to sin^(2^(l-1) - 1)(2*psi) on [0, pi/2]
    """
    if level == 1:
        return np.ones_like(psi) / (2 * np.pi)
    exponent = 2 ** (level - 1) - 1
    raw = np.sin(2 * psi) ** exponent
    # Normalize numerically
    norm_const, _ = integrate.quad(lambda p: np.sin(2 * p) ** exponent, 0, np.pi / 2)
    return raw / norm_const


def lloyd_max_1d(
    density_fn,
    support: tuple[float, float],
    bits: int,
    max_iter: int = 200,
    n_samples: int = 10_000,
) -> np.ndarray:
    """Compute optimal Lloyd-Max centroids for a 1D distribution.

    Args:
        density_fn: Callable returning density values for an array of points.
        support: (low, high) interval of the distribution.
        bits: Number of quantization bits → 2^bits centroids.
        max_iter: Maximum Lloyd iterations.
        n_samples: Grid resolution for numerical integration.

    Returns:
        Sorted array of 2^bits centroids.
    """
    k = 2**bits
    lo, hi = support

    # Initialize centroids uniformly
    centroids = np.linspace(lo, hi, k + 2)[1:-1]  # skip endpoints

    x = np.linspace(lo, hi, n_samples)
    dx = x[1] - x[0]
    pdf = density_fn(x)
    pdf = pdf / (pdf.sum() * dx)  # renormalize on grid

    for _ in range(max_iter):
        # Compute boundaries (midpoints between centroids)
        boundaries = np.concatenate([[lo], (centroids[:-1] + centroids[1:]) / 2, [hi]])

        new_centroids = np.zeros(k)
        for i in range(k):
            mask = (x >= boundaries[i]) & (x < boundaries[i + 1])
            if mask.sum() == 0:
                new_centroids[i] = centroids[i]
                continue
            weights = pdf[mask] * dx
            total_w = weights.sum()
            if total_w < 1e-15:
                new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = (x[mask] * weights).sum() / total_w

        if np.allclose(centroids, new_centroids, atol=1e-8):
            break
        centroids = new_centroids

    return np.sort(centroids)


# ---------------------------------------------------------------------------
# Precomputed codebook cache
# ---------------------------------------------------------------------------
_CODEBOOK_CACHE: dict[tuple[str, int, int], mx.array] = {}


def get_gaussian_codebook(bits: int) -> mx.array:
    """Get Lloyd-Max centroids for Gaussian marginal (used by TurboQuant MSE stage)."""
    key = ("gaussian", bits, 0)
    if key not in _CODEBOOK_CACHE:
        # Support covers ±4 sigma — beyond that density is negligible
        centroids_np = lloyd_max_1d(_gaussian_density, (-4.0, 4.0), bits)
        _CODEBOOK_CACHE[key] = mx.array(centroids_np, dtype=mx.float32)
    return _CODEBOOK_CACHE[key]


def get_polar_codebook(level: int, bits: int) -> mx.array:
    """Get Lloyd-Max centroids for polar angle at recursion level."""
    key = ("polar", bits, level)
    if key not in _CODEBOOK_CACHE:
        if level == 1:
            support = (0.0, 2 * np.pi)
        else:
            support = (0.0, np.pi / 2)
        centroids_np = lloyd_max_1d(
            lambda psi: _beta_angle_density(psi, level), support, bits
        )
        _CODEBOOK_CACHE[key] = mx.array(centroids_np, dtype=mx.float32)
    return _CODEBOOK_CACHE[key]


def quantize_scalar(values: mx.array, codebook: mx.array) -> mx.array:
    """Quantize values to nearest codebook centroid. Returns indices.

    Args:
        values: (...,) tensor of scalar values.
        codebook: (K,) sorted centroids.

    Returns:
        Integer indices of shape (...,) into the codebook.
    """
    # Expand for broadcasting: values[..., None] vs codebook[None, ...]
    diffs = mx.abs(mx.expand_dims(values, -1) - codebook)
    return mx.argmin(diffs, axis=-1)


def dequantize_scalar(indices: mx.array, codebook: mx.array) -> mx.array:
    """Look up centroid values from indices."""
    return codebook[indices]


# ---------------------------------------------------------------------------
# Bit-packing utilities
# ---------------------------------------------------------------------------


def pack_indices(indices: mx.array, bits: int) -> tuple[mx.array, int]:
    """Pack quantization indices into uint8 bytes.

    For 1-bit: 8 values per byte
    For 2-bit: 4 values per byte
    For 3-bit: rounded up to 4-bit, 2 values per byte
    For 4-bit: 2 values per byte

    Args:
        indices: (..., D) integer indices in [0, 2^bits).
        bits: Number of bits per index.

    Returns:
        (packed, orig_dim): packed uint8 array and original last dimension.
    """
    orig_shape = indices.shape
    orig_dim = orig_shape[-1]
    batch_shape = orig_shape[:-1]

    # Determine packing parameters
    if bits <= 1:
        vals_per_byte = 8
        pack_bits = 1
    elif bits <= 2:
        vals_per_byte = 4
        pack_bits = 2
    elif bits <= 4:
        vals_per_byte = 2
        pack_bits = 4  # round 3-bit up to 4-bit for packing
    else:
        # 5-8 bits: 1 value per byte
        return indices.astype(mx.uint8), orig_dim

    # Pad last dim to multiple of vals_per_byte
    padded_dim = ((orig_dim + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_dim > orig_dim:
        pad_width = padded_dim - orig_dim
        indices = mx.concatenate(
            [indices, mx.zeros((*batch_shape, pad_width), dtype=indices.dtype)],
            axis=-1,
        )

    # Reshape to groups of vals_per_byte
    flat = indices.reshape(*batch_shape, -1, vals_per_byte)

    # Shift each value and OR together into a single byte
    shifts = mx.array([i * pack_bits for i in range(vals_per_byte)], dtype=mx.uint32)
    packed = mx.sum(flat.astype(mx.uint32) << shifts, axis=-1).astype(mx.uint8)

    return packed, orig_dim


def unpack_indices(packed: mx.array, bits: int, orig_dim: int) -> mx.array:
    """Unpack uint8 bytes back to quantization indices.

    Args:
        packed: (..., packed_dim) uint8 array.
        bits: Number of bits per index (as used during packing).
        orig_dim: Original last dimension before packing.

    Returns:
        (..., orig_dim) integer indices.
    """
    if bits <= 1:
        vals_per_byte = 8
        pack_bits = 1
    elif bits <= 2:
        vals_per_byte = 4
        pack_bits = 2
    elif bits <= 4:
        vals_per_byte = 2
        pack_bits = 4
    else:
        return packed.astype(mx.int32)[..., :orig_dim]

    batch_shape = packed.shape[:-1]
    mask = (1 << pack_bits) - 1

    # Expand each byte into vals_per_byte values
    shifts = mx.array([i * pack_bits for i in range(vals_per_byte)], dtype=mx.uint32)
    expanded = (mx.expand_dims(packed.astype(mx.uint32), -1) >> shifts) & mask

    # Reshape and trim
    unpacked = expanded.reshape(*batch_shape, -1)
    return unpacked[..., :orig_dim].astype(mx.int32)


def pack_signs(signs: mx.array) -> tuple[mx.array, int]:
    """Pack sign bits ({-1, +1} as int8) into uint8, 8 signs per byte.

    Args:
        signs: (..., D) int8 array of {-1, +1}.

    Returns:
        (packed, orig_dim): packed uint8 array and original last dimension.
    """
    orig_dim = signs.shape[-1]
    batch_shape = signs.shape[:-1]

    # Convert {-1, +1} to {0, 1}
    bits = ((signs.astype(mx.int32) + 1) // 2).astype(mx.uint32)

    # Pad to multiple of 8
    padded_dim = ((orig_dim + 7) // 8) * 8
    if padded_dim > orig_dim:
        pad_width = padded_dim - orig_dim
        bits = mx.concatenate(
            [bits, mx.zeros((*batch_shape, pad_width), dtype=mx.uint32)],
            axis=-1,
        )

    # Reshape to groups of 8 and pack
    flat = bits.reshape(*batch_shape, -1, 8)
    powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint32)
    packed = mx.sum(flat * powers, axis=-1).astype(mx.uint8)

    return packed, orig_dim


def unpack_signs(packed: mx.array, orig_dim: int) -> mx.array:
    """Unpack uint8 bytes back to sign bits {-1, +1}.

    Args:
        packed: (..., packed_dim) uint8 array.
        orig_dim: Original last dimension before packing.

    Returns:
        (..., orig_dim) int8 array of {-1, +1}.
    """
    batch_shape = packed.shape[:-1]

    # Expand each byte into 8 bits
    powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint32)
    expanded = (mx.expand_dims(packed.astype(mx.uint32), -1) >> mx.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=mx.uint32)) & 1

    # Reshape, trim, and convert {0,1} back to {-1,+1}
    unpacked = expanded.reshape(*batch_shape, -1)[..., :orig_dim]
    return (unpacked.astype(mx.int8) * 2 - 1)
