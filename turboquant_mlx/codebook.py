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
