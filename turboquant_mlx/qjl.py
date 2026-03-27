"""QJL: Quantized Johnson-Lindenstrauss 1-bit residual projection.

Implements the QJL component from arXiv:2504.19874:
- Random Gaussian projection matrix
- Sign quantization (1 bit per projected dimension)
- Unbiased inner-product estimation via sqrt(pi/2)/d scaling

The QJL residual captures information lost during the primary quantization
stage, providing an unbiased dot-product estimator when combined with
the MSE-quantized representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .codebook import pack_signs, unpack_signs


@dataclass
class QJLState:
    """Compressed QJL representation of residual vectors."""

    # Sign bits of projected residual: bit-packed uint8, (batch, packed_dim)
    sign_bits_packed: mx.array
    # Original projection dimension before packing
    sign_orig_dim: int
    # Residual norms: (batch,)
    norms: mx.array


class QJLProjection:
    """1-bit Quantized Johnson-Lindenstrauss projection for residuals.

    Given a residual vector r, QJL compresses it to sign bits:
        signs = sign(S @ r)
    where S is a random Gaussian matrix.

    Decompression gives an unbiased inner-product estimate:
        r_hat = sqrt(pi/2) / d * ||r|| * S^T @ signs
    """

    def __init__(self, dim: int, proj_dim: int | None = None, seed: int = 42):
        """Initialize QJL projection.

        Args:
            dim: Input vector dimension.
            proj_dim: Projection dimension (defaults to dim).
            seed: Random seed for reproducible projection matrix.
        """
        self.dim = dim
        self.proj_dim = proj_dim or dim

        # Generate random Gaussian projection matrix
        # Using numpy for deterministic init, then convert to MLX
        rng = np.random.RandomState(seed)
        # S ~ N(0, 1) — raw Gaussian entries, no normalization
        # The scale factor in dequantize() accounts for the projection dimension
        s_np = rng.randn(self.proj_dim, dim).astype(np.float32)
        self.projection_matrix = mx.array(s_np)  # (proj_dim, dim)

        # Precompute scaling factor: sqrt(pi/2) / proj_dim
        self._scale = float(np.sqrt(np.pi / 2) / self.proj_dim)

    def quantize(self, residuals: mx.array) -> QJLState:
        """Compress residual vectors to 1-bit sign projections.

        Args:
            residuals: (batch, dim) residual vectors.

        Returns:
            QJLState with sign bits and norms.
        """
        # Compute norms before projection
        norms = mx.sqrt(mx.sum(residuals * residuals, axis=-1))  # (batch,)

        # Project: (batch, dim) @ (dim, proj_dim) → (batch, proj_dim)
        projected = residuals @ self.projection_matrix.T

        # Sign quantize
        sign_bits = mx.sign(projected).astype(mx.int8)  # {-1, 0, +1}
        # Replace zeros with +1 (arbitrary but consistent)
        sign_bits = mx.where(sign_bits == 0, mx.array(1, dtype=mx.int8), sign_bits)

        # Bit-pack signs: 8 per byte
        packed, orig_dim = pack_signs(sign_bits)

        return QJLState(sign_bits_packed=packed, sign_orig_dim=orig_dim, norms=norms)

    def dequantize(self, state: QJLState) -> mx.array:
        """Reconstruct approximate residuals from QJL state.

        The reconstruction is not exact but provides an unbiased
        inner-product estimator: E[<y, r_hat>] = <y, r>.

        Args:
            state: QJLState from quantize().

        Returns:
            (batch, dim) approximate residual vectors.
        """
        # Unpack sign bits
        sign_bits = unpack_signs(state.sign_bits_packed, state.sign_orig_dim)

        # r_hat = scale * ||r|| * S^T @ signs
        # (batch, proj_dim) @ (proj_dim, dim) → (batch, dim)
        signs_float = sign_bits.astype(mx.float32)
        reconstructed = signs_float @ self.projection_matrix  # (batch, dim)

        # Scale by norms and constant
        norms = mx.expand_dims(state.norms, -1)  # (batch, 1)
        reconstructed = reconstructed * norms * self._scale

        return reconstructed
