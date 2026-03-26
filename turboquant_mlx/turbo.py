"""TurboQuant: Online vector quantization combining MSE quantization + QJL residual.

Implements the full TurboQuant pipeline from arXiv:2504.19874:
  Stage 1 (TurboQuant_mse): Random rotation → scalar quantization with
    Lloyd-Max codebooks on Gaussian marginals.
  Stage 2 (TurboQuant_prod): MSE quantization at (b-1) bits + 1-bit QJL
    residual projection for unbiased inner-product estimation.

This module provides the main TurboQuantCompressor class that manages
KV cache compression for an attention head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np

from .codebook import (
    dequantize_scalar,
    get_gaussian_codebook,
    quantize_scalar,
)
from .qjl import QJLProjection, QJLState


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression."""

    # Total bit budget per dimension (e.g., 3 = 2 bits MSE + 1 bit QJL)
    bits: int = 3
    # Whether to use QJL residual (TurboQuant_prod) or just MSE (TurboQuant_mse)
    mode: Literal["prod", "mse"] = "prod"

    @property
    def mse_bits(self) -> int:
        """Bits allocated to MSE stage."""
        if self.mode == "prod":
            return self.bits - 1
        return self.bits


@dataclass
class TurboQuantState:
    """Compressed representation of a KV cache vector batch."""

    # MSE quantization indices: (batch, dim), dtype int8/int16
    mse_indices: mx.array
    # Vector norms for rescaling: (batch,)
    norms: mx.array
    # QJL state for residual (None if mode="mse")
    qjl_state: QJLState | None
    # Config reference
    config: TurboQuantConfig


class TurboQuantCompressor:
    """TurboQuant compressor for a single attention head dimension.

    Manages the random rotation matrix and QJL projection for
    compressing/decompressing KV cache vectors.
    """

    def __init__(
        self,
        head_dim: int,
        config: TurboQuantConfig | None = None,
        seed: int = 0,
    ):
        self.head_dim = head_dim
        self.config = config or TurboQuantConfig()

        # Generate random orthogonal matrix via QR of Gaussian
        rng = np.random.RandomState(seed)
        gaussian = rng.randn(head_dim, head_dim).astype(np.float32)
        q_np, _ = np.linalg.qr(gaussian)
        self.rotation_matrix = mx.array(q_np, dtype=mx.float32)

        # Precompute MSE codebook
        self._mse_codebook = get_gaussian_codebook(self.config.mse_bits)

        # Initialize QJL if using prod mode
        self._qjl: QJLProjection | None = None
        if self.config.mode == "prod":
            self._qjl = QJLProjection(head_dim, seed=seed + 1000)

    def quantize(self, x: mx.array) -> TurboQuantState:
        """Compress vectors using TurboQuant.

        Args:
            x: (batch, head_dim) vectors to compress.

        Returns:
            TurboQuantState with compressed representation.
        """
        batch = x.shape[0]

        # Compute and save norms
        norms = mx.sqrt(mx.sum(x * x, axis=-1))  # (batch,)

        # Normalize to unit vectors for rotation
        x_normalized = x / mx.maximum(mx.expand_dims(norms, -1), mx.array(1e-8))

        # Random rotation
        y = x_normalized @ self.rotation_matrix.T  # (batch, d)

        # Scalar quantization of each coordinate
        # After rotation of unit vectors, coordinates ≈ N(0, 1/d)
        # Scale to standard normal for codebook lookup
        scale = mx.array(np.sqrt(self.head_dim), dtype=mx.float32)
        y_scaled = y * scale

        mse_indices = quantize_scalar(y_scaled, self._mse_codebook)

        # QJL residual
        qjl_state = None
        if self._qjl is not None:
            # Reconstruct MSE approximation to compute residual
            y_hat_scaled = dequantize_scalar(mse_indices, self._mse_codebook)
            y_hat = y_hat_scaled / scale
            x_hat_normalized = y_hat @ self.rotation_matrix
            x_hat = x_hat_normalized * mx.expand_dims(norms, -1)

            residual = x - x_hat
            qjl_state = self._qjl.quantize(residual)

        return TurboQuantState(
            mse_indices=mse_indices,
            norms=norms,
            qjl_state=qjl_state,
            config=self.config,
        )

    def dequantize(self, state: TurboQuantState) -> mx.array:
        """Decompress vectors from TurboQuantState.

        Returns:
            (batch, head_dim) reconstructed vectors.
        """
        scale = mx.array(np.sqrt(self.head_dim), dtype=mx.float32)

        # MSE reconstruction
        y_hat_scaled = dequantize_scalar(state.mse_indices, self._mse_codebook)
        y_hat = y_hat_scaled / scale
        x_hat_normalized = y_hat @ self.rotation_matrix
        x_hat = x_hat_normalized * mx.expand_dims(state.norms, -1)

        # Add QJL residual if available
        if state.qjl_state is not None and self._qjl is not None:
            residual_hat = self._qjl.dequantize(state.qjl_state)
            x_hat = x_hat + residual_hat

        return x_hat
