"""PolarQuant: KV cache quantization via polar coordinate transformation.

Implements the algorithm from arXiv:2502.02617:
1. Random preconditioning (orthogonal rotation)
2. Recursive polar transform (atan2 + norm at each level)
3. Scalar quantization of angles using level-specific Lloyd-Max codebooks
4. Inverse transform for dequantization

The polar angles at each recursion level follow Beta-like distributions
with known shapes, enabling near-optimal codebook design.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .codebook import (
    dequantize_scalar,
    get_polar_codebook,
    quantize_scalar,
)


@dataclass
class PolarQuantConfig:
    """Configuration for PolarQuant compression."""

    # Bit allocations per level (level 1 = base pairs, level 2+ = sub-norms)
    level_bits: tuple[int, ...] = (4, 2, 2, 2)
    # Number of recursion levels to quantize (remaining stored as float16)
    num_levels: int = 4
    # Norm precision
    norm_dtype: mx.Dtype = mx.float16

    @property
    def total_bits_per_dim(self) -> float:
        """Approximate bits per dimension for the quantized representation."""
        # Each level l has d/2^l angles at level_bits[l] bits
        # Plus float16 norm
        bits = sum(b / (2**i) for i, b in enumerate(self.level_bits))
        return bits


@dataclass
class PolarQuantState:
    """Compressed representation of a vector batch after PolarQuant."""

    # List of index tensors, one per level: shapes (batch, d/2^l)
    angle_indices: list[mx.array]
    # Original norms: (batch,)
    norms: mx.array
    # Residual sub-norms for unquantized levels (if any): (batch, d/2^L)
    residual_subnorms: mx.array | None
    # The random matrix used (stored by reference in the layer)
    config: PolarQuantConfig


class PolarQuantLayer:
    """PolarQuant compression/decompression for a single attention head dimension."""

    def __init__(self, head_dim: int, config: PolarQuantConfig | None = None, seed: int = 0):
        self.head_dim = head_dim
        self.config = config or PolarQuantConfig()

        # Ensure head_dim is power of 2 (pad if needed)
        self._padded_dim = 1
        while self._padded_dim < head_dim:
            self._padded_dim *= 1  # Will handle non-power-of-2 below
            self._padded_dim = head_dim  # For now, assume power of 2
            break

        # Generate random orthogonal matrix via QR of Gaussian
        rng_key = mx.random.key(seed)
        gaussian = mx.random.normal(shape=(head_dim, head_dim), key=rng_key)
        # QR decomposition for orthogonal matrix
        # MLX doesn't have QR, so use numpy for init (one-time cost)
        g_np = np.array(gaussian)
        q_np, _ = np.linalg.qr(g_np)
        self.rotation_matrix = mx.array(q_np, dtype=mx.float32)  # (d, d)

        # Precompute codebooks for each level
        self._codebooks: list[mx.array] = []
        for level_idx in range(self.config.num_levels):
            level = level_idx + 1  # 1-indexed
            bits = self.config.level_bits[level_idx]
            cb = get_polar_codebook(level, bits)
            self._codebooks.append(cb)

    def _polar_transform(self, z: mx.array) -> tuple[list[mx.array], mx.array]:
        """Recursive polar coordinate decomposition.

        Args:
            z: (batch, d) preconditioned vectors.

        Returns:
            angles: List of angle tensors per level.
            norms: (batch,) overall norms.
        """
        batch = z.shape[0]
        d = z.shape[1]
        norms = mx.sqrt(mx.sum(z * z, axis=-1))  # (batch,)

        angles_per_level: list[mx.array] = []

        # Level 1: pair adjacent coordinates → atan2
        # z reshaped to (batch, d/2, 2)
        pairs = z.reshape(batch, d // 2, 2)
        level1_angles = mx.arctan2(pairs[:, :, 1], pairs[:, :, 0])  # (batch, d/2)
        # Shift to [0, 2*pi)
        level1_angles = mx.where(level1_angles < 0, level1_angles + 2 * mx.array(np.pi), level1_angles)
        angles_per_level.append(level1_angles)

        # Compute sub-norms for level 1: ||pair||
        subnorms = mx.sqrt(mx.sum(pairs * pairs, axis=-1))  # (batch, d/2)

        # Levels 2+: pair adjacent sub-norms → atan2
        for level_idx in range(1, self.config.num_levels):
            n_groups = subnorms.shape[1] // 2
            if n_groups == 0:
                break
            paired = subnorms.reshape(batch, n_groups, 2)
            # atan2(right_norm, left_norm) → angle in [0, pi/2]
            level_angles = mx.arctan2(paired[:, :, 1], paired[:, :, 0])  # (batch, n_groups)
            angles_per_level.append(level_angles)

            # New sub-norms: ||pair_of_subnorms||
            subnorms = mx.sqrt(mx.sum(paired * paired, axis=-1))  # (batch, n_groups)

        return angles_per_level, norms

    def _inverse_polar(
        self,
        quantized_angles: list[mx.array],
        norms: mx.array,
        residual_subnorms: mx.array | None,
    ) -> mx.array:
        """Reconstruct vectors from quantized polar representation.

        Works from deepest level back to level 1.
        """
        batch = norms.shape[0]
        num_levels = len(quantized_angles)

        # Start reconstruction from the deepest level
        if residual_subnorms is not None:
            current = residual_subnorms  # (batch, d/2^L)
        else:
            # If all levels quantized, start from the norm
            current = norms.reshape(batch, 1)

        # Reconstruct from deepest level to level 2
        for level_idx in range(num_levels - 1, 0, -1):
            angles = quantized_angles[level_idx]  # (batch, n_groups)
            n_groups = angles.shape[1]
            # Expand current to pairs using cos/sin
            cos_a = mx.cos(angles)  # (batch, n_groups)
            sin_a = mx.sin(angles)
            # current should have n_groups elements
            if current.shape[1] != n_groups:
                # Broadcast norm to match
                current = mx.broadcast_to(
                    norms.reshape(batch, 1),
                    (batch, n_groups),
                )
            left = current * cos_a
            right = current * sin_a
            # Interleave: (batch, n_groups, 2) → (batch, 2*n_groups)
            current = mx.concatenate(
                [mx.expand_dims(left, -1), mx.expand_dims(right, -1)], axis=-1
            ).reshape(batch, 2 * n_groups)

        # Level 1: reconstruct coordinate pairs from angles and sub-norms
        level1_angles = quantized_angles[0]  # (batch, d/2)
        subnorms = current  # (batch, d/2) — the reconstructed sub-norms
        cos_a = mx.cos(level1_angles)
        sin_a = mx.sin(level1_angles)
        x_even = subnorms * cos_a
        x_odd = subnorms * sin_a
        # Interleave to get (batch, d)
        z_hat = mx.concatenate(
            [mx.expand_dims(x_even, -1), mx.expand_dims(x_odd, -1)], axis=-1
        ).reshape(batch, -1)

        return z_hat

    def quantize(self, x: mx.array) -> PolarQuantState:
        """Compress vectors using PolarQuant.

        Args:
            x: (batch, head_dim) or (seq_len, num_heads, head_dim) vectors.

        Returns:
            PolarQuantState with compressed representation.
        """
        orig_shape = x.shape
        # Flatten to (N, head_dim)
        x_flat = x.reshape(-1, self.head_dim)

        # Step 1: Random preconditioning
        z = x_flat @ self.rotation_matrix.T  # (N, d)

        # Step 2: Polar transform
        angles_per_level, norms = self._polar_transform(z)

        # Step 3: Quantize angles at each level
        angle_indices: list[mx.array] = []
        for level_idx, angles in enumerate(angles_per_level):
            if level_idx < len(self._codebooks):
                indices = quantize_scalar(angles, self._codebooks[level_idx])
                angle_indices.append(indices)
            else:
                break

        return PolarQuantState(
            angle_indices=angle_indices,
            norms=norms.astype(self.config.norm_dtype),
            residual_subnorms=None,
            config=self.config,
        )

    def dequantize(self, state: PolarQuantState) -> mx.array:
        """Decompress vectors from PolarQuantState.

        Returns:
            Reconstructed vectors with same shape as original input.
        """
        # Dequantize angles
        quantized_angles: list[mx.array] = []
        for level_idx, indices in enumerate(state.angle_indices):
            values = dequantize_scalar(indices, self._codebooks[level_idx])
            quantized_angles.append(values)

        norms = state.norms.astype(mx.float32)

        # Inverse polar transform
        z_hat = self._inverse_polar(quantized_angles, norms, state.residual_subnorms)

        # Inverse rotation
        x_hat = z_hat @ self.rotation_matrix  # rotation_matrix is orthogonal, so R @ R^T = I

        return x_hat
