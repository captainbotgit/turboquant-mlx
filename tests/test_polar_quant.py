"""Tests for PolarQuant module."""

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.polar_quant import PolarQuantConfig, PolarQuantLayer


class TestPolarQuantLayer:
    def test_roundtrip_shape(self):
        """Quantize→dequantize should preserve shape."""
        layer = PolarQuantLayer(head_dim=64, seed=0)
        x = mx.random.normal(shape=(8, 64))
        state = layer.quantize(x)
        x_hat = layer.dequantize(state)
        assert x_hat.shape == (8, 64)

    def test_rotation_matrix_orthogonal(self):
        """Random rotation matrix should be orthogonal (R @ R^T ≈ I)."""
        layer = PolarQuantLayer(head_dim=32, seed=0)
        R = np.array(layer.rotation_matrix)
        product = R @ R.T
        np.testing.assert_allclose(product, np.eye(32), atol=1e-5)

    def test_polar_transform_levels(self):
        """Polar transform should produce correct number of angle levels."""
        layer = PolarQuantLayer(head_dim=64, seed=0)
        x = mx.random.normal(shape=(4, 64))
        z = x @ layer.rotation_matrix.T
        angles, norms = layer._polar_transform(z)

        # Level 1: d/2 = 32 angles
        assert angles[0].shape == (4, 32)
        # Level 2: d/4 = 16 angles
        assert angles[1].shape == (4, 16)
        # Level 3: d/8 = 8 angles
        assert angles[2].shape == (4, 8)
        # Level 4: d/16 = 4 angles
        assert angles[3].shape == (4, 4)

    def test_norms_match_input(self):
        """Extracted norms should match input vector norms."""
        layer = PolarQuantLayer(head_dim=64, seed=0)
        x = mx.random.normal(shape=(10, 64)) * 2.0
        z = x @ layer.rotation_matrix.T

        # Norms of z should equal norms of x (rotation preserves norms)
        x_norms = np.array(mx.sqrt(mx.sum(x * x, axis=-1)))
        z_norms = np.array(mx.sqrt(mx.sum(z * z, axis=-1)))
        np.testing.assert_allclose(z_norms, x_norms, rtol=1e-5)

    def test_reconstruction_not_nan(self):
        """Reconstruction should not contain NaN or Inf."""
        layer = PolarQuantLayer(head_dim=128, seed=42)
        x = mx.random.normal(shape=(16, 128))
        state = layer.quantize(x)
        x_hat = layer.dequantize(state)

        assert not mx.any(mx.isnan(x_hat)).item()
        assert not mx.any(mx.isinf(x_hat)).item()

    def test_config_bits_per_dim(self):
        """Config should report reasonable bits per dimension."""
        config = PolarQuantConfig(level_bits=(4, 2, 2, 2))
        bpd = config.total_bits_per_dim
        # 4/1 + 2/2 + 2/4 + 2/8 = 4 + 1 + 0.5 + 0.25 = 5.75
        assert abs(bpd - 5.75) < 0.01

    def test_single_vector(self):
        layer = PolarQuantLayer(head_dim=32, seed=0)
        x = mx.random.normal(shape=(1, 32))
        state = layer.quantize(x)
        x_hat = layer.dequantize(state)
        assert x_hat.shape == (1, 32)
