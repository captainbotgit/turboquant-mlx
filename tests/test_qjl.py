"""Tests for QJL 1-bit residual projection."""

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.qjl import QJLProjection, QJLState


class TestQJLProjection:
    def test_quantize_shape(self):
        """Output shapes should match input batch and projection dim."""
        qjl = QJLProjection(dim=64, proj_dim=64, seed=0)
        x = mx.random.normal(shape=(10, 64))
        state = qjl.quantize(x)

        assert state.sign_bits.shape == (10, 64)
        assert state.norms.shape == (10,)

    def test_sign_bits_are_binary(self):
        """Sign bits should be {-1, +1} only."""
        qjl = QJLProjection(dim=32, seed=0)
        x = mx.random.normal(shape=(5, 32))
        state = qjl.quantize(x)

        signs = np.array(state.sign_bits)
        unique = set(signs.flatten())
        assert unique <= {-1, 1}

    def test_dequantize_shape(self):
        """Dequantized output should match input shape."""
        qjl = QJLProjection(dim=64, seed=0)
        x = mx.random.normal(shape=(8, 64))
        state = qjl.quantize(x)
        x_hat = qjl.dequantize(state)

        assert x_hat.shape == (8, 64)

    def test_unbiased_dot_product(self):
        """Inner product estimation should be approximately unbiased over many samples."""
        dim = 128
        n_trials = 500
        qjl = QJLProjection(dim=dim, seed=42)

        true_dots = []
        est_dots = []

        for i in range(n_trials):
            key = mx.random.key(i)
            x = mx.random.normal(shape=(1, dim), key=key)
            y = mx.random.normal(shape=(1, dim), key=mx.random.key(i + 10000))

            # True dot product
            true_dot = mx.sum(x * y).item()
            true_dots.append(true_dot)

            # Estimated via QJL
            state = qjl.quantize(x)
            x_hat = qjl.dequantize(state)
            est_dot = mx.sum(x_hat * y).item()
            est_dots.append(est_dot)

        # Mean estimated dot should be close to mean true dot (unbiased)
        true_mean = np.mean(true_dots)
        est_mean = np.mean(est_dots)
        # Allow generous tolerance — this is a statistical test
        assert abs(est_mean - true_mean) < 0.5 * abs(true_mean) + 0.5

    def test_norm_preservation(self):
        """Stored norms should match input norms."""
        qjl = QJLProjection(dim=64, seed=0)
        x = mx.random.normal(shape=(4, 64))
        state = qjl.quantize(x)

        input_norms = np.array(mx.sqrt(mx.sum(x * x, axis=-1)))
        stored_norms = np.array(state.norms)
        np.testing.assert_allclose(stored_norms, input_norms, rtol=1e-5)
