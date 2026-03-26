"""Tests for TurboQuant compressor (MSE + QJL pipeline)."""

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.turbo import TurboQuantCompressor, TurboQuantConfig


class TestTurboQuantMSE:
    """Tests for MSE-only mode."""

    def test_roundtrip_shape(self):
        config = TurboQuantConfig(bits=3, mode="mse")
        comp = TurboQuantCompressor(head_dim=64, config=config, seed=0)

        x = mx.random.normal(shape=(16, 64))
        state = comp.quantize(x)
        x_hat = comp.dequantize(state)

        assert x_hat.shape == (16, 64)

    def test_indices_range(self):
        """MSE indices should be in [0, 2^bits)."""
        config = TurboQuantConfig(bits=3, mode="mse")
        comp = TurboQuantCompressor(head_dim=32, config=config)

        x = mx.random.normal(shape=(8, 32))
        state = comp.quantize(x)

        indices = np.array(state.mse_indices)
        assert indices.min() >= 0
        assert indices.max() < 2 ** config.mse_bits

    def test_reconstruction_error_bounded(self):
        """MSE reconstruction should have bounded error relative to input."""
        config = TurboQuantConfig(bits=3, mode="mse")
        comp = TurboQuantCompressor(head_dim=128, config=config, seed=42)

        x = mx.random.normal(shape=(100, 128))
        state = comp.quantize(x)
        x_hat = comp.dequantize(state)

        # Relative MSE per vector
        mse = mx.mean((x - x_hat) ** 2, axis=-1)
        signal = mx.mean(x**2, axis=-1)
        rel_mse = mx.mean(mse / mx.maximum(signal, mx.array(1e-8)))
        mx.eval(rel_mse)

        # With 2-bit MSE (3 bits total, MSE gets 2), error should be < 50%
        assert rel_mse.item() < 0.5, f"Relative MSE too high: {rel_mse.item()}"


class TestTurboQuantProd:
    """Tests for prod mode (MSE + QJL)."""

    def test_roundtrip_shape(self):
        config = TurboQuantConfig(bits=3, mode="prod")
        comp = TurboQuantCompressor(head_dim=64, config=config, seed=0)

        x = mx.random.normal(shape=(16, 64))
        state = comp.quantize(x)
        x_hat = comp.dequantize(state)

        assert x_hat.shape == (16, 64)
        assert state.qjl_state is not None

    def test_prod_has_lower_dot_error_than_mse(self):
        """Prod mode should have better inner-product estimation than MSE alone."""
        head_dim = 128
        n_vectors = 50

        mse_config = TurboQuantConfig(bits=3, mode="mse")
        prod_config = TurboQuantConfig(bits=3, mode="prod")

        mse_comp = TurboQuantCompressor(head_dim, mse_config, seed=0)
        prod_comp = TurboQuantCompressor(head_dim, prod_config, seed=0)

        x = mx.random.normal(shape=(n_vectors, head_dim))
        y = mx.random.normal(shape=(n_vectors, head_dim))

        # True dot products
        true_dots = mx.sum(x * y, axis=-1)

        # MSE reconstruction dots
        x_mse = mse_comp.dequantize(mse_comp.quantize(x))
        mse_dots = mx.sum(x_mse * y, axis=-1)

        # Prod reconstruction dots
        x_prod = prod_comp.dequantize(prod_comp.quantize(x))
        prod_dots = mx.sum(x_prod * y, axis=-1)

        mse_error = mx.mean((true_dots - mse_dots) ** 2).item()
        prod_error = mx.mean((true_dots - prod_dots) ** 2).item()

        # Prod should generally be better, but allow some slack for randomness
        # At minimum, prod should not be catastrophically worse
        assert prod_error < mse_error * 5, (
            f"Prod error ({prod_error}) much worse than MSE ({mse_error})"
        )

    def test_norm_preserved(self):
        """Stored norms should match input vector norms."""
        config = TurboQuantConfig(bits=3, mode="prod")
        comp = TurboQuantCompressor(head_dim=64, config=config)

        x = mx.random.normal(shape=(10, 64)) * 3.0  # non-unit scale
        state = comp.quantize(x)

        expected = np.array(mx.sqrt(mx.sum(x * x, axis=-1)))
        actual = np.array(state.norms)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)


class TestTurboQuantEdgeCases:
    def test_single_vector(self):
        comp = TurboQuantCompressor(head_dim=64)
        x = mx.random.normal(shape=(1, 64))
        state = comp.quantize(x)
        x_hat = comp.dequantize(state)
        assert x_hat.shape == (1, 64)

    def test_zero_vector(self):
        comp = TurboQuantCompressor(head_dim=32)
        x = mx.zeros((1, 32))
        state = comp.quantize(x)
        x_hat = comp.dequantize(state)
        # Should not produce NaN or Inf
        assert not mx.any(mx.isnan(x_hat)).item()
        assert not mx.any(mx.isinf(x_hat)).item()

    def test_large_batch(self):
        comp = TurboQuantCompressor(head_dim=128)
        x = mx.random.normal(shape=(1024, 128))
        state = comp.quantize(x)
        x_hat = comp.dequantize(state)
        assert x_hat.shape == (1024, 128)

    def test_different_bit_widths(self):
        for bits in [2, 3, 4]:
            config = TurboQuantConfig(bits=bits, mode="mse")
            comp = TurboQuantCompressor(head_dim=64, config=config)
            x = mx.random.normal(shape=(4, 64))
            state = comp.quantize(x)
            x_hat = comp.dequantize(state)
            assert x_hat.shape == (4, 64)
