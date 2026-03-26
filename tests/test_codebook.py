"""Tests for Lloyd-Max codebook computation."""

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.codebook import (
    dequantize_scalar,
    get_gaussian_codebook,
    get_polar_codebook,
    lloyd_max_1d,
    quantize_scalar,
)


class TestLloydMax:
    def test_uniform_1bit(self):
        """1-bit uniform quantization should give 2 centroids near ±0.5."""
        centroids = lloyd_max_1d(lambda x: np.ones_like(x), (0.0, 1.0), bits=1)
        assert len(centroids) == 2
        assert centroids[0] < centroids[1]
        np.testing.assert_allclose(centroids, [0.25, 0.75], atol=0.05)

    def test_gaussian_2bit(self):
        """2-bit Gaussian codebook should have 4 symmetric centroids."""
        centroids = lloyd_max_1d(
            lambda x: np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi),
            (-4.0, 4.0),
            bits=2,
        )
        assert len(centroids) == 4
        # Should be approximately symmetric
        np.testing.assert_allclose(centroids[0], -centroids[3], atol=0.1)
        np.testing.assert_allclose(centroids[1], -centroids[2], atol=0.1)

    def test_gaussian_3bit(self):
        """3-bit Gaussian codebook should have 8 centroids."""
        centroids = lloyd_max_1d(
            lambda x: np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi),
            (-4.0, 4.0),
            bits=3,
        )
        assert len(centroids) == 8
        # Should be sorted
        assert all(centroids[i] < centroids[i + 1] for i in range(7))


class TestGaussianCodebook:
    def test_caching(self):
        """Repeated calls should return same object (cached)."""
        cb1 = get_gaussian_codebook(2)
        cb2 = get_gaussian_codebook(2)
        assert cb1 is cb2

    def test_shapes(self):
        for bits in [1, 2, 3]:
            cb = get_gaussian_codebook(bits)
            assert cb.shape == (2**bits,)
            assert cb.dtype == mx.float32


class TestPolarCodebook:
    def test_level1_uniform(self):
        """Level 1 codebook (uniform) should span [0, 2*pi)."""
        cb = get_polar_codebook(level=1, bits=4)
        assert cb.shape == (16,)
        vals = np.array(cb)
        assert vals[0] >= 0
        assert vals[-1] < 2 * np.pi

    def test_level2_concentrated(self):
        """Level 2+ codebook should be concentrated around pi/4."""
        cb = get_polar_codebook(level=3, bits=2)
        vals = np.array(cb)
        # Should be near pi/4 ≈ 0.785
        assert all(0 < v < np.pi / 2 for v in vals)


class TestQuantizeDequantize:
    def test_roundtrip_identity(self):
        """Quantize→dequantize of centroid values should be exact."""
        codebook = mx.array([0.0, 1.0, 2.0, 3.0])
        values = mx.array([0.0, 1.0, 2.0, 3.0])
        indices = quantize_scalar(values, codebook)
        recovered = dequantize_scalar(indices, codebook)
        np.testing.assert_allclose(np.array(recovered), np.array(values), atol=1e-6)

    def test_nearest_neighbor(self):
        """Values between centroids should map to nearest."""
        codebook = mx.array([0.0, 1.0, 2.0, 3.0])
        values = mx.array([0.3, 0.7, 1.6, 2.9])
        indices = quantize_scalar(values, codebook)
        expected = mx.array([0, 1, 2, 3])
        np.testing.assert_array_equal(np.array(indices), np.array(expected))

    def test_batch(self):
        """Should handle batched inputs."""
        codebook = mx.array([-1.0, 0.0, 1.0])
        values = mx.array([[0.5, -0.5], [0.1, -0.8]])
        indices = quantize_scalar(values, codebook)
        assert indices.shape == (2, 2)
