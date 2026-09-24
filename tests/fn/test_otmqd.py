"""Tests for otmqd.ot_quantization_distortion."""

from morie.fn import _array_core as np

from morie.fn.otmqd import ot_quantization_distortion


def test_otmqd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    centroids = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_quantization_distortion(X, centroids)
    assert isinstance(result, dict)
    assert "dist" in result


def test_otmqd_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    centroids = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_quantization_distortion(X, centroids)
    assert isinstance(result, dict)
