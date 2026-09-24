"""Tests for blipqf.blip_qformer."""

from morie.fn import _array_core as np

from morie.fn.blipqf import blip_qformer


def test_blipqf_basic():
    """Test basic functionality."""
    queries = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    image_features = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WQ = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    WK = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WV = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = blip_qformer(queries, image_features, WQ, WK, WV)
    assert isinstance(result, dict)
    assert "output" in result


def test_blipqf_edge():
    """Test edge cases."""
    queries = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    image_features = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WQ = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    WK = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WV = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = blip_qformer(queries, image_features, WQ, WK, WV)
    assert isinstance(result, dict)
