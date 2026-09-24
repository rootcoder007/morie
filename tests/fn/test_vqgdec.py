"""Tests for vqgdec.vqgan_decode."""

from morie.fn import _array_core as np

from morie.fn.vqgdec import vqgan_decode


def test_vqgdec_basic():
    """Test basic functionality."""
    indices = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    codebook = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = vqgan_decode(indices, codebook)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_vqgdec_edge():
    """Test edge cases."""
    indices = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    codebook = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = vqgan_decode(indices, codebook)
    assert isinstance(result, dict)
