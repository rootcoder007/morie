"""Tests for seckdf.hkdf_extract_expand."""

from morie.fn import _array_core as np

from morie.fn.seckdf import hkdf_extract_expand


def test_seckdf_basic():
    """Test basic functionality."""
    ikm = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = hkdf_extract_expand(ikm)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_seckdf_edge():
    """Test edge cases."""
    ikm = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = hkdf_extract_expand(ikm)
    assert isinstance(result, dict)
