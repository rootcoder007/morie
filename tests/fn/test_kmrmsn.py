"""Tests for kmrmsn.kamath_rms_norm."""

from morie.fn import _array_core as np

from morie.fn.kmrmsn import kamath_rms_norm


def test_kmrmsn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_rms_norm(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "y" in result


def test_kmrmsn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_rms_norm(x)
    assert isinstance(result, dict)
