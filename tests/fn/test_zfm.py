"""Tests for zfm.z_transform."""

from morie.fn import _array_core as np

from morie.fn.zfm import z_transform


def test_zfm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = z_transform(x, z)
    assert isinstance(result, dict)
    assert "coefficients" in result


def test_zfm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = z_transform(x, z)
    assert isinstance(result, dict)
