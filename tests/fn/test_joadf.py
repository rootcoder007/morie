"""Tests for joadf.joseph_adf_unit_root_test."""

from morie.fn import _array_core as np

from morie.fn.joadf import joseph_adf_unit_root_test


def test_joadf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_adf_unit_root_test(x)
    assert isinstance(result, dict)
    assert "stat" in result or "stat" in result


def test_joadf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_adf_unit_root_test(x)
    assert isinstance(result, dict)
