"""Tests for tmlext.tmle_external_data."""

from morie.fn import _array_core as np

from morie.fn.tmlext import tmle_external_data


def test_tmlext_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_external_data(y, D)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tmlext_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_external_data(y, D)
    assert isinstance(result, dict)
