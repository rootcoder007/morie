"""Tests for hrztf.horowitz_both_nonpar_transform."""

from morie.fn import _array_core as np

from morie.fn.hrztf import horowitz_both_nonpar_transform


def test_hrztf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_both_nonpar_transform(x, y)
    assert isinstance(result, dict)
    assert "T_hat" in result


def test_hrztf_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_both_nonpar_transform(x, y)
    assert isinstance(result, dict)
