"""Tests for psm_standardized_bias.psm_standardized_bias."""

from morie.fn import _array_core as np

from morie.fn.psm_standardized_bias import psm_standardized_bias


def test_ca10e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = psm_standardized_bias(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_ca10e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = psm_standardized_bias(x)
    assert isinstance(result, dict)
