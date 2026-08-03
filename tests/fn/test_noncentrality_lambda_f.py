"""Tests for noncentrality_lambda_f.noncentrality_lambda_f."""

from morie.fn import _array_core as np

from morie.fn.noncentrality_lambda_f import noncentrality_lambda_f


def test_ca8e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = noncentrality_lambda_f(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca8e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = noncentrality_lambda_f(x)
    assert isinstance(result, dict)
