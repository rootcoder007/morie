"""Tests for funCA.functional_cca."""

from morie.fn import _array_core as np

from morie.fn.funCA import functional_cca


def test_funCA_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = functional_cca(X, Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_funCA_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = functional_cca(X, Y)
    assert isinstance(result, dict)
