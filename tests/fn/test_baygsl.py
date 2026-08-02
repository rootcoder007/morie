"""Tests for baygsl.gibbs_slice."""

from morie.fn import _array_core as np

from morie.fn.baygsl import gibbs_slice


def test_baygsl_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = gibbs_slice(model, x0, n_iter)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_baygsl_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = gibbs_slice(model, x0, n_iter)
    assert isinstance(result, dict)
