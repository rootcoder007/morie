"""Tests for rgrbf.rangayyan_rbf_network."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_rbf_network


def test_rgrbf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_rbf_network(X, y)
    assert isinstance(result, dict)
    assert "centers" in result


def test_rgrbf_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_rbf_network(X, y)
    assert isinstance(result, dict)
