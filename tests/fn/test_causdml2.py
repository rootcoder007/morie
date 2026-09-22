"""Tests for causdml2.causal_dml_partial_lin."""

from morie.fn import _array_core as np

from morie.fn.causdml2 import causal_dml_partial_lin


def test_causdml2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_dml_partial_lin(y, D, X)
    assert isinstance(result, dict)
    assert "theta" in result
def test_causdml2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_dml_partial_lin(y, D, X)
    assert isinstance(result, dict)
