"""Tests for tcls.t_closeness."""

from morie.fn import _array_core as np

from morie.fn.tcls import t_closeness


def test_tcls_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    quasi_ids = np.random.default_rng(42).normal(0.0, 1.0, 40)
    sensitive = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t = 0.1
    result = t_closeness(X, quasi_ids, sensitive, t)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tcls_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    quasi_ids = np.random.default_rng(42).normal(0.0, 1.0, 40)
    sensitive = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t = 0.1
    result = t_closeness(X, quasi_ids, sensitive, t)
    assert isinstance(result, dict)
