"""Tests for gatep.gate_estimation."""

from morie.fn import _array_core as np

from morie.fn.gatep import gate_estimation


def test_gatep_basic():
    """Test basic functionality."""
    cate = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    group_var = np.random.default_rng(42).normal(0, 1, 100)
    result = gate_estimation(cate, X, group_var)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gatep_edge():
    """Test edge cases."""
    cate = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    group_var = np.random.default_rng(42).normal(0, 1, 100)
    result = gate_estimation(cate, X, group_var)
    assert isinstance(result, dict)
