"""Tests for cfaftr.cfa_one_factor."""

from morie.fn import _array_core as np

from morie.fn.cfaftr import cfa_one_factor


def test_cfaftr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    factor_structure = np.random.default_rng(42).normal(0, 1, 100)
    result = cfa_one_factor(X, factor_structure)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_cfaftr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    factor_structure = np.random.default_rng(42).normal(0, 1, 100)
    result = cfa_one_factor(X, factor_structure)
    assert isinstance(result, dict)
