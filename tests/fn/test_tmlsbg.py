"""Tests for tmlsbg.tmle_subgroup."""

from morie.fn import _array_core as np

from morie.fn.tmlsbg import tmle_subgroup


def test_tmlsbg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    subgroup = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_subgroup(y, D, X, subgroup)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tmlsbg_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    subgroup = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_subgroup(y, D, X, subgroup)
    assert isinstance(result, dict)
