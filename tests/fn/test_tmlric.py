"""Tests for tmlric.tmle_rare_outcome."""

from morie.fn import _array_core as np

from morie.fn.tmlric import tmle_rare_outcome


def test_tmlric_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    prevalence = 0.1
    result = tmle_rare_outcome(y, D, X, prevalence)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tmlric_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    prevalence = 0.1
    result = tmle_rare_outcome(y, D, X, prevalence)
    assert isinstance(result, dict)
