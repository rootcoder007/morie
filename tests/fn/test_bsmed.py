"""Tests for bsmed.bootstrap_mediation_ci."""

from morie.fn import _array_core as np

from morie.fn.bsmed import bootstrap_mediation_ci


def test_bsmed_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = bootstrap_mediation_ci(X, M, Y, B)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bsmed_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = bootstrap_mediation_ci(X, M, Y, B)
    assert isinstance(result, dict)
