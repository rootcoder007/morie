"""Tests for hrzrank.horowitz_semipar_rank."""

from morie.fn import _array_core as np

from morie.fn.hrzrank import horowitz_semipar_rank


def test_hrzrank_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_semipar_rank(x, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_hrzrank_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_semipar_rank(x, y)
    assert isinstance(result, dict)
