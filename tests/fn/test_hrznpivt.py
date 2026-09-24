"""Tests for hrznpivt.horowitz_npiv_unknown_T."""

from morie.fn import _array_core as np

from morie.fn.hrznpivt import horowitz_npiv_unknown_T


def test_hrznpivt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    W = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_npiv_unknown_T(X, W)
    assert isinstance(result, dict)
    assert "T" in result


def test_hrznpivt_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    W = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_npiv_unknown_T(X, W)
    assert isinstance(result, dict)
