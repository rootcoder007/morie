"""Tests for posvt.positivity_assumption."""

from morie.fn import _array_core as np

from morie.fn.posvt import positivity_assumption


def test_posvt_basic():
    """Test basic functionality."""
    treat = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    stratum = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = positivity_assumption(treat, stratum)
    assert isinstance(result, dict)
    assert "minprob" in result


def test_posvt_edge():
    """Test edge cases."""
    treat = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    stratum = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = positivity_assumption(treat, stratum)
    assert isinstance(result, dict)
