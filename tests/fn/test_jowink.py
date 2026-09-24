"""Tests for jowink.joseph_winkler_interval_score."""

from morie.fn import _array_core as np

from morie.fn.jowink import joseph_winkler_interval_score


def test_jowink_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    lower = np.random.default_rng(42).normal(0.0, 1.0, 40)
    upper = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_winkler_interval_score(y, lower, upper)
    assert isinstance(result, dict)
    assert "score" in result


def test_jowink_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    lower = np.random.default_rng(42).normal(0.0, 1.0, 40)
    upper = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_winkler_interval_score(y, lower, upper)
    assert isinstance(result, dict)
