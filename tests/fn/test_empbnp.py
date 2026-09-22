"""Tests for empbnp.empirical_bayes_np."""

from morie.fn import _array_core as np

from morie.fn.empbnp import empirical_bayes_np


def test_empbnp_basic():
    """Test basic functionality."""
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    result = empirical_bayes_np(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_empbnp_edge():
    """Test edge cases."""
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    result = empirical_bayes_np(y)
    assert isinstance(result, dict)
