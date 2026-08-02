"""Tests for gh_emp_bayes.ghosal_empirical_bayes_np."""

from morie.fn import _array_core as np

from morie.fn.gh_emp_bayes import ghosal_empirical_bayes_np


def test_gh_emp_bayes_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_empirical_bayes_np(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_emp_bayes_edge():
    """Test edge cases."""
    result = ghosal_empirical_bayes_np(np.array([42.0]))
    assert result["n"] == 1
