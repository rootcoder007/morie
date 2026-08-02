"""Tests for gh_c1_1.ghosal_bayes_rule_infinite."""

from morie.fn import _array_core as np

from morie.fn.gh_c1_1 import ghosal_bayes_rule_infinite


def test_gh_c1_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_bayes_rule_infinite(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c1_1_edge():
    """Test edge cases."""
    result = ghosal_bayes_rule_infinite(np.array([42.0]))
    assert result["n"] == 1
