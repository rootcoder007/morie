"""Tests for gh_c14_20.ghosal_probit_sbp."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_20 import ghosal_probit_sbp


def test_gh_c14_20_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_probit_sbp(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c14_20_edge():
    """Test edge cases."""
    result = ghosal_probit_sbp(np.array([42.0]))
    assert result["n"] == 1
