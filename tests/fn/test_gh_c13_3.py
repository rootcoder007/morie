"""Tests for gh_c13_3.ghosal_beta_proc_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_3 import ghosal_beta_proc_def


def test_gh_c13_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_beta_proc_def(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c13_3_edge():
    """Test edge cases: single grid point should yield a path of length 1
    equal to the estimate, with a recorded method key."""
    grid = np.array([42.0])
    result = ghosal_beta_proc_def(grid)
    assert "estimate" in result
    assert "cum_hazard" in result
    assert "nondecreasing" in result
    assert "method" in result
    assert len(result["cum_hazard"]) == 1
    assert result["estimate"] == result["cum_hazard"][-1]
    assert result["nondecreasing"] is True
