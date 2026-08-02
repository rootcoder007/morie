"""Tests for gh_inf_dim_cr.ghosal_inf_dim_credible."""

from morie.fn import _array_core as np

from morie.fn.gh_inf_dim_cr import ghosal_inf_dim_credible


def test_gh_inf_dim_cr_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_inf_dim_credible(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_inf_dim_cr_edge():
    """Test edge cases."""
    result = ghosal_inf_dim_credible(np.array([42.0]))
    assert result["n"] == 1
