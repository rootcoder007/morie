"""Tests for gh_c2_3.ghosal_gp_increasing_prior."""

from morie.fn import _array_core as np

from morie.fn.gh_c2_3 import ghosal_gp_increasing_prior


def test_gh_c2_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_gp_increasing_prior(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c2_3_edge():
    """Test edge case: single point integrates to zero."""
    x = np.array([42.0])
    result = ghosal_gp_increasing_prior(x)
    # With a single point there is no integration interval,
    # so the cumulative integral F is [0.0] and the estimate is 0.
    assert "estimate" in result
    assert result["estimate"] == 0.0
    assert np.all(np.isfinite(np.asarray(result["F"], dtype=float)))
    assert result["increasing"] is True
    assert len(result["F"]) == 1
    assert result["F"][0] == 0.0
