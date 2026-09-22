"""Tests for gh_c8_4.ghosal_prior_mass_cnd."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_4 import ghosal_prior_mass_cnd


def test_gh_c8_4_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_prior_mass_cnd(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert "log_mass" in result
    assert "positive" in result
    assert result["positive"] is True or result["positive"] is False
    assert result["method"].startswith("B_2")
    mass = result["estimate"]
    assert 0.0 <= mass <= 1.0


def test_gh_c8_4_edge():
    """Test edge cases: degenerate single-component weight."""
    p0 = np.array([42.0])
    result = ghosal_prior_mass_cnd(p0)
    assert "estimate" in result
    mass = result["estimate"]
    assert 0.0 <= mass <= 1.0
