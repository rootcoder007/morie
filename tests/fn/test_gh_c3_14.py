"""Tests for gh_c3_14.ghosal_mpt_prior."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_14 import ghosal_mpt_prior


def test_gh_c3_14_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_mpt_prior(x)
    # The mixture-of-PT formula returns the density evaluated on a grid.
    assert "grid" in result
    assert "density" in result
    assert np.all(np.isfinite(np.asarray(result["density"], dtype=float)))

    # Shapes: grid and density must agree; density must integrate ~ to 1
    # over the grid (the components are densities on [lo, hi]).
    g = np.asarray(result["grid"], dtype=float)
    d = np.asarray(result["density"], dtype=float)
    assert g.shape == d.shape
    area = float(np.trapz(d, g))
    assert abs(area - 1.0) < 0.05

    # n and n_components from the documented bookkeeping.
    assert int(result["n"]) == int(np.asarray(x, dtype=float).ravel().size)
    assert int(result["n_components"]) == 8  # default linspace(0, 0.5, 8)

    # Smoothing: max jump of the mixture is bounded by the max jump of
    # any single component, and the result must report it as such.
    jm = float(result["max_jump"])
    js = float(result["max_jump_single"])
    assert jm <= js + 1e-12
    assert bool(result["smoother_than_single"]) is True


def test_gh_c3_14_edge():
    """Test edge cases."""
    # The docstring requires at least 4 observations; 1 must raise.
    import pytest
    with pytest.raises(ValueError):
        ghosal_mpt_prior(np.array([42.0]))
