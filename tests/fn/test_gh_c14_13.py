"""Tests for gh_c14_13.ghosal_pk_levy."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_13 import ghosal_pk_levy


def test_gh_c14_13_basic():
    """Test basic functionality."""
    u_grid_max = 8.0
    n_grid = 4000
    result = ghosal_pk_levy(u_grid_max=u_grid_max, n_grid=n_grid)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # Independent quadrature of int_0^infty u * (1/u) * e^{-u} du = int_0^infty e^{-u} du = 1
    # using midpoints on the same grid.
    est_indep = 0.0
    for i in range(n_grid):
        u = (i + 0.5) * u_grid_max / n_grid
        est_indep += (1.0 / u) * np.exp(-u) * u * (u_grid_max / n_grid)
    assert np.isfinite(est_indep)
    assert abs(float(result["estimate"]) - float(est_indep)) < 1e-9
    assert abs(float(result["estimate"]) - 1.0) < 5e-3


def test_gh_c14_13_edge():
    """Test edge cases."""
    result = ghosal_pk_levy(u_grid_max=42.0, n_grid=1000)
    assert "estimate" in result
    assert np.isfinite(float(result["estimate"]))
    assert abs(float(result["estimate"]) - 1.0) < 5e-3
