"""Tests for gh_c7_6.ghosal_pt_dens_con."""

from morie.fn import _array_core as np

from morie.fn.gh_c7_6 import ghosal_pt_dens_con


def test_gh_c7_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_pt_dens_con(x)
    # Documented keys: grid, density, levels, a_rule,
    # absolutely_continuous_prior, mass, consistent_at, n, method.
    assert "grid" in result
    assert "density" in result
    assert "levels" in result
    assert "a_rule" in result
    assert "absolutely_continuous_prior" in result
    assert "mass" in result
    assert "consistent_at" in result
    assert "n" in result
    assert "method" in result

    dens = np.asarray(result["density"], dtype=float)
    g = np.asarray(result["grid"], dtype=float)
    assert np.all(np.isfinite(dens))
    assert np.all(dens >= 0.0)

    # The function's closed-form posterior-mean density is a product of
    # Beta posterior means at every split. Here we just sanity-check the
    # bookkeeping: defaults and shape relationships.
    assert result["levels"] == 6
    assert result["absolutely_continuous_prior"] is True
    assert result["n"] == 5
    # When no grid is supplied, the implementation uses 200 evaluation
    # points spanning [min(x), max(x)].
    assert g.size == 200
    assert g[0] == 1.0 and g[-1] == 5.0

    # Independent computation of mass: trapezoidal rule on the returned
    # grid/density pair. (Same formula as np.trapezoid; written out to
    # avoid copying the implementation's result.)
    diffs = np.diff(g)
    mids = 0.5 * (dens[:-1] + dens[1:])
    mass_indep = float(np.sum(diffs * mids))
    assert np.isfinite(result["mass"])
    assert abs(result["mass"] - mass_indep) < 1e-12


def test_gh_c7_6_edge():
    """Test edge cases."""
    import pytest
    # The docstring says: "need at least 4 observations"; a single point
    # must therefore raise rather than return n=1.
    with pytest.raises(ValueError):
        ghosal_pt_dens_con(np.array([42.0]))

    # With enough data the call succeeds and n matches.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = ghosal_pt_dens_con(x)
    assert result["n"] == 4
