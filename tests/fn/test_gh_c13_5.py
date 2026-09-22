"""Tests for gh_c13_5.ghosal_bp_cont."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_5 import ghosal_bp_cont


def test_gh_c13_5_basic():
    """Test basic functionality."""
    c = 2.0
    t_max = 1.0
    n_grid = 2000
    result = ghosal_bp_cont(c=c, t_max=t_max, n_grid=n_grid)
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(estimate)
    # Quadrature check: the integral equals H0(t_max) = t_max
    expected = t_max
    assert np.isclose(estimate, expected)
    # gap should reflect the quadrature error
    gap = float(np.asarray(result["gap"], dtype=float))
    assert gap >= 0.0
    assert np.isclose(gap, abs(estimate - t_max))


def test_gh_c13_5_edge():
    """Test edge cases."""
    c = 2.0
    t_max = 3.5
    n_grid = 500
    result = ghosal_bp_cont(c=c, t_max=t_max, n_grid=n_grid)
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(estimate)
    # Independent check using the closed-form integral of c*(1-u)^(c-1) over [0,1]
    # which equals 1, so the expected total mass is t_max.
    expected = t_max
    assert np.isclose(estimate, expected, atol=5e-3)
    # Sanity on stored metadata
    assert result["H0_t_max"] == t_max
    assert result["method"] == "BP Levy measure (GvdV 2017 sec. 13.3.2)"
