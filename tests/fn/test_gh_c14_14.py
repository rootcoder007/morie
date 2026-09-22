"""Tests for gh_c14_14.ghosal_nig_proc."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_14 import ghosal_nig_proc


def test_gh_c14_14_basic():
    """Test basic functionality against the documented theory value 1/alpha."""
    alpha_par = 1.0
    u_max = 10.0
    n_grid = 6000
    result = ghosal_nig_proc(alpha_par=alpha_par, u_max=u_max, n_grid=n_grid)
    assert "estimate" in result
    estimate = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(estimate))  # N6: was a generator-guessed value

    # Independent expectation from the documented formula using midpoint quadrature:
    # estimate ~= sum_{i=0}^{n_grid-1} u_i * u_i^{-3/2} * exp(-alpha^2 u_i / 2)
    #          * (2 pi)^{-1/2} * (u_max / n_grid),  with u_i = (i + 0.5) * u_max / n_grid
    h = u_max / n_grid
    i = np.arange(n_grid, dtype=float)
    u = (i + 0.5) * h
    integrand = u * u ** (-1.5) * np.exp(-alpha_par ** 2 * u / 2.0) \
        / np.sqrt(2.0 * np.pi)
    expected_estimate = float(np.sum(integrand) * h)
    theory = 1.0 / alpha_par

    # The quadrature is what the function computes; allow a small tolerance for
    # the trapezoid-like midpoint rule on a half-infinite interval.
    assert np.all(np.isfinite(expected_estimate))
    assert abs(float(estimate) - expected_estimate) < 1e-3
    # The closed-form integral over u in (0, inf) of u*rho(u) du = 1/alpha.
    assert abs(float(estimate) - theory) < 0.05


def test_gh_c14_14_edge():
    """Test edge cases: scalar inputs and an alternate alpha produce finite estimate."""
    # Scalar alpha_par
    result = ghosal_nig_proc(alpha_par=2.0)
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(estimate)

    # Scalar tuple inputs (positional, matching documented arity)
    result2 = ghosal_nig_proc(3.0, 5.0, 2000)
    estimate2 = float(np.asarray(result2["estimate"], dtype=float))
    assert np.isfinite(estimate2)
