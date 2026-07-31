"""Tests for spols: OLS semivariogram fitting (Schabenberger & Gotway Sec 4.5.1).

OLS is eq (4.31) under the simplification R = phi*I named just after (4.34).
The fit is by Gauss-Newton, which is the algorithm the text names for this
problem (after 4.43), with the derivatives of (4.42) taken analytically.
"""

import numpy as np
import pytest

from morie.fn._schab_fit import _objective, _start_and_bounds
from morie.fn._schab_gn import semivariogram_jacobian
from morie.fn._schab_vario import semivariogram
from morie.fn.spols import schabenberger_ols_variogram as spols


def test_ols_objective_is_the_plain_residual_sum_of_squares(schab_fit_table):
    lag, gam, cnt = schab_fit_table
    obj, ok = _objective("ols", lag, gam, cnt, "exponential")
    theta = np.array([0.3, 2.0, 6.0])
    resid = gam[ok] - semivariogram(lag[ok], *theta, "exponential")
    assert obj(theta) == pytest.approx(float(resid @ resid), rel=1e-12)


def test_analytic_jacobian_matches_the_derivative_it_claims_to_be():
    """(4.42) is written in terms of d gamma / d theta, so those derivatives
    carry the fit. Check them against central differences rather than
    assuming the algebra was transcribed correctly."""
    theta = np.array([0.3, 2.0, 6.0])
    h = np.linspace(0.5, 20.0, 7)
    for model in ("exponential", "gaussian", "spherical"):
        jac = semivariogram_jacobian(h, *theta, model)
        num = np.zeros_like(jac)
        for i in range(3):
            e = np.zeros(3)
            e[i] = 1e-7
            num[:, i] = (semivariogram(h, *(theta + e), model)
                         - semivariogram(h, *(theta - e), model)) / 2e-7
        assert np.abs(jac - num).max() < 1e-6


def test_recovers_the_parameters_the_table_was_built_from(schab_fit_table):
    """The table IS the model plus a 2 per cent deterministic wiggle, so the
    answer is known exactly -- this is a recovery test, not a self-check."""
    res = spols(schab_fit_table, "exponential")
    assert res["nugget"] == pytest.approx(0.3, abs=0.05)
    assert res["partial_sill"] == pytest.approx(2.0, rel=0.05)
    assert res["range"] == pytest.approx(6.0, rel=0.10)
    assert res["converged"]


def test_fit_actually_improves_on_the_starting_values(schab_fit_table):
    """Regression guard for a real defect: an earlier version minimised the
    scalar objective with a quasi-Newton solver, which stopped after one
    iteration and returned the starting heuristic as the fit while reporting
    success. The start sat near the truth, so it looked plausible."""
    lag, gam, cnt = schab_fit_table
    res = spols((lag, gam, cnt), "exponential")
    obj, ok = _objective("ols", lag, gam, cnt, "exponential")
    start, _ = _start_and_bounds(lag[ok], gam[ok])
    got = np.array([res["nugget"], res["partial_sill"], res["range"]])
    assert not np.allclose(got, start)
    assert res["objective"] < obj(start)


def test_parameters_stay_in_the_valid_space(schab_fit_table):
    """A nugget and a partial sill are variances; a range is a distance."""
    res = spols(schab_fit_table, "exponential")
    assert res["nugget"] >= 0.0
    assert res["partial_sill"] >= 0.0
    assert res["range"] > 0.0
    assert res["sill"] == pytest.approx(res["nugget"] + res["partial_sill"])


def test_agrees_with_the_r_arm(schab_fit_table):
    """Both arms run the same Gauss-Newton, so these are pinned to all the
    digits they actually share -- see the R suite for the other side."""
    res = spols(schab_fit_table, "exponential")
    assert res["nugget"] == pytest.approx(0.300720377187438, rel=1e-11)
    assert res["partial_sill"] == pytest.approx(2.00611785482791, rel=1e-11)
    assert res["range"] == pytest.approx(6.0714734193828, rel=1e-11)


def test_accepts_a_mapping_as_well_as_a_tuple(schab_fit_table):
    lag, gam, cnt = schab_fit_table
    a = spols((lag, gam, cnt), "exponential")["range"]
    b = spols({"lags": lag, "gamma": gam, "counts": cnt}, "exponential")["range"]
    assert a == pytest.approx(b, rel=1e-12)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        spols(np.array([[1.0], [2.0]]), "exponential")
    with pytest.raises(ValueError):
        spols((np.array([1.0]), np.array([1.0]), np.array([1.0])), "exponential")
