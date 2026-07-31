"""Tests for spols: OLS semivariogram fitting (Schabenberger & Gotway Sec 4.5.1).

OLS is eq (4.31) under the simplification R = phi*I named just after (4.34):
it ignores both the correlation among the gamma-hat(h_m) and their unequal
dispersion.
"""

import numpy as np
import pytest

from morie.fn._schab_fit import _objective, _start_and_bounds
from morie.fn._schab_vario import semivariogram
from morie.fn.spols import schabenberger_ols_variogram as spols


def test_ols_objective_is_the_plain_residual_sum_of_squares(schab_sites, schab_simulate, schab_ev):
    coords = schab_sites()
    lag, gam, cnt = schab_ev(coords, schab_simulate(coords, 7))
    obj, ok = _objective("ols", lag, gam, cnt, "exponential")
    theta = np.array([0.3, 2.0, 6.0])
    resid = gam[ok] - semivariogram(lag[ok], *theta, "exponential")
    assert obj(theta) == pytest.approx(float(resid @ resid), rel=1e-12)


def test_fit_actually_improves_on_the_starting_values(schab_sites, schab_simulate, schab_ev):
    """Regression guard for a real defect: with a scalar objective and a
    quasi-Newton solver this family stopped after one iteration and returned
    the starting heuristic as the fit, reporting success. The start sat near
    the truth, so the numbers looked plausible."""
    coords = schab_sites()
    lag, gam, cnt = schab_ev(coords, schab_simulate(coords, 7))
    res = spols((lag, gam, cnt), "exponential")
    obj, ok = _objective("ols", lag, gam, cnt, "exponential")
    start, _ = _start_and_bounds(lag[ok], gam[ok])
    got = np.array([res["nugget"], res["partial_sill"], res["range"]])
    assert not np.allclose(got, start)
    assert res["objective"] < obj(start)


def test_recovers_the_generating_parameters_in_median(schab_sites, schab_simulate, schab_ev):
    """One realization is a poor guide -- the range especially, and on some
    draws the fitted model genuinely beats the truth on the objective. The
    claim being made is about the estimator, so judge it over replicates."""
    coords = schab_sites()
    est = []
    for s in range(12):
        r = spols(schab_ev(coords, schab_simulate(coords, 100 + s)), "exponential")
        est.append([r["nugget"], r["partial_sill"], r["range"]])
    med = np.median(np.array(est), axis=0)
    assert med[0] == pytest.approx(0.3, abs=0.3)
    assert med[1] == pytest.approx(2.0, rel=0.4)
    assert med[2] == pytest.approx(6.0, rel=0.6)


def test_parameters_stay_in_the_valid_space(schab_sites, schab_simulate, schab_ev):
    """A nugget and a partial sill are variances; a range is a distance."""
    coords = schab_sites()
    r = spols(schab_ev(coords, schab_simulate(coords, 7)), "exponential")
    assert r["nugget"] >= 0.0
    assert r["partial_sill"] >= 0.0
    assert r["range"] > 0.0
    assert r["sill"] == pytest.approx(r["nugget"] + r["partial_sill"])


def test_accepts_a_mapping_as_well_as_a_tuple(schab_sites, schab_simulate, schab_ev):
    coords = schab_sites()
    lag, gam, cnt = schab_ev(coords, schab_simulate(coords, 7))
    a = spols((lag, gam, cnt), "exponential")["range"]
    b = spols({"lags": lag, "gamma": gam, "counts": cnt}, "exponential")["range"]
    assert a == pytest.approx(b, rel=1e-12)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        spols(np.array([[1.0], [2.0]]), "exponential")
    with pytest.raises(ValueError):
        spols((np.array([1.0]), np.array([1.0]), np.array([1.0])), "exponential")
