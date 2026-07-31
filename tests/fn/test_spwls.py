"""Tests for spwls: Cressie WLS semivariogram fitting (Schabenberger Sec 4.5.1).

Minimises eq (4.34), the generalized sum of squares (4.31) with R(theta)
replaced by the diagonal W(theta) built from Cressie's (1985) approximation
(4.33). The weights are functions of theta, so the fit is re-weighted rather
than fixed-weight.
"""

import numpy as np
import pytest

from morie.fn._schab_fit import _objective, _start_and_bounds
from morie.fn._schab_vario import semivariogram
from morie.fn.spols import schabenberger_ols_variogram as spols
from morie.fn.spwls import schabenberger_wls_variogram as spwls


def test_wls_objective_is_equation_4_34(schab_sites, schab_simulate, schab_ev):
    """eq (4.34) is sum |N| / (2 gamma^2) * resid^2. Dividing through gives
    the familiar (1/2) sum |N| [gamma_hat/gamma - 1]^2; assert the identity
    rather than assuming the two forms were transcribed consistently."""
    coords = schab_sites()
    lag, gam, cnt = schab_ev(coords, schab_simulate(coords, 7))
    obj, ok = _objective("wls", lag, gam, cnt, "exponential")
    theta = np.array([0.3, 2.0, 6.0])
    fitted = semivariogram(lag[ok], *theta, "exponential")
    alt = 0.5 * np.sum(cnt[ok] * (gam[ok] / fitted - 1.0) ** 2)
    assert obj(theta) == pytest.approx(alt, rel=1e-12)


def test_fit_actually_improves_on_the_starting_values(schab_sites, schab_simulate, schab_ev):
    """Regression guard: this family once stopped after one iteration and
    reported the starting heuristic as the fit, with success=True."""
    coords = schab_sites()
    lag, gam, cnt = schab_ev(coords, schab_simulate(coords, 7))
    res = spwls((lag, gam, cnt), "exponential")
    obj, ok = _objective("wls", lag, gam, cnt, "exponential")
    start, _ = _start_and_bounds(lag[ok], gam[ok])
    got = np.array([res["nugget"], res["partial_sill"], res["range"]])
    assert not np.allclose(got, start)
    assert res["objective"] < obj(start)


def test_recovers_the_generating_parameters_in_median(schab_sites, schab_simulate, schab_ev):
    coords = schab_sites()
    est = []
    for s in range(12):
        r = spwls(schab_ev(coords, schab_simulate(coords, 100 + s)), "exponential")
        est.append([r["nugget"], r["partial_sill"], r["range"]])
    med = np.median(np.array(est), axis=0)
    assert med[0] == pytest.approx(0.3, abs=0.25)
    assert med[1] == pytest.approx(2.0, rel=0.35)
    assert med[2] == pytest.approx(6.0, rel=0.45)


def test_counts_reach_wls_and_do_not_leak_into_ols(schab_sites, schab_simulate, schab_ev):
    """|N(h_m)| appears in (4.34) and nowhere in the OLS criterion. If
    reweighting the lag classes moved the OLS answer the weights would be
    leaking; if it left WLS unmoved they would not be arriving."""
    coords = schab_sites()
    lag, gam, cnt = schab_ev(coords, schab_simulate(coords, 7))
    skew = cnt.astype(float).copy()
    skew[: len(skew) // 2] *= 40.0
    assert (spwls((lag, gam, cnt), "exponential")["range"]
            != spwls((lag, gam, skew), "exponential")["range"])
    assert spols((lag, gam, cnt), "exponential")["range"] == \
        pytest.approx(spols((lag, gam, skew), "exponential")["range"], rel=1e-9)


def test_pure_nugget_field_can_report_no_spatial_structure(schab_sites, schab_ev):
    """Can the estimator return the null? White noise has none, so the
    partial sill must be able to collapse -- a fit that cannot say "no
    correlation" will always find some."""
    coords = schab_sites(n=200, seed=3)
    z = np.random.default_rng(21).normal(size=coords.shape[0])
    res = spwls(schab_ev(coords, z), "exponential")
    assert res["partial_sill"] < 0.5 * res["sill"]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        spwls((np.array([1.0]), np.array([1.0]), np.array([1.0])), "exponential")
