"""Tests for spwls: Cressie WLS semivariogram fitting (Schabenberger Sec 4.5.1).

Minimises eq (4.34), the generalized sum of squares (4.31) with R(theta)
replaced by the diagonal W(theta) from Cressie's (1985) approximation (4.33).
The fit is by Gauss-Newton -- the text names it for the composite-likelihood
form too (after 4.44) -- with the weights refreshed every outer iteration,
because Sec 4.5.1 requires that updates to theta be followed by updates to
R(theta).
"""

from morie.fn import _array_core as np
import pytest

from morie.fn._schab_fit import _objective, _start_and_bounds
from morie.fn._schab_vario import semivariogram
from morie.fn.spols import schabenberger_ols_variogram as spols
from morie.fn.spwls import schabenberger_wls_variogram as spwls


def test_wls_objective_is_equation_4_34(schab_fit_table):
    """eq (4.34) is sum |N| / (2 gamma^2) * resid^2. Dividing through gives
    the familiar (1/2) sum |N| [gamma_hat/gamma - 1]^2; assert the identity
    rather than assuming the two forms were transcribed consistently."""
    lag, gam, cnt = schab_fit_table
    obj, ok = _objective("wls", lag, gam, cnt, "exponential")
    theta = np.array([0.3, 2.0, 6.0])
    fitted = semivariogram(lag[ok], *theta, "exponential")
    alt = 0.5 * np.sum(cnt[ok] * (gam[ok] / fitted - 1.0) ** 2)
    assert obj(theta) == pytest.approx(alt, rel=1e-12)


def test_recovers_the_parameters_the_table_was_built_from(schab_fit_table):
    res = spwls(schab_fit_table, "exponential")
    assert res["nugget"] == pytest.approx(0.3, abs=0.05)
    assert res["partial_sill"] == pytest.approx(2.0, rel=0.05)
    assert res["range"] == pytest.approx(6.0, rel=0.10)
    assert res["converged"]


def test_fit_actually_improves_on_the_starting_values(schab_fit_table):
    """Regression guard: this family once stopped after one iteration and
    reported the starting heuristic as the fit, with success=True."""
    lag, gam, cnt = schab_fit_table
    res = spwls((lag, gam, cnt), "exponential")
    obj, ok = _objective("wls", lag, gam, cnt, "exponential")
    start, _ = _start_and_bounds(lag[ok], gam[ok])
    got = np.array([res["nugget"], res["partial_sill"], res["range"]])
    assert not np.allclose(got, start)
    assert res["objective"] < obj(start)


def test_counts_reach_wls_and_do_not_leak_into_ols(schab_fit_table):
    """|N(h_m)| appears in (4.34) and nowhere in the OLS criterion. If
    reweighting the lag classes moved the OLS answer the weights would be
    leaking; if it left WLS unmoved they would not be arriving."""
    lag, gam, cnt = schab_fit_table
    skew = cnt.copy()
    skew[: len(skew) // 2] *= 40.0
    assert (spwls((lag, gam, cnt), "exponential")["range"]
            != spwls((lag, gam, skew), "exponential")["range"])
    assert spols((lag, gam, cnt), "exponential")["range"] == \
        pytest.approx(spols((lag, gam, skew), "exponential")["range"], rel=1e-9)


def test_weights_are_refreshed_not_frozen(schab_fit_table):
    """Sec 4.5.1: updates to theta must be followed by updates to R(theta).
    A frozen-weight fit is a different estimator, so the converged answer must
    satisfy (4.34) evaluated with ITS OWN weights -- which is what the
    reported objective is."""
    lag, gam, cnt = schab_fit_table
    res = spwls((lag, gam, cnt), "exponential")
    obj, _ = _objective("wls", lag, gam, cnt, "exponential")
    theta = np.array([res["nugget"], res["partial_sill"], res["range"]])
    assert res["objective"] == pytest.approx(obj(theta), rel=1e-10)


def test_agrees_with_the_r_arm(schab_fit_table):
    res = spwls(schab_fit_table, "exponential")
    assert res["nugget"] == pytest.approx(0.292391714297037, rel=1e-11)
    assert res["partial_sill"] == pytest.approx(2.00846165741921, rel=1e-11)
    assert res["range"] == pytest.approx(5.98562346376273, rel=1e-11)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        spwls((np.array([1.0]), np.array([1.0]), np.array([1.0])), "exponential")
