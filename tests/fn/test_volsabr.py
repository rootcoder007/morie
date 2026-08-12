"""Tests for volsabr (SABR implied volatility, Hagan et al. 2002).

Replaces the generated stub, which imported ``vol_sabr_implied``.
"""

import math

from morie.fn.volsabr import volsabr


def test_the_at_the_money_formula_at_beta_one():
    # at K = f and beta = 1 the leading term is alpha, with the usual
    # (1 + [...] T) correction
    f, T, alpha, rho, nu = 100.0, 1.0, 0.2, -0.3, 0.4
    res = volsabr(f, f, T, alpha, 1.0, rho, nu)
    corr = 1.0 + (0.25 * rho * nu * alpha +
                  (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2) * T
    assert abs(res["estimate"] - alpha * corr) < 1e-9
    assert abs(res["atm"] - res["estimate"]) < 1e-12


def test_zero_vol_of_vol_gives_the_flat_smile_at_beta_one():
    f, T, alpha = 100.0, 0.5, 0.25
    a = volsabr(80.0, f, T, alpha, 1.0, 0.0, 0.0)["estimate"]
    b = volsabr(120.0, f, T, alpha, 1.0, 0.0, 0.0)["estimate"]
    assert abs(a - alpha) < 1e-9
    assert abs(b - alpha) < 1e-9


def test_z_vanishes_at_the_money():
    res = volsabr(100.0, 100.0, 1.0, 0.2, 0.5, -0.2, 0.3)
    assert abs(res["z"]) < 1e-12


def test_a_negative_correlation_tilts_the_smile_downward_in_strike():
    f, T, alpha, beta, nu = 100.0, 1.0, 0.2, 1.0, 0.5
    low = volsabr(80.0, f, T, alpha, beta, -0.5, nu)["estimate"]
    high = volsabr(120.0, f, T, alpha, beta, -0.5, nu)["estimate"]
    assert low > high
    low_p = volsabr(80.0, f, T, alpha, beta, 0.5, nu)["estimate"]
    high_p = volsabr(120.0, f, T, alpha, beta, 0.5, nu)["estimate"]
    assert high_p > low_p


def test_more_vol_of_vol_makes_a_deeper_smile():
    f, T, alpha, beta, rho = 100.0, 1.0, 0.2, 1.0, 0.0
    wings_flat = volsabr(70.0, f, T, alpha, beta, rho, 0.05)["estimate"]
    wings_curved = volsabr(70.0, f, T, alpha, beta, rho, 0.8)["estimate"]
    assert wings_curved > wings_flat


def test_the_volatility_is_positive_across_a_range_of_strikes():
    for k in (50.0, 80.0, 100.0, 130.0, 200.0):
        v = volsabr(k, 100.0, 2.0, 0.3, 0.7, -0.4, 0.6)["estimate"]
        assert v > 0 and v == v


def test_validation():
    for call in (lambda: volsabr(0.0, 100.0, 1.0, 0.2, 1.0, 0.0, 0.3),
                 lambda: volsabr(100.0, 0.0, 1.0, 0.2, 1.0, 0.0, 0.3),
                 lambda: volsabr(100.0, 100.0, -1.0, 0.2, 1.0, 0.0, 0.3),
                 lambda: volsabr(100.0, 100.0, 1.0, 0.0, 1.0, 0.0, 0.3),
                 lambda: volsabr(100.0, 100.0, 1.0, 0.2, 1.5, 0.0, 0.3),
                 lambda: volsabr(100.0, 100.0, 1.0, 0.2, 1.0, 1.5, 0.3),
                 lambda: volsabr(100.0, 100.0, 1.0, 0.2, 1.0, 0.0, -0.1)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
