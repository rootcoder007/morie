"""Linear regression with real inference, checked against R.

Anchors come from R itself -- lm/summary.lm, sandwich::vcovHC,
sandwich::NeweyWest, lmtest::bptest, lmtest::dwtest, car::vif -- run on
the same 40-observation fixture, which is deliberately heteroskedastic
so the robust machinery has something to do.
"""
import math

import pytest

from morie.fn import _regression_core as rg

N = 40
X1 = [((i * 7) % 13) + 0.5 * ((i * 3) % 5) for i in range(N)]
X2 = [((i * 5) % 11) - 0.25 * ((i * 2) % 7) for i in range(N)]
E = [((i * 13) % 17) / 17 - 0.5 for i in range(N)]
Y = [3 + 1.5 * X1[i] - 0.8 * X2[i] + E[i] * (1 + 0.05 * X1[i])
     for i in range(N)]
X = [[X1[i], X2[i]] for i in range(N)]
FIT = rg.ols(Y, X)


def test_ols_coefficients_match_r_lm():
    for got, want in zip(FIT["coef"], (2.83062921605554, 1.50887378011902,
                                       -0.786477982373891)):
        assert abs(got - want) < 1e-12


def test_ols_standard_errors_and_tests_match_summary_lm():
    for got, want in zip(FIT["se"], (0.211842232683341, 0.0190862901909392,
                                     0.0230017208671273)):
        assert abs(got - want) < 1e-14
    for got, want in zip(FIT["t"], (13.3619683865716, 79.0553724701995,
                                    -34.192136619564)):
        assert abs(got - want) < 1e-11
    for got, want in zip(FIT["p_value"], (9.88338046119056e-16,
                                          7.19392533819999e-43,
                                          1.34105014318614e-29)):
        assert abs(got - want) / want < 1e-11


def test_ols_summary_statistics_match_r():
    assert abs(FIT["r_squared"] - 0.997397231487757) < 1e-14
    assert abs(FIT["adj_r_squared"] - 0.997256541297906) < 1e-14
    assert abs(FIT["sigma"] - 0.396084329507282) < 1e-14
    assert abs(FIT["f_statistic"] - 7089.31612463033) < 1e-8


def test_f_p_value_does_not_underflow():
    # computed as the upper tail directly; as 1 - lower it collapses to
    # exactly 0 for a model this strong, throwing the p-value away
    assert abs(FIT["f_p_value"] - 1.53305439891331e-48) / \
        1.53305439891331e-48 < 1e-10
    assert FIT["f_p_value"] > 0


def test_robust_standard_errors_match_sandwich_vcovhc():
    want = {
        "HC0": (0.191679800355021, 0.0172033171162341, 0.0212173733280863),
        "HC1": (0.199299166307748, 0.0178871573981345, 0.0220607743105732),
        "HC2": (0.21338832217753, 0.0190716967679078, 0.0234116174677771),
        "HC3": (0.239053563666654, 0.0212641516186098, 0.0259807964174029),
    }
    for kind, target in want.items():
        got = rg.robust_vcov(FIT, kind)["se"]
        for g, w in zip(got, target):
            assert abs(g - w) < 1e-14, kind


def test_robust_se_ordering_hc0_to_hc3():
    # HC3 penalises high-leverage points most, so it is the widest
    ses = [rg.robust_vcov(FIT, k)["se"][1]
           for k in ("HC0", "HC1", "HC2", "HC3")]
    assert ses[0] < ses[1] < ses[2] < ses[3]


def test_robust_se_differs_from_classical_under_heteroskedasticity():
    # the fixture's error variance grows with x1, so the classical SE
    # is not merely a rounding away from the robust one
    assert abs(rg.robust_vcov(FIT, "HC3")["se"][1]
               - FIT["se"][1]) > 1e-3


def test_newey_west_matches_sandwich_with_the_same_lag():
    # R's NeweyWest picks its bandwidth automatically; pin the lag and
    # the two agree exactly (R default adjust = TRUE, which we apply)
    got = rg.newey_west_vcov(FIT, lags=3)["se"]
    for g, w in zip(got, (0.190394553869704, 0.0167770191608669,
                          0.0214790245326673)):
        assert abs(g - w) < 1e-13


def test_newey_west_default_lag_rule():
    # floor(4 (n/100)^(2/9)); Newey and West's own rule, as statsmodels
    assert rg.newey_west_vcov(FIT)["lags"] == \
        int(math.floor(4.0 * (N / 100.0) ** (2.0 / 9.0)))


def test_newey_west_with_zero_lags_reduces_to_hc0_style_meat():
    nw = rg.newey_west_vcov(FIT, lags=0)["se"]
    hc1 = rg.robust_vcov(FIT, "HC1")["se"]
    for a, b in zip(nw, hc1):
        assert abs(a - b) < 1e-12


def test_breusch_pagan_matches_lmtest_bptest():
    bp = rg.breusch_pagan(FIT)
    assert abs(bp["statistic"] - 0.792751135858652) < 1e-11
    assert bp["df"] == 2
    assert abs(bp["p_value"] - 0.672753983664454) < 1e-12


def test_durbin_watson_matches_lmtest_dwtest():
    dw = rg.durbin_watson(FIT)
    assert abs(dw["statistic"] - 2.09214798929709) < 1e-13
    # DW is about 2(1 - rho)
    assert abs(dw["statistic"] - 2 * (1 - dw["rho"])) < 0.15


def test_vif_matches_car_vif():
    v = rg.variance_inflation_factors(X)["vif"]
    for got in v:
        assert abs(got - 1.38183972779563) < 1e-12
    # two predictors -> both VIFs are equal by construction
    assert abs(v[0] - v[1]) < 1e-15


def test_vif_rises_with_collinearity():
    z = [[X1[i], X1[i] + 1e-3 * X2[i]] for i in range(N)]
    assert rg.variance_inflation_factors(z)["vif"][0] > 100


def test_ols_rejects_collinear_and_underdetermined_designs():
    dup = [[X1[i], X1[i]] for i in range(N)]
    with pytest.raises(ValueError):
        rg.ols(Y, dup)
    with pytest.raises(ValueError):
        rg.ols(Y[:2], [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        rg.ols(Y[:5], X)


def test_residuals_are_orthogonal_to_the_design():
    for j in range(FIT["k"]):
        s = sum(FIT["design"][i][j] * FIT["residuals"][i]
                for i in range(N))
        assert abs(s) < 1e-9


def test_r_squared_identity():
    assert abs(FIT["tss"] - (FIT["rss"] + (FIT["tss"] - FIT["rss"]))) < 1e-9
    assert abs(FIT["r_squared"]
               - (1 - FIT["rss"] / FIT["tss"])) < 1e-15
