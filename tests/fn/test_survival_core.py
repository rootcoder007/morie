"""Survival analysis, anchored on R's survival package.

Values come from survfit, survdiff, coxph and concordance run on the
same 40-subject fixture with censoring in both groups.
"""
import math

import pytest

from morie.fn import _survival_core as sv

T = [5, 8, 12, 3, 15, 7, 20, 11, 4, 18, 9, 22, 6, 14, 25, 10, 17, 2,
     13, 19, 16, 21, 1, 24, 23, 27, 26, 30, 28, 29, 31, 33, 32, 35,
     34, 37, 36, 39, 38, 40]
E = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1,
     1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
G = [1] * 20 + [2] * 20
X1 = [((i * 7) % 11) / 5 - 1 for i in range(40)]
X2 = [((i * 5) % 7) / 3 - 1 for i in range(40)]
X = [[X1[i], X2[i]] for i in range(40)]


def _at(km, t):
    return dict(zip(km["time"], range(len(km["time"]))))[t]


def test_kaplan_meier_matches_survfit():
    km = sv.kaplan_meier(T, E)
    assert len(km["time"]) == 30            # distinct event times
    assert abs(km["surv"][0] - 0.975) < 1e-15
    i = _at(km, 10)
    assert abs(km["surv"][i] - 0.774193548387097) < 1e-14
    assert abs(km["lower"][i] - 0.610837039143546) < 1e-13
    assert abs(km["upper"][i] - 0.87556658555096) < 1e-13


def test_kaplan_meier_std_err_is_the_cumulative_hazard_one():
    # survival::survfit's `std.err` is the error of the CUMULATIVE
    # HAZARD, sqrt(Greenwood sum); the error of S(t) carries a further
    # factor of S.  Both are returned, and they differ by exactly that.
    km = sv.kaplan_meier(T, E)
    assert abs(km["se_cumhaz"][0] - 0.0253184841770917) < 1e-15
    assert abs(km["se_cumhaz"][4] - 0.0597614304667197) < 1e-15
    for j in range(len(km["time"])):
        assert abs(km["se"][j]
                   - km["surv"][j] * km["se_cumhaz"][j]) < 1e-15


def test_kaplan_meier_is_monotone_and_bounded():
    km = sv.kaplan_meier(T, E)
    assert all(a >= b - 1e-15 for a, b in zip(km["surv"], km["surv"][1:]))
    assert all(0.0 <= s <= 1.0 for s in km["surv"])
    # the log-log interval cannot leave [0, 1], which is why it is used
    assert all(0.0 <= l <= 1.0 for l in km["lower"])
    assert all(0.0 <= u <= 1.0 for u in km["upper"])


def test_censoring_is_not_treated_as_an_event():
    # marking every censored subject as an event must lower survival
    km = sv.kaplan_meier(T, E)
    allev = sv.kaplan_meier(T, [1] * len(T))
    assert allev["surv"][-1] <= km["surv"][-1]
    assert km["n_events"] == sum(E)


def test_nelson_aalen_matches_survfit_fh():
    na = sv.nelson_aalen(T, E)
    assert abs(na["surv"][0] - 0.975309912028333) < 1e-14
    assert abs(na["surv"][4] - 0.876549922217358) < 1e-14
    # exp(-H) exceeds the product-limit estimate, as it must
    km = sv.kaplan_meier(T, E)
    assert all(a >= b - 1e-12 for a, b in zip(na["surv"], km["surv"]))


def test_logrank_matches_survdiff():
    r = sv.logrank_test(T, E, G)
    assert abs(r["statistic"] - 20.5279554955505) < 1e-11
    assert r["df"] == 1
    assert abs(r["p_value"] - 5.87666766058383e-06) / \
        5.87666766058383e-06 < 1e-10


def test_logrank_observed_and_expected_totals_agree():
    r = sv.logrank_test(T, E, G)
    assert abs(sum(r["observed"]) - sum(r["expected"])) < 1e-9
    assert abs(sum(r["observed"]) - sum(E)) < 1e-9


def test_logrank_is_null_when_groups_are_identical():
    g = [1 if i % 2 == 0 else 2 for i in range(40)]
    # interleaved assignment: no systematic difference to detect
    r = sv.logrank_test(T, E, g)
    assert r["p_value"] > 0.05


def test_cox_matches_coxph_efron():
    c = sv.cox_ph(T, E, X)
    assert abs(c["coef"][0] - (-0.427673034112693)) < 1e-7
    assert abs(c["coef"][1] - (-0.157489759400814)) < 1e-7
    assert abs(c["se"][0] - 0.304717144602484) < 1e-9
    assert abs(c["se"][1] - 0.285683063051433) < 1e-9
    assert abs(c["loglik"] - (-81.1317187602372)) < 1e-10
    assert abs(c["loglik_null"] - (-82.4119335691052)) < 1e-10


def test_cox_hazard_ratio_is_exp_of_the_coefficient():
    c = sv.cox_ph(T, E, X)
    for b, hr in zip(c["coef"], c["hazard_ratio"]):
        assert abs(hr - math.exp(b)) < 1e-14


def test_cox_likelihood_ratio_is_positive_and_consistent():
    c = sv.cox_ph(T, E, X)
    assert c["loglik"] >= c["loglik_null"]
    assert abs(c["lr_statistic"]
               - 2 * (c["loglik"] - c["loglik_null"])) < 1e-12


def test_cox_partial_loglik_at_zero_is_the_null_loglik():
    c = sv.cox_ph(T, E, X)
    ll0 = sv.cox_partial_loglik(T, E, X, [0.0, 0.0])
    assert abs(ll0 - c["loglik_null"]) < 1e-12


def test_efron_and_breslow_differ_only_with_ties():
    # this fixture has no tied event times, so the two coincide
    a = sv.cox_ph(T, E, X, ties="efron")["coef"]
    b = sv.cox_ph(T, E, X, ties="breslow")["coef"]
    for u, v in zip(a, b):
        assert abs(u - v) < 1e-8
    # introduce ties and they must part company
    tied = [min(t, 10) for t in T]
    a2 = sv.cox_ph(tied, E, X, ties="efron")["coef"]
    b2 = sv.cox_ph(tied, E, X, ties="breslow")["coef"]
    assert max(abs(u - v) for u, v in zip(a2, b2)) > 1e-6


def test_concordance_matches_survival_concordance():
    c = sv.cox_ph(T, E, X)
    risk = [math.exp(c["coef"][0] * X1[i] + c["coef"][1] * X2[i])
            for i in range(40)]
    assert abs(sv.concordance_index(T, E, risk)["c_index"]
               - 0.483443708609272) < 1e-12


def test_concordance_is_one_for_a_perfect_predictor():
    # risk exactly reversing the event order discriminates perfectly
    risk = [-t for t in T]
    assert abs(sv.concordance_index(T, E, risk)["c_index"] - 1.0) < 1e-12
    # and flipping it gives 0
    assert abs(sv.concordance_index(T, E, T)["c_index"]) < 1e-12


def test_survival_inputs_are_validated():
    with pytest.raises(ValueError):
        sv.kaplan_meier(T, E[:5])
    with pytest.raises(ValueError):
        sv.kaplan_meier(T, [2] * len(T))
    with pytest.raises(ValueError):
        sv.logrank_test(T, E, [1] * len(T))
    with pytest.raises(ValueError):
        sv.cox_ph(T, [0] * len(T), X)
