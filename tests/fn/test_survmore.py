"""The fifteen survival methods restored after de-externalization.

Expected values are computed from the definitions or are exact
properties of the estimators, and the R arm asserts the same numbers on
the same fixtures.
"""

import math

import pytest

from morie.fn.survmore import (aftfit, cif, coxsnell, devresid, finegray,
                               hazratio, landmark, ltkm, martingale,
                               paracompare, parasurv, rmst, rmstdiff,
                               schoenfeld, turnbull)

T = [5, 6, 6, 2.5, 4, 4, 3, 3, 1, 2, 2, 3, 7, 8, 9, 10]
E = [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1]
X = [[float(i % 2)] for i in range(len(T))]
C = [1, 0, 2, 1, 1, 0, 2, 1, 1, 1, 0, 2, 1, 0, 1, 2]


def test_rmst_of_an_uncensored_sample_is_the_stepwise_area():
    # S steps 0.75, 0.5, 0.25, 0; area to tau = 4 is 1 + .75 + .5 + .25
    assert rmst([1, 2, 3, 4], [1, 1, 1, 1], tau=4)["rmst"] == \
        pytest.approx(2.5)


def test_rmst_interval_brackets_the_estimate():
    r = rmst(T, E, tau=8)
    assert r["se"] > 0
    assert r["lower"] < r["rmst"] < r["upper"]


def test_rmst_flags_a_horizon_past_the_data():
    assert rmst(T, E, tau=99)["tau_beyond_data"] is True
    assert rmst(T, E, tau=5)["tau_beyond_data"] is False
    with pytest.raises(ValueError):
        rmst(T, E, tau=0)


def test_rmstdiff_adds_variances_over_a_common_horizon():
    g = [0] * 8 + [1] * 8
    r = rmstdiff(T, E, g)
    a = rmst(T[:8], E[:8], tau=r["tau"])
    b = rmst(T[8:], E[8:], tau=r["tau"])
    assert r["difference"] == pytest.approx(a["rmst"] - b["rmst"])
    assert r["se"] == pytest.approx(
        math.sqrt(a["variance"] + b["variance"]))
    assert rmstdiff(T, E, g, tau=99)["tau_capped"] is True


def test_rmstdiff_needs_exactly_two_groups():
    with pytest.raises(ValueError):
        rmstdiff(T, E, [1] * 16)


def test_martingale_residuals_sum_to_zero():
    m = martingale(T, E, X, [0.3])
    assert m["sum"] == pytest.approx(0.0, abs=1e-8)
    assert m["sums_to_zero"] is True
    assert m["max"] <= 1.0


def test_coxsnell_is_delta_minus_the_martingale():
    m = martingale(T, E, X, [0.3])["residuals"]
    cs = coxsnell(T, E, X, [0.3])["residuals"]
    assert cs == pytest.approx([E[i] - m[i] for i in range(len(T))])
    assert all(v >= 0 for v in cs)


def test_devresid_is_the_symmetrized_martingale():
    d = devresid(T, E, X, [0.3])
    m = d["martingale"]
    for a, b in zip(d["residuals"], m):
        assert (a >= 0) == (b >= 0)
    assert d["is_model_deviance"] is False

    def skew(v):
        n = len(v)
        mu = sum(v) / n
        m2 = sum((x - mu) ** 2 for x in v) / n
        m3 = sum((x - mu) ** 3 for x in v) / n
        return abs(m3) / m2 ** 1.5

    assert skew(d["residuals"]) < skew(m)


def test_schoenfeld_returns_one_residual_per_event_time():
    s = schoenfeld(T, E, X, [0.3], vcov=[[0.25]])
    assert len(s["residuals"]) == len(set(T[i] for i in range(len(T))
                                          if E[i] == 1))
    assert len(s["time"]) == len(s["residuals"])
    assert abs(s["ph_test"][0]["rho"]) <= 1.0


def test_schoenfeld_scaling_needs_the_covariance():
    with pytest.raises(ValueError):
        schoenfeld(T, E, X, [0.3])
    assert "scaled" not in schoenfeld(T, E, X, [0.3], scaled=False)


def test_hazratio_interval_is_asymmetric_and_positive():
    r = hazratio([0.5, -0.2], [0.2, 0.1])
    assert r["hazard_ratio"] == pytest.approx([math.exp(0.5),
                                               math.exp(-0.2)])
    assert all(v > 0 for v in r["lower"])
    lo_gap = r["hazard_ratio"][0] - r["lower"][0]
    hi_gap = r["upper"][0] - r["hazard_ratio"][0]
    assert lo_gap != pytest.approx(hi_gap)
    assert r["interval_on_log_scale"] is True


def test_hazratio_rejects_negative_standard_errors():
    with pytest.raises(ValueError):
        hazratio([0.5], [-0.1])


def test_cif_is_below_the_naive_one_minus_km():
    r = cif(T, C, code=1)
    assert r["cif"][-1] < r["naive_one_minus_km"][-1]
    assert r["naive_overstates_by"] > 0
    assert all(b >= a - 1e-12 for a, b in zip(r["cif"], r["cif"][1:]))
    assert all(0.0 <= v <= 1.0 for v in r["cif"])


def test_cifs_over_all_causes_sum_to_one_minus_survival():
    a = cif(T, C, code=1)
    b = cif(T, C, code=2)
    assert a["cif"][-1] + b["cif"][-1] == pytest.approx(
        1.0 - a["overall_survival_at_end"], abs=1e-9)


def test_cif_rejects_the_censoring_code():
    with pytest.raises(ValueError):
        cif(T, C, code=0)
    with pytest.raises(ValueError):
        cif(T, C, code=9)


def test_finegray_fits_the_subdistribution_hazard():
    r = finegray(T, C, X, code=1)
    assert len(r["coef"]) == 1
    assert r["subdistribution_hazard_ratio"] == pytest.approx(
        [math.exp(r["coef"][0])])
    assert r["n_competing"] > 0
    assert r["differs_from_cause_specific"] is True


def test_ltkm_with_entry_at_zero_reproduces_plain_km():
    r = ltkm([0.0] * len(T), T, E)
    assert r["max_difference"] == pytest.approx(0.0, abs=1e-12)
    assert r["surv"] == pytest.approx(r["ignoring_truncation"])


def test_ltkm_with_real_entry_times_differs():
    en = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2]
    r = ltkm(en, T, E)
    assert r["max_difference"] > 0
    assert all(v <= len(T) for v in r["n_risk"])


def test_ltkm_requires_entry_before_followup():
    with pytest.raises(ValueError):
        ltkm(T, T, E)


def test_landmark_drops_early_subjects_and_resets_the_clock():
    r = landmark(T, E, 3.0)
    assert r["n_retained"] == sum(1 for v in T if v > 3)
    assert r["n_dropped"] == len(T) - r["n_retained"]
    assert r["time"] == pytest.approx([v - 3 for v in T if v > 3])
    assert r["conditional_on_surviving_to_landmark"] is True


def test_landmark_refuses_a_landmark_past_the_follow_up():
    with pytest.raises(ValueError):
        landmark(T, E, 9.5)


def test_turnbull_puts_unit_mass_on_its_intervals():
    r = turnbull([0, 1, 2, 3, 1, 2], [2, 3, 4, float("inf"), 2, 5])
    assert r["converged"] is True
    assert sum(r["mass"]) == pytest.approx(1.0, abs=1e-8)
    assert all(v >= -1e-12 for v in r["mass"])
    assert all(b <= a + 1e-12 for a, b in zip(r["surv"], r["surv"][1:]))


def test_turnbull_on_exact_observations_is_the_empirical_cdf():
    x = [1, 2, 3, 4]
    r = turnbull(x, x)
    assert r["n_intervals"] == 4
    assert r["mass"] == pytest.approx([0.25] * 4, abs=1e-6)


def test_parasurv_exponential_is_nested_in_weibull():
    w = parasurv(T, E, "weibull")
    ex = parasurv(T, E, "exponential")
    assert w["loglik"] >= ex["loglik"] - 1e-6
    assert w["lr_vs_exponential"] == pytest.approx(
        2.0 * (w["loglik"] - ex["loglik"]), abs=1e-6)
    assert ex["fixed_scale"] is True
    assert w["fixed_scale"] is False


def test_parasurv_rejects_an_unknown_family():
    with pytest.raises(ValueError):
        parasurv(T, E, "gompertz")


def test_aft_time_ratio_and_hazard_ratio_point_opposite_ways():
    a = aftfit(T, E, X, "weibull")
    assert a["time_ratio"] == pytest.approx(
        [math.exp(v) for v in a["beta"]])
    assert a["ph_equivalent"] is True
    assert a["hazard_ratio"] == pytest.approx(
        [math.exp(-v / a["scale"]) for v in a["beta"]])
    assert (a["time_ratio"][0] > 1) == (a["hazard_ratio"][0] < 1)


def test_aft_lognormal_has_no_ph_equivalent():
    ln = aftfit(T, E, X, "lognormal")
    assert ln["ph_equivalent"] is False
    assert "hazard_ratio" not in ln


def test_paracompare_ranks_by_aic_and_reports_failures():
    r = paracompare(T, E)
    assert len(r["table"]) == 4
    assert [row["aic"] for row in r["table"]] == sorted(
        row["aic"] for row in r["table"])
    assert r["best_aic"] in [row["dist"] for row in r["table"]]
    assert r["families_not_nested"] is True
    assert r["failed"] == {}
    assert "lr_weibull_vs_exponential" in r


def test_paracompare_keeps_a_failing_family_visible():
    r = paracompare(T, E, dists=["weibull", "gompertz"])
    assert "gompertz" in r["failed"]
    assert r["best_aic"] == "weibull"
