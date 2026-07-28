"""The staggered-DiD shelf, tested against the properties that define it.

Every estimator here exists because two-way fixed effects fails under
heterogeneous treatment timing. The tests are built around that: a
design with staggered adoption and dynamic effects has a KNOWN average
effect, TWFE misses it, and each replacement recovers it.
"""

import numpy as np
import pytest

from morie.fn.avtdid import avg_treatment_did
from morie.fn.boryis import borusyak_jaravel_spiess
from morie.fn.cssant import callaway_santanna
from morie.fn.didtwfe import twoway_fixed_effects_did
from morie.fn.drcsa import dr_callaway_santanna
from morie.fn.drsza import dr_did_santanna_zhao
from morie.fn.gbacon import goodman_bacon_decomp
from morie.fn.sdiff import synthetic_did
from morie.fn.synct import synthetic_control
from morie.fn.wbcide import wooldridge_bjs_estimator


def staggered(n_per=3, T=8, gs=(3.0, 5.0), dynamic=0.5, base=1.0, noise=0.0,
              seed=0):
    """Panel with staggered adoption and (optionally) growing effects.

    The true average effect over treated cells is returned, computed
    from the design rather than from any estimator.
    """
    gv = np.concatenate([np.full(n_per, g) for g in gs]
                        + [np.full(n_per, np.inf)])
    n_u = gv.size
    unit = np.repeat(np.arange(n_u), T)
    time = np.tile(np.arange(T), n_u)
    g = np.repeat(gv, T)
    D = (time >= g).astype(float)
    rel = np.where(np.isfinite(g), time - g, -1.0)
    tau = np.where(D > 0, base + dynamic * rel, 0.0)
    y = unit * 0.3 + time * 0.2 + tau
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise, y.size)
    truth = float(tau[D > 0].mean())
    return y, D, unit, time, gv, truth


def test_twfe_is_exact_without_timing_variation():
    y, D, unit, time, _, truth = staggered(gs=(4.0, 4.0), dynamic=0.0)
    out = twoway_fixed_effects_did(y, D, unit, time)
    assert out["estimate"] == pytest.approx(truth, abs=1e-10)
    assert out["timing_varies"] is False
    assert out["trustworthy"] is True
    assert "single adoption date" in out["diagnosis"]


def test_twfe_misses_the_truth_under_staggered_dynamic_effects():
    y, D, unit, time, _, truth = staggered()
    twfe = twoway_fixed_effects_did(y, D, unit, time)
    assert truth == pytest.approx(1.8125)
    # the failure the whole shelf exists for
    assert abs(twfe["estimate"] - truth) > 0.3
    assert twfe["estimate"] < truth
    assert twfe["timing_varies"] is True
    assert twfe["trustworthy"] is False
    assert twfe["already_treated_share"] > 0


def test_twfe_refuses_designs_it_cannot_describe():
    y, D, unit, time, _, _ = staggered()
    with pytest.raises(ValueError, match="absorbing"):
        twoway_fixed_effects_did(y, 1 - D, unit, time)
    with pytest.raises(ValueError, match="unbalanced"):
        twoway_fixed_effects_did(y[:-1], D[:-1], unit[:-1], time[:-1])
    with pytest.raises(ValueError, match="binary"):
        twoway_fixed_effects_did(y, D * 2, unit, time)
    with pytest.raises(ValueError, match="no within-panel variation"):
        twoway_fixed_effects_did(y, np.ones_like(D), unit, time)


def test_goodman_bacon_identity_holds_exactly():
    y, D, unit, time, _, _ = staggered(noise=0.05, seed=3)
    out = goodman_bacon_decomp(y, D, unit, time)
    twfe = twoway_fixed_effects_did(y, D, unit, time)["estimate"]
    assert out["estimate"] == pytest.approx(twfe, rel=1e-12)
    assert out["weight_sum"] == pytest.approx(1.0, abs=1e-10)
    assert out["identity_residual"] == pytest.approx(0.0, abs=1e-10)
    assert out["recomposed"] == pytest.approx(twfe, abs=1e-10)


def test_goodman_bacon_isolates_the_forbidden_comparison():
    y, D, unit, time, _, truth = staggered()
    out = goodman_bacon_decomp(y, D, unit, time)
    forbidden = [c for c in out["components"] if c["forbidden"]]
    clean = [c for c in out["components"] if not c["forbidden"]]
    assert len(forbidden) == 1 and len(clean) == 3
    # equal cohorts of 3 adopting at t = 3 and t = 5 of 8 put exactly a
    # seventh of the TWFE coefficient on the late-vs-early comparison
    assert out["forbidden_weight"] == pytest.approx(1 / 7)
    # the forbidden comparison differences out the early cohort's own
    # growing effect, so it understates the late cohort's effect
    assert forbidden[0]["beta"] < min(c["beta"] for c in clean)
    # and that is what drags the TWFE coefficient below the truth
    assert out["estimate"] < truth


def test_goodman_bacon_weights_depend_only_on_timing():
    a = staggered(dynamic=0.0, base=1.0)
    b = staggered(dynamic=0.9, base=4.0)
    wa = goodman_bacon_decomp(a[0], a[1], a[2], a[3])["weights"]
    wb = goodman_bacon_decomp(b[0], b[1], b[2], b[3])["weights"]
    assert np.allclose(wa, wb)


def test_callaway_santanna_recovers_the_truth_cell_by_cell():
    y, D, unit, time, gv, truth = staggered()
    out = callaway_santanna(y, D, unit, time)
    assert out["estimate"] == pytest.approx(truth, abs=1e-10)
    assert out["pretrend_max_abs"] == pytest.approx(0.0, abs=1e-10)
    # the event study is the design's own dynamic path
    for rel, v in out["event"].items():
        assert v["att"] == pytest.approx(1.0 + 0.5 * rel, abs=1e-10)
    # each cohort's own average
    assert out["cohort_att"][3.0]["att"] == pytest.approx(2.0, abs=1e-10)
    assert out["cohort_att"][5.0]["att"] == pytest.approx(1.5, abs=1e-10)


def test_callaway_santanna_control_group_choice_is_reported():
    y, D, unit, time, _, truth = staggered()
    ny = callaway_santanna(y, D, unit, time, control="notyet")
    nv = callaway_santanna(y, D, unit, time, control="never")
    assert ny["estimate"] == pytest.approx(truth, abs=1e-10)
    assert nv["estimate"] == pytest.approx(truth, abs=1e-10)
    assert ny["control_group"] == "notyet"
    assert nv["n_cells"] == ny["n_cells"]
    with pytest.raises(ValueError, match="'notyet' or 'never'"):
        callaway_santanna(y, D, unit, time, control="whatever")


def test_callaway_santanna_needs_never_treated_for_that_option():
    y, D, unit, time, _, _ = staggered(gs=(3.0, 5.0, 7.0))
    y2, D2, u2, t2, _, _ = staggered(n_per=3, T=8, gs=(3.0, 5.0))
    # every unit treated: drop the never-treated block
    keep = np.repeat([True] * 6 + [False] * 3, 8)
    with pytest.raises(ValueError, match="control='never' needs"):
        callaway_santanna(y2[keep], D2[keep], u2[keep], t2[keep],
                          control="never")


def test_imputation_and_saturated_regression_are_the_same_number():
    y, D, unit, time, _, truth = staggered(noise=0.2, seed=7)
    bjs = borusyak_jaravel_spiess(y, D, unit, time)
    etwfe = wooldridge_bjs_estimator(y, D, unit, time)
    assert etwfe["estimate"] == pytest.approx(bjs["estimate"], abs=1e-10)
    assert etwfe["matches_imputation"] == pytest.approx(0.0, abs=1e-9)
    for rel in bjs["event"]:
        assert etwfe["event"][rel] == pytest.approx(bjs["event"][rel],
                                                    abs=1e-9)


def test_imputation_is_exact_and_linear_in_the_outcome():
    y, D, unit, time, _, truth = staggered()
    out = borusyak_jaravel_spiess(y, D, unit, time)
    assert out["estimate"] == pytest.approx(truth, abs=1e-9)
    assert out["pretrend_max_abs"] == pytest.approx(0.0, abs=1e-9)
    # tau_hat = sum(v * Y) exactly, by construction of the weights
    assert out["linearity_residual"] == pytest.approx(0.0, abs=1e-9)
    # noiseless AND homogeneous: nothing is left for the SE to measure
    flat = staggered(dynamic=0.0)
    exact = borusyak_jaravel_spiess(flat[0], flat[1], flat[2], flat[3])
    assert exact["se"] == pytest.approx(0.0, abs=1e-9)
    # with dynamic effects the conservative SE picks up the real
    # heterogeneity across treated cells, not noise
    assert out["se"] > 0.05


def test_imputation_recovers_covariate_coefficients():
    y, D, unit, time, gv, truth = staggered()
    x = np.repeat(np.random.default_rng(3).normal(size=gv.size), 8) * time
    out = borusyak_jaravel_spiess(y + 1.5 * x, D, unit, time, X=x)
    assert out["covariate_coef"][0] == pytest.approx(1.5, abs=1e-8)
    assert out["estimate"] == pytest.approx(truth, abs=1e-8)


def test_imputation_refuses_an_unidentified_design():
    y, D, unit, time, _, _ = staggered()
    always = (time >= 0).astype(float)
    with pytest.raises(ValueError, match="treated in every period|not "
                                         "identified"):
        borusyak_jaravel_spiess(y, always, unit, time)


def test_imputation_standard_error_tracks_the_sampling_spread():
    base = staggered(n_per=8, T=8)
    ests, ses = [], []
    for s in range(150):
        noise = np.random.default_rng(s).normal(0, 0.5, base[0].size)
        out = borusyak_jaravel_spiess(base[0] + noise, base[1], base[2],
                                      base[3])
        ests.append(out["estimate"])
        ses.append(out["se"])
    assert np.mean(ests) == pytest.approx(base[5], abs=4 * np.std(ests) / 12)
    assert 0.75 < np.mean(ses) / np.std(ests, ddof=1) < 1.3


def test_doubly_robust_beats_the_unadjusted_did():
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.normal(size=n)
    D = (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(float)
    pre = x + rng.normal(size=n)
    post = pre + 0.8 * x + 2.0 * D + rng.normal(size=n)   # trend depends on x
    out = dr_did_santanna_zhao(pre, post, D, x)
    assert abs(out["estimate"] - 2.0) < 4 * out["se"]
    assert out["att_unadjusted"] > 2.4          # biased by the x-trend
    assert abs(out["covariate_adjustment"]) > 0.4


def test_double_robustness_survives_either_model_failing():
    rng = np.random.default_rng(5)
    n = 8000
    x = rng.normal(size=n)
    D = (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(float)
    pre = x + rng.normal(size=n)
    post = pre + 0.8 * x + 2.0 * D + rng.normal(size=n)
    out = dr_did_santanna_zhao(pre, post, D, x)
    checks = out["dr_check"]
    assert abs(checks["misspecified_propensity"] - 2.0) < 0.15
    assert abs(checks["misspecified_outcome"] - 2.0) < 0.15
    # with BOTH wrong there is no protection left, and it shows
    assert abs(checks["both_misspecified"] - 2.0) > 0.3
    assert checks["both_misspecified"] == pytest.approx(
        out["att_unadjusted"], abs=1e-9
    )


def test_doubly_robust_normalised_weights_are_shift_invariant():
    rng = np.random.default_rng(11)
    n = 2000
    x = rng.normal(size=n)
    D = (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(float)
    pre = x + rng.normal(size=n)
    post = pre + 0.5 * x + 1.5 * D + rng.normal(size=n)
    a = dr_did_santanna_zhao(pre, post, D, x)["estimate"]
    b = dr_did_santanna_zhao(pre + 100, post + 100, D, x)["estimate"]
    assert a == pytest.approx(b, abs=1e-9)


def test_doubly_robust_bootstrap_matches_the_influence_function():
    rng = np.random.default_rng(2)
    n = 3000
    x = rng.normal(size=n)
    D = (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(float)
    pre = x + rng.normal(size=n)
    post = pre + 0.5 * x + 1.0 * D + rng.normal(size=n)
    a = dr_did_santanna_zhao(pre, post, D, x)
    b = dr_did_santanna_zhao(pre, post, D, x, n_boot=400, seed=1)
    assert b["estimate"] == pytest.approx(a["estimate"])
    assert b["se"] == pytest.approx(a["se"], rel=0.25)
    assert b["se_method"].startswith("Mammen")


def test_doubly_robust_validates_its_inputs():
    rng = np.random.default_rng(1)
    y0, y1 = rng.normal(size=50), rng.normal(size=50)
    D = np.r_[np.ones(25), np.zeros(25)]
    with pytest.raises(ValueError, match="same length"):
        dr_did_santanna_zhao(y0[:10], y1, D)
    with pytest.raises(ValueError, match="binary"):
        dr_did_santanna_zhao(y0, y1, D * 3)
    with pytest.raises(ValueError, match="at least 2 treated"):
        dr_did_santanna_zhao(y0, y1, np.zeros(50))
    with pytest.raises(ValueError, match="X has 10 rows"):
        dr_did_santanna_zhao(y0, y1, D, rng.normal(size=10))


def test_dr_group_time_handles_covariate_dependent_trends():
    rng = np.random.default_rng(2)
    nu, T = 60, 6
    gv = np.where(np.arange(nu) < 20, 3.0,
                  np.where(np.arange(nu) < 40, 4.0, np.inf))
    xu = rng.normal(size=nu)
    unit = np.repeat(np.arange(nu), T)
    time = np.tile(np.arange(T), nu)
    g = np.repeat(gv, T)
    D = (time >= g).astype(float)
    X = np.repeat(xu, T)
    y = X + time * (1 + 0.5 * X) + 2.0 * D        # trend varies with x
    out = dr_callaway_santanna(y, D, unit, time, X=X)
    assert out["estimate"] == pytest.approx(2.0, abs=1e-8)
    assert abs(out["att_unadjusted"] - 2.0) > 0.15
    assert out["covariate_adjustment"] == pytest.approx(
        out["estimate"] - out["att_unadjusted"], abs=1e-12
    )
    assert max(abs(v) for v in out["pretrend"].values()) < 1e-8


def test_dr_group_time_matches_unweighted_when_covariates_are_irrelevant():
    y, D, unit, time, gv, truth = staggered()
    x = np.repeat(np.random.default_rng(9).normal(size=gv.size), 8)
    dr = dr_callaway_santanna(y, D, unit, time, X=x)
    cs = callaway_santanna(y, D, unit, time)
    assert dr["estimate"] == pytest.approx(cs["estimate"], abs=1e-8)


def test_average_effect_names_the_assumption_it_needs():
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.normal(size=n)
    D = (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(float)
    dY = 0.5 * x + D * (1.0 + 2.0 * x) + rng.normal(scale=0.3, size=n)
    out = avg_treatment_did(dY, D, x)
    assert abs(out["estimate"] - 1.0) < 4 * out["se"]     # ATE = 1 + 2 E[x]
    assert out["att"] > out["estimate"]                   # selection on gains
    assert out["atu"] < out["estimate"]
    # the decomposition is an identity, and the misnamed quantity is not it
    assert out["identity_check"] == pytest.approx(0.0, abs=1e-10)
    assert out["treated_contribution"] == pytest.approx(
        out["att"] * out["p_treated"]
    )
    assert out["treated_contribution"] != pytest.approx(out["estimate"])
    assert "not the ATE" in out["contribution_note"]


def test_average_effect_homogeneous_assumption_returns_the_att():
    rng = np.random.default_rng(4)
    n = 2000
    D = np.r_[np.ones(n // 2), np.zeros(n // 2)]
    dY = 1.0 * D + rng.normal(scale=0.5, size=n)
    out = avg_treatment_did(dY, D, assume="homogeneous")
    assert out["estimate"] == pytest.approx(out["att"])
    assert out["atu"] == pytest.approx(out["att"])
    assert abs(out["estimate"] - 1.0) < 4 * out["se"]
    with pytest.raises(ValueError, match="needs covariates"):
        avg_treatment_did(dY, D)


def test_synthetic_control_recovers_a_known_shift():
    rng = np.random.default_rng(0)
    f = rng.normal(size=20)
    load = np.array([1.0, 0.5, 1.5, 0.2, 1.0])
    Y = np.outer(load, f) + rng.normal(scale=0.01, size=(5, 20))
    Y[0, 12:] += 3.0
    out = synthetic_control(Y, None, None, 0, 12)
    assert out["estimate"] == pytest.approx(3.0, abs=0.1)
    # weights are a convex combination that reproduces the factor loading
    assert out["weights"].min() >= 0
    assert out["weights"].sum() == pytest.approx(1.0)
    assert float(out["weights"] @ load[1:]) == pytest.approx(1.0, abs=0.05)
    assert out["pre_rmspe"] < 0.05
    assert out["fit_quality"].startswith("good")


def test_synthetic_control_placebo_p_has_an_honest_floor():
    rng = np.random.default_rng(0)
    f = rng.normal(size=20)
    load = np.array([1.0, 0.5, 1.5, 0.2, 1.0])
    Y = np.outer(load, f) + rng.normal(scale=0.01, size=(5, 20))
    Y[0, 12:] += 3.0
    out = synthetic_control(Y, None, None, 0, 12)
    assert out["placebo_p"] == pytest.approx(out["p_value_floor"])
    assert out["p_value_floor"] == pytest.approx(0.2)     # 4 donors
    assert out["rmspe_ratio"] > max(v["ratio"] for v in
                                    out["placebo_ratios"].values())
    # with no effect the treated unit is unremarkable among the placebos
    Y0 = np.outer(load, f) + rng.normal(scale=0.01, size=(5, 20))
    null = synthetic_control(Y0, None, None, 0, 12)
    assert null["placebo_p"] > 0.2
    assert abs(null["estimate"]) < 0.1


def test_synthetic_control_refuses_impossible_setups():
    Y = np.random.default_rng(0).normal(size=(4, 10))
    with pytest.raises(ValueError, match="not in the unit set"):
        synthetic_control(Y, None, None, 99, 5)
    with pytest.raises(ValueError, match="after the last period"):
        synthetic_control(Y, None, None, 0, 50)
    with pytest.raises(ValueError, match="pre-treatment period"):
        synthetic_control(Y, None, None, 0, 1)
    with pytest.raises(ValueError, match="at least 3 units"):
        synthetic_control(Y[:2], None, None, 0, 5)


def test_synthetic_did_beats_plain_did_when_loadings_differ():
    rng = np.random.default_rng(1)
    f = np.cumsum(rng.normal(size=16))
    # treated units load heavily on the common factor, controls lightly:
    # plain DiD attributes the differential trend to treatment
    load = np.concatenate([[1.8, 1.9], rng.uniform(0.3, 0.7, size=10)])
    Y = np.outer(load, f) + rng.normal(scale=0.05, size=(12, 16))
    Y[:2, 10:] += 2.0
    out = synthetic_did(Y, None, None, [0, 1], 10)
    assert abs(out["estimate"] - 2.0) < abs(out["did_estimate"] - 2.0)
    assert out["unit_weights"].sum() == pytest.approx(1.0)
    assert out["unit_weights"].min() >= 0
    assert out["time_weights"].sum() == pytest.approx(1.0)
    assert out["zeta"] > 0
    assert np.isfinite(out["se"])
    assert out["se_method"].startswith("leave-one")


def test_synthetic_did_ridge_spreads_weight_across_donors():
    rng = np.random.default_rng(1)
    f = np.cumsum(rng.normal(size=16))
    load = np.concatenate([[1.0, 1.1], rng.uniform(0.4, 1.6, size=10)])
    Y = np.outer(load, f) + rng.normal(scale=0.05, size=(12, 16))
    Y[:2, 10:] += 2.0
    tight = synthetic_did(Y, None, None, [0, 1], 10, zeta=0.0)
    loose = synthetic_did(Y, None, None, [0, 1], 10, zeta=5.0)
    assert loose["n_donors_used"] >= tight["n_donors_used"]
    assert np.max(loose["unit_weights"]) <= np.max(tight["unit_weights"])


def test_synthetic_did_validates_its_setup():
    Y = np.random.default_rng(0).normal(size=(6, 12))
    with pytest.raises(ValueError, match="no treated unit"):
        synthetic_did(Y, None, None, [99], 6)
    with pytest.raises(ValueError, match="at least 2 control"):
        synthetic_did(Y, None, None, [0, 1, 2, 3, 4], 6)
    with pytest.raises(ValueError, match="pre-treatment period"):
        synthetic_did(Y, None, None, [0], 1)
    with pytest.raises(ValueError, match="after the last period"):
        synthetic_did(Y, None, None, [0], 99)


def test_the_shelf_agrees_where_it_should():
    """Every heterogeneity-robust estimator hits the same known truth."""
    y, D, unit, time, _, truth = staggered()
    cs = callaway_santanna(y, D, unit, time)["estimate"]
    bjs = borusyak_jaravel_spiess(y, D, unit, time)["estimate"]
    etwfe = wooldridge_bjs_estimator(y, D, unit, time)["estimate"]
    for est in (cs, bjs, etwfe):
        assert est == pytest.approx(truth, abs=1e-8)
    # and TWFE is the odd one out
    assert twoway_fixed_effects_did(y, D, unit, time)["estimate"] < truth - 0.3
