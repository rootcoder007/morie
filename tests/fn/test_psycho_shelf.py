# morie.fn -- test file (rootcoder007/morie)
"""The reliability, IRT-ability and meta-analysis shelf.

The ICC tests are against Shrout and Fleiss's (1979) OWN worked
example -- their Table 2 data and the six coefficients they print for
it -- so the arithmetic is checked against the paper rather than
against itself. The IRT tests turn on existence: the maximum
likelihood estimate does NOT exist for a perfect response pattern,
and MAP, EAP and Warm's WLE each return a finite number there by a
different mechanism, which is the whole reason all four modules
exist. The meta-analysis tests turn on tau^2 being part of the
answer: a different heterogeneity estimator is a different pooled
effect, not a footnote.
"""

import numpy as np
import pytest

from morie.fn._psycho import anova_two_way, spearman_brown
from morie.fn.eapth import eap_theta_estimator
from morie.fn.icc1k import icc_one_way_average
from morie.fn.icc2k import icc_two_way_random_avg
from morie.fn.icc3k import icc_two_way_mixed_avg
from morie.fn.maloo import ma_leave_one_out
from morie.fn.mapaule import ma_paule_mandel
from morie.fn.mapth import map_theta_estimator
from morie.fn.mareml import ma_random_reml
from morie.fn.marve import ma_robust_variance_est
from morie.fn.mleth import mle_theta_estimator
from morie.fn.theteap import theta_eap
from morie.fn.theteap2 import theta_map
from morie.fn.wleth import weighted_likelihood_theta


# Shrout and Fleiss (1979), Table 2: 6 targets rated by 4 judges.
SF_TABLE = np.array([[9, 2, 5, 8],
                     [6, 1, 3, 2],
                     [8, 4, 6, 8],
                     [7, 1, 2, 6],
                     [10, 5, 6, 9],
                     [6, 2, 4, 7]], dtype=float)


def sf_long():
    n, k = SF_TABLE.shape
    return (SF_TABLE.ravel(),
            np.repeat(np.arange(n), k),
            np.tile(np.arange(k), n))


# ------------------------------------------------- ICC


def test_the_three_iccs_reproduce_shrout_and_fleiss_table_2():
    """Their published values: ICC(1,1)=.17, ICC(2,1)=.29,
    ICC(3,1)=.71, ICC(1,k)=.44, ICC(2,k)=.62, ICC(3,k)=.91."""
    y, sub, rat = sf_long()
    a = icc_one_way_average(y, sub)
    b = icc_two_way_random_avg(y, sub, rat)
    c = icc_two_way_mixed_avg(y, sub, rat)
    assert a["icc_single"] == pytest.approx(0.17, abs=0.005)
    assert b["icc_single"] == pytest.approx(0.29, abs=0.005)
    assert c["icc_single"] == pytest.approx(0.71, abs=0.006)
    assert a["value"] == pytest.approx(0.44, abs=0.005)
    assert b["value"] == pytest.approx(0.62, abs=0.005)
    assert c["value"] == pytest.approx(0.91, abs=0.005)


def test_the_cases_are_ordered_and_the_ordering_has_a_reason():
    """ICC(1,*) <= ICC(2,*) <= ICC(3,*) on the same data, because
    each case charges less of the rater variance against
    reliability. Reporting the wrong case is how reliability gets
    overstated."""
    y, sub, rat = sf_long()
    a = icc_one_way_average(y, sub)
    b = icc_two_way_random_avg(y, sub, rat)
    c = icc_two_way_mixed_avg(y, sub, rat)
    assert a["value"] < b["value"] < c["value"]
    assert a["icc_single"] < b["icc_single"] < c["icc_single"]
    # the gap between cases 2 and 3 IS the rater penalty
    assert b["rater_penalty"] == pytest.approx(c["value"] - b["value"],
                                               rel=1e-9)
    assert c["icc2k"] == pytest.approx(b["value"], rel=1e-12)


def test_average_measure_follows_spearman_brown_exactly():
    y, sub, rat = sf_long()
    for o in (icc_one_way_average(y, sub),
              icc_two_way_random_avg(y, sub, rat),
              icc_two_way_mixed_avg(y, sub, rat)):
        assert o["value"] == pytest.approx(
            spearman_brown(o["icc_single"], o["k"]), rel=1e-9)


def test_a_constant_rater_offset_costs_case_3_nothing():
    """The consistency-versus-agreement distinction, made concrete:
    raters differing by a fixed constant on every target agree
    perfectly in RANK and not at all in LEVEL. Case 3 scores 1;
    Case 2 does not."""
    base = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    M = np.column_stack([base, base + 4.0, base + 8.0])
    y = M.ravel()
    sub = np.repeat(np.arange(6), 3)
    rat = np.tile(np.arange(3), 6)
    c3 = icc_two_way_mixed_avg(y, sub, rat)
    c2 = icc_two_way_random_avg(y, sub, rat)
    assert c3["value"] == pytest.approx(1.0, abs=1e-9)
    assert c2["value"] < 0.85
    assert c3["max_rater_offset"] == pytest.approx(8.0, rel=1e-12)


def test_icc_refuses_designs_its_formula_cannot_describe():
    y, sub, rat = sf_long()
    # unbalanced one-way
    with pytest.raises(ValueError, match="k ratings per target"):
        icc_one_way_average(y[:-1], sub[:-1])
    # incomplete crossed table
    with pytest.raises(ValueError, match="complete and crossed"):
        icc_two_way_random_avg(y[:-1], sub[:-1], rat[:-1])
    with pytest.raises(ValueError, match="at least 2 subjects"):
        anova_two_way([1.0, 2.0], [0, 0], [0, 1])


# ------------------------------------------------- IRT


def irt_items(m=20):
    return np.full(m, 1.2), np.linspace(-2.0, 2.0, m)


def test_no_finite_mle_for_a_perfect_pattern_but_the_others_deliver():
    """The organising fact of this shelf. ML has no maximum for an
    all-correct or all-wrong pattern; MAP and EAP get one from the
    prior, WLE from the information weight -- three different
    mechanisms, all finite."""
    a, b = irt_items()
    for pattern, sign in ((np.ones(20), 1), (np.zeros(20), -1)):
        ml = mle_theta_estimator(pattern, a=a, b=b)
        assert ml["finite"] is False
        assert ml["theta"] == sign * np.inf
        assert "no maximum exists" in ml["why_infinite"]
        for f in (map_theta_estimator, eap_theta_estimator,
                  weighted_likelihood_theta):
            o = f(pattern, a=a, b=b)
            assert np.isfinite(o["theta"])
            assert sign * o["theta"] > 0        # right direction
    assert map_theta_estimator(np.ones(20), a=a,
                               b=b)["exists_for_perfect_patterns"] is True
    assert weighted_likelihood_theta(
        np.ones(20), a=a, b=b)["finite_for_perfect_patterns"] is True


def test_map_and_eap_shrink_toward_the_prior_mean():
    a, b = irt_items()
    rng = np.random.default_rng(0)
    p = 1 / (1 + np.exp(-a * (1.5 - b)))
    y = (rng.random(20) < p).astype(float)
    ml = mle_theta_estimator(y, a=a, b=b)
    mp = map_theta_estimator(y, a=a, b=b)
    ep = eap_theta_estimator(y, a=a, b=b)
    assert ml["finite"] is True
    # both shrink toward zero
    assert abs(mp["theta"]) < abs(ml["theta"])
    assert mp["shrinkage_vs_ml"] is not None
    # the prior adds information, so the MAP interval is narrower
    assert mp["se"] < ml["se"]
    assert mp["posterior_information"] == pytest.approx(
        mp["information"] + 1.0, rel=1e-12)
    # EAP is a genuine posterior SD, not a curvature approximation
    assert ep["posterior_sd"] == pytest.approx(ep["se"], rel=1e-15)
    assert ep["no_optimisation"] is True


def test_warms_weighted_likelihood_reduces_the_ml_bias():
    """Warm's claim, measured by simulation rather than asserted:
    at a fixed true theta the WLE's mean error is smaller than the
    ML estimator's."""
    a, b = irt_items()
    true = 1.0
    ml_err, wl_err = [], []
    for r in range(400):
        rng = np.random.default_rng(1000 + r)
        p = 1 / (1 + np.exp(-a * (true - b)))
        y = (rng.random(20) < p).astype(float)
        m = mle_theta_estimator(y, a=a, b=b)
        if not m["finite"]:
            continue
        w = weighted_likelihood_theta(y, a=a, b=b)
        ml_err.append(m["theta"] - true)
        wl_err.append(w["theta"] - true)
    assert abs(np.mean(wl_err)) < abs(np.mean(ml_err))
    assert weighted_likelihood_theta(np.r_[np.ones(10), np.zeros(10)],
                                     a=a, b=b)["bias_corrected"] is True


def test_the_matrix_aliases_share_the_single_pattern_implementations():
    a, b = irt_items()
    items = np.column_stack([a, b])
    rng = np.random.default_rng(5)
    X = (rng.random((5, 20)) < 0.6).astype(float)
    te = theta_eap(X, items)
    tm = theta_map(X, items)
    assert te["theta"][0] == pytest.approx(
        eap_theta_estimator(X[0], a=a, b=b)["theta"], rel=1e-15)
    assert tm["theta"][0] == pytest.approx(
        map_theta_estimator(X[0], a=a, b=b)["theta"], rel=1e-15)
    assert te["alias_of"] == "morie.fn.eapth.eap_theta_estimator"
    assert tm["alias_of"] == "morie.fn.mapth.map_theta_estimator"
    # mode and mean are NOT the same number
    assert np.max(np.abs(te["theta"] - tm["theta"])) > 0


def test_irt_validates_its_inputs():
    a, b = irt_items()
    with pytest.raises(ValueError, match="binary"):
        mle_theta_estimator(np.full(20, 2.0), a=a, b=b)
    with pytest.raises(ValueError, match="difficulties b are required"):
        mle_theta_estimator(np.ones(20))
    with pytest.raises(ValueError, match="one entry per item"):
        mle_theta_estimator(np.ones(20), a=a[:5], b=b)
    with pytest.raises(ValueError, match="prior standard deviation"):
        map_theta_estimator(np.ones(20), a=a, b=b, prior=(0.0, 0.0))


# ------------------------------------------------- meta-analysis


def meta_sim(k=25, mu=0.5, tau2=0.09, seed=0):
    rng = np.random.default_rng(seed)
    vi = rng.uniform(0.01, 0.15, k)
    yi = mu + rng.normal(scale=np.sqrt(tau2), size=k) + \
        rng.normal(scale=np.sqrt(vi))
    return yi, vi


def test_paule_mandel_solves_its_defining_equation():
    """tau^2 is defined as the root of generalised Q = k - 1, so the
    test is that the returned value satisfies it -- not that it
    matches a recorded number."""
    yi, vi = meta_sim()
    o = ma_paule_mandel(yi, vi)
    w = 1.0 / (vi + o["tau2"])
    mu = float(np.sum(w * yi) / np.sum(w))
    genq = float(np.sum(w * (yi - mu) ** 2))
    assert genq == pytest.approx(len(yi) - 1, rel=1e-6)
    assert o["at_boundary"] is False
    assert o["mu"] == pytest.approx(mu, rel=1e-12)
    # PM exceeds the downward-biased DL on heterogeneous data
    assert o["tau2"] > o["tau2_dl"]


def test_paule_mandel_reports_a_boundary_truncation_as_one():
    """Homogeneous studies: Q falls below its expectation and the
    estimate is a truncation at zero, which is a different thing
    from an interior estimate of zero."""
    rng = np.random.default_rng(3)
    vi = rng.uniform(0.05, 0.2, 15)
    yi = 0.4 + rng.normal(scale=np.sqrt(vi))     # tau^2 = 0 truly
    o = ma_paule_mandel(yi, vi)
    assert o["tau2"] == 0.0
    assert o["at_boundary"] is True
    assert o["boundary_note"] is not None


def test_reml_exceeds_ml_by_the_degree_of_freedom_correction():
    """The 1/sum(w) term is the whole difference and it is
    positive, so REML's tau^2 is always at least ML's -- the same
    reason s^2 exceeds the MLE variance."""
    yi, vi = meta_sim(seed=7)
    o = ma_random_reml(yi, vi)
    assert o["tau2"] >= o["tau2_ml"]
    assert o["reml_correction"] == pytest.approx(o["tau2"] - o["tau2_ml"],
                                                 rel=1e-12)
    assert o["converged"] is True
    # the pooled estimate uses 1/(v + tau^2) weights
    w = 1.0 / (vi + o["tau2"])
    assert o["mu"] == pytest.approx(float(np.sum(w * yi) / np.sum(w)),
                                    rel=1e-12)


def test_the_tau2_estimator_changes_the_pooled_effect():
    """The claim that makes tau^2 part of the answer: swapping the
    estimator moves mu and the interval, not just a heterogeneity
    footnote."""
    yi, vi = meta_sim(k=30, tau2=0.15, seed=11)
    pm = ma_paule_mandel(yi, vi)
    rl = ma_random_reml(yi, vi)
    assert pm["tau2"] != pytest.approx(rl["tau2"], rel=1e-6)
    assert pm["mu"] != pytest.approx(rl["mu"], rel=1e-9)
    assert pm["se"] != pytest.approx(rl["se"], rel=1e-9)
    # both bracket the truth on this design
    assert pm["ci"][0] < 0.5 < pm["ci"][1]


def test_robust_variance_estimation_beats_naive_ses_under_clustering():
    """Dependent effect sizes: the naive standard error ignores the
    within-study correlation and understates. RVE is consistent in
    the number of STUDIES."""
    rng = np.random.default_rng(1)
    G, per = 20, 3
    cl = np.repeat(np.arange(G), per)
    u = rng.normal(scale=1.0, size=G)
    x = rng.normal(size=G * per)
    y = 0.4 + 0.6 * x + u[cl] + rng.normal(scale=0.3, size=G * per)
    o = ma_robust_variance_est(y, x, cl)
    assert o["beta"][0] == pytest.approx(0.4, abs=0.3)
    assert o["beta"][1] == pytest.approx(0.6, abs=0.2)
    Xd = np.column_stack([np.ones(y.size), x])
    bn = np.linalg.lstsq(Xd, y, rcond=None)[0]
    e = y - Xd @ bn
    s2 = float(e @ e) / (y.size - 2)
    naive = np.sqrt(np.diag(s2 * np.linalg.inv(Xd.T @ Xd)))
    assert o["se"][0] > naive[0]
    assert o["n_clusters"] == G
    assert o["n_effects"] == G * per
    assert np.all(o["df"] > 0)


def test_rve_refuses_fewer_clusters_than_parameters():
    rng = np.random.default_rng(13)
    y = rng.normal(size=9)
    X = rng.normal(size=(9, 4))
    cl = np.repeat(np.arange(3), 3)
    with pytest.raises(ValueError, match="number of STUDIES"):
        ma_robust_variance_est(y, X, cl)


def test_leave_one_out_refits_tau2_and_flags_conclusion_changes():
    """Deleting a study changes tau^2 and therefore EVERY weight --
    the channel a quick recomputation with the full-data tau^2
    misses."""
    rng = np.random.default_rng(3)
    vi = rng.uniform(0.005, 0.03, 12)
    yi = 0.3 + rng.normal(scale=0.4, size=12) + rng.normal(scale=np.sqrt(vi))
    o = ma_leave_one_out(yi, vi, method="PM")
    assert o["tau2_loo"].std() > 1e-6           # tau^2 really is refit
    assert np.max(np.abs(o["tau2_loo"] - o["tau2_full"])) > 1e-3
    assert o["mu_loo"].size == 12
    assert o["ci_loo"].shape == (12, 2)
    assert 0 <= o["most_influential"] < 12
    assert o["max_abs_delta"] == pytest.approx(
        float(np.max(np.abs(o["delta_mu"]))), rel=1e-12)
    assert o["flips_significance"].dtype == bool
    with pytest.raises(ValueError, match="at least 3 studies"):
        ma_leave_one_out(yi[:2], vi[:2])
    with pytest.raises(ValueError, match="PM"):
        ma_leave_one_out(yi, vi, method="magic")
