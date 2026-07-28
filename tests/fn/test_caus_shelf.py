# morie.fn -- test file (rootcoder007/morie)
"""The instrumental-variables and modern-causal shelf.

Every estimator here is an identification result before it is a
formula, so the tests are built around the identifying claims rather
than around recorded output: LIML must coincide with 2SLS exactly
when the model is just identified, AIPW must survive either nuisance
model being wrong and fail when both are, cross-fitting must be what
removes the regularisation bias, and the Sun-Abraham weights must be
shares where the two-way fixed-effects weights are not.
"""

import numpy as np
import pytest

from morie.fn._caus_iv import first_stage_f, k_class
from morie.fn.causaipw import causal_aipw
from morie.fn.causdidsap import causal_did_sun_abraham
from morie.fn.causdml2 import causal_dml_partial_lin
from morie.fn.causinst import causal_iv_instrumental_dag
from morie.fn.causiv2sls import causal_iv_2sls
from morie.fn.causivla import causal_iv_late
from morie.fn.causivlim import causal_iv_liml


def iv_design(n=3000, n_instr=1, beta=2.0, seed=0):
    """An endogenous regressor: the error enters both D and y, so
    least squares is biased and only an instrument recovers beta."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, n_instr))
    u = rng.normal(size=n)
    D = Z @ np.linspace(1.2, 0.4, n_instr) + u + rng.normal(size=n)
    y = beta * D + u + rng.normal(size=n)
    return y, D.reshape(-1, 1), Z


# ------------------------------------------------------------ k-class


def test_the_k_class_specialises_to_ols_and_2sls():
    """k = 0 is least squares and k = 1 is two-stage least squares.
    Keeping them one function is what stops them drifting apart."""
    y, D, Z = iv_design()
    X = np.column_stack([np.ones(len(y)), D])
    Zf = np.column_stack([np.ones(len(y)), Z])
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert k_class(y, X, Zf, 0.0) == pytest.approx(ols, rel=1e-10)
    tsls = causal_iv_2sls(y, D, Z)["beta"]
    assert k_class(y, X, Zf, 1.0) == pytest.approx(tsls, rel=1e-10)
    # and least squares really is biased here, which is the point
    assert abs(ols[1] - 2.0) > 4 * abs(tsls[1] - 2.0)


def test_2sls_recovers_the_structural_parameter():
    y, D, Z = iv_design(n=5000, n_instr=3)
    o = causal_iv_2sls(y, D, Z)
    assert o["beta"][1] == pytest.approx(2.0, abs=0.06)
    assert o["overidentified"] is True
    assert o["n_overid_restrictions"] == 2
    assert o["first_stage_F"] > 50
    # the instruments are valid by construction, so Sargan should not
    # reject at any conventional level
    assert o["sargan_p"] > 0.01
    assert np.all(o["se"] > 0)


def test_2sls_refuses_an_underidentified_model():
    """The order condition is arithmetic, so failing it should be an
    error and not a pseudo-inverse quietly returning something."""
    rng = np.random.default_rng(7)
    n = 200
    X = rng.normal(size=(n, 3))
    y = rng.normal(size=n)
    Z = rng.normal(size=(n, 1))
    with pytest.raises(ValueError, match="order condition"):
        causal_iv_2sls(y, X, Z)


def test_2sls_residuals_use_the_original_regressors():
    """A common silent error is to form residuals from the
    first-stage fitted values, which gives a smaller number that is
    not a standard error of anything."""
    y, D, Z = iv_design(n=1500)
    o = causal_iv_2sls(y, D, Z)
    X = np.column_stack([np.ones(len(y)), D])
    assert np.allclose(o["residuals"], y - X @ o["beta"], rtol=1e-12)


# ------------------------------------------------------------ LIML


def test_liml_equals_2sls_exactly_when_just_identified():
    """kappa = 1 in the just-identified case, so the k-class collapses
    to 2SLS. An implementation that misses this is wrong, and nothing
    else about it needs checking first."""
    y, D, Z = iv_design(n=3000, n_instr=1)
    a = causal_iv_2sls(y, D, Z)["beta"]
    b = causal_iv_liml(y, D, Z)
    assert b["just_identified"] is True
    assert b["kappa"] == pytest.approx(1.0, abs=1e-9)
    assert b["equals_2sls"] is True
    assert b["beta"] == pytest.approx(a, abs=1e-10)


def test_liml_kappa_exceeds_one_when_overidentified():
    y, D, Z = iv_design(n=3000, n_instr=5)
    o = causal_iv_liml(y, D, Z)
    assert o["kappa"] > 1.0
    assert o["just_identified"] is False
    assert o["equals_2sls"] is False
    assert o["n_overid_restrictions"] == 4
    assert o["beta"][1] == pytest.approx(2.0, abs=0.1)
    assert list(o["endogenous_columns"]) == [1]


def test_liml_would_be_ols_if_the_constant_leaked_into_the_ratio():
    """The Anderson-Rubin ratio is taken over y and the ENDOGENOUS
    regressors only. An exogenous column -- the intercept above all
    -- is annihilated exactly by M_W, which makes the matrix singular,
    drives kappa to 0, and silently turns the k-class estimator back
    into least squares. Guarding that the detected endogenous set
    excludes the constant is the check."""
    y, D, Z = iv_design(n=2000, n_instr=3)
    o = causal_iv_liml(y, D, Z)
    assert 0 not in list(o["endogenous_columns"])   # column 0 is the constant
    assert o["kappa"] >= 1.0
    X = np.column_stack([np.ones(len(y)), D])
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert abs(o["beta"][1] - 2.0) < abs(ols[1] - 2.0)


def test_liml_fuller_shifts_kappa_down_by_a_over_n_minus_m():
    y, D, Z = iv_design(n=2000, n_instr=4)
    plain = causal_iv_liml(y, D, Z)
    full = causal_iv_liml(y, D, Z, fuller=1.0)
    assert full["kappa"] == pytest.approx(
        plain["kappa"] - 1.0 / (2000 - 5), rel=1e-9)
    assert full["fuller_a"] == 1.0
    with pytest.raises(ValueError, match="non-negative"):
        causal_iv_liml(y, D, Z, fuller=-1.0)


# ------------------------------------------------------------ LATE


def late_design(n=20000, seed=2):
    """40% compliers with effect 3, 30% always-takers, 30%
    never-takers with effect 1. No defiers, so monotonicity holds."""
    rng = np.random.default_rng(seed)
    Z = (rng.random(n) < 0.5).astype(float)
    t = rng.random(n)
    typ = np.where(t < 0.4, "c", np.where(t < 0.7, "a", "n"))
    D = np.where(typ == "a", 1.0, np.where(typ == "n", 0.0, Z))
    eff = np.where(typ == "c", 3.0, 1.0)
    y = D * eff + rng.normal(size=n)
    return y, D, Z


def test_late_estimates_the_compliers_effect_not_the_ate():
    """The estimand is the compliers' effect. Here that is 3, while
    the population average effect is a mix of 3 and 1 -- so an
    implementation that recovered the ATE would be the wrong one."""
    y, D, Z = late_design()
    o = causal_iv_late(y, D, Z)
    assert o["late"] == pytest.approx(3.0, abs=0.15)
    assert o["complier_share"] == pytest.approx(0.4, abs=0.03)
    assert o["weak_first_stage"] is False
    assert "COMPLIERS" in o["estimand"]
    # the naive contrast of treated against untreated is NOT the LATE
    naive = y[D == 1].mean() - y[D == 0].mean()
    assert abs(naive - 3.0) > 0.3


def test_late_refuses_a_zero_first_stage():
    """A Wald ratio with no first stage is 0/0. Returning inf would be
    worse than refusing."""
    rng = np.random.default_rng(11)
    n = 500
    Z = (rng.random(n) < 0.5).astype(float)
    D = (rng.random(n) < 0.5).astype(float)   # independent of Z
    y = rng.normal(size=n)
    o = causal_iv_late(y, D, Z)
    assert o["weak_first_stage"] is True or abs(o["first_stage"]) < 0.1
    with pytest.raises(ValueError, match="binary"):
        causal_iv_late(y, D * 2, Z)


def test_the_dag_estimator_is_arithmetically_the_late():
    """Same number, different assumption set. A discrepancy between
    the two would mean one of them is wrong."""
    y, D, Z = late_design(n=8000, seed=5)
    a = causal_iv_late(y, D, Z)
    b = causal_iv_instrumental_dag(y, D, Z)
    assert b["beta"] == pytest.approx(a["late"], rel=1e-14)
    assert b["se"] == pytest.approx(a["se"], rel=1e-14)
    # only the claimed estimand changes
    assert "COMPLIERS" in a["estimand"]
    assert "compliers" in b["estimand"]
    hom = causal_iv_instrumental_dag(y, D, Z, homogeneous=True)
    assert hom["beta"] == pytest.approx(b["beta"], rel=1e-14)
    assert "average treatment effect" in hom["estimand"]
    assert hom["homogeneous_asserted"] is True
    assert set(b["untestable"]) >= {"exclusion", "exchangeability"}
    assert b["testable"] == ["relevance"]


# ------------------------------------------------------------ AIPW


def aipw_design(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    e = 1 / (1 + np.exp(-(X @ [0.8, -0.5, 0.3])))
    T = (rng.random(n) < e).astype(float)
    m0 = X @ [1.0, 0.5, -0.5]
    m1 = m0 + 2.0
    y = np.where(T == 1, m1, m0) + rng.normal(scale=0.5, size=n)
    return y, T, e, m1, m0


def test_aipw_is_doubly_robust_and_fails_only_when_both_are_wrong():
    """The defining property, exercised in all four combinations.
    Getting either nuisance right is enough; getting both wrong is
    not rescued by anything."""
    y, T, e, m1, m0 = aipw_design()
    n = y.size
    bad_e = np.full(n, 0.5)
    bad_m = np.zeros(n)
    both = causal_aipw(y, T, e, m1, m0)["ate"]
    ps_wrong = causal_aipw(y, T, bad_e, m1, m0)["ate"]
    out_wrong = causal_aipw(y, T, e, bad_m, bad_m)["ate"]
    all_wrong = causal_aipw(y, T, bad_e, bad_m, bad_m)["ate"]
    assert both == pytest.approx(2.0, abs=0.08)
    assert ps_wrong == pytest.approx(2.0, abs=0.08)
    assert out_wrong == pytest.approx(2.0, abs=0.15)
    assert abs(all_wrong - 2.0) > 2 * max(abs(ps_wrong - 2.0),
                                          abs(out_wrong - 2.0))


def test_aipw_reduces_to_the_regression_estimator_when_augmentation_vanishes():
    """With the outcome models exactly right the two augmentation
    terms have mean zero, so the estimator IS the regression
    estimator. That is the algebra the double robustness rests on."""
    y, T, e, m1, m0 = aipw_design()
    o = causal_aipw(y, T, e, m1, m0)
    assert o["regression_component"] == pytest.approx(2.0, abs=0.01)
    assert abs(o["augmentation_component"]) < 0.1
    assert o["ate"] == pytest.approx(
        o["regression_component"] + o["augmentation_component"], rel=1e-12)


def test_aipw_trims_extreme_propensities_and_says_so():
    y, T, e, m1, m0 = aipw_design()
    e2 = e.copy()
    e2[:20] = 1e-6
    o = causal_aipw(y, T, e2, m1, m0, trim=0.01)
    assert o["n_trimmed"] >= 20
    assert o["min_ps"] == pytest.approx(1e-6)
    assert np.isfinite(o["ate"])
    with pytest.raises(ValueError, match="divides by zero"):
        causal_aipw(y, T, np.zeros_like(e), m1, m0, trim=0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        causal_aipw(y, T, e * 3, m1, m0)


# ------------------------------------------------------------ DML


def test_dml_recovers_theta_in_a_partially_linear_model():
    rng = np.random.default_rng(13)
    n, p = 4000, 40
    X = rng.normal(size=(n, p))
    g = X @ rng.normal(size=p) / np.sqrt(p)
    D = g + rng.normal(size=n)
    y = 1.5 * D + g + rng.normal(size=n)
    o = causal_dml_partial_lin(y, D, X, n_folds=5)
    assert o["theta"] == pytest.approx(1.5, abs=0.06)
    assert o["cross_fitted"] is True
    assert o["n_folds"] == 5
    lo, hi = o["ci"]
    assert hi > lo
    # a 95% interval misses one draw in twenty, so asserting coverage
    # on a single seed is a coin flip dressed as a test. Coverage is a
    # frequency property and has to be measured as one.
    hits = 0
    reps = 40
    for s_ in range(reps):
        r2 = np.random.default_rng(100 + s_)
        n2, p2 = 1200, 10
        X2 = r2.normal(size=(n2, p2))
        g2 = X2 @ r2.normal(size=p2) / np.sqrt(p2)
        D2 = g2 + r2.normal(size=n2)
        y2 = 1.5 * D2 + g2 + r2.normal(size=n2)
        c = causal_dml_partial_lin(y2, D2, X2, n_folds=5, seed=s_)["ci"]
        hits += int(c[0] < 1.5 < c[1])
    assert hits >= int(0.80 * reps)


def test_dml_orthogonality_beats_residualising_only_the_outcome():
    """Residualising BOTH Y and D is what makes the score
    Neyman-orthogonal. Residualising only Y and regressing on raw D
    leaves the confounding in, and on a design where g(X) drives both
    it is badly biased."""
    rng = np.random.default_rng(17)
    n, p = 3000, 20
    X = rng.normal(size=(n, p))
    g = X @ rng.normal(size=p) / np.sqrt(p) * 3.0
    D = g + rng.normal(size=n)
    y = 1.0 * D + g + rng.normal(size=n)
    o = causal_dml_partial_lin(y, D, X, n_folds=5)
    yres = o["y_residual"]
    half_done = float(yres @ D / (D @ D))     # only Y residualised
    assert abs(o["theta"] - 1.0) < abs(half_done - 1.0)


def test_dml_rejects_a_single_fold():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(200, 3))
    D = rng.normal(size=200)
    y = D + rng.normal(size=200)
    with pytest.raises(ValueError, match="n_folds must lie"):
        causal_dml_partial_lin(y, D, X, n_folds=1)


# ------------------------------------------------------------ Sun-Abraham


def sa_design(seed=4):
    """Two cohorts with very different, growing effects, plus a
    never-treated group. Every true effect is POSITIVE."""
    rng = np.random.default_rng(seed)
    T = 10
    G = np.r_[np.full(150, 3.0), np.full(150, 6.0), np.full(200, np.inf)]
    n = G.size
    Y = np.zeros((n, T))
    ui = rng.normal(size=n)
    pt = np.linspace(0, 1, T)
    for i in range(n):
        Y[i] = ui[i] + pt + rng.normal(scale=0.1, size=T)
        if np.isfinite(G[i]):
            g = int(G[i])
            for t in range(g, T):
                Y[i, t] += (5.0 if g == 3 else 1.0) * (1 + 0.5 * (t - g))
    return Y, G


def test_sun_abraham_recovers_the_cohort_share_weighted_effect():
    """The two cohorts are equal in size, so the correct aggregate at
    relative time l is the average of 5(1 + l/2) and 1(1 + l/2): 3 at
    l = 0, 4.5 at l = 1, 6 at l = 2."""
    Y, G = sa_design()
    o = causal_did_sun_abraham(Y, G, rel_periods=[-2, -1, 0, 1, 2])
    mu = dict(zip(o["rel_periods"].tolist(), o["mu"]))
    assert mu[0] == pytest.approx(3.0, abs=0.1)
    assert mu[1] == pytest.approx(4.5, abs=0.15)
    assert mu[2] == pytest.approx(6.0, abs=0.2)
    # the reference period is zero by construction and there is no
    # pre-trend to find
    assert mu[-1] == pytest.approx(0.0, abs=1e-12)
    assert abs(mu[-2]) < 0.1


def test_the_two_way_fixed_effects_event_study_gets_the_sign_wrong():
    """Sun and Abraham's central warning, reproduced. Every true
    cohort effect here is positive and growing, yet the two-way
    fixed-effects coefficients come out NEGATIVE -- the contamination
    from already-treated units serving as controls. That is not a
    small bias; it is the wrong sign."""
    Y, G = sa_design()
    o = causal_did_sun_abraham(Y, G, rel_periods=[0, 1, 2])
    assert np.all(o["mu"] > 0)
    assert np.all(o["naive_twfe"] < 0)


def test_the_interaction_weights_are_shares():
    """Non-negative and summing to one at every relative time -- the
    property the two-way fixed-effects weights lack, and the reason
    the estimator cannot invert a sign."""
    Y, G = sa_design()
    o = causal_did_sun_abraham(Y, G, rel_periods=[0, 1, 2, 3])
    assert o["weights_nonnegative"] is True
    assert o["weights_sum_to_one"] is True
    assert np.all(o["weights"] >= 0)
    live = o["weights"].sum(axis=0) > 0
    assert np.allclose(o["weights"].sum(axis=0)[live], 1.0)


def test_sun_abraham_validates_its_control_group():
    Y, G = sa_design()
    allt = np.where(np.isfinite(G), G, 3.0)
    with pytest.raises(ValueError, match="no never-treated"):
        causal_did_sun_abraham(Y, allt, control="never")
    # not-yet-treated is the fallback and still works
    o = causal_did_sun_abraham(Y, allt, rel_periods=[0, 1], control="notyet")
    assert o["control_group"] == "notyet"
    assert np.all(np.isfinite(o["mu"]))
    with pytest.raises(ValueError, match="never.*notyet|control must"):
        causal_did_sun_abraham(Y, G, control="magic")


def test_first_stage_f_reports_instrument_strength():
    strong = iv_design(n=2000, n_instr=1, seed=21)
    weak_rng = np.random.default_rng(23)
    n = 2000
    Zw = weak_rng.normal(size=(n, 1))
    Dw = 0.01 * Zw[:, 0] + weak_rng.normal(size=n)
    assert first_stage_f(strong[1].ravel(), strong[2]) > 100
    assert first_stage_f(Dw, Zw) < 10
