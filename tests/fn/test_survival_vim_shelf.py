"""Survival TMLE and algorithm-agnostic variable importance.

Every expectation here is derived from a design whose truth is known in
closed form, or from a property the estimator must satisfy by
construction. Nothing is anchored to whatever the code happened to
print.
"""

import numpy as np
import pytest

from morie.fn._did import add_intercept, ols_fit
from morie.fn._survtmle import discretise_times, survival_from_hazard
from morie.fn._vimp import gateaux_check, predictiveness, vim
from morie.fn.tmlavp import tmle_average_predictiveness
from morie.fn.tmlsur import tmle_survival

# --------------------------------------------------------------------
# survival fixture: discrete hazard, known counterfactual survival
# --------------------------------------------------------------------
#
#   logit lambda(k | A, W) = -1.5 + 0.5 W0 - 1.2 A          (constant in k)
#   so S_a(5) = E_W (1 - lambda_a)^5, computed exactly below.
#
# Treatment is confounded through W0, which also drives the hazard, so a
# naive comparison of arms is biased and the adjustment has something to
# do.

B0, BW, BA = -1.5, 0.5, -1.2
HORIZON = 5


def _true_survival():
    rng = np.random.default_rng(0)
    w = rng.normal(size=2_000_000)
    l1 = 1 / (1 + np.exp(-(B0 + BW * w + BA)))
    l0 = 1 / (1 + np.exp(-(B0 + BW * w)))
    return float(np.mean((1 - l1) ** HORIZON)), float(
        np.mean((1 - l0) ** HORIZON)
    )


TRUE_S1, TRUE_S0 = _true_survival()


def survival_data(seed=0, n=1500, censor=True):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, 2))
    g = 1 / (1 + np.exp(-0.8 * W[:, 0]))
    A = (rng.uniform(size=n) < g).astype(float)
    lam = 1 / (1 + np.exp(-(B0 + BW * W[:, 0] + BA * A)))
    T = np.full(n, 10.0)
    for k in range(1, 10):
        hit = (rng.uniform(size=n) < lam) & (T == 10.0)
        T[hit] = k
    C = rng.integers(3, 13, size=n).astype(float) if censor else np.full(n, 99.0)
    return np.minimum(T, C), (T <= C).astype(int), A, W


def test_true_survival_difference_is_what_the_design_says():
    # guards the fixture itself: treatment lowers the hazard, so the
    # treated survive more often, and the gap is roughly a third
    assert TRUE_S1 > TRUE_S0
    assert 0.30 < TRUE_S1 - TRUE_S0 < 0.37


def test_hazard_to_survival_is_the_running_product():
    lam = np.array([[0.1, 0.2, 0.5]])
    s = survival_from_hazard(lam)
    assert s[0, 0] == pytest.approx(0.9)
    assert s[0, 1] == pytest.approx(0.9 * 0.8)
    assert s[0, 2] == pytest.approx(0.9 * 0.8 * 0.5)


def test_time_grid_keeps_a_short_discrete_scale_intact():
    t = np.array([1.0, 2.0, 2.0, 5.0, 3.0])
    k, edges = discretise_times(t)
    # five distinct values or fewer means the observed times ARE the grid
    assert np.array_equal(np.unique(edges), np.array([1.0, 2.0, 3.0, 5.0]))
    assert np.array_equal(k, np.array([1, 2, 2, 4, 3]))


def test_survival_tmle_recovers_the_counterfactual_difference():
    Y, E, A, W = survival_data(seed=11, n=3000)
    out = tmle_survival(Y, E, A, W, tau=HORIZON)
    assert abs(out["estimate"] - (TRUE_S1 - TRUE_S0)) < 0.05
    assert abs(out["s1"] - TRUE_S1) < 0.05
    assert abs(out["s0"] - TRUE_S0) < 0.05


def test_targeting_solves_the_influence_curve_equation():
    Y, E, A, W = survival_data(seed=12, n=1500)
    out = tmle_survival(Y, E, A, W, tau=HORIZON)
    # the whole point of the targeting step: P_n D* is driven to zero,
    # and 1/n is the threshold the loop is written against
    assert out["converged"]
    assert abs(out["eif_mean"]) <= 1.0 / out["n"]


def test_estimate_stays_a_probability_difference():
    # substitution estimators cannot leave the parameter space, however
    # the nuisance fits behave; an estimating-equation correction can
    Y, E, A, W = survival_data(seed=13, n=400)
    out = tmle_survival(Y, E, A, W, tau=HORIZON)
    assert 0.0 <= out["s1"] <= 1.0
    assert 0.0 <= out["s0"] <= 1.0
    assert -1.0 <= out["estimate"] <= 1.0


def test_survival_curves_are_monotone_and_start_below_one():
    Y, E, A, W = survival_data(seed=14, n=1200)
    out = tmle_survival(Y, E, A, W)
    for curve in (out["curve1"], out["curve0"]):
        assert np.all(np.diff(curve) <= 1e-12)
        assert np.all((curve >= 0) & (curve <= 1))


def test_confounding_is_actually_removed():
    # the naive contrast ignores that treatment is assigned on W0, which
    # also drives the hazard; the adjusted estimate must be closer to
    # truth than the naive one
    Y, E, A, W = survival_data(seed=15, n=3000)
    naive = float(np.mean(Y[A == 1] > HORIZON)) - float(
        np.mean(Y[A == 0] > HORIZON)
    )
    out = tmle_survival(Y, E, A, W, tau=HORIZON)
    truth = TRUE_S1 - TRUE_S0
    assert abs(out["estimate"] - truth) < abs(naive - truth)


def test_standard_error_tracks_the_sampling_variation():
    ests, ses = [], []
    for s in range(25):
        Y, E, A, W = survival_data(seed=200 + s, n=1500)
        o = tmle_survival(Y, E, A, W, tau=HORIZON)
        ests.append(o["estimate"])
        ses.append(o["se"])
    mc_sd = float(np.std(ests, ddof=1))
    mean_se = float(np.mean(ses))
    # measured: MC sd 0.0236, mean reported SE 0.0247 -> ratio 1.05
    assert 0.75 < mean_se / mc_sd < 1.4


def test_censoring_is_adjusted_for_not_ignored():
    # with heavy censoring the naive complete-case survival is biased
    # upward at a fixed horizon; the estimator must not inherit that
    Y, E, A, W = survival_data(seed=16, n=3000, censor=True)
    out = tmle_survival(Y, E, A, W, tau=HORIZON)
    assert out["n_censored"] > 0
    assert abs(out["estimate"] - (TRUE_S1 - TRUE_S0)) < 0.06


def test_survival_input_validation():
    Y, E, A, W = survival_data(seed=17, n=200)
    with pytest.raises(ValueError, match="binary"):
        tmle_survival(Y, np.full(Y.size, 2), A, W)
    with pytest.raises(ValueError, match="agree in length"):
        tmle_survival(Y[:-1], E, A, W)
    with pytest.raises(ValueError, match="each arm"):
        tmle_survival(Y, E, np.zeros_like(A), W)


# --------------------------------------------------------------------
# variable importance
# --------------------------------------------------------------------
#
#   y = 2 X0 + X1 + N(0, 1),  X ~ iid N(0, 1)
#   Var(y) = 4 + 1 + 1 = 6
#   drop X0 -> residual variance 5 -> psi = (5 - 1)/6 = 4/6
#   drop X1 -> residual variance 2 -> psi = (2 - 1)/6 = 1/6
#   drop X2 -> psi = 0 exactly (the null)

VIM_TRUTH = {0: 4 / 6, 1: 1 / 6, 2: 0.0}


def vim_data(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 2 * X[:, 0] + X[:, 1] + rng.normal(scale=1.0, size=n)
    return y, X


def ols_learner(Xtr, ytr):
    b = ols_fit(add_intercept(Xtr), ytr)
    return lambda Z: add_intercept(Z) @ b


@pytest.mark.parametrize("measure", ["r_squared", "accuracy", "auc",
                                     "deviance"])
def test_every_gradient_is_the_real_gateaux_derivative(measure):
    # tilt the empirical distribution towards one point EXACTLY (append
    # a duplicate, eps = 1/(n+1)) and difference; a resampled tilt is
    # useless here because its noise divided by eps swamps the gradient
    rng = np.random.default_rng(0)
    n = 1500
    if measure == "r_squared":
        y = rng.normal(size=n)
        p = 0.7 * y + rng.normal(scale=0.5, size=n)
    else:
        y = (rng.uniform(size=n) < 0.4).astype(float)
        p = np.clip(0.3 + 0.4 * y + rng.normal(scale=0.2, size=n), 0.01, 0.99)
    _, grad = predictiveness(y, p, measure)
    assert abs(float(np.mean(grad))) < 1e-12
    assert gateaux_check(y, p, measure) < 0.02


def test_gateaux_error_is_curvature_and_shrinks_with_n():
    errs = []
    for n in (250, 1000, 4000):
        rng = np.random.default_rng(0)
        y = rng.normal(size=n)
        p = 0.7 * y + rng.normal(scale=0.5, size=n)
        errs.append(gateaux_check(y, p, "r_squared"))
    # the discrepancy is the O(eps) curvature term with eps = 1/(n+1),
    # so it must fall roughly in proportion to n
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.1 * errs[0]


def test_r_squared_matches_its_definition():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.5, 2.0, 2.5, 4.0])
    v, _ = predictiveness(y, p, "r_squared")
    mse = np.mean((y - p) ** 2)
    assert v == pytest.approx(1 - mse / np.var(y))


def test_auc_matches_the_mann_whitney_count():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.4, 0.3, 0.9])
    v, _ = predictiveness(y, p, "auc")
    # pairs (case, control): (0.3,0.1) win, (0.3,0.4) loss,
    # (0.9,0.1) win, (0.9,0.4) win -> 3/4
    assert v == pytest.approx(0.75)


def test_perfect_and_useless_predictions_bracket_the_measures():
    rng = np.random.default_rng(2)
    y = (rng.uniform(size=400) < 0.5).astype(float)
    for m in ("accuracy", "auc"):
        good, _ = predictiveness(y, y * 0.99 + 0.005, m)
        flat, _ = predictiveness(y, np.full(y.size, 0.5), m)
        assert good == pytest.approx(1.0)
        assert flat == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize("j", [0, 1, 2])
def test_importance_recovers_the_analytic_value(j):
    ests = []
    for s in range(30):
        y, X = vim_data(seed=1000 + s)
        ests.append(vim(y, X, j, f=ols_learner, seed=s)["estimate"])
    # measured over 120 reps: bias +0.008, +0.008, +0.003 for j = 0,1,2
    assert abs(float(np.mean(ests)) - VIM_TRUTH[j]) < 0.03


def test_importance_ordering_follows_the_signal():
    y, X = vim_data(seed=4)
    est = [vim(y, X, j, f=ols_learner, seed=1)["estimate"] for j in range(3)]
    assert est[0] > est[1] > est[2]


def test_sample_splitting_is_what_makes_the_null_testable():
    # under psi = 0 the two influence functions coincide, so WITHOUT
    # splitting their difference is identically zero and the estimator
    # has no non-degenerate limit -- visible as an MC standard
    # deviation that collapses, and a test that never rejects
    spread = {}
    reject = {}
    for split in (True, False):
        ests, rej = [], 0
        for s in range(60):
            y, X = vim_data(seed=5000 + s, n=1200)
            o = vim(y, X, 2, f=ols_learner, seed=s, sample_split=split)
            ests.append(o["estimate"])
            rej += o["p_value"] < 0.05
        spread[split] = float(np.std(ests, ddof=1))
        reject[split] = rej / 60.0
    # measured: split 0.0177 vs unsplit 0.0012 -- an order of magnitude
    assert spread[True] > 5 * spread[False]
    # measured type-I over 200 reps: 0.055 split, 0.000 unsplit
    assert 0.0 <= reject[True] <= 0.15
    assert reject[False] < 0.02


def test_null_flag_is_reported_not_left_to_the_reader():
    y, X = vim_data(seed=6)
    on = tmle_average_predictiveness(y, 2, X, f=ols_learner)
    off = tmle_average_predictiveness(y, 2, X, f=ols_learner,
                                      sample_split=False)
    assert on["null_inference_valid"] is True
    assert on["null_note"] is None
    assert off["null_inference_valid"] is False
    assert "degenerate" in off["null_note"]


def test_learner_is_refitted_per_fold_without_the_dropped_columns():
    seen = []

    def spy(Xtr, ytr):
        seen.append(Xtr.shape[1])
        return ols_learner(Xtr, ytr)

    y, X = vim_data(seed=7, n=600)
    vim(y, X, 0, f=spy, n_folds=3)
    # three folds for the full model (3 columns) and three for the
    # reduced one (2 columns); a single pre-fitted f could not do this
    assert sorted(seen) == [2, 2, 2, 3, 3, 3]


def test_group_importance_is_at_least_the_largest_member():
    y, X = vim_data(seed=8, n=2000)
    both = vim(y, X, [0, 1], f=ols_learner, seed=2)["estimate"]
    alone = vim(y, X, 0, f=ols_learner, seed=2)["estimate"]
    # dropping both signals must cost at least as much as dropping one
    assert both > alone - 0.02
    assert both == pytest.approx(5 / 6, abs=0.06)


def test_default_learner_runs_and_orders_the_variables():
    y, X = vim_data(seed=9, n=800)
    est = [tmle_average_predictiveness(y, j, X, seed=1)["estimate"]
           for j in range(3)]
    assert est[0] > est[1] > est[2]


def test_binary_measures_reject_a_continuous_outcome():
    y, X = vim_data(seed=10, n=400)
    for m in ("accuracy", "auc", "deviance"):
        with pytest.raises(ValueError, match="binary"):
            tmle_average_predictiveness(y, 0, X, loss=m)


def test_vim_input_validation():
    y, X = vim_data(seed=11, n=400)
    with pytest.raises(ValueError, match="outside the"):
        vim(y, X, 9)
    with pytest.raises(ValueError, match="empty"):
        vim(y, X, [])
    with pytest.raises(ValueError, match="must be one of"):
        tmle_average_predictiveness(y, 0, X, loss="hinge")
    with pytest.raises(TypeError, match="fitting function"):
        vim(y, X, 0, f=lambda a, b: 3.0)
