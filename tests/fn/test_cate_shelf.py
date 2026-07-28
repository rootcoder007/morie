"""Targeted learning and heterogeneous-effect estimation.

The through-line: every estimator here reports something the number
alone does not say. TMLE reports whether its own targeting converged
and whether positivity held; the forest reports out-of-bag spread next
to in-sample; the variance estimator reports how much of itself the
finite-tree correction removed; the GATE reports whether its groups
were cut from the estimate they summarise.
"""

import numpy as np
import pytest

from morie.fn.catep import cate_estimation
from morie.fn.cfst import causal_forest
from morie.fn.crfvar import causal_forest_variance
from morie.fn.gatep import gate_estimation
from morie.fn.tmleat import tmle_ate
from morie.fn.tmlhte import tmle_heterogeneous


def confounded(n=4000, effect=0.25, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, 2))
    p = 1 / (1 + np.exp(-(0.8 * W[:, 0] + 0.4 * W[:, 1])))
    A = (rng.uniform(size=n) < p).astype(float)
    y = 0.3 + 0.2 * W[:, 0] + effect * A + rng.normal(scale=0.1, size=n)
    return y, A, W


def test_tmle_recovers_the_effect_under_confounding():
    y, A, W = confounded()
    out = tmle_ate(y, A, W)
    assert abs(out["estimate"] - 0.25) < 4 * out["se"]
    naive = float(y[A == 1].mean() - y[A == 0].mean())
    assert abs(naive - 0.25) > 4 * out["se"]        # confounding is real
    assert out["ey1"] - out["ey0"] == pytest.approx(out["estimate"], abs=1e-9)


def test_tmle_targeting_solves_the_influence_equation():
    y, A, W = confounded(seed=3)
    out = tmle_ate(y, A, W)
    # this is the definition of the targeting step, not a coincidence
    assert abs(out["eif_mean"]) < 1e-6
    assert out["se"] == pytest.approx(
        float(np.sqrt(np.var(out["eif"], ddof=0) / out["n"])), rel=1e-6
    )


def test_tmle_reports_positivity_rather_than_hiding_it():
    rng = np.random.default_rng(1)
    n = 2000
    W = rng.normal(size=(n, 1))
    p = 1 / (1 + np.exp(-(6.0 * W[:, 0])))         # near-deterministic
    A = (rng.uniform(size=n) < p).astype(float)
    y = 0.1 * W[:, 0] + 0.2 * A + rng.normal(scale=0.1, size=n)
    out = tmle_ate(y, A, W, trunc=0.05)
    assert out["n_truncated"] > 0
    assert out["positivity_warning"] is not None
    clean = tmle_ate(*confounded(seed=5))
    assert clean["n_truncated"] == 0
    assert clean["positivity_warning"] is None


def test_tmle_accepts_a_known_propensity():
    rng = np.random.default_rng(2)
    n = 3000
    W = rng.normal(size=(n, 1))
    A = (rng.uniform(size=n) < 0.5).astype(float)   # randomised
    y = 0.2 * W[:, 0] + 0.3 * A + rng.normal(scale=0.1, size=n)
    known = tmle_ate(y, A, W, g=np.full(n, 0.5))
    assert known["propensity_supplied"] is True
    assert known["propensity"]["min"] == pytest.approx(0.5)
    assert abs(known["estimate"] - 0.3) < 4 * known["se"]


def test_stratum_tmle_finds_real_heterogeneity_and_not_imaginary():
    rng = np.random.default_rng(0)
    n = 4000
    s = rng.integers(0, 2, size=n)
    W = rng.normal(size=(n, 1))
    A = (rng.uniform(size=n) < 0.5).astype(float)
    y = 0.3 + 0.1 * W[:, 0] + A * (0.1 + 0.3 * s) + \
        rng.normal(scale=0.1, size=n)
    out = tmle_heterogeneous(y, A, W, s)
    assert out["by_stratum"][0]["estimate"] == pytest.approx(0.1, abs=0.03)
    assert out["by_stratum"][1]["estimate"] == pytest.approx(0.4, abs=0.03)
    assert out["heterogeneity_p"] < 1e-6

    flat = 0.3 + 0.1 * W[:, 0] + A * 0.2 + rng.normal(scale=0.1, size=n)
    null = tmle_heterogeneous(flat, A, W, s)
    assert null["heterogeneity_p"] > 0.05
    assert null["heterogeneity_q"] < 10


def test_stratum_tmle_drops_strata_it_cannot_estimate_with_a_reason():
    rng = np.random.default_rng(4)
    n = 1200
    s = rng.integers(0, 3, size=n)
    W = rng.normal(size=(n, 1))
    A = (rng.uniform(size=n) < 0.5).astype(float)
    A[s == 2] = 1.0                                 # stratum 2: no controls
    y = 0.2 * A + 0.1 * W[:, 0] + rng.normal(scale=0.1, size=n)
    out = tmle_heterogeneous(y, A, W, s)
    assert 2 in out["dropped"]
    assert "one arm" in out["dropped"][2]
    assert 2 not in out["by_stratum"]
    assert out["n_strata"] == 2
    with pytest.raises(ValueError, match="at least 2 strata"):
        tmle_heterogeneous(y, A, W, np.zeros(n))


def heterogeneous_forest(n=800, seed=1, trees=60):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    T = (rng.uniform(size=n) < 0.5).astype(float)
    Y = X[:, 0] + T * (X[:, 0] > 0) + rng.normal(scale=0.2, size=n)
    return causal_forest(Y, T, X, n_trees=trees, seed=seed), X


def test_causal_forest_orders_the_effect_by_the_driving_covariate():
    out, X = heterogeneous_forest()
    hi = out["cate"][X[:, 0] > 1].mean()
    lo = out["cate"][X[:, 0] < -1].mean()
    assert hi > 0.6 and lo < 0.25
    assert out["honest"] is True
    assert out["ate"] == pytest.approx(float(np.mean(out["cate"])))


def test_causal_forest_importance_needs_the_depth_weighting():
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n, 3))
    T = (rng.uniform(size=n) < 0.5).astype(float)
    Y = X[:, 0] + T * (X[:, 0] > 0) + rng.normal(scale=0.2, size=n)
    out = causal_forest(Y, T, X, n_trees=200, max_depth=6, seed=1)
    vi = out["var_importance"]
    raw = out["split_counts"] / out["split_counts"].sum()
    assert vi.sum() == pytest.approx(1.0)
    # only the driving covariate should stand out, and only once the
    # deep, noisy splits are downweighted
    assert vi[0] == vi.max() and vi[0] > 0.4
    assert raw[0] < vi[0]


def test_causal_forest_reports_out_of_bag_spread():
    rng = np.random.default_rng(5)
    n = 600
    X = rng.normal(size=(n, 3))
    T = (rng.uniform(size=n) < 0.5).astype(float)
    Y = rng.normal(scale=0.5, size=n)              # no effect at all
    out = causal_forest(Y, T, X, n_trees=100, seed=2)
    assert np.isfinite(out["oob_spread"])
    assert abs(out["ate"]) < 0.15
    assert out["n_oob_missing"] == 0


def test_infinitesimal_jackknife_intervals_cover_and_are_finite():
    rng = np.random.default_rng(7)
    n = 600
    X = rng.normal(size=(n, 2))
    T = (rng.uniform(size=n) < 0.5).astype(float)
    Y = T * X[:, 0] + rng.normal(scale=0.3, size=n)
    forest = causal_forest(Y, T, X, n_trees=2000, seed=1)
    Xq = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    v = causal_forest_variance(forest, Xq)
    assert np.all(v["se"] > 0)
    assert np.all(v["ci_lower"] < v["predictions"])
    assert np.all(v["predictions"] < v["ci_upper"])
    # tau(x) = x0, so the ordering must survive the noise
    assert v["predictions"][0] < v["predictions"][1] < v["predictions"][2]


def test_the_finite_tree_correction_shrinks_like_one_over_b():
    rng = np.random.default_rng(7)
    n = 600
    X = rng.normal(size=(n, 2))
    T = (rng.uniform(size=n) < 0.5).astype(float)
    Y = T * X[:, 0] + rng.normal(scale=0.3, size=n)
    Xq = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    shares = []
    for B in (200, 800, 2000):
        f = causal_forest(Y, T, X, n_trees=B, seed=1)
        v = causal_forest_variance(f, Xq)
        shares.append(float(np.median(v["correction_share"])))
    # the correction is Monte-Carlo noise in the forest, so it falls as
    # trees are added -- it does not fall as the sample grows
    assert shares[0] > shares[1] > shares[2]
    small = causal_forest_variance(
        causal_forest(Y, T, X, n_trees=200, seed=1), Xq
    )
    big = causal_forest_variance(
        causal_forest(Y, T, X, n_trees=2000, seed=1), Xq
    )
    assert small["reliable"] is False
    assert small["reliability_note"] is not None
    assert big["reliable"] is True
    # uncorrected, the estimate would be inflated at every tree count
    raw = causal_forest_variance(
        causal_forest(Y, T, X, n_trees=2000, seed=1), Xq, bias_correct=False
    )
    assert np.all(raw["variance"] >= big["variance"])


def test_variance_estimator_refuses_what_it_cannot_compute():
    with pytest.raises(ValueError, match="fitted causal forest"):
        causal_forest_variance(None)
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(size=(n, 2))
    T = (rng.uniform(size=n) < 0.5).astype(float)
    Y = rng.normal(size=n)
    with pytest.raises(ValueError, match="subsample < 1"):
        causal_forest_variance(causal_forest(Y, T, X, n_trees=20,
                                             subsample=1.0, seed=0))


def linear_cate(n=2000, seed=0, confound=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    if confound:
        p = 1 / (1 + np.exp(-X[:, 0]))
        T = (rng.uniform(size=n) < p).astype(float)
    else:
        T = (rng.uniform(size=n) < 0.5).astype(float)
    Y = X[:, 0] + T * (1 + X[:, 0]) + rng.normal(scale=0.3, size=n)
    return Y, T, X, 1 + X[:, 0]


def test_every_meta_learner_recovers_a_linear_cate():
    Y, T, X, tau = linear_cate()
    out = cate_estimation(Y, T, X)
    for name, est in out["by_estimator"].items():
        rmse = float(np.sqrt(np.mean((est - tau) ** 2)))
        assert rmse < (0.3 if name == "forest" else 0.1), name
    assert out["ate"] == pytest.approx(1.0, abs=0.05)
    assert out["cate"] is out["by_estimator"]["x"]


def test_meta_learners_survive_a_confounded_assignment():
    Y, T, X, tau = linear_cate(seed=4, confound=True)
    out = cate_estimation(Y, T, X, estimator="r")
    assert float(np.sqrt(np.mean((out["cate"] - tau) ** 2))) < 0.1
    assert out["propensity_range"][0] < 0.3
    assert out["propensity_range"][1] > 0.7


def test_meta_learner_disagreement_is_reported():
    Y, T, X, _ = linear_cate(seed=2)
    out = cate_estimation(Y, T, X)
    # on a design every learner fits, they agree -- and the correlation
    # matrix is what says so
    corr = out["agreement"]
    assert corr.shape == (5, 5)
    assert np.all(np.diag(corr) == pytest.approx(1.0))
    assert corr[0, 2] > 0.95
    assert np.mean(out["uncertainty"]) < 0.2
    with pytest.raises(ValueError, match="estimator must be one of"):
        cate_estimation(Y, T, X, estimator="q")
    with pytest.raises(ValueError, match="at least 5 units in each arm"):
        cate_estimation(Y[:20], np.zeros(20), X[:20])


def test_gate_averages_within_groups():
    tau = np.linspace(-1, 1, 400)
    g = (np.arange(400) >= 200).astype(int)
    out = gate_estimation(tau, group_var=g)
    assert out["gate"][0] == pytest.approx(-0.5, abs=0.01)
    assert out["gate"][1] == pytest.approx(0.5, abs=0.01)
    assert out["monotone"] is True
    assert out["selection_on_estimate"] is False
    assert out["selection_warning"] is None
    assert list(out["n_by_group"]) == [200, 200]


def test_gate_flags_sorting_on_the_estimate_it_summarises():
    # pure noise: there is NO heterogeneity to find
    noise = np.random.default_rng(0).normal(size=2000)
    selected = gate_estimation(noise)
    assert selected["selection_on_estimate"] is True
    assert selected["selection_warning"] is not None
    # sorted on itself, noise produces a large fake spread
    assert selected["spread"] > 2.0
    # grouped on an independent covariate, the same noise produces none
    x = np.random.default_rng(1).normal(size=2000)
    honest = gate_estimation(noise, X=x[:, None], group_var=0)
    assert honest["selection_on_estimate"] is False
    assert honest["spread"] < 0.3
    assert honest["spread"] < selected["spread"] / 5


def test_gate_standard_errors_and_input_checks():
    tau = np.linspace(-1, 1, 400)
    g = (np.arange(400) >= 200).astype(int)
    out = gate_estimation(tau, group_var=g, se=np.full(400, 0.05))
    assert out["se"] == pytest.approx(np.full(2, 0.05 / np.sqrt(200)))
    assert out["difference"] == pytest.approx(1.0, abs=0.02)
    assert out["difference_p"] < 1e-6
    assert out["se_note"] is not None
    labelled = gate_estimation(tau, group_var=g, labels=["low", "high"])
    assert labelled["groups"] == ["low", "high"]
    with pytest.raises(ValueError, match="labels has 3 entries"):
        gate_estimation(tau, group_var=g, labels=["a", "b", "c"])
    with pytest.raises(ValueError, match="se has 3 entries"):
        gate_estimation(tau, group_var=g, se=np.zeros(3))
    with pytest.raises(ValueError, match="out of range"):
        gate_estimation(tau, X=np.zeros((400, 2)), group_var=9)


def test_the_shelf_composes_end_to_end():
    """Forest -> variance -> groups, the way the modules are meant to chain."""
    rng = np.random.default_rng(3)
    n = 1200
    X = rng.normal(size=(n, 2))
    T = (rng.uniform(size=n) < 0.5).astype(float)
    Y = T * X[:, 0] + rng.normal(scale=0.3, size=n)
    forest = causal_forest(Y, T, X, n_trees=400, seed=1)
    v = causal_forest_variance(forest)
    # group on the covariate, not on the fitted effect
    out = gate_estimation(forest["cate"], X=X, group_var=0, n_groups=4,
                          se=v["se"])
    assert out["selection_on_estimate"] is False
    assert out["monotone"] is True
    assert out["gate"][0] < 0 < out["gate"][-1]
    assert out["difference"] > 1.0
