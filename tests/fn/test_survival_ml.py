"""Random survival forests, AFT boosting, DeepSurv, Deep Survival Machines."""
import importlib
import math

import pytest

R = importlib.import_module("morie.fn.survrsf")
G = importlib.import_module("morie.fn.surxgb")
N = importlib.import_module("morie.fn.survnnr")
V = importlib.import_module("morie.fn.survvae")
C = importlib.import_module("morie.fn.coxph")


def simulate(n, seed):
    rng = R._Rng(seed)
    X, t, e = [], [], []
    for _ in range(n):
        X.append([rng.next(), rng.next()])
        lam = 0.2 + 6.0 * X[-1][0] ** 2
        ti = -math.log(rng.next()) / lam
        ci = -math.log(rng.next()) / 0.15
        t.append(min(ti, ci))
        e.append(1 if ti <= ci else 0)
    return X, t, e


X, T, E = simulate(160, 11)


# ------------------------------------------------------------- survrsf
def test_conservation_of_events_is_exact():
    c = R.conservation_check(T, E)
    assert c["conserved"]
    assert c["sum_chf"] == pytest.approx(sum(E), abs=1e-9)


def test_conservation_on_a_hand_checkable_case():
    c = R.conservation_check([1.0, 2.0, 3.0], [1, 0, 1])
    assert c["sum_chf"] == pytest.approx(2.0)


def test_nelson_aalen_steps_only_at_deaths():
    na = R.nelson_aalen([1.0, 2.0, 3.0], [1, 0, 1])
    assert na["time"] == [1.0, 3.0]
    assert na["chf"] == pytest.approx([1.0 / 3.0, 1.0 / 3.0 + 1.0])


def test_nelson_aalen_needs_matching_lengths():
    with pytest.raises(ValueError):
        R.nelson_aalen([1.0, 2.0], [1])


def test_c_index_is_one_for_a_perfect_ranking():
    assert R.c_index([1.0, 2.0, 3.0], [1, 1, 1],
                     [3.0, 2.0, 1.0])["c_index"] == 1.0
    assert R.c_index([1.0, 2.0, 3.0], [1, 1, 1],
                     [1.0, 2.0, 3.0])["c_index"] == 0.0


def test_ties_score_a_half():
    assert R.c_index([1.0, 2.0], [1, 1],
                     [5.0, 5.0])["c_index"] == pytest.approx(0.5)


def test_a_censored_shorter_time_makes_a_pair_impermissible():
    assert R.c_index([1.0, 2.0, 3.0], [0, 1, 1],
                     [1.0, 2.0, 3.0])["permissible"] == 1.0


def test_no_permissible_pairs_is_an_error():
    with pytest.raises(ValueError):
        R.c_index([1.0, 2.0], [0, 0], [1.0, 2.0])


def test_the_unsourced_splitting_rules_are_refused():
    for rule in ("conserve", "logrankscore"):
        assert not R.rule_status(rule)["available"]
        with pytest.raises(ValueError):
            R.forest(X, T, E, n_trees=2, rule=rule)


def test_an_unknown_splitting_rule_is_refused():
    with pytest.raises(ValueError):
        R.rule_status("magic")


def test_the_forest_leaves_a_third_of_the_data_out_of_bag():
    f = R.forest(X, T, E, n_trees=20, mtry=2, min_deaths=5, seed=3)
    assert 0.30 < f["oob_fraction"] < 0.42


def test_terminal_nodes_respect_the_death_minimum():
    tree = R.grow_tree(X, T, E, mtry=2, min_deaths=5, seed=1)
    assert all(l["na"]["deaths"] >= 5 for l in R._leaves(tree["root"]))


def test_the_out_of_bag_ensemble_is_less_optimistic_than_in_bag():
    f = R.forest(X, T, E, n_trees=25, mtry=2, min_deaths=5, seed=3)
    oob = R.c_index(T, E, R.mortality(f, X))["c_index"]
    inbag = R.c_index(T, E, R.mortality(f, X, oob=False))["c_index"]
    assert inbag > oob > 0.5


# -------------------------------------------------------------- surxgb
@pytest.mark.parametrize("dist", G.DISTRIBUTIONS)
@pytest.mark.parametrize("bounds", [(2.0, 2.0), (2.0, float("inf")),
                                    (0.0, 3.0), (1.5, 4.0)])
def test_analytic_derivatives_match_the_loss(dist, bounds):
    a = G.aft_gradient_hessian(bounds[0], bounds[1], 0.3, 1.1, dist,
                               "analytic")
    b = G.aft_gradient_hessian(bounds[0], bounds[1], 0.3, 1.1, dist,
                               "numeric")
    assert a["gradient"] == pytest.approx(b["gradient"], abs=1e-4)
    assert a["hessian"] == pytest.approx(b["hessian"], rel=1e-2,
                                         abs=1e-4)


def test_the_extreme_distribution_is_asymmetric():
    assert G.cdf(0.0, "extreme") == pytest.approx(1 - math.exp(-1.0))
    assert G.cdf(0.0, "logistic") == pytest.approx(0.5)
    assert G.cdf(0.0, "normal") == pytest.approx(0.5)


def test_an_unknown_distribution_is_refused():
    with pytest.raises(ValueError):
        G.pdf(0.0, "cauchy")


def test_the_censoring_types_use_the_right_pieces():
    u = math.log(2.0)
    assert G.aft_loss(2.0, float("inf"), u) == pytest.approx(
        -math.log(0.5))
    assert G.aft_loss(0.0, 2.0, u) == pytest.approx(-math.log(0.5))
    assert G.aft_loss(2.0, 2.0, u) == pytest.approx(
        -math.log(G.pdf(0.0) / 2.0))


def test_an_inverted_interval_is_refused():
    with pytest.raises(ValueError):
        G.aft_loss(3.0, 1.0, 0.0)


def test_a_non_positive_sigma_is_refused():
    with pytest.raises(ValueError):
        G.aft_loss(2.0, 2.0, 0.0, sigma=0.0)


def test_the_leaf_weight_and_the_split_gain():
    assert G.leaf_weight(4.0, 2.0, 1.0) == pytest.approx(-4.0 / 3.0)
    assert G.split_gain(1.0, 1.0, 1.0, 1.0, 1.0, 0.0) < 0.0
    assert G.split_gain(3.0, 1.0, -3.0, 1.0, 1.0, 0.0) > 0.0


def test_boosting_reduces_the_loss():
    lo = [max(t, 1e-3) for t in T]
    hi = [lo[i] if E[i] else float("inf") for i in range(len(lo))]
    f = G.boost(X, lo, hi, n_rounds=12, eta=0.2, sigma=0.7)
    assert f["loss_history"][-1] < f["loss_history"][0]


# -------------------------------------------------------------- survnnr
def test_the_linear_network_reproduces_coxph():
    rng = R._Rng(2)
    n = 120
    Xc = [[rng.next(), rng.next()] for _ in range(n)]
    t, e = [], []
    for i in range(n):
        lam = math.exp(1.5 * Xc[i][0] - Xc[i][1])
        ti = -math.log(rng.next()) / lam
        ci = -math.log(rng.next()) / 0.5
        t.append(min(ti, ci))
        e.append(1 if ti <= ci else 0)
    cox = C.coxph(t, e, Xc)["coefficients"]
    lin = N.fit(Xc, t, e, hidden=(), lr=1.0, n_epochs=20000, tol=1e-14)
    for j in range(2):
        assert lin["coefficients"][j] == pytest.approx(float(cox[j]),
                                                       abs=5e-3)


def test_the_partial_likelihood_ignores_an_additive_shift():
    risk = [0.1 * i for i in range(len(T))]
    a = N.partial_loglik(T, E, risk)["loglik"]
    b = N.partial_loglik(T, E, [v + 5.0 for v in risk])["loglik"]
    assert a == pytest.approx(b)


def test_the_partial_likelihood_needs_an_event():
    with pytest.raises(ValueError):
        N.partial_loglik(T, [0] * len(T), [0.0] * len(T))


def test_an_unknown_activation_is_refused():
    with pytest.raises(ValueError):
        N.fit(X, T, E, activation="sigmoid")


def test_breslow_hazard_is_non_decreasing():
    fit = N.fit(X, T, E, hidden=(), lr=0.5, n_epochs=300)
    bh = N.baseline_hazard(fit)
    assert all(bh["cumulative_hazard"][i]
               <= bh["cumulative_hazard"][i + 1]
               for i in range(len(bh["time"]) - 1))
    sf = N.survival_function(fit, X[0])
    assert all(0.0 <= v <= 1.0 for v in sf["survival"])


# -------------------------------------------------------------- survvae
def test_the_weibull_survival_is_exact():
    assert V.log_survival(2.0, 1.5, 3.0) == pytest.approx(
        -((2.0 / 3.0) ** 1.5))


def test_the_lognormal_is_at_its_median_at_the_scale():
    assert math.exp(V.log_survival(3.0, 1.0, 3.0,
                                   "lognormal")) == pytest.approx(0.5)


@pytest.mark.parametrize("p", V.PRIMITIVES)
def test_survival_starts_at_one(p):
    assert math.exp(V.log_survival(0.0, 1.3, 2.0, p)) == 1.0


def test_an_unknown_primitive_is_refused():
    with pytest.raises(ValueError):
        V.log_pdf(1.0, 1.0, 1.0, "gompertz")


def test_a_non_positive_time_has_no_density():
    with pytest.raises(ValueError):
        V.log_pdf(0.0, 1.0, 1.0)


def test_the_gates_are_a_probability_vector():
    g = V.gates([0.3, 1.0], [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
    assert sum(g) == pytest.approx(1.0)
    assert all(v > 0.0 for v in g)


def test_the_elbo_sits_below_the_exact_likelihood():
    W = [[0.0, 0.0], [0.0, 0.0]]
    e = V.elbo(X, T, E, W, [0.0, 0.0], [1.5, 3.0], [2.0, 6.0])
    x = V.exact_loglik(X, T, E, W, [0.0, 0.0], [1.5, 3.0], [2.0, 6.0])
    assert e["elbo"] < x["loglik"]


def test_alpha_zero_removes_the_censored_contribution():
    W = [[0.0, 0.0], [0.0, 0.0]]
    moved = [T[i] * (3.0 if not E[i] else 1.0) for i in range(len(T))]
    a = V.elbo(X, T, E, W, [0.0, 0.0], [1.5, 3.0], [2.0, 6.0],
               alpha=0.0)["elbo"]
    b = V.elbo(X, moved, E, W, [0.0, 0.0], [1.5, 3.0], [2.0, 6.0],
               alpha=0.0)["elbo"]
    assert a == pytest.approx(b)


def test_an_alpha_outside_the_unit_interval_is_refused():
    W = [[0.0, 0.0], [0.0, 0.0]]
    with pytest.raises(ValueError):
        V.elbo(X, T, E, W, [0.0, 0.0], [1.5, 3.0], [2.0, 6.0],
               alpha=1.5)


def test_competing_risks_need_a_non_censored_cause():
    with pytest.raises(ValueError):
        V.fit_competing(X, T, [0] * len(T), K=1)


def test_predicted_survival_is_a_decreasing_probability():
    fit = V.fit(X, T, E, K=2, seed=1, restarts=1)
    ps = V.predict_survival(fit, X[0], [0.5, 1.0, 2.0, 4.0])
    assert all(0.0 <= v <= 1.0 for v in ps["survival"])
    assert all(ps["survival"][i] >= ps["survival"][i + 1]
               for i in range(3))
