"""trclrn -- tree-based ITR. Sources: Laber, E. B. & Zhao, Y. Q.
(2015) Biometrika 102(3), 501-514, doi:10.1093/biomet/asv028; Tao, Y.
& Wang, L. (2017) Biometrics 73, 145-155, doi:10.1111/biom.12539."""
import pytest

from morie.fn import _array_core as np
from morie.fn.trclrn import (fit_tree, predict_rule, rule_value,
                             tree_rules)


def trial(n=800, seed=9, hetero=True):
    rng = np.random.default_rng(seed)
    Y, A, X = [], [], []
    for _ in range(n):
        x0 = float(rng.uniform())
        x1 = float(rng.uniform())
        a = 1 if float(rng.uniform()) < 0.5 else 0
        tau = (2.0 if x0 > 0.5 else -2.0) if hetero else 1.0
        Y.append(1.0 + 3.0 * x1 + a * tau
                 + float(rng.normal(0.0, 0.4)))
        A.append(a)
        X.append([x0, x1])
    return Y, A, X


def test_ipw_value_is_twice_the_concordant_mean_at_half():
    Y, A, X = trial()
    v = rule_value(Y, A, X, lambda x: 1, propensity=0.5)
    conc = sum(Y[i] for i in range(len(Y)) if A[i] == 1)
    assert v == pytest.approx(2.0 * conc / len(Y), abs=1e-12)


def test_the_oracle_rule_beats_both_fixed_arms():
    Y, A, X = trial()
    v_or = rule_value(Y, A, X, lambda x: 1 if x[0] > 0.5 else 0,
                      propensity=0.5)
    v_1 = rule_value(Y, A, X, lambda x: 1, propensity=0.5)
    v_0 = rule_value(Y, A, X, lambda x: 0, propensity=0.5)
    assert v_or > v_1 and v_or > v_0


def test_the_fitted_tree_beats_the_best_fixed_arm():
    Y, A, X = trial()
    f = fit_tree(Y, A, X, propensity=0.5, max_depth=2, min_leaf=40)
    assert f["value"] > max(f["fixed_arm_values"].values()) + 0.2


def test_the_tree_splits_on_the_effect_modifier():
    Y, A, X = trial()
    f = fit_tree(Y, A, X, propensity=0.5, max_depth=2, min_leaf=40)
    assert not f["tree"]["leaf"]
    assert f["tree"]["feature"] == 0


def test_the_split_lands_near_the_planted_threshold():
    Y, A, X = trial()
    f = fit_tree(Y, A, X, propensity=0.5, max_depth=2, min_leaf=40)
    assert abs(f["tree"]["threshold"] - 0.5) < 0.12


def test_the_learned_rule_mostly_agrees_with_the_oracle():
    Y, A, X = trial()
    f = fit_tree(Y, A, X, propensity=0.5, max_depth=2, min_leaf=40)
    pred = predict_rule(f["tree"], X)
    agree = sum(1 for i in range(len(X))
                if pred[i] == (1 if X[i][0] > 0.5 else 0)) / len(X)
    assert agree > 0.85


def test_a_homogeneous_design_gains_little_from_splitting():
    Y, A, X = trial(hetero=False)
    f = fit_tree(Y, A, X, propensity=0.5, max_depth=2, min_leaf=40)
    assert f["value"] - max(f["fixed_arm_values"].values()) < 0.3


def test_the_augmented_route_also_finds_the_rule():
    Y, A, X = trial()

    def om(x, a):
        return 1.0 + 3.0 * x[1] + a * (2.0 if x[0] > 0.5 else -2.0)

    f = fit_tree(Y, A, X, propensity=0.5, method="augmented",
                 outcome_model=om, max_depth=2, min_leaf=40)
    assert f["tree"]["feature"] == 0


def test_tree_rules_print_one_line_per_node():
    Y, A, X = trial()
    f = fit_tree(Y, A, X, propensity=0.5, max_depth=1, min_leaf=40)
    lines = tree_rules(f["tree"], names=["x0", "x1"])
    assert any("if x0 <" in ln for ln in lines)
    assert sum(1 for ln in lines if "treat with" in ln) == \
        f["n_leaves"]


def test_depth_zero_gives_a_single_leaf():
    Y, A, X = trial()
    f = fit_tree(Y, A, X, propensity=0.5, max_depth=0)
    assert f["n_leaves"] == 1


def test_a_positivity_violation_is_refused():
    Y, A, X = trial()
    with pytest.raises(ValueError):
        rule_value(Y, A, X, lambda x: 1, propensity=0.001)


def test_a_single_treatment_arm_is_refused():
    Y, A, X = trial()
    with pytest.raises(ValueError):
        fit_tree(Y, [1] * len(A), X, propensity=0.5)


def test_augmented_without_an_outcome_model_is_refused():
    Y, A, X = trial()
    with pytest.raises(ValueError):
        fit_tree(Y, A, X, propensity=0.5, method="augmented")


def test_an_unknown_method_is_refused():
    Y, A, X = trial()
    with pytest.raises(ValueError):
        fit_tree(Y, A, X, propensity=0.5, method="owl")


def test_mismatched_lengths_are_refused():
    Y, A, X = trial()
    with pytest.raises(ValueError):
        fit_tree(Y, A[:-1], X, propensity=0.5)


def test_a_min_leaf_below_one_is_refused():
    Y, A, X = trial()
    with pytest.raises(ValueError):
        fit_tree(Y, A, X, propensity=0.5, min_leaf=0)
