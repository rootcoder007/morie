"""Tests for dtrsp.decision_tree_split."""

from morie.fn import _array_core as np
import pytest

from morie.fn.dtrsp import decision_tree_split


def test_dtrsp_finds_the_true_split_variable_and_threshold():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] > 0.0).astype(int)
    r = decision_tree_split(X, y, criterion="gini", max_depth=1)
    assert int(r["root_feature"]) == 0
    assert abs(float(r["root_threshold"])) < 0.15
    assert int(r["n_leaves"]) == 2
    assert float(r["train_accuracy"]) >= 0.99
    imp = np.asarray(r["feature_importances"], dtype=float)
    assert imp[0] == pytest.approx(1.0, abs=1e-9)


def test_dtrsp_root_gini_matches_esl_eq_9_17():
    """Root impurity is sum_k p_k (1 - p_k) of the class shares before any
    split (Hastie et al. 2009, eq. 9.17, p. 309) -- computable by hand
    from the label counts alone."""
    y = np.array([0] * 30 + [1] * 10)
    X = np.arange(40.0).reshape(-1, 1)
    r = decision_tree_split(X, y, criterion="gini", max_depth=1)
    p = np.array([30 / 40, 10 / 40])
    assert float(r["root_impurity"]) == pytest.approx(float((p * (1 - p)).sum()), rel=1e-12)
    # Two classes: 2p(1-p) with p = 1/4 gives 3/8.
    assert float(r["root_impurity"]) == pytest.approx(0.375, rel=1e-12)


def test_dtrsp_entropy_criterion_reports_entropy():
    y = np.array([0] * 20 + [1] * 20)
    X = np.arange(40.0).reshape(-1, 1)
    r = decision_tree_split(X, y, criterion="entropy", max_depth=1)
    # Balanced two-class entropy is log(2) nats = 1 bit; sklearn uses log2.
    assert float(r["root_impurity"]) == pytest.approx(1.0, rel=1e-9)


def test_dtrsp_pure_node_has_zero_impurity():
    r = decision_tree_split(np.arange(10.0).reshape(-1, 1), np.zeros(10, dtype=int))
    assert float(r["root_impurity"]) == pytest.approx(0.0, abs=1e-12)
