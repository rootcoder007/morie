"""Tests for hmcdt.geron_classification_tree (CART, Gini/entropy)."""

from morie.fn.hmcdt import geron_classification_tree


def _gini(ys):
    n = len(ys)
    return 1.0 - sum((ys.count(c) / n) ** 2 for c in set(ys))


def test_hmcdt_basic():
    """A stump on 1-D data: the chosen split is the exhaustive minimiser
    of the weighted Gini impurity, recomputed here, and each leaf's
    probabilities are its class frequencies."""
    X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0, 0, 1, 0, 1]
    result = geron_classification_tree(X, y, "gini", 1)
    assert isinstance(result, dict)
    costs = []
    for k in range(1, 5):
        L, R = y[:k], y[k:]
        costs.append((len(L) * _gini(L) + len(R) * _gini(R)) / 5.0)
    k = costs.index(min(costs)) + 1
    assert sorted(costs).count(min(costs)) == 1
    for i in range(5):
        leaf = y[:k] if i < k else y[k:]
        assert result["probabilities"][i] == [leaf.count(0) / len(leaf),
                                              leaf.count(1) / len(leaf)]
    assert result["n_leaves"] == 2


def test_hmcdt_edge():
    """Separable classes on feature 0 with a noise feature 1: the tree
    fits perfectly and all importance goes to feature 0; entropy agrees."""
    X = [[0.1, 5.0], [0.2, -1.0], [0.3, 2.0], [0.7, 4.0], [0.8, -3.0], [0.9, 0.5]]
    y = [0, 0, 0, 1, 1, 1]
    for crit in ("gini", "entropy"):
        r = geron_classification_tree(X, y, crit)
        assert r["predictions"] == y
        assert r["train_accuracy"] == 1.0
        assert list(r["feature_importances"]) == [1.0, 0.0]


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.hmcdt as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
