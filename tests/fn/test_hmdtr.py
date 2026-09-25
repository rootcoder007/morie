"""Tests for hmdtr.geron_tree_regularization."""

import math

import pytest

from morie.fn.hmdtr import geron_tree_regularization


X = [[math.sin(i), math.cos(1.7 * i)] for i in range(60)]
Y = [1 if (a + 0.6 * b + 0.4 * math.sin(3.1 * i)) > 0 else 0
     for i, (a, b) in enumerate(X)]


def test_hmdtr_basic():
    """Leaves, depth and training accuracy equal sklearn 1.9
    DecisionTreeClassifier(random_state=0) on the same data."""
    sk = {(): (11, 6, 60), (("max_depth", 2),): (4, 2, 55),
          (("min_samples_leaf", 5),): (6, 4, 54),
          (("min_samples_split", 10),): (7, 4, 56)}
    for kw, (leaves, depth, correct) in sk.items():
        r = geron_tree_regularization(X, Y, **dict(kw))
        assert (r["n_leaves"], r["depth"]) == (leaves, depth)
        assert r["train_score"] == pytest.approx(correct / 60, rel=1e-15)


def test_hmdtr_edge():
    """Depth 0 is a single majority leaf; constraints never add leaves."""
    r = geron_tree_regularization([[1.0], [2.0], [3.0], [4.0]], [0, 0, 1, 1], max_depth=0)
    assert (r["n_leaves"], r["baseline_leaves"]) == (1, 2)
    assert (r["train_score"], r["baseline_train_score"]) == (0.5, 1.0)
    r = geron_tree_regularization(X, Y, max_depth=3, min_samples_leaf=3)
    assert r["n_leaves"] <= r["baseline_leaves"]
    assert r["leaves_saved"] == r["baseline_leaves"] - r["n_leaves"]


