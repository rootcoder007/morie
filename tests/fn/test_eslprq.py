"""Tests for eslprq.esl_prototype_lvq."""

import pytest

from morie.fn.eslprq import esl_prototype_lvq


def _two_clusters():
    """Twenty points around (-3, -3) and twenty around (3, 3)."""
    X, y = [], []
    for i in range(20):
        off = 0.1 * (i % 5) - 0.2
        X.append([-3.0 + off, -3.0 - off])
        y.append(0)
    for i in range(20):
        off = 0.1 * (i % 5) - 0.2
        X.append([3.0 + off, 3.0 - off])
        y.append(1)
    return X, y


def test_eslprq_basic():
    """Well-separated classes are classified without error."""
    X, y = _two_clusters()
    result = esl_prototype_lvq(X, y, n_prototypes=2, seed=1)

    assert result["accuracy"] == 1.0
    assert result["n_prototypes"] == 2
    assert [int(c) for c in result["classes"]] == [0, 1]
    # Two prototypes per class, two classes, two predictors.
    assert result["prototypes"].shape == (4, 2)
    assert [int(c) for c in result["prototype_class"]] == [0, 0, 1, 1]
    # class_ defaults to the training predictions, which are all correct here.
    assert [int(c) for c in result["class_"]] == y
    # Each prototype stays inside its own cluster.
    P = result["prototypes"]
    for j in range(2):
        assert float(P[j][0]) < 0.0
    for j in (2, 3):
        assert float(P[j][0]) > 0.0


def test_eslprq_edge():
    """newdata classification and the documented input checks."""
    X, y = _two_clusters()
    result = esl_prototype_lvq(X, y, n_prototypes=1, seed=1,
                               newdata=[[-5.0, -5.0], [5.0, 5.0], [-2.5, -2.5]])
    assert [int(c) for c in result["class_"]] == [0, 1, 0]
    assert result["prototypes"].shape == (2, 2)

    with pytest.raises(ValueError, match="n_prototypes must be at least 1"):
        esl_prototype_lvq(X, y, n_prototypes=0)
    with pytest.raises(ValueError, match="eta must be in"):
        esl_prototype_lvq(X, y, eta=0.0)
    with pytest.raises(ValueError, match="y has"):
        esl_prototype_lvq(X, y[:-1])
    with pytest.raises(ValueError, match="fewer than n_prototypes"):
        esl_prototype_lvq(X, y, n_prototypes=21)
    with pytest.raises(ValueError, match="newdata has"):
        esl_prototype_lvq(X, y, newdata=[[0.0, 0.0, 0.0]])
