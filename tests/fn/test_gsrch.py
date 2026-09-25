"""Tests for gsrch.grid_search_cv."""

import pytest

from morie.fn import _array_core as np
from morie.fn.gsrch import grid_search_cv


def _data():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 2))
    y = 1.5 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(0, 0.5, 40)
    return X.tolist(), y.tolist()


def _ridge_r2(Xtr, ytr, Xte, yte, alpha):
    # centred ridge with an unpenalised intercept; 2x2 normal equations
    n = len(ytr)
    mx = [sum(r[j] for r in Xtr) / n for j in range(2)]
    my = sum(ytr) / n
    Xc = [[r[j] - mx[j] for j in range(2)] for r in Xtr]
    A = [[sum(r[i] * r[j] for r in Xc) + (alpha if i == j else 0.0) for j in range(2)]
         for i in range(2)]
    b = [sum(r[i] * (v - my) for r, v in zip(Xc, ytr)) for i in range(2)]
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    w = [(b[0] * A[1][1] - A[0][1] * b[1]) / det, (A[0][0] * b[1] - b[0] * A[1][0]) / det]
    b0 = my - w[0] * mx[0] - w[1] * mx[1]
    pred = [b0 + w[0] * r[0] + w[1] * r[1] for r in Xte]
    m = sum(yte) / len(yte)
    return 1.0 - sum((a - p) ** 2 for a, p in zip(yte, pred)) / sum((a - m) ** 2 for a in yte)


def test_gsrch_basic():
    """Continuous y selects ridge regression; each grid point's score is
    the mean R^2 over 5 contiguous folds (sklearn's KFold), recomputed."""
    X, y = _data()
    grid = [0.01, 1.0, 50.0]
    result = grid_search_cv(X, y, param_grid={"alpha": grid})
    assert isinstance(result, dict)
    assert result["task"] == "regression"
    exp = []
    for a in grid:
        sc = []
        for f in range(5):
            te = list(range(8 * f, 8 * f + 8))
            tr = [i for i in range(40) if i not in te]
            sc.append(_ridge_r2([X[i] for i in tr], [y[i] for i in tr],
                                [X[i] for i in te], [y[i] for i in te], a))
        exp.append(sum(sc) / 5)
    assert result["cv_results_mean_score"] == pytest.approx(exp, rel=1e-10, abs=1e-12)
    k = exp.index(max(exp))
    assert result["best_params"] == {"alpha": grid[k]}
    assert result["best_score"] == max(result["cv_results_mean_score"])


def test_gsrch_edge():
    """Integer labels with three classes select multinomial logistic;
    the best score is the maximum of the per-setting means."""
    X, _ = _data()
    y = [int(r[0] > 0.5) + int(r[1] > 0.0) for r in X]
    r = grid_search_cv(X, y, param_grid={"C": [0.1, 10.0]}, cv=3)
    assert r["task"] == "classification"
    assert r["best_score"] == max(r["cv_results_mean_score"])
    assert all(0.0 <= s <= 1.0 for s in r["cv_results_mean_score"])
