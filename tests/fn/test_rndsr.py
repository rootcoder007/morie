"""Tests for rndsr.random_search_cv."""

from morie.fn import _array_core as np
from morie.fn.rndsr import random_search_cv


def _data():
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (45, 2))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.3, 45)).tolist()
    return X.tolist(), y


def test_rndsr_basic():
    """n_iter draws from loguniform(1e-3, 1e2) for ridge alpha; the
    reported best is the arg-max of the sampled mean scores."""
    X, y = _data()
    result = random_search_cv(X, y, n_iter=6)
    assert isinstance(result, dict)
    assert result["task"] == "regression"
    ps, sc = result["sampled_params"], result["sampled_scores"]
    assert len(ps) == len(sc) == 6
    assert all(1e-3 <= p["alpha"] <= 1e2 for p in ps)
    k = sc.index(max(sc))
    assert result["best_score"] == sc[k]
    assert result["best_params"] == ps[k]


def test_rndsr_edge():
    """The same seed reproduces the draws; three integer classes run the
    multinomial logistic search."""
    X, y = _data()
    a = random_search_cv(X, y, n_iter=4, seed=3)
    b = random_search_cv(X, y, n_iter=4, seed=3)
    assert a["sampled_params"] == b["sampled_params"]
    yc = [0 if v < -0.5 else (1 if v < 0.5 else 2) for v in y]
    c = random_search_cv(X, yc, n_iter=3, cv=3)
    assert c["task"] == "classification"
    assert c["best_score"] == max(c["sampled_scores"])
