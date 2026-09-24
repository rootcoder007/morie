"""Tests for hmrsc.geron_randomized_search."""

import math

from morie.fn import _array_core as np

from morie.fn.hmrsc import geron_randomized_search


# Workaround: geron_cross_validation_score (called internally) passes
# `assume_unique=False` to np.setdiff1d, but the pure-Python _array_core
# implementation does not accept that keyword. Wrap it to drop unknown kwargs.
_orig_setdiff1d = np.setdiff1d


def _setdiff1d(a, b, **kwargs):
    return _orig_setdiff1d(a, b)


np.setdiff1d = _setdiff1d


def test_hmrsc_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    param_dist = {"alpha": [0.0, 0.1, 1.0, 10.0]}
    n_iter = 5
    result = geron_randomized_search(param_dist, n_iter, X, y, seed=0)
    assert isinstance(result, dict)
    expected_keys = {"best_params", "best_score", "candidates", "scores", "estimate", "n", "method"}
    assert expected_keys.issubset(result.keys())
    assert isinstance(result["candidates"], list)
    assert len(result["candidates"]) == n_iter
    assert len(result["scores"]) == n_iter
    assert isinstance(result["best_params"], dict)
    assert "alpha" in result["best_params"]
    assert math.isfinite(result["best_score"])
    assert isinstance(result["n"], int)
    assert isinstance(result["method"], str)
    for cand in result["candidates"]:
        assert isinstance(cand, dict)
        assert "alpha" in cand


def test_hmrsc_edge():
    """Test edge case with interval distribution."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    param_dist = {"alpha": (0.0, 1.0)}
    n_iter = 3
    result = geron_randomized_search(param_dist, n_iter, X, y, seed=1)
    assert isinstance(result, dict)
    expected_keys = {"best_params", "best_score", "candidates", "scores", "estimate", "n", "method"}
    assert expected_keys.issubset(result.keys())
    assert len(result["candidates"]) == n_iter
    assert len(result["scores"]) == n_iter
    for cand in result["candidates"]:
        assert isinstance(cand, dict)
        alpha = cand["alpha"]
        assert 0.0 <= alpha <= 1.0
    assert math.isfinite(result["best_score"])
