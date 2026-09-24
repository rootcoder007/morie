"""Tests for grgs.geron_grid_search_cv."""

import math

from morie.fn import _array_core as np

from morie.fn.grgs import geron_grid_search_cv


def test_grgs_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    m, p = 40, 3
    X = rng.normal(0, 1, (m, p))
    y = rng.normal(0, 1, m)
    param_grid = {"a": [1, 2, 3], "b": [0, 10]}
    K = 4

    def fit_score(Xtr, ytr, Xva, yva, params):
        return -abs(params["a"] - 2) - 0.1 * params["b"]

    result = geron_grid_search_cv(X, y, param_grid, K, fit_score)
    assert isinstance(result, dict)
    for key in ("best_params", "best_score", "best_index",
                "mean_scores", "std_scores", "all_scores",
                "candidates", "n_fits", "estimate", "n", "method"):
        assert key in result
    assert result["best_params"] == {"a": 2, "b": 0}
    assert math.isfinite(result["best_score"])


def test_grgs_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    m, p = 20, 2
    X = rng.normal(0, 1, (m, p))
    y = rng.normal(0, 1, m)
    param_grid = {"alpha": [0.1, 1.0, 10.0]}
    K = 2

    def fit_score(Xtr, ytr, Xva, yva, params):
        return -abs(params["alpha"] - 1.0)

    result = geron_grid_search_cv(X, y, param_grid, K, fit_score)
    assert isinstance(result, dict)
    assert "best_params" in result
    assert "best_score" in result
    assert math.isfinite(result["best_score"])
    assert len(result["candidates"]) == 3
    assert result["n_fits"] == 6


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grgs as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
