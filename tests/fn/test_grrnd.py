"""Tests for grrnd.geron_randomized_search_cv."""

from morie.fn import _array_core as np

from morie.fn.grrnd import geron_randomized_search_cv


def test_grrnd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    param_dist = {"alpha": (0.0, 10.0)}
    n_iter = 5
    K = 4

    def fit_score(Xtr, ytr, Xva, yva, params):
        # Prefer the smallest sampled alpha.
        return -params["alpha"]

    result = geron_randomized_search_cv(
        X, y, param_dist, n_iter, K, fit_score=fit_score, seed=1
    )
    assert isinstance(result, dict)
    assert "best_params" in result
    assert "best_score" in result
    assert "results" in result
    assert "fold_sizes" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert len(result["results"]) == n_iter
    assert result["best_score"] == max(row["mean_score"] for row in result["results"])


def test_grrnd_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    param_dist = {
        "alpha": (0.0, 10.0),
        "kernel": ["linear", "rbf", "poly"],
    }
    n_iter = 3
    K = 2

    def fit_score(Xtr, ytr, Xva, yva, params):
        return -params["alpha"]

    result = geron_randomized_search_cv(
        X, y, param_dist, n_iter, K, fit_score=fit_score, seed=1
    )
    assert isinstance(result, dict)
    assert "best_params" in result
    assert len(result["results"]) == n_iter
    assert all(set(row["params"].keys()) == {"alpha", "kernel"}
               for row in result["results"])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grrnd as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
