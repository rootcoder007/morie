"""Tests for hmoob.geron_oob_score."""

import math

from morie.fn import _array_core as np

from morie.fn.hmoob import geron_oob_score


def test_hmoob_basic():
    """Test basic functionality with a regression task."""
    rng_x = np.random.default_rng(42)
    X = rng_x.normal(0, 1, (100, 5))
    rng_y = np.random.default_rng(43)
    y = list(rng_y.normal(0, 1, 100))

    n_models = 10

    def predict_fn(A):
        rows = np.asarray(A, dtype=float)
        return [float(r[0]) for r in rows]

    rng_bag = np.random.default_rng(44)
    models = []
    for _ in range(n_models):
        indices = list(rng_bag.integers(0, 100, 100))
        in_bag = [False] * 100
        for idx in indices:
            in_bag[int(idx)] = True
        models.append((predict_fn, in_bag))

    result = geron_oob_score(X, y, models)
    assert isinstance(result, dict)
    for key in ("oob_score", "oob_predictions", "covered",
                "mean_oob_votes", "estimate", "n", "method"):
        assert key in result
    assert int(result["n"]) == 100
    assert math.isfinite(float(result["oob_score"]))
    assert math.isfinite(float(result["mean_oob_votes"]))
    assert 0.0 <= float(result["mean_oob_votes"]) <= n_models
    assert len(result["covered"]) == 100
    covered_count = sum(1 for v in result["covered"] if float(v) > 0)
    assert 0 <= covered_count <= 100
    assert len(result["oob_predictions"]) == 100
    assert isinstance(result["method"], str)


def test_hmoob_edge():
    """Test with an explicit classification task and binary labels."""
    rng_x = np.random.default_rng(42)
    X = rng_x.normal(0, 1, (100, 5))
    rng_y = np.random.default_rng(43)
    y = [float(v) for v in list(rng_y.integers(0, 2, 100))]

    n_models = 5

    def predict_fn(A):
        rows = np.asarray(A, dtype=float)
        return [1.0 if r[0] > 0 else 0.0 for r in rows]

    rng_bag = np.random.default_rng(44)
    models = []
    for _ in range(n_models):
        indices = list(rng_bag.integers(0, 100, 100))
        in_bag = [False] * 100
        for idx in indices:
            in_bag[int(idx)] = True
        models.append((predict_fn, in_bag))

    result = geron_oob_score(X, y, models, task="classification")
    assert isinstance(result, dict)
    for key in ("oob_score", "oob_predictions", "covered",
                "mean_oob_votes", "estimate", "n", "method"):
        assert key in result
    assert int(result["n"]) == 100
    assert 0.0 <= float(result["oob_score"]) <= 1.0
    assert math.isfinite(float(result["oob_score"]))
    assert math.isfinite(float(result["mean_oob_votes"]))
    assert 0.0 <= float(result["mean_oob_votes"]) <= n_models
    assert len(result["oob_predictions"]) == 100


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmoob as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
