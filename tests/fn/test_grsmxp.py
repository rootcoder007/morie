"""Tests for grsmxp.geron_softmax_probability."""

import math

from morie.fn import _array_core as np

from morie.fn.grsmxp import geron_softmax_probability


def test_grsmxp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    theta = rng.normal(0, 1, (3, 4))
    result = geron_softmax_probability(X, theta)
    assert isinstance(result, dict)
    for key in ("probabilities", "predictions", "scores", "estimate", "n", "method"):
        assert key in result
    probs = result["probabilities"]
    assert len(probs) == 40
    assert len(probs[0]) == 4
    for row in probs:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-6)
        for v in row:
            assert 0.0 <= v <= 1.0
            assert math.isfinite(v)
    preds = result["predictions"]
    assert len(preds) == 40
    for p in preds:
        assert 0 <= p < 4
    assert result["n"] == 40


def test_grsmxp_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (8, 2))
    theta = rng.normal(0, 1, (2, 3))
    result = geron_softmax_probability(X, theta)
    assert isinstance(result, dict)
    probs = result["probabilities"]
    assert len(probs) == 8
    assert len(probs[0]) == 3
    for row in probs:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-6)
    scores = result["scores"]
    assert len(scores) == 8
    assert len(scores[0]) == 3
    assert result["n"] == 8


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grsmxp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
