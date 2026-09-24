"""Tests for kmgev.kamath_g_eval."""

import math

from morie.fn import _array_core as np

from morie.fn.kmgev import kamath_g_eval


def test_kmgev_basic():
    """Test basic functionality with uniform logits over a 3-point rubric."""
    rubric = [1, 2, 3]
    model = lambda x, y, r: [0.0, 0.0, 0.0]
    result = kamath_g_eval("q", "a", rubric, model)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "probabilities" in result
    assert "score_points" in result
    assert "logits" in result
    assert "n" in result
    assert "method" in result
    assert math.isclose(result["estimate"], 2.0)
    assert result["n"] == 3
    assert list(result["score_points"]) == [1.0, 2.0, 3.0]
    assert math.isclose(sum(result["probabilities"]), 1.0)
    assert all(math.isfinite(p) and p > 0.0 for p in result["probabilities"])


def test_kmgev_edge():
    """Test edge case with a 2-point rubric and non-uniform logits."""
    rubric = [1, 5]
    model = lambda x, y, r: [0.0, math.log(3)]
    result = kamath_g_eval("q", "a", rubric, model)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isclose(result["estimate"], 1 * 0.25 + 5 * 0.75)
    assert result["n"] == 2
    assert math.isclose(sum(result["probabilities"]), 1.0)
    assert math.isclose(result["probabilities"][0], 0.25)
    assert math.isclose(result["probabilities"][1], 0.75)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmgev as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
