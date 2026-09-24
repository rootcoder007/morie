"""Tests for hmvth.geron_voting_hard."""

import math

from morie.fn import _array_core as np

from morie.fn.hmvth import geron_voting_hard


def test_hmvth_basic():
    """Test basic functionality."""
    # Simple models returning labels for 4 rows
    a = lambda X: [1, 1, 0, 0]
    b = lambda X: [1, 0, 1, 0]
    c = lambda X: [1, 1, 0, 1]

    X = [[0.0], [1.0], [2.0], [3.0]]
    y_true = [1, 0, 1, 0]

    result = geron_voting_hard([a, b, c], X, y_true=y_true)

    # Check all documented keys are present
    assert "predicted" in result
    assert "votes" in result
    assert "member_predictions" in result
    assert "member_accuracy" in result
    assert "accuracy" in result
    assert "agreement" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result

    # Check basic shapes
    assert len(result["predicted"]) == 4
    assert result["n"] == 4
    assert len(result["member_accuracy"]) == 3
    assert len(result["agreement"]) == 3
    assert len(result["member_predictions"]) == 3

    # Accuracy should be a finite probability
    acc = float(result["accuracy"])
    assert math.isfinite(acc)
    assert 0.0 <= acc <= 1.0


def test_hmvth_edge():
    """Test edge cases using the docstring example."""
    a = lambda X: [1, 1]
    b = lambda X: [1, 0]
    c = lambda X: [0, 0]

    X = [[0.0], [1.0]]
    y_true = [1, 0]

    result = geron_voting_hard([a, b, c], X, y_true=y_true)

    # These values are explicitly given in the docstring
    assert list(result["predicted"]) == [1, 0]
    assert float(result["accuracy"]) == 1.0
    assert list(result["member_accuracy"]) == [0.5, 1.0, 0.5]


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmvth as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
