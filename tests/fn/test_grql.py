"""Tests for grql.geron_q_learning_update."""

import math

from morie.fn import _array_core as np

from morie.fn.grql import geron_q_learning_update


def test_grql_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    Q = rng.normal(0, 1, (10, 4)).tolist()
    s = 0
    a = 0
    r = 1.0
    s_next = 1
    alpha = 0.1
    gamma = 0.9
    result = geron_q_learning_update(Q, s, a, r, s_next, alpha, gamma)
    assert isinstance(result, dict)
    for key in ("Q", "old_value", "new_value", "target", "td_error", "estimate", "n", "method"):
        assert key in result
    assert math.isfinite(result["target"])
    assert math.isfinite(result["new_value"])
    assert math.isfinite(result["old_value"])
    assert math.isfinite(result["td_error"])
    assert math.isfinite(result["estimate"])
    assert len(result["Q"]) == 10
    assert len(result["Q"][0]) == 4


def test_grql_edge():
    """Test edge cases: terminal transition drops the bootstrap."""
    rng = np.random.default_rng(42)
    Q = rng.normal(0, 1, (10, 4)).tolist()
    s = 0
    a = 0
    r = 1.0
    s_next = 1
    alpha = 0.1
    gamma = 0.9
    result = geron_q_learning_update(Q, s, a, r, s_next, alpha, gamma, done=True)
    assert isinstance(result, dict)
    for key in ("Q", "old_value", "new_value", "target", "td_error", "estimate", "n", "method"):
        assert key in result
    # Terminal transition: target is just r (no bootstrap).
    assert result["target"] == 1.0
    assert math.isfinite(result["new_value"])
    assert len(result["Q"]) == 10
    assert len(result["Q"][0]) == 4


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grql as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
