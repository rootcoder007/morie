"""Tests for hmdqn.geron_dqn."""

from morie.fn import _array_core as np

from morie.fn.hmdqn import geron_dqn


def test_hmdqn_basic():
    """Test basic functionality."""
    env = None
    Q = np.zeros((4, 2))
    Q_target = np.zeros((4, 2))
    buffer = [
        (0, 0, 1.0, 1, False),
        (1, 1, 0.5, 2, False),
        (2, 0, 0.0, 3, True),
        (3, 1, -0.5, 0, False),
    ]
    result = geron_dqn(env, Q, Q_target, buffer, epochs=5, lr=0.1)
    assert isinstance(result, dict)
    assert "Q" in result
    assert "Q_target" in result
    assert "loss_history" in result
    assert "td_errors" in result
    assert "greedy_policy" in result
    assert "sync_epochs" in result
    assert "n_updates" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert len(result["Q"]) == 4
    assert len(result["Q"][0]) == 2
    assert len(result["loss_history"]) == 5
    assert len(result["greedy_policy"]) == 4
    assert isinstance(result["n_updates"], int)
    assert result["n_updates"] >= 0


def test_hmdqn_edge():
    """Test edge cases."""
    env = None
    Q = [[0.0, 0.0]]
    Q_target = [[0.0, 0.0]]
    buffer = [(0, 0, 1.0, 0, True)]
    result = geron_dqn(env, Q, Q_target, buffer, epochs=1, lr=0.5)
    assert isinstance(result, dict)
    assert "Q" in result
    assert "loss_history" in result
    assert "td_errors" in result
    assert "greedy_policy" in result
    assert len(result["Q"]) == 1
    assert len(result["Q"][0]) == 2
    assert len(result["loss_history"]) == 1
    assert len(result["greedy_policy"]) == 1
    import math
    assert math.isfinite(result["loss_history"][0])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmdqn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
