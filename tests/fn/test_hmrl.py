"""Tests for hmrl.geron_reinforcement_learning."""

import math

from morie.fn import _array_core as np

from morie.fn.hmrl import geron_reinforcement_learning


def _make_env():
    """Create a simple environment that pays 1 for 3 steps then terminates."""
    clock = {"t": 0}

    def reset():
        clock["t"] = 0
        return 0

    def step(a):
        clock["t"] += 1
        return clock["t"], 1.0, clock["t"] >= 3

    return {"reset": reset, "step": step}


def test_hmrl_basic():
    """Test basic functionality."""
    env = _make_env()
    pi = lambda s: 0
    result = geron_reinforcement_learning(env, pi, gamma=0.5)
    assert isinstance(result, dict)
    assert "mean_return" in result
    assert "returns" in result
    assert "lengths" in result
    assert "effective_horizon" in result
    assert math.isfinite(result["mean_return"])
    assert len(result["returns"]) == 1
    assert len(result["lengths"]) == 1


def test_hmrl_edge():
    """Test edge cases."""
    env = _make_env()
    pi = lambda s: 0
    # Undiscounted case: each episode returns 3.0 (3 steps of reward 1)
    result = geron_reinforcement_learning(env, pi, gamma=1.0, n_episodes=3)
    assert isinstance(result, dict)
    assert "mean_return" in result
    assert math.isfinite(result["mean_return"])
    assert len(result["returns"]) == 3
    assert len(result["lengths"]) == 3


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmrl as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
