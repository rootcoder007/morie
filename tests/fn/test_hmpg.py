"""Tests for hmpg.geron_policy_gradient."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.hmpg import geron_policy_gradient


def test_hmpg_basic():
    """Test basic functionality."""
    # Policy returns a gradient vector (one entry per parameter)
    def policy(state, action):
        return np.array([1.0, 0.0, 0.0])

    # Two episodes of (state, action, reward) steps
    trajectories = [
        [(0, 0, 1.0), (1, 1, 0.5)],
        [(0, 0, 2.0), (1, 0, 1.0), (2, 1, 0.0)],
    ]
    result = geron_policy_gradient(trajectories, policy, gamma=0.99)
    # RichResult is dict-like
    assert isinstance(result, dict)
    # Keys named in the docstring's return section
    assert "gradient" in result
    assert "returns" in result
    assert "mean_return" in result
    assert "n_steps" in result
    assert "method" in result
    # gradient length matches the policy gradient size
    grad = result["gradient"]
    assert len(grad) == 3
    # n_steps is the total number of (state, action, reward) steps
    assert result["n_steps"] == 5
    # returns array has one entry per step
    assert len(result["returns"]) == 5
    # mean_return is a finite number
    assert math.isfinite(result["mean_return"])


def test_hmpg_edge():
    """Test edge cases."""
    def policy(state, action):
        return np.array([1.0])
    # An empty trajectory list is explicitly rejected by the docstring
    with pytest.raises(ValueError):
        geron_policy_gradient([], policy, gamma=1.0)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmpg as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
