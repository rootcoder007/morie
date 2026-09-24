"""Tests for hmrnfc.geron_reinforce."""

import math

from morie.fn import _array_core as np

from morie.fn.hmrnfc import geron_reinforce


def test_hmrnfc_basic():
    """Test basic functionality with a single episode."""
    def policy(state, action):
        if action == 0:
            return np.array([1.0, 0.0])
        return np.array([0.0, 1.0])

    episodes = [[(0, 0, 1.0), (1, 1, 1.0)]]
    result = geron_reinforce(episodes, policy, gamma=0.5, eta=0.1)
    assert isinstance(result, dict)
    # Verify the keys the docstring / return statement advertises.
    for key in ("theta", "step", "gradient"):
        assert key in result
    # The gradient is 2-D (one component per "action indicator" basis vector).
    grad = result["gradient"]
    assert len(grad) == 2
    theta = result["theta"]
    assert len(theta) == 2
    for v in theta:
        assert math.isfinite(float(v))


def test_hmrnfc_edge():
    """Test edge case: multiple episodes, baseline off, with a custom theta."""
    def policy(state, action):
        if action == 0:
            return np.array([1.0, 0.0])
        return np.array([0.0, 1.0])

    episodes = [
        [(0, 0, 1.0), (1, 1, 2.0)],
        [(0, 0, 0.5), (1, 1, 1.5)],
    ]
    theta_init = np.zeros(2)
    result = geron_reinforce(
        episodes, policy, gamma=0.9, eta=0.05,
        theta=theta_init, baseline=False,
    )
    assert isinstance(result, dict)
    for key in ("theta", "step", "gradient"):
        assert key in result
    grad = result["gradient"]
    assert len(grad) == 2
    theta = result["theta"]
    assert len(theta) == 2
    for v in theta:
        assert math.isfinite(float(v))


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmrnfc as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
