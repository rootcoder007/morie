"""Tests for hmrl.geron_reinforcement_learning."""

from morie.fn import _array_core as np

from morie.fn.hmrl import geron_reinforcement_learning


def test_hmrl_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_reinforcement_learning(env, pi, gamma)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmrl_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_reinforcement_learning(env, pi, gamma)
    assert isinstance(result, dict)


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
