"""Tests for grrein.geron_reinforce_policy_gradient."""

from morie.fn import _array_core as np

from morie.fn.grrein import geron_reinforce_policy_gradient


def test_grrein_basic():
    """Test basic functionality."""
    theta = [0.0, 0.0]
    log_probs = [[1.0, 0.0], [0.0, 1.0]]
    returns_G = [2.0, -1.0]
    alpha = 0.5
    result = geron_reinforce_policy_gradient(theta, log_probs, returns_G, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grrein_edge():
    """Test edge cases."""
    theta = [0.0, 0.0]
    log_probs = [[1.0, 0.0], [0.0, 1.0]]
    returns_G = [2.0, -1.0]
    alpha = 0.5
    result = geron_reinforce_policy_gradient(theta, log_probs, returns_G, alpha)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grrein as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
