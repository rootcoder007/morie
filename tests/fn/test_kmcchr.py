"""Tests for kmcchr.kamath_christiano_deep_rl_feedback."""

from morie.fn import _array_core as np

from morie.fn.kmcchr import kamath_christiano_deep_rl_feedback


def test_kmcchr_basic():
    """Test basic functionality."""
    trajectory_pairs = [(2.0, 0.0), (0.0, 1.0)]
    r_phi = lambda s: s
    result = kamath_christiano_deep_rl_feedback(trajectory_pairs, r_phi)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmcchr_edge():
    """Test edge cases."""
    trajectory_pairs = [(2.0, 0.0), (0.0, 1.0)]
    r_phi = lambda s: s
    result = kamath_christiano_deep_rl_feedback(trajectory_pairs, r_phi)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmcchr as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
