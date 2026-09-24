"""Tests for grppo.geron_ppo_clipped_objective."""

from morie.fn import _array_core as np

from morie.fn.grppo import geron_ppo_clipped_objective


def test_grppo_basic():
    """Test basic functionality."""
    ratios = [1.5, 0.5]
    advantages = [1.0, -1.0]
    result = geron_ppo_clipped_objective(ratios, advantages)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grppo_edge():
    """Test edge cases."""
    ratios = [1.5, 0.5]
    advantages = [1.0, -1.0]
    result = geron_ppo_clipped_objective(ratios, advantages)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grppo as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
