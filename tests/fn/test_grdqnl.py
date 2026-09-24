"""Tests for grdqnl.geron_dqn_loss."""

from morie.fn import _array_core as np

from morie.fn.grdqnl import geron_dqn_loss


def test_grdqnl_basic():
    """Test basic functionality."""
    Q = [[0.5, 0.0], [0.0, 0.0]]
    Q_target = [[0.0, 0.0], [4.0, 1.0]]
    batch = [(0, 0, 1.0, 1, False)]
    result = geron_dqn_loss(Q, Q_target, batch)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grdqnl_edge():
    """Test edge cases."""
    Q = [[0.5, 0.0], [0.0, 0.0]]
    Q_target = [[0.0, 0.0], [4.0, 1.0]]
    batch = [(0, 0, 1.0, 1, False)]
    result = geron_dqn_loss(Q, Q_target, batch)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grdqnl as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
