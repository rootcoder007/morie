"""Tests for grmcol.geron_gan_mode_collapse_metric."""

from morie.fn import _array_core as np

from morie.fn.grmcol import geron_gan_mode_collapse_metric


def test_grmcol_basic():
    """Test basic functionality."""
    samples = [[0.05], [-0.05], [10.1], [5.0]]
    true_modes = [[0.0], [10.0], [20.0]]
    result = geron_gan_mode_collapse_metric(samples, true_modes)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grmcol_edge():
    """Test edge cases."""
    samples = [[0.05], [-0.05], [10.1], [5.0]]
    true_modes = [[0.0], [10.0], [20.0]]
    result = geron_gan_mode_collapse_metric(samples, true_modes)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grmcol as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
