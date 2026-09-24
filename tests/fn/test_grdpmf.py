"""Tests for grdpmf.geron_ddpm_forward_process."""

from morie.fn import _array_core as np

from morie.fn.grdpmf import geron_ddpm_forward_process


def test_grdpmf_basic():
    """Test basic functionality."""
    x0 = [2.0, -4.0]
    t = 1
    alpha_bar = [1.0, 0.36]
    result = geron_ddpm_forward_process(x0, t, alpha_bar)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grdpmf_edge():
    """Test edge cases."""
    x0 = [2.0, -4.0]
    t = 1
    alpha_bar = [1.0, 0.36]
    result = geron_ddpm_forward_process(x0, t, alpha_bar)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grdpmf as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
