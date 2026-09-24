"""Tests for grdpmr.geron_ddpm_reverse_step."""

from morie.fn import _array_core as np

from morie.fn.grdpmr import geron_ddpm_reverse_step


def test_grdpmr_basic():
    """Test basic functionality."""
    x_t = 0.5
    t = 0.5
    eps_pred = 0.5
    alpha = np.random.default_rng(42).normal(0.0, 1.0, 40)
    alpha_bar = 0.5
    sigma = 0.5
    result = geron_ddpm_reverse_step(x_t, t, eps_pred, alpha, alpha_bar, sigma)
    assert isinstance(result, dict)
    assert "estimate" in result or "x_prev" in result


def test_grdpmr_edge():
    """Test edge cases."""
    x_t = 0.5
    t = 0.5
    eps_pred = 0.5
    alpha = np.random.default_rng(42).normal(0.0, 1.0, 40)
    alpha_bar = 0.5
    sigma = 0.5
    result = geron_ddpm_reverse_step(x_t, t, eps_pred, alpha, alpha_bar, sigma)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grdpmr as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
