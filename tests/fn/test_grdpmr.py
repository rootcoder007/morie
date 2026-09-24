"""Tests for grdpmr.geron_ddpm_reverse_step."""

from morie.fn import _array_core as np

from morie.fn.grdpmr import geron_ddpm_reverse_step


def test_grdpmr_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    eps_pred = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    alpha_bar = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = geron_ddpm_reverse_step(x_t, t, eps_pred, alpha, alpha_bar, sigma)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grdpmr_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    eps_pred = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    alpha_bar = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
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
