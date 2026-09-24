"""Tests for grlrco.geron_lr_cosine_annealing."""

from morie.fn import _array_core as np

from morie.fn.grlrco import geron_lr_cosine_annealing


def test_grlrco_basic():
    """Test basic functionality."""
    eta_min = 0
    eta_max = 100
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_lr_cosine_annealing(eta_min, eta_max, t, T)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grlrco_edge():
    """Test edge cases."""
    eta_min = 0
    eta_max = 100
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_lr_cosine_annealing(eta_min, eta_max, t, T)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grlrco as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
