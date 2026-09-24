"""Tests for hmarma.geron_arma."""

import doctest as _doctest

import pytest

import morie.fn.hmarma as _doctest_module
from morie.fn import _array_core as np
from morie.fn.hmarma import geron_arma


def test_hmarma_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    y = rng_y.normal(0, 1, 100)
    p = 3
    q = 2
    result = geron_arma(y, p, q)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmarma_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    y = rng_y.normal(0, 1, 60)
    p = 1
    q = 1
    result = geron_arma(y, p, q)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
