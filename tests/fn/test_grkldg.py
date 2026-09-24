"""Tests for grkldg.geron_kl_divergence_gaussian."""

from morie.fn import _array_core as np

from morie.fn.grkldg import geron_kl_divergence_gaussian


def test_grkldg_basic():
    """Test basic functionality."""
    mu = [0.5, -1.0]
    logvar = [0.2, -0.3]
    result = geron_kl_divergence_gaussian(mu, logvar)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grkldg_edge():
    """Test edge cases."""
    mu = [0.5, -1.0]
    logvar = [0.2, -0.3]
    result = geron_kl_divergence_gaussian(mu, logvar)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grkldg as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
