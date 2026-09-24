"""Tests for grgmem.geron_gmm_em_step."""

from morie.fn import _array_core as np

from morie.fn.grgmem import geron_gmm_em_step


def test_grgmem_basic():
    """Test basic functionality."""
    X = [[0.0], [0.4], [5.0], [5.5]]
    pi = [0.5, 0.5]
    means = [[1.0], [4.0]]
    covars = [[[1.0]], [[1.0]]]
    result = geron_gmm_em_step(X, pi, means, covars)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grgmem_edge():
    """Test edge cases."""
    X = [[0.0], [0.4], [5.0], [5.5]]
    pi = [0.5, 0.5]
    means = [[1.0], [4.0]]
    covars = [[[1.0]], [[1.0]]]
    result = geron_gmm_em_step(X, pi, means, covars)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grgmem as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
