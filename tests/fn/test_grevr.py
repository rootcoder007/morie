"""Tests for grevr.geron_explained_variance_ratio."""

from morie.fn import _array_core as np

from morie.fn.grevr import geron_explained_variance_ratio


def test_grevr_basic():
    """Test basic functionality."""
    singular_values = [3.0, 4.0]
    result = geron_explained_variance_ratio(singular_values)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grevr_edge():
    """Test edge cases."""
    singular_values = [3.0, 4.0]
    result = geron_explained_variance_ratio(singular_values)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grevr as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
