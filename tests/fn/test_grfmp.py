"""Tests for grfmp.geron_feature_map_dim."""

from morie.fn import _array_core as np

from morie.fn.grfmp import geron_feature_map_dim


def test_grfmp_basic():
    """Test basic functionality."""
    H_out = 150
    W_out = 100
    C_out = 200
    result = geron_feature_map_dim(H_out, W_out, C_out)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grfmp_edge():
    """Test edge cases."""
    H_out = 150
    W_out = 100
    C_out = 200
    result = geron_feature_map_dim(H_out, W_out, C_out)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grfmp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
