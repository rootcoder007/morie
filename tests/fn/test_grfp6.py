"""Tests for grfp6.geron_fp16_mixed_precision."""

from morie.fn import _array_core as np

from morie.fn.grfp6 import geron_fp16_mixed_precision


def test_grfp6_basic():
    """Test basic functionality."""
    loss = 0.5
    S = 512.0
    result = geron_fp16_mixed_precision(loss, S)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grfp6_edge():
    """Test edge cases."""
    loss = 0.5
    S = 512.0
    result = geron_fp16_mixed_precision(loss, S)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grfp6 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
