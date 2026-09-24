"""Tests for kmrwkv.kamath_rwkv_time_mix."""

from morie.fn import _array_core as np

from morie.fn.kmrwkv import kamath_rwkv_time_mix


def test_kmrwkv_basic():
    """Test basic functionality."""
    k = 0.5
    v = 0.5
    w = 0.5
    result = kamath_rwkv_time_mix(k, v, w)
    assert isinstance(result, dict)
    assert "estimate" in result or "wkv" in result


def test_kmrwkv_edge():
    """Test edge cases."""
    k = 0.5
    v = 0.5
    w = 0.5
    result = kamath_rwkv_time_mix(k, v, w)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmrwkv as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
