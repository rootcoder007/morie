"""Tests for hmkrn.geron_filter_kernel."""

from morie.fn import _array_core as np

from morie.fn.hmkrn import geron_filter_kernel


def test_hmkrn_basic():
    """Test basic functionality."""
    kh, kw, c_in, c_out = 3, 3, 4, 5
    result = geron_filter_kernel(kh, kw, c_in, c_out, seed=42)
    assert isinstance(result, dict)
    assert result["shape"] == (kh, kw, c_in, c_out)
    assert result["fan_in"] == kh * kw * c_in
    assert int(result["kernel"].size) == kh * kw * c_in * c_out
    assert int(result["bias"].size) == c_out
    assert result["n_parameters"] == kh * kw * c_in * c_out + c_out


def test_hmkrn_edge():
    """Test edge cases."""
    # 1x1 kernel minimal case (matches the docstring example).
    result = geron_filter_kernel(1, 1, 3, 2)
    assert isinstance(result, dict)
    assert result["shape"] == (1, 1, 3, 2)
    assert result["fan_in"] == 3
    assert result["n_parameters"] == 8


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmkrn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
