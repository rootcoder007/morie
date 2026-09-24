"""Tests for grsdpa.geron_scaled_dot_product_attention."""

from morie.fn import _array_core as np

from morie.fn.grsdpa import geron_scaled_dot_product_attention


def test_grsdpa_basic():
    """Test basic functionality."""
    Q = [[1.0, 0.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 0.0], [0.0, 1.0]]
    result = geron_scaled_dot_product_attention(Q, K, V)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grsdpa_edge():
    """Test edge cases."""
    Q = [[1.0, 0.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 0.0], [0.0, 1.0]]
    result = geron_scaled_dot_product_attention(Q, K, V)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grsdpa as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
