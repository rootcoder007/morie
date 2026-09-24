"""Tests for grsa.geron_self_attention."""

from morie.fn import _array_core as np

from morie.fn.grsa import geron_self_attention


def test_grsa_basic():
    """Test basic functionality."""
    X = np.array([[1.0, 2.0], [0.0, -1.0]])
    WQ = np.array([[1.0, 0.0], [0.5, 1.0]])
    WK = np.array([[0.0, 1.0], [1.0, 0.0]])
    WV = np.array([[2.0, 0.0], [0.0, 3.0]])
    result = geron_self_attention(X, WQ, WK, WV)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grsa_edge():
    """Test edge cases."""
    X = np.array([[1.0, 2.0], [0.0, -1.0]])
    WQ = np.array([[1.0, 0.0], [0.5, 1.0]])
    WK = np.array([[0.0, 1.0], [1.0, 0.0]])
    WV = np.array([[2.0, 0.0], [0.0, 3.0]])
    result = geron_self_attention(X, WQ, WK, WV)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grsa as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
