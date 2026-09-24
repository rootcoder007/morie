"""Tests for grmha.geron_multi_head_attention."""

from morie.fn import _array_core as np

from morie.fn.grmha import geron_multi_head_attention


def test_grmha_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    WQ = np.random.default_rng(42).normal(0, 1, 100)
    WK = np.random.default_rng(42).normal(0, 1, 100)
    WV = np.random.default_rng(42).normal(0, 1, 100)
    WO = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = geron_multi_head_attention(Q, K, V, WQ, WK, WV, WO, h)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grmha_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    WQ = np.random.default_rng(42).normal(0, 1, 100)
    WK = np.random.default_rng(42).normal(0, 1, 100)
    WV = np.random.default_rng(42).normal(0, 1, 100)
    WO = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = geron_multi_head_attention(Q, K, V, WQ, WK, WV, WO, h)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grmha as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
