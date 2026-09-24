"""Tests for grkvc.geron_kv_cache_compression."""

from morie.fn import _array_core as np

from morie.fn.grkvc import geron_kv_cache_compression


def test_grkvc_basic():
    """Test basic functionality."""
    seq_len = 100
    num_layers = np.random.default_rng(42).normal(0, 1, 100)
    num_heads = np.random.default_rng(42).normal(0, 1, 100)
    d_head = np.random.default_rng(42).normal(0, 1, 100)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_kv_cache_compression(seq_len, num_layers, num_heads, d_head, bits)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grkvc_edge():
    """Test edge cases."""
    seq_len = 100
    num_layers = np.random.default_rng(42).normal(0, 1, 100)
    num_heads = np.random.default_rng(42).normal(0, 1, 100)
    d_head = np.random.default_rng(42).normal(0, 1, 100)
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_kv_cache_compression(seq_len, num_layers, num_heads, d_head, bits)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grkvc as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
