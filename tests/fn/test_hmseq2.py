"""Tests for hmseq2.geron_seq2seq."""

from morie.fn import _array_core as np

from morie.fn.hmseq2 import geron_seq2seq


def test_hmseq2_basic():
    """Test basic functionality."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_seq2seq(src, tgt, encoder, decoder)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmseq2_edge():
    """Test edge cases."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_seq2seq(src, tgt, encoder, decoder)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmseq2 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
