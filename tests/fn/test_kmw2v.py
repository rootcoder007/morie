"""Tests for kmw2v.kamath_word2vec_skipgram."""

from morie.fn import _array_core as np

from morie.fn.kmw2v import kamath_word2vec_skipgram


def test_kmw2v_basic():
    """Test basic functionality."""
    center_indices = np.random.default_rng(42).normal(0, 1, 100)
    context_indices = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_word2vec_skipgram(center_indices, context_indices, V, U)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmw2v_edge():
    """Test edge cases."""
    center_indices = np.random.default_rng(42).normal(0, 1, 100)
    context_indices = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_word2vec_skipgram(center_indices, context_indices, V, U)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmw2v as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
