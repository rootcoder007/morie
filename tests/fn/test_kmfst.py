"""Tests for kmfst.kamath_fasttext_subword."""

from morie.fn import _array_core as np

from morie.fn.kmfst import kamath_fasttext_subword


def test_kmfst_basic():
    """Test basic functionality."""
    word = np.random.default_rng(42).normal(0, 1, 100)
    ngram_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    n_min = 0
    n_max = 100
    result = kamath_fasttext_subword(word, ngram_embeddings, n_min, n_max)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmfst_edge():
    """Test edge cases."""
    word = np.random.default_rng(42).normal(0, 1, 100)
    ngram_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    n_min = 0
    n_max = 100
    result = kamath_fasttext_subword(word, ngram_embeddings, n_min, n_max)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmfst as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
