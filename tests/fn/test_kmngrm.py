"""Tests for kmngrm.kamath_ngram_language_model."""

from morie.fn import _array_core as np

from morie.fn.kmngrm import kamath_ngram_language_model


def test_kmngrm_basic():
    """Test basic functionality."""
    counts_ngram = 0.5
    counts_prefix = 0.5
    result = kamath_ngram_language_model(counts_ngram, counts_prefix)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kmngrm_edge():
    """Test edge cases."""
    counts_ngram = 0.5
    counts_prefix = 0.5
    result = kamath_ngram_language_model(counts_ngram, counts_prefix)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmngrm as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
