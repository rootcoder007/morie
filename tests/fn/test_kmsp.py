"""Tests for kmsp.kamath_sentencepiece_tokenizer."""

from morie.fn import _array_core as np

import math

import pytest

from morie.fn.kmsp import kamath_sentencepiece_tokenizer


def test_kmsp_basic():
    """Test basic functionality."""
    corpus = ["abab", "abab", "abc"]
    vocab_size = 4
    result = kamath_sentencepiece_tokenizer(corpus, vocab_size)
    assert isinstance(result, dict)
    assert result["vocab_size"] == vocab_size
    assert "vocab" in result
    assert "log_likelihood" in result
    assert "segmentations" in result
    assert math.isfinite(result["log_likelihood"])
    assert result["log_likelihood"] <= 0
    chars = sorted({c for s in corpus for c in s})
    for c in chars:
        assert c in result["vocab"]
    assert len(result["segmentations"]) == len(corpus)
    for seg in result["segmentations"]:
        assert isinstance(seg, list)
        assert len(seg) > 0
        for piece in seg:
            assert isinstance(piece, str)


def test_kmsp_edge():
    """Test edge cases."""
    with pytest.raises(ValueError):
        kamath_sentencepiece_tokenizer([], 4)

    corpus = ["abab", "abab", "abc"]
    with pytest.raises(ValueError):
        kamath_sentencepiece_tokenizer(corpus, 2)

    with pytest.raises(ValueError):
        kamath_sentencepiece_tokenizer(corpus, 4, max_piece_len=1)

    with pytest.raises(ValueError):
        kamath_sentencepiece_tokenizer(corpus, 4, shrink=1.5)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmsp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
