"""Tests for hmsent.geron_sentiment_analysis."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.hmsent import geron_sentiment_analysis


def test_hmsent_basic():
    """Test basic functionality."""
    # Simple lexicon-based model that returns 2-class scores per token sequence.
    pos = {"good", "great", "excellent", "wonderful"}
    neg = {"bad", "awful", "terrible", "horrible"}

    def model(tokens):
        s = sum(t in pos for t in tokens) - sum(t in neg for t in tokens)
        return [-s, s]

    texts = ["good great", "awful bad", "good terrible", "excellent wonderful"]
    y_true = [1, 0, 1, 1]

    result = geron_sentiment_analysis(texts, model, y_true=y_true)

    assert isinstance(result, dict)
    # Keys named in the return statement / docstring.
    assert "probabilities" in result
    assert "predicted" in result
    assert "accuracy" in result
    assert "confusion" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result
    assert "macro_f1" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    # Shape sanity: one prediction per document.
    assert len(result["predicted"]) == len(texts)
    # Accuracy is a real number in [0, 1].
    acc = float(result["accuracy"])
    assert math.isfinite(acc)
    assert 0.0 <= acc <= 1.0


def test_hmsent_edge():
    """Test edge cases."""
    pos = {"good"}
    neg = {"bad"}

    def model(tokens):
        s = sum(t in pos for t in tokens) - sum(t in neg for t in tokens)
        return [-s, s]

    # The docstring states texts must be non-empty; empty input is invalid.
    with pytest.raises(ValueError):
        geron_sentiment_analysis([], model)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmsent as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
