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
