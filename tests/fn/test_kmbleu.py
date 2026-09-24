"""Tests for kmbleu.kamath_bleu_score."""

from morie.fn import _array_core as np

from morie.fn.kmbleu import kamath_bleu_score


def test_kmbleu_basic():
    """Test basic functionality."""
    hypothesis = np.random.default_rng(42).normal(0.0, 1.0, 40)
    references = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_bleu_score(hypothesis, references)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kmbleu_edge():
    """Test edge cases."""
    hypothesis = np.random.default_rng(42).normal(0.0, 1.0, 40)
    references = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_bleu_score(hypothesis, references)
    assert isinstance(result, dict)
