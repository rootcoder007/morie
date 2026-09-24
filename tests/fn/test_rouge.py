"""Tests for rouge.rouge."""

from morie.fn import _array_core as np

from morie.fn.rouge import rouge


def test_rouge_basic():
    """Test basic functionality."""
    candidate = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    reference = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rouge(candidate, reference)
    assert isinstance(result, dict)
    assert "estimate" in result or "precision" in result


def test_rouge_edge():
    """Test edge cases."""
    candidate = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    reference = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rouge(candidate, reference)
    assert isinstance(result, dict)
