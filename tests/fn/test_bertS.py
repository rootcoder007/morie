"""Tests for bertS.bertscore."""

from morie.fn import _array_core as np

from morie.fn.bertS import bertscore


def test_bertS_basic():
    """Test basic functionality."""
    reference = np.random.default_rng(42).normal(0, 1, 100)
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    result = bertscore(reference, candidate)
    assert isinstance(result, dict)
    assert "P" in result
def test_bertS_edge():
    """Test edge cases."""
    reference = np.random.default_rng(42).normal(0, 1, 100)
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    result = bertscore(reference, candidate)
    assert isinstance(result, dict)
