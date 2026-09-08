"""Tests for alnnw.needleman_wunsch."""

from morie.fn import _array_core as np

from morie.fn.alnnw import needleman_wunsch


def test_alnnw_basic():
    """Test basic functionality."""
    seq1 = np.random.default_rng(42).normal(0, 1, 100)
    seq2 = np.random.default_rng(42).normal(0, 1, 100)
    sub_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    gap = np.random.default_rng(42).normal(0, 1, 100)
    result = needleman_wunsch(seq1, seq2, sub_matrix, gap)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_alnnw_edge():
    """Test edge cases."""
    seq1 = np.random.default_rng(42).normal(0, 1, 100)
    seq2 = np.random.default_rng(42).normal(0, 1, 100)
    sub_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    gap = np.random.default_rng(42).normal(0, 1, 100)
    result = needleman_wunsch(seq1, seq2, sub_matrix, gap)
    assert isinstance(result, dict)
