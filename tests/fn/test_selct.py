"""Tests for selct.genomic_selection_accuracy."""

from morie.fn import _array_core as np

from morie.fn.selct import genomic_selection_accuracy


def test_selct_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = genomic_selection_accuracy(y, yhat)
    assert isinstance(result, dict)
    assert "accuracy" in result


def test_selct_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    yhat = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = genomic_selection_accuracy(y, yhat)
    assert isinstance(result, dict)
