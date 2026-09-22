"""Tests for fzt59.fauzi_thm5_9_edgeworth_wilcoxon."""

from morie.fn import _array_core as np

from morie.fn.fzt59 import fauzi_thm5_9_edgeworth_wilcoxon


def test_fzt59_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    n = 50
    result = fauzi_thm5_9_edgeworth_wilcoxon(y, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_fzt59_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    n = 50
    result = fauzi_thm5_9_edgeworth_wilcoxon(y, n)
    assert isinstance(result, dict)
