"""Tests for cumulative_logit.cumulative_logit."""

from morie.fn import _array_core as np

from morie.fn.cumulative_logit import cumulative_logit


def test_ca5e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cumulative_logit(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca5e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cumulative_logit(x)
    assert isinstance(result, dict)
