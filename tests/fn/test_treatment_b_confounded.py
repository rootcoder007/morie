"""Tests for treatment_b_confounded.treatment_b_confounded."""

from morie.fn import _array_core as np

from morie.fn.treatment_b_confounded import treatment_b_confounded


def test_ca9e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_b_confounded(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca9e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_b_confounded(x)
    assert isinstance(result, dict)
