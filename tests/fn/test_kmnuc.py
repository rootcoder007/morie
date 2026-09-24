"""Tests for kmnuc.kamath_nucleus_sampling."""

from morie.fn import _array_core as np

from morie.fn.kmnuc import kamath_nucleus_sampling


def test_kmnuc_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    p = 0.1
    result = kamath_nucleus_sampling(logits, p)
    assert isinstance(result, dict)
    assert "estimate" in result or "probabilities" in result


def test_kmnuc_edge():
    """Test edge cases."""
    logits = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    p = 0.1
    result = kamath_nucleus_sampling(logits, p)
    assert isinstance(result, dict)
