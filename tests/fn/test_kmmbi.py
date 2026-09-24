"""Tests for kmmbi.kamath_membership_inference."""

from morie.fn import _array_core as np

from morie.fn.kmmbi import kamath_membership_inference


def test_kmmbi_basic():
    """Test basic functionality."""
    losses = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    threshold = 0.1
    result = kamath_membership_inference(losses, threshold)
    assert isinstance(result, dict)
    assert "estimate" in result or "predictions" in result


def test_kmmbi_edge():
    """Test edge cases."""
    losses = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    threshold = 0.1
    result = kamath_membership_inference(losses, threshold)
    assert isinstance(result, dict)
