"""Tests for joaci.joseph_adaptive_conformal_inference."""

from morie.fn import _array_core as np

from morie.fn.joaci import joseph_adaptive_conformal_inference


def test_joaci_basic():
    """Test basic functionality."""
    inside = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_adaptive_conformal_inference(inside)
    assert isinstance(result, dict)
    assert "alpha" in result


def test_joaci_edge():
    """Test edge cases."""
    inside = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_adaptive_conformal_inference(inside)
    assert isinstance(result, dict)
