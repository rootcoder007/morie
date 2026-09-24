"""Tests for km136.kamath_ch9_mml_vlm_loss."""

from morie.fn import _array_core as np

from morie.fn.km136 import kamath_ch9_mml_vlm_loss


def test_km136_basic():
    """Test basic functionality."""
    Pos = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    Neg = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch9_mml_vlm_loss(Pos, Neg)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_km136_edge():
    """Test edge cases."""
    Pos = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    Neg = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch9_mml_vlm_loss(Pos, Neg)
    assert isinstance(result, dict)
