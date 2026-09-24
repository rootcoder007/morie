"""Tests for km115.kamath_ch8_bleu_n_geom_mean."""

from morie.fn import _array_core as np

from morie.fn.km115 import kamath_ch8_bleu_n_geom_mean


def test_km115_basic():
    """Test basic functionality."""
    p_n = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch8_bleu_n_geom_mean(p_n)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km115_edge():
    """Test edge cases."""
    p_n = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch8_bleu_n_geom_mean(p_n)
    assert isinstance(result, dict)
