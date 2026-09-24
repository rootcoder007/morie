"""Tests for km098.kamath_ch6_log_prob_ratio_attr."""

from morie.fn import _array_core as np

from morie.fn.km098 import kamath_ch6_log_prob_ratio_attr


def test_km098_basic():
    """Test basic functionality."""
    a_i = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    a_j = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch6_log_prob_ratio_attr(a_i, a_j)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km098_edge():
    """Test edge cases."""
    a_i = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    a_j = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch6_log_prob_ratio_attr(a_i, a_j)
    assert isinstance(result, dict)
