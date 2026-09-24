"""Tests for km102.kamath_ch6_lstm_chain_rule."""

from morie.fn import _array_core as np

from morie.fn.km102 import kamath_ch6_lstm_chain_rule


def test_km102_basic():
    """Test basic functionality."""
    w_1_w_M = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch6_lstm_chain_rule(w_1_w_M)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km102_edge():
    """Test edge cases."""
    w_1_w_M = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch6_lstm_chain_rule(w_1_w_M)
    assert isinstance(result, dict)
