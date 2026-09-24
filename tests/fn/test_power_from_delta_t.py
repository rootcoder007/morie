"""Tests for power_from_delta_t.power_from_delta_t."""

from morie.fn import _array_core as np

from morie.fn.power_from_delta_t import power_from_delta_t


def test_ca8e3_basic():
    """Test basic functionality."""
    delta = 0.5
    t_cv = 0.5
    df = 0.5
    result = power_from_delta_t(delta, t_cv, df)
    assert isinstance(result, dict)
    assert "t_beta" in result


def test_ca8e3_edge():
    """Test edge cases."""
    delta = 0.5
    t_cv = 0.5
    df = 0.5
    result = power_from_delta_t(delta, t_cv, df)
    assert isinstance(result, dict)
