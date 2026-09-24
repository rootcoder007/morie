"""Tests for powsrv.power_survey."""

from morie.fn import _array_core as np

from morie.fn.powsrv import power_survey


def test_powsrv_basic():
    """Test basic functionality."""
    effect_size = 0.1
    result = power_survey(effect_size)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_powsrv_edge():
    """Test edge cases."""
    effect_size = 0.1
    result = power_survey(effect_size)
    assert isinstance(result, dict)
