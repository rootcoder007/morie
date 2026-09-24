"""Tests for hrzsmsrc.horowitz_sms_rate."""

from morie.fn import _array_core as np

from morie.fn.hrzsmsrc import horowitz_sms_rate


def test_hrzsmsrc_basic():
    """Test basic functionality."""
    n = 5
    result = horowitz_sms_rate(n)
    assert isinstance(result, dict)
    assert "rate" in result


def test_hrzsmsrc_edge():
    """Test edge cases."""
    n = 5
    result = horowitz_sms_rate(n)
    assert isinstance(result, dict)
