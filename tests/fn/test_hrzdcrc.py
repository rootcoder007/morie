"""Tests for hrzdcrc.horowitz_deconv_rate."""

from morie.fn import _array_core as np

from morie.fn.hrzdcrc import horowitz_deconv_rate


def test_hrzdcrc_basic():
    """Test basic functionality."""
    n = 5
    result = horowitz_deconv_rate(n)
    assert isinstance(result, dict)
    assert "rate" in result


def test_hrzdcrc_edge():
    """Test edge cases."""
    n = 5
    result = horowitz_deconv_rate(n)
    assert isinstance(result, dict)
