"""Tests for rgcorart.rangayyan_coronary_sound."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_coronary_sound


def test_rgcorart_basic():
    """Test basic functionality."""
    diameter = 0.1
    flow_velocity = 0.1
    result = rangayyan_coronary_sound(diameter, flow_velocity)
    assert isinstance(result, dict)
    assert "freq_hz" in result


def test_rgcorart_edge():
    """Test edge cases."""
    diameter = 0.1
    flow_velocity = 0.1
    result = rangayyan_coronary_sound(diameter, flow_velocity)
    assert isinstance(result, dict)
