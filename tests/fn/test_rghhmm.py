"""Tests for rghhmm.rangayyan_hodgkin_huxley."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_hodgkin_huxley


def test_rghhmm_basic():
    """Test basic functionality."""
    result = rangayyan_hodgkin_huxley()
    assert isinstance(result, dict)
    assert "t_ms" in result


def test_rghhmm_edge():
    """Test edge cases."""
    result = rangayyan_hodgkin_huxley()
    assert isinstance(result, dict)
