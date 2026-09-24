"""Tests for sppois.schabenberger_poisson_process."""

from morie.fn import _array_core as np

from morie.fn.sppois import schabenberger_poisson_process


def test_sppois_basic():
    """Test basic functionality."""
    result = schabenberger_poisson_process()
    assert isinstance(result, dict)
    assert "points" in result


def test_sppois_edge():
    """Test edge cases."""
    result = schabenberger_poisson_process()
    assert isinstance(result, dict)
