"""Tests for spbino.schabenberger_binomial_process."""

from morie.fn.spbino import schabenberger_binomial_process


def test_spbino_basic():
    """Test basic functionality."""
    result = schabenberger_binomial_process()
    assert isinstance(result, dict)
    assert "points" in result


def test_spbino_edge():
    """Test edge cases."""
    result = schabenberger_binomial_process()
    assert isinstance(result, dict)
