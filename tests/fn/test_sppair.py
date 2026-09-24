"""Tests for sppair.schabenberger_pair_correlation."""

from morie.fn import _array_core as np

from morie.fn.sppair import schabenberger_pair_correlation


def test_sppair_basic():
    """Test basic functionality."""
    points = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = schabenberger_pair_correlation(points)
    assert isinstance(result, dict)
    assert "r" in result


def test_sppair_edge():
    """Test edge cases."""
    points = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = schabenberger_pair_correlation(points)
    assert isinstance(result, dict)
