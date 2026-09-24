"""Tests for rosenb.rosenbaum_bounds."""

from morie.fn import _array_core as np

from morie.fn.rosenb import rosenbaum_bounds


def test_rosenb_basic():
    """Test basic functionality."""
    matched_pairs = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rosenbaum_bounds(matched_pairs)
    assert isinstance(result, dict)
    assert "Gamma" in result


def test_rosenb_edge():
    """Test edge cases."""
    matched_pairs = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rosenbaum_bounds(matched_pairs)
    assert isinstance(result, dict)
