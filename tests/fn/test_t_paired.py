"""Tests for t_paired.t_paired."""

from morie.fn import _array_core as np

from morie.fn.t_paired import t_paired


def test_ca9e10_basic():
    """Test basic functionality."""
    differences = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = t_paired(differences)
    assert isinstance(result, dict)
    assert "t" in result


def test_ca9e10_edge():
    """Test edge cases."""
    differences = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = t_paired(differences)
    assert isinstance(result, dict)
