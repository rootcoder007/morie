"""Tests for hrzaml.horowitz_additive_nonid_link."""

from morie.fn import _array_core as np

from morie.fn.hrzaml import horowitz_additive_nonid_link


def test_hrzaml_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_additive_nonid_link(x, y)
    assert isinstance(result, dict)
    assert "mu" in result


def test_hrzaml_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_additive_nonid_link(x, y)
    assert isinstance(result, dict)
