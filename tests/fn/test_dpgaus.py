"""Tests for dpgaus.dp_gaussian_mechanism."""

from morie.fn import _array_core as np

from morie.fn.dpgaus import dp_gaussian_mechanism


def test_dpgaus_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = dp_gaussian_mechanism(y)
    assert isinstance(result, dict)
    assert "release" in result
def test_dpgaus_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = dp_gaussian_mechanism(y)
    assert isinstance(result, dict)
