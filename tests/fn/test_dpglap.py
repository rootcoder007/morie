"""Tests for dpglap.dp_laplace_mechanism."""

from morie.fn import _array_core as np

from morie.fn.dpglap import dp_laplace_mechanism


def test_dpglap_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = dp_laplace_mechanism(y)
    assert isinstance(result, dict)
    assert "release" in result
def test_dpglap_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = dp_laplace_mechanism(y)
    assert isinstance(result, dict)
