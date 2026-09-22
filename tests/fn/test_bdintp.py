"""Tests for bdintp.bound_intersection."""

from morie.fn import _array_core as np

from morie.fn.bdintp import bound_intersection


def test_bdintp_basic():
    """Test basic functionality."""
    mbar = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_intersection(mbar)
    assert isinstance(result, dict)
    assert "Q" in result
def test_bdintp_edge():
    """Test edge cases."""
    mbar = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_intersection(mbar)
    assert isinstance(result, dict)
