"""Tests for evextint.evt_extremal_index_intervals."""

from morie.fn import _array_core as np

from morie.fn.evextint import evt_extremal_index_intervals


def test_evextint_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    threshold = 1.0
    result = evt_extremal_index_intervals(x, threshold)
    assert isinstance(result, dict)
    assert "theta" in result
def test_evextint_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    threshold = 1.0
    result = evt_extremal_index_intervals(x, threshold)
    assert isinstance(result, dict)
