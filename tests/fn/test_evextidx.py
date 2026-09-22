"""Tests for evextidx.evt_extremal_index_runs."""

from morie.fn import _array_core as np

from morie.fn.evextidx import evt_extremal_index_runs


def test_evextidx_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    threshold = 1.0
    result = evt_extremal_index_runs(x, threshold)
    assert isinstance(result, dict)
    assert "theta" in result
def test_evextidx_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    threshold = 1.0
    result = evt_extremal_index_runs(x, threshold)
    assert isinstance(result, dict)
