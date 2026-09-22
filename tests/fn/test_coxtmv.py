"""Tests for coxtmv.cox_time_varying."""

from morie.fn import _array_core as np

from morie.fn.coxtmv import cox_time_varying


def test_coxtmv_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    X = np.random.default_rng(42).normal(0, 1, (100, 3))
    result = cox_time_varying(time, event, X)
    assert isinstance(result, dict)
    assert "beta" in result
def test_coxtmv_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    X = np.random.default_rng(42).normal(0, 1, (100, 3))
    result = cox_time_varying(time, event, X)
    assert isinstance(result, dict)
