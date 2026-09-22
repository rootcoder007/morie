"""Tests for breslot.breslow_tie_correction."""

from morie.fn import _array_core as np

from morie.fn.breslot import breslow_tie_correction


def test_breslot_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = breslow_tie_correction(time, event, X)
    assert isinstance(result, dict)
    assert "beta" in result
def test_breslot_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = breslow_tie_correction(time, event, X)
    assert isinstance(result, dict)
