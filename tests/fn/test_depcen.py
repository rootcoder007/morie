"""Tests for depcen.dependent_censoring_hazard."""

from morie.fn import _array_core as np

from morie.fn.depcen import dependent_censoring_hazard


def test_depcen_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dependent_censoring_hazard(time, event, X)
    assert isinstance(result, dict)
    assert "beta_censoring" in result
def test_depcen_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dependent_censoring_hazard(time, event, X)
    assert isinstance(result, dict)
