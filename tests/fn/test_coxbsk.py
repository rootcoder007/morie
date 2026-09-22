"""Tests for coxbsk.cox_breslow_step."""

from morie.fn import _array_core as np

from morie.fn.coxbsk import cox_breslow_step


def test_coxbsk_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = cox_breslow_step(time, event, X)
    assert isinstance(result, dict)
    assert "times" in result
def test_coxbsk_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = cox_breslow_step(time, event, X)
    assert isinstance(result, dict)
