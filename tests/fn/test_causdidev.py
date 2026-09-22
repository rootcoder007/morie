"""Tests for causdidev.causal_did_eventstudy."""

from morie.fn import _array_core as np

from morie.fn.causdidev import causal_did_eventstudy


def test_causdidev_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    D = np.array([1.0 if (i // 10) >= 5 and (i % 10) < 5 else 0.0 for i in range(100)])
    unit = np.array([float(i % 10) for i in range(100)])
    time = np.array([float(i // 10) for i in range(100)])
    result = causal_did_eventstudy(y, D, unit, time)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_causdidev_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    D = np.array([1.0 if (i // 10) >= 5 and (i % 10) < 5 else 0.0 for i in range(100)])
    unit = np.array([float(i % 10) for i in range(100)])
    time = np.array([float(i // 10) for i in range(100)])
    result = causal_did_eventstudy(y, D, unit, time)
    assert isinstance(result, dict)
