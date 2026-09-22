"""Tests for cssant.callaway_santanna."""

from morie.fn import _array_core as np

from morie.fn.cssant import callaway_santanna


def test_cssant_basic():
    """Test basic functionality."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([1.0 if (i // 10) >= 5 and (i % 10) < 5 else 0.0 for i in range(100)])
    unit = np.array([float(i % 10) for i in range(100)])
    time = np.array([float(i // 10) for i in range(100)])
    result = callaway_santanna(y, D, unit, time)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_cssant_edge():
    """Test edge cases."""
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    D = np.array([1.0 if (i // 10) >= 5 and (i % 10) < 5 else 0.0 for i in range(100)])
    unit = np.array([float(i % 10) for i in range(100)])
    time = np.array([float(i // 10) for i in range(100)])
    result = callaway_santanna(y, D, unit, time)
    assert isinstance(result, dict)
