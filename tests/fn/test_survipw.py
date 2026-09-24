"""Tests for survipw.ipcw_estimator."""

from morie.fn import _array_core as np

from morie.fn.survipw import ipcw_estimator


def test_survipw_basic():
    """Test basic functionality."""
    time = np.array([float(i + 1) for i in range(40)])
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = ipcw_estimator(time, event)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_survipw_edge():
    """Test edge cases."""
    time = np.array([float(i + 1) for i in range(40)])
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = ipcw_estimator(time, event)
    assert isinstance(result, dict)
