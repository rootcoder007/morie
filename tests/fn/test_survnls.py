"""Tests for survnls.nonlinear_least_squares_surv."""

from morie.fn import _array_core as np

from morie.fn.survnls import nonlinear_least_squares_surv


def test_survnls_basic():
    """Test basic functionality."""
    time = np.array([float(i + 1) for i in range(40)])
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = nonlinear_least_squares_surv(time, event)
    assert isinstance(result, dict)
    assert "params" in result


def test_survnls_edge():
    """Test edge cases."""
    time = np.array([float(i + 1) for i in range(40)])
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = nonlinear_least_squares_surv(time, event)
    assert isinstance(result, dict)
