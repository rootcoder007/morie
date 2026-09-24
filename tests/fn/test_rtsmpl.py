"""Tests for rtsmpl.rt_serial_interval."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.rtsmpl import rt_serial_interval


def test_rtsmpl_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # Daily incidence counts (nonnegative integers)
    incidence = rng.integers(0, 20, 30)
    # Discrete serial-interval distribution, normalised to sum to 1
    serial_interval = rng.uniform(0.0, 1.0, 14)
    total = float(sum(serial_interval))
    serial_interval = [float(v) / total for v in serial_interval]
    window = 7
    result = rt_serial_interval(incidence, serial_interval, window)
    assert isinstance(result, dict)
    # The function documents these keys in its return value
    assert "r_mean" in result
    assert "r_std" in result
    assert "a_posterior" in result
    assert "b_posterior" in result
    assert "lambda" in result
    assert "t_start" in result
    assert "t_end" in result
    assert "n_windows" in result
    assert "n" in result
    assert "window" in result
    # Shape / length sanity
    n = len(incidence)
    assert result["n"] == n
    assert len(result["lambda"]) == n
    n_windows = n - window
    assert len(result["r_mean"]) == n_windows
    assert len(result["r_std"]) == n_windows
    assert len(result["t_start"]) == n_windows
    assert len(result["t_end"]) == n_windows
    assert result["n_windows"] == n_windows
    assert result["window"] == window
    # Every estimate and infectiousness value must be finite
    for v in result["r_mean"]:
        assert math.isfinite(v)
    for v in result["r_std"]:
        assert math.isfinite(v)
    for v in result["lambda"]:
        assert math.isfinite(v)
    for v in result["a_posterior"]:
        assert math.isfinite(v)
    for v in result["b_posterior"]:
        assert math.isfinite(v)


def test_rtsmpl_edge():
    """Test edge cases."""
    # An empty incidence sequence is documented as invalid.
    with pytest.raises(ValueError):
        rt_serial_interval([], [0.5, 0.5], 7)
