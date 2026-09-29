"""Tests for joavg.joseph_average_forecast."""

from morie.fn import _array_core as np
from morie.fn.joavg import joseph_average_forecast


def test_joavg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_average_forecast(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_joavg_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_average_forecast(y)
    assert isinstance(result, dict)


def test_baselines_recomputed():
    import math

    import pytest

    y = [3.0, 4.0, 3.5, 5.0, 6.0, 5.5]
    T, h = 6, 3
    mu = sum(y) / T
    sd = math.sqrt(sum((v - mu) ** 2 for v in y) / (T - 1))
    r = joseph_average_forecast(y, horizon=h, seasonal_period=2)
    assert [float(v) for v in r["forecast"]] == pytest.approx([mu] * h, rel=1e-14)
    assert r["se"] == pytest.approx(sd * math.sqrt(1 + 1 / T), rel=1e-13)
    slope = (y[-1] - y[0]) / (T - 1)
    assert [float(v) for v in r["drift_forecast"]] == pytest.approx([5.5 + slope * k for k in (1, 2, 3)], rel=1e-14)
    assert [float(v) for v in r["seasonal_naive"]] == [6.0, 5.5, 6.0]
