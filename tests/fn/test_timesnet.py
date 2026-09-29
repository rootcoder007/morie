"""Tests for timesnet.timesnet."""

from morie.fn import _array_core as np
from morie.fn.timesnet import timesnet


def test_timesnet_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = timesnet(x)
    assert isinstance(result, dict)
    assert "frequency" in result


def test_timesnet_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = timesnet(x)
    assert isinstance(result, dict)


def test_dominant_periods_from_the_dft():
    import math

    import pytest

    n = 12
    x = [math.sin(2 * math.pi * 3 * t / n) + 0.4 * math.cos(2 * math.pi * 2 * t / n) for t in range(n)]
    amps = []
    for k in range(n):
        re = sum(x[t] * math.cos(-2 * math.pi * k * t / n) for t in range(n))
        im = sum(x[t] * math.sin(-2 * math.pi * k * t / n) for t in range(n))
        amps.append(math.hypot(re, im))
    r = timesnet(x, k=2)
    assert r["frequency"] == [3, 2]
    assert r["period"] == [4.0, 6.0]
    assert r["amplitude"] == pytest.approx([amps[3], amps[2]], rel=1e-12)
