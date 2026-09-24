"""Tests for kpmsmp.km_simultaneous_band."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.kpmsmp import km_simultaneous_band


def _make_risk_table(n_times=5, n_start=10):
    """Build a valid risk-table mapping for km_simultaneous_band.

    The table has ``n_times`` event times with one event per time and a
    strictly decreasing number at risk, ensuring ``n_risk > n_event`` at
    every time so the Greenwood variance stays finite.
    """
    times = list(range(1, n_times + 1))
    n_risk = [n_start - i for i in range(n_times)]
    n_event = [1] * n_times
    return {"time": times, "n_risk": n_risk, "n_event": n_event}


def test_kpmsmp_basic():
    """Test basic functionality."""
    fit = _make_risk_table(n_times=5, n_start=10)
    alpha = 0.05
    result = km_simultaneous_band(fit, alpha)

    payload = getattr(result, "payload", result)
    assert isinstance(payload, dict)

    # Every key documented in the function's return statement must be present.
    for key in (
        "estimate", "time", "surv", "half_width", "sigma2",
        "lower", "upper", "h", "alpha", "n_times",
        "n_risk_start", "n", "method",
    ):
        assert key in payload, f"missing key: {key}"

    m = 5
    assert len(payload["time"]) == m
    assert len(payload["surv"]) == m
    assert len(payload["half_width"]) == m
    assert len(payload["sigma2"]) == m
    assert len(payload["lower"]) == m
    assert payload["n"] == m
    assert payload["n_times"] == float(m)
    assert payload["n_risk_start"] == 10
    assert payload["alpha"] == alpha
    assert isinstance(payload["method"], str)

    # The final half-width (the "estimate" field) must be finite.
    assert math.isfinite(payload["estimate"])

    # Survival curve must lie inside its simultaneous band at every finite time.
    for s, lo, hi in zip(payload["surv"], payload["lower"], payload["upper"]):
        if math.isnan(lo) or math.isnan(hi):
            continue
        assert lo <= s <= hi


def test_kpmsmp_edge():
    """Test edge case with a small but valid risk table."""
    fit = _make_risk_table(n_times=3, n_start=5)
    alpha = 0.5
    result = km_simultaneous_band(fit, alpha)

    payload = getattr(result, "payload", result)
    assert isinstance(payload, dict)

    m = 3
    assert len(payload["time"]) == m
    assert len(payload["surv"]) == m
    assert len(payload["lower"]) == m
    assert len(payload["upper"]) == m
    assert payload["n"] == m
    assert payload["n_risk_start"] == 5
    assert payload["alpha"] == alpha

    assert math.isfinite(payload["estimate"])
    assert math.isfinite(payload["h"])

    for s, lo, hi in zip(payload["surv"], payload["lower"], payload["upper"]):
        if math.isnan(lo) or math.isnan(hi):
            continue
        assert lo <= s <= hi
