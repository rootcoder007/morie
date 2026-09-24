"""Tests for kpmnci.km_pointwise_ci."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.kpmnci import km_pointwise_ci


def _make_risk_table():
    """Construct a small valid risk table."""
    times = [1.0, 2.0, 3.0, 4.0, 5.0]
    n_risk = [40, 35, 30, 25, 20]
    n_event = [2, 1, 3, 2, 1]
    return {"time": times, "n_risk": n_risk, "n_event": n_event}


def test_kpmnci_basic():
    """Test basic functionality."""
    fit = _make_risk_table()
    alpha = 0.05
    result = km_pointwise_ci(fit, alpha)
    assert isinstance(result, dict)
    # Check that all expected payload keys are present
    expected_keys = [
        "estimate",
        "time",
        "surv",
        "se",
        "sigma2",
        "lower",
        "upper",
        "z",
        "alpha",
        "n_times",
        "n_risk_start",
        "n",
        "method",
    ]
    for key in expected_keys:
        assert key in result
    # Length checks
    m = len(fit["time"])
    assert len(result["time"]) == m
    assert len(result["surv"]) == m
    assert len(result["se"]) == m
    assert len(result["lower"]) == m
    assert len(result["upper"]) == m
    # Estimate is a probability in [0, 1] and finite
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    # Alpha is echoed back
    assert result["alpha"] == alpha
    # Pointwise CI brackets the survival curve
    for lo, sv, hi in zip(result["lower"], result["surv"], result["upper"]):
        assert lo <= sv <= hi


def test_kpmnci_edge():
    """Test edge cases: invalid alpha raises ValueError."""
    fit = _make_risk_table()
    # alpha must be in (0, 1)
    with pytest.raises(ValueError):
        km_pointwise_ci(fit, 1.5)
    with pytest.raises(ValueError):
        km_pointwise_ci(fit, 0.0)
    with pytest.raises(ValueError):
        km_pointwise_ci(fit, -0.1)
