"""Tests for coxstr.cox_stratified."""

import math

from morie.fn import _array_core as np

from morie.fn.coxstr import cox_stratified


def test_coxstr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    time = rng.uniform(0.1, 10.0, n)
    raw_event = rng.integers(0, 2, n)
    event = [float(e) for e in raw_event]
    stratum = rng.integers(0, 3, n)

    result = cox_stratified(time, event, X, stratum)

    assert isinstance(result, dict)
    for key in ("beta", "se", "z", "p_value", "hazard_ratio", "loglik",
                "strata", "events_per_stratum", "empty_strata"):
        assert key in result

    assert len(result["beta"]) == p
    assert len(result["se"]) == p
    assert len(result["z"]) == p
    assert len(result["p_value"]) == p
    assert len(result["hazard_ratio"]) == p

    assert math.isfinite(result["loglik"])
    assert all(s >= 0 for s in result["se"])
    assert all(0 <= pv <= 1 for pv in result["p_value"])

    assert sorted(result["strata"].tolist()) == [0, 1, 2]


def test_coxstr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    time = rng.uniform(0.1, 10.0, n)
    raw_event = rng.integers(0, 2, n)
    event = [float(e) for e in raw_event]
    stratum = rng.integers(0, 2, n)

    result = cox_stratified(time, event, X, stratum, ties="breslow")

    assert isinstance(result, dict)
    assert "beta" in result
    assert "empty_strata" in result
    assert len(result["beta"]) == p
    assert math.isfinite(result["loglik"])
    assert sorted(result["strata"].tolist()) == [0, 1]
