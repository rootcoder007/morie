"""Tests for pwpgt.pwp_gap_time."""

import math
from morie.fn import _array_core as np
from morie.fn.pwpgt import pwp_gap_time


def test_pwpgt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    p = 3

    # Valid gap-time recurrent event data: clock resets at each occurrence,
    # so start=0 and stop equals a positive gap for every interval.
    start = np.zeros(n)
    gap = rng.uniform(0.1, 5.0, n)
    stop = start + gap
    event = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, p))
    occurrence = rng.integers(1, 4, n)

    result = pwp_gap_time(start, stop, event, X, occurrence)

    assert isinstance(result, dict)
    for key in ("estimate", "se", "cov", "loglik", "n_iter", "n_events"):
        assert key in result

    assert len(result["estimate"]) == p
    assert len(result["se"]) == p
    for v in result["estimate"]:
        assert math.isfinite(v)
    for v in result["se"]:
        assert math.isfinite(v)
    assert math.isfinite(result["loglik"])
    assert result["n_iter"] >= 0
    assert result["n_events"] >= 0


def test_pwpgt_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    p = 2

    # Edge: all events observed, single occurrence stratum (degenerate strata).
    start = np.zeros(n)
    gap = rng.uniform(0.1, 3.0, n)
    stop = start + gap
    event = np.ones(n)
    X = rng.normal(0, 1, (n, p))
    occurrence = np.ones(n)

    result = pwp_gap_time(start, stop, event, X, occurrence)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert len(result["estimate"]) == p
    for v in result["estimate"]:
        assert math.isfinite(v)
