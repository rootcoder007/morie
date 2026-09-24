"""Tests for hrzlam.horowitz_baseline_hazard_est."""

import math

from morie.fn import _array_core as np

from morie.fn.hrzlam import horowitz_baseline_hazard_est


def test_hrzlam_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    t = np.linspace(0, 10, n)
    x = rng.normal(0, 1, (n, p))
    event = rng.integers(0, 2, n)
    beta_hat = rng.normal(0, 1, p)
    result = horowitz_baseline_hazard_est(t, x, event, beta_hat)
    assert isinstance(result, dict)
    assert "lambda0_hat" in result
    assert "grid" in result
    assert "bandwidth" in result
    assert "A_K" in result
    assert "rate_exponent" in result
    assert math.isfinite(result["bandwidth"])


def test_hrzlam_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    t = np.linspace(0, 10, n)
    x = rng.normal(0, 1, (n, p))
    event = rng.integers(0, 2, n)
    beta_hat = rng.normal(0, 1, p)
    result = horowitz_baseline_hazard_est(t, x, event, beta_hat)
    assert isinstance(result, dict)
    assert "cumhaz" in result
    assert "cumhaz_times" in result
    assert "n_events" in result
    assert "method" in result
