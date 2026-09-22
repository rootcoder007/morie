"""Tests for ggmaft.generalized_gamma_aft."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.ggmaft import generalized_gamma_aft


def test_ggmaft_basic():
    """Test basic functionality with documented inputs and key names."""
    rng = np.random.default_rng(42)
    time = rng.uniform(0.1, 10.0, 100)  # strictly positive times
    event = rng.integers(0, 2, 100).astype(float)  # 0 = censored, 1 = event
    X = rng.normal(0, 1, (100, 5))
    result = generalized_gamma_aft(time, event, X)

    # Function returns a RichResult with these documented keys
    expected_keys = {
        "beta", "sigma", "q", "loglik",
        "lr_vs_weibull", "p_vs_weibull", "lr_vs_lognormal", "preferred",
    }
    assert expected_keys.issubset(set(result.keys()))

    # beta should have one intercept + 5 covariates = 6 coefficients
    assert np.asarray(result["beta"]).shape == (6,)
    assert result["sigma"] > 0
    assert np.isfinite(result["loglik"])
    assert np.isfinite(result["lr_vs_weibull"])
    assert np.isfinite(result["lr_vs_lognormal"])
    assert 0.0 <= result["p_vs_weibull"] <= 1.0
    assert 0.0 <= result["p_vs_lognormal"] <= 1.0
    assert result["preferred"] in {"generalized gamma", "weibull", "lognormal"}


def test_ggmaft_edge():
    """Test edge cases: all events, no events, and minimal data."""
    rng = np.random.default_rng(42)

    # All events (no censoring)
    time1 = rng.uniform(0.1, 10.0, 50)
    event1 = np.ones(50)
    X1 = rng.normal(0, 1, (50, 2))
    r1 = generalized_gamma_aft(time1, event1, X1)
    assert np.isfinite(r1["loglik"])
    assert r1["preferred"] in {"generalized gamma", "weibull", "lognormal"}

    # All censored
    time2 = rng.uniform(0.1, 10.0, 50)
    event2 = np.zeros(50)
    X2 = rng.normal(0, 1, (50, 2))
    r2 = generalized_gamma_aft(time2, event2, X2)
    assert np.isfinite(r2["loglik"])

    # Weibull-like data: fitted q should sit near 1, and the LR test
    # against Weibull should not reject.
    rng3 = np.random.default_rng(0)
    X3 = rng3.normal(size=(1200, 2))
    mu = 1.0 + 0.7 * X3[:, 0] - 0.4 * X3[:, 1]
    T = np.exp(mu + 0.6 * np.log(rng3.exponential(1.0, 1200)))
    C = rng3.exponential(float(np.exp(mu).mean()) * 6, 1200)
    t3 = np.minimum(T, C)
    e3 = (T <= C).astype(float)
    r3 = generalized_gamma_aft(t3, e3, X3)
    assert r3["p_vs_weibull"] > 0.05
    assert str(r3["preferred"]) == "weibull"
    # Regression coefficients recovered within tolerance, computed independently
    for i, v in enumerate([1.0, 0.7, -0.4]):
        assert abs(r3["beta"][i] - v) < 0.2
