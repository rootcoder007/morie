"""Tests for egrch.egarch_model."""

import numpy as np
import pytest

from morie.fn.egrch import egarch_model


def _dgp(seed, n=3000, omega=-0.2, alpha=0.2, gamma=-0.15, beta=0.95):
    """log s2_t = omega + alpha(|z| - E|z|) + gamma z + beta log s2_{t-1}
    (Nelson 1991). gamma < 0 is the leverage effect."""
    rng = np.random.default_rng(seed)
    e_abs = np.sqrt(2 / np.pi)
    log_s2, y = 0.0, np.empty(n)
    for t in range(n):
        s = np.exp(0.5 * log_s2)
        z = rng.standard_normal()
        y[t] = s * z
        log_s2 = omega + alpha * (abs(z) - e_abs) + gamma * z + beta * log_s2
    return y


def test_egrch_recovers_persistence_and_leverage_sign():
    """beta is the log-variance persistence; gamma < 0 (bad news raises
    volatility more) must come back negative. Mean over seeds 1..3."""
    bs, gs = [], []
    for s in (1, 2, 3):
        r = egarch_model(_dgp(s))
        bs.append(float(r["beta"]))
        gs.append(float(r["gamma"]))
    assert np.mean(bs) == pytest.approx(0.95, abs=0.05)
    assert np.mean(gs) < 0


def test_egrch_variance_is_positive_without_constraints():
    """The point of modelling LOG variance: no parameter constraint is
    needed for positivity, so every fitted variance must be positive."""
    r = egarch_model(_dgp(7, n=600))
    assert np.all(np.asarray(r["conditional_variance"], dtype=float) > 0)


def test_egrch_rejects_short_series():
    with pytest.raises(ValueError, match="at least 20"):
        egarch_model(np.arange(10.0))
