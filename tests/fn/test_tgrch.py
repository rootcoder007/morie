"""Tests for tgrch.tgarch_model."""

import numpy as np
import pytest

from morie.fn.tgrch import tgarch_model


def _dgp(seed, n=4000, omega=0.05, alpha=0.05, gamma=0.10, beta=0.85):
    """GJR: s2_t = omega + (alpha + gamma 1[eps<0]) eps^2 + beta s2
    (Glosten, Jagannathan & Runkle 1993)."""
    rng = np.random.default_rng(seed)
    s2, y = omega / (1 - alpha - gamma / 2 - beta), np.empty(n)
    for t in range(n):
        eps = np.sqrt(s2) * rng.standard_normal()
        y[t] = eps
        s2 = omega + (alpha + gamma * (eps < 0)) * eps**2 + beta * s2
    return y


def test_tgrch_recovers_the_asymmetry():
    """gamma > 0 means negative shocks raise volatility more. Mean over
    seeds 1..3; the persistence alpha + gamma/2 + beta should recover 0.95."""
    gs, pers = [], []
    for s in (1, 2, 3):
        r = tgarch_model(_dgp(s))
        gs.append(float(r["gamma"]))
        pers.append(float(r["persistence"]))
    assert np.mean(gs) > 0.02
    assert np.mean(pers) == pytest.approx(0.95, abs=0.05)


def test_tgrch_symmetric_data_gives_small_gamma():
    """Plain GARCH data (gamma = 0): the asymmetry term must not be
    invented. Mean |gamma_hat| over seeds measured ~0.02."""
    gs = []
    for s in (5, 6, 7):
        r = tgarch_model(_dgp(s, gamma=0.0, alpha=0.10))
        gs.append(float(r["gamma"]))
    assert abs(np.mean(gs)) < 0.06


def test_tgrch_rejects_short_series():
    with pytest.raises(ValueError, match="at least 20"):
        tgarch_model(np.arange(6.0))
