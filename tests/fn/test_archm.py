"""Tests for archm.arch_in_mean."""

import numpy as np
import pytest

from morie.fn.archm import arch_in_mean


def _dgp(seed, n=3000, omega=0.2, alpha=0.4, delta=0.8, mu=0.1):
    """y_t = mu + delta sigma_t + eps_t, sigma_t^2 = omega + alpha eps_{t-1}^2
    (Engle, Lilien & Robins 1987)."""
    rng = np.random.default_rng(seed)
    eps_prev, y = 0.0, np.empty(n)
    for t in range(n):
        s2 = omega + alpha * eps_prev**2
        eps = np.sqrt(s2) * rng.standard_normal()
        y[t] = mu + delta * np.sqrt(s2) + eps
        eps_prev = eps
    return y


def test_archm_recovers_the_risk_premium_and_arch_parameters():
    """Mean over seeds 1..3 at (omega, alpha, delta) = (0.2, 0.4, 0.8)."""
    al, dl = [], []
    for s in (1, 2, 3):
        r = arch_in_mean(_dgp(s))
        al.append(float(r["alpha"]))
        dl.append(float(r["delta"]))
    assert np.mean(al) == pytest.approx(0.4, abs=0.12)
    assert np.mean(dl) == pytest.approx(0.8, abs=0.3)


def test_archm_no_arch_data_gives_small_alpha():
    rng = np.random.default_rng(9)
    r = arch_in_mean(0.1 + rng.standard_normal(2000))
    assert float(r["alpha"]) < 0.1


def test_archm_conditional_variance_is_positive_and_aligned():
    r = arch_in_mean(_dgp(5, n=500))
    s2 = np.asarray(r["conditional_variance"], dtype=float)
    assert s2.shape == (500,)
    assert np.all(s2 > 0)


def test_archm_rejects_short_series():
    with pytest.raises(ValueError, match="at least 20"):
        arch_in_mean(np.arange(5.0))
