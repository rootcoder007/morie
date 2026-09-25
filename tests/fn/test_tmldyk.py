"""Tests for tmldyk (Laplace-mechanism private TMLE)."""

import math
import statistics

import pytest

from morie.fn import _array_core as np
from morie.fn.tmldyk import (ate_sensitivity, composition_budget,
                             laplace_noise, private_ci, private_release,
                             tmle_diff_kernel)


def _expit(x):
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p):
    return math.log(p / (1.0 - p))


def _lap(b, seed):
    """Inverse CDF of Lap(0, b) at the first uniform of the seeded stream."""
    u = float(np.random.default_rng(seed).uniform()) - 0.5
    return -b * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))


def _tmle(Y, A, g, Q1, Q0):
    H = [a / p - (1 - a) / (1 - p) for a, p in zip(A, g)]
    qa = [x if a else z for a, x, z in zip(A, Q1, Q0)]
    lo, hi = -50.0, 50.0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        s = sum(h * (y - _expit(_logit(q) + m * h)) for h, y, q in zip(H, Y, qa))
        lo, hi = (m, hi) if s > 0 else (lo, m)
    e = 0.5 * (lo + hi)
    q1s = [_expit(_logit(q) + e / p) for q, p in zip(Q1, g)]
    q0s = [_expit(_logit(q) - e / (1 - p)) for q, p in zip(Q0, g)]
    psi = statistics.fmean(x - z for x, z in zip(q1s, q0s))
    d = [h * (y - (x if a else z)) + x - z - psi for h, y, a, x, z in zip(H, Y, A, q1s, q0s)]
    return psi, statistics.pstdev(d) / math.sqrt(len(d))


def _data(n=40):
    W = [math.sin(2.3 * k) for k in range(n)]
    A = [1.0 if ((17 * k + 4) % 61 + 0.5) / 61.0 < _expit(0.5 * w) else 0.0 for k, w in enumerate(W)]
    Y = [1.0 if ((23 * k + 2) % 67 + 0.5) / 67.0 < _expit(-0.2 + 0.6 * a + w) else 0.0
         for k, (a, w) in enumerate(zip(A, W))]
    g = [_expit(0.1 + 0.4 * w) for w in W]
    Q1 = [_expit(0.4 + 0.9 * w) for w in W]
    Q0 = [_expit(-0.2 + 0.9 * w) for w in W]
    return Y, A, [[w] for w in W], g, Q1, Q0


def test_tmldyk_basic():
    """The released value is the TMLE plus Lap(2/(n g_min eps)) drawn by
    inverse transform, and the private variance adds 2 b^2."""
    Y, A, X, g, Q1, Q0 = _data()
    n = len(Y)
    psi, se = _tmle(Y, A, g, Q1, Q0)
    b = 2.0 / (n * 0.05) / 0.5
    noise = _lap(b, 11)
    r = tmle_diff_kernel(Y, A, X, epsilon=0.5, g_min=0.05, seed=11, g=g, Q1=Q1, Q0=Q0)
    assert r["non_private_psi"] == pytest.approx(psi, abs=1e-9)
    assert r["sensitivity"] == pytest.approx(2.0 / (n * 0.05), rel=1e-15)
    assert r["estimate"] == pytest.approx(psi + noise, abs=1e-9)
    assert r["se_private"] == pytest.approx(math.sqrt(se * se + 2 * b * b), rel=1e-9)
    lo, hi = r["ci"]
    assert hi - lo == pytest.approx(2 * 1.96 * r["se_private"], rel=1e-12)


def test_tmldyk_edge():
    """Parameter guards, composition, and the ci width identity."""
    with pytest.raises(ValueError):
        ate_sensitivity(10, 0.6)
    with pytest.raises(ValueError):
        private_release(0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        laplace_noise(0.0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        tmle_diff_kernel([2.0, 0.0], [1.0, 0.0], [[0.0], [1.0]])
    assert composition_budget([0.1, 0.2, 0.3])["total_epsilon"] == pytest.approx(0.6, abs=1e-15)
    s = ate_sensitivity(100, 0.1, 2.0)
    assert s["sensitivity"] == pytest.approx(0.4, rel=1e-15)
    rel = private_release(1.0, 0.2, 2.0, seed=4)
    assert rel["noise"] == pytest.approx(_lap(0.1, 4), abs=1e-15)
    c = private_ci(1.0, 0.2, 2.0, 0.3, seed=4)
    assert c["se_private"] == pytest.approx(math.sqrt(0.09 + 0.02), rel=1e-12)
