"""Tests for ksr048 (Kosorok eq. 2.12, stochastic equicontinuity)."""

import math

import pytest

from morie.fn.ksr048 import kosorok_ch2_z_master_stochastic_equicontinuity as seq

GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def test_ksr048_basic():
    """Psi_n(theta)(t) = t theta (1 + 1/n), Psi(theta)(t) = t theta: the
    ratio is sqrt(n) max_t |t| |theta_n - theta_0| / n over
    1 + sqrt(n) |theta_n - theta_0|, recomputed; with theta_n = 1/sqrt(n)
    it is (1/n) / 2 and shrinks, so the condition holds."""
    ns = [4, 16, 64, 256]
    ths = [1.0 / math.sqrt(n) for n in ns]
    state = {}

    def psi_n(th, t):
        return t * th * (1.0 + 1.0 / state["n"])

    def psi(th, t):
        return t * th

    # the module evaluates psi_n per (theta, n) pair in order; feed n
    out_r = []
    for n, th in zip(ns, ths):
        state["n"] = n
        r = seq(psi_n, psi, [th], 0.0, [n], grid=GRID)
        out_r.append(r["ratio"][0])
        num = math.sqrt(n) * max(abs(t * th / n) for t in GRID)
        assert r["numerator"][0] == pytest.approx(num, rel=1e-14)
        assert r["denominator"][0] == pytest.approx(1 + math.sqrt(n) * th, rel=1e-14)
    assert out_r == pytest.approx([(1.0 / n) / 2 for n in ns], rel=1e-13)


def test_ksr048_edge():
    """A theta-free discrepancy cancels: ratio 0 up to rounding; the
    difference (t th + 0.3) - t th - 0.3 carries at most ~2 ulp of
    |t th| + 0.3 <= 0.8, times sqrt(100) = 10, so < 4e-15; mismatched
    sequence lengths are refused."""
    r = seq(lambda th, t: t * th + 0.3, lambda th, t: t * th, [0.5, 0.2], 0.0, [10, 100], grid=GRID)
    assert all(abs(float(v)) < 4e-15 for v in r["ratio"])
    with pytest.raises(ValueError):
        seq(lambda th, t: 0.0, lambda th, t: 0.0, [0.1, 0.2, 0.3], 0.0, [10])
