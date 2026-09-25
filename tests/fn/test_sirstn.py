"""Tests for sirstn.sir_stochastic."""

import math

import pytest

from morie.fn.sirstn import sir_stochastic


def test_sirstn_basic():
    """Major outbreaks end near the deterministic final size z = 1 -
    exp(-R0 z) (0.7968 for R0 = 2): over 60 seeds the mean attack rate of
    the large outbreaks has standard error about 0.004."""
    z = 0.5
    for _ in range(200):
        z = 1 - math.exp(-2 * z)
    big = [f for f in (sir_stochastic(995, 5, 0.4, 0.2, 1e9, seed=s)["estimate"] / 1000
                       for s in range(1, 61)) if f > 0.3]
    assert abs(sum(big) / len(big) - z) < 0.02


def test_sirstn_edge():
    """Bookkeeping is exact: S + I + R = N, every event is an infection or
    a recovery; with no transmission every case simply recovers."""
    r = sir_stochastic(95, 5, 0.5, 0.25, 20.0, seed=4)
    assert r["S"] + r["I"] + r["R"] == r["N"] == 100
    assert r["n_events"] == r["n_infections"] + r["R"]
    assert r["S"] == 95 - r["n_infections"]
    q = sir_stochastic(50, 7, 0.0, 1.0, 1e9, seed=2)
    assert (q["S"], q["I"], q["R"]) == (50.0, 0.0, 7.0)
    assert sir_stochastic(95, 5, 0.5, 0.25, 20.0, seed=4)["R"] == r["R"]
    with pytest.raises(ValueError, match="non-negative"):
        sir_stochastic(-1, 5, 0.5, 0.25, 10.0)


