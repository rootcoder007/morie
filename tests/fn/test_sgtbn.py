"""Tests for turning bands simulation (sgtbn wrapper)."""

import pytest

from morie.fn.sgtbn import sgtbn
from morie.fn.zstbs import turning_bands


def test_sgtbn_matches_turning_bands():
    P = [(0.3 * i, (0.7 * i) % 2.1) for i in range(30)]
    r = sgtbn(P, cov_params={"sill": 2.0, "range": 1.5}, n_bands=50, seed=25)
    want = turning_bands(P, "exponential", sill=2.0, range_=1.5, n_bands=50, n_waves=50, seed=25).field
    assert r.name == "turning_bands_sim"
    assert r.extra["simulated_values"] == pytest.approx(want, abs=1e-15)
    assert r.statistic == pytest.approx(sum(want) / 30, abs=1e-15)
    assert r.extra["n_bands"] == 50


def test_sgtbn_nugget_adds_stream_noise():
    from morie.fn._rng import random_normal

    P = [(0.0, 0.0), (1.0, 1.0)]
    base = sgtbn(P, seed=3).extra["simulated_values"]
    r = sgtbn(P, cov_params={"nugget": 0.25}, seed=3).extra["simulated_values"]
    e = random_normal(2, seed=3, stream=10**6)
    assert r == pytest.approx([b + 0.5 * float(w) for b, w in zip(base, e)], abs=1e-15)


def test_docstring_example():
    r = sgtbn([(0.0, 0.0), (1.0, 0.5)], "gaussian", n_bands=4, seed=2, n_waves=3)
    assert [round(v, 6) for v in r.extra["simulated_values"]] == [0.881358, 0.400421]


def test_cheatsheet():
    from morie.fn.sgtbn import cheatsheet

    assert "turning" in cheatsheet()
