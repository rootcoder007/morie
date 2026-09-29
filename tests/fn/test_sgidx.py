"""Tests for index of dispersion."""

from morie.fn import _array_core as np
from morie.fn.sgidx import sgidx


def test_sgidx_smoke():
    counts = np.array([5, 3, 7, 4, 6, 5, 4, 8, 3, 5])
    r = sgidx(counts)
    assert r.name == "index_of_dispersion"
    assert "VMR" in r.extra
    assert "pattern" in r.extra


def test_sgidx_poisson():
    rng = np.random.default_rng(42)
    counts = rng.poisson(10, 100)
    r = sgidx(counts)
    assert 0.5 < r.extra["VMR"] < 2.0


def test_vmr_and_chi2_recomputed():
    import math

    import pytest

    c = [3.0, 0.0, 7.0, 2.0, 1.0, 5.0]
    n = 6
    m = sum(c) / n
    v = sum((t - m) ** 2 for t in c) / (n - 1)
    r = sgidx(c)
    assert r.value == pytest.approx(v / m, rel=1e-13)
    assert r.extra["chi2"] == pytest.approx(v / m * (n - 1), rel=1e-13)
    assert r.extra["pattern"] == "clustered"
    # df = 2 survival is exp(-x/2); here df = 5 so only check monotone sanity
    assert 0.0 < r.extra["p_value"] < 1.0
    assert math.isfinite(r.extra["p_value"])
