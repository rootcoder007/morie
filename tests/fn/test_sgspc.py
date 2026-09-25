"""Tests for spectral density estimation."""

import pytest

from morie.fn import _array_core as np

from morie.fn.sgspc import sgspc


def test_sgspc_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (100, 2))
    Z = rng.normal(0, 1, 100)
    r = sgspc(Z, coords, n_freq=20)
    assert r.name == "spectral_density"
    assert "frequencies" in r.extra
    assert "power" in r.extra


def test_cheatsheet():
    from morie.fn.sgspc import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0


def test_sgspc_cosine_peak_and_periodogram_scale():
    """A unit cosine at 0.25 cycles per step along x on a 16 x 16 grid:
    |FFT|^2 / N^2 is 16^2 / 4 = 64 at each of the two frequencies
    (+-0.25, 0), zero elsewhere; both sit in the radial bin holding 0.25."""
    import math
    nf = 16
    coords = [(float(i), float(j)) for j in range(nf) for i in range(nf)]
    z = [math.cos(2 * math.pi * 0.25 * x) for x, _ in coords]
    r = sgspc(z, coords, n_freq=nf)
    mids = r.extra["frequencies"]
    power = r.extra["power"]
    width = mids[1] - mids[0]
    assert abs(r.extra["peak_frequency"] - 0.25) < width
    # every other bin is empty of energy
    k = power.index(max(power))
    for i, v in enumerate(power):
        if i != k:
            assert abs(v) < 1e-20
    # radial average: 2 points at 64 among all grid points in that ring
    import itertools
    freqs = [(q - nf // 2) / nf for q in range(nf)]
    edges = [0.0 + t * (math.sqrt(0.5) / (nf // 2 - 1)) for t in range(nf // 2)]
    ring = sum(1 for fy, fx in itertools.product(freqs, freqs)
               if edges[k] <= math.hypot(fx, fy) < edges[k + 1])
    assert power[k] == pytest.approx(2 * 64.0 / ring, rel=1e-12)
