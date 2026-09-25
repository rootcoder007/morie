"""Tests for bsatf.rangayyan_seizure_wavelet (sec. 8.17)."""

import math

import pytest

from morie.fn.bsatf import rangayyan_seizure_wavelet


FS = 256.0
E = [math.sin(2 * math.pi * 10 * t / FS) + 0.1 * math.sin(1.7 * t) for t in range(1024)]


def test_rgseizwv_basic():
    """Scale j covers fs/2^(j+1) .. fs/2^j Hz; the fluctuation index (sum
    of absolute first differences of the coefficients) is linear in the
    signal amplitude and the energy quadratic -- the DWT is linear; the
    total is the sum over the chosen scales; a 10 Hz rhythm puts most
    energy in the 8-16 Hz scale."""
    r = rangayyan_seizure_wavelet(E, FS)
    r2 = rangayyan_seizure_wavelet([2 * v for v in E], FS)
    assert r["bands"] == [(16.0, 32.0), (8.0, 16.0), (4.0, 8.0)]
    assert r2["fi"] == pytest.approx([2 * v for v in r["fi"]], rel=1e-12)
    assert r2["energies"] == pytest.approx([4 * v for v in r["energies"]], rel=1e-12)
    assert r["fi_total"] == pytest.approx(sum(r["fi"]), rel=1e-15)
    assert max(range(3), key=lambda k: r["energies"][k]) == 1


def test_rgseizwv_edge():
    """With a threshold the decision is fi_total > threshold."""
    r = rangayyan_seizure_wavelet(E, FS)
    assert rangayyan_seizure_wavelet(E, FS, threshold=r["fi_total"] - 1e-9)["seizure_detected"] in (True, 1)
    assert rangayyan_seizure_wavelet(E, FS, threshold=r["fi_total"] + 1e-9)["seizure_detected"] in (False, 0)
