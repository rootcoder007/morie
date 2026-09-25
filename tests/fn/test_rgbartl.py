"""Tests for rgbartl.rangayyan_bartlett_psd."""

import cmath
import math

import pytest

from morie.fn.bsacorr import rangayyan_bartlett_psd


def _dft_power(seg):
    m = len(seg)
    return [abs(sum(v * cmath.exp(-2j * math.pi * k * n / m) for n, v in enumerate(seg))) ** 2
            for k in range(m // 2 + 1)]


def test_rgbartl_basic():
    """Eq. (6.16): the mean of |DFT|^2 / M over K disjoint segments; the
    trailing samples that do not fill a segment are dropped."""
    x = [math.sin(0.7 * i) + 0.5 * math.cos(2.3 * i) for i in range(53)]
    r = rangayyan_bartlett_psd(x, fs=2.0, segment_length=10)
    assert (r["n_segments"], r["segment_length"]) == (5, 10)
    want = [sum(p / 10 for p in col) / 5
            for col in zip(*[_dft_power(x[i * 10:(i + 1) * 10]) for i in range(5)])]
    assert r["psd"] == pytest.approx(want, rel=1e-12, abs=1e-14)
    assert r["freqs"] == pytest.approx([k * 2.0 / 10 for k in range(6)], rel=1e-15)


def test_rgbartl_edge():
    """One segment is the plain periodogram; exactly one of n_segments and
    segment_length must be given."""
    x = [1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 2.0, 1.5]
    r = rangayyan_bartlett_psd(x, n_segments=1)
    assert r["psd"] == pytest.approx([p / 8 for p in _dft_power(x)], rel=1e-12, abs=1e-14)
    with pytest.raises(ValueError, match="exactly one"):
        rangayyan_bartlett_psd(x)
    with pytest.raises(ValueError, match="exactly one"):
        rangayyan_bartlett_psd(x, n_segments=2, segment_length=4)


