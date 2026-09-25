"""Tests for bsaphys.rangayyan_freq_domain_feat (sec. 6.4.1-6.4.2)."""

import cmath
import math

import pytest

from morie.fn.bsaphys import rangayyan_freq_domain_feat


FS = 1000.0
X = [math.sin(2 * math.pi * 50 * t / FS) + 0.3 * math.sin(2 * math.pi * 400 * t / FS)
     + 0.05 * math.sin(7.3 * t) for t in range(256)]


def _psd(x, fs):
    """Mean-removed, Hann-windowed periodogram over 0..fs/2 (n = 256 is
    already a power of two, so no zero padding)."""
    n = len(x)
    m = sum(x) / n
    w = [0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]
    s2 = sum(v * v for v in w)
    xs = [(v - m) * wi for v, wi in zip(x, w)]
    P = [abs(sum(xs[t] * cmath.exp(-2j * math.pi * k * t / n) for t in range(n))) ** 2 / s2
         for k in range(n // 2 + 1)]
    return [k * fs / n for k in range(n // 2 + 1)], P


def test_rgfrqdom_basic():
    """PSD moments (eqs. 6.32-6.41) with the PSD as a density, and the
    default low/mid/high band fractions cut at Nyquist, which sum to 1."""
    f, P = _psd(X, FS)
    tot = sum(P)
    mf = sum(a * b for a, b in zip(f, P)) / tot
    m2 = sum((a - mf) ** 2 * b for a, b in zip(f, P)) / tot
    m3 = sum((a - mf) ** 3 * b for a, b in zip(f, P)) / tot
    m4 = sum((a - mf) ** 4 * b for a, b in zip(f, P)) / tot
    r = rangayyan_freq_domain_feat(X, FS)
    assert r["total_power"] == pytest.approx(tot, rel=1e-10)
    assert r["mean_freq_hz"] == pytest.approx(mf, rel=1e-10)
    assert r["fm2_hz2"] == pytest.approx(m2, rel=1e-9)
    assert r["spectral_skewness"] == pytest.approx(m3 / m2 ** 1.5, rel=1e-9)
    assert r["spectral_kurtosis"] == pytest.approx(m4 / m2 ** 2, rel=1e-9)
    bands = r["band_power_fraction"]
    assert [(lo, hi) for lo, hi, _ in bands] == [(0.0, 100.0), (100.0, 300.0), (300.0, 500.0)]
    assert sum(fr for _, _, fr in bands) == pytest.approx(1.0, abs=1e-12)
    assert bands[0][2] == pytest.approx(sum(p for a, p in zip(f, P) if a < 100) / tot, abs=1e-12)


def test_rgfrqdom_edge():
    """A reversed band raises; at a higher sampling rate the default
    partition has four bands."""
    with pytest.raises(ValueError):
        rangayyan_freq_domain_feat(X, FS, bands=[(200.0, 100.0)])
    assert len(rangayyan_freq_domain_feat(X, 4000.0)["band_power_fraction"]) == 4
