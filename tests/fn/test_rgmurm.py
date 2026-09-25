"""Tests for rgmurm.rangayyan_murmur_analysis."""

import math

import pytest

from morie.fn.bsaphys import rangayyan_murmur_analysis


def _tones(parts, fs=1000.0, n=1000):
    return [sum(a * math.sin(2 * math.pi * f * t / fs) for f, a in parts) for t in range(n)]


def test_rgmurm_basic():
    """Eq. (6.45) integrates MAGNITUDES: a 50 Hz tone (constant area,
    25-75 Hz) and a 100 Hz tone of amplitude a (predictive area, 75-150
    Hz), both on exact bins (fs = N = 1024), give PA / CA = a. The
    symmetric Hann window (period N - 1) is not quite periodic in N, which
    leaks a few parts in 1e5; the ratio is also gain-free."""
    for a in (0.4, 2.0):
        r = rangayyan_murmur_analysis(_tones([(50, 1.0), (100, a)], 1024.0, 1024), 1024.0)
        assert r["pa_over_ca"] == pytest.approx(a, rel=1e-4)
    x = _tones([(50, 1.0), (100, 0.4)], 1024.0, 1024)
    assert rangayyan_murmur_analysis([7.0 * v for v in x], 1024.0)["pa_over_ca"] == \
        pytest.approx(rangayyan_murmur_analysis(x, 1024.0)["pa_over_ca"], rel=1e-12)
    assert r["dominant_freq_hz"] == 100.0


def test_rgmurm_edge():
    with pytest.raises(ValueError, match="f1 < f2 < f3"):
        rangayyan_murmur_analysis(_tones([(50, 1.0)]), 1000.0, f1=80.0)
    with pytest.raises(ValueError, match="Nyquist"):
        rangayyan_murmur_analysis(_tones([(50, 1.0)]), 250.0)
    with pytest.raises(ValueError, match="no energy"):
        rangayyan_murmur_analysis([0.0] * 256, 1000.0)


