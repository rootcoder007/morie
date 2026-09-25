"""Tests for rgpcgmrm.rangayyan_pcg_murmur_detect."""

import math

import pytest

from morie.fn.bsaphys import rangayyan_pcg_murmur_detect


def _tones(parts, fs=1000.0, n=1000):
    return [sum(a * math.sin(2 * math.pi * f * t / fs) for f, a in parts) for t in range(n)]


def test_rgpcgmrm_basic():
    """Eq. (6.44): with 50 Hz at unit amplitude and 300 Hz at amplitude a,
    the 150-600 Hz power fraction is a^2 / (1 + a^2)."""
    r = rangayyan_pcg_murmur_detect(_tones([(50, 1.0), (300, 0.6)], 2048.0, 2048), 2048.0)
    assert r["hf_power_fraction"] == pytest.approx(0.36 / 1.36, rel=1e-9)
    assert r["murmur_present"] is True


def test_rgpcgmrm_edge():
    """A pure low-frequency sound has no high-band power: no murmur."""
    r = rangayyan_pcg_murmur_detect(_tones([(50, 1.0)], 2048.0, 2048), 2048.0)
    assert r["hf_power_fraction"] < 1e-8 and r["murmur_present"] is False
    with pytest.raises(ValueError, match="threshold"):
        rangayyan_pcg_murmur_detect(_tones([(50, 1.0)], 2048.0, 2048), 2048.0, threshold=1.5)
    with pytest.raises(ValueError, match="Nyquist"):
        rangayyan_pcg_murmur_detect(_tones([(50, 1.0)]), 1000.0)


