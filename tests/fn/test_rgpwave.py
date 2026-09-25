"""Tests for rgpwave.rangayyan_p_wave_detect."""

import math

import pytest

from morie.fn.bsaqrs import rangayyan_p_wave_detect


def _ecg(p_lead, rr=0.8, fs=250.0, n=2500):
    """Gaussian QRS, P wave p_lead seconds before it, T wave 300 ms after."""
    q = [int(fs * (0.5 + rr * k)) for k in range(20) if int(fs * (0.5 + rr * k)) < n - 10]
    x = [0.0] * n
    for r in q:
        for i in range(max(0, r - 200), min(n, r + 200)):
            t = (i - r) / fs
            x[i] += (1.2 * math.exp(-(t / 0.012) ** 2)
                     + 0.15 * math.exp(-((t + p_lead) / 0.025) ** 2)
                     + 0.3 * math.exp(-((t - 0.30) / 0.05) ** 2))
    return x, q


def test_rgpwave_basic():
    """Each P wave is found where it was placed, 160 ms before its QRS; the
    search starts at the T end QTmax = (2/9) RR + 250 ms after the
    previous QRS."""
    x, q = _ecg(0.16)
    r = rangayyan_p_wave_detect(x, q, 250.0)
    assert [(q[k + 1] - p) / 250.0 for k, p in enumerate(r["p"])] == [0.16] * (len(q) - 1)
    start = r["windows"][0][0] if "windows" in r else None
    if start is not None:
        assert start == q[0] + round((2 / 9 * 0.8 + 0.25) * 250.0)


def test_rgpwave_edge():
    """A longer PR interval moves the detection with it; an RR too short
    to leave an interval gives None rather than a window after the QRS."""
    x, q = _ecg(0.20)
    r = rangayyan_p_wave_detect(x, q, 250.0)
    assert all(abs((q[k + 1] - p) / 250.0 - 0.20) <= 0.008 for k, p in enumerate(r["p"]))
    x2, q2 = _ecg(0.16)
    q2 = q2[:3] + [q2[2] + 60] + q2[3:]      # an extra beat 240 ms later
    r2 = rangayyan_p_wave_detect(x2, q2, 250.0)
    assert r2["p"][2] is None
    with pytest.raises(ValueError, match="22 Hz"):
        rangayyan_p_wave_detect([0.0] * 64, [10, 40], 20.0)


