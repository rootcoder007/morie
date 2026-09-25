"""Tests for bsatf.rangayyan_pcg_envelope_avg (ECG-triggered PCG averaging)."""

import math

import pytest

from morie.fn.bsatf import rangayyan_pcg_envelope_avg


FS = 1000.0
ECG = [3 * math.exp(-((t % 800) - 100) ** 2 / 20.0) for t in range(8000)]
PCG = [math.exp(-((t % 800) - 150) ** 2 / 200.0) * math.sin(0.9 * t)
       + 0.3 * math.exp(-((t % 800) - 450) ** 2 / 200.0) * math.sin(1.3 * t)
       + 0.02 * math.sin(7.7 * t) for t in range(8000)]


def test_rgpcgenl_basic():
    """Triggers sit on the QRS peaks every 800 ms; S1 is found ~50 ms and
    S2 ~350 ms after the trigger, with the planted 0.3 amplitude ratio
    (within the envelope smoothing), and the cycle count is the number of
    complete cycles."""
    r = rangayyan_pcg_envelope_avg(PCG, ECG, FS)
    trig = [int(v) for v in r["triggers"]]
    assert all(abs(b - a - 800) <= 2 for a, b in zip(trig, trig[1:]))
    assert r["s1_time"] == pytest.approx(0.05, abs=0.01)
    assert r["s2_time"] == pytest.approx(0.35, abs=0.01)
    assert r["s2_s1_ratio"] == pytest.approx(0.3, abs=0.05)
    assert r["n_cycles"] >= 8


def test_rgpcgenl_edge():
    """Signals of different lengths raise."""
    with pytest.raises(ValueError):
        rangayyan_pcg_envelope_avg(PCG[:-10], ECG, FS)
