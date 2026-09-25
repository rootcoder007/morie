"""Tests for bsaqrs.rangayyan_pcg_segments (Rangayyan 2024 sec. 4.9)."""

import math

import pytest

from morie.fn.bsaqrs import hsoundid, qrsdetect, rangayyan_pcg_segments


FS = 500.0


def _g(t, c, w):
    return math.exp(-((t - c) ** 2) / (2 * w * w))


# 0.8 s beats: QRS at 0.1 s, carotid upstroke peak 0.25 s, dicrotic wave 0.45 s
ECG = [2.0 * _g((i % 400) / FS, 0.1, 0.008) - 0.3 * _g((i % 400) / FS, 0.12, 0.01)
       + 0.3 * _g((i % 400) / FS, 0.35, 0.04) for i in range(4000)]
CP = [_g((i % 400) / FS, 0.25, 0.05) + 0.5 * _g((i % 400) / FS, 0.45, 0.04) for i in range(4000)]
PCG = [_g((i % 400) / FS, 0.13, 0.01) * math.sin(0.8 * i) + 0.6 * _g((i % 400) / FS, 0.42, 0.01) * math.sin(1.1 * i)
       + 0.01 * math.sin(0.37 * i) for i in range(4000)]


def test_rgpcg_basic():
    """S1 is the Pan-Tompkins QRS and S2 the carotid notch less 52.6 ms
    (steps 1-4); systole runs S1 to the next S2 and diastole S2 to the
    next S1 (steps 5-6), with the PCG RMS over each segment."""
    r = rangayyan_pcg_segments(PCG, ECG, CP, FS)
    ids = hsoundid(ECG, CP, FS)
    assert r["s1"] == list(ids["s1"]) == list(qrsdetect(ECG, FS)["qrs"])
    assert r["s2"] == [max(0, d - round(0.0526 * FS)) for d in ids["notch"]]
    sysl = [(a, min(b for b in r["s2"] if b > a)) for a in r["s1"] if any(b > a for b in r["s2"])]
    assert r["systole"] == sysl
    dia = [(b, min(c for c in r["s1"] if c > b)) for _, b in sysl if any(c > b for c in r["s1"])]
    assert r["diastole"] == dia
    rms = [math.sqrt(sum(v * v for v in PCG[a:b]) / (b - a)) for a, b in sysl]
    assert r["systolerms"] == pytest.approx(rms, rel=1e-12)
    # QRS timing: within 10 ms of the planted R peak at 0.1 s into each beat
    assert all(abs(q % 400 - 50) <= 5 for q in r["s1"])
    assert len(r["s1"]) == 10


def test_rgpcg_edge():
    """Unequal signal lengths raise."""
    with pytest.raises(ValueError):
        rangayyan_pcg_segments(PCG[:-1], ECG, CP, FS)
