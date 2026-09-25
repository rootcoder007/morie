"""Tests for bsaphys.rangayyan_speech_features (sec. 7.2.3)."""

import math

import pytest

from morie.fn.bsaphys import rangayyan_speech_features


FS = 8000.0
# a 125 Hz harmonic series: period 64 samples exactly
V = [sum(math.sin(2 * math.pi * k * 125 * t / FS) / k for k in range(1, 15)) for t in range(1024)]


def test_rgspeech_basic():
    """Voiced segment with a 64-sample period: f0 = 125 Hz and pitch
    period 8 ms; the zero-crossing rate is the count of sign changes per
    second."""
    r = rangayyan_speech_features(V, FS)
    assert r["voiced"] in (True, 1, 1.0)
    assert r["f0_hz"] == pytest.approx(125.0, rel=1e-9)
    assert r["pitch_period_ms"] == pytest.approx(8.0, rel=1e-9)
    zc = sum(1 for a, b in zip(V, V[1:]) if (a >= 0) != (b >= 0))
    zcr = r["zero_crossing_rate"]
    assert zcr in (pytest.approx(zc / (len(V) / FS), rel=1e-9), pytest.approx(zc / (len(V) - 1), rel=1e-9))


def test_rgspeech_edge():
    """An f0 range beyond Nyquist raises."""
    with pytest.raises(ValueError):
        rangayyan_speech_features(V, FS, f0_range=(60.0, 5000.0))
