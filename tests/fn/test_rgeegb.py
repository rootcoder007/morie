"""Tests for bsaclass.rangayyan_eeg_rhythms (sec. 1.2.6, eq. 6.44)."""

import math

import pytest

from morie.fn.bsaclass import rangayyan_eeg_rhythms


def _psd(x):
    n = len(x)
    m = sum(x) / n
    xc = [v - m for v in x]
    out = []
    for k in range(n // 2 + 1):
        re = sum(xc[t] * math.cos(-2 * math.pi * k * t / n) for t in range(n))
        im = sum(xc[t] * math.sin(-2 * math.pi * k * t / n) for t in range(n))
        out.append(re * re + im * im)
    return out


def test_rgeegb_basic():
    """Fraction of |X(k)|^2 in each band over the total, with the book's
    limits: delta [0.5, 4), theta [4, 8), alpha [8, 13], beta (13, fs/2].
    A 4 Hz line is theta, 8 and 13 Hz lines are alpha -- each counted once."""
    x = [math.sin(2 * math.pi * 4 * t / 100) + 0.5 * math.sin(2 * math.pi * 13 * t / 100)
         + 0.2 * math.sin(2 * math.pi * 8 * t / 100) + 0.3 * math.sin(2 * math.pi * 20 * t / 100)
         for t in range(200)]
    P = _psd(x)
    f = [k * 100 / 200 for k in range(len(P))]
    tot = sum(P)
    r = rangayyan_eeg_rhythms(x, 100)
    exp = {"delta": sum(p for p, v in zip(P, f) if 0.5 <= v < 4),
           "theta": sum(p for p, v in zip(P, f) if 4 <= v < 8),
           "alpha": sum(p for p, v in zip(P, f) if 8 <= v <= 13),
           "beta": sum(p for p, v in zip(P, f) if v > 13)}
    for k, v in exp.items():
        assert r["fraction"][k] == pytest.approx(v / tot, abs=1e-12)
    assert sum(r["fraction"][k] for k in exp) == pytest.approx(sum(p for p, v in zip(P, f) if v >= 0.5) / tot, abs=1e-12)
    assert r["dominant"] == "theta"


def test_rgeegb_edge():
    """Custom bands are half-open; bad sampling rate or band raises."""
    x = [math.sin(2 * math.pi * 5 * t / 50) for t in range(100)]
    r = rangayyan_eeg_rhythms(x, 50, bands={"a": (0.0, 5.0), "b": (5.0, 25.0)})
    assert r["fraction"]["a"] == pytest.approx(0.0, abs=1e-12)
    assert r["fraction"]["b"] == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        rangayyan_eeg_rhythms(x, 0)
    with pytest.raises(ValueError):
        rangayyan_eeg_rhythms(x, 50, bands={"a": (5.0, 1.0)})
