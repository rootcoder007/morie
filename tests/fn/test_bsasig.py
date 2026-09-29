"""Tests for morie.fn.bsasig: suppressed-carrier AM and synchronous demodulation recomputed."""

import math

from morie.fn.bsasig import amsig


def test_modulation_and_demodulation():
    x = [math.sin(0.05 * n) for n in range(40)]
    fc, fs = 100.0, 1000.0
    r = amsig(x, fc, fs)
    w = 2 * math.pi * fc / fs
    for n in range(40):
        c = math.cos(w * n)
        assert abs(r["y"][n] - x[n] * c) < 1e-15
        assert abs(r["demodulated"][n] - 0.5 * x[n] - 0.5 * x[n] * math.cos(2 * w * n)) < 1e-12
    conv = amsig(x, fc, fs, conventional=True, depth=0.5)
    assert abs(conv["y"][3] - (1 + 0.5 * x[3]) * math.cos(3 * w)) < 1e-15
