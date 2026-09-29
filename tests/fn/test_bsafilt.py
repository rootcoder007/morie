"""Tests for morie.fn.bsafilt: Butterworth highpass gains recomputed on the unit circle."""

import cmath
import math

from morie.fn.bsafilt import bwhp


def _H(b, a, w):
    z = cmath.exp(1j * w)
    return sum(c * z ** (-k) for k, c in enumerate(b)) / sum(c * z ** (-k) for k, c in enumerate(a))


def test_highpass_gain_at_dc_nyquist_and_cutoff():
    fs, fc = 1000.0, 120.0
    for order in (2, 4):
        r = bwhp(fc, order=order, fs=fs)
        b, a = r["b"], r["a"]
        assert abs(_H(b, a, 0.0)) < 1e-12
        assert abs(abs(_H(b, a, math.pi)) - 1.0) < 1e-12
        # bilinear transform with prewarping keeps the -3 dB point at the cutoff
        assert abs(abs(_H(b, a, 2 * math.pi * fc / fs)) - 1 / math.sqrt(2)) < 1e-9
