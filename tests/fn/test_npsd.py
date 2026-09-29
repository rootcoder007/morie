"""Tests for morie.fn.npsd."""

import math

from morie.fn.npsd import noise_psd

X = [math.sin(1.7 * n) for n in range(50)]


def test_levels():
    m = sum(X) / 50
    v = math.fsum((t - m) ** 2 for t in X) / 50
    assert abs(noise_psd(X, fs=100.0).value - v / 100) < 1e-15
    assert abs(noise_psd(X, fs=100.0, onesided=True).value - 2 * v / 100) < 1e-15
