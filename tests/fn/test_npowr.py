"""Tests for morie.fn.npowr."""

import math

from morie.fn.npowr import noise_power

X = [math.sin(0.1 * n) + (0.3 if n % 2 else -0.3) for n in range(200)]


def test_reference_and_difference_estimators():
    s = [math.sin(0.1 * n) for n in range(200)]
    assert abs(noise_power(X, signal=s).value - math.fsum((a - b) ** 2 for a, b in zip(X, s)) / 200) < 1e-15
    ref = math.fsum((X[i + 1] - X[i]) ** 2 for i in range(199)) / (2 * 199)
    r = noise_power(X)
    assert abs(r.value - ref) < 1e-15
    # the alternating +-0.3 noise has power 0.09; the smooth sine barely contributes to the differences
    assert abs(r.value - 2 * 0.09) < 0.01
