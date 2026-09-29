"""Tests for morie.fn.bsaar: the Burg recursion recomputed at order one."""

import math

from morie.fn.bsaar import burg_psd

X = [math.sin(0.3 * t) + 0.5 * math.cos(1.1 * t) + ((t * 7) % 5 - 2) / 10 for t in range(64)]


def test_order_one_reflection_coefficient_and_psd():
    n = len(X)
    num = -2.0 * sum(X[i] * X[i - 1] for i in range(1, n))
    den = sum(X[i] ** 2 for i in range(1, n)) + sum(X[i] ** 2 for i in range(n - 1))
    k = num / den
    s2 = sum(v * v for v in X) / n * (1 - k * k)
    r = burg_psd(X, order=1, nfft=5, fs=2.0)
    assert abs(float(r.extra["ar_coeffs"][0]) - k) < 1e-12
    assert abs(float(r.extra["noise_variance"]) - s2) < 1e-12
    for f, p in zip(r.extra["frequencies"], r.extra["psd"]):
        w = 2 * math.pi * float(f) / 2.0
        a = complex(1.0, 0.0) + k * complex(math.cos(w), -math.sin(w))
        assert abs(float(p) - s2 / abs(a) ** 2) < 1e-10


def test_higher_order_residual_variance_decreases():
    v = [float(burg_psd(X, order=p, nfft=4).extra["noise_variance"]) for p in (1, 2, 4)]
    assert v[0] >= v[1] >= v[2] > 0
