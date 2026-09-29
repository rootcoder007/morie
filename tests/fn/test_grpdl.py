"""Tests for morie.fn.grpdl: exact group delay against the phase derivative."""

import cmath
import math

from morie.fn.grpdl import group_delay

B = [0.2, 0.5, 0.3]
A = [1.0, -0.4, 0.1]


def _phase(w):
    h = sum(b * cmath.exp(-1j * w * k) for k, b in enumerate(B)) / sum(
        a * cmath.exp(-1j * w * k) for k, a in enumerate(A)
    )
    return cmath.phase(h)


def test_against_numerical_phase_derivative():
    r = group_delay(B, A, worN=32)
    for i in (3, 10, 25):
        w = r.extra["frequencies"][i]
        h = 1e-6
        d = _phase(w + h) - _phase(w - h)
        d = (d + math.pi) % (2 * math.pi) - math.pi
        assert abs(r.value[i] - (-d / (2 * h))) < 1e-6


def test_linear_phase_fir():
    r = group_delay([1.0, 3.0, 3.0, 1.0], [1.0], worN=8)
    assert all(abs(v - 1.5) < 1e-12 for v in r.value)
