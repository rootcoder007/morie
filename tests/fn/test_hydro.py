"""Tests for hydro (L-moments, flood frequency, flow duration, runoff)."""

import math

import pytest

from morie.fn.hydro import (
    channel_slope,
    convolve_runoff,
    flood_frequency,
    flow_duration_curve,
    sample_lmoments,
)


def _pwm(x, r):
    xs = sorted(x)
    n = len(xs)
    tot = 0.0
    for j, v in enumerate(xs, start=1):
        w = 1.0
        for k in range(1, r + 1):
            w *= (j - k) / (n - k)
        tot += w * v
    return tot / n


def test_sample_lmoments_from_unbiased_pwms():
    x = [21.0, 34.5, 19.2, 55.1, 28.3, 31.0, 44.2, 25.5]
    b = [_pwm(x, r) for r in range(4)]
    l1, l2 = b[0], 2 * b[1] - b[0]
    l3 = 6 * b[2] - 6 * b[1] + b[0]
    l4 = 20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]
    assert sample_lmoments(x) == pytest.approx([l1, l2, l3 / l2, l4 / l2], rel=1e-12)


def test_gumbel_flood_quantiles_recomputed():
    """Gumbel by L-moments: alpha = l2 / log 2, xi = l1 - Euler alpha,
    Q_T = xi - alpha log(-log(1 - 1/T))."""
    q = [120, 95, 310, 180, 150, 220, 90, 260, 140, 175]
    l1 = _pwm(q, 0)
    l2 = 2 * _pwm(q, 1) - l1
    a = l2 / math.log(2)
    xi = l1 - 0.57721566490153286 * a
    r = flood_frequency(q, dist="gumbel", return_periods=(2, 100))
    want = [xi - a * math.log(-math.log(1 - 1 / T)) for T in (2, 100)]
    assert r.quantiles == pytest.approx(want, rel=1e-12)


def test_flow_duration_runoff_and_slope():
    fdc = flow_duration_curve([3.0, 1.0, 2.0, 4.0, 6.0], [0.5])
    # descending flows 6,4,3,2,1 at p = i/6; p = 0.5 is exactly the third
    assert fdc.quantiles == pytest.approx([3.0], rel=1e-15)
    P, U = [1.0, 0.5, 2.0], [0.2, 1.0, 0.4]
    want = [sum(P[m] * U[j - m] for m in range(3) if 0 <= j - m < 3) for j in range(5)]
    assert convolve_runoff(P, U) == pytest.approx(want, rel=1e-15)
    r = channel_slope([0.0, 50.0, 150.0, 300.0], [10.0, 10.5, 12.0, 16.0])
    assert r.simple == pytest.approx(6.0 / 300.0, rel=1e-14)
