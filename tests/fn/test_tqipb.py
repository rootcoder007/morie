"""Tests for tqipb.turboquant_inner_product_distortion_bound."""

import math

import pytest

from morie.fn.tqipb import (bits_required, distortion_constant, tail_probability,
                           turboquant_inner_product_distortion_bound, variance_bound)


def test_tqipb_basic():
    """The constants: Theorem 2's sqrt(3) pi^2 / (d 4^b), Theorem 3's
    1 / (d 4^b), the b = 1..4 table and the QJL pi / (2 d); the variance
    does not depend on how the d coordinates are split into blocks."""
    d = 128
    assert distortion_constant(2, d, "panter_dite") == pytest.approx(math.sqrt(3) * math.pi ** 2 / (d * 16), rel=1e-15)
    assert distortion_constant(2, d, "lower_bound") == pytest.approx(1 / (d * 16), rel=1e-15)
    assert distortion_constant(1, d, "table") == pytest.approx(1.57 / d, rel=1e-15)
    assert distortion_constant(3, d, "qjl") == pytest.approx(math.pi / (2 * d), rel=1e-15)
    one = variance_bound(3, d, norm_sq=2.0, x_norm_sq=0.5)
    for k in (2, 3, 7):
        assert variance_bound(3, d, norm_sq=2.0, x_norm_sq=0.5, n_blocks=k) == pytest.approx(one, rel=1e-13)
    assert one == pytest.approx(0.18 / d * 2.0 * 0.5, rel=1e-13)


def test_tqipb_edge():
    """Chebyshev: delta = V / (eps^2 |x|^2 |y|^2), capped at 1; the
    sub-Gaussian tail is 2 exp(-eps^2 |x|^2 |y|^2 / (2 V)); bits_required
    is the smallest b meeting (eps, delta)."""
    v = variance_bound(2, 64)
    assert tail_probability(v, 0.1) == pytest.approx(min(1.0, v / 0.01), rel=1e-15)
    assert tail_probability(v, 0.1, tail="sub_gaussian") == pytest.approx(
        min(1.0, 2 * math.exp(-0.01 / (2 * v))), rel=1e-15)
    b = bits_required(0.1, 0.05, 256)
    assert tail_probability(variance_bound(b, 256), 0.1) <= 0.05
    assert b == 1 or tail_probability(variance_bound(b - 1, 256), 0.1) > 0.05
    r = turboquant_inner_product_distortion_bound(3, d=256, eps=0.1, delta=0.05)
    assert r["bits_needed"] == b
    with pytest.raises(ValueError, match="route"):
        distortion_constant(2, 8, "exact")


