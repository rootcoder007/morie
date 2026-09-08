"""Tests for bicarp.bic_ar_order (Schwarz 1978)."""

import math

import pytest

from morie.fn.bicarp import bic_ar_order


def noise(n):
    """Lehmer minimal-standard LCG; exact in IEEE doubles."""
    out, s = [], 12345
    for _ in range(n):
        s = (16807 * s) % 2147483647
        out.append(s / 2147483647.0 - 0.5)
    return out


def ar1(n, phi):
    e = noise(n)
    x = [0.0]
    for t in range(1, n):
        x.append(phi * x[-1] + e[t])
    return x


def test_consistency_selects_the_true_order():
    assert bic_ar_order(ar1(120, 0.7), 4)["order"] == 1
    assert bic_ar_order(noise(120), 4)["order"] == 0
    assert bic_ar_order(ar1(120, 0.0), 4)["order"] == 0


def test_penalty_is_exactly_k_log_T():
    r = bic_ar_order(ar1(120, 0.7), 4)
    T = r["T"]
    logT = math.log(T)
    fit = [T * (math.log(2 * math.pi * s) + 1) for s in r["sigma2"]]
    pen = [r["bic_raw"][p] - fit[p] for p in range(5)]
    for p in range(5):
        assert abs(pen[p] - (p + 2) * logT) < 1e-9
    assert logT > 2  # stricter than AIC


def test_the_two_scalings_differ_by_a_constant_in_p():
    r = bic_ar_order(ar1(120, 0.7), 4)
    T = r["T"]
    d = [r["bic"][p] - r["bic_raw"][p] / T for p in range(5)]
    assert max(d) - min(d) < 1e-12


def test_ar1_coefficient_is_recovered():
    r = bic_ar_order(ar1(120, 0.7), 4)
    assert abs(r["coefficients"][1] - 0.7) < 0.05


def test_error_paths():
    with pytest.raises(ValueError):
        bic_ar_order(noise(50), -1)
    with pytest.raises(ValueError):
        bic_ar_order([1.0, 2.0, 3.0, 4.0], 3)
    with pytest.raises(ValueError):
        bic_ar_order([5.0] * 40, 2)
