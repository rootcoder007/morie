"""Tests for bsaqrs.rangayyan_ch4_length_transformation (Rangayyan Eq. 4.21)."""

import math

import pytest

from morie.fn.bsaqrs import rangayyan_ch4_length_transformation as lengthxfm

A = [0.0, 0.3, 1.2, 0.8, -0.1, 0.0, 0.4, 0.9, 0.2, 0.0]
B = [0.1, 0.1, -0.5, 0.6, 0.2, 0.3, 0.3, -0.2, 0.0, 0.1]


def test_rng195_basic():
    """L(n) = sum_{k=n}^{n+W-1} sqrt(sum_j (x_j(k+1) - x_j(k))^2), the
    multichannel arc length over W = round(w fs) steps, recomputed; a
    window that would run off the record is left at 0."""
    fs, w = 100.0, 0.03                  # W = 3 samples
    r = lengthxfm([A, B], w, fs)
    step = [math.hypot(A[k + 1] - A[k], B[k + 1] - B[k]) for k in range(9)]
    exp = [sum(step[n:n + 3]) for n in range(7)] + [0.0] * 3
    assert r["length"] == pytest.approx(exp, rel=1e-14, abs=0)
    assert (r["wsamp"], r["nchan"], r["n"]) == (3, 2, 10)


def test_rng195_edge():
    """One channel is its absolute first difference summed; channels of
    unequal length are refused."""
    r = lengthxfm(A, 0.02, 100.0)
    assert r["length"][0] == pytest.approx(abs(A[1] - A[0]) + abs(A[2] - A[1]), rel=1e-15)
    with pytest.raises(ValueError):
        lengthxfm([A, B[:-1]], 0.03, 100.0)
