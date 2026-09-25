"""Tests for ssmpar (affine-composition scan for x_t = A_t x_{t-1} + b_t)."""

import math

import pytest

from morie.fn.ssmpar import (check_associativity, compose, parallel_scan,
                             scan_depth, sequential_scan, ssm_parallel_scan)


def _pairs(L):
    return [(0.9 * math.cos(0.7 * t), math.sin(1.3 * t) + 0.1 * t) for t in range(L)]


def test_ssmpar_basic():
    """For every length 1..40 the scan's states equal the recurrence run
    by hand, and every prefix composes to (prod A, x-offset)."""
    for L in range(1, 41):
        P = _pairs(L)
        x, ref = 0.5, []
        for A, b in P:
            x = A * x + b
            ref.append(x)
        r = ssm_parallel_scan(P, x0=0.5)
        assert r["states"] == pytest.approx(ref, abs=1e-12)
        assert sequential_scan(P, 0.5)["states"] == pytest.approx(ref, abs=1e-15)
        # the last prefix is the whole composition: (prod A_t, x_L at x0 = 0)
        prodA = math.prod(A for A, _ in P)
        A_, b_ = r["prefix"][-1]
        assert A_ == pytest.approx(prodA, abs=1e-12)
        assert b_ == pytest.approx(parallel_scan(P, 0.0)["states"][-1], abs=1e-12)


def test_ssmpar_edge():
    """compose applies its left argument first; associativity holds;
    depth is ceil(log2 L); an empty sequence and length 0 raise."""
    assert compose((2.0, 1.0), (3.0, 4.0)) == (6.0, 7.0)
    assert check_associativity((2.0, 1.0), (0.5, -1.0), (3.0, 2.0))["associative"]
    assert scan_depth(1000)["parallel_depth"] == 10
    assert scan_depth(1024)["parallel_depth"] == 10
    assert scan_depth(1)["parallel_depth"] == 1
    with pytest.raises(ValueError):
        parallel_scan([])
    with pytest.raises(ValueError):
        scan_depth(0)
