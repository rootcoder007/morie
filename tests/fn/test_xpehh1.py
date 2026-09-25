"""Tests for xpehh1.xp_ehh (Sabeti et al. 2007 XP-EHH)."""

import math
from collections import Counter

import pytest

from morie.fn.xpehh1 import xp_ehh


def _hap(N, L, seed, sweep=0):
    """Deterministic 0/1 haplotypes; `sweep` copies the first haplotype
    into that many others, giving long shared haplotypes."""
    H = [[int(((math.sin(12.9898 * (seed * 1000 + i * L + j)) * 43758.5453) % 1) < 0.5)
          for j in range(L)] for i in range(N)]
    for i in range(1, sweep + 1):
        H[i] = list(H[0])
    return H


def _ehh(H, core, j):
    """Site EHH between core and marker j (inclusive): sum_i C(n_i,2) /
    C(N,2) over groups of identical haplotypes on that stretch."""
    lo, hi = min(core, j), max(core, j)
    c = Counter(tuple(h[lo:hi + 1]) for h in H)
    N = len(H)
    return sum(v * (v - 1) / 2 for v in c.values()) / (N * (N - 1) / 2)


def _ihh(H, core, min_ehh=0.05):
    """Trapezoid area outward from the core on both sides, including the
    segment to the first marker whose EHH falls below min_ehh."""
    L, tot = len(H[0]), 0.0
    for step in (1, -1):
        prev = _ehh(H, core, core)
        j = core + step
        while 0 <= j < L:
            e = _ehh(H, core, j)
            tot += 0.5 * (prev + e)
            prev = e
            if e < min_ehh:
                break
            j += step
    return tot


def test_xpehh1_basic():
    """XP-EHH = ln(I_A / I_B), I the integrated site EHH on index
    positions; the population carrying a long shared haplotype scores
    positive."""
    A = _hap(20, 15, 1, sweep=9)
    B = _hap(20, 15, 2)
    r = xp_ehh(A, B, core=7)
    IA, IB = _ihh(A, 7), _ihh(B, 7)
    assert r["I_A"] == pytest.approx(IA, rel=1e-12)
    assert r["I_B"] == pytest.approx(IB, rel=1e-12)
    assert r["xpehh_unstandardized"] == pytest.approx(math.log(IA / IB), abs=1e-12)
    assert r["estimate"] > 0


def test_xpehh1_edge():
    """Standardising subtracts the reference mean and divides by its sd;
    a non-positive sd raises."""
    A = _hap(20, 15, 1, sweep=9)
    B = _hap(20, 15, 2)
    u = xp_ehh(A, B, core=7)["xpehh_unstandardized"]
    assert xp_ehh(A, B, core=7, standardize=(0.2, 0.5))["estimate"] == pytest.approx((u - 0.2) / 0.5, abs=1e-12)
    with pytest.raises(ValueError):
        xp_ehh(A, B, core=7, standardize=(0.0, 0.0))


