"""Tests for twoldp.two_locus_dprime (Lewontin's D' from unphased genotypes)."""

import math

import pytest

from morie.fn.twoldp import two_locus_dprime

G1 = [2, 2, 1, 1, 1, 0, 2, 1, 0, 1, 2, 1, 1, 0, 2, 1, 1, 1, 2, 0]
G2 = [2, 1, 1, 1, 0, 0, 2, 1, 1, 1, 2, 2, 1, 0, 1, 1, 0, 1, 2, 1]


def _score(pAB, pA, pB, g1, g2):
    """d/dpAB of the genotype log-likelihood under random union of the
    haplotypes AB, Ab, aB, ab (frequencies pAB, pA - pAB, pB - pAB,
    1 - pA - pB + pAB; derivatives +1, -1, -1, +1): each individual's
    probability sums over the phased pairs giving its genotype."""
    h = {"AB": pAB, "Ab": pA - pAB, "aB": pB - pAB, "ab": 1 - pA - pB + pAB}
    dh = {"AB": 1.0, "Ab": -1.0, "aB": -1.0, "ab": 1.0}
    hap = lambda a, b: ("A" if a else "a") + ("B" if b else "b")
    sc = 0.0
    for a, b in zip(g1, g2):
        pr = dpr = 0.0
        for a1 in (0, 1):
            for b1 in (0, 1):
                a2, b2 = a - a1, b - b1
                if a2 in (0, 1) and b2 in (0, 1):
                    u, v = hap(a1, b1), hap(a2, b2)
                    pr += h[u] * h[v]
                    dpr += dh[u] * h[v] + h[u] * dh[v]
        sc += dpr / pr
    return sc


def test_twoldp_basic():
    """pAB solves the genotype-likelihood score equation (bisection);
    D = pAB - pA pB and D' = D/Dmax for D > 0.  genetics::LD gives D
    0.15174 and D' 0.64229 here (its optimize() stops at ~1e-4)."""
    n = len(G1)
    pA, pB = sum(G1) / (2 * n), sum(G2) / (2 * n)
    lo, hi = max(0.0, pA + pB - 1) + 1e-12, min(pA, pB) - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _score(mid, pA, pB, G1, G2) > 0 else (lo, mid)
    pAB = 0.5 * (lo + hi)
    D = pAB - pA * pB
    Dmax = min(pA * (1 - pB), (1 - pA) * pB)
    r = two_locus_dprime(G1, G2)
    assert r["pAB"] == pytest.approx(pAB, abs=1e-9)
    assert r["D"] == pytest.approx(D, abs=1e-9)
    assert r["estimate"] == pytest.approx(D / Dmax, abs=1e-8)
    assert r["r"] == pytest.approx(D / math.sqrt(pA * pB * (1 - pA) * (1 - pB)), abs=1e-8)


def test_twoldp_edge():
    """Complete association gives D' = 1; missing codes are dropped;
    fewer than two complete pairs raise."""
    r = two_locus_dprime([2, 0, 2, 0, 2], [2, 0, 2, 0, 2])
    assert r["estimate"] == pytest.approx(1.0, abs=1e-12)
    assert two_locus_dprime(G1 + [-1], G2 + [1])["n"] == 20
    with pytest.raises(ValueError):
        two_locus_dprime([1], [1])


