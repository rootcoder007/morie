"""Anchored tests for ldcmpr.ld_r2 (LD r^2, Hill-Robertson 1968)."""

import math

from morie.fn.ldcmpr import ld_r2


def test_phased_hand_anchor():
    """Hand-computed haplotype table: AB=50, Ab=10, aB=10, ab=30.

    pA = 0.6, pB = 0.6, pAB = 0.5, D = 0.5 - 0.36 = 0.14,
    r = 0.14 / sqrt(0.6*0.4*0.6*0.4) = 0.14 / 0.24 = 0.5833...,
    r^2 = 0.14^2 / 0.0576 = 0.3402777...,
    D' = 0.14 / min(0.24, 0.24) = 0.5833...
    """
    a = [1] * 50 + [1] * 10 + [0] * 10 + [0] * 30
    b = [1] * 50 + [0] * 10 + [1] * 10 + [0] * 30
    res = ld_r2(a, b, phased=True)
    assert abs(res["pA"] - 0.6) < 1e-12
    assert abs(res["pB"] - 0.6) < 1e-12
    assert abs(res["pAB"] - 0.5) < 1e-12
    assert abs(res["D"] - 0.14) < 1e-12
    assert abs(res["r"] - 0.14 / 0.24) < 1e-12
    assert abs(res["estimate"] - 0.14 ** 2 / 0.0576) < 1e-12
    assert abs(res["Dprime"] - 0.14 / 0.24) < 1e-12


def test_unphased_no_double_het_exact():
    """Without double heterozygotes the EM is closed-form.

    Individuals AB/AB x2, Ab/Ab, aB/aB, ab/ab x2: 12 haplotypes,
    AB=4, Ab=2, aB=2, ab=4.  pA = pB = 0.5, pAB = 1/3,
    D = 1/3 - 1/4 = 1/12, r = (1/12)/0.25 = 1/3, r^2 = 1/9.
    Genotypic r^2 (PLINK): x = (2,2,2,0,0,0), y = (2,2,0,2,0,0):
    sxy = 2, sxx = syy = 6, r^2 = 4/36 = 1/9.
    """
    g1 = [2, 2, 2, 0, 0, 0]
    g2 = [2, 2, 0, 2, 0, 0]
    res = ld_r2(g1, g2)
    assert abs(res["pA"] - 0.5) < 1e-12
    assert abs(res["pB"] - 0.5) < 1e-12
    assert abs(res["pAB"] - 1.0 / 3.0) < 1e-9
    assert abs(res["D"] - 1.0 / 12.0) < 1e-9
    assert abs(res["estimate"] - 1.0 / 9.0) < 1e-9
    assert abs(res["r2_genotypic"] - 1.0 / 9.0) < 1e-12


def test_unphased_matches_twoldp():
    """The EM path must agree exactly with two_locus_dprime (reuse)."""
    from morie.fn.twoldp import two_locus_dprime
    g1 = [0, 1, 2, 1, 0, 2, 1, 1, 2, 0, 1, 2]
    g2 = [0, 1, 1, 2, 0, 2, 0, 1, 2, 1, 1, 2]
    res = ld_r2(g1, g2)
    base = two_locus_dprime(g1, g2)
    assert res["estimate"] == base["r2"]
    assert res["Dprime"] == base["estimate"]
    assert res["pAB"] == base["pAB"]


def test_perfect_ld():
    """Identical loci: r^2 = 1 on both paths."""
    g = [0, 1, 2, 0, 1, 2, 2, 0]
    res = ld_r2(g, g)
    assert abs(res["estimate"] - 1.0) < 1e-9
    assert abs(res["r2_genotypic"] - 1.0) < 1e-12
    hap = [0, 1, 0, 1, 1, 0]
    resp = ld_r2(hap, hap, phased=True)
    assert abs(resp["estimate"] - 1.0) < 1e-12
