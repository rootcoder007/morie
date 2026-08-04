# morie.fn -- function file (rootcoder007/morie)
"""Lewontin's normalised two-locus linkage disequilibrium D'."""

from __future__ import annotations

import math

from . import _t4core as T

from ._richresult import RichResult

__all__ = ["two_locus_dprime"]

_EM_ITERS = 500


def two_locus_dprime(geno1, geno2):
    """Lewontin's D' for two biallelic loci from unphased genotypes.

    Formula: with ``pA``, ``pB`` the major-allele frequencies,
    ``pa = 1 - pA``, ``pb = 1 - pB``, and ``pAB`` the haplotype
    frequency,

        ``D = pAB - pA pB``,
        ``Dmax = min(pA pb, pa pB)``,  ``Dmin = max(-pA pB, -pa pb)``,
        ``D' = D / Dmax`` if ``D > 0`` else ``D / Dmin``.

    Normalising by the attainable extreme is the whole content of
    Lewontin's proposal: raw ``D`` is bounded by the allele frequencies,
    so it cannot be compared across loci, whereas ``D'`` always lies in
    ``[-1, 1]`` and reaches ``|D'| = 1`` exactly when one of the four
    haplotypes is absent.

    Genotypes are unphased, so ``pAB`` is not observed: only the double
    heterozygote is ambiguous, and it is resolved by expectation
    maximisation over the two possible phases, run for a fixed
    ``500`` iterations with no convergence test, so the answer is
    deterministic and identical in both language arms.

    Parameters
    ----------
    geno1, geno2 : array-like
        Genotypes at the two loci, coded as the count 0, 1, 2 of the
        allele at that locus.  Individuals with a missing (negative or
        non-integer) code at either locus are dropped pairwise.

    Returns
    -------
    RichResult
        ``estimate`` (D'), ``D``, ``pAB``, ``pA``, ``pB``, ``Dmax``,
        ``Dmin``, ``r``, ``r2``, ``n``, ``method``.

    References
    ----------
    Lewontin (1964), The interaction of selection and linkage. I.
    General considerations; heterotic models, Genetics 49:49-67.  The
    Genetics PDF at PMC could not be retrieved from this host (the
    fetch returned a 1.8 kB error page, not the article), so the coded
    form was read instead from Warnes and Leisch's CRAN package
    ``genetics``, R/LD.R (tarball genetics_1.3.8.1.3 fetched from CRAN),
    which gives ``Dmin <- max(-pA*pB, -pa*pb)``,
    ``Dmax <- min(pA*pb, pB*pa)`` and
    ``estDp <- if (estD > 0) estD/Dmax else estD/Dmin`` verbatim.
    ``genetics`` maximises the same likelihood with ``optimize()``; a
    fixed-iteration EM is used here instead because a golden-section
    search is not reproducible across language arms.
    """
    g1 = T.vec(geno1)
    g2 = T.vec(geno2)
    if len(g1) != len(g2):
        raise ValueError("geno1 and geno2 must be the same length")
    pairs = [(int(a), int(b)) for a, b in zip(g1, g2)
             if a == int(a) and b == int(b) and 0 <= a <= 2 and 0 <= b <= 2]
    n = len(pairs)
    if n < 2:
        raise ValueError("need at least 2 complete genotype pairs")
    # major allele at each locus, so pA, pB >= 1/2 and the tables below
    # are oriented the way genetics::LD orients them
    cnt1 = sum(a for a, _ in pairs)
    cnt2 = sum(b for _, b in pairs)
    pA = cnt1 / (2.0 * n)
    pB = cnt2 / (2.0 * n)
    flip1 = pA < 0.5
    flip2 = pB < 0.5
    if flip1:
        pairs = [(2 - a, b) for a, b in pairs]
        pA = 1.0 - pA
    if flip2:
        pairs = [(a, 2 - b) for a, b in pairs]
        pB = 1.0 - pB
    pa = 1.0 - pA
    pb = 1.0 - pB
    tab = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for a, b in pairs:
        tab[a][b] += 1
    # unambiguous AB haplotype count, plus the ambiguous double heterozygotes
    nAB = 2 * tab[2][2] + tab[2][1] + tab[1][2]
    namb = tab[1][1]
    dmin = max(-pA * pB, -pa * pb)
    dmax = min(pA * pb, pa * pB)
    pab = pA * pB  # start EM at linkage equilibrium
    for _ in range(_EM_ITERS):
        num = pab * (1.0 - pA - pB + pab)
        den = num + (pA - pab) * (pB - pab)
        w = num / den if den > 0 else 0.5
        pab = (nAB + 2.0 * namb * w) / (2.0 * n)
        lo = max(0.0, pA + pB - 1.0)
        hi = min(pA, pB)
        if pab < lo:
            pab = lo
        if pab > hi:
            pab = hi
    d = pab - pA * pB
    if d > 0:
        dp = d / dmax if dmax > 0 else float("nan")
    elif d < 0:
        dp = d / dmin if dmin < 0 else float("nan")
    else:
        dp = 0.0
    denom = pA * pB * pa * pb
    r = d / math.sqrt(denom) if denom > 0 else float("nan")
    return RichResult(
        payload={
            "estimate": float(dp),
            "D": float(d),
            "pAB": float(pab),
            "pA": float(pA),
            "pB": float(pB),
            "Dmax": float(dmax),
            "Dmin": float(dmin),
            "r": float(r),
            "r2": float(r * r) if r == r else float("nan"),
            "n": int(n),
            "method": "Lewontin D' two-locus disequilibrium",
        }
    )


def cheatsheet():
    return "two_locus_dprime(geno1, geno2): D' = D/Dmax (or D/Dmin), EM-phased."


# compact alias per ledger/NAMING.md
twolocusdprime = two_locus_dprime
