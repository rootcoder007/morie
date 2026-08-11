# morie.fn -- function file (rootcoder007/morie)
"""Linkage disequilibrium r^2 between two biallelic loci."""

import math

from ._richresult import RichResult
from .twoldp import two_locus_dprime

__all__ = ["ld_r2"]


def ld_r2(geno1, geno2, phased=False):
    """Linkage disequilibrium r^2 (Hill-Robertson) for two biallelic loci.

    With haplotype frequency ``pAB`` and allele frequencies ``pA``,
    ``pB`` (``pa = 1 - pA``, ``pb = 1 - pB``),

        ``D = pAB - pA pB``,   ``r^2 = D^2 / (pA pa pB pb)``,

    the squared correlation of allelic state over haplotypes.  This is
    the measure Hill and Robertson (1968) write as sigma_d^2's
    single-pair analogue r^2 = D^2/(p(1-p)q(1-q)) and adopt in place of
    D because its expectation relates directly to sample size and
    drift.

    Two input conventions are supported.

    * ``phased=True``: ``geno1``, ``geno2`` are 0/1 allele indicators
      on aligned haplotypes; ``pAB`` is the observed frequency of the
      1-1 haplotype, so r^2 is computed exactly.
    * ``phased=False`` (default): ``geno1``, ``geno2`` are unphased
      diploid genotypes coded 0/1/2.  ``pAB`` is not observed (the
      double heterozygote is phase-ambiguous) and is estimated by the
      fixed-iteration EM of :func:`morie.fn.twoldp.two_locus_dprime`,
      whose orientation is to the major allele at each locus (r^2 is
      invariant to that flip).  The squared Pearson correlation of the
      genotype counts is also reported as ``r2_genotypic``; this is
      what PLINK 1.9 computes for ``--r2`` ("currently based on
      correlations between genotype allele counts; phase is not
      considered", PLINK 1.9 LD docs).

    Parameters
    ----------
    geno1, geno2 : array-like
        Allele indicators (phased) or genotype counts (unphased).
    phased : bool
        Input convention, see above.

    Returns
    -------
    RichResult
        Keys ``estimate`` (r^2), ``r``, ``D``, ``Dprime``, ``pA``,
        ``pB``, ``pAB``, ``r2_genotypic`` (unphased only), ``n``,
        ``method``.

    References
    ----------
    Hill, W. G. and Robertson, A. (1968). Linkage disequilibrium in
    finite populations. Theoretical and Applied Genetics 38(6),
    226-231 (r^2 = D^2/(p(1-p)q(1-q)), their eq. for sigma_d^2's
    single-locus-pair form, sec. 2).
    Lewontin, R. C. (1964). Genetics 49(1), 49-67 (D, D').
    PLINK 1.9 LD documentation, https://www.cog-genomics.org/plink/1.9/ld
    (genotype-allele-count correlation r^2; fetched 2026-08-09).
    EM phase resolution: CRAN package genetics R/LD.R as documented in
    morie.fn.twoldp (local file src/morie/fn/twoldp.py).
    """
    if phased:
        a = [float(v) for v in geno1]
        b = [float(v) for v in geno2]
        if len(a) != len(b):
            raise ValueError("geno1 and geno2 must be the same length")
        n = len(a)
        if n < 2:
            raise ValueError("need at least 2 haplotypes")
        for v in a + b:
            if v not in (0.0, 1.0):
                raise ValueError("phased inputs must be coded 0/1")
        pA = sum(a) / n
        pB = sum(b) / n
        pAB = sum(x * y for x, y in zip(a, b)) / n
        d = pAB - pA * pB
        pa = 1.0 - pA
        pb = 1.0 - pB
        denom = pA * pa * pB * pb
        r = d / math.sqrt(denom) if denom > 0 else float("nan")
        dmax = min(pA * pb, pa * pB)
        dmin = max(-pA * pB, -pa * pb)
        if d > 0:
            dp = d / dmax if dmax > 0 else float("nan")
        elif d < 0:
            dp = d / dmin if dmin < 0 else float("nan")
        else:
            dp = 0.0
        return RichResult(payload={
            "estimate": float(r * r) if r == r else float("nan"),
            "r": float(r), "D": float(d), "Dprime": float(dp),
            "pA": float(pA), "pB": float(pB), "pAB": float(pAB),
            "n": int(n),
            "method": "LD r^2 (Hill-Robertson 1968), phased haplotypes",
        })
    base = two_locus_dprime(geno1, geno2)
    # PLINK-style genotypic r^2 on the complete pairs, same pairwise
    # deletion rule as two_locus_dprime.
    g1 = [float(v) for v in geno1]
    g2 = [float(v) for v in geno2]
    pairs = [(x, y) for x, y in zip(g1, g2)
             if x in (0.0, 1.0, 2.0) and y in (0.0, 1.0, 2.0)]
    n = len(pairs)
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    r2g = (sxy * sxy) / (sxx * syy) if sxx > 0 and syy > 0 else float("nan")
    return RichResult(payload={
        "estimate": float(base["r2"]),
        "r": float(base["r"]), "D": float(base["D"]),
        "Dprime": float(base["estimate"]),
        "pA": float(base["pA"]), "pB": float(base["pB"]),
        "pAB": float(base["pAB"]),
        "r2_genotypic": float(r2g),
        "n": int(base["n"]),
        "method": "LD r^2 (Hill-Robertson 1968), EM-phased genotypes",
    })


def cheatsheet():
    return "ldcmpr: LD r^2 = D^2/(pA pa pB pb); EM-phased (unphased 0/1/2) or exact (phased 0/1)."


# compact alias per ledger/NAMING.md
ldr2 = ld_r2
ldcmpr = ld_r2
