# morie.fn -- function file (rootcoder007/morie)
"""Minor allele frequency (MAF) calculation for SNP markers."""

import math

from ._richresult import RichResult

__all__ = ["maf_calculation"]


def maf_calculation(marker_matrix, coding="012"):
    """
    Minor allele frequency (MAF) of every SNP in a marker matrix.

    Formula: with genotypes coded as the count of one allele (0, 1, 2),
    the allele frequency at locus j is
    ``p_j = sum_i x_ij / (2 n_j)`` over the ``n_j`` genotyped individuals,
    and the minor allele frequency is ``MAF_j = min(p_j, 1 - p_j)``.

    Parameters
    ----------
    marker_matrix : array-like
        Individuals x loci genotype matrix (a vector is one locus).
        Missing genotypes (NaN or None) are skipped locus by locus.
    coding : {"012", "-101"}
        Allele-count coding 0/1/2 (default) or the centred -1/0/1 coding
        used by BGLR, which is shifted by +1 first.

    Returns
    -------
    result : dict
        Keys: ``maf`` (vector, one per locus), ``p`` (allele frequency of
        the counted allele), ``n_genotyped``, ``estimate`` (mean MAF),
        ``n``, ``n_loci``, ``method``.

    References
    ----------
    Montesinos Lopez, O. A., Montesinos Lopez, A. and Crossa, J. (2022).
    Multivariate Statistical Machine Learning Methods for Genomic
    Prediction. Springer, Ch 2 (marker quality control: MAF filtering).

    Examples
    --------
    >>> r = maf_calculation([[0, 2], [1, 2], [2, 1], [0, 2]])
    >>> r["maf"], r["p"]
    ([0.375, 0.125], [0.375, 0.875])
    """
    if coding not in ("012", "-101"):
        raise ValueError("coding must be '012' or '-101'")
    shift = 1.0 if coding == "-101" else 0.0
    M = marker_matrix.tolist() if hasattr(marker_matrix, "tolist") else list(marker_matrix)
    if not M:
        raise ValueError("marker_matrix is empty")
    if not isinstance(M[0], (list, tuple)):
        M = [[v] for v in M]
    n = len(M)
    L = len(M[0])
    maf, pv, ng = [], [], []
    for j in range(L):
        vals = []
        for i in range(n):
            v = M[i][j]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            g = float(v) + shift
            if g not in (0.0, 1.0, 2.0):
                raise ValueError(f"genotype {v!r} at locus {j} is not a valid {coding} code")
            vals.append(g)
        if not vals:
            raise ValueError(f"locus {j} has no genotyped individuals")
        s = 0.0
        for g in vals:
            s += g
        p = s / (2.0 * len(vals))
        pv.append(p)
        maf.append(min(p, 1.0 - p))
        ng.append(len(vals))
    tot = 0.0
    for m in maf:
        tot += m
    return RichResult(
        payload={
            "maf": maf,
            "p": pv,
            "n_genotyped": ng,
            "estimate": tot / L,
            "n": n,
            "n_loci": L,
            "method": "Minor allele frequency min(p, 1 - p), p = allele count / (2 n)",
        }
    )


def cheatsheet():
    return "mafcl: Minor allele frequency (MAF) per SNP, min(p, 1-p) with p = count/(2n)"


# compact alias per ledger/NAMING.md
mafcalculation = maf_calculation
