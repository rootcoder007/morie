# morie.fn -- function file (rootcoder007/morie)
"""Marker heterozygosity and frequency of heterogeneous loci."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["heterozygosity_locus"]


def heterozygosity_locus(marker_matrix):
    r"""Marker heterozygosity and frequency of heterogeneous loci.

    For an ``n x m`` matrix of biallelic genotypes coded 0, 1, 2 (copies of
    the counted allele; missing values as ``None`` or NaN are skipped), per
    marker ``j``: allele frequency ``p_j = sum x_ij / (2 n_j)``, expected
    heterozygosity ``H_exp,j = 2 p_j (1 - p_j)`` (Nei 1973) and observed
    heterozygosity ``H_obs,j`` = share of individuals coded 1; per
    individual ``i``, the frequency of heterozygous (heterogeneous) loci =
    share of its called markers coded 1 -- the marker quality-control
    summaries of Montesinos Lopez, Montesinos Lopez and Crossa (2022, ch. 2).

    Parameters
    ----------
    marker_matrix : array-like, shape (n, m)
        Genotype codes 0/1/2.

    Returns
    -------
    RichResult
        ``H_obs`` and ``H_exp`` (length m), ``allele_freq``,
        ``het_freq_individual`` (length n), ``estimate`` (mean ``H_exp``),
        ``n``, ``m``.

    References
    ----------
    Nei, M. (1973). Analysis of gene diversity in subdivided populations. *PNAS*, 70(12), 3321-3323.

    Montesinos Lopez, O. A., Montesinos Lopez, A. and Crossa, J. (2022). *Multivariate Statistical
    Machine Learning Methods for Genomic Prediction*. Springer, ch. 2.

    Examples
    --------
    >>> r = heterozygosity_locus([[0, 1, 2], [1, 1, 2], [2, 0, 1], [1, 1, 0]])
    >>> r["H_obs"], r["H_exp"]
    ([0.5, 0.75, 0.25], [0.5, 0.46875, 0.46875])
    """
    M = marker_matrix.tolist() if hasattr(marker_matrix, "tolist") else marker_matrix
    M = [[None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in row] for row in M]
    n, m = len(M), len(M[0])
    for row in M:
        for v in row:
            if v is not None and v not in (0.0, 1.0, 2.0):
                raise ValueError("genotypes must be coded 0, 1 or 2")
    freq, hexp, hobs = [], [], []
    for j in range(m):
        col = [row[j] for row in M if row[j] is not None]
        if not col:
            raise ValueError(f"marker {j} has no called genotypes")
        p = sum(col) / (2.0 * len(col))
        freq.append(p)
        hexp.append(2.0 * p * (1.0 - p))
        hobs.append(sum(1 for v in col if v == 1.0) / len(col))
    ind = []
    for row in M:
        called = [v for v in row if v is not None]
        ind.append(sum(1 for v in called if v == 1.0) / len(called) if called else math.nan)
    return RichResult(
        payload={
            "H_obs": hobs,
            "H_exp": hexp,
            "allele_freq": freq,
            "het_freq_individual": ind,
            "estimate": math.fsum(hexp) / m,
            "n": n,
            "m": m,
            "method": "Marker heterozygosity (Nei 1973) and per-individual heterozygous-locus frequency",
        }
    )


hetlc = heterozygosity_locus


def cheatsheet():
    return "hetlc: per-marker H_obs, H_exp = 2p(1-p) and per-individual heterozygous-locus frequency"
