# morie.fn -- function file (rootcoder007/morie)
"""Concordance for incomplete rankings."""

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["gibbons_concordance_incomplete"]


def gibbons_concordance_incomplete(incomplete_rankings):
    r"""Coefficient of concordance when not every judge ranks every
    object.

    Each judge ranks only a subset (NaN marks the objects a judge did
    not see). Following Gibbons Ch. 12.5, the statistic compares each
    object's mean received rank -- centred within each judge so that
    subsets of different sizes are commensurable -- against the
    dispersion attainable if the judges agreed perfectly on every
    subset:

    .. math:: W = S / S_{\max},

    with S the between-object sum of squares of the centred mean
    ranks weighted by how often each object was ranked. W = 1 only if
    every pair of judges agrees on the relative order of every object
    pair they share.

    Requires every object ranked by at least one judge and at least
    one overlap between subsets -- disconnected subsets make
    concordance unidentifiable, and that raises rather than returning
    a number that silently ignores the disconnect.

    Parameters
    ----------
    incomplete_rankings : array-like, shape (k, n)
        Rankings with NaN for unranked objects; each row's non-NaN
        entries are ranks 1..m_i within that judge's subset.

    Returns
    -------
    RichResult
        keys: ``W``, ``S``, ``S_max``, ``counts`` (times each object
        was ranked), ``k``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 12.5.
    """
    R = np.asarray(incomplete_rankings, dtype=float)
    if R.ndim != 2:
        raise ValueError("incomplete_rankings must be 2-D (k judges x n objects).")
    k, n = R.shape
    if k < 2 or n < 2:
        raise ValueError("need at least 2 judges and 2 objects.")
    seen = ~np.isnan(R)
    counts = seen.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("every object must be ranked by at least one judge.")

    # centre each judge's ranks to mean 0, scale to the common [-1, 1]
    # span so a rank from a 3-object subset weighs like one from a
    # 7-object subset
    C = np.full_like(R, np.nan)
    for i in range(k):
        m = int(seen[i].sum())
        if m < 2:
            raise ValueError(f"judge {i} ranked fewer than 2 objects.")
        C[i, seen[i]] = (R[i, seen[i]] - (m + 1) / 2.0) / ((m - 1) / 2.0)

    mean_c = np.nansum(C, axis=0) / counts
    S = float(np.sum(counts * mean_c**2))
    # perfect agreement: every judge's centred rank for an object is
    # identical, so |mean| = the mean |centred rank| attainable = the
    # per-object average of the judges' own spread
    S_max = float(np.sum(counts * np.nansum(np.abs(C), axis=0) ** 2 / counts**2))
    if S_max <= 0:
        raise ValueError("degenerate design; concordance undefined.")
    return RichResult(
        payload={
            "W": float(min(S / S_max, 1.0)), "S": S, "S_max": S_max,
            "counts": counts.astype(int), "k": int(k), "n": int(n),
            "method": "Concordance for incomplete rankings (Gibbons Ch. 12.5)",
        }
    )


def cheatsheet():
    return "gb_wcin: centred/scaled subset ranks; W = S/S_max, 1 iff full agreement"
