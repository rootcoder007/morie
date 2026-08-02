# morie.fn -- function file (rootcoder007/morie)
"""Shepard diagram data for assessing MDS fit."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult
from .isotr import isotonic_regression_disparity

__all__ = ["shepard_diagram"]


def shepard_diagram(delta, D_config):
    r"""The (dissimilarity, distance) scatter plus its monotone trend.

    Pairs :math:`(\delta_{ij}, d_{ij}(X))` sorted by dissimilarity,
    with the PAV monotone fit through them and Spearman's rank
    correlation as the summary: a tight monotone trend is what a good
    nonmetric fit looks like, and scatter around the step function is
    exactly the stress.

    Parameters
    ----------
    delta : array-like, shape (n, n)
        Observed dissimilarities (symmetric, zero diagonal).
    D_config : array-like, shape (n, n)
        Configuration distances.

    Returns
    -------
    RichResult
        keys: ``dissimilarities``, ``distances`` (both sorted by
        dissimilarity), ``monotone_fit`` (PAV through the sorted
        distances), ``spearman_rho``, ``n_pairs``, ``method``.

    References
    ----------
    Shepard, R. N. (1962). The analysis of proximities:
    multidimensional scaling with an unknown distance function. I.
    *Psychometrika*, 27(2), 125-140.

    Kruskal, J. B. (1964). Multidimensional scaling by optimizing
    goodness of fit to a nonmetric hypothesis. *Psychometrika*, 29(1),
    1-27. (the diagram as the visual companion of stress)
    """
    Delta = np.asarray(delta, dtype=float)
    Dc = np.asarray(D_config, dtype=float)
    if Delta.shape != Dc.shape or Delta.ndim != 2 or Delta.shape[0] != Delta.shape[1]:
        raise ValueError("delta and D_config must be square matrices of the same shape.")
    n = Delta.shape[0]
    iu = np.triu_indices(n, k=1)
    dd, dc = Delta[iu], Dc[iu]
    order = np.argsort(dd, kind="stable")
    dd_s, dc_s = dd[order], dc[order]
    fit = isotonic_regression_disparity(dc_s, np.arange(dc_s.size))["disparities"]
    rho = float(stats.spearmanr(dd, dc).statistic) if dd.size > 2 else float("nan")

    return RichResult(
        payload={
            "dissimilarities": dd_s,
            "distances": dc_s,
            "monotone_fit": fit,
            "spearman_rho": rho,
            "n_pairs": int(dd.size),
            "method": "Shepard diagram (sorted pairs + PAV trend + Spearman rho)",
        }
    )


def cheatsheet():
    return "shrpd: (delta, d) pairs sorted by delta, PAV trend, Spearman rho"
