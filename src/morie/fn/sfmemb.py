# morie.fn -- function file (rootcoder007/morie)
"""MEM Bonferroni-corrected eigenvector selection."""

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rrng_core import pt
from .sfilter import moran_eigenvectors


def sfmemb(y, W, alpha=0.05, candidates="positive"):
    r"""Moran eigenvector maps (MEM) significantly related to y after a Bonferroni correction.

    Each candidate MEM v_j (unit length, orthogonal to the constant) is
    tested in the simple regression of y on it: with r_j = v_j'(y -
    ybar) / ||y - ybar|| the statistic t_j = r_j sqrt((n - 2)/(1 - r_j^2))
    has n - 2 df; MEMs with two-sided p-value below alpha / m (m
    candidates) are kept (Dray, Legendre and Peres-Neto 2006; Bauman et al.
    2018, who recommend correcting for the number of MEMs tested). Because the
    MEMs are orthonormal the selected set's coefficients are the v_j'y.

    References
    ----------
    Dray, S., Legendre, P. and Peres-Neto, P. R. (2006). Spatial modelling: a
    comprehensive framework for principal coordinate analysis of neighbour
    matrices (PCNM). *Ecological Modelling* 196, 483-493.
    Bauman, D., Drouet, T., Dray, S. and Vleminckx, J. (2018). Disentangling
    good from bad practices in the selection of spatial or phylogenetic
    eigenvectors. *Ecography* 41, 1638-1649.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(8)] for i in range(8)]
    >>> r = sfmemb([1.0, 2.0, 2.5, 4.0, 3.5, 3.0, 1.5, 1.0], W, alpha=0.2)
    >>> r["selected"]
    [1]
    """
    yv = [float(v) for v in y]
    n = len(yv)
    me = moran_eigenvectors(W)
    pool = {"positive": me["positive"], "negative": me["negative"], "all": list(range(len(me["eigenvalues"])))}[
        candidates
    ]
    m = len(pool)
    ybar = ssum(yv) / n
    yc = [v - ybar for v in yv]
    ss = math.sqrt(ssum(v * v for v in yc))
    E = me["vectors"]
    rows = []
    for k in pool:
        r = ssum(E[i][k] * yc[i] for i in range(n)) / ss
        t = r * math.sqrt((n - 2) / (1.0 - r * r))
        p = 2.0 * float(pt(abs(t), n - 2, lower_tail=False))
        rows.append((k, r, t, p))
    sel = [k for k, _, _, p in rows if p < alpha / m]
    return RichResult(
        payload={
            "selected": sel,
            "p_values": [p for _, _, _, p in rows],
            "t": [t for _, _, t, _ in rows],
            "candidates": pool,
            "threshold": alpha / m,
            "coefficients": [ssum(E[i][k] * yv[i] for i in range(n)) for k in sel],
        }
    )


sfmemb_fn = sfmemb


def cheatsheet() -> str:
    return "sfmemb(y, W, alpha=0.05) -> MEMs with Bonferroni-significant simple-regression t tests."
