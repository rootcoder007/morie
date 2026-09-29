# morie.fn -- function file (rootcoder007/morie)
"""Spatial filter residual autocorrelation reduction."""

from ._richresult import RichResult
from .sfilter import eigenvector_filtering


def sfredu(resid, W, tol=0.1):
    r"""Reduction of residual spatial autocorrelation achieved by an eigenvector filter.

    Moran eigenvectors are added to the regression of resid on a
    constant until the residual \|Moran's I\| drops below tol
    (Tiefelsdorf and Griffith 2007); reports Moran's I before and after, the
    absolute and relative reduction (I_0 - I_1) / I_0 and the selected
    eigenvectors (:func:`morie.fn.sfilter.eigenvector_filtering`).

    References
    ----------
    Tiefelsdorf, M. and Griffith, D. A. (2007). Semiparametric filtering of
    spatial autocorrelation: the eigenvector approach. *Environment and
    Planning A* 39, 1193-1221.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(8)] for i in range(8)]
    >>> r = sfredu([1.0, 2.0, 2.5, 4.0, 3.5, 3.0, 1.5, 1.0], W)
    >>> round(r["moran_before"], 10), round(r["moran_after"], 10)
    (0.5092085615, -0.7534180361)
    """
    base = eigenvector_filtering(resid, W, max_vectors=0)
    r = eigenvector_filtering(resid, W, criterion="moran", tol=tol)
    i0, i1 = base["residual_moran"], r["residual_moran"]
    return RichResult(
        payload={
            "moran_before": i0,
            "moran_after": i1,
            "reduction": i0 - i1,
            "relative_reduction": (i0 - i1) / i0 if i0 != 0 else float("nan"),
            "selected": r["selected"],
        }
    )


sfredu_fn = sfredu


def cheatsheet() -> str:
    return "sfredu(resid, W) -> residual Moran's I before/after eigenvector filtering and its reduction."
