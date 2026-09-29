# morie.fn -- function file (rootcoder007/morie)
"""Local eigenvector spatial filter."""

from ._qpcore import ssum
from ._richresult import RichResult
from .sfilter import eigenvector_filtering, moran_eigenvectors


def sfloc(y, X, W, i=0, criterion="moran", tol=0.1):
    r"""The fitted eigenvector spatial filter E_S gamma and its value at unit i.

    The filter selected by :func:`morie.fn.sfilter.eigenvector_filtering` in
    y = b0 + X b + E_S gamma + e is a synthetic map variable whose value
    at each location is the local spatial component removed from y
    (Griffith 2003, ch. 4; Tiefelsdorf and Griffith 2007). Returns the whole
    filter, its value at unit i (0-based) and the selected eigenvectors.

    References
    ----------
    Griffith, D. A. (2003). *Spatial Autocorrelation and Spatial Filtering*.
    Springer.
    Tiefelsdorf, M. and Griffith, D. A. (2007). Semiparametric filtering of
    spatial autocorrelation: the eigenvector approach. *Environment and
    Planning A* 39, 1193-1221.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(8)] for i in range(8)]
    >>> X = [[0.1 * k] for k in range(8)]
    >>> r = sfloc([1.0, 2.0, 2.5, 4.0, 3.5, 3.0, 1.5, 1.0], X, W, i=3, criterion="aic")
    >>> round(r["value"], 10)
    1.4952559259
    """
    r = eigenvector_filtering(y, W, X, criterion=criterion, tol=tol)
    E = moran_eigenvectors(W)["vectors"]
    sel = r["selected"]
    gam = r["coefficients"][len(r["coefficients"]) - len(sel) :]
    filt = [ssum(E[u][k] * g for k, g in zip(sel, gam)) for u in range(len(E))]
    return RichResult(payload={"filter": filt, "value": filt[int(i)], "selected": sel, "coefficients": gam})


sfloc_fn = sfloc


def cheatsheet() -> str:
    return "sfloc(y, X, W, i=0) -> fitted eigenvector spatial filter E_S gamma and its value at unit i."
