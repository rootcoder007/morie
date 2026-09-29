# morie.fn -- function file (rootcoder007/morie)
"""SLX spatial filter (de-mean with WX)."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def slxflt(X, W):
    r"""SLX spatial filter (de-mean with WX).

    ``(I - W) X``: each unit's regressors minus the weighted average of its
    neighbours' (the local deviation, spatial first difference used to sweep
    out neighbourhood-level effects; Anselin 1988, sec. 6.2, with ``rho =
    1``). Constant columns become zero under a row-standardised ``W``.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Regressors.
    W : array-like, shape (n, n)
        Spatial weights.

    Returns
    -------
    SpatialResult
        ``statistic`` is the number of columns; ``extra["filtered"]`` the
        n x p filtered matrix.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    Examples
    --------
    >>> r = slxflt([[1, 0.2], [1, 0.9], [1, 0.4]], [[0, 1, 0], [.5, 0, .5], [0, 1, 0]])
    >>> [[round(v, 12) for v in row] for row in r.extra["filtered"]]
    [[0.0, -0.7], [0.0, 0.6], [0.0, -0.5]]
    """
    Xm, Wm = sd._mat(X), sd._mat(W)
    WX = sd._mm(Wm, Xm)
    F = [[a - b for a, b in zip(r, s)] for r, s in zip(Xm, WX)]
    return SpatialResult(name="slxflt", statistic=float(len(Xm[0])), extra={"filtered": F})


slxflt_fn = slxflt


def cheatsheet() -> str:
    return "slxflt(X, W) -> (I - W) X"
