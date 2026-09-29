# morie.fn -- function file (rootcoder007/morie)
"""SEM Cochrane-Orcutt spatial filter transform."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def semflt(y, W, lam=0.3):
    r"""SEM Cochrane-Orcutt spatial filter transform.

    The spatial Cochrane-Orcutt transform ``y* = (I - lam W) y`` that
    whitens a spatial error process (Anselin 1988, sec. 6.2): after it, OLS of ``y*`` on
    the equally filtered regressors is the GLS / ML step for a given
    ``lam``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Series to filter.
    W : array-like, shape (n, n)
        Spatial weights.
    lam : float
        Autoregressive parameter.

    Returns
    -------
    SpatialResult
        ``statistic`` is ``lam``; ``local_values`` and
        ``extra["filtered"]`` the filtered series.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    Examples
    --------
    >>> [round(v, 12) for v in semflt([1.0, 2.0, 4.0], [[0, 1, 0], [.5, 0, .5], [0, 1, 0]], 0.5).local_values]
    [0.0, 0.75, 3.0]
    """
    yv, Wm = sd._vec(y), sd._mat(W)
    f = [a - float(lam) * b for a, b in zip(yv, sd._mv(Wm, yv))]
    return SpatialResult(name="semflt", statistic=float(lam), local_values=f, extra={"filtered": f})


semflt_fn = semflt


def cheatsheet() -> str:
    return "semflt(y, W, lam) -> (I - lam W) y"
