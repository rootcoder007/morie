# morie.fn -- function file (rootcoder007/morie)
"""SAR Cochrane-Orcutt-style spatial filter."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def sarfilt(y, W, rho=0.3):
    r"""SAR Cochrane-Orcutt-style spatial filter.

    The spatial Cochrane-Orcutt transform ``y* = (I - rho W) y`` that
    whitens a spatial lag process (Anselin 1988, sec. 6.2): after it, OLS of ``y*`` on
    the equally filtered regressors is the GLS / ML step for a given
    ``rho``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Series to filter.
    W : array-like, shape (n, n)
        Spatial weights.
    rho : float
        Autoregressive parameter.

    Returns
    -------
    SpatialResult
        ``statistic`` is ``rho``; ``local_values`` and
        ``extra["filtered"]`` the filtered series.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    Examples
    --------
    >>> [round(v, 12) for v in sarfilt([1.0, 2.0, 4.0], [[0, 1, 0], [.5, 0, .5], [0, 1, 0]], 0.5).local_values]
    [0.0, 0.75, 3.0]
    """
    yv, Wm = sd._vec(y), sd._mat(W)
    f = [a - float(rho) * b for a, b in zip(yv, sd._mv(Wm, yv))]
    return SpatialResult(name="sarfilt", statistic=float(rho), local_values=f, extra={"filtered": f})


sarfilt_fn = sarfilt


def cheatsheet() -> str:
    return "sarfilt(y, W, rho) -> (I - rho W) y"
