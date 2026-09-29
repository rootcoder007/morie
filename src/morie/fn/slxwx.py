# morie.fn -- function file (rootcoder007/morie)
"""SLX WX construction and summary."""

from __future__ import annotations

import math

from . import _spdiag as sd
from ._containers import SpatialResult


def slxwx(X, W):
    r"""SLX WX construction and summary.

    The spatially lagged regressors ``W X_*`` of the SLX / spatial Durbin
    design ``[X, W X_*]``, ``X_*`` the non-constant columns of ``X`` (the
    lag of a constant is a constant for row-standardised ``W``; Halleck
    Vega and Elhorst 2015), with a per-column summary: the mean of ``W x_k``
    and the Pearson correlation of ``x_k`` with ``W x_k``.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    W : array-like, shape (n, n)
        Spatial weights.

    Returns
    -------
    SpatialResult
        ``statistic`` is the number of lagged columns; ``extra`` has ``WX``
        (n x k rows), ``lagged_columns``, ``design`` ``[X, W X_*]``,
        ``means`` and ``correlations``.

    References
    ----------
    Halleck Vega, S. and Elhorst, J. P. (2015). The SLX model. *Journal of Regional Science*, 55(3),
    339-363.

    Examples
    --------
    >>> r = slxwx([[1, 0.2], [1, 0.9], [1, 0.4]], [[0, 1, 0], [.5, 0, .5], [0, 1, 0]])
    >>> r.statistic, [round(v, 12) for v in r.extra["means"]]
    (1.0, [0.7])
    """
    Z, lag = sd.durbin(X, W)
    p = len(Z[0]) - len(lag)
    WX = [r[p:] for r in Z]
    means, cors = [], []
    for j, k in enumerate(lag):
        x = [r[k] for r in Z]
        w = [r[j] for r in WX]
        mx, mw = sd.ssum(x) / len(x), sd.ssum(w) / len(w)
        sxy = sd.ssum((a - mx) * (b - mw) for a, b in zip(x, w))
        sxx = sd.ssum((a - mx) ** 2 for a in x)
        syy = sd.ssum((b - mw) ** 2 for b in w)
        means.append(mw)
        cors.append(sxy / math.sqrt(sxx * syy) if syy > 0 else math.nan)
    return SpatialResult(
        name="slxwx",
        statistic=float(len(lag)),
        extra={"WX": WX, "lagged_columns": lag, "design": Z, "means": means, "correlations": cors},
    )


slxwx_fn = slxwx


def cheatsheet() -> str:
    return "slxwx(X, W) -> W X_* and its summary"
