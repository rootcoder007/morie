# morie.fn -- function file (rootcoder007/morie)
"""Biweight IRLS robust regression."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tukey_regression"]


def tukey_regression(X, y, c=4.685, n_iter=25):
    """Robust regression by iteratively reweighted least squares.

    Least squares gives a single bad row unbounded influence, because
    the squared loss has an unbounded derivative.  Reweighting with the
    biweight bounds it and, past the tuning constant, removes it.  The
    scale is re-estimated from the current residuals by MAD on each
    sweep, which is what stops the weights from being decided by a
    scale the outliers themselves inflated.

    Determinism: fixed sweeps, fixed start (ordinary least squares), no
    tolerance test.

    Formula: iterate ``beta <- argmin sum_i w_i (y_i - x_i' beta)^2``
    with ``w_i = [1 - (r_i / (c s))^2]^2`` inside ``|r_i| <= c s`` and
    ``0`` outside.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix; supply your own intercept column.
    y : array-like, shape (n,)
        Response.
    c : float, default 4.685
        Tuning constant.
    n_iter : int, default 25
        Reweighting sweeps.

    Returns
    -------
    RichResult
        ``estimate`` (coefficients), ``scale``, ``weights``,
        ``fitted``, ``resid``, ``n``.

    References
    ----------
    Beaton, A. E. & Tukey, J. W. (1974).  Technometrics 16:147-185.  The
    IRLS scheme and the MAD scale step follow Holland, P. W. & Welsch,
    R. E. (1977), Robust regression using iteratively reweighted
    least-squares, Communications in Statistics 6:813-827.
    """
    Xm = C.mat(X)
    yv = C.vec(y)
    n, p = C.shape(Xm)
    beta, fitted, resid, _ = C.lstsq(Xm, yv)
    w = [1.0] * n
    s = 1.0
    for _ in range(int(n_iter)):
        med = S.median(resid)
        s = S.median([abs(t - med) for t in resid]) / 0.6744897501960817
        if s <= 0.0:
            s = 1.0
        w = []
        for t in resid:
            u = t / (c * s)
            w.append((1.0 - u * u) ** 2 if abs(u) < 1.0 else 0.0)
        Xw = [[math.sqrt(w[i]) * Xm[i][j] for j in range(p)] for i in range(n)]
        yw = [math.sqrt(w[i]) * yv[i] for i in range(n)]
        beta, _, _, _ = C.lstsq(Xw, yw)
        fitted = [C.dot(Xm[i], beta) for i in range(n)]
        resid = [yv[i] - fitted[i] for i in range(n)]
    return RichResult(payload={
        "estimate": beta, "scale": s, "weights": w, "fitted": fitted,
        "resid": resid, "n": n,
        "method": "Biweight IRLS robust regression"})


def cheatsheet():
    return "tukrr: Biweight IRLS robust regression."
