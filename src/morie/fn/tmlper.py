# morie.fn -- function file (rootcoder007/morie)
"""TMLE with a seasonal basis."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_periodic"]


def tmle_periodic(y, D, X, period, n_fourier=2):
    """Point-treatment TMLE with seasonality in both nuisance models.

    Season enters twice: it drives the outcome and it drives who gets
    treated.  Putting a Fourier basis into the propensity as well as the
    outcome model is what stops calendar time from masquerading as a
    treatment effect, and a basis is used rather than month dummies so
    that January and December stay neighbours.

    Formula: augment the covariates with
    ``cos(2 pi j t / p), sin(2 pi j t / p)`` for ``j = 1 .. n_fourier``,
    then target with ``H = D / g - (1 - D) / (1 - g)``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Covariates; the first column is taken as calendar time.
    period : float
        Length of one cycle in the units of that time column.
    n_fourier : int, default 2
        Harmonics.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``n_basis``, ``n``.

    References
    ----------
    Westreich, D. & Cole, S. R. (2010).  Invited commentary: positivity
    in practice.  American Journal of Epidemiology 171:674-677.  The
    targeting step is van der Laan & Rubin (2006), IJB 2(1):11.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    Xm = C.mat(X)
    n = len(yv)
    p = float(period)
    rows = []
    for i in range(n):
        t = Xm[i][0]
        r = [1.0] + list(Xm[i])
        for j in range(1, int(n_fourier) + 1):
            r.append(math.cos(2.0 * math.pi * j * t / p))
            r.append(math.sin(2.0 * math.pi * j * t / p))
        rows.append(r)
    res = S.tmle(yv, Dv, rows)
    return RichResult(payload={
        "estimate": res["psi"], "se": res["se"], "eps": res["eps"],
        "n_basis": 2 * int(n_fourier), "n": n,
        "method": "TMLE with a Fourier seasonal basis"})


tmleperiodic = tmle_periodic


def cheatsheet():
    return "tmlper: TMLE with a seasonal basis."
