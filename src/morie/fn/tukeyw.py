# morie.fn -- function file (rootcoder007/morie)
"""Tukey biweight location estimate."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tukey_biweight"]


def tukey_biweight(y, c=4.685, n_iter=20):
    """Biweight M-estimate of location, with redescending weights.

    Huber weights taper but never vanish, so one point far enough out
    still moves the estimate.  The biweight redescends to exactly zero
    past ``c``, which is what makes the estimator resist a gross
    outlier rather than merely dampen it.  The price is a non-convex
    objective with more than one local solution, so the starting point
    matters -- the median and a MAD scale are used, which is the
    standard remedy.

    Determinism: a fixed number of reweighting sweeps, no convergence
    tolerance.

    Formula: ``w(r) = (1 - (r / c)^2)^2`` for ``|r| <= c`` and 0
    otherwise, with ``r = (y - mu) / s`` and ``s`` the MAD scaled to be
    consistent at the normal.

    Parameters
    ----------
    y : array-like
        Sample.
    c : float, default 4.685
        Tuning constant; 4.685 gives 95 percent Gaussian efficiency.
    n_iter : int, default 20
        Reweighting sweeps.

    Returns
    -------
    RichResult
        ``estimate`` (location), ``scale``, ``weights``, ``n``.

    References
    ----------
    Beaton, A. E. & Tukey, J. W. (1974).  The fitting of power series,
    meaning polynomials, illustrated on band-spectroscopic data.
    Technometrics 16:147-185 -- the published statement of the biweight
    Tukey had circulated from 1960.
    """
    v = C.vec(y)
    n = len(v)
    mu = S.median(v)
    s = S.median([abs(t - mu) for t in v]) / 0.6744897501960817
    if s <= 0.0:
        s = 1.0
    w = [1.0] * n
    for _ in range(int(n_iter)):
        w = []
        for t in v:
            r = (t - mu) / (c * s)
            w.append((1.0 - r * r) ** 2 if abs(r) < 1.0 else 0.0)
        sw = sum(w)
        if sw > 0.0:
            mu = sum(w[i] * v[i] for i in range(n)) / sw
    return RichResult(payload={
        "estimate": mu, "scale": s, "weights": w, "n": n,
        "method": "Tukey biweight location, MAD scale"})


def cheatsheet():
    return "tukeyw: Tukey biweight location estimate."
