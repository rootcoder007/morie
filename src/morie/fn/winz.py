# morie.fn -- function file (rootcoder007/morie)
"""Winsorized mean."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["winsorized_mean"]


def winsorized_mean(x, alpha=0.1):
    """Mean after pulling the tails in to the quantiles.

    Trimming discards extreme observations; winsorizing keeps them but
    caps their value.  The difference matters for the variance, because
    the winsorized sample still has ``n`` observations, so the estimator
    has a defined and smaller standard error than the trimmed mean at
    the same alpha -- which is why Winsor rule survives in robust
    scale estimation.

    Formula: replace values below the alpha quantile with it, values
    above the ``1 - alpha`` quantile with that, then take the mean.

    Parameters
    ----------
    x : array-like
        Sample.
    alpha : float, default 0.1
        Fraction winsorized at each tail.

    Returns
    -------
    RichResult
        ``estimate``, ``lower``, ``upper``, ``n_changed``, ``n``.

    References
    ----------
    Dixon, W. J. (1960).  Simplified estimation from censored normal
    samples.  Annals of Mathematical Statistics 31:385-391, which is
    where Charles Winsor rule -- he did not publish it himself -- is
    set out and attributed to him.
    """
    v = C.vec(x)
    n = len(v)
    lo = S.quantile7(v, alpha)
    hi = S.quantile7(v, 1.0 - alpha)
    w = [lo if t < lo else (hi if t > hi else t) for t in v]
    changed = sum(1 for i in range(n) if w[i] != v[i])
    return RichResult(payload={
        "estimate": sum(w) / n, "lower": lo, "upper": hi,
        "n_changed": changed, "n": n, "method": "Winsorized mean"})


def cheatsheet():
    return "winz: Winsorized mean."
