# morie.fn -- function file (rootcoder007/morie)
"""Inverse of the Freeman-Tukey double arcsine transformation."""

import math

from ._richresult import RichResult

__all__ = ["ma_freeman_tukey_inverse"]


def ma_freeman_tukey_inverse(ft, n_harmonic):
    """Get a proportion back out of the double-arcsine scale.

    The double arcsine stabilises the variance of a proportion, which is
    what makes it poolable, but the pooled value is then on a scale nobody
    can read.  Back-transforming is not a matter of inverting the formula
    twice: the transform depends on the sample size, and Miller's inverse
    substitutes the harmonic mean of the study sizes for it.  Using the
    arithmetic mean instead is a known way to get a proportion outside
    ``[0, 1]``.

    Formula: ``p = 0.5 (1 - sgn(cos t) sqrt(1 - (sin t + (sin t - 1/sin
    t)/n)^2))`` for ``t`` the double-arcsine value and ``n`` the harmonic
    mean sample size -- Miller (1978).

    Parameters
    ----------
    ft : float or array-like
        Value(s) on the double-arcsine scale.
    n_harmonic : float
        Harmonic mean of the study sample sizes, positive.

    Returns
    -------
    RichResult
        ``p`` (the back-transformed proportion(s)), ``n_harmonic``,
        ``clamped`` (how many values were pinned to 0 or 1).

    References
    ----------
    Miller, J. J. (1978).  The inverse of the Freeman-Tukey double arcsine
    transformation.  The American Statistician 32(4):138.
    doi:10.1080/00031305.1978.10479283.
    """
    n = float(n_harmonic)
    if n <= 0.0:
        raise ValueError("the harmonic mean sample size must be positive")
    scalar = not hasattr(ft, "__len__")
    vals = [float(ft)] if scalar else [float(t) for t in ft]
    out = []
    clamped = 0
    for t in vals:
        st = math.sin(t)
        ct = math.cos(t)
        if abs(st) < 1e-12:
            p = 0.0 if ct > 0.0 else 1.0
            clamped += 1
            out.append(p)
            continue
        inner = st + (st - 1.0 / st) / n
        q = 1.0 - inner * inner
        if q < 0.0:
            q = 0.0
        sgn = 1.0 if ct > 0.0 else (-1.0 if ct < 0.0 else 0.0)
        p = 0.5 * (1.0 - sgn * math.sqrt(q))
        if p < 0.0:
            p = 0.0
            clamped += 1
        elif p > 1.0:
            p = 1.0
            clamped += 1
        out.append(p)
    return RichResult(payload={
        "p": out[0] if scalar else out, "n_harmonic": n, "clamped": clamped,
        "method": "Freeman-Tukey double arcsine back-transformation"})


def cheatsheet():
    return "mafrti: Miller's inverse of the Freeman-Tukey double arcsine"
