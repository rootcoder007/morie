"""Influence function of an estimator at a point."""

from __future__ import annotations

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["influence_function"]


def wmean(v, w):
    s = 0.0
    sw = 0.0
    for i in range(len(v)):
        s += w[i] * v[i]
        sw += w[i]
    return s / sw


def wvar(v, w):
    m = wmean(v, w)
    s = 0.0
    sw = 0.0
    for i in range(len(v)):
        s += w[i] * (v[i] - m) * (v[i] - m)
        sw += w[i]
    return s / sw


def wmedian(v, w):
    """Lower weighted median: the smallest value whose cumulative weight
    reaches half the total."""
    o = sorted(range(len(v)), key=lambda i: v[i])
    tot = 0.0
    for e in w:
        tot += e
    acc = 0.0
    for i in o:
        acc += w[i]
        if acc >= 0.5 * tot:
            return v[i]
    return v[o[-1]]


_NAMED = {"mean": wmean, "var": wvar, "median": wmedian}


def _resolve(estimator, who):
    if callable(estimator):
        return estimator
    if isinstance(estimator, str) and estimator in _NAMED:
        return _NAMED[estimator]
    raise ValueError(who + ": estimator must be callable or one of 'mean', 'var', 'median'")


def influence_function(estimator, F, x, eps=1e-3):
    """IF(x; T, F) = lim_{e->0} [T((1-e)F + e delta_x) - T(F)] / e.

    Hampel, F. R. (1974), "The influence curve and its role in robust
    estimation", *Journal of the American Statistical Association*
    69(346), 383-393, doi:10.1080/01621459.1974.10482962, defines the
    influence curve as this limit; the paper is closed access with no
    open copy in any repository (Unpaywall reports oa_status "closed"),
    and the definition used here is the one already written in this
    module's own stub docstring, which matches every later statement of
    it (e.g. Hampel, Ronchetti, Rousseeuw and Stahel 1986, Section 2.1).

    F is a sample, taken as the empirical distribution putting mass
    1/n on each point, so the contaminated mixture is exactly
    representable: the same points with weight (1-e)/n and x with
    weight e.  T must therefore be a functional of a weighted sample,
    T(values, weights).  The three named ones are supplied:

        "mean"    weighted mean
        "var"     weighted variance, sum w (v - mu)^2 / sum w
        "median"  lower weighted median

    The quotient is evaluated at eps and at eps/2 and combined by
    Richardson extrapolation, 2 Q(eps/2) - Q(eps), which removes the
    O(eps) term.  For the mean the quotient is exactly x - mean(F) at
    every eps, since the mean is linear in the mixing weight; for the
    variance the limit is (x - mu)^2 - sigma^2.

    The median shows what an empirical F cannot do.  For odd n the
    lower weighted median does not move at all under a small added
    weight, so the quotient is exactly 0 whatever x is -- including
    x far outside the data.  For even n it is the opposite: the
    unweighted lower median sits exactly at the halfway crossing, so
    any weight added above it pushes the crossing to the next order
    statistic and the quotient is (x_(n/2+1) - x_(n/2))/eps, which
    diverges as eps -> 0.  Neither is the sign-based population
    formula IF = sign(x - m) / (2 f(m)); that one needs a density,
    which an empirical distribution does not have.  Both behaviours
    are properties of the empirical version rather than defects here,
    and both are asserted as such.

    Parameters
    ----------
    estimator : callable or str
        T(values, weights), or "mean", "var", "median".
    F : array-like
        The sample standing for F.
    x : float
        Where to evaluate the influence.
    eps : float
        Contamination weight used for the difference quotient.

    Returns
    -------
    estimate : the Richardson-extrapolated influence value
    raw      : the plain quotient at eps
    tf       : T(F) itself
    """
    T = _resolve(estimator, "influence_function")
    v = core.vec(F)
    n = len(v)
    if n == 0:
        raise ValueError("influence_function: F is empty")
    xs = core.vec(x)
    if len(xs) != 1:
        raise ValueError("influence_function: x must be a single point")
    x0 = xs[0]
    e = float(eps)
    if not 0.0 < e < 1.0:
        raise ValueError("influence_function: eps must lie strictly between 0 and 1")
    base = T(v, [1.0 / n] * n)

    def quot(h):
        vals = v + [x0]
        w = [(1.0 - h) / n] * n + [h]
        return (T(vals, w) - base) / h

    q1 = quot(e)
    q2 = quot(e / 2.0)
    return RichResult(payload={
        "estimate": 2.0 * q2 - q1,
        "raw": q1,
        "half": q2,
        "tf": base,
        "eps": e,
        "n": n,
        "method": "Influence function",
    })


def cheatsheet():
    return "infcrv: Influence function"
