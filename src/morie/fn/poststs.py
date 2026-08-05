# morie.fn -- function file (rootcoder007/morie)
"""Post-stratified estimator of a population mean."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["poststratify"]


def _strata(y, stratum, Nh, who):
    y = [float(v) for v in C.vec(y)]
    s = [str(v) for v in (stratum if isinstance(stratum, (list, tuple)) else list(stratum))]
    if len(y) == 0:
        raise ValueError(who + ": y is empty")
    if len(s) != len(y):
        raise ValueError(who + ": stratum must have one entry per observation")
    if isinstance(Nh, dict):
        keys = [str(k) for k in Nh]
        sizes = {str(k): float(Nh[k]) for k in Nh}
    else:
        keys = []
        for v in s:
            if v not in keys:
                keys.append(v)
        vals = [float(v) for v in C.vec(Nh)]
        if len(vals) != len(keys):
            raise ValueError(who + ": Nh must give one size per stratum")
        sizes = {keys[i]: vals[i] for i in range(len(keys))}
    for k in s:
        if k not in sizes:
            raise ValueError(who + ": stratum " + k + " has no population size")
    for k in sizes:
        if sizes[k] <= 0.0:
            raise ValueError(who + ": stratum sizes must be positive")
    order = []
    for v in s:
        if v not in order:
            order.append(v)
    return y, s, sizes, order


def poststratify(y, stratum, Nh):
    """Reweight the sample means by known population stratum sizes.

    Post-stratification uses stratum sizes that were NOT used to draw the
    sample, so it can only be applied after the fact; the payoff is that
    it removes exactly the part of the sampling error that comes from the
    realised strata proportions differing from the population ones.  When
    the sample happens to be proportionally allocated the estimator
    collapses to the plain sample mean -- that identity is the anchor
    used here.

    Formula: ``ybar_post = sum_h (N_h / N) ybar_h`` with
    ``var = sum_h (N_h / N)^2 s_h^2 / n_h``.

    Parameters
    ----------
    y : array-like
        Observed values.
    stratum : sequence
        Stratum label per observation.
    Nh : dict or array-like
        Population size per stratum; an array is matched to the strata in
        order of first appearance.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``strata`` (count), ``N``, ``n``.

    References
    ----------
    Holt, D. & Smith, T. M. F. (1979).  Post stratification.  Journal of
    the Royal Statistical Society Series A 142(1):33-46.
    doi:10.2307/2344652.  The estimator and its conditional variance are
    the standard form given there; the paper is paywalled and not held
    locally, so the standard form was used rather than a rendered page.
    """
    y, s, sizes, order = _strata(y, stratum, Nh, "poststratify")
    N = 0.0
    for k in order:
        N += sizes[k]
    est = 0.0
    var = 0.0
    for k in order:
        vals = [y[i] for i in range(len(y)) if s[i] == k]
        nh = len(vals)
        if nh == 0:
            raise ValueError("poststratify: stratum " + k + " has no observations")
        mh = 0.0
        for v in vals:
            mh += v
        mh /= nh
        w = sizes[k] / N
        est += w * mh
        if nh > 1:
            ss = 0.0
            for v in vals:
                ss += (v - mh) * (v - mh)
            var += w * w * (ss / (nh - 1)) / nh
    return RichResult(payload={
        "estimate": est, "se": var ** 0.5, "variance": var,
        "strata": len(order), "N": N, "n": len(y),
        "method": "Post-stratified mean, sum_h (N_h/N) ybar_h"})


def cheatsheet():
    return "poststs: Post-stratification"
