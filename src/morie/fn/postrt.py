# morie.fn -- function file (rootcoder007/morie)
"""Post-stratification as a weight adjustment."""

from . import _tail1core as C

from ._richresult import RichResult
from .poststs import _strata

__all__ = ["post_stratification"]


def post_stratification(y, weights, stratum, N_h):
    """Rescale design weights so each stratum's weights sum to ``N_h``.

    This is the same method as ``morie.fn.poststs.poststratify`` seen
    from the weight side rather than the estimator side: with unit design
    weights the two return the same number, which is asserted in the
    batch anchors.  The calibration property -- the adjusted weights sum
    to ``N_h`` within every stratum, exactly -- is what makes the
    adjustment worth doing and is also asserted.

    Formula: ``w_i' = w_i N_h / sum_{j in h} w_j``; the estimate is the
    weighted mean ``sum w_i' y_i / sum w_i'``.

    Parameters
    ----------
    y : array-like
        Observed values.
    weights : array-like
        Design weights, positive.
    stratum : sequence
        Stratum label per observation.
    N_h : dict or array-like
        Population size per stratum.

    Returns
    -------
    RichResult
        ``estimate``, ``weights`` (the adjusted weights), ``factors``
        (one adjustment factor per stratum, in order of first
        appearance), ``N``, ``n``.

    References
    ----------
    Holt, D. & Smith, T. M. F. (1979).  Post stratification.  Journal of
    the Royal Statistical Society Series A 142(1):33-46.
    doi:10.2307/2344652.  Standard form, as for ``poststs``.
    """
    y, s, sizes, order = _strata(y, stratum, N_h, "post_stratification")
    w = [float(v) for v in C.vec(weights)]
    if len(w) != len(y):
        raise ValueError("post_stratification: weights must have one entry per observation")
    for v in w:
        if v <= 0.0:
            raise ValueError("post_stratification: weights must be positive")
    fac = []
    wa = list(w)
    for k in order:
        tot = 0.0
        for i in range(len(y)):
            if s[i] == k:
                tot += w[i]
        if tot <= 0.0:
            raise ValueError("post_stratification: stratum " + k + " has no weight")
        f = sizes[k] / tot
        fac.append(f)
        for i in range(len(y)):
            if s[i] == k:
                wa[i] = w[i] * f
    sw = 0.0
    swy = 0.0
    for i in range(len(y)):
        sw += wa[i]
        swy += wa[i] * y[i]
    return RichResult(payload={
        "estimate": swy / sw, "weights": wa, "factors": fac,
        "N": sw, "n": len(y), "strata": len(order),
        "method": "Post-stratification weight adjustment w_i N_h / sum_h w"})


def cheatsheet():
    return "postrt: Post-stratification weight adjustment"
