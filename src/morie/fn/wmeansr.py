"""Weighted (Hajek) survey mean with linearised variance."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["weighted_mean_survey"]


def weighted_mean_survey(y, weights):
    """
    Horvitz-Thompson weighted survey mean

    Formula: ybar_w = sum_i w_i y_i / sum_i w_i

    With design weights ``w_i = 1 / pi_i`` the Horvitz-Thompson total is
    ``that_y = sum_i w_i y_i`` and the estimated population size is
    ``that_N = sum_i w_i``.  Their ratio is the Hajek mean written above.
    It is a ratio of two estimated totals rather than a linear
    statistic, so its variance is obtained by linearisation rather than
    directly.

    The variance estimator used is the standard with-replacement
    linearisation of that ratio,

        v(ybar_w) = n / ((n - 1) * (sum_i w_i) ** 2)
                    * sum_i w_i ** 2 * (y_i - ybar_w) ** 2,

    the residual technique of Section 5.6 of Sarndal, Swensson &
    Wretman applied to the Hajek ratio.  This is the same quantity
    ``survey::svymean`` reports for a design declared with
    ``ids = ~1``; the identity was checked numerically against that
    package when this function was written.

    ``weights`` are design weights ``1 / pi_i``, not frequencies.  Zero
    weights drop a unit; negative weights are rejected.

    Parameters
    ----------
    y : array-like
        Observed values for the sampled units.
    weights : array-like
        Design weights ``w_i = 1 / pi_i``, the same length as ``y``.

    Returns
    -------
    result : RichResult
        Keys: estimate, se, sum_weights, n, method.

    See Also
    --------
    morie.fn.ht_tot.horvitz_thompson_total : the unratioed HT total,
        which takes inclusion probabilities rather than weights.

    References
    ----------
    Horvitz D G & Thompson D J (1952).  A generalization of sampling
    without replacement from a finite universe.  Journal of the American
    Statistical Association 47(260), 663-685.

    Thompson S K (2012).  Sampling, 3rd ed.  Wiley.  Chapter 6 gives the
    ratio estimator and its linearised variance.

    Sarndal C-E, Swensson B & Wretman J (1992).  Model Assisted Survey
    Sampling.  Springer.  Section 5.6, the residual technique for ratios.
    """
    yv = [float(v) for v in np.atleast_1d(np.asarray(y, dtype=float)).tolist()]
    wv = [float(v) for v in np.atleast_1d(np.asarray(weights, dtype=float)).tolist()]
    n = len(yv)
    if len(wv) != n:
        raise ValueError("y and weights must have the same length")
    if n < 2:
        raise ValueError("need at least two sampled units")
    if any(v < 0.0 for v in wv):
        raise ValueError("design weights must be non-negative")
    sw = 0.0
    for v in wv:
        sw += v
    if sw <= 0.0:
        raise ValueError("design weights must have positive total")

    num = 0.0
    for i in range(n):
        num += wv[i] * yv[i]
    est = num / sw

    ss = 0.0
    for i in range(n):
        r = yv[i] - est
        ss += wv[i] * wv[i] * r * r
    var = n / ((n - 1.0) * sw * sw) * ss
    return RichResult(
        payload={
            "estimate": float(est),
            "se": float(np.sqrt(var)),
            "sum_weights": float(sw),
            "n": n,
            "method": "Hajek weighted survey mean (Horvitz-Thompson weights)",
        }
    )


def cheatsheet():
    return "wmeansr: Hajek weighted survey mean with linearised SE"

