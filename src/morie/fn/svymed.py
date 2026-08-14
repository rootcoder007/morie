# morie.fn -- function file (rootcoder007/morie)
"""Median of a complex survey sample."""

from . import _tail1core as C
from .svyqtl import survey_quantile

from ._richresult import RichResult

__all__ = ["survey_median"]


def survey_median(y, weights=None):
    """Design-weighted median: the weighted CDF inverted at one half.

    This is ``survey_quantile`` at ``p = 0.5``; the estimator, its
    Woodruff interval and its payload keys are shared rather than
    reimplemented.  Reported alongside is the weighted mean, so the
    skewness of the weighted distribution is visible at a glance.

    Formula: ``M = inf{t : F_w(t) >= 1/2}`` with
    ``F_w(t) = sum_{y_i <= t} w_i / sum_i w_i``.

    Parameters
    ----------
    y : array-like
        Observations.
    weights : array-like, optional
        Design weights; equal weights if omitted.

    Returns
    -------
    RichResult
        ``estimate`` (median), ``se``, ``lower``, ``upper``, ``F``,
        ``mean`` (weighted mean), ``sumw``, ``n``, ``method``.

    References
    ----------
    Francisco, C. A. & Fuller, W. A. (1991).  Quantile estimation with a
    complex survey design.  The Annals of Statistics 19(1):454-469.
    <https://doi.org/10.1214/aos/1176347993>
    """
    r = survey_quantile(y, weights, 0.5)
    yy = C.vec(y)
    n = len(yy)
    w = [1.0] * n if weights is None else C.vec(weights)
    tot = sum(w)
    mu = sum(w[i] * yy[i] for i in range(n)) / tot
    return RichResult(payload={
        "estimate": float(r["estimate"]), "se": float(r["se"]),
        "lower": float(r["lower"]), "upper": float(r["upper"]),
        "F": float(r["F"]), "mean": float(mu), "sumw": float(tot), "n": n,
        "method": "weighted median, F_w inverted at 1/2 [Francisco & Fuller 1991]"})


# CANONICAL TEST
# >>> r = survey_median([1.0, 2.0, 3.0, 4.0, 5.0], None)
# >>> assert abs(r["estimate"] - 3.0) < 1e-12
# >>> assert abs(r["mean"] - 3.0) < 1e-12


def cheatsheet():
    return "svymed(y, weights): weighted median with a Woodruff interval."

# public names resolved by fn/_lazy_map.json
surveymedian = survey_median
