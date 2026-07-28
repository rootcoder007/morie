# morie.fn -- function file (rootcoder007/morie)
"""Hajek ratio-of-weights estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["hajek_estimator"]


def hajek_estimator(y, pi):
    r"""Hajek estimator of a population mean:

    .. math:: \hat{\bar Y}_{Haj}
              = \frac{\sum_{i\in s} y_i/\pi_i}
                     {\sum_{i\in s} 1/\pi_i}.

    The Horvitz-Thompson total divided by the ESTIMATED population
    size rather than a known one. It is biased in finite samples --
    a ratio of two random quantities is -- and it is nonetheless the
    usual default, because the bias is :math:`O(n^{-1})` while the
    variance reduction is first order.

    The reason it works is that numerator and denominator move
    together: a sample that happens to over-represent large weights
    inflates both, and the ratio cancels much of the error. That
    cancellation is the entire argument, and it fails when y is
    uncorrelated with the weights, where Hajek offers little over
    Horvitz-Thompson.

    Parameters
    ----------
    y : array-like
        Observed values.
    pi : array-like
        Inclusion probabilities in (0, 1].

    Returns
    -------
    RichResult
        keys: ``mean``, ``ht_mean_if_N_known``, ``weight_sum``,
        ``design_unbiased`` (False), ``bias_order``, ``n``,
        ``method``.
    """
    from ._survey import hajek_mean, ht_total

    yv = np.asarray(y, dtype=float).ravel()
    p = np.asarray(pi, dtype=float).ravel()
    if yv.size < 2:
        raise ValueError(f"need at least 2 sampled units, got {yv.size}.")
    m = hajek_mean(yv, p)
    wsum = float(np.sum(1.0 / p))
    return RichResult(payload={
        "mean": m, "ht_mean_if_N_known": ht_total(yv, p) / wsum,
        "weight_sum": wsum, "design_unbiased": False,
        "bias_order": "O(1/n), against a first-order variance reduction",
        "cancellation_note": "numerator and denominator move together; the gain "
                             "vanishes when y is uncorrelated with the weights",
        "n": int(yv.size),
        "method": "Hajek ratio estimator; biased but usually far less variable than HT"})


def cheatsheet():
    return "hjkest: trades O(1/n) bias for a first-order variance cut -- and the trade fails if y is unrelated to the weights"
