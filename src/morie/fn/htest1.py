# morie.fn -- function file (rootcoder007/morie)
"""Horvitz-Thompson estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["horvitz_thompson"]


def horvitz_thompson(y, pi, N=None):
    r"""Horvitz-Thompson estimator of a population total:

    .. math:: \hat T_{HT} = \sum_{i \in s} \frac{y_i}{\pi_i}.

    Unbiased for ANY sampling design with strictly positive
    inclusion probabilities, and that is its whole claim: no model
    for y, no distributional assumption, only the design. Each unit
    stands for :math:`1/\pi_i` units of the population.

    Its weakness is the other side of the same coin. Because it uses
    no information beyond the weights, it does not know that the
    weights should sum to the population size. When they happen to
    sum to much more or less, the estimate inherits that error
    directly -- which is exactly what the Hajek estimator of
    :mod:`morie.fn.hjkest` corrects, at the cost of introducing
    bias. ``weight_sum`` and ``implied_N`` are returned so that
    discrepancy is visible.

    Parameters
    ----------
    y : array-like
        Observed values for the sampled units.
    pi : array-like
        Inclusion probabilities in (0, 1].
    N : int, optional
        Known population size, for comparison with the implied one.

    Returns
    -------
    RichResult
        keys: ``total``, ``mean``, ``weight_sum``, ``implied_N``,
        ``N``, ``design_unbiased`` (True), ``uses_known_N`` (False),
        ``n``, ``method``.
    """
    from ._survey import ht_total

    yv = np.asarray(y, dtype=float).ravel()
    p = np.asarray(pi, dtype=float).ravel()
    if yv.size < 1:
        raise ValueError("need at least one sampled unit.")
    total = ht_total(yv, p)
    wsum = float(np.sum(1.0 / p))
    return RichResult(payload={
        "total": total, "mean": total / wsum,
        "weight_sum": wsum, "implied_N": wsum,
        "N": None if N is None else int(N),
        "design_unbiased": True, "uses_known_N": False,
        "n": int(yv.size),
        "method": "Horvitz-Thompson total; design-unbiased for any positive-pi design"})


def cheatsheet():
    return "htest1: unbiased for ANY design -- but it never uses the known N, so errors in the weights pass straight through"
