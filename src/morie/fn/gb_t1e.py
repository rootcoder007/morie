# morie.fn -- function file (rootcoder007/morie)
"""Attainable exact sizes of a test with a discrete null distribution."""

import math

from ._richresult import RichResult

__all__ = ['exactsize', 'gibbons_type1_error']


def exactsize(pmf, alpha=0.05, upper=True):
    """Attainable significance levels and the p-value of an observation.

    Section 1.2.9 (book p. 26).  For a discrete statistic only a finite
    set of exact sizes is attainable; the book lists them explicitly
    for the Bernoulli example (n = 5, theta = 0.5).  This returns the
    whole ladder of attainable one-tailed sizes, the largest that does
    not exceed alpha, and the corresponding critical value.

    Parameters
    ----------
    pmf : sequence of float
        Null probabilities over the support, in increasing order of the
        statistic.  Need not sum to exactly 1.
    alpha : float, optional
        Nominal level (default 0.05).
    upper : bool, optional
        Upper-tail rejection region (default True).

    Returns
    -------
    RichResult
        keys ``sizes`` (attainable tails, one per cut point),
        ``alpha_exact``, ``cut`` (index into the support),
        ``nlevels``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 1.2.9, p. 26.
    """
    p = [float(v) for v in pmf]
    k = len(p)
    if k < 1:
        raise ValueError("pmf must be non-empty.")
    alpha = float(alpha)
    if upper:
        sizes = []
        acc = 0.0
        for i in range(k - 1, -1, -1):
            acc += p[i]
            sizes.append(acc)
        sizes = list(reversed(sizes))
    else:
        sizes = []
        acc = 0.0
        for i in range(k):
            acc += p[i]
            sizes.append(acc)
    best = float("nan")
    cut = -1
    rng = range(k - 1, -1, -1) if upper else range(k)
    for i in rng:
        if sizes[i] <= alpha:
            best = sizes[i]
            cut = i
            break
    return RichResult(
        payload={
            "sizes": sizes,
            "alpha_exact": float(best),
            "cut": int(cut),
            "nlevels": int(k),
            "method": "attainable exact sizes of a discrete test (Sec. 1.2.9)",
        }
    )


gibbons_type1_error = exactsize
