"""Categorical distribution: probability mass, distribution function and draws.

Johnson, N. L., Kemp, A. W. & Kotz, S. (2005). Univariate Discrete Distributions, 3rd ed. Wiley, ch. 10.
"""

from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["categoricaldist"]


def categoricaldist(k=None, probs=(0.5, 0.5), n=0, seed=0):
    r"""P(Y = j) = p_j, j = 0, ..., K-1, with the p_j normalised to sum to one.

    Draws invert the cumulative probabilities at morie's Philox uniforms
    (identical in the R arm).

    Parameters
    ----------
    k : int or sequence, optional
    probs : sequence of non-negative float
    n : int
    seed : int

    Returns
    -------
    RichResult
        Keys: probs, pmf, cdf (at k), random, mean, var (of the category index).

    References
    ----------
    Johnson, Kemp & Kotz (2005). Univariate Discrete Distributions, 3rd ed., ch. 10.

    Examples
    --------
    >>> categoricaldist(1, probs=[1, 3])["pmf"]
    0.75
    """
    w = [float(v) for v in probs]
    if not w or min(w) < 0 or sum(w) <= 0:
        raise ValueError("probs must be non-negative with a positive sum")
    tot = sum(w)
    pr = [v / tot for v in w]
    cum = []
    s = 0.0
    for v in pr:
        s += v
        cum.append(s)
    cum[-1] = 1.0
    mean = sum(j * v for j, v in enumerate(pr))
    payload = {"probs": pr, "mean": mean, "var": sum((j - mean) ** 2 * v for j, v in enumerate(pr))}
    if k is not None:
        ks = [k] if isinstance(k, int) else list(k)
        if any(not 0 <= j < len(pr) for j in ks):
            raise ValueError("categories must be in 0..K-1")
        payload["pmf"] = pr[k] if isinstance(k, int) else [pr[j] for j in ks]
        payload["cdf"] = cum[k] if isinstance(k, int) else [cum[j] for j in ks]
    if n:
        payload["random"] = [
            next(j for j, v in enumerate(cum) if u <= v) for u in random_uniform(int(n), seed=seed, stream=0)
        ]
    return RichResult(title="Categorical distribution", summary_lines=[("categories", len(pr))], payload=payload)


def cheatsheet():
    return "categoricaldist: categorical pmf, cdf and draws."
