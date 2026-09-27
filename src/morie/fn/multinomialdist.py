"""Multinomial distribution: probability mass function and draws.

Johnson, N. L., Kotz, S. & Balakrishnan, N. (1997). Discrete Multivariate Distributions. Wiley, ch. 35.
"""

import math

from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["multinomialdist"]


def multinomialdist(x=None, size=None, probs=(0.5, 0.5), n=0, seed=0):
    r"""P(X = x) = size! / prod x_j! prod p_j^{x_j}, sum x_j = size, probabilities normalised.

    Draws: each of the ``size`` trials picks a category by inverting the cumulative
    probabilities at a Philox uniform (identical in the R arm).

    Parameters
    ----------
    x : sequence of counts, optional
    size : int
        Number of trials (defaults to sum(x)).
    probs : sequence of non-negative floats
    n : int
        Number of random count vectors.
    seed : int

    Returns
    -------
    RichResult
        Keys: pmf, logpmf, mean, cov, random.

    References
    ----------
    Johnson, Kotz & Balakrishnan (1997). Discrete Multivariate Distributions, ch. 35.
    Matches ``stats::dmultinom(x, size, prob)``.

    Examples
    --------
    >>> round(multinomialdist([1, 2, 1], probs=[0.2, 0.5, 0.3])["pmf"], 12)
    0.18
    """
    w = [float(v) for v in probs]
    if not w or min(w) < 0 or sum(w) <= 0:
        raise ValueError("probs must be non-negative with a positive sum")
    p = [v / sum(w) for v in w]
    K = len(p)
    if size is None:
        if x is None:
            raise ValueError("give size or x")
        size = int(sum(x))
    payload = {
        "mean": [size * v for v in p],
        "cov": [[size * (p[i] * (i == j) - p[i] * p[j]) for j in range(K)] for i in range(K)],
    }
    if x is not None:
        xs = [int(v) for v in x]
        if len(xs) != K or sum(xs) != size or min(xs) < 0:
            raise ValueError("x must have one non-negative count per category summing to size")
        lp = math.lgamma(size + 1) + sum(-math.lgamma(c + 1) + (c * math.log(q) if c else 0.0) for c, q in zip(xs, p))
        if any(c > 0 and q == 0 for c, q in zip(xs, p)):
            lp = -math.inf
        payload["logpmf"] = lp
        payload["pmf"] = math.exp(lp) if lp > -math.inf else 0.0
    if n:
        cum = []
        s = 0.0
        for v in p:
            s += v
            cum.append(s)
        cum[-1] = 1.0
        us = random_uniform(int(n) * int(size), seed=seed, stream=0)
        draws = []
        for r in range(int(n)):
            cnt = [0] * K
            for t in range(int(size)):
                u = us[r * int(size) + t]
                cnt[next(j for j, v in enumerate(cum) if u <= v)] += 1
            draws.append(cnt)
        payload["random"] = draws
    return RichResult(title="Multinomial distribution", summary_lines=[("size", size)], payload=payload)


def cheatsheet():
    return "multinomialdist: multinomial pmf and Philox draws."
