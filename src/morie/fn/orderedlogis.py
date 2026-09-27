"""Ordered logistic (cumulative logit) distribution of an ordinal response.

McCullagh, P. (1980). Regression models for ordinal data. JRSS B 42, 109-142.
"""

import math

from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["orderedlogis"]


def orderedlogis(k=None, eta=0.0, cuts=(-1.0, 1.0), n=0, seed=0):
    r"""P(Y = j) = F(c_j - eta) - F(c_{j-1} - eta), j = 0, ..., J-1, F logistic, c_{-1} = -inf, c_{J-1} = inf.

    ``cuts`` are the J - 1 increasing cut points and ``eta`` the linear predictor,
    as in Stan's ordered_logistic. Draws invert morie's Philox uniforms.

    Parameters
    ----------
    k : int or sequence, optional
        Categories 0..J-1 at which to evaluate.
    eta : float
    cuts : sequence of float
        Increasing cut points.
    n : int
    seed : int

    Returns
    -------
    RichResult
        Keys: probs (all categories), pmf, cdf (at k), random.

    References
    ----------
    McCullagh, P. (1980). JRSS B 42, 109-142.

    Examples
    --------
    >>> round(sum(orderedlogis(eta=0.3, cuts=[-1, 0.5, 2])["probs"]), 12)
    1.0
    """
    c = [float(v) for v in cuts]
    if any(b <= a for a, b in zip(c, c[1:])) or not c:
        raise ValueError("cuts must be non-empty and strictly increasing")

    def F(t):
        return 1 / (1 + math.exp(-t)) if t >= 0 else math.exp(t) / (1 + math.exp(t))

    cum = [F(v - eta) for v in c] + [1.0]
    probs = [cum[0]] + [cum[j] - cum[j - 1] for j in range(1, len(cum))]
    payload = {"probs": probs}
    if k is not None:
        ks = [k] if isinstance(k, int) else list(k)
        if any(not 0 <= j < len(probs) for j in ks):
            raise ValueError("categories must be in 0..J-1")
        payload["pmf"] = [probs[j] for j in ks] if not isinstance(k, int) else probs[k]
        payload["cdf"] = [cum[j] for j in ks] if not isinstance(k, int) else cum[k]
    if n:
        payload["random"] = [
            next(j for j, v in enumerate(cum) if u <= v) for u in random_uniform(int(n), seed=seed, stream=0)
        ]
    return RichResult(
        title="Ordered logistic distribution", summary_lines=[("categories", len(probs))], payload=payload
    )


def cheatsheet():
    return "orderedlogis: ordinal probabilities F(c_j - eta) - F(c_{j-1} - eta)."
