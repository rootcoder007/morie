# morie.fn -- function file (rootcoder007/morie)
"""Blackwell-MacQueen predictive rule of the Dirichlet process."""

import math

from ._richresult import RichResult

__all__ = ["dp_exchangeable_distribution"]


def dp_exchangeable_distribution(partition, alpha=1.0):
    """
    Blackwell-MacQueen predictive rule

    Formula: P(z_n = k | z_{1:n-1}) = n_k / (n - 1 + alpha)

    with the remaining alpha/(n - 1 + alpha) going to a new block.  The
    rule is exchangeable: the probability of a partition depends only on
    the block sizes, through the EPPF
    alpha^K prod (n_k - 1)! / (alpha)_n, and the expected number of
    blocks is sum_{i=1..n} alpha/(alpha + i - 1).

    Parameters
    ----------
    partition : array-like
        Block label of each of the n observations already seated.
    alpha : float
        Concentration, strictly positive.

    Returns
    -------
    result : dict
        Keys: estimate (probability of a new block), probs, p_new,
        counts, K, log_eppf, expected_K, n.

    References
    ----------
    Blackwell & MacQueen (1973), Ann. Statist. 1(2):353-355.
    Pitman (2006), Combinatorial Stochastic Processes, Springer, ch. 3.
    """
    if not (alpha > 0.0):
        raise ValueError("alpha must be strictly positive")
    lab = list(partition)
    n = len(lab)
    if n == 0:
        raise ValueError("empty input: partition has no observations")
    keys = []
    for v in lab:
        if v not in keys:
            keys.append(v)
    counts = [sum(1 for w in lab if w == k) for k in keys]
    K = len(keys)
    denom = n + alpha
    probs = [c / denom for c in counts]
    p_new = alpha / denom
    log_eppf = K * math.log(alpha) + sum(math.lgamma(c) for c in counts) \
        + math.lgamma(alpha) - math.lgamma(alpha + n)
    expected_K = sum(alpha / (alpha + i) for i in range(n))
    return RichResult(payload={
        "estimate": p_new,
        "probs": probs,
        "p_new": p_new,
        "counts": counts,
        "K": K,
        "log_eppf": log_eppf,
        "expected_K": expected_K,
        "n": n,
        "method": "Blackwell-MacQueen predictive rule of the DP",
    })


def cheatsheet():
    return "dpedt: Blackwell-MacQueen predictive rule of the DP"


# compact alias per ledger/NAMING.md
dpexchangeabledistribution = dp_exchangeable_distribution
