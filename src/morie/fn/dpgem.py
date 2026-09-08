# morie.fn -- function file (rootcoder007/morie)
"""GEM (stick-breaking) weights of a Dirichlet process."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["gem_distribution"]


def gem_distribution(alpha=1.0, K=10, seed=42):
    """
    GEM stick-breaking weights

    Formula: GEM(alpha) = stick-breaking with V_k ~ Beta(1, alpha)

    w_1 = V_1 and w_k = V_k prod_{j<k} (1 - V_j), so the weights and the
    unbroken remainder sum to exactly one at every truncation.  The
    marginal mean of the first weight is 1/(1 + alpha), and the expected
    remaining mass after K breaks is (alpha/(1 + alpha))^K.

    Parameters
    ----------
    alpha : float
        Concentration, strictly positive.  Larger alpha spreads the mass
        over more sticks.
    K : int
        Truncation level.
    seed : int
        Seed of the deterministic stream.

    Returns
    -------
    result : dict
        Keys: estimate (largest weight), weights, V, remaining,
        expected_remaining, K.

    References
    ----------
    Sethuraman (1994), Statistica Sinica 4(2):639-650.
    Pitman (2002), Poisson-Dirichlet and GEM invariant distributions,
    Technical Report 621, U.C. Berkeley.
    """
    if not (alpha > 0.0):
        raise ValueError("alpha must be strictly positive")
    K = int(K)
    if K < 1:
        raise ValueError("K must be at least 1")
    rng = np.random.default_rng(seed)
    V, w = [], []
    rest = 1.0
    for _ in range(K):
        v = float(rng.beta(1.0, alpha))
        V.append(v)
        w.append(v * rest)
        rest *= (1.0 - v)
    return RichResult(payload={
        "estimate": max(w),
        "weights": w,
        "V": V,
        "remaining": rest,
        "expected_remaining": (alpha / (1.0 + alpha)) ** K,
        "K": K,
        "method": "GEM stick-breaking weights",
    })


def cheatsheet():
    return "dpgem: GEM stick-breaking weights"


# compact alias per ledger/NAMING.md
gemdistribution = gem_distribution
