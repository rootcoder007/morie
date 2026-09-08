# morie.fn -- function file (rootcoder007/morie)
"""Pitman-Yor two-parameter seating process."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["pitman_yor_process"]


def pitman_yor_process(n=100, alpha=1.0, sigma=0.5, seed=42):
    """
    Pitman-Yor two-parameter process

    Formula: P(new) = (alpha + sigma K) / (n + alpha)

    An occupied table takes (n_k - sigma)/(n + alpha), so the discount
    sigma moves mass from large tables to new ones and the number of
    blocks grows like n^sigma instead of log n.  Setting sigma = 0
    recovers the Dirichlet process exactly.

    Parameters
    ----------
    n : int
        Number of customers to seat.
    alpha : float
        Concentration; must exceed -sigma.
    sigma : float
        Discount in [0, 1).
    seed : int
        Seed of the deterministic stream.

    Returns
    -------
    result : dict
        Keys: estimate (number of blocks), K, counts, p_new, n.

    References
    ----------
    Pitman & Yor (1997), Ann. Probab. 25(2):855-900.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    if not (0.0 <= sigma < 1.0):
        raise ValueError("sigma must lie in [0, 1)")
    if not (alpha > -sigma):
        raise ValueError("alpha must exceed -sigma")
    rng = np.random.default_rng(seed)
    counts = [1]
    for i in range(1, n):
        K = len(counts)
        w = [(c - sigma) / (i + alpha) for c in counts]
        w.append((alpha + sigma * K) / (i + alpha))
        u = float(rng.uniform(0.0, 1.0))
        acc = 0.0
        pick = len(w) - 1
        for c in range(len(w)):
            acc += w[c]
            if u <= acc:
                pick = c
                break
        if pick == len(counts):
            counts.append(0)
        counts[pick] += 1
    K = len(counts)
    return RichResult(payload={
        "estimate": K,
        "K": K,
        "counts": counts,
        "p_new": (alpha + sigma * K) / (n + alpha),
        "n": n,
        "method": "Pitman-Yor two-parameter seating process",
    })


def cheatsheet():
    return "dpparit: Pitman-Yor two-parameter process"


# compact alias per ledger/NAMING.md
pitmanyorprocess = pitman_yor_process
