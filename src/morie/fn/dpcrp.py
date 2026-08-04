# morie.fn -- function file (rootcoder007/morie)
"""Chinese restaurant process seating."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['crp', 'chinese_restaurant_process']


def crp(n, alpha=1.0, u=None, seed=1):
    """Chinese restaurant process seating.

    The number of tables grows like alpha log n, not like n, which is the property that makes the process usable as a prior over an unbounded number of clusters. The expected table count sum_i alpha/(alpha + i - 1) is returned in closed form next to the realised count, so the draw can be checked against what it should be on average without simulating anything. Seating uses the shared minstd stream unless the caller supplies uniforms.


    Formula: P(z_i = k) = n_k / (i - 1 + alpha) for an occupied table k, and alpha / (i - 1 + alpha) for a new one

    Parameters
    ----------
    n : int
        Number of customers.
    alpha : float
        Concentration parameter, positive.
    u : array-like, optional
        Caller-supplied uniforms, one per customer.
    seed : int
        Seed of the shared minstd stream when ``u`` is omitted.

    Returns
    -------
    RichResult
        ``table`` (assignment per customer), ``counts``, ``n_tables``, ``expected_tables``, ``alpha``, ``n``.

    References
    ----------
    Aldous (1985), Exchangeability and related topics, Ecole d'Ete de
    Probabilites de Saint-Flour XIII; Pitman (2006), Combinatorial
    Stochastic Processes.  Neither is held locally; the seating rule and
    the E[K] = sum_i alpha/(alpha + i - 1) identity are the standard
    published forms.
    """
    n = int(n); a = float(alpha)
    if a <= 0:
        raise ValueError("alpha must be positive")
    if n < 1:
        raise ValueError("n must be at least 1")
    us = C.vec(u) if u is not None else None
    g = C.Lcg(seed) if us is None else None
    counts, table = [], []
    for i in range(n):
        draw = us[i] if us is not None else g.unif()
        tot = i + a
        acc, pick = 0.0, -1
        for k in range(len(counts)):
            acc += counts[k] / tot
            if draw < acc:
                pick = k
                break
        if pick < 0:
            counts.append(0)
            pick = len(counts) - 1
        counts[pick] += 1
        table.append(pick)
    ek = sum(a / (a + i) for i in range(n))
    return RichResult(payload={
        "table": table, "counts": counts, "n_tables": len(counts),
        "expected_tables": ek, "alpha": a, "n": n,
        "method": "Chinese restaurant process"})


chinese_restaurant_process = crp


def cheatsheet():
    return "dpcrp: Chinese restaurant process seating."
