# morie.fn -- function file (rootcoder007/morie)
"""Hadamard response estimator for local differential privacy."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['ldphr', 'hadamard_response']


def ldphr(counts, epsilon, n=None):
    """Hadamard response estimator for local differential privacy.

    Each user reports a single symbol drawn from the +1 support set C_x of row x of a Hadamard matrix, landing inside C_x with probability e^eps/(e^eps+1). Because distinct Hadamard rows agree on exactly half their entries, the probability that a report lands in C_x has a closed-form linear relation to p(x), and inverting it gives an unbiased estimator. The estimates can be negative -- the paper says so, and projecting onto the simplex is left to the caller rather than done silently here.


    Formula: phat(x) = 2(e^eps + 1)/(e^eps - 1) * (phat(C_x) - 1/2)

    Parameters
    ----------
    counts : array-like
        Per-symbol counts of reports falling in C_x.
    epsilon : float
        Local privacy parameter.
    n : int, optional
        Number of users; the sum of ``counts`` if omitted.

    Returns
    -------
    RichResult
        ``p``, ``p_set`` (the empirical p(C_x)), ``scale``, ``epsilon``, ``n``, ``k``.

    References
    ----------
    Acharya, Sun and Zhang (2019), Hadamard Response: Estimating
    Distributions Privately, Efficiently, and with Little Communication,
    AISTATS, PMLR 89.  Equations (8), (9) and (10).  Verified against the
    paper.
    """
    counts = C.vec(counts)
    eps = float(epsilon)
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    total = float(n) if n is not None else sum(counts)
    if total <= 0:
        raise ValueError("need at least one report")
    e = math.exp(eps)
    scale = 2.0 * (e + 1.0) / (e - 1.0)
    pset = [c / total for c in counts]
    p = [scale * (v - 0.5) for v in pset]
    return RichResult(payload={
        "p": p, "p_set": pset, "scale": scale, "epsilon": eps,
        "n": total, "k": len(counts),
        "method": "Hadamard response LDP distribution estimator"})


hadamard_response = ldphr


def cheatsheet():
    return "hkonly: Hadamard response estimator for local differential privacy."
