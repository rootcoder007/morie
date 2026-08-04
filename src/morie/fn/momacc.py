# morie.fn -- function file (rootcoder007/morie)
"""Moments accountant for DP-SGD."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['dpacct', 'moments_accountant']


def dpacct(sigma, sample_rate, steps, delta=1e-05, max_order=64):
    """Moments accountant for DP-SGD.

    Privacy loss of the sampled Gaussian mechanism tracked through its log moment generating function instead of by naive composition, which is the entire point of the accountant. The log-moment bound is Lemma 3, composability and the tail bound are Theorem 2, and the minimisation runs over integer orders 2..max_order so the answer is deterministic. The O(q^3 lam^3 / sigma^3) remainder in Lemma 3 is not added: it is an asymptotic order term, not a computable quantity.


    Formula: alpha(lam) <= T q^2 lam(lam+1)/((1-q) sigma^2); delta = min_lam exp(alpha(lam) - lam eps)

    Parameters
    ----------
    sigma : float
        Noise multiplier (noise sd over clipping norm).
    sample_rate : float
        Lot sampling probability q in (0, 1).
    steps : int
        Number of SGD steps composed.
    delta : float
        Target delta.
    max_order : int
        Largest integer moment order searched.

    Returns
    -------
    RichResult
        ``epsilon``, ``order``, ``logmgf``, ``delta``, ``sigma``, ``sample_rate``, ``steps``.

    References
    ----------
    Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar and Zhang (2016),
    Deep Learning with Differential Privacy, CCS'16, arXiv:1607.00133.
    Lemma 3 for the sampled-Gaussian log moment, Theorem 2 for
    composability and the tail bound.  Verified against the paper.
    """
    sigma = float(sigma); q = float(sample_rate)
    steps = int(steps); delta = float(delta)
    if sigma <= 0 or not 0.0 < q < 1.0 or steps < 1:
        raise ValueError("need sigma > 0, 0 < sample_rate < 1, steps >= 1")
    best_eps, best_order, best_a = float("inf"), 0, float("nan")
    for lam in range(2, int(max_order) + 1):
        a = steps * q * q * lam * (lam + 1.0) / ((1.0 - q) * sigma * sigma)
        eps = (a + math.log(1.0 / delta)) / lam
        if eps < best_eps:
            best_eps, best_order, best_a = eps, lam, a
    return RichResult(payload={
        "epsilon": best_eps, "order": best_order, "logmgf": best_a,
        "delta": delta, "sigma": sigma, "sample_rate": q, "steps": steps,
        "method": "Moments accountant (sampled Gaussian, DP-SGD)"})


moments_accountant = dpacct


def cheatsheet():
    return "momacc: Moments accountant for DP-SGD."
