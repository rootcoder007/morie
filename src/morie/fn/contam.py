# morie.fn -- function file (rootcoder007/morie)
"""Huber's epsilon-contamination neighbourhood."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["epsilon_contamination"]


def _huber_k(eps):
    """Solve 2 phi(k)/k - 2 Phi(-k) = eps/(1-eps) for k, by bisection."""
    if eps <= 0.0:
        return float("inf")
    if eps >= 1.0:
        return 0.0
    target = eps / (1.0 - eps)

    def g(k):
        return 2.0 * math.exp(-0.5 * k * k) / math.sqrt(2.0 * math.pi) / k \
            - 2.0 * core.pnorm(-k)

    lo, hi = 1e-8, 40.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if g(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def epsilon_contamination(epsilon, H, x=None):
    """
    Huber's epsilon-contamination neighbourhood

    Formula: F = (1 - eps) Phi + eps H

    The gross-error model: a fraction eps of the data comes from an
    arbitrary H instead of the nominal normal.  The least favourable
    member of this neighbourhood has a density that is normal in the
    middle and exponential in the tails, whose score is Huber's psi
    clipped at k, where k solves 2 phi(k)/k - 2 Phi(-k) = eps/(1-eps).
    That k is reported alongside the mixture itself.

    Parameters
    ----------
    epsilon : float
        Contamination fraction in [0, 1).
    H : array-like
        Sample from the contaminating distribution, used as its
        empirical distribution function.
    x : array-like or None
        Points at which the mixture cdf is evaluated.  None uses
        -3, -2, ..., 3.

    Returns
    -------
    result : dict
        Keys: estimate (Huber k), k, F, mean, var, eps, n_H.

    References
    ----------
    Huber (1964), Ann. Math. Statist. 35(1):73-101.
    """
    eps = float(epsilon)
    if not (0.0 <= eps < 1.0):
        raise ValueError("epsilon must lie in [0, 1)")
    h = core.vec(H)
    m = len(h)
    if m == 0:
        raise ValueError("empty input: H has no observations")
    if x is None:
        x = [float(i) for i in range(-3, 4)]
    else:
        x = core.vec(x)
    hs = sorted(h)
    F = []
    for v in x:
        c = 0
        for w in hs:
            if w <= v:
                c += 1
        F.append((1.0 - eps) * core.pnorm(v) + eps * c / m)
    mh = sum(hs) / m
    vh = sum((w - mh) ** 2 for w in hs) / m
    mean = eps * mh
    var = (1.0 - eps) * 1.0 + eps * (vh + mh * mh) - mean * mean
    k = _huber_k(eps)
    return RichResult(payload={
        "estimate": k,
        "k": k,
        "F": F,
        "mean": mean,
        "var": var,
        "eps": eps,
        "n_H": m,
        "method": "Huber epsilon-contamination neighbourhood",
    })


def cheatsheet():
    return "contam: Huber epsilon-contamination neighbourhood"


# compact alias per ledger/NAMING.md
epsiloncontamination = epsilon_contamination
