# morie.fn -- function file (rootcoder007/morie)
"""Trimmed L-moments TL(s,t) of a sample."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["evt_trimmed_lmom"]


def _lchoose(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def evt_trimmed_lmom(x, s=0, t=0, order=2):
    """
    Trimmed L-moments TL(s,t) of a sample

    Formula: lambda_r^(s,t) = (1/r) sum_k (-1)^k C(r-1,k) E[X_(r+s-k):(r+s+t)]

    with the order statistic expectation estimated by the unbiased
    combinatorial weights
    E[X_(j:m)] = sum_i C(i-1,j-1) C(n-i,m-j) / C(n,m) x_(i).
    Trimming s observations from the left and t from the right makes the
    moments exist for heavy tails where ordinary L-moments do not.  With
    s = t = 0 they reduce to Hosking's L-moments, so lambda_1 is the
    sample mean and lambda_2 the L-scale.

    Parameters
    ----------
    x : array-like
        Sample.
    s : int
        Left trimming.
    t : int
        Right trimming.
    order : int
        Highest order r to compute.

    Returns
    -------
    result : dict
        Keys: lambda, estimate (lambda_order), tau (ratios), n.

    References
    ----------
    Elamir & Seheult (2003), Comput. Statist. Data Anal. 43(3):299-314.
    """
    xs = sorted(core.vec(x))
    n = len(xs)
    s = int(s)
    t = int(t)
    order = int(order)
    if n == 0:
        raise ValueError("empty input: x has no observations")
    if s < 0 or t < 0:
        raise ValueError("trimming parameters must be non-negative")
    if order < 1:
        raise ValueError("order must be at least 1")
    if n < order + s + t:
        raise ValueError("sample too small for the requested order and trimming")
    lam = []
    for r in range(1, order + 1):
        m = r + s + t
        tot = 0.0
        for k in range(r):
            j = r + s - k
            w = 0.0
            for i in range(1, n + 1):
                a = _lchoose(i - 1, j - 1) + _lchoose(n - i, m - j) - _lchoose(n, m)
                if a > float("-inf"):
                    w += math.exp(a) * xs[i - 1]
            sgn = -1.0 if (k % 2) else 1.0
            tot += sgn * math.exp(_lchoose(r - 1, k)) * w
        lam.append(tot / r)
    tau = [float("nan")] * len(lam)
    if len(lam) >= 2 and lam[1] != 0.0:
        for r in range(2, len(lam)):
            tau[r] = lam[r] / lam[1]
    return RichResult(payload={
        "lambda": lam,
        "estimate": lam[-1],
        "tau": tau,
        "n": n,
        "method": "trimmed L-moments TL(s,t)",
    })


def cheatsheet():
    return "evtlmom: trimmed L-moments TL(s,t)"


# compact alias per ledger/NAMING.md
evttrimmedlmom = evt_trimmed_lmom
