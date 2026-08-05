# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic-variance check for a marginal structural model."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["check_asymptote_msm"]


def _lcg(seed):
    """Park-Miller multiplicative LCG: a bootstrap needs an INDEPENDENT
    stream, not a low-discrepancy one -- a shared quasi-random sequence
    correlates the resample positions and shrinks the bootstrap spread."""
    state = [int(seed) % 2147483647 or 1]

    def nxt():
        state[0] = (state[0] * 48271) % 2147483647
        return (state[0] - 1) / 2147483646.0
    return nxt


def check_asymptote_msm(y, A=None, H=None, B=200, seed=42):
    """
    Asymptotic-variance check for an IPW marginal structural model

    Formula: compare E[psi^2]/n to the bootstrap

    The influence function of the IPW mean is
    psi_i = w_i (Y_i - theta), so the sandwich variance is
    sum w_i^2 (Y_i - theta)^2 / (sum w_i)^2.  If the estimator really is
    asymptotically linear that number matches the bootstrap variance;
    a systematic gap is the signal that the weights are too unstable for
    the normal approximation.  With unit weights the sandwich reduces to
    the plain sample variance over n, which is how the algebra is
    checked.

    Parameters
    ----------
    y : array-like
        Outcome.
    A : array-like or None
        Treatment indicator; None treats every unit as one arm.
    H : array-like or None
        IPW weights; None uses unit weights.
    B : int
        Bootstrap replicates.
    seed : int
        Seed of the deterministic bootstrap stream.

    Returns
    -------
    result : dict
        Keys: estimate (ratio of variances), theta, var_if, var_boot,
        ratio, ess, agree, n, B.

    References
    ----------
    van der Laan & Rose (2011), Targeted Learning, Springer, ch. 5.
    Funk, Westreich, Wiesen, Sturmer, Brookhart & Davidian (2011),
    Am. J. Epidemiology 173(7):761-767.
    """
    yv = core.vec(y)
    n = len(yv)
    if n < 2:
        raise ValueError("need at least two observations")
    w = [1.0] * n if H is None else core.vec(H)
    if len(w) != n:
        raise ValueError("y and H must have the same length")
    if any(v < 0.0 for v in w):
        raise ValueError("weights must be non-negative")
    if A is not None:
        av = core.vec(A)
        if len(av) != n:
            raise ValueError("y and A must have the same length")
    sw = sum(w)
    if sw <= 0.0:
        raise ValueError("weights must not sum to zero")
    theta = sum(w[i] * yv[i] for i in range(n)) / sw
    var_if = sum((w[i] * (yv[i] - theta)) ** 2 for i in range(n)) / (sw * sw)
    u = _lcg(seed)
    B = int(B)
    if B < 2:
        raise ValueError("B must be at least 2")
    reps = []
    for _ in range(B):
        sw_b = 0.0
        sy_b = 0.0
        for _ in range(n):
            k = int(u() * n)
            if k >= n:
                k = n - 1
            sw_b += w[k]
            sy_b += w[k] * yv[k]
        reps.append(sy_b / sw_b if sw_b > 0.0 else float("nan"))
    mb = sum(reps) / B
    var_b = sum((v - mb) ** 2 for v in reps) / (B - 1)
    ess = sw * sw / sum(v * v for v in w)
    ratio = var_b / var_if if var_if > 0.0 else float("nan")
    return RichResult(payload={
        "estimate": ratio,
        "theta": theta,
        "var_if": var_if,
        "var_boot": var_b,
        "ratio": ratio,
        "ess": ess,
        "agree": 1 if (ratio == ratio and 0.5 <= ratio <= 2.0) else 0,
        "n": n,
        "B": B,
        "method": "asymptotic-variance check for an IPW MSM",
    })


def cheatsheet():
    return "chasym: asymptotic-variance check for an IPW MSM"


# compact alias per ledger/NAMING.md
checkasymptotemsm = check_asymptote_msm
