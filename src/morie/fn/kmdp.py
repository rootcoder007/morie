# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 6: the (eps, delta)-differential privacy guarantee."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_differential_privacy"]


def kamath_differential_privacy(eps, delta, p_D=None, p_Dp=None):
    r"""P(M(D) in S) <= exp(eps) P(M(D') in S) + delta, for all D ~ D', S.

    With ``p_D`` and ``p_Dp`` given -- the mechanism's output
    probabilities on a neighboring pair of datasets over a set of
    events S -- the definition is CHECKED, in both directions (the
    neighbor relation is symmetric), and the worst-case slack is
    returned; a negative slack is a certified violation. Without
    them, the multiplicative bound exp(eps) and the additive delta
    are returned as the guarantee itself.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Differential
    Privacy; Dwork and Roth (2014).

    Examples
    --------
    >>> out = kamath_differential_privacy(1.0, 0.0, [0.5], [0.2])
    >>> round(out["estimate"], 6)      # e*0.2 - 0.5
    0.043656
    >>> out["guarantee"]
    True
    """
    e = float(eps)
    d = float(delta)
    if e < 0:
        raise ValueError("epsilon is a privacy loss and cannot be "
                         "negative.")
    if not (0.0 <= d <= 1.0):
        raise ValueError(f"delta must lie in [0, 1]; got {d}.")
    if (p_D is None) != (p_Dp is None):
        raise ValueError("give both p_D and p_Dp, or neither.")
    if p_D is None:
        return RichResult(payload={
            "estimate": math.exp(e), "guarantee": None,
            "multiplicative_bound": math.exp(e), "epsilon": e,
            "delta": d, "n": 0,
            "method": "(eps, delta)-DP bound, no mechanism supplied "
                      "(Kamath Ch 6)"})
    a = np.atleast_1d(np.asarray(p_D, dtype=float))
    b = np.atleast_1d(np.asarray(p_Dp, dtype=float))
    if a.shape != b.shape:
        raise ValueError(
            f"{a.size} probabilities on D but {b.size} on D'.")
    if a.size == 0:
        raise ValueError("no events S were given to check.")
    if np.any((a < 0) | (a > 1)) or np.any((b < 0) | (b > 1)):
        raise ValueError("mechanism outputs must be probabilities in "
                         "[0, 1].")
    slack = np.minimum(math.exp(e) * b + d - a,
                       math.exp(e) * a + d - b)
    worst = float(slack.min())
    return RichResult(payload={
        "estimate": worst, "guarantee": bool(worst >= -1e-12),
        "slack": [float(v) for v in slack],
        "worst_event": int(np.argmin(slack)),
        "multiplicative_bound": math.exp(e), "epsilon": e, "delta": d,
        "n": int(a.size),
        "method": "(eps, delta)-DP definition check (Kamath Ch 6)"})


def cheatsheet():
    return "kmdp: check P_D <= e^eps P_D' + delta both ways, worst slack"
