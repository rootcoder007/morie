# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.32: the differential privacy guarantee."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_differential_privacy"]


def _mass(M, D, S, name):
    dist = M(D)
    if not isinstance(dist, dict) or not dist:
        raise ValueError(f"M({name}) must return a non-empty outcome "
                         "distribution.")
    p = np.asarray([float(v) for v in dist.values()], dtype=float)
    if np.any(p < 0) or abs(float(p.sum()) - 1.0) > 1e-8:
        raise ValueError(
            f"M({name}) must be a distribution; it sums to "
            f"{float(p.sum()):.6g}.")
    missing = [s for s in S if s not in dist]
    if missing:
        raise ValueError(
            f"the outcomes {missing!r} are absent from M({name}).")
    return float(sum(float(dist[s]) for s in S))


def kamath_ch6_differential_privacy(M, A, B, S, epsilon):
    """P[M(A) in S] <= e^epsilon P[M(B) in S] for neighbouring datasets.

    CHECKS the guarantee for one output set rather than asserting it:
    ``M`` is a callable dataset -> outcome distribution, ``A`` and
    ``B`` differ by one record, ``S`` is the outcome subset. The
    smallest epsilon that would hold, log(P_A / P_B), is returned as
    ``epsilon_required`` -- the interesting number when the test fails.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.32, printed
    p. 258.

    Examples
    --------
    >>> import math
    >>> M = lambda D: ({"o1": 0.6, "o2": 0.4} if D == "A"
    ...                else {"o1": 0.3, "o2": 0.7})
    >>> out = kamath_ch6_differential_privacy(M, "A", "B", ["o1"], 1.0)
    >>> out["satisfied"], abs(out["epsilon_required"] - math.log(2)) < 1e-12
    (True, True)
    >>> kamath_ch6_differential_privacy(M, "A", "B", ["o1"],
    ...                                 0.5)["satisfied"]
    False
    """
    if not callable(M):
        raise ValueError("M must be a callable dataset -> outcome "
                         "distribution.")
    outs = list(S)
    if not outs:
        raise ValueError("S is empty; the guarantee is vacuous for the "
                         "empty outcome set.")
    eps = float(epsilon)
    if eps < 0:
        raise ValueError("epsilon must be non-negative.")
    pA = _mass(M, A, outs, "A")
    pB = _mass(M, B, outs, "B")
    if pB <= 0:
        raise ValueError(
            "P[M(B) in S] is 0; no finite epsilon can bound a positive "
            "P[M(A) in S] against it.")
    required = float(math.log(pA / pB)) if pA > 0 else float("-inf")
    return RichResult(payload={
        "estimate": required, "epsilon_required": required,
        "satisfied": bool(pA <= math.exp(eps) * pB + 1e-12),
        "p_A": pA, "p_B": pB, "ratio": pA / pB, "epsilon": eps,
        "bound": float(math.exp(eps) * pB), "n": len(outs),
        "method": "differential privacy check (Kamath Eq 6.32)"})


def cheatsheet():
    return "km108: checks P[M(A) in S] <= e^eps P[M(B) in S]"
