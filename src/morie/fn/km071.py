# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 5.7: the reward implied by an optimal policy (DPO)."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch5_dpo_reward_optimal"]


def _ratio_logs(pi_star, pi_ref):
    p = np.atleast_1d(np.asarray(
        [float(v) for v in (pi_star.values() if isinstance(pi_star, dict)
                            else pi_star)], dtype=float))
    q = np.atleast_1d(np.asarray(
        [float(v) for v in (pi_ref.values() if isinstance(pi_ref, dict)
                            else pi_ref)], dtype=float))
    if p.size == 0 or q.size == 0:
        raise ValueError("pi_star or pi_ref is empty.")
    if p.shape != q.shape:
        raise ValueError(
            f"pi_star has {p.size} entries but pi_ref has {q.size}.")
    if np.any(p <= 0) or np.any(q <= 0) or np.any(p > 1) or np.any(q > 1):
        raise ValueError("every probability must lie in (0, 1]; a zero "
                         "makes the log ratio undefined.")
    return np.log(p / q), p, q


def kamath_ch5_dpo_reward_optimal(pi_star, pi_ref, beta, Z=None):
    """r*(x,y) = beta log[pi*(y|x) / pi_ref(y|x)] + beta log Z(x).

    Eq 5.6 solved for r: every optimal policy IS a reward model, up to
    the prompt-level constant beta log Z(x). Feed km070's output back
    in with its Z and the original rewards come out -- the tests check
    exactly that round trip. Z defaults to 1, leaving r* determined up
    to that constant, which is all DPO needs since it cancels.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Eq 5.7, printed
    p. 209.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch5_dpo_reward_optimal([0.75, 0.25], [0.5, 0.5],
    ...                                     1.0, Z=2.0)
    >>> abs(out["r"][0] - math.log(3.0)) < 1e-12, round(out["r"][1], 12)
    (True, 0.0)
    """
    beta = float(beta)
    if beta <= 0:
        raise ValueError("beta must be strictly positive.")
    logs, p, q = _ratio_logs(pi_star, pi_ref)
    Zv = 1.0 if Z is None else float(Z)
    if Zv <= 0:
        raise ValueError("Z must be strictly positive.")
    r = beta * logs + beta * math.log(Zv)
    return RichResult(payload={
        "r": [float(v) for v in r], "log_ratio": [float(v) for v in logs],
        "beta": beta, "Z": Zv, "offset": float(beta * math.log(Zv)),
        "estimate": float(r[0]), "n": int(r.size),
        "method": "reward implied by an optimal policy (Kamath Eq 5.7)"})


def cheatsheet():
    return "km071: r* = beta log(pi*/pi_ref) + beta log Z"
