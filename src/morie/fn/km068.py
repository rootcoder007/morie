# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 5.4: the PPO loss used to update the policy."""

import numpy as np

from ._richresult import RichResult
from .km069 import kamath_ch5_rlhf_objective

__all__ = ["kamath_ch5_ppo_loss"]


def kamath_ch5_ppo_loss(phi, x, y, r_theta, beta, pi_ref=None):
    """L(phi) = -E[r_theta(x,y) - beta D_KL(pi_RL || pi_REF)].

    The NEGATED Eq 5.5 objective averaged over prompts -- a loss, so
    smaller is better. Per prompt the quantity inside the expectation
    is exactly km069's, so it is delegated: ``phi`` gives the policy's
    distribution over each prompt's response set ``y``, ``pi_ref`` the
    reference model's, and ``r_theta`` is a callable (x, y) -> reward.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Eq 5.4, printed
    p. 200.

    Examples
    --------
    >>> r = lambda xi, yi: 1.0 if yi == "a" else 0.0
    >>> out = kamath_ch5_ppo_loss([[0.5, 0.5]], ["p"], [["a", "b"]],
    ...                           r, 1.0, pi_ref=[[0.5, 0.5]])
    >>> out["estimate"]
    -0.5
    """
    if pi_ref is None:
        raise ValueError("pi_ref is required: the KL term of Eq 5.4 has no "
                         "meaning without the reference policy.")
    pol, refs, xs, ys = list(phi), list(pi_ref), list(x), list(y)
    if not xs:
        raise ValueError("no prompts; an expectation over nothing is "
                         "undefined, not 0.")
    if not (len(pol) == len(refs) == len(xs) == len(ys)):
        raise ValueError(
            f"phi, pi_ref, x and y must have equal length; got {len(pol)}, "
            f"{len(refs)}, {len(xs)}, {len(ys)}.")
    if not callable(r_theta):
        raise ValueError("r_theta must be a callable (x, y) -> reward.")
    per = []
    for p_i, q_i, x_i, y_i in zip(pol, refs, xs, ys):
        rewards = [float(r_theta(x_i, resp)) for resp in y_i]
        per.append(kamath_ch5_rlhf_objective(p_i, q_i, rewards, beta))
    obj = np.asarray([float(o["estimate"]) for o in per], dtype=float)
    return RichResult(payload={
        "estimate": float(-obj.mean()),
        "per_prompt_objective": [float(v) for v in obj],
        "kl": [float(o["kl"]) for o in per],
        "expected_reward": [float(o["expected_reward"]) for o in per],
        "beta": float(beta), "n": len(xs),
        "method": "PPO loss = -mean RLHF objective (Kamath Eq 5.4)"})


def cheatsheet():
    return "km068: L = -mean(E[r] - beta KL), km069 negated"
