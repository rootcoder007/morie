# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 5.12: the DPO loss."""

import numpy as np

from ._richresult import RichResult
from .km065 import _bt_loss
from .km075 import _implicit_rewards

__all__ = ["kamath_ch5_dpo_loss"]


def kamath_ch5_dpo_loss(pi_theta, pi_ref, beta):
    """L_DPO = -E[log sigma(beta log[pi_theta(y_w)/pi_ref(y_w)]
    - beta log[pi_theta(y_l)/pi_ref(y_l)])].

    The maximum-likelihood loss over Eq 5.11's preference probability:
    a Bradley-Terry loss whose implicit reward is the policy's log
    ratio, so the MARGIN construction is km075's and the loss is
    km065's, both delegated. ``pi_theta`` and ``pi_ref`` are (n, 2)
    arrays of (winner, loser) probabilities.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Eq 5.12, printed
    p. 210.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch5_dpo_loss([[0.75, 0.25]], [[0.5, 0.5]], 1.0)
    >>> abs(out["estimate"] + math.log(0.75)) < 1e-12
    True
    >>> kamath_ch5_dpo_loss([[0.5, 0.5]], [[0.5, 0.5]], 1.0)["margins"]
    [0.0]
    """
    pairs_t = list(pi_theta)
    pairs_r = list(pi_ref)
    if not pairs_t:
        raise ValueError("no preference pairs; an expectation over an "
                         "empty dataset is undefined, not 0.")
    if len(pairs_t) != len(pairs_r):
        raise ValueError(
            f"pi_theta has {len(pairs_t)} pairs but pi_ref has "
            f"{len(pairs_r)}.")
    margins, rw_all, rl_all = [], [], []
    for p_i, q_i in zip(pairs_t, pairs_r):
        rw, rl, _ = _implicit_rewards(p_i, q_i, beta)
        margins.append(float(rw - rl))
        rw_all.append(float(rw))
        rl_all.append(float(rl))
    loss, per = _bt_loss(margins)
    return RichResult(payload={
        "estimate": loss, "margins": margins,
        "per_pair": [float(v) for v in per],
        "implicit_reward_w": rw_all, "implicit_reward_l": rl_all,
        "beta": float(beta), "n": len(margins),
        "method": "DPO loss (Kamath Eq 5.12)"})


def cheatsheet():
    return "km076: -mean log sigma(beta log-ratio margin)"
