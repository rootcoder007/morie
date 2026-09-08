# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 5.11: the DPO preference with Z(x) already cancelled."""

from . import _array_core as np

from ._richresult import RichResult
from .km073 import kamath_ch5_pref_sigmoid_form

__all__ = ["kamath_ch5_dpo_pref_simplified"]


def _implicit_rewards(pi_star, pi_ref, beta):
    """(beta log(pi*/pi_ref)) for the (winner, loser) pair.

    Shared with km074 so the substituted and simplified forms cannot
    disagree.
    """
    beta = float(beta)
    if beta <= 0:
        raise ValueError("beta must be strictly positive.")
    p = [float(v) for v in (pi_star.values() if isinstance(pi_star, dict)
                            else pi_star)]
    q = [float(v) for v in (pi_ref.values() if isinstance(pi_ref, dict)
                            else pi_ref)]
    if len(p) != 2 or len(q) != 2:
        raise ValueError(
            "pi_star and pi_ref must each hold exactly two probabilities, "
            f"(winner, loser); got {len(p)} and {len(q)}.")
    if any(not (0.0 < v <= 1.0) for v in p + q):
        raise ValueError("every probability must lie in (0, 1].")
    return (beta * np.log(p[0] / q[0]), beta * np.log(p[1] / q[1]), beta)


def kamath_ch5_dpo_pref_simplified(pi_star, pi_ref, beta):
    """p* = sigma(beta log[pi*(y_w)/pi_ref(y_w)] -
    beta log[pi*(y_l)/pi_ref(y_l)]).

    The policy IS the reward model: substitute Eq 5.7 into Eq 5.9 and
    the intractable Z(x) cancels, leaving a quantity computable from
    two policies alone. The sigmoid is delegated to km073.
    ``pi_star`` and ``pi_ref`` are (winner, loser) probability pairs.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Eq 5.11, printed
    p. 210.

    Examples
    --------
    >>> out = kamath_ch5_dpo_pref_simplified([0.75, 0.25], [0.5, 0.5], 1.0)
    >>> round(out["estimate"], 12)      # sigma(log 3) = 3/4
    0.75
    >>> kamath_ch5_dpo_pref_simplified([0.5, 0.5], [0.5, 0.5],
    ...                                2.0)["estimate"]
    0.5
    """
    rw, rl, beta = _implicit_rewards(pi_star, pi_ref, beta)
    inner = kamath_ch5_pref_sigmoid_form([rw, rl])
    return RichResult(payload={
        "estimate": inner["estimate"], "margin": inner["margin"],
        "implicit_reward_w": float(rw), "implicit_reward_l": float(rl),
        "beta": beta, "n": 2,
        "method": "DPO preference, Z cancelled (Kamath Eq 5.11)"})


def cheatsheet():
    return "km075: sigma(beta log ratio_w - beta log ratio_l)"
