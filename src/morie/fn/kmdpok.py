# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 5: the Direct Preference Optimization (DPO) loss."""

from . import _array_core as np

from ._richresult import RichResult
from .alrmt import alammar_reward_model_training_bt

__all__ = ["kamath_dpo_loss"]


def kamath_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta):
    r"""L_DPO = -E log sigmoid(beta * (logratio_w - logratio_l)).

    DPO's insight is that beta times the policy/reference log-ratio IS
    an implicit reward, and the loss around it is the ordinary
    Bradley-Terry one -- so the implicit rewards are formed here and
    the -log sigmoid of their difference is taken from
    ``morie.fn.alrmt`` instead of being written again.

    All four log-probability arguments are LOG probabilities (<= 0 is
    the usual case, and a positive one is rejected as a probability
    passed by mistake).

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Direct Preference
    Optimization; Rafailov et al. (2023).

    Examples
    --------
    >>> import math
    >>> out = kamath_dpo_loss(-1.0, -3.0, -2.0, -3.0, 1.0)
    >>> abs(out["estimate"] - math.log(1 + math.exp(-1.0))) < 1e-12
    True
    >>> out["implicit_reward_w"]
    [1.0]
    """
    arrays = [np.atleast_1d(np.asarray(v, dtype=float))
              for v in (logp_w, logp_l, logp_ref_w, logp_ref_l)]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError("the four log-probability arrays must line "
                         f"up; got shapes {sorted(shapes)}.")
    if arrays[0].size == 0:
        raise ValueError("no preference pairs were given.")
    for name, a in zip(("logp_w", "logp_l", "logp_ref_w",
                        "logp_ref_l"), arrays):
        if np.any(a > 0):
            raise ValueError(
                f"{name} holds LOG probabilities; a positive entry "
                "means a probability was passed instead.")
    b = float(beta)
    if b <= 0:
        raise ValueError(f"beta must be positive; got {b}.")
    lw, ll, rw, rl = arrays
    rew_w = b * (lw - rw)
    rew_l = b * (ll - rl)
    bt = alammar_reward_model_training_bt(rew_w, rew_l)
    return RichResult(payload={
        "estimate": bt["estimate"], "loss": bt["estimate"],
        "per_pair": bt["losses"], "pair_accuracy": bt["pair_accuracy"],
        "implicit_reward_w": [float(v) for v in rew_w],
        "implicit_reward_l": [float(v) for v in rew_l],
        "beta": b, "n": int(lw.size),
        "method": "DPO loss (Kamath Ch 5; the Bradley-Terry core in "
                  "alrmt)"})


def cheatsheet():
    return "kmdpok: BT loss on beta-scaled policy/reference log-ratios"
