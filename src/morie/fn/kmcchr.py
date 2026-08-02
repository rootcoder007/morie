# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 5: deep RL from human preferences (Christiano et al. 2017)."""

from . import _array_core as np

from ._richresult import RichResult
from .alrmt import alammar_reward_model_training_bt

__all__ = ["kamath_christiano_deep_rl_feedback"]


def kamath_christiano_deep_rl_feedback(trajectory_pairs, r_phi):
    r"""L = sum over preferred/rejected segment pairs of -log P(w > l).

    ``trajectory_pairs`` is a sequence of ``(sigma_w, sigma_l)`` pairs
    and ``r_phi`` the learned reward, a callable applied to each
    trajectory segment. The preference model is Bradley-Terry on the
    two segment returns, which is exactly ``morie.fn.alrmt``, so the
    loss and the pair accuracy come from there; what is added here is
    the segment scoring and the SUM (Christiano's objective is over
    the whole comparison set, so both the sum and the mean are
    reported).

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Deep RL from Human
    Preferences; Christiano et al. (2017).

    Examples
    --------
    >>> out = kamath_christiano_deep_rl_feedback([(2.0, 0.0)],
    ...                                          lambda seg: seg)
    >>> round(out["estimate"], 6)      # -log sigmoid(2)
    0.126928
    """
    if not callable(r_phi):
        raise ValueError("r_phi must be a callable reward applied to a "
                         "trajectory segment.")
    pairs = list(trajectory_pairs)
    if len(pairs) == 0:
        raise ValueError("no preference comparisons were given.")
    rw, rl = [], []
    for k, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError(
                f"comparison {k} is not a (sigma_w, sigma_l) pair.")
        rw.append(float(r_phi(pair[0])))
        rl.append(float(r_phi(pair[1])))
    if not np.all(np.isfinite(rw + rl)):
        raise ValueError("r_phi returned a non-finite return.")
    bt = alammar_reward_model_training_bt(rw, rl)
    return RichResult(payload={
        "estimate": float(np.sum(bt["losses"])),
        "mean_loss": bt["estimate"], "losses": bt["losses"],
        "pair_accuracy": bt["pair_accuracy"],
        "returns_preferred": rw, "returns_rejected": rl,
        "n": len(pairs),
        "method": "preference-based reward learning (Kamath Ch 5; the "
                  "Bradley-Terry core in alrmt)"})


def cheatsheet():
    return "kmcchr: score segment pairs with r_phi, then the alrmt BT loss"
