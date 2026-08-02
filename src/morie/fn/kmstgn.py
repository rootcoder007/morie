# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stiennon et al. summarisation-from-human-feedback: reward-model
loss plus the KL-penalised RL objective."""

from . import _array_core as np

from ._richresult import RichResult
from .kmppok import kamath_ppo_rlhf_objective
from .kmrmloss import kamath_reward_model_training_loss

__all__ = ["kamath_summarize_from_feedback"]


def kamath_summarize_from_feedback(preferences, rewards, pi_logprobs,
                                   ref_logprobs, beta):
    """L_RM over summary preferences, and
    J_RLHF = E[r_phi] - beta * KL(pi || pi_ref).

    Both halves already exist here, so both are DELEGATED: the
    preference loss to ``morie.fn.kmrmloss`` (Bradley-Terry NLL) and
    the policy objective to ``morie.fn.kmppok``. What this module adds
    is the pairing -- the two stages reported together, with the
    reward model's pairwise accuracy next to the policy's objective,
    because a policy optimising a 50%-accurate reward model is
    optimising noise.

    ``preferences`` is a sequence of ``(score_chosen, score_rejected)``
    pairs from the reward model.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, learning to
    summarise from human feedback (Stiennon et al. 2020).

    Examples
    --------
    >>> import math
    >>> out = kamath_summarize_from_feedback(
    ...     [(2.0, 1.0), (3.0, 0.0)], [1.0, 3.0],
    ...     [math.log(0.5), math.log(0.5)],
    ...     [math.log(0.25), math.log(0.5)], 2.0)
    >>> exp_rm = (math.log(1 + math.exp(-1.0))
    ...           + math.log(1 + math.exp(-3.0))) / 2
    >>> abs(out["loss_rm"] - exp_rm) < 1e-12
    True
    >>> abs(out["objective"] - (2.0 - 2.0 * math.log(2) / 2)) < 1e-12
    True
    >>> out["rm_accuracy"]
    1.0
    """
    pairs = [tuple(p) for p in preferences]
    if not pairs:
        raise ValueError("no summary preference pairs supplied.")
    if any(len(p) != 2 for p in pairs):
        raise ValueError(
            "each preference must be (score_chosen, score_rejected).")
    w = np.array([p[0] for p in pairs], dtype=float)
    l = np.array([p[1] for p in pairs], dtype=float)
    rm = kamath_reward_model_training_loss(w, l)
    rl = kamath_ppo_rlhf_objective(rewards, pi_logprobs, ref_logprobs, beta)
    return RichResult(payload={
        "loss_rm": float(rm["estimate"]),
        "rm_accuracy": float(rm["accuracy"]),
        "rm_mean_margin": float(rm["mean_margin"]),
        "objective": float(rl["estimate"]),
        "mean_reward": float(rl["mean_reward"]),
        "kl_estimate": float(rl["kl_estimate"]),
        "beta": float(beta),
        "n_preferences": len(pairs),
        "estimate": float(rl["estimate"]),
        "n": int(rl["n"]),
        "method": "Summarisation from human feedback "
                  "(kmrmloss + kmppok)"})


def cheatsheet():
    return "kmstgn: kmrmloss on summary pairs + kmppok's KL-penalised objective"
