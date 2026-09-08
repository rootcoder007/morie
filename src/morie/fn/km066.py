# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 5.2: the KL-penalised RLHF reward."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch5_reward_kl_penalty"]


def _pos_prob(v, name):
    p = np.atleast_1d(np.asarray(v, dtype=float))
    if p.size == 0:
        raise ValueError(f"{name} is empty.")
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError(
            f"every entry of {name} must lie in (0, 1]; a zero makes the "
            "log ratio undefined.")
    return p


def kamath_ch5_reward_kl_penalty(x, y, pi_RL, pi_SFT, beta, r_theta=None):
    """R(x,y) = r_theta(x,y) - beta log[pi_RL(y|x) / pi_SFT(y|x)].

    A per-SAMPLE penalty (the sampled point estimate of the KL term),
    not an expectation: it charges the policy for the log-odds it has
    moved away from the supervised model on the response it actually
    produced. ``r_theta`` is a scalar or a callable (x, y) -> scalar
    and is required. beta = 0 returns the raw reward.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Eq 5.2, printed
    p. 199.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch5_reward_kl_penalty("p", "resp", 0.5, 0.25, 2.0,
    ...                                    r_theta=1.0)
    >>> abs(out["estimate"] - (1.0 - 2.0 * math.log(2.0))) < 1e-12
    True
    """
    if r_theta is None:
        raise ValueError("r_theta is required: Eq 5.2 penalises a reward, "
                         "and there is no default reward model.")
    beta = float(beta)
    if beta < 0:
        raise ValueError("beta must be non-negative.")
    p_rl = _pos_prob(pi_RL, "pi_RL")
    p_sft = _pos_prob(pi_SFT, "pi_SFT")
    if p_rl.shape != p_sft.shape:
        raise ValueError(
            f"pi_RL has shape {p_rl.shape} but pi_SFT has {p_sft.shape}.")
    r = float(r_theta(x, y)) if callable(r_theta) else float(r_theta)
    penalty = beta * np.log(p_rl / p_sft)
    R = r - penalty
    return RichResult(payload={
        "estimate": float(R[0]) if R.size == 1 else float(R.mean()),
        "penalised_reward": [float(v) for v in R],
        "raw_reward": r, "penalty": [float(v) for v in penalty],
        "beta": beta, "n": int(R.size),
        "method": "KL-penalised RLHF reward (Kamath Eq 5.2)"})


def cheatsheet():
    return "km066: R = r_theta - beta log(pi_RL / pi_SFT)"
