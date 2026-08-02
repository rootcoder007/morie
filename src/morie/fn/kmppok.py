# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PPO-based RLHF policy objective: reward model minus a KL penalty."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ppo_rlhf_objective"]


def kamath_ppo_rlhf_objective(rewards, logp_theta, logp_ref, beta):
    """J(theta) = E[r_phi(x, y) - beta * log(pi_theta(y|x)/pi_ref(y|x))].

    The expectation is estimated by the sample mean over the batch,
    and the KL term by the per-sample log-ratio -- the k1 estimator
    the pipeline actually uses, which is unbiased but can be negative
    on a single sample. That is reported as ``kl_estimate`` and NOT
    clipped to zero, because clipping it silently biases the penalty.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, the RLHF PPO
    pipeline.

    Examples
    --------
    >>> import math
    >>> out = kamath_ppo_rlhf_objective(
    ...     [1.0, 3.0], [math.log(0.5), math.log(0.5)],
    ...     [math.log(0.25), math.log(0.5)], 2.0)
    >>> abs(out["kl_estimate"] - math.log(2) / 2) < 1e-12
    True
    >>> abs(out["estimate"] - (2.0 - 2.0 * math.log(2) / 2)) < 1e-12
    True
    """
    r = np.atleast_1d(np.asarray(rewards, dtype=float)).ravel()
    lt = np.atleast_1d(np.asarray(logp_theta, dtype=float)).ravel()
    lr = np.atleast_1d(np.asarray(logp_ref, dtype=float)).ravel()
    beta = float(beta)
    if not (r.size == lt.size == lr.size):
        raise ValueError(
            f"batch sizes disagree: {r.size} rewards, {lt.size} policy "
            f"log-probs, {lr.size} reference log-probs.")
    if r.size == 0:
        raise ValueError("the batch is empty.")
    if beta < 0:
        raise ValueError(
            f"beta must be non-negative; got {beta}. A negative "
            "coefficient pays the policy to leave the reference.")
    if np.any(lt > 0) or np.any(lr > 0):
        raise ValueError(
            "log-probabilities must be <= 0; these look like "
            "probabilities rather than logs.")
    ratio = lt - lr
    kl = float(ratio.mean())
    per = r - beta * ratio
    J = float(per.mean())
    return RichResult(payload={
        "estimate": J, "objective": J,
        "mean_reward": float(r.mean()),
        "kl_estimate": kl, "penalty": beta * kl,
        "per_sample": [float(v) for v in per],
        "beta": beta, "n": int(r.size),
        "method": "PPO-RLHF objective E[r] - beta * E[log pi/pi_ref]"})


def cheatsheet():
    return "kmppok: mean(r - beta*(logp_theta - logp_ref)); KL never clipped"
