# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""RLHF objective: expected reward minus a KL penalty against the reference model."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_rlhf_reward_kl_objective"]

_METHOD = "RLHF reward minus KL-to-reference objective"


def geron_rlhf_reward_kl_objective(rewards, policy_logprobs, ref_logprobs, beta=0.1):
    r"""Reward with a leash.

    .. math::
        J = \mathbb{E}[r(x, y)]
            - \beta\, \mathrm{KL}\bigl(\pi_{\theta}(\cdot|x)
              \,\|\, \pi_{\text{ref}}(\cdot|x)\bigr)

    The KL term is estimated from the sampled sequences as
    :math:`\mathbb{E}_{y \sim \pi_\theta}[\log\pi_\theta - \log\pi_{\text{ref}}]`,
    which is exactly what you can compute when you only have the
    completions the policy actually produced.  Without it the policy
    reward-hacks: it finds the degenerate strings the reward model scores
    highly and stops speaking like the reference model at all.  The
    estimator is unbiased but can go negative on a small sample, so the
    per-sample terms are returned rather than only their mean.

    Parameters
    ----------
    rewards : array-like, shape (n,)
        Reward-model scores of the sampled completions.
    policy_logprobs, ref_logprobs : array-like, shape (n,)
        Sequence log-probabilities under the policy and the frozen
        reference. Must be non-positive.
    beta : float, optional
        Non-negative KL coefficient.

    Returns
    -------
    RichResult
        Payload keys ``objective``, ``mean_reward``, ``kl``,
        ``per_sample``, ``kl_terms``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 15, RLHF section.

    Examples
    --------
    Mean reward 1, KL estimate ``(-1) - (-2) = 1``, ``beta = 0.5``:

    >>> r = geron_rlhf_reward_kl_objective([1.0], [-1.0], [-2.0], beta=0.5)
    >>> r["mean_reward"], r["kl"]
    (1.0, 1.0)
    >>> r["objective"]
    0.5

    A policy identical to the reference pays no penalty:

    >>> geron_rlhf_reward_kl_objective([1.0], [-1.0], [-1.0], beta=0.5)["objective"]
    1.0
    """
    r = np.asarray(rewards, dtype=float).ravel()
    lp = np.asarray(policy_logprobs, dtype=float).ravel()
    lr = np.asarray(ref_logprobs, dtype=float).ravel()
    if r.size == 0:
        raise ValueError("rewards is empty.")
    if not (r.shape == lp.shape == lr.shape):
        raise ValueError(
            f"rewards, policy_logprobs and ref_logprobs must have equal length; "
            f"got {r.size}, {lp.size}, {lr.size}."
        )
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(lp)) and np.all(np.isfinite(lr))):
        raise ValueError("rewards and log-probabilities must be finite.")
    if np.any(lp > 0) or np.any(lr > 0):
        raise ValueError("log-probabilities must be non-positive; these look like raw probabilities.")
    beta = float(beta)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError(f"beta must be finite and non-negative, got {beta}.")

    kl_terms = lp - lr
    kl = float(kl_terms.mean())
    per = r - beta * kl_terms
    obj = float(r.mean() - beta * kl)

    return RichResult(
        title="RLHF objective",
        summary_lines=[("Objective", obj), ("Mean reward", float(r.mean())), ("KL", kl)],
        payload={
            "objective": obj,
            "mean_reward": float(r.mean()),
            "kl": kl,
            "kl_terms": kl_terms.tolist(),
            "per_sample": per.tolist(),
            "beta": beta,
            "estimate": obj,
            "n": int(r.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grrlhf: J = mean r - beta * mean(log pi - log pi_ref); the KL leash stops reward hacking"
