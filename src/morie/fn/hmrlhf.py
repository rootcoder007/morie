# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reinforcement learning from human feedback."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_rlhf"]


def _softmax_rows(Z):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def geron_rlhf(policy, reward_model, prompts=None, beta=0.1, lr=0.5, epochs=500):
    """
    Reinforcement learning from human feedback (RLHF).

    Formula: train reward model then PPO on policy maximizing reward - KL penalty

    The KL penalty is not a regulariser bolted on for stability -- it is
    what stops reward hacking. The reward model is a fit to human
    comparisons and is only valid near the distribution it was trained
    on; without the penalty the policy walks off that distribution to
    wherever the model's errors are largest, and the "reward" it collects
    there is fiction.

    With a finite response set the objective
    E_pi[r] - beta * KL(pi || pi_ref) has a closed-form maximiser,

        pi*(a) proportional to pi_ref(a) * exp(r(a) / beta),

    so the ascent here can be CHECKED rather than trusted: the fitted
    policy is returned next to that optimum and the largest deviation is
    reported. beta -> 0 recovers the greedy argmax of the reward model,
    beta -> infinity pins the policy to the reference.

    Parameters
    ----------
    policy : array-like, shape (n_prompts, n_responses)
        Initial logits; their softmax is also the reference policy.
    reward_model : callable or array-like
        ``reward_model(prompt, response_index) -> float``, or a matrix of
        the same shape as ``policy``.
    prompts : sequence, optional
        Prompt identifiers passed to a callable reward model; defaults to
        the row indices.
    beta : float, default 0.1
        KL weight (positive).
    lr : float, default 0.5
        Ascent step (positive).
    epochs : int, default 500

    Returns
    -------
    result : RichResult
        Keys: policy, optimal_policy, max_deviation, objective,
        objective_history, mean_reward, kl, estimate, n, method.

    Examples
    --------
    One prompt, two responses, rewards 0 and 1, beta = 1 from a uniform
    reference: the optimum is softmax([0, 1]).

    >>> r = geron_rlhf([[0.0, 0.0]], [[0.0, 1.0]], beta=1.0, epochs=800, lr=0.5)
    >>> [round(float(p), 6) for p in r["optimal_policy"][0]]
    [0.268941, 0.731059]
    >>> bool(r["max_deviation"] < 1e-4)
    True

    The KL penalty is what keeps it off the greedy answer; drop beta and
    the policy chases the reward:

    >>> hot = geron_rlhf([[0.0, 0.0]], [[0.0, 1.0]], beta=0.01, epochs=800, lr=0.5)
    >>> bool(hot["policy"][0][1] > r["policy"][0][1])
    True
    >>> bool(hot["kl"] > r["kl"])
    True

    References
    ----------
    Geron Ch 15
    """
    Z0 = np.atleast_2d(np.asarray(policy, dtype=float)).astype(float)
    if Z0.ndim != 2 or Z0.size == 0:
        raise ValueError(f"geron_rlhf: policy must be a non-empty (n_prompts, n_responses) logit matrix, got shape {Z0.shape}")
    P, R = Z0.shape
    if R < 2:
        raise ValueError(f"geron_rlhf: need at least 2 candidate responses, got {R}")
    b = float(beta)
    if not np.isfinite(b) or b <= 0:
        raise ValueError(f"geron_rlhf: beta must be positive and finite, got {beta!r}")
    eta = float(lr)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"geron_rlhf: lr must be positive and finite, got {lr!r}")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_rlhf: epochs must be >= 1, got {epochs!r}")

    keys = list(range(P)) if prompts is None else list(prompts)
    if len(keys) != P:
        raise ValueError(f"geron_rlhf: {len(keys)} prompts for {P} policy rows")
    if callable(reward_model):
        rew = np.array([[float(reward_model(keys[i], j)) for j in range(R)] for i in range(P)])
    else:
        rew = np.asarray(reward_model, dtype=float)
        if rew.shape != (P, R):
            raise ValueError(f"geron_rlhf: reward matrix has shape {rew.shape}, expected {(P, R)}")
    if not np.all(np.isfinite(rew)):
        raise ValueError("geron_rlhf: the reward model produced non-finite values")

    ref = _softmax_rows(Z0)
    Z = Z0.copy()
    hist = []
    for _ in range(E):
        pi = _softmax_rows(Z)
        adv = rew - b * (np.log(pi / ref) + 1.0)
        obj = float(np.mean(np.sum(pi * rew, axis=1) - b * np.sum(pi * np.log(pi / ref), axis=1)))
        hist.append(obj)
        # dJ/dz = pi * (adv - sum_a pi_a adv_a)
        grad = pi * (adv - np.sum(pi * adv, axis=1, keepdims=True))
        Z = Z + eta * grad

    pi = _softmax_rows(Z)
    opt = ref * np.exp((rew - rew.max(axis=1, keepdims=True)) / b)
    opt = opt / opt.sum(axis=1, keepdims=True)
    kl = float(np.mean(np.sum(pi * np.log(pi / ref), axis=1)))
    mean_r = float(np.mean(np.sum(pi * rew, axis=1)))
    obj = float(mean_r - b * kl)
    hist.append(obj)

    return RichResult(
        title="RLHF (reward minus KL)",
        summary_lines=[("Mean reward", mean_r), ("KL from reference", kl), ("Objective", obj)],
        interpretation="The KL term keeps the policy where the reward model is valid; without it, reward hacking.",
        payload={
            "policy": pi,
            "reference_policy": ref,
            "optimal_policy": opt,
            "max_deviation": float(np.max(np.abs(pi - opt))),
            "objective": obj,
            "objective_history": hist,
            "mean_reward": mean_r,
            "kl": kl,
            "beta": b,
            "estimate": pi,
            "n": int(P),
            "method": "RLHF objective E[r] - beta KL(pi||pi_ref) maximised by gradient ascent",
        },
    )


def cheatsheet():
    return "hmrlhf: RLHF, reward maximisation under a KL penalty"
