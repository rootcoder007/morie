# morie.fn -- function file (rootcoder007/morie)
"""Counterfactual (off-policy) evaluation of a recommendation policy."""

import numpy as np

from ._richresult import RichResult

__all__ = ["counterfactual_rec"]


def counterfactual_rec(actions, rewards, logging_prob, target_prob, reward_model=None, clip=None):
    r"""Off-policy value estimates: IPS, self-normalised IPS, and DR.

    From logged bandit feedback :math:`(a_i, r_i, \pi_0(a_i \mid x_i))`
    the value of a new policy is

    .. math::
        \hat V_{IPS} &= \frac1n \sum_i w_i r_i,
        \qquad w_i = \frac{\pi(a_i \mid x_i)}{\pi_0(a_i \mid x_i)}, \\
        \hat V_{SNIPS} &= \frac{\sum_i w_i r_i}{\sum_i w_i}, \\
        \hat V_{DR} &= \frac1n \sum_i
            \big[\hat q_i^\pi + w_i (r_i - \hat q_i(a_i))\big].

    IPS is unbiased but high-variance; SNIPS trades a little bias for
    much lower variance; DR uses a reward model and stays consistent if
    either the model or the propensities are right. Weight clipping
    bounds the variance at the cost of bias, and the effective sample
    size shows how much the reweighting cost.

    Parameters
    ----------
    actions : array-like of int, shape (n,)
        Logged action per round.
    rewards : array-like, shape (n,)
        Observed reward for the logged action.
    logging_prob : array-like, shape (n,)
        pi_0(a_i | x_i), strictly positive.
    target_prob : array-like, shape (n,) or (n, k)
        The new policy's probability of the logged action, or the full
        action distribution per round (needed for the DR direct term).
    reward_model : array-like, shape (n, k), optional
        Predicted reward for every action; enables the DR estimate.
    clip : float, optional
        Upper bound on the importance weights.

    Returns
    -------
    RichResult
        keys: ``ips``, ``snips``, ``dr`` (None without a reward
        model), ``ess``, ``max_weight``, ``n_clipped``, ``n``,
        ``method``.

    References
    ----------
    Dudik, M., Langford, J. & Li, L. (2011). Doubly robust policy
    evaluation and learning. *Proceedings of ICML-11*, 1097-1104.

    Swaminathan, A. & Joachims, T. (2015). The self-normalized
    estimator for counterfactual learning. *Advances in Neural
    Information Processing Systems 28*.
    """
    a = np.asarray(actions).ravel().astype(int)
    r = np.asarray(rewards, dtype=float).ravel()
    p0 = np.asarray(logging_prob, dtype=float).ravel()
    n = r.size
    if not (a.size == n and p0.size == n):
        raise ValueError("actions, rewards, logging_prob must have equal length.")
    if np.any(p0 <= 0):
        raise ValueError("logging_prob must be strictly positive.")

    tp = np.asarray(target_prob, dtype=float)
    if tp.ndim == 2:
        if tp.shape[0] != n:
            raise ValueError(f"target_prob has {tp.shape[0]} rows but rewards has {n}.")
        pi_a = tp[np.arange(n), a]
    else:
        pi_a = tp.ravel()
        if pi_a.size != n:
            raise ValueError("target_prob must have one entry per round or be (n, k).")

    w = pi_a / p0
    n_clipped = 0
    if clip is not None:
        clip = float(clip)
        if clip <= 0:
            raise ValueError(f"clip must be positive, got {clip}.")
        n_clipped = int((w > clip).sum())
        w = np.minimum(w, clip)

    ips = float((w * r).mean())
    snips = float((w * r).sum() / w.sum()) if w.sum() > 0 else float("nan")

    dr = None
    if reward_model is not None:
        q = np.asarray(reward_model, dtype=float)
        if q.ndim != 2 or q.shape[0] != n:
            raise ValueError("reward_model must be (n, k).")
        if tp.ndim != 2:
            raise ValueError("DR needs the full target distribution as (n, k) target_prob.")
        if tp.shape != q.shape:
            raise ValueError("target_prob and reward_model must have the same shape.")
        direct = (tp * q).sum(axis=1)
        dr = float((direct + w * (r - q[np.arange(n), a])).mean())

    return RichResult(
        payload={
            "ips": ips,
            "snips": snips,
            "dr": dr,
            "ess": float(w.sum() ** 2 / (w**2).sum()) if (w**2).sum() > 0 else 0.0,
            "max_weight": float(w.max()),
            "n_clipped": n_clipped,
            "n": int(n),
            "method": "Off-policy evaluation: IPS, self-normalised IPS, doubly robust",
        }
    )


def cheatsheet():
    return "counRS: IPS/SNIPS/DR off-policy value with weight clipping and ESS"
