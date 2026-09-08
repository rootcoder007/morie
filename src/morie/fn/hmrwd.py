# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reward function R(s, a, s')."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_reward_function"]


def geron_reward_function(s, a, s_next, R=None, gamma=1.0):
    """
    Reward function R(s, a, s').

    Formula: r_t = R(s_t, a_t, s_{t+1})

    Evaluates the reward for one transition or a whole trajectory. `R` is
    either a callable ``R(s, a, s') -> float`` or a lookup table indexed
    ``[s, a, s']`` (3-D) or ``[s, a]`` (2-D, reward independent of the
    landing state). Alongside the per-step rewards it returns the return
    ``G_t = sum_k gamma^k r_{t+k}`` computed backwards
    (``G_t = r_t + gamma*G_{t+1}``), which is the quantity every value
    method is trying to estimate.

    Parameters
    ----------
    s, a, s_next : scalar or array-like
        Transition, or three equal-length trajectory arrays.
    R : callable or array-like
        Reward function or table. Required.
    gamma : float, default 1.0
        Discount factor in [0, 1].

    Returns
    -------
    result : RichResult
        Keys: rewards, total_reward, returns, discounted_return,
        estimate, n, method.

    Examples
    --------
    >>> r = geron_reward_function(0, 1, 1, R=lambda s, a, sp: 1.0 if sp == 1 else 0.0)
    >>> float(r["total_reward"])
    1.0

    A three-step trajectory with rewards 1, 0, 2 and gamma = 0.5 has
    return 1 + 0.5*0 + 0.25*2 = 1.5:

    >>> table = [[[0.0, 1.0], [2.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]]
    >>> traj = geron_reward_function([0, 1, 1], [0, 0, 1], [1, 0, 1], R=table, gamma=0.5)
    >>> [float(v) for v in traj["rewards"]]
    [1.0, 0.0, 2.0]
    >>> float(traj["discounted_return"])
    1.5
    >>> [float(v) for v in traj["returns"]]
    [1.5, 1.0, 2.0]

    References
    ----------
    Géron Ch 19
    """
    if R is None:
        raise ValueError("geron_reward_function: R is required -- pass a callable R(s, a, s') or a lookup table")
    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"geron_reward_function: gamma must lie in [0, 1], got {g}")

    sa = np.atleast_1d(np.asarray(s))
    aa = np.atleast_1d(np.asarray(a))
    sn = np.atleast_1d(np.asarray(s_next))
    if not (sa.size == aa.size == sn.size):
        raise ValueError(
            f"geron_reward_function: s, a and s_next must be the same length, got {sa.size}, {aa.size}, {sn.size}"
        )
    if sa.size == 0:
        raise ValueError("geron_reward_function: the transition is empty")

    if callable(R):
        rewards = np.asarray([float(R(si, ai, spi)) for si, ai, spi in zip(sa, aa, sn)], dtype=float)
    else:
        T = np.asarray(R, dtype=float)
        if T.ndim not in (2, 3):
            raise ValueError(f"geron_reward_function: a reward table must be 2-D [s, a] or 3-D [s, a, s'], got {T.ndim}-D")
        idx = [sa.astype(int), aa.astype(int)] + ([sn.astype(int)] if T.ndim == 3 else [])
        for k, (name, arr) in enumerate(zip(("s", "a", "s_next"), idx)):
            if arr.min() < 0 or arr.max() >= T.shape[k]:
                raise ValueError(
                    f"geron_reward_function: {name} index out of range for a reward table of shape {T.shape}"
                )
        rewards = T[tuple(idx)].astype(float)
    if not np.all(np.isfinite(rewards)):
        raise ValueError("geron_reward_function: R produced non-finite rewards")

    returns = np.empty_like(rewards)
    acc = 0.0
    for t in range(rewards.size - 1, -1, -1):
        acc = rewards[t] + g * acc
        returns[t] = acc

    return RichResult(
        title="Reward function",
        summary_lines=[
            ("Transitions", int(rewards.size)),
            ("Total reward", float(np.sum(rewards))),
            ("Discounted return", float(returns[0])),
            ("gamma", g),
        ],
        interpretation=(
            "The reward is the only channel through which the task is specified; gamma < 1 makes "
            "distant rewards worth exponentially less and keeps the return finite on infinite horizons."
        ),
        payload={
            "rewards": rewards,
            "total_reward": float(np.sum(rewards)),
            "returns": returns,
            "discounted_return": float(returns[0]),
            "gamma": g,
            "estimate": float(returns[0]),
            "n": int(rewards.size),
            "method": "R(s, a, s') evaluated per transition with backward discounted returns",
        },
    )


def cheatsheet():
    return "hmrwd: Reward function R(s, a, s')"
