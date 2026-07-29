# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Actor-critic with learned value baseline; advantage = r + gamma*V(s') - V(s)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_actor_critic_advantage"]

_METHOD = "Actor-critic one-step advantage (A2C)"


def geron_actor_critic_advantage(V, s, s_next, r, gamma, done=None):
    r"""One-step temporal-difference advantage used by A2C/A3C actors.

    .. math::
        A(s_t, a_t) = r_t + \gamma V(s_{t+1}) - V(s_t)

    The advantage replaces the Monte-Carlo return :math:`G_t` in the
    policy-gradient estimator, trading a little bias for a large drop in
    variance.  When a transition is terminal the bootstrap term is
    dropped, so ``A = r - V(s)``.

    Parameters
    ----------
    V : array-like
        Value table, ``V[i]`` the estimated value of state ``i``.
    s, s_next : array-like of int
        State indices before and after each transition.
    r : array-like
        Rewards, one per transition.
    gamma : float
        Discount factor in ``[0, 1]``.
    done : array-like of bool, optional
        Terminal flags; the bootstrap ``gamma*V(s')`` is zeroed where true.

    Returns
    -------
    RichResult
        Payload keys ``advantage``, ``td_target``, ``value_s``,
        ``value_s_next``, ``critic_loss`` (mean squared TD error),
        ``estimate`` (mean advantage), ``n``, ``method``.

    References
    ----------
    Géron Ch 19, Actor-Critic (A2C/A3C) section.

    Examples
    --------
    >>> r = geron_actor_critic_advantage([0.0, 1.0, 2.0], [0], [1], [1.0], 0.9)
    >>> round(r["advantage"][0], 6)
    1.9
    >>> term = geron_actor_critic_advantage([0.0, 1.0], [0], [1], [1.0], 0.9,
    ...                                     done=[True])
    >>> round(term["advantage"][0], 6)
    1.0
    """
    V = np.asarray(V, dtype=float).ravel()
    if V.size == 0:
        raise ValueError("V must contain at least one state value.")
    if not np.all(np.isfinite(V)):
        raise ValueError("V contains non-finite values.")
    s = np.asarray(s).ravel()
    s_next = np.asarray(s_next).ravel()
    r = np.asarray(r, dtype=float).ravel()
    if not (s.size == s_next.size == r.size):
        raise ValueError(
            f"s, s_next and r must have equal length, got {s.size}, "
            f"{s_next.size}, {r.size}."
        )
    if s.size == 0:
        raise ValueError("no transitions supplied.")
    if not np.issubdtype(s.dtype, np.integer) or not np.issubdtype(s_next.dtype, np.integer):
        s = s.astype(int)
        s_next = s_next.astype(int)
    if s.min() < 0 or s.max() >= V.size or s_next.min() < 0 or s_next.max() >= V.size:
        raise ValueError(f"state indices must lie in [0, {V.size - 1}].")
    gamma = float(gamma)
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}.")
    if done is None:
        mask = np.zeros(s.size, dtype=float)
    else:
        done_arr = np.asarray(done).ravel()
        if done_arr.size != s.size:
            raise ValueError(
                f"done must have one flag per transition ({s.size}), got {done_arr.size}."
            )
        mask = done_arr.astype(bool).astype(float)

    v_s = V[s]
    v_next = V[s_next]
    target = r + gamma * (1.0 - mask) * v_next
    adv = target - v_s

    return RichResult(
        title="Actor-critic advantage",
        summary_lines=[("Mean advantage", float(adv.mean()))],
        payload={
            "advantage": adv.tolist(),
            "td_target": target.tolist(),
            "value_s": v_s.tolist(),
            "value_s_next": v_next.tolist(),
            "critic_loss": float(np.mean(adv**2)),
            "gamma": gamma,
            "estimate": float(adv.mean()),
            "n": int(adv.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grac: actor-critic advantage A = r + gamma*V(s') - V(s)"
