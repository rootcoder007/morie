# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Credit assignment problem: which past actions caused reward."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_credit_assignment"]


def geron_credit_assignment(trajectory, gamma=0.95, lam=None, normalize=False):
    """
    Credit assignment problem: which past actions caused reward.

    Formula: eligibility traces or discounted returns assign credit

    Two answers to the same question are computed exactly:

    * ``returns[t] = sum_k gamma^k r_{t+k}``, the discounted return, which
      is what REINFORCE multiplies the log-probability by. It is built by
      the backward recursion ``G_t = r_t + gamma G_{t+1}``, so it is exact
      rather than a truncated sum.
    * ``eligibility[t]``, the accumulating trace ``e_t = gamma*lam*e_{t-1}
      + 1`` at each step, which is how TD(lambda) spreads a late reward
      over the actions that preceded it.

    ``credit`` is the share of the *total* discounted return attributable
    to each step, so it sums to 1 and reads directly as "how much of the
    outcome does this action own".

    ``trajectory`` may be a flat reward sequence or a sequence of
    ``(state, action, reward)`` tuples.

    Parameters
    ----------
    trajectory : sequence
        Rewards, or ``(state, action, reward)`` triples.
    gamma : float, default 0.95
        Discount factor in [0, 1].
    lam : float, optional
        Trace decay for eligibility; defaults to ``gamma``.
    normalize : bool, default False
        Standardise the returns (the usual variance-reduction trick).

    Returns
    -------
    result : RichResult
        Keys: returns, rewards, eligibility, credit, total_return,
        horizon, actions, estimate, n, method.

    Examples
    --------
    A single reward at the end is discounted back through the episode:

    >>> r = geron_credit_assignment([0.0, 0.0, 10.0], gamma=0.5)
    >>> [round(v, 12) for v in r["returns"]]
    [2.5, 5.0, 10.0]
    >>> round(r["total_return"], 12)
    2.5

    Credit sums to one and, with all reward at the end, is uniform in
    discounted terms:

    >>> r2 = geron_credit_assignment([1.0, 1.0, 1.0], gamma=1.0)
    >>> [round(v, 12) for v in r2["returns"]]
    [3.0, 2.0, 1.0]
    >>> round(sum(r2["credit"]), 12)
    1.0

    The eligibility trace accumulates while the episode runs:

    >>> [round(v, 12) for v in geron_credit_assignment([0.0, 0.0], gamma=1.0, lam=0.5)["eligibility"]]
    [1.0, 1.5]

    The effective horizon 1/(1-gamma) says how far credit reaches:

    >>> round(geron_credit_assignment([1.0], gamma=0.9)["horizon"], 6)
    10.0

    References
    ----------
    Géron Ch 19
    """
    if trajectory is None or len(trajectory) == 0:
        raise ValueError("geron_credit_assignment: trajectory is empty")
    seq = list(trajectory)
    actions = None
    if all(isinstance(e, (tuple, list)) and len(e) == 3 for e in seq):
        actions = [e[1] for e in seq]
        rewards = np.asarray([float(e[2]) for e in seq])
    else:
        try:
            rewards = np.asarray([float(e) for e in seq])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "geron_credit_assignment: trajectory must be rewards or (state, action, reward) triples"
            ) from exc
    if not np.all(np.isfinite(rewards)):
        raise ValueError("geron_credit_assignment: trajectory contains non-finite rewards")

    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"geron_credit_assignment: gamma must lie in [0, 1], got {gamma!r}")
    l = g if lam is None else float(lam)
    if not (0.0 <= l <= 1.0):
        raise ValueError(f"geron_credit_assignment: lam must lie in [0, 1], got {lam!r}")

    T = rewards.size
    ret = np.empty(T)
    acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = rewards[t] + g * acc
        ret[t] = acc

    elig = np.empty(T)
    e = 0.0
    for t in range(T):
        e = g * l * e + 1.0
        elig[t] = e

    total = float(ret[0])
    disc = np.array([g**t * rewards[t] for t in range(T)])
    credit = disc / total if total != 0 else np.zeros(T)

    out = ret.copy()
    if normalize:
        sd = float(ret.std())
        if sd == 0:
            raise ValueError("geron_credit_assignment: returns have zero variance; normalisation is undefined")
        out = (ret - ret.mean()) / sd

    return RichResult(
        title="Credit assignment",
        summary_lines=[("Total return", total), ("gamma", g), ("Effective horizon", float(1.0 / (1.0 - g)) if g < 1 else float("inf"))],
        interpretation="Discounted returns credit an action by what followed it; eligibility traces credit it by how recently it fired.",
        payload={
            "returns": out.tolist(),
            "raw_returns": ret.tolist(),
            "rewards": rewards.tolist(),
            "eligibility": elig.tolist(),
            "credit": credit.tolist(),
            "discounted_rewards": disc.tolist(),
            "total_return": total,
            "horizon": float(1.0 / (1.0 - g)) if g < 1 else float("inf"),
            "gamma": g,
            "lam": l,
            "actions": actions,
            "estimate": total,
            "n": int(T),
            "method": "discounted returns by backward recursion plus accumulating eligibility traces",
        },
    )


def cheatsheet():
    return "hmcrd: Credit assignment problem: which past actions caused reward"
