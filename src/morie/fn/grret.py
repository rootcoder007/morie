# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Discounted return G_t from step t onward."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_discounted_return", "returns_curve"]

_METHOD = "Discounted return"


def returns_curve(rewards, gamma):
    """Full ``G_t`` curve by one backward sweep; shared with REINFORCE."""
    r = np.asarray(rewards, dtype=float).ravel()
    if r.size == 0:
        raise ValueError("rewards is empty; the return of an empty episode is undefined.")
    if not np.all(np.isfinite(r)):
        raise ValueError("rewards contains non-finite values.")
    gamma = float(gamma)
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}.")
    G = np.empty_like(r)
    acc = 0.0
    for t in range(r.size - 1, -1, -1):
        acc = r[t] + gamma * acc
        G[t] = acc
    return G, gamma


def geron_discounted_return(rewards, gamma):
    r"""Discounted sum of future rewards, for every step of the episode.

    .. math::
        G_t = \sum_{k=0}^{\infty}\gamma^{k}\, r_{t+k+1}

    Computed by a single backward pass using
    :math:`G_t = r_t + \gamma G_{t+1}` -- linear rather than quadratic in
    the episode length, and free of the accumulated rounding you get from
    summing :math:`\gamma^k r` forwards.  The effective horizon
    :math:`1/(1-\gamma)` is reported, because that number, not
    :math:`\gamma` itself, is what says how far the agent can see:
    ``gamma=0.95`` is a 20-step horizon, ``gamma=0.99`` is 100.

    Parameters
    ----------
    rewards : array-like, shape (T,)
        Rewards in time order.
    gamma : float
        Discount factor in ``[0, 1]``.

    Returns
    -------
    RichResult
        Payload keys ``returns`` (per step), ``G0``,
        ``effective_horizon``, ``estimate`` (G0), ``n``, ``method``.

    References
    ----------
    Géron Ch 19, Discounted Return section.

    Examples
    --------
    Géron's worked example: rewards ``10, 0, -50`` at ``gamma = 0.8``
    give ``10 + 0 - 50*0.64 = -22``.

    >>> r = geron_discounted_return([10.0, 0.0, -50.0], gamma=0.8)
    >>> [round(v, 10) for v in r["returns"]]
    [-22.0, -40.0, -50.0]
    >>> r["G0"]
    -22.0

    ``gamma = 0`` is a myopic agent: the return is just the next reward.

    >>> geron_discounted_return([10.0, 0.0, -50.0], gamma=0.0)["returns"]
    [10.0, 0.0, -50.0]
    """
    G, gamma = returns_curve(rewards, gamma)
    horizon = float("inf") if gamma == 1.0 else 1.0 / (1.0 - gamma)
    return RichResult(
        title="Discounted return",
        summary_lines=[("gamma", gamma), ("G_0", float(G[0])), ("Effective horizon", horizon)],
        payload={
            "returns": G.tolist(),
            "G0": float(G[0]),
            "gamma": gamma,
            "effective_horizon": horizon,
            "estimate": float(G[0]),
            "n": int(G.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grret: G_t = r_t + gamma G_{t+1} by backward sweep; horizon ~ 1/(1-gamma)"


# compact alias per ledger/NAMING.md
returnscurve = returns_curve
