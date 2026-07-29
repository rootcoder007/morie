# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Epsilon-greedy action selection."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_epsilon_greedy"]

_METHOD = "Epsilon-greedy action selection"


def _lcg(seed):
    """Reference LCG: s = (1664525 s + 1013904223) mod 2**32, u = (s+0.5)/2**32."""
    s = int(seed) % 2**32
    while True:
        s = (1664525 * s + 1013904223) % 2**32
        yield (s + 0.5) / 2**32


def geron_epsilon_greedy(Q_s, eps, seed=0):
    r"""Exploit with probability :math:`1-\varepsilon`, explore otherwise.

    .. math::
        a = \arg\max_a Q(s, a) \text{ with prob } 1-\varepsilon,
        \quad \text{else } a \sim \mathrm{Uniform}

    The exploration branch is uniform over *all* actions, the greedy one
    included, so the greedy action's total probability is
    :math:`1 - \varepsilon + \varepsilon/|A|`, not :math:`1 -
    \varepsilon`.  That is the standard definition and the source of the
    usual off-by-one when the policy's probabilities are needed for
    importance weighting -- so the exact distribution is returned in
    ``action_probabilities``.

    Draws come from the deterministic LCG above.

    Parameters
    ----------
    Q_s : array-like, shape (A,)
        Action values in the current state.
    eps : float
        Exploration rate in ``[0, 1]``.
    seed : int, optional

    Returns
    -------
    RichResult
        Payload keys ``action``, ``greedy_action``, ``explored``,
        ``action_probabilities``, ``greedy_probability``, ``eps``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 19, Epsilon-Greedy Exploration section.

    Examples
    --------
    At ``eps = 0`` the policy is purely greedy and the distribution is
    a point mass:

    >>> r = geron_epsilon_greedy([1.0, 5.0, 2.0], eps=0.0)
    >>> r["action"], r["explored"]
    (1, False)
    >>> r["action_probabilities"]
    [0.0, 1.0, 0.0]

    At ``eps = 0.3`` over three actions the greedy one keeps
    ``0.7 + 0.3/3 = 0.8``, not 0.7:

    >>> r2 = geron_epsilon_greedy([1.0, 5.0, 2.0], eps=0.3)
    >>> [round(p, 10) for p in r2["action_probabilities"]]
    [0.1, 0.8, 0.1]
    >>> round(sum(r2["action_probabilities"]), 10)
    1.0

    At ``eps = 1`` every action is equally likely -- pure exploration:

    >>> geron_epsilon_greedy([1.0, 5.0], eps=1.0)["greedy_probability"]
    0.5
    """
    Q = np.asarray(Q_s, dtype=float).ravel()
    if Q.size == 0:
        raise ValueError("Q_s is empty; there is no action to choose.")
    if not np.all(np.isfinite(Q)):
        raise ValueError("Q_s must be finite.")
    eps = float(eps)
    if not (0.0 <= eps <= 1.0):
        raise ValueError(f"eps must lie in [0, 1], got {eps}.")
    A = Q.size

    greedy = int(Q.argmax())
    probs = np.full(A, eps / A)
    probs[greedy] += 1.0 - eps

    u = _lcg(seed)
    explored = next(u) < eps
    action = min(int(next(u) * A), A - 1) if explored else greedy

    return RichResult(
        title="Epsilon-greedy",
        summary_lines=[("Action", action), ("Greedy", greedy), ("eps", eps)],
        payload={
            "action": int(action),
            "greedy_action": greedy,
            "explored": bool(explored),
            "action_probabilities": probs.tolist(),
            "greedy_probability": float(probs[greedy]),
            "eps": eps,
            "seed": int(seed),
            "estimate": int(action),
            "n": int(A),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grepl: greedy w.p. 1-eps, else uniform over ALL actions -> P(greedy) = 1-eps+eps/|A|"
