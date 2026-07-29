# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Action-value function Q^pi(s, a) for a finite MDP."""

import numpy as np

from ._richresult import RichResult
from .grvpi import policy_evaluation

__all__ = ["geron_action_value_function"]

_METHOD = "Action-value function Q^pi"


def geron_action_value_function(state, action, policy, transitions, rewards, gamma):
    r"""Value of taking one action now and following the policy afterwards.

    .. math::
        Q^{\pi}(s, a) = \sum_{s'} P(s'|s,a)
            \bigl[R(s,a,s') + \gamma V^{\pi}(s')\bigr]

    which is the expectation form of
    :math:`\mathbb{E}[\sum_k \gamma^k r_{t+k+1} \mid S_t=s, A_t=a, \pi]`.
    The state values come from :func:`morie.fn.grvpi.policy_evaluation`
    -- Q is one Bellman backup away from V, not a separate computation.
    The gap :math:`Q^{\pi}(s,a) - V^{\pi}(s)` is the advantage, and it is
    returned because it, not Q, is what a policy-improvement step acts on.

    Parameters
    ----------
    state, action : int
    policy : array-like
        ``(S,)`` deterministic or ``(S, A)`` stochastic.
    transitions : array-like, shape (S, A, S)
    rewards : array-like, shape (S, A, S) or (S, A)
    gamma : float

    Returns
    -------
    RichResult
        Payload keys ``q_value``, ``q_values`` (S x A), ``values``,
        ``advantage``, ``greedy_action``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 19, Action-Value Q function.

    Examples
    --------
    Two actions in one absorbing state, paying 1 and 0, ``gamma = 0.5``,
    policy always takes action 0 so ``V = 2``.  Then
    ``Q(0, 1) = 0 + 0.5 * 2 = 1``:

    >>> P = [[[1.0], [1.0]]]
    >>> R = [[1.0, 0.0]]
    >>> r = geron_action_value_function(0, 1, [0], P, R, gamma=0.5)
    >>> round(r["q_value"], 10)
    1.0
    >>> round(r["q_values"][0][0], 10)
    2.0

    The advantage of the worse action is negative:

    >>> round(r["advantage"], 10)
    -1.0
    """
    V, pi, P, R, r_sa, gamma = policy_evaluation(policy, transitions, rewards, gamma)
    S, A = P.shape[0], P.shape[1]
    s, a = int(state), int(action)
    if not (0 <= s < S):
        raise ValueError(f"state must lie in [0, {S - 1}], got {s}.")
    if not (0 <= a < A):
        raise ValueError(f"action must lie in [0, {A - 1}], got {a}.")

    Q = r_sa + gamma * np.einsum("sap,p->sa", P, V)
    return RichResult(
        title="Action-value function",
        summary_lines=[("State", s), ("Action", a), ("Q^pi(s,a)", float(Q[s, a]))],
        payload={
            "q_value": float(Q[s, a]),
            "q_values": Q.tolist(),
            "values": V.tolist(),
            "advantage": float(Q[s, a] - V[s]),
            "greedy_action": int(np.argmax(Q[s])),
            "estimate": float(Q[s, a]),
            "n": int(S),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grqpi: Q^pi(s,a) = sum_s' P[R + gamma V^pi(s')]; V from grvpi, advantage = Q - V"
