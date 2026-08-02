# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""State value function V^pi(s) for a finite MDP."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_state_value_function", "policy_evaluation"]

_METHOD = "State value function V^pi under a fixed policy"


def _check_mdp(policy, transitions, rewards, gamma):
    P = np.asarray(transitions, dtype=float)
    R = np.asarray(rewards, dtype=float)
    pi = np.asarray(policy, dtype=float)
    if P.ndim != 3:
        raise ValueError(
            f"transitions must be (n_states, n_actions, n_states), got shape {P.shape}."
        )
    S, A, S2 = P.shape
    if S2 != S:
        raise ValueError(f"transitions last axis is {S2} but there are {S} states.")
    if R.shape != P.shape:
        if R.shape == (S, A):
            R = np.repeat(R[:, :, None], S, axis=2)
        else:
            raise ValueError(
                f"rewards must be (S, A, S') = {P.shape} or (S, A) = {(S, A)}, got {R.shape}."
            )
    if pi.shape == (S,):
        det = pi.astype(int)
        if np.any(det < 0) or np.any(det >= A):
            raise ValueError(f"deterministic policy actions must lie in [0, {A - 1}].")
        pi = np.zeros((S, A))
        pi[np.arange(S), det] = 1.0
    elif pi.shape != (S, A):
        raise ValueError(f"policy must be (S,) or (S, A) = {(S, A)}, got {pi.shape}.")
    if np.any(pi < 0) or not np.allclose(pi.sum(axis=1), 1.0):
        raise ValueError("policy rows must be non-negative and sum to 1.")
    if np.any(P < 0) or not np.allclose(P.sum(axis=2), 1.0):
        raise ValueError("transition rows P(.|s, a) must be non-negative and sum to 1.")
    if not np.all(np.isfinite(R)):
        raise ValueError("rewards contains non-finite values.")
    gamma = float(gamma)
    if not (0.0 <= gamma < 1.0):
        raise ValueError(
            f"gamma must lie in [0, 1) for the value system to have a unique solution, got {gamma}."
        )
    return pi, P, R, gamma


def policy_evaluation(policy, transitions, rewards, gamma):
    """Exact ``V^pi`` by solving ``(I - gamma P_pi) V = r_pi``."""
    pi, P, R, gamma = _check_mdp(policy, transitions, rewards, gamma)
    S = P.shape[0]
    r_sa = np.einsum("sap,sap->sa", P, R)          # expected immediate reward
    r_pi = np.einsum("sa,sa->s", pi, r_sa)
    P_pi = np.einsum("sa,sap->sp", pi, P)
    V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
    return V, pi, P, R, r_sa, gamma


def geron_state_value_function(state, policy, transitions, rewards, gamma):
    r"""Expected discounted return from a state under a fixed policy.

    .. math::
        V^{\pi}(s) = \mathbb{E}\Bigl[\sum_k \gamma^{k} r_{t+k+1}
                     \,\Big|\, S_t = s, \pi\Bigr]

    Rather than iterating the Bellman expectation backup to convergence,
    this solves it: the backup is affine, so
    :math:`(I - \gamma P_{\pi})V = r_{\pi}` has one exact solution
    whenever :math:`\gamma < 1`.  Iterative evaluation only approximates
    that fixed point, and its error is invisible unless you track it.

    Parameters
    ----------
    state : int
        State whose value is reported.
    policy : array-like
        ``(S,)`` deterministic actions or ``(S, A)`` action probabilities.
    transitions : array-like, shape (S, A, S)
        ``P(s' | s, a)``; rows must sum to 1.
    rewards : array-like, shape (S, A, S) or (S, A)
    gamma : float
        Discount in ``[0, 1)``.

    Returns
    -------
    RichResult
        Payload keys ``value`` (for ``state``), ``values`` (all states),
        ``state``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 19, Value Function.

    Examples
    --------
    A single absorbing state paying 1 per step has value
    ``1/(1-gamma) = 2`` at ``gamma = 0.5``:

    >>> P = [[[1.0]]]
    >>> r = geron_state_value_function(0, [0], P, [[1.0]], gamma=0.5)
    >>> round(r["value"], 10)
    2.0

    Two states, deterministic move to the other, rewards 1 and 0:
    solving the pair gives ``V_0 = 1 + 0.9 V_1`` and ``V_1 = 0.9 V_0``.

    >>> P2 = [[[0.0, 1.0]], [[1.0, 0.0]]]
    >>> R2 = [[1.0], [0.0]]
    >>> v = geron_state_value_function(0, [0, 0], P2, R2, gamma=0.9)
    >>> round(v["values"][0], 6), round(v["values"][1], 6)
    (5.263158, 4.736842)
    """
    V, pi, P, R, r_sa, gamma = policy_evaluation(policy, transitions, rewards, gamma)
    s = int(state)
    if not (0 <= s < V.size):
        raise ValueError(f"state must lie in [0, {V.size - 1}], got {s}.")
    return RichResult(
        title="State value function",
        summary_lines=[("State", s), ("V^pi(s)", float(V[s])), ("gamma", gamma)],
        payload={
            "value": float(V[s]),
            "values": V.tolist(),
            "state": s,
            "estimate": float(V[s]),
            "n": int(V.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grvpi: V^pi solves (I - gamma P_pi) V = r_pi exactly; no iteration, no convergence error"
