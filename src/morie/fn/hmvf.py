# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""State value function V^pi(s)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_value_function"]


def geron_value_function(s, pi, gamma, P=None, R=None):
    """
    State value function V^pi(s).

    Formula: V^pi(s) = E_pi[sum_t gamma^t r_t | s_0 = s]

    Solved exactly, not sampled: for a finite MDP the Bellman expectation
    equation ``V = r_pi + gamma * P_pi V`` is linear, so
    ``V = (I - gamma*P_pi)^-1 r_pi``. The residual of that equation is
    returned as proof the solution is a fixed point. For ``gamma = 1``
    the system is singular whenever the chain is recurrent, and that is
    reported as an error rather than a silent pseudo-solution.

    Parameters
    ----------
    s : int or None
        State whose value is reported as `estimate`; None means state 0.
    pi : array-like
        Policy: (n_states, n_actions) action probabilities, or a length
        n_states vector of deterministic action indices.
    gamma : float
        Discount factor in [0, 1).
    P : array-like
        Transition tensor ``P[s, a, s']``, rows summing to 1. Required.
    R : array-like
        Rewards ``R[s, a, s']`` or ``R[s, a]``. Required.

    Returns
    -------
    result : RichResult
        Keys: V, value, r_pi, P_pi, residual, estimate, n, method.

    Examples
    --------
    A two-state chain: state 0 always moves to state 1 with reward 1,
    state 1 absorbs with reward 0. With gamma = 0.9 the values are
    V(1) = 0 and V(0) = 1.

    >>> P = [[[0.0, 1.0]], [[0.0, 1.0]]]
    >>> R = [[[0.0, 1.0]], [[0.0, 0.0]]]
    >>> r = geron_value_function(0, [[1.0], [1.0]], 0.9, P=P, R=R)
    >>> [round(float(v), 12) for v in r["V"]]
    [1.0, 0.0]
    >>> bool(r["residual"] < 1e-12)
    True

    Self-loop with reward 1 gives the geometric sum 1/(1 - gamma) = 10:

    >>> r2 = geron_value_function(0, [0], 0.9, P=[[[1.0]]], R=[[[1.0]]])
    >>> round(float(r2["estimate"]), 9)
    10.0

    References
    ----------
    Géron Ch 19
    """
    if P is None or R is None:
        raise ValueError("geron_value_function: both P (transitions) and R (rewards) are required to evaluate a policy")
    Pt = np.asarray(P, dtype=float)
    if Pt.ndim != 3 or Pt.shape[0] != Pt.shape[2]:
        raise ValueError(f"geron_value_function: P must have shape (n_states, n_actions, n_states), got {Pt.shape}")
    n_s, n_a = Pt.shape[0], Pt.shape[1]
    if not np.allclose(Pt.sum(axis=2), 1.0):
        raise ValueError("geron_value_function: every P[s, a, :] must be a probability distribution summing to 1")
    Rt = np.asarray(R, dtype=float)
    if Rt.shape == (n_s, n_a):
        Rsa = Rt
    elif Rt.shape == (n_s, n_a, n_s):
        Rsa = np.sum(Pt * Rt, axis=2)
    else:
        raise ValueError(
            f"geron_value_function: R must have shape {(n_s, n_a)} or {(n_s, n_a, n_s)}, got {Rt.shape}"
        )
    g = float(gamma)
    if not (0.0 <= g < 1.0):
        raise ValueError(f"geron_value_function: gamma must lie in [0, 1) for a well-posed linear solve, got {g}")

    Pi = np.asarray(pi, dtype=float)
    if Pi.ndim == 1:
        if Pi.size != n_s:
            raise ValueError(f"geron_value_function: deterministic pi must have {n_s} entries, got {Pi.size}")
        act = Pi.astype(int)
        if act.min() < 0 or act.max() >= n_a:
            raise ValueError(f"geron_value_function: pi selects an action outside 0..{n_a - 1}")
        Pi = np.zeros((n_s, n_a))
        Pi[np.arange(n_s), act] = 1.0
    if Pi.shape != (n_s, n_a):
        raise ValueError(f"geron_value_function: pi must have shape {(n_s, n_a)}, got {Pi.shape}")
    if np.any(Pi < 0) or not np.allclose(Pi.sum(axis=1), 1.0):
        raise ValueError("geron_value_function: every pi[s, :] must be a probability distribution summing to 1")

    P_pi = np.einsum("sa,sat->st", Pi, Pt)
    r_pi = np.sum(Pi * Rsa, axis=1)
    A = np.eye(n_s) - g * P_pi
    if abs(float(np.linalg.det(A))) < 1e-14:
        raise ValueError("geron_value_function: (I - gamma*P_pi) is singular; the policy's value is unbounded")
    V = np.linalg.solve(A, r_pi)
    residual = float(np.max(np.abs(V - (r_pi + g * P_pi @ V))))

    idx = 0 if s is None else int(s)
    if not (0 <= idx < n_s):
        raise ValueError(f"geron_value_function: state {idx} is outside 0..{n_s - 1}")

    return RichResult(
        title="State value function",
        summary_lines=[("States", n_s), ("Actions", n_a), ("gamma", g), (f"V({idx})", float(V[idx]))],
        interpretation=(
            "V^pi answers 'how good is this state under this policy'; because the Bellman equation is "
            "linear in V it can be solved exactly for finite MDPs, no sampling needed."
        ),
        payload={
            "V": V,
            "value": float(V[idx]),
            "r_pi": r_pi,
            "P_pi": P_pi,
            "residual": residual,
            "gamma": g,
            "state": idx,
            "estimate": float(V[idx]),
            "n": int(n_s),
            "method": "Exact policy evaluation V = (I - gamma P_pi)^-1 r_pi",
        },
    )


def cheatsheet():
    return "hmvf: State value function V^pi(s)"
