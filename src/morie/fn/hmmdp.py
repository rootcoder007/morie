# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Markov decision process (S, A, P, R, gamma)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_mdp"]

_METHOD = "MDP validation and value iteration"


def geron_mdp(states, actions, P, R, gamma=0.95, max_iter=1000, tol=1e-10):
    """
    Markov decision process (S, A, P, R, gamma).

    Formula: (s, a) -> s' ~ P; r ~ R; discount gamma

    Builds the tuple, checks it is actually an MDP, and solves it by
    value iteration:

    ``Q(s,a) = sum_s' P(s,a,s') [ R(s,a,s') + gamma * max_a' Q(s',a') ]``

    Two things are enforced rather than assumed.  Each ``P[s, a, :]``
    must sum to 1 -- a row summing to 0.9 is a leak that silently
    discounts every future reward by an extra 10%, and it looks exactly
    like a lower ``gamma``.  And ``gamma`` must lie in ``[0, 1)``: at
    ``gamma = 1`` the Bellman operator stops being a contraction, so
    value iteration need not converge at all on a continuing task.

    The contraction rate is ``gamma``, so the error falls by that factor
    per sweep and the effective horizon is about ``1/(1-gamma)`` steps;
    both are returned, because they explain why ``gamma = 0.99`` costs
    so much more iteration than ``0.9``.

    Parameters
    ----------
    states : sequence
        State labels; the length defines ``n_states``.
    actions : sequence
        Action labels.
    P : array-like, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R : array-like, shape (n_states, n_actions, n_states) or (n_states, n_actions)
        Rewards.
    gamma : float
        Discount in [0, 1).
    max_iter : int
        Sweep cap.
    tol : float
        Convergence threshold on the max change in V.

    Returns
    -------
    result : RichResult
        Keys: V, Q, policy, policy_labels, n_iter, effective_horizon,
        estimate, n, method.

    Examples
    --------
    One state, two actions, rewards 1 and 0.  The optimal value of a
    self-loop with reward 1 and gamma 0.9 is the geometric sum
    ``1/(1-0.9) = 10``:

    >>> P = [[[1.0], [1.0]]]
    >>> R = [[1.0, 0.0]]
    >>> r = geron_mdp(["s"], ["good", "bad"], P, R, gamma=0.9)
    >>> round(float(r["V"][0]), 6)
    10.0
    >>> r["policy_labels"]
    ['good']

    A shorter horizon values the same loop less: at gamma 0.5 it is 2.

    >>> round(float(geron_mdp(["s"], ["good", "bad"], P, R, gamma=0.5)["V"][0]), 9)
    2.0

    Effective horizon ``1/(1-gamma)``:

    >>> round(r["effective_horizon"], 6)
    10.0

    A transition row that does not sum to one is refused:

    >>> geron_mdp(["s"], ["a"], [[[0.9]]], [[1.0]], gamma=0.9)
    Traceback (most recent call last):
        ...
    ValueError: geron_mdp: P[0, 0] sums to 0.9, not 1; a leaking transition row silently discounts every future reward

    gamma = 1 is refused because the Bellman operator stops contracting:

    >>> geron_mdp(["s"], ["a"], [[[1.0]]], [[1.0]], gamma=1.0)
    Traceback (most recent call last):
        ...
    ValueError: geron_mdp: gamma must lie in [0, 1); at gamma = 1.0 the Bellman operator is not a contraction and value iteration need not converge

    References
    ----------
    Géron Ch 19
    """
    S = list(states)
    A = list(actions)
    n_s, n_a = len(S), len(A)
    if n_s < 1 or n_a < 1:
        raise ValueError(f"geron_mdp: need at least one state and one action, got {n_s} and {n_a}")
    Pa = np.asarray(P, dtype=float)
    if Pa.shape != (n_s, n_a, n_s):
        raise ValueError(
            f"geron_mdp: P must have shape ({n_s}, {n_a}, {n_s}), got {Pa.shape}"
        )
    if not np.all(np.isfinite(Pa)):
        raise ValueError("geron_mdp: P contains non-finite values")
    if np.any(Pa < 0):
        raise ValueError("geron_mdp: P contains negative probabilities")
    sums = Pa.sum(axis=2)
    bad = np.argwhere(np.abs(sums - 1.0) > 1e-9)
    if bad.size:
        s, a = int(bad[0, 0]), int(bad[0, 1])
        raise ValueError(
            f"geron_mdp: P[{s}, {a}] sums to {sums[s, a]:g}, not 1; a leaking transition row silently "
            f"discounts every future reward"
        )

    Ra = np.asarray(R, dtype=float)
    if Ra.shape == (n_s, n_a):
        Ra = np.repeat(Ra[:, :, None], n_s, axis=2)
    if Ra.shape != (n_s, n_a, n_s):
        raise ValueError(
            f"geron_mdp: R must have shape ({n_s}, {n_a}, {n_s}) or ({n_s}, {n_a}), got {np.asarray(R).shape}"
        )
    if not np.all(np.isfinite(Ra)):
        raise ValueError("geron_mdp: R contains non-finite values")

    g = float(gamma)
    if not (0.0 <= g < 1.0):
        raise ValueError(
            f"geron_mdp: gamma must lie in [0, 1); at gamma = {g} the Bellman operator is not a "
            f"contraction and value iteration need not converge"
        )
    iters = int(max_iter)
    if iters < 1:
        raise ValueError(f"geron_mdp: max_iter must be at least 1, got {max_iter!r}")

    expected_r = np.sum(Pa * Ra, axis=2)  # (n_s, n_a)
    V = np.zeros(n_s)
    n_iter = 0
    for n_iter in range(1, iters + 1):
        Q = expected_r + g * (Pa @ V)
        V_new = Q.max(axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        if delta <= float(tol):
            break
    Q = expected_r + g * (Pa @ V)
    policy = np.argmax(Q, axis=1)
    horizon = 1.0 / (1.0 - g) if g < 1 else float("inf")

    return RichResult(
        title="Markov decision process",
        summary_lines=[
            ("States x actions", f"{n_s} x {n_a}"),
            ("gamma", g),
            ("Sweeps", n_iter),
            ("Effective horizon", horizon),
        ],
        warnings=(
            [f"value iteration hit the {iters}-sweep cap without converging; raise max_iter or lower gamma."]
            if n_iter >= iters and delta > float(tol)
            else []
        ),
        interpretation=(
            "Error falls by a factor of gamma per sweep, so a gamma near 1 buys a long horizon at the "
            "cost of slow convergence."
        ),
        payload={
            "V": V,
            "Q": Q,
            "policy": policy,
            "policy_labels": [A[int(i)] for i in policy],
            "expected_reward": expected_r,
            "n_iter": int(n_iter),
            "gamma": g,
            "effective_horizon": horizon,
            "estimate": float(np.max(V)),
            "n": int(n_s),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmdp: MDP tuple validation plus value iteration; enforces row-stochastic P and gamma < 1"


# compact alias per ledger/NAMING.md
geronmdp = geron_mdp
