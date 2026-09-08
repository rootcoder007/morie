# SPDX-License-Identifier: AGPL-3.0-or-later
"""SARSA (on-policy TD control)."""

from . import _array_core as np

from ._richresult import RichResult
from .mdpval import _mdp_args
from .qlearn import _greedy, _sample_row

__all__ = ["sarsa", "sarsa_control"]

_METHOD = "Tabular SARSA, epsilon-greedy on-policy TD control"


def sarsa_control(P, R, gamma, alpha=0.1, epsilon=0.1, n_episodes=100,
                  start=0, terminal=(), max_steps=1000, seed=0, Q0=None):
    r"""
    Tabular SARSA on an explicit finite MDP.

    Runs the boxed algorithm "Sarsa (on-policy TD control)" of Sutton
    and Barto (2018), Section 6.4, p. 130: after taking A in S and
    observing R and S2, the NEXT action A2 is chosen (epsilon-greedily)
    from S2 first, and the update uses the quintuple (S, A, R, S2, A2):

        Q(S, A) <- Q(S, A) + alpha [ R + gamma Q(S2, A2) - Q(S, A) ]

    then S <- S2, A <- A2.  The name and the algorithm come from the
    "modified connectionist Q-learning" of Rummery and Niranjan (1994).
    Q(terminal, .) = 0, so gamma Q(S2, A2) contributes 0 when S2 is
    terminal.

    Determinism conventions (mirrored bit-exactly in the R arm): ties in
    the greedy argmax break to the lowest action index; the epsilon-
    greedy draw consumes one uniform for the epsilon test plus one for
    the random action when exploring (a = floor(u * A)); the next state
    consumes one uniform by inverse CDF on the P row.  The initial
    action of an episode is drawn before the step loop, exactly as in
    the source box.  With ``epsilon = 0`` and deterministic rows the
    trajectory is fully deterministic and hand-computable.

    Parameters
    ----------
    P : sequence of A matrices, each (S, S)
        Transition probabilities ``P[a][s, s2] = P(s2 | s, a)``.
    R : array-like (S, A), or sequence of A matrices (S, S)
        Expected one-step rewards r(s, a).
    gamma : float
        Discount factor in [0, 1].
    alpha : float
        Step size in (0, 1].
    epsilon : float
        Exploration probability in [0, 1].
    n_episodes : int
        Number of episodes.
    start : int
        0-based start state of every episode.
    terminal : sequence of int
        0-based absorbing terminal states.
    max_steps : int
        Step cap per episode.
    seed : int
        SplitMix64 seed.
    Q0 : array-like (S, A), optional
        Initial action values (default zeros; terminal rows forced 0).

    Returns
    -------
    result : dict
        Keys: ``estimate`` ((S, A) learned Q table), ``policy`` (greedy
        0-based action per state), ``v`` (greedy state values),
        ``n_steps``, ``n_episodes``, ``method``.

    References
    ----------
    Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An
    Introduction, 2nd ed., MIT Press, Section 6.4, boxed algorithm
    p. 130.  Local source:
    fetched-wave3/sutton-barto-2018-reinforcement-learning-2nd-ed.pdf.
    Rummery, G. A. and Niranjan, M. (1994). On-line Q-learning using
    connectionist systems. Technical Report CUED/F-INFENG/TR 166,
    Cambridge University Engineering Department.  Local source:
    fetched-wave3/rummery-niranjan-1994-sarsa-tr166.pdf.
    """
    Pm, R, S, A = _mdp_args(P, R)
    gamma = float(gamma)
    alpha = float(alpha)
    epsilon = float(epsilon)
    start = int(start)
    term = set(int(t) for t in terminal)
    if start < 0 or start >= S:
        raise ValueError("start out of range")
    rng = np.random.default_rng(seed)
    Q = np.zeros((S, A))
    if Q0 is not None:
        Q0 = np.asarray(Q0, dtype=float)
        for s in range(S):
            for a in range(A):
                Q[s, a] = 0.0 if s in term else float(Q0[s, a])

    def _eps_greedy(s):
        u1 = float(rng.uniform())
        if u1 < epsilon:
            a = int(float(rng.uniform()) * A)
            return A - 1 if a >= A else a
        return _greedy(Q[s], A)

    n_steps = 0
    for _ in range(int(n_episodes)):
        s = start
        if s in term:
            continue
        a = _eps_greedy(s)
        for _ in range(int(max_steps)):
            u3 = float(rng.uniform())
            s2 = _sample_row(Pm[a][s], S, u3)
            r = float(R[s, a])
            if s2 in term:
                target = r
                a2 = 0
            else:
                a2 = _eps_greedy(s2)
                target = r + gamma * float(Q[s2, a2])
            Q[s, a] = float(Q[s, a]) + alpha * (target - float(Q[s, a]))
            n_steps += 1
            if s2 in term:
                break
            s = s2
            a = a2
    pol = np.zeros(S)
    V = np.zeros(S)
    for s in range(S):
        b = _greedy(Q[s], A)
        pol[s] = float(b)
        V[s] = float(Q[s, b])
    return RichResult(payload={
        "estimate": Q,
        "policy": pol,
        "v": V,
        "n_steps": n_steps,
        "n_episodes": int(n_episodes),
        "method": _METHOD,
    })


sarsa = sarsa_control


def cheatsheet():
    return "sarsa(P, R, gamma, alpha, epsilon, n_episodes) -> on-policy SARSA per Sutton-Barto 2018 Sec 6.4."
