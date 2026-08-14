# SPDX-License-Identifier: AGPL-3.0-or-later
"""Soft policy iteration (the exact tabular core of soft actor-critic)."""

from . import _array_core as np

from ._richresult import RichResult
from .mdpval import _mdp_args

__all__ = ["sacc", "soft_policy_iteration"]

_METHOD = "Soft policy iteration (maximum-entropy RL, exact tabular SAC core)"


def soft_policy_iteration(P, R, gamma, temp=1.0, tol=1e-12, max_eval=100000,
                          max_improve=1000):
    r"""
    Soft policy iteration on an explicit finite MDP.

    The exact tabular algorithm underlying soft actor-critic (Haarnoja
    et al. 2018, Section 4.1), which the authors state "can perform ...
    in its exact form only in the tabular case".  Soft policy
    evaluation repeatedly applies the soft Bellman backup operator of
    their eq. (2)-(3),

        Q(s, a) <- r(s, a) + gamma E_{s2}[ V(s2) ],
        V(s)    = E_{a ~ pi}[ Q(s, a) - temp log pi(a|s) ],

    and soft policy improvement projects onto the softmax policy of
    their eq. (4),

        pi_new(a|s) = exp(Q(s, a)/temp) / Z(s),

    which for the unrestricted tabular policy class is the exact
    minimizer, giving V(s) = temp log sum_a exp(Q(s, a)/temp)
    (log-sum-exp).  Lemmas 1-2 and Theorem 1 of the source give
    monotone improvement and convergence to the maximum-entropy
    optimum.  The temperature ``temp`` is the entropy weight alpha of
    their eq. (1) (the paper fixes alpha = 1 by scaling rewards; here
    it is explicit).  As temp -> 0 the fixed point approaches the
    standard optimal V* of value iteration.

    Parameters
    ----------
    P : sequence of A matrices, each (S, S)
        Transition probabilities ``P[a][s, s2] = P(s2 | s, a)``.
    R : array-like (S, A), or sequence of A matrices (S, S)
        Expected one-step rewards r(s, a).
    gamma : float
        Discount factor in [0, 1).
    temp : float
        Entropy temperature alpha > 0.
    tol : float
        Convergence threshold for both loops.
    max_eval : int
        Cap on evaluation sweeps per improvement round.
    max_improve : int
        Cap on improvement rounds.

    Returns
    -------
    result : dict
        Keys: ``estimate`` (soft V*, length S), ``policy`` ((S, A)
        maximum-entropy stochastic policy), ``q`` ((S, A) soft action
        values), ``entropy`` (policy entropy per state),
        ``n_improve``, ``n_eval``, ``converged``, ``method``.

    References
    ----------
    Haarnoja, T., Zhou, A., Abbeel, P. and Levine, S. (2018). Soft
    actor-critic: off-policy maximum entropy deep reinforcement
    learning with a stochastic actor. ICML 2018 (arXiv:1801.01290).
    Section 4.1, eqs. (2)-(4), Lemmas 1-2, Theorem 1.  Local source:
    fetched-wave3/haarnoja-etal-2018-sac-arxiv1801.01290.pdf.
    """
    Pm, R, S, A = _mdp_args(P, R)
    gamma = float(gamma)
    temp = float(temp)
    tol = float(tol)
    if temp <= 0.0:
        raise ValueError("temp must be positive")
    Q = np.zeros((S, A))
    pi = np.full((S, A), 1.0 / A)
    logpi = np.full((S, A), -float(np.log(float(A))))
    n_eval = 0
    converged = False
    rounds = 0
    for rounds in range(1, int(max_improve) + 1):
        # soft policy evaluation (eq. 2-3) under the current pi
        for _ in range(int(max_eval)):
            n_eval += 1
            V = np.zeros(S)
            for s in range(S):
                acc = 0.0
                for a in range(A):
                    acc += float(pi[s, a]) * (float(Q[s, a])
                                              - temp * float(logpi[s, a]))
                V[s] = acc
            delta = 0.0
            for s in range(S):
                for a in range(A):
                    q = float(R[s, a]) + gamma * float(np.sum(Pm[a][s] * V))
                    d = abs(q - float(Q[s, a]))
                    if d > delta:
                        delta = d
                    Q[s, a] = q
            if delta < tol:
                break
        # soft policy improvement (eq. 4): pi <- softmax(Q / temp)
        moved = 0.0
        for s in range(S):
            m = float(Q[s, 0])
            for a in range(1, A):
                if float(Q[s, a]) > m:
                    m = float(Q[s, a])
            z = 0.0
            ex = [0.0] * A
            for a in range(A):
                ex[a] = float(np.exp((float(Q[s, a]) - m) / temp))
                z += ex[a]
            for a in range(A):
                p = ex[a] / z
                d = abs(p - float(pi[s, a]))
                if d > moved:
                    moved = d
                pi[s, a] = p
                logpi[s, a] = float(np.log(p)) if p > 0.0 else -1e300
        if moved < tol:
            converged = True
            break
    V = np.zeros(S)
    H = np.zeros(S)
    for s in range(S):
        acc = 0.0
        h = 0.0
        for a in range(A):
            acc += float(pi[s, a]) * (float(Q[s, a]) - temp * float(logpi[s, a]))
            if float(pi[s, a]) > 0.0:
                h -= float(pi[s, a]) * float(logpi[s, a])
        V[s] = acc
        H[s] = h
    return RichResult(payload={
        "estimate": V,
        "policy": pi,
        "q": Q,
        "entropy": H,
        "n_improve": rounds,
        "n_eval": n_eval,
        "converged": converged,
        "method": _METHOD,
    })


sacc = soft_policy_iteration


def cheatsheet():
    return "sacc(P, R, gamma, temp) -> exact soft policy iteration (Haarnoja et al 2018, Sec 4.1)."

# public names resolved by fn/_lazy_map.json
sac = soft_policy_iteration
