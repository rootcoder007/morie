# morie.fn -- function file (rootcoder007/morie)
"""TMLE for the long-run value of a policy in a finite Markov decision process."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_markov"]


def tmle_markov(state, action, reward, policy):
    """Targeted long-run average reward of a deterministic policy.

    The value reported is the stationary average reward,
    ``V^pi = sum_s d^pi(s) r*(s, pi(s))``, where ``d^pi`` is the
    stationary distribution of the transition matrix induced by ``pi``.
    Using the stationary average rather than a discounted sum avoids
    inventing a discount factor the caller never supplied, and it is the
    quantity that is identified from a single long trajectory.

    The reward model is the cell mean ``r(s, a)``; the behaviour policy
    ``b(a | s)`` is the empirical action frequency in each state.  The
    clever covariate is the importance ratio restricted to the actions
    the policy would take,

        ``H = I(a = pi(s)) / b(pi(s) | s)``,

    and ``r*(s, pi(s)) = r(s, pi(s)) + eps / b(pi(s) | s)`` with
    ``eps = sum H (R - r(s, a)) / sum H^2``.  Transitions are counted
    from consecutive entries of the input, which therefore must be a
    single trajectory in time order.

    Parameters
    ----------
    state : array-like, shape (n,)
        State label at each step, in time order.
    action : array-like, shape (n,)
        Action taken at each step.
    reward : array-like, shape (n,)
        Reward received at each step.
    policy : array-like, shape (n_states,)
        Action the evaluated policy takes in each distinct state, in the
        sorted order of the state labels.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``n_states``, ``n``.

    References
    ----------
    Murphy, S. A. (2003).  Optimal dynamic treatment regimes.  Journal
    of the Royal Statistical Society Series B 65(2):331-355.
    doi:10.1111/1467-9868.00389.  The targeting step is van der Laan,
    M. J. & Rubin, D. (2006), IJB 2(1):11.
    """
    sv = C.vec(state)
    av = C.vec(action)
    rv = C.vec(reward)
    pv = C.vec(policy)
    n = len(sv)
    if n < 2 or len(av) != n or len(rv) != n:
        raise ValueError("tmle_markov: state, action and reward must share one length >= 2")
    states = sorted(set(sv))
    ns = len(states)
    if len(pv) != ns:
        raise ValueError("tmle_markov: policy must give one action per distinct state")
    si = {}
    for k in range(ns):
        si[states[k]] = k
    pol = [pv[k] for k in range(ns)]

    b = [0.0] * n
    for i in range(n):
        s = sv[i]
        tot = sum(1.0 for j in range(n) if sv[j] == s)
        hit = sum(1.0 for j in range(n) if sv[j] == s and abs(av[j] - pol[si[s]]) < 1e-9)
        b[i] = S.clip(hit / tot, 0.01, 1.0)

    rbar = [[0.0, 0.0] for _ in range(ns)]
    for i in range(n):
        k = si[sv[i]]
        if abs(av[i] - pol[k]) < 1e-9:
            rbar[k][0] += rv[i]
            rbar[k][1] += 1.0
    rhat = [rbar[k][0] / rbar[k][1] if rbar[k][1] > 0 else 0.0 for k in range(ns)]

    H = [(1.0 if abs(av[i] - pol[si[sv[i]]]) < 1e-9 else 0.0) / b[i] for i in range(n)]
    Qobs = [rhat[si[sv[i]]] for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (rv[i] - Qobs[i]) for i in range(n)) / den if den != 0.0 else 0.0
    bst = [0.0] * ns
    for k in range(ns):
        rows = [i for i in range(n) if si[sv[i]] == k]
        bst[k] = b[rows[0]] if rows else 1.0
    rstar = [rhat[k] + eps / bst[k] for k in range(ns)]

    P = [[0.0] * ns for _ in range(ns)]
    cnt = [0.0] * ns
    for i in range(n - 1):
        k = si[sv[i]]
        if abs(av[i] - pol[k]) < 1e-9:
            P[k][si[sv[i + 1]]] += 1.0
            cnt[k] += 1.0
    for k in range(ns):
        if cnt[k] > 0:
            for j in range(ns):
                P[k][j] /= cnt[k]
        else:
            P[k][k] = 1.0

    A = [[(1.0 if j == k else 0.0) - P[j][k] for j in range(ns)] for k in range(ns)]
    A[ns - 1] = [1.0] * ns
    rhs = [0.0] * ns
    rhs[ns - 1] = 1.0
    d = C.solvev(A, rhs)
    V = sum(d[k] * rstar[k] for k in range(ns))

    emp = [sum(1.0 for i in range(n) if si[sv[i]] == k) / n for k in range(ns)]
    ic = []
    for i in range(n):
        k = si[sv[i]]
        w = d[k] / emp[k] if emp[k] > 0 else 0.0
        ic.append(w * H[i] * (rv[i] - Qobs[i] - eps * H[i]) + rstar[k] - V)
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": V, "se": se, "eps": eps, "n_states": float(ns), "n": n,
        "method": "TMLE for the long-run average reward of a policy in an MDP"})


def cheatsheet():
    return "tmlmrk: TMLE for the long-run policy value in a Markov decision process."
