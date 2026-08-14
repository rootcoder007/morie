# morie.fn -- function file (rootcoder007/morie)
"""Backward-targeted Q-learning for a multi-stage treatment regime."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_qlearning"]


def tmle_qlearning(state, action, reward, time):
    """Value of the optimal multi-stage regime, targeted stage by stage.

    Plain Q-learning is a sequence of regressions and inherits a bias at
    every stage, because the maximisation is applied to a fitted value
    and a fitted value is biased in the direction the maximiser looks.
    Targeting each stage before taking it forward removes the
    first-order part of that bias: at stage ``t`` the clever covariate

        ``H = I(a = a*(s)) / b(a*(s) | s)``

    is used to fluctuate ``Q_t`` along the regime actually being
    evaluated, and only the targeted ``V_t(s) = Q*_t(s, a*(s))`` is
    passed back to stage ``t - 1`` as part of the pseudo-outcome.  The
    behaviour policy ``b`` is fitted by logistic regression on the state
    at that stage, so binary actions are assumed.

    Rows must be grouped by stage: within each stage the subjects appear
    in the same order, and every stage has the same number of rows.

    Parameters
    ----------
    state : array-like, shape (n,)
        State at each subject-stage.
    action : array-like, shape (n,)
        Binary action at each subject-stage.
    reward : array-like, shape (n,)
        Reward collected at each subject-stage.
    time : array-like, shape (n,)
        Stage index of each row.

    Returns
    -------
    RichResult
        ``estimate`` (mean stage-1 value), ``se``, ``n_stages``,
        ``n_subj``, ``n``.

    References
    ----------
    Murphy, S. A. (2003).  Optimal dynamic treatment regimes.  Journal
    of the Royal Statistical Society Series B 65(2):331-355.
    doi:10.1111/1467-9868.00389.  The stagewise targeting step is
    Petersen, M. et al. (2014), Targeted maximum likelihood estimation
    for dynamic and static longitudinal marginal structural working
    models, Journal of Causal Inference 2(2):147-185.
    doi:10.1515/jci-2013-0007.
    """
    sv = C.vec(state)
    av = C.vec(action)
    rv = C.vec(reward)
    tv = C.vec(time)
    n = len(sv)
    if n == 0 or len(av) != n or len(rv) != n or len(tv) != n:
        raise ValueError("tmle_qlearning: state, action, reward and time must share one length")
    stages = sorted(set(tv))
    T = len(stages)
    rows = [[i for i in range(n) if tv[i] == s] for s in stages]
    m = len(rows[0])
    for r in rows:
        if len(r) != m:
            raise ValueError("tmle_qlearning: every stage must have the same number of rows")
    V = [0.0] * m
    eps_all = []
    ic = [0.0] * m
    for t in range(T - 1, -1, -1):
        idx = rows[t]
        des = [[1.0, sv[i], av[i], sv[i] * av[i]] for i in idx]
        pseudo = [rv[idx[k]] + (V[k] if t < T - 1 else 0.0) for k in range(m)]
        qb, _, _, _ = S.ols(des, pseudo)
        bb = S.glmbin([[1.0, sv[i]] for i in idx], [av[i] for i in idx])
        b1 = [S.clip(S.expit(C.dot([1.0, sv[i]], bb)), 0.025, 0.975) for i in idx]

        def q(k, a):
            i = idx[k]
            return C.dot([1.0, sv[i], a, sv[i] * a], qb)

        astar = [1.0 if q(k, 1.0) >= q(k, 0.0) else 0.0 for k in range(m)]
        ba = [b1[k] if astar[k] > 0.5 else 1.0 - b1[k] for k in range(m)]
        H = [(1.0 if abs(av[idx[k]] - astar[k]) < 0.5 else 0.0) / ba[k] for k in range(m)]
        Qobs = [q(k, av[idx[k]]) for k in range(m)]
        den = sum(h * h for h in H)
        eps = sum(H[k] * (pseudo[k] - Qobs[k]) for k in range(m)) / den if den != 0.0 else 0.0
        eps_all.append(eps)
        Vnew = [q(k, astar[k]) + eps / ba[k] for k in range(m)]
        ic = [H[k] * (pseudo[k] - Qobs[k] - eps * H[k]) + ic[k] for k in range(m)]
        V = Vnew
    psi = sum(V) / m
    ic = [ic[k] + V[k] - psi for k in range(m)]
    mu = sum(ic) / m
    se = math.sqrt(sum((v - mu) ** 2 for v in ic) / (m - 1) / m) if m > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "n_stages": float(T), "n_subj": float(m), "n": n,
        "method": "Backward-targeted Q-learning for a multi-stage regime"})


def cheatsheet():
    return "tmlqlc: stagewise-targeted Q-learning for a dynamic regime."

# public names resolved by fn/_lazy_map.json
tmleqlearning = tmle_qlearning
