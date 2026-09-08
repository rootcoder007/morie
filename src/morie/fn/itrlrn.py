# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Iterative Q-learning (backward induction) for a dynamic treatment regime.

Murphy (2003), "Optimal dynamic treatment regimes", Journal of the
Royal Statistical Society Series B 65(2):331-355,
doi:10.1111/1467-9868.00389, and Petersen, Schwab, Gruber, Blaser,
Schomaker and van der Laan (2014), "Targeted maximum likelihood
estimation for dynamic and static longitudinal marginal structural
working models", Journal of Causal Inference 2(2):147-185,
doi:10.1515/jci-2013-0007.

Q-learning solves the regime backwards.  At the final stage T the
Q-function is fitted by least squares on the stage covariates, the
action and their interaction; at stage t the same regression is run on
the pseudo-outcome

    R_t + gamma * max_{a'} Q_{t+1}(S_{t+1}, a'),

so that the optimal rule at stage t is d_t(s) = 1{Q_t(s, 1) > Q_t(s, 0)}.
The value of the regime is the average of max_a Q_1(S_1, a).  Everything
is a closed-form least-squares solve, so the two language arms agree
exactly; nothing here is simulated.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["iterative_q_learning"]


def iterative_q_learning(state, action, reward, time, gamma=1.0):
    """Backward-induction Q-learning on balanced panel data.

    Parameters
    ----------
    state : array-like
        Stage covariates, one row per (subject, stage) record.
    action : array-like
        Binary action taken in that record.
    reward : array-like
        Reward received in that record.
    time : array-like
        Stage index.  Every stage must contain the same number of
        records, given in the same subject order.
    gamma : float
        Discount factor.
    """
    a = core.vec(action)
    r = core.vec(reward)
    tm = core.vec(time)
    n = len(r)
    if n == 0:
        raise ValueError("iterative_q_learning: reward is empty")
    if len(a) != n or len(tm) != n:
        raise ValueError("iterative_q_learning: action, reward and time have different lengths")
    for v in a:
        if v not in (0.0, 1.0):
            raise ValueError("iterative_q_learning: action must be 0 or 1")
    S = core.mat(state) if state is not None else [[] for _ in range(n)]
    if len(S) != n:
        raise ValueError("iterative_q_learning: state and reward have different lengths")
    k = len(S[0]) if S and S[0] is not None else 0
    stages = sorted(set(tm))
    T = len(stages)
    if T == 0:
        raise ValueError("iterative_q_learning: no stages")
    idx = [[i for i in range(n) if tm[i] == s] for s in stages]
    m = len(idx[0])
    for g in idx:
        if len(g) != m:
            raise ValueError("iterative_q_learning: stages have different numbers of records")
    p = 2 + 2 * k
    if m <= p:
        raise ValueError("iterative_q_learning: too few records per stage for the Q-model")

    def row(sv, av):
        out = [1.0]
        out.extend(sv)
        out.append(av)
        out.extend([av * z for z in sv])
        return out

    Vnext = [0.0] * m
    betas = []
    shares = []
    values = []
    for t in range(T - 1, -1, -1):
        g = idx[t]
        Z = [row(S[i], a[i]) for i in g]
        tgt = [r[g[u]] + gamma * Vnext[u] for u in range(m)]
        b = core.lstsq(Z, tgt)
        q0 = [sum(row(S[i], 0.0)[j] * b[j] for j in range(p)) for i in g]
        q1 = [sum(row(S[i], 1.0)[j] * b[j] for j in range(p)) for i in g]
        Vnext = [q1[u] if q1[u] > q0[u] else q0[u] for u in range(m)]
        betas.append(b)
        shares.append(sum(1.0 for u in range(m) if q1[u] > q0[u]) / m)
        values.append(sum(Vnext) / m)
    betas.reverse()
    shares.reverse()
    values.reverse()
    flat = []
    for b in betas:
        flat.extend(b)
    value = values[0]
    return RichResult(
        title="Iterative Q-learning for dynamic regimes",
        summary_lines=[("stages", T), ("subjects", m), ("regime value", value)],
        payload={
            "estimate": value,
            "value": value,
            "stage_value": values,
            "coef": flat,
            "share_treated": shares,
            "n_stages": T,
            "n_subjects": m,
            "gamma": gamma,
            "n": n,
            "method": "Q_t(s,a) <- R_t + gamma max_a' Q_{t+1}(s',a') by least squares, Murphy (2003)",
        },
    )


def cheatsheet():
    return "itrlrn: Iterative Q-learning for dynamic regimes"


# compact alias per ledger/NAMING.md
iterativeqlearning = iterative_q_learning
