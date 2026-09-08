# morie.fn -- slice s03 (rootcoder007/morie)
"""Model-based RL: fit a tabular model, then plan in it.

Source consulted: Sutton, R. S. (1991).  Dyna, an integrated
architecture for learning, planning, and reacting.  *SIGART Bulletin*
2(4), 160-163, whose whole point is that the same experience can be used
twice -- once to learn a model, once to plan with it.  The model here is
the maximum-likelihood tabular one,

    phat(s' | s, a) = N(s, a, s') / N(s, a)
    rhat(s, a)      = ( sum of rewards on (s, a) ) / N(s, a)

and the planner is value iteration (Bellman, R. 1957, *Dynamic
Programming*, Princeton),

    V(s) <- max_a [ rhat(s,a) + gamma sum_s' phat(s'|s,a) V(s') ]

run to a fixed tolerance.  Neither the 1991 bulletin nor the 1957 book
was available here as a full text; both equations are quoted in their
standard published form and are reproduced identically in Sutton and
Barto (2018) sections 8.1-8.2 and 4.4 (FETCHED from
incompleteideas.net).  Unvisited (s, a) pairs contribute nothing and are
excluded from the maximisation rather than assigned an invented value.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["model_based_rl"]


def model_based_rl(env, model=None, planner="vi", n_states=None,
                   n_actions=None, gamma=0.95, tol=1e-12, max_iter=1000):
    """Fit phat and rhat from transitions, then value-iterate.

    Parameters
    ----------
    env : 2-D array-like
        Transitions, one row per experience: (s, a, r, s').
    model : any, optional
        Ignored; present for signature stability.
    planner : str
        Only "vi" (value iteration) is implemented; anything else raises.
    n_states, n_actions : int, optional
        Sizes of the tabular model; inferred from the data when absent.
    gamma : float
        Discount.
    tol : float
        Sweep-to-sweep tolerance on max |V change|.
    max_iter : int
        Sweep cap.

    Returns
    -------
    RichResult with payload:
        estimate : V of state 0
        v        : the value function
        policy   : the greedy action per state (-1 where unvisited)
        sweeps   : number of sweeps performed
    """
    if planner != "vi":
        raise ValueError("only planner='vi' (value iteration) is implemented")
    rows = k.mat(env)
    ns = int(n_states) if n_states is not None else int(
        max([max(r[0], r[3]) for r in rows]) + 1) if rows else 0
    na = int(n_actions) if n_actions is not None else int(
        max([r[1] for r in rows]) + 1) if rows else 0
    cnt = [[0.0] * na for _ in range(ns)]
    rsum = [[0.0] * na for _ in range(ns)]
    trans = [[[0.0] * ns for _ in range(na)] for _ in range(ns)]
    for row in rows:
        s, a, rw, s2 = int(row[0]), int(row[1]), row[2], int(row[3])
        cnt[s][a] += 1.0
        rsum[s][a] += rw
        trans[s][a][s2] += 1.0
    V = [0.0] * ns
    sweeps = 0
    for _ in range(int(max_iter)):
        sweeps += 1
        delta = 0.0
        for s in range(ns):
            best = None
            for a in range(na):
                if cnt[s][a] <= 0.0:
                    continue
                q = rsum[s][a] / cnt[s][a]
                acc = 0.0
                for s2 in range(ns):
                    if trans[s][a][s2] > 0.0:
                        acc += (trans[s][a][s2] / cnt[s][a]) * V[s2]
                q += float(gamma) * acc
                if best is None or q > best:
                    best = q
            nv = best if best is not None else V[s]
            d = abs(nv - V[s])
            if d > delta:
                delta = d
            V[s] = nv
        if delta <= float(tol):
            break
    pol = []
    for s in range(ns):
        best = None
        ba = -1
        for a in range(na):
            if cnt[s][a] <= 0.0:
                continue
            q = rsum[s][a] / cnt[s][a]
            acc = 0.0
            for s2 in range(ns):
                if trans[s][a][s2] > 0.0:
                    acc += (trans[s][a][s2] / cnt[s][a]) * V[s2]
            q += float(gamma) * acc
            if best is None or q > best:
                best = q
                ba = a
        pol.append(ba)
    return RichResult(
        title="Model-based RL",
        summary_lines=[("states", ns), ("sweeps", sweeps)],
        payload={
            "estimate": V[0] if V else float("nan"),
            "v": V,
            "policy": pol,
            "sweeps": sweeps,
            "n_states": ns,
            "n_actions": na,
            "method": "Maximum-likelihood tabular model plus value iteration (Dyna)",
        },
    )


def cheatsheet():
    return "mfomf: Model-based RL planning"


modelbasedrl = model_based_rl
