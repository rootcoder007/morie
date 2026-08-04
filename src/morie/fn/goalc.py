# morie.fn -- slice s03 (rootcoder007/morie)
"""Goal-conditioned value functions.

Source consulted: Schaul, T., Horgan, D., Gregor, K. and Silver, D.
(2015).  Universal value function approximators.  *ICML* 37, 1312-1320.
The paper generalises a value function to V(s, g), a single
approximator over states *and* goals, with a goal-dependent
pseudo-reward r_g(s, a, s'); the canonical choice, and the one used in
the paper's experiments and in Andrychowicz et al. (2017), Hindsight
experience replay (arXiv:1707.01495), is the sparse indicator

    r_g(s) = 0  if s == g,  -1 otherwise

so that V(s, g) is the negated expected number of steps to reach g.  The
ICML proceedings version is free but was not retrievable here, so the
definition above is quoted in its standard published form; it is
unambiguous and reproduced identically in both papers.

Given a deterministic transition list the goal-conditioned values are
computed exactly, by a backward breadth-first sweep from each goal,
which is value iteration specialised to a deterministic MDP with unit
costs -- no sampling, so no generator.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["goal_conditioned"]


def goal_conditioned(env, policy=None, goal_dist=None, n_states=None,
                     gamma=1.0, step_cost=-1.0):
    """Goal-conditioned values over a deterministic transition list.

    Parameters
    ----------
    env : 2-D array-like
        Deterministic transitions, one row per (s, a, s').
    policy : any, optional
        Ignored; present for signature stability (the values returned are
        the optimal ones, V*(s, g)).
    goal_dist : array-like, optional
        Goals, or (goal, weight) pairs.  Defaults to every state.
    n_states : int, optional
        Number of states; inferred when absent.
    gamma : float
        Discount; 1 gives negated shortest-path length.
    step_cost : float
        The per-step pseudo-reward; -1 in the sparse indicator scheme.

    Returns
    -------
    RichResult with payload:
        estimate      : expected V(0, g) over the goal distribution
        v             : V[s][g] for every state and goal
        expected_value: expected V(s, .) per state
        reachable     : share of (s, g) pairs that are reachable
    """
    rows = k.mat(env)
    ns = int(n_states) if n_states is not None else (
        int(max([max(r[0], r[2]) for r in rows]) + 1) if rows else 0)
    succ = [[] for _ in range(ns)]
    pred = [[] for _ in range(ns)]
    for row in rows:
        s, s2 = int(row[0]), int(row[2])
        succ[s].append(s2)
        pred[s2].append(s)
    gd = k.mat(goal_dist) if goal_dist is not None else [[float(s), 1.0] for s in range(ns)]
    if gd and len(gd[0]) == 1:
        gd = [[r[0], 1.0] for r in gd]
    goals = [int(r[0]) for r in gd]
    wts = [r[1] for r in gd]
    wtot = 0.0
    for x in wts:
        wtot += x
    V = [[float("-inf")] * len(goals) for _ in range(ns)]
    reach = 0
    for gi in range(len(goals)):
        g = goals[gi]
        dist = [-1] * ns
        dist[g] = 0
        frontier = [g]
        while frontier:
            nxt = []
            for u in frontier:
                for p in pred[u]:
                    if dist[p] < 0:
                        dist[p] = dist[u] + 1
                        nxt.append(p)
            frontier = nxt
        for s in range(ns):
            if dist[s] < 0:
                V[s][gi] = float("-inf")
            else:
                d = dist[s]
                if float(gamma) == 1.0:
                    V[s][gi] = float(step_cost) * d
                else:
                    acc = 0.0
                    for j in range(d):
                        acc += (float(gamma) ** j) * float(step_cost)
                    V[s][gi] = acc
                reach += 1
    ev = []
    for s in range(ns):
        acc = 0.0
        for gi in range(len(goals)):
            acc += wts[gi] * (V[s][gi] if V[s][gi] != float("-inf") else 0.0)
        ev.append(acc / wtot if wtot > 0.0 else float("nan"))
    return RichResult(
        title="Goal-conditioned values",
        summary_lines=[("states", ns), ("goals", len(goals))],
        payload={
            "estimate": ev[0] if ev else float("nan"),
            "v": V,
            "expected_value": ev,
            "reachable": reach / (ns * len(goals)) if ns and goals else float("nan"),
            "n_states": ns,
            "method": "Goal-conditioned V*(s, g) with the sparse pseudo-reward (UVFA)",
        },
    )


def cheatsheet():
    return "goalc: Goal-conditioned RL"
