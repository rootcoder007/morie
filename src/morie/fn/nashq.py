r"""Nash Q-learning for general-sum stochastic games.

Hu, J., & Wellman, M. P. (2003) "Nash Q-Learning for General-Sum
Stochastic Games", *Journal of Machine Learning Research* 4,
1039-1069.

Single-agent Q-learning bootstraps off :math:`\max_{a'} Q(s', a')`. In
a stochastic game that is meaningless: the payoff of an action depends
on what everyone else does, so there is no "own maximum" to take. Nash
Q-learning replaces the max with an **equilibrium operator**. Agent
:math:`i`'s Nash Q-function is defined over the *joint* action
(Definition 5):

.. math:: Q^i_*(s, a^1, \dots, a^n) = r^i(s, a^1, \dots, a^n)
          + \beta \sum_{s'} p(s' \mid s, a^1,\dots,a^n)\,
          v^i(s', \pi^1_*, \dots, \pi^n_*),

the current reward plus the future reward when *all* agents play a
joint Nash equilibrium strategy from the next period on. The learning
rule (eqs. 6-7) is then

.. math:: Q^i_{t+1}(s, \mathbf{a}) = (1-\alpha_t) Q^i_t(s, \mathbf{a})
          + \alpha_t\big[r^i_t + \beta\,\mathrm{NashQ}^i_t(s')\big],
          \qquad
          \mathrm{NashQ}^i_t(s') = \pi^1(s')\cdots\pi^n(s')\cdot
          Q^i_t(s'),

where :math:`(\pi^1(s'),\dots,\pi^n(s'))` is a Nash equilibrium of the
**stage game** :math:`(Q^1_t(s'), \dots, Q^n_t(s'))` -- the one-period
game whose payoff matrices are the current Q-values at :math:`s'`.
:math:`\mathrm{NashQ}^i_t(s')` is a scalar: agent :math:`i`'s expected
payoff under that equilibrium.

Two consequences the paper is explicit about and this implementation
inherits:

* the learner must observe **every** agent's reward and action, not
  just its own, and maintains a model :math:`Q^j` of every other
  agent's Q-function (Table 2 updates all :math:`n` of them);
* "different methods for selecting among multiple Nash equilibria will
  in general yield different updates". Equilibrium selection is not a
  detail. Section 4 proves convergence only when every stage game
  arising during learning has a **global optimal point**
  (Definition 12: every agent simultaneously gets its highest payoff)
  or a **saddle point** (Definition 13: a Nash equilibrium at which
  each agent gains when someone else deviates), and agents update at
  that point. ``selection=`` exposes exactly those choices, and
  :func:`stage_game_type` reports which -- if either -- a given stage
  game has, so a user can check the theorem's hypothesis on their own
  problem instead of assuming it.

The stage-game solver is exact support enumeration for two players,
which is appropriate here because the stage games are small (one per
state, over the joint action space) and because an approximate
equilibrium would quietly break the update rule.
"""

import itertools
import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["nashq", "nash_q_learning", "nash_equilibria_bimatrix",
           "stage_game_type"]

_SELECTIONS = ("global_optimal", "saddle", "first", "best_for_agent")


def _mat(M, name):
    rows = [[float(v) for v in r]
            for r in np.atleast_2d(np.asarray(M, dtype=float))]
    if not rows or not rows[0]:
        raise ValueError("nashq: %s must be a non-empty matrix" % name)
    w = len(rows[0])
    for r in rows:
        if len(r) != w:
            raise ValueError("nashq: %s must be rectangular" % name)
    return rows


def _solve(A, b):
    """Small dense solve with partial pivoting; None if singular."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= pv
        for r in range(n):
            if r == c:
                continue
            f = M[r][c]
            if f == 0.0:
                continue
            for j in range(c, n + 1):
                M[r][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]


def nash_equilibria_bimatrix(A, B, tol=1e-9):
    r"""All Nash equilibria of a two-player bimatrix game, by support
    enumeration.

    ``A`` is the row player's payoff matrix and ``B`` the column
    player's, both ``(m, n)``. Returns a list of ``(p, q)`` pairs of
    mixed strategies.

    Support enumeration is exhaustive rather than approximate: for each
    candidate pair of supports it solves the indifference conditions
    exactly and then verifies the result really is an equilibrium (no
    profitable deviation to any pure action). Exponential in the number
    of actions, which is fine for the stage games Nash Q-learning
    produces and wrong to approximate.
    """
    A = _mat(A, "A")
    B = _mat(B, "B")
    m, n = len(A), len(A[0])
    if len(B) != m or len(B[0]) != n:
        raise ValueError("nashq: A and B must have the same shape")

    out = []
    seen = set()
    for k in range(1, min(m, n) + 1):
        for I in itertools.combinations(range(m), k):
            for J in itertools.combinations(range(n), k):
                # Column player mixes over J to make the row player
                # indifferent across I; and vice versa.
                q = _indifference([[A[i][j] for j in J] for i in I], k)
                p = _indifference([[B[i][j] for i in I] for j in J], k)
                if q is None or p is None:
                    continue
                if min(q) < -tol or min(p) < -tol:
                    continue
                P = [0.0] * m
                Q = [0.0] * n
                for a, i in enumerate(I):
                    P[i] = max(0.0, p[a])
                for a, j in enumerate(J):
                    Q[j] = max(0.0, q[a])
                sp, sq = sum(P), sum(Q)
                if sp <= tol or sq <= tol:
                    continue
                P = [v / sp for v in P]
                Q = [v / sq for v in Q]
                if not _is_equilibrium(A, B, P, Q, tol):
                    continue
                key = (tuple(round(v, 9) for v in P),
                       tuple(round(v, 9) for v in Q))
                if key in seen:
                    continue
                seen.add(key)
                out.append((P, Q))
    return out


def _indifference(payoff, k):
    """Mix making the opponent indifferent across the k rows of payoff.

    Solves payoff . x = v * 1 with sum(x) = 1, for unknown (x, v).
    """
    # unknowns: x_0..x_{k-1}, v
    Aeq = []
    beq = []
    for i in range(k):
        Aeq.append(list(payoff[i]) + [-1.0])
        beq.append(0.0)
    Aeq.append([1.0] * k + [0.0])
    beq.append(1.0)
    sol = _solve(Aeq, beq)
    if sol is None:
        return None
    return sol[:k]


def _payoff(M, p, q):
    tot = 0.0
    for i in range(len(p)):
        if p[i] == 0.0:
            continue
        row = M[i]
        for j in range(len(q)):
            if q[j] != 0.0:
                tot += p[i] * q[j] * row[j]
    return tot


def _is_equilibrium(A, B, p, q, tol):
    va = _payoff(A, p, q)
    vb = _payoff(B, p, q)
    for i in range(len(A)):
        dev = sum(q[j] * A[i][j] for j in range(len(q)))
        if dev > va + tol:
            return False
    for j in range(len(A[0])):
        dev = sum(p[i] * B[i][j] for i in range(len(p)))
        if dev > vb + tol:
            return False
    return True


def stage_game_type(A, B, tol=1e-9):
    r"""Classify a stage game against the paper's convergence conditions.

    Returns a RichResult carrying ``equilibria``, and the booleans
    ``has_global_optimal`` (Definition 12 -- some equilibrium at which
    *every* agent gets its highest attainable payoff) and
    ``has_saddle`` (Definition 13 -- some equilibrium at which each
    agent is no worse off when the others deviate).

    Section 4's convergence theorem holds when every stage game arising
    during learning is of one of these types. The paper's own Grid Game
    2 satisfies neither, and does not always converge -- so this is
    worth checking rather than assuming.
    """
    A = _mat(A, "A")
    B = _mat(B, "B")
    eqs = nash_equilibria_bimatrix(A, B, tol)
    best_a = max(max(r) for r in A)
    best_b = max(max(r) for r in B)
    glob = []
    sad = []
    for p, q in eqs:
        va, vb = _payoff(A, p, q), _payoff(B, p, q)
        if va >= best_a - tol and vb >= best_b - tol:
            glob.append((p, q))
        if _is_saddle(A, B, p, q, tol):
            sad.append((p, q))
    return RichResult(payload={
        "estimate": len(eqs),
        "equilibria": eqs,
        "n_equilibria": len(eqs),
        "has_global_optimal": bool(glob),
        "has_saddle": bool(sad),
        "global_optimal": glob,
        "saddle": sad,
        "method": "stage game classification (Hu & Wellman 2003 "
                  "Defs 12-13)",
    })


def _is_saddle(A, B, p, q, tol):
    """Definition 13: Nash, and each agent gains when the OTHER deviates."""
    for j in range(len(A[0])):
        pure = [0.0] * len(A[0])
        pure[j] = 1.0
        if _payoff(A, p, pure) < _payoff(A, p, q) - tol:
            return False
    for i in range(len(A)):
        pure = [0.0] * len(A)
        pure[i] = 1.0
        if _payoff(B, pure, q) < _payoff(B, p, q) - tol:
            return False
    return True


def _select(A, B, selection, agent, tol):
    eqs = nash_equilibria_bimatrix(A, B, tol)
    if not eqs:
        return None
    if selection == "first":
        return eqs[0]
    if selection == "best_for_agent":
        M = A if agent == 0 else B
        return max(eqs, key=lambda e: _payoff(M, e[0], e[1]))
    if selection == "global_optimal":
        ba = max(max(r) for r in A)
        bb = max(max(r) for r in B)
        for p, q in eqs:
            if (_payoff(A, p, q) >= ba - tol
                    and _payoff(B, p, q) >= bb - tol):
                return p, q
        return eqs[0]
    for p, q in eqs:                       # "saddle"
        if _is_saddle(A, B, p, q, tol):
            return p, q
    return eqs[0]


def nashq(states, actions, step, rewards, gamma=0.9, alpha=0.5,
          epsilon=0.1, episodes=500, horizon=50, start=None,
          selection="global_optimal", terminal=(), seed=0, agent=0,
          tol=1e-9):
    r"""Tabular Nash Q-learning for a two-player general-sum stochastic
    game (Table 2 of the paper).

    Parameters
    ----------
    states : sequence
        Hashable states :math:`S`.
    actions : sequence of two sequences
        ``(A1, A2)``, each player's action set.
    step : callable
        ``step(s, a1, a2) -> s_next``. The joint action drives the
        transition, which is the whole point of a stochastic game.
    rewards : callable
        ``rewards(s, a1, a2, s_next) -> (r1, r2)``. Both rewards are
        required: the learner cannot form the stage game without the
        other agent's payoffs, and the paper is explicit that where
        they are unobservable a proxy must be substituted, with results
        depending on how good the proxy is.
    gamma : float
        :math:`\beta`, the discount.
    alpha : float
        Learning rate :math:`\alpha_t`.
    epsilon : float
        Exploration rate for behaviour; ties break at random.
    episodes, horizon : int
        Training length.
    start : callable or state, optional
        Initial state; defaults to ``states[0]``.
    selection : {"global_optimal", "saddle", "first", "best_for_agent"}
        Which equilibrium to update against when the stage game has
        several. The first two are the paper's convergence conditions.
    terminal : iterable
        States that end an episode.
    seed : int
        Seed for exploration.
    agent : int
        Which player's policy is reported in ``policy`` (both agents'
        Q-functions are learned regardless).
    tol : float
        Tolerance in the equilibrium solver.

    Returns
    -------
    RichResult
        ``estimate`` / ``q`` is ``{(player, state): payoff matrix}``.
        ``policy`` gives the selected equilibrium mixed strategy per
        state, ``nash_values`` the scalar
        :math:`\mathrm{NashQ}^i(s)`, ``returns`` the per-episode
        undiscounted return of each player, and ``stage_game_types``
        the Definition 12/13 classification of the final stage game at
        each state -- i.e. whether the convergence theorem's hypothesis
        actually held.

    References
    ----------
    Hu & Wellman (2003), *JMLR* 4, 1039-1069: Definitions 5-7 and
    12-13, eqs. 5-7, Table 2.
    """
    if selection not in _SELECTIONS:
        raise ValueError("nashq: selection must be one of %r, got %r"
                         % (_SELECTIONS, selection))
    S = list(states)
    if len(actions) != 2:
        raise ValueError("nashq: this implementation covers two players; "
                         "pass actions as (A1, A2)")
    A1, A2 = list(actions[0]), list(actions[1])
    if not S or not A1 or not A2:
        raise ValueError("nashq: states and both action sets must be "
                         "non-empty")
    if not callable(step) or not callable(rewards):
        raise TypeError("nashq: step and rewards must be callable")
    term = set(terminal)
    s0 = start if callable(start) else (lambda: S[0] if start is None
                                        else start)
    rng = np.random.default_rng(seed)

    # Q[(player, state)] is a |A1| x |A2| payoff matrix -- the joint
    # action Q-function of Definition 5, one per player.
    Q = {}
    for pl in (0, 1):
        for s in S:
            Q[(pl, s)] = [[0.0] * len(A2) for _ in range(len(A1))]

    returns = []
    for _ep in range(int(episodes)):
        s = s0()
        tot = [0.0, 0.0]
        for _t in range(int(horizon)):
            if s in term:
                break
            i = _pick(Q[(0, s)], A1, 0, epsilon, rng)
            j = _pick(Q[(1, s)], A2, 1, epsilon, rng)
            s1 = step(s, A1[i], A2[j])
            r1, r2 = rewards(s, A1[i], A2[j], s1)
            tot[0] += r1
            tot[1] += r2
            # NashQ(s') -- eq. 7. One equilibrium of the stage game
            # (Q^1(s'), Q^2(s')), the SAME one for both updates.
            if s1 in term:
                nv = (0.0, 0.0)
            else:
                eq = _select(Q[(0, s1)], Q[(1, s1)], selection, agent, tol)
                if eq is None:
                    nv = (0.0, 0.0)
                else:
                    p, q = eq
                    nv = (_payoff(Q[(0, s1)], p, q),
                          _payoff(Q[(1, s1)], p, q))
            for pl, r in ((0, r1), (1, r2)):
                cur = Q[(pl, s)][i][j]
                Q[(pl, s)][i][j] = ((1.0 - alpha) * cur
                                    + alpha * (r + gamma * nv[pl]))
            s = s1
        returns.append(tot)

    policy = {}
    nash_values = {}
    types = {}
    for s in S:
        cls = stage_game_type(Q[(0, s)], Q[(1, s)], tol)
        types[s] = ("global_optimal" if cls["has_global_optimal"]
                    else "saddle" if cls["has_saddle"]
                    else "neither" if cls["n_equilibria"]
                    else "none_found")
        eq = _select(Q[(0, s)], Q[(1, s)], selection, agent, tol)
        if eq is None:
            continue
        p, q = eq
        policy[s] = (p, q)
        nash_values[s] = (_payoff(Q[(0, s)], p, q),
                          _payoff(Q[(1, s)], p, q))

    tenth = max(1, int(episodes) // 10)
    return RichResult(payload={
        "estimate": Q,
        "q": Q,
        "policy": policy,
        "nash_values": nash_values,
        "stage_game_types": types,
        "returns": returns,
        "mean_return_last": [
            sum(r[pl] for r in returns[-tenth:]) / tenth for pl in (0, 1)],
        "selection": selection,
        "method": "Nash Q-learning (Hu & Wellman 2003, Table 2)",
    })


def _pick(M, A, who, epsilon, rng):
    """epsilon-greedy on the agent's own row/column marginal, ties random."""
    if rng.random() < epsilon:
        return int(rng.random() * len(A))
    if who == 0:
        vals = [sum(r) / len(r) for r in M]
    else:
        vals = [sum(M[i][j] for i in range(len(M))) / len(M)
                for j in range(len(M[0]))]
    bv = max(vals)
    best = [k for k, v in enumerate(vals) if v >= bv - 1e-15]
    return best[int(rng.random() * len(best))] if len(best) > 1 else best[0]


def cheatsheet():
    return ("nashq: Q^i over JOINT actions; update with the stage-game "
            "Nash payoff instead of a max -- Q^i <- (1-a)Q^i + "
            "a[r^i + beta pi^1...pi^n Q^i(s')] (Hu & Wellman 2003 "
            "eqs. 6-7). Needs every agent's reward. Equilibrium "
            "selection changes the update: convergence is proved only "
            "for global optimal (Def 12) or saddle (Def 13) stage "
            "games. stage_game_type() reports which you have.")


# compact aliases per ledger/NAMING.md
nash_q_learning = nashq
nashqlearning = nashq
