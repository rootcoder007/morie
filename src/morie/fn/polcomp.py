# morie.fn -- function file (rootcoder007/morie)
"""Spatial party competition and legislative bargaining: proximity and logit vote shares with
salience weights, best-response (Hotelling-Downs) equilibria on a grid, Duvergerian strategic
voting, Palfrey entry deterrence, the effective number of parties, Rubinstein bargaining, the median (core) party, Laver's adaptive party dynamics, party
mergers and splits, and the Romer-Rosenthal setter and Denzau-Mackay committee models."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = [
    "vote_shares",
    "best_response_equilibrium",
    "strategic_vote_shares",
    "entry_game",
    "effective_number_parties",
    "rubinstein_bargaining",
    "median_party",
    "laver_dynamics",
    "merge_split_parties",
    "setter_outcome",
    "committee_outcome",
]


def _pts(x):
    return [[float(v)] if not isinstance(v, (list, tuple)) else [float(a) for a in v] for v in x]


def _dist2(a, b, sal):
    return ssum(s * (u - v) ** 2 for u, v, s in zip(a, b, sal))


def vote_shares(positions, voters, *, weights=None, rule: str = "proximity", beta: float = 1.0, salience=None) -> list:
    r"""Party vote shares under proximity (Downs/Hotelling) or logit (probabilistic) voting.

    Distances are salience-weighted Euclidean,
    ``d_ij^2 = sum_k s_k (v_ik - p_jk)^2`` (Rabinowitz and Macdonald's
    salience weights; 1 by default). ``proximity``: each voter supports the
    nearest party, ties split equally. ``logit``: ``P_ij = exp(-beta d_ij^2) / sum_l exp(-beta d_il^2)``.
    Voters are points (1-D values or vectors) with optional weights.

    References
    ----------
    Downs, A. (1957). An Economic Theory of Democracy. Hotelling, H. (1929).
    Stability in competition. Economic J. 39, 41-57. Adams, J., Merrill, S.
    and Grofman, B. (2005). A Unified Theory of Party Competition, ch. 2.

    Examples
    --------
    >>> vote_shares([0.0, 1.0], [0.1, 0.4, 0.5, 0.9])
    [0.625, 0.375]
    """
    P, V = _pts(positions), _pts(voters)
    w = [1.0] * len(V) if weights is None else [float(v) for v in weights]
    sal = [1.0] * len(P[0]) if salience is None else [float(v) for v in salience]
    tot = ssum(w)
    sh = [0.0] * len(P)
    for v, wv in zip(V, w):
        d = [_dist2(v, p, sal) for p in P]
        if rule == "logit":
            m = min(d)
            e = [math.exp(-beta * (x - m)) for x in d]
            s = ssum(e)
            for j in range(len(P)):
                sh[j] += wv * e[j] / s
        else:
            m = min(d)
            win = [j for j in range(len(P)) if d[j] == m]
            for j in win:
                sh[j] += wv / len(win)
    return [v / tot for v in sh]


def best_response_equilibrium(
    voters,
    n_parties: int,
    grid,
    *,
    weights=None,
    start=None,
    rule: str = "proximity",
    beta: float = 1.0,
    max_rounds: int = 200,
) -> RichResult:
    r"""Vote-maximising positions by sequential best responses on a grid (Hotelling-Downs competition).

    Parties in turn move to the grid point maximising their own share given
    the others (a move needs a gain above 1e-12; among
    improving points the lowest index wins); a round
    with no move is a pure Nash equilibrium on the grid. With two parties and
    proximity voting this is the median voter; with more parties or a
    double-peaked electorate equilibria may fail to exist (Eaton and Lipsey
    1975), reported as ``converged = False``.

    References
    ----------
    Eaton, B. C. and Lipsey, R. G. (1975). The principle of minimum
    differentiation reconsidered. Rev. Econ. Stud. 42, 27-49. Cox, G. W.
    (1990). Centripetal and centrifugal incentives in electoral systems.
    AJPS 34, 903-935.

    Examples
    --------
    >>> r = best_response_equilibrium([0.1, 0.2, 0.3, 0.8, 0.9], 2, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    >>> r.positions, r.converged
    ([[0.3], [0.3]], True)
    """
    G = _pts(grid)
    idx = list(range(n_parties)) if start is None else list(start)
    idx = [min(i, len(G) - 1) for i in idx]
    rounds = 0
    converged = False
    for _ in range(max_rounds):
        rounds += 1
        moved = False
        for j in range(n_parties):
            best = idx[j]
            bs = vote_shares([G[i] for i in idx], voters, weights=weights, rule=rule, beta=beta)[j]
            for g in range(len(G)):
                pos = [G[idx[q]] if q != j else G[g] for q in range(n_parties)]
                s = vote_shares(pos, voters, weights=weights, rule=rule, beta=beta)[j]
                if s > bs + 1e-12:
                    best, bs = g, s
            if best != idx[j]:
                idx[j] = best
                moved = True
        if not moved:
            converged = True
            break
    pos = [G[i] for i in idx]
    return RichResult(
        payload={
            "positions": pos,
            "shares": vote_shares(pos, voters, weights=weights, rule=rule, beta=beta),
            "grid_index": idx,
            "converged": converged,
            "rounds": rounds,
        }
    )


def strategic_vote_shares(positions, voters, *, weights=None, magnitude: int = 1) -> RichResult:
    r"""Duvergerian strategic voting: supporters of non-viable parties switch to the nearest viable one.

    Sincere proximity shares identify the ``M + 1`` viable parties (Cox's
    ``M + 1`` rule for district magnitude ``M``; ties to the lower index);
    voters whose sincere choice is not viable vote for their nearest viable
    party (the wasted-vote logic).

    References
    ----------
    Cox, G. W. (1997). Making Votes Count, ch. 4. Duverger, M. (1954).
    Political Parties.

    Examples
    --------
    >>> r = strategic_vote_shares([0.0, 0.5, 1.0], [0.0, 0.1, 0.45, 0.6, 0.9, 1.0, 0.95])
    >>> r.viable, r.strategic
    ([0, 2], [0.42857142857142855, 0.0, 0.5714285714285714])
    """
    P, V = _pts(positions), _pts(voters)
    w = [1.0] * len(V) if weights is None else [float(v) for v in weights]
    sinc = vote_shares(P, V, weights=w)
    order = sorted(range(len(P)), key=lambda j: (-sinc[j], j))
    viable = sorted(order[: magnitude + 1])
    tot = ssum(w)
    sh = [0.0] * len(P)
    one = [1.0] * len(P[0])
    for v, wv in zip(V, w):
        d = [_dist2(v, p, one) for p in P]
        m = min(d[j] for j in viable)
        win = [j for j in viable if d[j] == m]
        for j in win:
            sh[j] += wv / len(win)
    return RichResult(payload={"sincere": sinc, "strategic": [v / tot for v in sh], "viable": viable})


def entry_game(voters, grid, *, weights=None, n_incumbents: int = 2, max_rounds: int = 100) -> RichResult:
    r"""Entry deterrence: incumbents position anticipating a vote-maximising entrant (Palfrey 1984).

    For incumbent positions the entrant takes its best grid response (ties to
    the lowest index); incumbents then best-respond to each other taking the
    entrant's reaction into account, sequentially until no incumbent moves
    (a subgame-perfect equilibrium on the grid). Incumbents are pushed apart
    from the median, the result Palfrey derived for two incumbents.

    References
    ----------
    Palfrey, T. R. (1984). Spatial equilibrium with entry. Rev. Econ. Stud.
    51, 139-156. Shepsle, K. A. (1991). Models of Multiparty Electoral Competition.

    Examples
    --------
    >>> v = [i / 20 for i in range(21)]
    >>> r = entry_game(v, [i / 10 for i in range(11)])
    >>> len(r.incumbents), len(r.shares)
    (2, 3)
    """
    G = _pts(grid)

    def entrant(inc):
        best, bs = 0, None
        for g in range(len(G)):
            s = vote_shares([G[i] for i in inc] + [G[g]], voters, weights=weights)[-1]
            if bs is None or s > bs + 1e-12:
                best, bs = g, s
        return best

    def payoff(inc, j):
        e = entrant(inc)
        return vote_shares([G[i] for i in inc] + [G[e]], voters, weights=weights)[j]

    mid = len(G) // 2
    inc = [max(0, mid - 1 - q) if q % 2 == 0 else min(len(G) - 1, mid + q) for q in range(n_incumbents)]
    for _ in range(max_rounds):
        moved = False
        for j in range(n_incumbents):
            cur = payoff(inc, j)
            best, bs = inc[j], cur
            for g in range(len(G)):
                trial = list(inc)
                trial[j] = g
                s = payoff(trial, j)
                if s > bs + 1e-12:
                    best, bs = g, s
            if best != inc[j]:
                inc[j] = best
                moved = True
        if not moved:
            break
    e = entrant(inc)
    pos = [G[i] for i in inc] + [G[e]]
    return RichResult(
        payload={
            "incumbents": [G[i] for i in inc],
            "entrant": G[e],
            "shares": vote_shares(pos, voters, weights=weights),
        }
    )


def effective_number_parties(shares) -> RichResult:
    r"""Effective number of parties: Laakso-Taagepera ``1 / sum p_i^2`` and Golosov's ``sum p_i / (p_i + p_1^2 - p_i^2)``.

    Shares are normalised to sum to one; ``p_1`` is the largest.

    References
    ----------
    Laakso, M. and Taagepera, R. (1979). Effective number of parties.
    Comparative Political Studies 12, 3-27. Golosov, G. V. (2010). The
    effective number of parties: a new approach. Party Politics 16, 171-192.

    Examples
    --------
    >>> r = effective_number_parties([0.5, 0.3, 0.2])
    >>> round(r.laakso_taagepera, 10), round(r.golosov, 10)
    (2.6315789474, 2.1399787911)
    """
    s = [float(v) for v in shares]
    t = ssum(s)
    p = [v / t for v in s]
    p1 = max(p)
    return RichResult(
        payload={
            "laakso_taagepera": 1.0 / ssum(v * v for v in p),
            "golosov": ssum(v / (v + p1 * p1 - v * v) for v in p),
        }
    )


def rubinstein_bargaining(delta1: float, delta2: float) -> RichResult:
    r"""Rubinstein (1982) alternating-offers bargaining: the proposer's share ``(1 - delta2) / (1 - delta1 delta2)``.

    Also reports the responder's share and, for discount factors
    ``exp(-r_i dt)``, the limiting asymmetric Nash bargaining weight of player
    one, ``r_2 / (r_1 + r_2)`` (Binmore, Rubinstein and Wolinsky 1986).

    References
    ----------
    Rubinstein, A. (1982). Perfect equilibrium in a bargaining model.
    Econometrica 50, 97-109. Binmore, K., Rubinstein, A. and Wolinsky, A.
    (1986). The Nash bargaining solution in economic modelling. RAND J. Econ. 17, 176-188.

    Examples
    --------
    >>> r = rubinstein_bargaining(0.9, 0.9)
    >>> round(r.proposer, 12), round(r.nash_weight, 12)
    (0.526315789474, 0.5)
    """
    x1 = (1.0 - delta2) / (1.0 - delta1 * delta2)
    r1, r2 = -math.log(delta1), -math.log(delta2)
    return RichResult(payload={"proposer": x1, "responder": 1.0 - x1, "nash_weight": r2 / (r1 + r2)})


def median_party(positions, seats) -> RichResult:
    r"""The median (core) party of a one-dimensional legislature (Laver and Schofield 1990).

    Parties ordered by position; the party holding the seat-weighted median
    legislator is in the core: no majority prefers another policy to its
    ideal point. Returns its index, position, seat share and whether it holds
    a majority on its own.

    References
    ----------
    Laver, M. and Schofield, N. (1990). Multiparty Government, ch. 5. Black,
    D. (1948). On the rationale of group decision-making. JPE 56, 23-34.

    Examples
    --------
    >>> median_party([-1.0, 0.2, 0.8], [40, 15, 45]).index
    1
    """
    pos = [float(v) for v in positions]
    s = [float(v) for v in seats]
    tot = ssum(s)
    order = sorted(range(len(pos)), key=lambda j: (pos[j], j))
    acc = 0.0
    med = order[-1]
    for j in order:
        acc += s[j]
        if acc > tot / 2:
            med = j
            break
    return RichResult(
        payload={"index": med, "position": pos[med], "seat_share": s[med] / tot, "majority": s[med] > tot / 2}
    )


def laver_dynamics(voters, positions, rules, n_steps: int, *, speed: float = 0.1, seed: int = 0) -> RichResult:
    r"""Laver's (2005) agent-based party competition with adaptive decision rules.

    Each step voters support the nearest party; then ``sticker`` parties stay
    put, ``aggregator`` parties jump to the centroid of their supporters,
    ``predator`` parties move ``speed`` towards the largest party, and
    ``hunter`` parties repeat their last move if it raised their share and
    otherwise turn around and take a random heading within 90 degrees of the
    reverse direction (Philox stream ``step * P + party``).

    References
    ----------
    Laver, M. (2005). Policy and the dynamics of political competition. APSR
    99, 263-281. Laver, M. and Sergenti, E. (2012). Party Competition: An
    Agent-Based Model. Princeton UP.

    Examples
    --------
    >>> v = [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    >>> r = laver_dynamics(v, [(0.5, 0.5), (-0.5, -0.5)], ["aggregator", "sticker"], 2)
    >>> [round(a, 12) for a in r.path[-1][0]]
    [0.5, 0.5]
    """
    V = _pts(voters)
    P = _pts(positions)
    k = len(P)
    heading = [[0.0, 0.0] for _ in range(k)]
    last = vote_shares(P, V)
    path = [[list(p) for p in P]]
    shares = [last]
    prev = list(last)
    for step in range(n_steps):
        sh = vote_shares(P, V)
        new = [list(p) for p in P]
        big = max(range(k), key=lambda j: (sh[j], -j))
        for j, rule in enumerate(rules):
            if rule == "aggregator":
                sup = [v for v in V if min(range(k), key=lambda q: (_dist2(v, P[q], [1.0, 1.0]), q)) == j]
                if sup:
                    new[j] = [ssum(v[0] for v in sup) / len(sup), ssum(v[1] for v in sup) / len(sup)]
            elif rule == "predator" and big != j:
                dx, dy = P[big][0] - P[j][0], P[big][1] - P[j][1]
                dd = math.hypot(dx, dy)
                if dd > 0:
                    m = min(speed, dd)
                    new[j] = [P[j][0] + m * dx / dd, P[j][1] + m * dy / dd]
            elif rule == "hunter":
                if step == 0 or sh[j] <= prev[j]:
                    u = float(random_uniform(1, seed=seed, stream=step * k + j)[0])
                    base = math.atan2(-heading[j][1], -heading[j][0]) if step > 0 else 0.0
                    ang = base + (u - 0.5) * math.pi if step > 0 else 2 * math.pi * u
                    heading[j] = [math.cos(ang), math.sin(ang)]
                new[j] = [P[j][0] + speed * heading[j][0], P[j][1] + speed * heading[j][1]]
        prev = sh
        P = new
        path.append([list(p) for p in P])
        shares.append(vote_shares(P, V))
    return RichResult(payload={"path": path, "shares": shares})


def merge_split_parties(positions, voters, *, weights=None, merge=None, split=None) -> RichResult:
    r"""Vote shares before and after a party merger or split under proximity voting.

    ``merge = (i, j)`` replaces parties ``i`` and ``j`` by one at their
    share-weighted mean position; ``split = (i, delta)`` replaces party ``i``
    by two parties at ``position -+ delta``. Returns the old and new shares
    and the combined gain of the merging or splitting parties.

    References
    ----------
    Kaminski, M. M. (2001). Coalitional stability of multi-party systems:
    evidence from Poland. AJPS 45, 294-312. Laver, M. and Benoit, K. (2003).
    The evolution of party systems between elections. AJPS 47, 215-233.

    Examples
    --------
    >>> r = merge_split_parties([0.0, 0.4, 1.0], [0.0, 0.2, 0.3, 0.5, 0.8, 1.0], merge=(0, 1))
    >>> round(r.gain, 12) + 0.0
    0.0
    """
    P = _pts(positions)
    old = vote_shares(P, voters, weights=weights)
    if merge is not None:
        i, j = merge
        wi, wj = old[i], old[j]
        mpos = [(wi * a + wj * b) / (wi + wj) if wi + wj > 0 else (a + b) / 2 for a, b in zip(P[i], P[j])]
        newP = [p for q, p in enumerate(P) if q not in (i, j)] + [mpos]
        new = vote_shares(newP, voters, weights=weights)
        gain = new[-1] - (wi + wj)
    else:
        i, delta = split
        newP = [p for q, p in enumerate(P) if q != i] + [[a - delta for a in P[i]], [a + delta for a in P[i]]]
        new = vote_shares(newP, voters, weights=weights)
        gain = new[-1] + new[-2] - old[i]
    return RichResult(payload={"old": old, "new": new, "positions": newP, "gain": gain})


def setter_outcome(setter: float, median: float, status_quo: float) -> RichResult:
    r"""Romer-Rosenthal agenda-setter outcome in one dimension.

    The floor (median ``m``) accepts any proposal at least as close to ``m``
    as the status quo ``q``: the interval ``[m - |q - m|, m + |q - m|]``; the
    setter proposes the point of that interval closest to its ideal. The
    chair's advantage is the outcome's distance from the median.

    References
    ----------
    Romer, T. and Rosenthal, H. (1978). Political resource allocation,
    controlled agendas, and the status quo. Public Choice 33, 27-43.

    Examples
    --------
    >>> setter_outcome(0.9, 0.5, 0.3).outcome
    0.7
    """
    r = abs(status_quo - median)
    x = min(max(setter, median - r), median + r)
    return RichResult(
        payload={"outcome": x, "advantage": abs(x - median), "accepted_interval": [median - r, median + r]}
    )


def committee_outcome(committee, floor, status_quo: float, *, rule: str = "closed") -> RichResult:
    r"""Committee gatekeeping and proposal power (Denzau and Mackay 1983).

    With committee median ``c`` and floor median ``m``: under the open rule a
    reported bill ends at ``m``; under the closed rule the committee proposes
    its best floor-acceptable point (:func:`setter_outcome`). The committee
    keeps the gates closed (status quo stands) unless it prefers the
    resulting outcome to ``q``.

    References
    ----------
    Denzau, A. T. and Mackay, R. J. (1983). Gatekeeping and monopoly power of
    committees: an analysis of sincere and sophisticated behavior. AJPS 27,
    740-761. Krehbiel, K. (1991). Information and Legislative Organization.

    Examples
    --------
    >>> committee_outcome([0.7, 0.8, 0.9], [0.1, 0.3, 0.5, 0.6, 0.9], 0.2).outcome
    0.8
    """

    def med(v):
        s = sorted(float(a) for a in v)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    c, m = med(committee), med(floor)
    prop = m if rule == "open" else setter_outcome(c, m, status_quo).outcome
    opened = abs(prop - c) < abs(status_quo - c)
    return RichResult(
        payload={
            "outcome": prop if opened else status_quo,
            "gate_opened": opened,
            "committee_median": c,
            "floor_median": m,
        }
    )


def cheatsheet() -> str:
    return (
        "vote_shares / best_response_equilibrium / strategic_vote_shares / entry_game / effective_number_parties / "
        "rubinstein_bargaining / median_party / laver_dynamics / merge_split_parties / "
        "setter_outcome / committee_outcome -> spatial party competition and legislative bargaining."
    )
