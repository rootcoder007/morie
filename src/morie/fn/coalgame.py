# morie.fn -- function file (rootcoder007/morie)
"""Coalition games in the spatial model: median lines, the core of a weighted voting game in the plane
(Schofield), the heart bounded by the median lines, quota solutions, minimal-range coalitions and Roemer's
party-unanimity Nash equilibrium."""

from __future__ import annotations

import itertools
import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._s03core import jacobi

__all__ = [
    "median_lines",
    "weighted_game_core",
    "spatial_heart",
    "quota_solution",
    "minimal_range_coalition",
    "roemer_pune",
]


def _pts(P):
    return [(float(a), float(b)) for a, b in P]


def _side(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def median_lines(positions, weights, quota: float) -> list:
    r"""Median lines of a weighted voting game with party positions in the plane.

    The line through the positions of parties ``i`` and ``j`` is a median
    line when each closed half-plane it bounds holds parties with total
    weight at least ``quota`` (a winning coalition on each side, the line
    included). Returned as index pairs ``(i, j)``, ``i < j`` (0-based).

    References
    ----------
    Schofield, N. (1986). Existence of a 'structurally stable' equilibrium for
    a non-collegial voting rule. *Public Choice*, 51, 267-284.
    McKelvey, R. D. (1986). Covering, dominance, and institution-free
    properties of social choice. *AJPS*, 30, 283-314.

    Examples
    --------
    >>> median_lines([(0, 0), (1, 0), (0, 1)], [1, 1, 1], 2)
    [(0, 1), (0, 2), (1, 2)]
    """
    P = _pts(positions)
    w = [float(v) for v in weights]
    n = len(P)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if P[i] == P[j]:
                continue
            s = [_side(P[k], P[i], P[j]) for k in range(n)]
            left = ssum(w[k] for k in range(n) if s[k] >= -1e-12)
            right = ssum(w[k] for k in range(n) if s[k] <= 1e-12)
            if left >= quota and right >= quota:
                out.append((i, j))
    return out


def weighted_game_core(positions, weights, quota: float) -> RichResult:
    r"""Core of a weighted majority game with Euclidean preferences in the plane (Schofield's core party).

    A party position ``z_j`` is in the core iff no open half-plane bounded
    by a line through ``z_j`` contains a winning coalition (weight at least
    ``quota``) -- otherwise its members would all prefer a point on their
    side. The lines are tested at every critical direction (towards each
    other party) and between consecutive critical directions. Returns the
    core parties (0-based) and the corresponding positions.

    References
    ----------
    Plott, C. R. (1967). A notion of equilibrium and its possibility under
    majority rule. *American Economic Review*, 57, 787-806.
    Schofield, N. (1993). Political competition and multiparty coalition
    governments. *European Journal of Political Research*, 23, 1-33.

    Examples
    --------
    >>> weighted_game_core([(0, 0), (1, 0), (-1, 0), (0, 1)], [40, 20, 20, 20], 51).core
    [0]
    """
    P = _pts(positions)
    w = [float(v) for v in weights]
    n = len(P)
    core = []
    for j in range(n):
        ang = []
        for k in range(n):
            if P[k] != P[j]:
                ang.append(math.atan2(P[k][1] - P[j][1], P[k][0] - P[j][0]) % math.pi)
        ang = sorted(set(ang))
        tests = list(ang)
        for a in range(len(ang)):
            nxt = ang[a + 1] if a + 1 < len(ang) else ang[0] + math.pi
            tests.append(0.5 * (ang[a] + nxt))
        if not tests:
            tests = [0.0]
        ok = True
        for th in tests:
            nx, ny = -math.sin(th), math.cos(th)
            pos = neg = 0.0
            for k in range(n):
                d = nx * (P[k][0] - P[j][0]) + ny * (P[k][1] - P[j][1])
                if d > 1e-12:
                    pos += w[k]
                elif d < -1e-12:
                    neg += w[k]
            if pos >= quota or neg >= quota:
                ok = False
                break
        if ok:
            core.append(j)
    return RichResult(payload={"core": core, "positions": [P[j] for j in core]})


def _hull(points):
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def spatial_heart(positions, weights, quota: float) -> RichResult:
    r"""The heart of a weighted voting game in the plane: the region bounded by the median lines.

    With a core party the heart is its position. Otherwise the heart is the
    region bounded by the median lines (Schofield 1999); it is returned as
    the convex hull of the pairwise intersections of median lines that fall
    inside the Pareto set (the hull of all positions), an outer polygon for
    the bounded region that is exact for three parties with no majority
    (the triangle of their positions).

    References
    ----------
    Schofield, N. (1999). The heart and the uncovered set. *Journal of
    Economics*, Supplement 8, 79-113.
    Schofield, N. and Sened, I. (2006). *Multiparty Democracy: Elections and
    Legislative Politics*. Cambridge University Press, ch. 3.

    Examples
    --------
    >>> spatial_heart([(0, 0), (4, 0), (0, 3)], [30, 35, 35], 51).vertices
    [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]
    """
    P = _pts(positions)
    core = weighted_game_core(P, weights, quota).core
    if core:
        return RichResult(payload={"vertices": [P[core[0]]], "core": True, "median_lines": []})
    lines = median_lines(P, weights, quota)
    pareto = _hull(P)

    def inside(q):
        m = len(pareto)
        if m < 3:
            return True
        return all(_side(q, pareto[i], pareto[(i + 1) % m]) >= -1e-9 for i in range(m))

    xs = []
    for (a, b), (c, d) in itertools.combinations(lines, 2):
        p1, p2, p3, p4 = P[a], P[b], P[c], P[d]
        den = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
        if abs(den) < 1e-15:
            continue
        t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / den
        q = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
        if inside(q):
            xs.append((round(q[0], 12) + 0.0, round(q[1], 12) + 0.0))
    return RichResult(payload={"vertices": _hull(xs), "core": False, "median_lines": lines})


def _minimal_winning(w, quota):
    n = len(w)
    out = []
    for r in range(1, n + 1):
        for S in itertools.combinations(range(n), r):
            tot = ssum(w[i] for i in S)
            if tot >= quota and all(tot - w[i] < quota for i in S):
                out.append(S)
    return out


def quota_solution(weights, quota: float) -> RichResult:
    r"""Quota (von Neumann-Morgenstern / Shapley) of a weighted majority game.

    The quota ``q`` gives each player a claim such that every minimal winning
    coalition ``S`` exactly exhausts the prize, ``sum_{i in S} q_i = 1``; the
    game is a quota game when this linear system is consistent (e.g. three
    players with simple majority: ``q = (1/2, 1/2, 1/2)``). Solved in least
    squares by the Moore-Penrose inverse (minimum norm when underdetermined), with the maximum residual reported.

    References
    ----------
    Shapley, L. S. (1953). Quota solutions of n-person games. In *Contributions
    to the Theory of Games II*, 343-359. Princeton University Press.
    von Neumann, J. and Morgenstern, O. (1944). *Theory of Games and Economic
    Behavior*. Princeton University Press, section 22.

    Examples
    --------
    >>> r = quota_solution([1, 1, 1], 2)
    >>> [round(v, 12) for v in r.quota], r.consistent
    ([0.5, 0.5, 0.5], True)
    """
    w = [float(v) for v in weights]
    n = len(w)
    mw = _minimal_winning(w, quota)
    A = [[1.0 if i in S else 0.0 for i in range(n)] for S in mw]
    m = len(A)
    AtA = [[ssum(A[k][i] * A[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
    At1 = [ssum(A[k][i] for k in range(m)) for i in range(n)]
    vals, vecs = jacobi(AtA)  # Moore-Penrose solution of A q = 1
    top = max(abs(v) for v in vals)
    q = [0.0] * n
    for c in range(n):
        if vals[c] > 1e-10 * top:
            proj = ssum(vecs[i][c] * At1[i] for i in range(n)) / vals[c]
            for i in range(n):
                q[i] += vecs[i][c] * proj
    res = max(abs(ssum(A[k][i] * q[i] for i in range(n)) - 1.0) for k in range(m))
    return RichResult(
        payload={"quota": q, "minimal_winning": [list(S) for S in mw], "max_residual": res, "consistent": res < 1e-9}
    )


def minimal_range_coalition(positions, weights, quota: float) -> RichResult:
    r"""Minimal-range (closed, ideologically compact) winning coalition (Leiserson 1968; de Swaan 1973).

    Among the minimal winning coalitions, the one whose members span the
    smallest ideological range ``max x - min x`` on the left-right dimension;
    ties go to fewer parties, then to smaller total weight, then to the
    lexicographically first.

    References
    ----------
    Leiserson, M. (1968). Factions and coalitions in one-party Japan.
    *American Political Science Review*, 62, 770-787.
    de Swaan, A. (1973). *Coalition Theories and Cabinet Formations*. Elsevier.

    Examples
    --------
    >>> minimal_range_coalition([-2, -1, 0, 1, 3], [20, 15, 10, 25, 30], 51).coalition
    [3, 4]
    """
    x = [float(v) for v in positions]
    w = [float(v) for v in weights]
    mw = _minimal_winning(w, quota)
    best = min(mw, key=lambda S: (max(x[i] for i in S) - min(x[i] for i in S), len(S), ssum(w[i] for i in S), S))
    return RichResult(
        payload={
            "coalition": list(best),
            "range": max(x[i] for i in best) - min(x[i] for i in best),
            "weight": ssum(w[i] for i in best),
        }
    )


def _pnorm(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def roemer_pune(
    a: float, b: float, mu: float, sigma: float, alpha_a: float, alpha_b: float, *, iters: int = 200
) -> RichResult:
    r"""Roemer's party-unanimity Nash equilibrium (PUNE) in one dimension with factional bargaining.

    Party A (ideal ``a``) and B (ideal ``b > a``) propose ``t`` and ``s``;
    the median voter is ``N(mu, sigma^2)`` so ``pi(t, s) = Phi(((t + s)/2 - mu)/sigma)``
    is A's win probability when ``t < s``. Each party's militants (quadratic
    utility around the party ideal) and opportunists (win probability)
    bargain; by Roemer's characterisation a PUNE is a Nash equilibrium of the
    weighted Nash products ``pi^{alpha} (EU - u(opponent policy))^{1 - alpha}``,
    which reduce to maximising ``log pi_A + (1 - alpha_A) log(u_A(t) - u_A(s))``
    (and symmetrically for B). Solved by best-response iteration with golden-section searches.

    References
    ----------
    Roemer, J. E. (1999). The democratic political economy of progressive
    income taxation. *Econometrica*, 67, 1-19.
    Roemer, J. E. (2001). *Political Competition: Theory and Applications*.
    Harvard University Press, ch. 8.

    Examples
    --------
    >>> r = roemer_pune(-1.0, 1.0, 0.0, 0.5, 0.5, 0.5)
    >>> abs(round(r.t + r.s, 6)), r.t < 0 < r.s
    (0.0, True)
    """

    def gold(f, lo, hi):
        gr = (math.sqrt(5) - 1) / 2
        x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
        f1, f2 = f(x1), f(x2)
        for _ in range(300):
            if f1 >= f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - gr * (hi - lo)
                f1 = f(x1)
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + gr * (hi - lo)
                f2 = f(x2)
            if hi - lo < 1e-13:
                break
        return 0.5 * (lo + hi)

    def obj_a(t, s):
        gain = (s - a) ** 2 - (t - a) ** 2
        p = _pnorm(((t + s) / 2 - mu) / sigma)
        return -math.inf if gain <= 0 or p <= 0 else math.log(p) + (1 - alpha_a) * math.log(gain)

    def obj_b(s, t):
        gain = (t - b) ** 2 - (s - b) ** 2
        p = 1 - _pnorm(((t + s) / 2 - mu) / sigma)
        return -math.inf if gain <= 0 or p <= 0 else math.log(p) + (1 - alpha_b) * math.log(gain)

    t, s = a, b
    for _ in range(iters):
        t_new = gold(lambda v, s=s: obj_a(v, s), a, s)
        s_new = gold(lambda v, t_new=t_new: obj_b(v, t_new), t_new, b)
        done = abs(t_new - t) < 1e-12 and abs(s_new - s) < 1e-12
        t, s = t_new, s_new
        if done:
            break
    return RichResult(payload={"t": t, "s": s, "win_probability_a": _pnorm(((t + s) / 2 - mu) / sigma)})


def cheatsheet() -> str:
    return (
        "median_lines / weighted_game_core / spatial_heart / quota_solution / minimal_range_coalition / roemer_pune "
        "-> coalition and party competition games."
    )
