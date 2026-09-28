# morie.fn -- function file (rootcoder007/morie)
"""Formal spatial voting theory: weighted-distance majority comparison, win sets, weighted/supermajority/veto/
bicameral pivots, Pareto sets, probabilistic voting (logit, probit, Cauchy, uniform, Laplace errors) with valence
and best-response candidate equilibria, Hotelling price-location and Salop circle markets, Kalai-Smorodinsky and
Baron-Ferejohn bargaining, Riker-Brams vote trading and principal-component ideal points."""

from __future__ import annotations

import itertools
import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult
from ._rrng_core import pnorm

__all__ = [
    "majority_compare",
    "pivot_points",
    "pareto_set",
    "win_set",
    "probabilistic_vote",
    "candidate_equilibrium",
    "hotelling_price_location",
    "salop_circle",
    "kalai_smorodinsky",
    "baron_ferejohn",
    "vote_trading_riker_brams",
    "pca_ideal_points",
]


def _pts(a):
    return [tuple(float(v) for v in (r if isinstance(r, (list, tuple)) else [r])) for r in a]


def _W(A, d):
    if A is None:
        return [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
    if isinstance(A[0], (list, tuple)):
        return [[float(v) for v in r] for r in A]
    return [[float(A[i]) if i == j else 0.0 for j in range(d)] for i in range(d)]


def _d2(p, x, M):
    h = [a - b for a, b in zip(p, x)]
    return ssum(h[i] * M[i][j] * h[j] for i in range(len(h)) for j in range(len(h)))


def _votes(P, x, y, A, w):
    M = _W(A, len(P[0]))
    vx = vy = 0.0
    for p, wi in zip(P, w):
        dx, dy = _d2(p, x, M), _d2(p, y, M)
        if dx < dy:
            vx += wi
        elif dy < dx:
            vy += wi
    return vx, vy


def majority_compare(ideals, x, y, *, weights=None, quota: float = 0.5, A=None) -> RichResult:
    r"""Weighted votes for ``x`` against ``y`` under sincere spatial voting with weighted distance ``(p - x)' A (p - x)``.

    ``A`` is a salience vector (separable preferences) or a positive
    definite matrix (non-separable; Hinich and Munger 1997); ``x`` wins with
    more than ``quota`` of the total weight (0.5 simple, 2/3 qualified
    majority); indifferent voters abstain.

    References
    ----------
    Hinich, M. J. and Munger, M. C. (1997). *Analytical Politics*. Cambridge
    University Press.

    Examples
    --------
    >>> r = majority_compare([(0,), (2,), (3,)], (1.0,), (2.5,))
    >>> r.votes_x, r.votes_y, r.winner
    (1.0, 2.0, 'y')
    """
    P = _pts(ideals)
    w = [1.0] * len(P) if weights is None else [float(v) for v in weights]
    vx, vy = _votes(P, tuple(float(v) for v in x), tuple(float(v) for v in y), A, w)
    tot = ssum(w)
    win = "x" if vx > quota * tot else "y" if vy > quota * tot else "none"
    return RichResult(payload={"votes_x": vx, "votes_y": vy, "winner": win})


def _wquantile(v, w, q):
    order = sorted(range(len(v)), key=lambda i: (v[i], i))
    tot = ssum(w)
    acc = 0.0
    for i in order:
        acc += w[i]
        if acc >= q * tot - 1e-12:
            return v[i]
    return v[order[-1]]


def pivot_points(
    ideals, *, weights=None, supermajority: float | None = None, veto: float | None = None, chambers=None
) -> RichResult:
    r"""One-dimensional pivots: weighted median, supermajority and veto pivots, and bicameral cores.

    The weighted median is the smallest ideal point with cumulative weight
    at least half (Black 1948). ``supermajority`` ``q`` gives the ``1 - q``
    and ``q`` weighted quantiles (filibuster pivots), ``veto`` ``v`` the
    ``1 - v`` quantile (veto-override pivot); the gridlock interval spans the
    pivots (Krehbiel 1998). ``chambers`` (lists of member indices) gives each
    chamber median and the bicameral core between them (Hammond and Miller
    1987).

    References
    ----------
    Black, D. (1948). On the rationale of group decision-making. *Journal of
    Political Economy*, 56(1), 23-34.
    Krehbiel, K. (1998). *Pivotal Politics*. University of Chicago Press.
    Hammond, T. H. and Miller, G. J. (1987). The core of the constitution.
    *American Political Science Review*, 81(4), 1155-1174.

    Examples
    --------
    >>> r = pivot_points([1.0, 2.0, 3.0, 4.0, 5.0], supermajority=0.6)
    >>> r.median, r.gridlock
    (3.0, [2.0, 3.0])
    """
    x = [float(v) for v in ideals]
    w = [1.0] * len(x) if weights is None else [float(v) for v in weights]
    med = _wquantile(x, w, 0.5)
    out = {"median": med}
    lo = hi = med
    if supermajority is not None:
        a, b = _wquantile(x, w, 1 - supermajority), _wquantile(x, w, supermajority)
        out["supermajority_pivots"] = [a, b]
        lo, hi = min(lo, a), max(hi, b)
    if veto is not None:
        out["veto_pivot"] = _wquantile(x, w, 1 - veto)
        lo = min(lo, out["veto_pivot"])
    if chambers is not None:
        meds = [_wquantile([x[i] for i in c], [w[i] for i in c], 0.5) for c in chambers]
        out["chamber_medians"] = meds
        out["bicameral_core"] = [min(meds), max(meds)]
        lo, hi = min(lo, min(meds)), max(hi, max(meds))
    out["gridlock"] = [lo, hi]
    return RichResult(payload=out)


def _hull(P):
    P = sorted(set(P))
    if len(P) < 3:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def pareto_set(ideals, points=None) -> RichResult:
    r"""Pareto set of Euclidean spatial preferences, the convex hull of the ideal points (an interval in 1-D).

    Returns the hull and, for ``points``, whether each lies in it: the
    unanimity core, where no veto player can be made better off without
    another losing (Tsebelis 2002).

    References
    ----------
    Tsebelis, G. (2002). *Veto Players*. Princeton University Press.

    Examples
    --------
    >>> r = pareto_set([(0, 0), (2, 0), (0, 2), (0.5, 0.5)], [(0.5, 0.5), (2, 2)])
    >>> r.hull, r.inside
    ([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)], [True, False])
    """
    P = _pts(ideals)
    Q = _pts(points or [])
    if len(P[0]) == 1:
        lo, hi = min(p[0] for p in P), max(p[0] for p in P)
        return RichResult(payload={"hull": [(lo,), (hi,)], "inside": [lo <= q[0] <= hi for q in Q]})
    H = _hull(P)
    ins = [
        all(
            (H[(i + 1) % len(H)][0] - H[i][0]) * (q[1] - H[i][1])
            - (H[(i + 1) % len(H)][1] - H[i][1]) * (q[0] - H[i][0])
            >= -1e-12
            for i in range(len(H))
        )
        for q in Q
    ]
    return RichResult(payload={"hull": H, "inside": ins})


def win_set(ideals, status_quo, grid, *, weights=None, quota: float = 0.5, A=None) -> RichResult:
    r"""Majority win set of the status quo on ``grid``: points a weighted ``quota`` majority prefers to it.

    An empty win set means the status quo is a majority equilibrium (Plott
    1967); the share of grid points estimates the win set's area on a
    regular grid.

    References
    ----------
    Plott, C. R. (1967). A notion of equilibrium and its possibility under
    majority rule. *American Economic Review*, 57(4), 787-806.

    Examples
    --------
    >>> win_set([(0,), (1,), (2,)], (1.0,), [(0.5,), (1.5,)]).members
    []
    """
    P = _pts(ideals)
    G = _pts(grid)
    w = [1.0] * len(P) if weights is None else [float(v) for v in weights]
    sq = tuple(float(v) for v in status_quo)
    tot = ssum(w)
    flags = [_votes(P, g, sq, A, w)[0] > quota * tot for g in G]
    return RichResult(
        payload={"in_win_set": flags, "members": [g for g, f in zip(G, flags) if f], "share": sum(flags) / len(flags)}
    )


def _cdf(z, error):
    if error == "logit":
        return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
    if error == "probit":
        return float(pnorm(z))
    if error == "cauchy":
        return 0.5 + math.atan(z) / math.pi
    if error == "laplace":
        return 0.5 * math.exp(z) if z < 0 else 1 - 0.5 * math.exp(-z)
    if error == "uniform":  # utility-difference noise uniform on [-1, 1]
        return min(1.0, max(0.0, (z + 1) / 2))
    raise ValueError("error must be logit, probit, cauchy, laplace or uniform")


def probabilistic_vote(
    ideals, positions, *, valence=None, beta: float = 1.0, error: str = "logit", weights=None, A=None
) -> RichResult:
    r"""Probabilistic spatial voting: each voter's choice probabilities and expected vote shares.

    Utility ``U_ij = valence_j - beta d_ij^2`` (weighted squared distance)
    plus random errors. Two candidates: ``P(vote 1) = F(U_i1 - U_i2)`` with
    ``F`` the logistic, standard normal (probit), Cauchy, standard Laplace or
    uniform-on-[-1,1] cdf of the utility-difference noise. More candidates:
    multinomial logit ``exp(U_ij) / sum_k exp(U_ik)`` (``error="logit"``
    only). Expected vote shares weight voters by ``weights``; the probability
    that candidate 1 wins a majority of ``n`` independent voters uses the
    normal approximation to the Poisson-binomial count.

    References
    ----------
    Enelow, J. M. and Hinich, M. J. (1984). *The Spatial Theory of Voting*.
    Cambridge University Press.
    Lindbeck, A. and Weibull, J. W. (1987). Balanced-budget redistribution as
    the outcome of political competition. *Public Choice*, 52(3), 273-297.

    Examples
    --------
    >>> r = probabilistic_vote([(0.0,), (1.0,)], [(0.0,), (1.0,)], error="logit")
    >>> [round(v, 6) for v in r.shares]
    [0.5, 0.5]
    """
    P = _pts(ideals)
    X = _pts(positions)
    J = len(X)
    val = [0.0] * J if valence is None else [float(v) for v in valence]
    w = [1.0] * len(P) if weights is None else [float(v) for v in weights]
    M = _W(A, len(P[0]))
    probs = []
    for p in P:
        U = [val[j] - beta * _d2(p, X[j], M) for j in range(J)]
        if J == 2:
            q = _cdf(U[0] - U[1], error)
            probs.append([q, 1 - q])
        else:
            if error != "logit":
                raise ValueError("more than two candidates needs error='logit'")
            mx = max(U)
            e = [math.exp(u - mx) for u in U]
            s = ssum(e)
            probs.append([v / s for v in e])
    tw = ssum(w)
    shares = [ssum(wi * pr[j] for wi, pr in zip(w, probs)) / tw for j in range(J)]
    out = {"probabilities": probs, "shares": shares}
    if J == 2:
        mean = ssum(pr[0] for pr in probs)
        var = ssum(pr[0] * (1 - pr[0]) for pr in probs)
        n = len(P)
        out["win_probability"] = 1 - float(pnorm((n / 2 - mean) / math.sqrt(var))) if var > 0 else float(mean > n / 2)
    return RichResult(payload=out)


def candidate_equilibrium(
    ideals, grid, *, valence=(0.0, 0.0), beta: float = 1.0, error: str = "logit", start=None, maxit: int = 100, A=None
) -> RichResult:
    r"""Two-candidate equilibrium of vote-share maximisation by alternating best responses on a candidate ``grid``.

    Each candidate in turn moves to the grid point maximising its expected
    vote share (:func:`probabilistic_vote`) against the other's position
    (first grid point on ties) until neither moves: a pure equilibrium on the
    grid. With symmetric logit errors and no valence both converge to the
    weighted mean (the mean voter theorem; Lin, Enelow and Dorussen 1999);
    a valence advantage makes the weaker candidate diverge (Groseclose 2001).
    Returns the positions, their distance (convergence/divergence), shares and
    whether a fixed point was reached.

    References
    ----------
    Groseclose, T. (2001). A model of candidate location when one candidate
    has a valence advantage. *American Journal of Political Science*, 45(4),
    862-886.
    Lin, T.-M., Enelow, J. M. and Dorussen, H. (1999). Equilibrium in
    multicandidate probabilistic spatial voting. *Public Choice*, 98(1), 59-82.

    Examples
    --------
    >>> g = [(k / 10,) for k in range(11)]
    >>> r = candidate_equilibrium([(0.0,), (0.4,), (0.8,)], g, beta=0.5)
    >>> r.positions, r.converged
    ([(0.4,), (0.4,)], True)
    """
    G = _pts(grid)
    pos = [G[0], G[-1]] if start is None else [tuple(float(v) for v in s) for s in start]
    ok = False
    for _ in range(maxit):
        moved = False
        for c in (0, 1):
            other = pos[1 - c]
            best, arg = -math.inf, pos[c]
            for g in G:
                pp = [g, other] if c == 0 else [other, g]
                sh = probabilistic_vote(ideals, pp, valence=list(valence), beta=beta, error=error, A=A)["shares"][c]
                if sh > best + 1e-12:
                    best, arg = sh, g
            if arg != pos[c]:
                pos[c] = arg
                moved = True
        if not moved:
            ok = True
            break
    sh = probabilistic_vote(ideals, pos, valence=list(valence), beta=beta, error=error, A=A)["shares"]
    return RichResult(payload={"positions": pos, "distance": math.dist(pos[0], pos[1]), "shares": sh, "converged": ok})


def hotelling_price_location(a: float, b: float, *, t: float = 1.0, c: float = 0.0) -> RichResult:
    r"""Hotelling duopoly on ``[0, 1]`` with quadratic transport costs (d'Aspremont, Gabszewicz and Thisse 1979).

    Firm 1 at ``a`` from the left, firm 2 at ``b`` from the right (``a + b <
    1``). Equilibrium prices ``p1 = c + t (1 - a - b)(1 + (a - b)/3)``, ``p2 =
    c + t (1 - a - b)(1 + (b - a)/3)``; the indifferent consumer is at ``a +
    (1 - a - b)(1 + (a - b)/3)/2``; profits are margin times demand.
    Firms gain by moving apart, so the location equilibrium is maximal
    differentiation (``a = b = 0``).

    References
    ----------
    d'Aspremont, C., Gabszewicz, J. J. and Thisse, J.-F. (1979). On
    Hotelling's "Stability in competition". *Econometrica*, 47(5), 1145-1150.

    Examples
    --------
    >>> r = hotelling_price_location(0.0, 0.0, t=1.0)
    >>> r.prices, r.demand, r.profits
    ([1.0, 1.0], [0.5, 0.5], [0.5, 0.5])
    """
    L = 1 - a - b
    p1 = c + t * L * (1 + (a - b) / 3)
    p2 = c + t * L * (1 + (b - a) / 3)
    xhat = a + L / 2 + (p2 - p1) / (2 * t * L)
    d1, d2 = xhat, 1 - xhat
    return RichResult(
        payload={"prices": [p1, p2], "demand": [d1, d2], "profits": [(p1 - c) * d1, (p2 - c) * d2], "indifferent": xhat}
    )


def salop_circle(
    n: int | None = None, *, t: float = 1.0, c: float = 0.0, fixed_cost: float | None = None, length: float = 1.0
) -> RichResult:
    r"""Salop (1979) circular city with linear transport cost ``t`` and ``n`` equally spaced firms.

    Symmetric equilibrium price ``c + t L / n``, profit ``t L^2 / n^2 - F``;
    with free entry and fixed cost ``F`` the number of firms is ``sqrt(t
    L^2 / F)`` (L the circumference).

    References
    ----------
    Salop, S. C. (1979). Monopolistic competition with outside goods. *Bell
    Journal of Economics*, 10(1), 141-156.

    Examples
    --------
    >>> r = salop_circle(4, t=2.0, fixed_cost=0.05)
    >>> r.price, round(r.profit, 6), round(r.free_entry_firms, 6)
    (0.5, 0.075, 6.324555)
    """
    out = {}
    if fixed_cost is not None:
        out["free_entry_firms"] = math.sqrt(t * length * length / fixed_cost)
    if n is not None:
        out["price"] = c + t * length / n
        out["profit"] = t * length * length / (n * n) - (fixed_cost or 0.0)
    return RichResult(payload=out)


def kalai_smorodinsky(frontier, disagreement=(0.0, 0.0)) -> RichResult:
    r"""Nash and Kalai-Smorodinsky bargaining solutions on a piecewise-linear Pareto frontier of utility pairs.

    ``frontier`` lists points ``(u1, u2)`` of a concave frontier in order of
    increasing ``u1``. The Nash solution maximises ``(u1 - d1)(u2 - d2)``
    (exactly, on each segment); the Kalai-Smorodinsky solution is the
    frontier point on the ray from ``d`` to the ideal point ``(max u1, max
    u2)`` (Kalai and Smorodinsky 1975).

    References
    ----------
    Nash, J. F. (1950). The bargaining problem. *Econometrica*, 18(2), 155-162.
    Kalai, E. and Smorodinsky, M. (1975). Other solutions to Nash's bargaining
    problem. *Econometrica*, 43(3), 513-518.

    Examples
    --------
    >>> r = kalai_smorodinsky([(0.0, 1.0), (1.0, 0.0)])
    >>> r.nash, r.kalai_smorodinsky
    ([0.5, 0.5], [0.5, 0.5])
    """
    F = [(float(a), float(b)) for a, b in frontier]
    d1, d2 = (float(v) for v in disagreement)
    best, nash = -math.inf, None
    for (x1, y1), (x2, y2) in zip(F, F[1:]):
        # product along the segment: (x1 + s dx - d1)(y1 + s dy - d2), s in [0, 1]
        dx, dy = x2 - x1, y2 - y1
        cand = [0.0, 1.0]
        if dx * dy != 0:
            s = -((x1 - d1) * dy + (y1 - d2) * dx) / (2 * dx * dy)
            if 0 <= s <= 1:
                cand.append(s)
        for s in cand:
            u1, u2 = x1 + s * dx, y1 + s * dy
            if u1 >= d1 and u2 >= d2 and (u1 - d1) * (u2 - d2) > best:
                best, nash = (u1 - d1) * (u2 - d2), [u1, u2]
    I1, I2 = max(p[0] for p in F), max(p[1] for p in F)
    ks = None
    for (x1, y1), (x2, y2) in zip(F, F[1:]):
        # intersect the segment with the ray d + t (I - d)
        ax, ay = I1 - d1, I2 - d2
        den = (x2 - x1) * ay - (y2 - y1) * ax
        if abs(den) < 1e-15:
            continue
        s = ((d1 - x1) * ay - (d2 - y1) * ax) / den
        if -1e-12 <= s <= 1 + 1e-12:
            ks = [x1 + s * (x2 - x1), y1 + s * (y2 - y1)]
            break
    return RichResult(payload={"nash": nash, "nash_product": best, "kalai_smorodinsky": ks, "ideal": [I1, I2]})


def baron_ferejohn(n: int, delta: float = 1.0, *, quota: int | None = None) -> RichResult:
    r"""Stationary equilibrium of Baron and Ferejohn (1989) closed-rule legislative bargaining over a unit dollar.

    With ``n`` members, a random recognised proposer and ``quota`` votes
    needed (default the simple majority ``(n + 1) // 2`` for odd ``n``), the
    proposer offers the continuation value ``delta / n`` to ``quota - 1``
    members and keeps ``1 - delta (quota - 1) / n``; proposals pass at once.

    References
    ----------
    Baron, D. P. and Ferejohn, J. A. (1989). Bargaining in legislatures.
    *American Political Science Review*, 83(4), 1181-1206.

    Examples
    --------
    >>> r = baron_ferejohn(5, 1.0)
    >>> r.proposer_share, r.coalition_share
    (0.6, 0.2)
    """
    q = (n + 1) // 2 if quota is None else int(quota)
    off = delta / n
    return RichResult(
        payload={
            "proposer_share": 1 - off * (q - 1),
            "coalition_share": off,
            "coalition_size": q,
            "continuation_value": 1 / n,
        }
    )


def vote_trading_riker_brams(valuations) -> RichResult:
    r"""Riker and Brams (1973) vote trading over binary issues decided by simple majority.

    ``valuations[i][k]`` is voter ``i``'s signed intensity for passing issue
    ``k`` (positive: wants it). Sincere outcomes pass issues with a
    majority in favour. A trade pairs voter ``i``, who cares more about issue ``a`` than
    ``b``, with voter ``j``, who cares more about ``b`` than ``a``: ``i`` votes
    ``j``'s way on ``b`` and ``j`` votes ``i``'s way on ``a``; it is made when it
    changes the outcome and both gain. Trades are made greedily (largest
    joint gain first) until none remains. Reports outcomes and total welfare before and after:
    the paradox of vote trading is that welfare can fall.

    References
    ----------
    Riker, W. H. and Brams, S. J. (1973). The paradox of vote trading.
    *American Political Science Review*, 67(4), 1235-1247.

    Examples
    --------
    >>> V = [[3, -1], [-1, 3], [-3, -3]]
    >>> r = vote_trading_riker_brams(V)
    >>> r.sincere, r.after_trade, r.welfare_before, r.welfare_after
    ([False, False], [True, True], 0.0, -2.0)
    """
    V = [[float(v) for v in r] for r in valuations]
    n, K = len(V), len(V[0])
    votes = [[v > 0 for v in r] for r in V]

    def outcome(vt):
        return [sum(1 for i in range(n) if vt[i][k]) > n / 2 for k in range(K)]

    def welfare(out):
        return ssum(V[i][k] for i in range(n) for k in range(K) if out[k])

    sincere = outcome(votes)
    trades = []
    while True:
        cur = outcome(votes)
        best = None
        for i, j in itertools.combinations(range(n), 2):
            for a, b in itertools.permutations(range(K), 2):
                # i cares more about a, j more about b: i votes j's way on b, j votes i's way on a
                if not (abs(V[i][a]) > abs(V[i][b]) and abs(V[j][b]) > abs(V[j][a])):
                    continue
                want_b, want_a = V[j][b] > 0, V[i][a] > 0
                if votes[i][b] == want_b or votes[j][a] == want_a:
                    continue
                trial = [row[:] for row in votes]
                trial[i][b] = want_b
                trial[j][a] = want_a
                new = outcome(trial)
                if new == cur:
                    continue
                gi = ssum(V[i][k] * (int(new[k]) - int(cur[k])) for k in range(K))
                gj = ssum(V[j][k] * (int(new[k]) - int(cur[k])) for k in range(K))
                if gi > 0 and gj > 0 and (best is None or gi + gj > best[0] + 1e-12):
                    best = (gi + gj, i, j, a, b, trial)
        if best is None:
            break
        _, i, j, a, b, votes = best
        trades.append((i, j, a, b))
    after = outcome(votes)
    return RichResult(
        payload={
            "sincere": sincere,
            "after_trade": after,
            "trades": trades,
            "welfare_before": welfare(sincere),
            "welfare_after": welfare(after),
        }
    )


def pca_ideal_points(votes, *, dims: int = 1) -> RichResult:
    r"""Principal-component ideal points from a legislator x roll-call matrix (1 yea, 0 nay, ``nan`` missing).

    Missing votes are replaced by the roll-call mean, the matrix is
    column-centred and the leading ``dims`` left singular vectors scaled by
    their singular values give the scores (the linear-probability / factor
    approach of Heckman and Snyder 1997); signs are fixed so the first
    legislator's score is non-positive on every dimension. Also returns the
    share of variance per dimension.

    References
    ----------
    Heckman, J. J. and Snyder, J. M. (1997). Linear probability models of the
    demand for attributes with an empirical application to estimating the
    preferences of legislators. *RAND Journal of Economics*, 28, S142-S189.

    Examples
    --------
    >>> r = pca_ideal_points([[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
    >>> r.scores[0][0] <= 0 < r.scores[2][0]
    True
    """
    M = [[float(v) if v is not None else float("nan") for v in r] for r in votes]
    n, p = len(M), len(M[0])
    for j in range(p):
        col = [r[j] for r in M if r[j] == r[j]]
        m = ssum(col) / len(col)
        for r in M:
            if r[j] != r[j]:
                r[j] = m
    means = [ssum(r[j] for r in M) / n for j in range(p)]
    X = [[r[j] - means[j] for j in range(p)] for r in M]
    C = [[ssum(X[i][a] * X[k][a] for a in range(p)) for k in range(n)] for i in range(n)]
    vals, vecs = np.linalg.eigh(np.asarray(C, dtype=float))
    vals = [float(v) for v in vals.tolist()]
    V = [[float(v) for v in r] for r in vecs.tolist()]
    order = sorted(range(n), key=lambda k: -vals[k])[:dims]
    scores = []
    for k in order:
        s = math.sqrt(max(vals[k], 0.0))
        col = [V[i][k] * s for i in range(n)]
        if col[0] > 0:
            col = [-v for v in col]
        scores.append(col)
    tot = ssum(v for v in vals if v > 0)
    return RichResult(
        payload={
            "scores": [[scores[d][i] for d in range(len(order))] for i in range(n)],
            "variance_share": [max(vals[k], 0.0) / tot for k in order],
        }
    )


def cheatsheet() -> str:
    return "majority_compare / pivot_points / probabilistic_vote / candidate_equilibrium / baron_ferejohn."
