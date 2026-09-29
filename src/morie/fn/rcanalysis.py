# morie.fn -- function file (rootcoder007/morie)
"""Roll-call analysis: participation, Rice cohesion, the Hix-Noury-Roland agreement index and
pairwise agreement; Snyder-Groseclose party influence on close votes; and one-dimensional
logistic (Luce choice) ideal-point estimation by penalised joint maximum likelihood."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import solve, ssum
from ._richresult import RichResult

__all__ = ["rollcall_cohesion", "party_influence", "logit_ideal_points"]


def _ok(v):
    return v is not None and v == v


def rollcall_cohesion(votes, party) -> RichResult:
    r"""Participation, party cohesion and agreement in a roll-call matrix.

    ``votes[i][j]`` is 1 (yea), 0 (nay) or ``None``/NaN (absent or
    abstaining); ``party[i]`` labels legislators. Per party and vote: Rice
    index ``|Y - N| / (Y + N)`` and the Hix-Noury-Roland agreement index
    ``(max(Y, N, A) - (Y + N + A - max(Y, N, A)) / 2) / (Y + N + A)`` (absent
    counted as ``A``); per legislator the participation rate; and the
    pairwise agreement (share of jointly cast votes that coincide).

    References
    ----------
    Rice, S. A. (1925). The behavior of legislative groups: a method of
    measurement. Political Science Quarterly 40, 60-72. Hix, S., Noury, A.
    and Roland, G. (2005). Power to the parties: cohesion and competition in
    the European Parliament. British J. Political Science 35, 209-234.

    Examples
    --------
    >>> r = rollcall_cohesion([[1, 1], [1, 0], [0, 0]], ["a", "a", "b"])
    >>> r.rice["a"], r.participation
    ([1.0, 0.0], [1.0, 1.0, 1.0])
    """
    n, m = len(votes), len(votes[0])
    parties = sorted(set(party), key=str)
    rice, agree = {}, {}
    for p in parties:
        rows = [i for i in range(n) if party[i] == p]
        rl, al = [], []
        for j in range(m):
            y = sum(1 for i in rows if _ok(votes[i][j]) and votes[i][j] == 1)
            no = sum(1 for i in rows if _ok(votes[i][j]) and votes[i][j] == 0)
            a = len(rows) - y - no
            rl.append(abs(y - no) / (y + no) if y + no else float("nan"))
            mx = max(y, no, a)
            al.append((mx - (y + no + a - mx) / 2) / (y + no + a))
        rice[p] = rl
        agree[p] = al
    part = [sum(1 for j in range(m) if _ok(votes[i][j])) / m for i in range(n)]
    pair = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            both = [j for j in range(m) if _ok(votes[i][j]) and _ok(votes[k][j])]
            pair[i][k] = sum(1 for j in both if votes[i][j] == votes[k][j]) / len(both) if both else float("nan")
    return RichResult(
        payload={"rice": rice, "agreement": agree, "participation": part, "pairwise": pair, "parties": parties}
    )


def _pc1(M):
    # first principal component scores of the columns-centred matrix (legislators x votes)
    n, m = len(M), len(M[0])
    mu = [ssum(M[i][j] for i in range(n)) / n for j in range(m)]
    C = [[M[i][j] - mu[j] for j in range(m)] for i in range(n)]
    G = [[ssum(C[a][j] * C[b][j] for j in range(m)) for b in range(n)] for a in range(n)]
    w, V = np.linalg.eigh(np.asarray(G, dtype=float))
    k = max(range(n), key=lambda i: float(w[i]))
    v = [float(V[r][k]) for r in range(n)]
    big = max(range(n), key=lambda r: abs(v[r]))
    return [-a for a in v] if v[big] < 0 else v


def party_influence(votes, party_dummy, *, lopsided: float = 0.65) -> RichResult:
    r"""Snyder-Groseclose party influence on close roll calls.

    Votes whose winning side has at least ``lopsided`` of those voting are
    lopsided; ideal points are the first principal component of the
    lopsided-vote matrix (absences filled with the vote mean), oriented to
    correlate positively with ``party_dummy``. For each remaining close vote
    the linear probability model ``y = a + b x + g party`` is fitted by least
    squares over legislators voting; ``g`` measures party influence.

    References
    ----------
    Snyder, J. M. and Groseclose, T. (2000). Estimating party influence in
    congressional roll-call voting. AJPS 44, 193-211.

    Examples
    --------
    >>> V = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 1], [0, 0, 0, 1], [1, 1, 1, 0], [0, 0, 1, 1]]
    >>> r = party_influence(V, [1, 1, 1, 0, 1, 0])
    >>> len(r.close_votes) + len(r.lopsided_votes)
    4
    """
    n, m = len(votes), len(votes[0])
    lop, close = [], []
    for j in range(m):
        cast = [votes[i][j] for i in range(n) if _ok(votes[i][j])]
        yes = sum(cast) / len(cast)
        (lop if max(yes, 1 - yes) >= lopsided else close).append(j)
    M = []
    for i in range(n):
        row = []
        for j in lop:
            if _ok(votes[i][j]):
                row.append(float(votes[i][j]))
            else:
                cast = [votes[q][j] for q in range(n) if _ok(votes[q][j])]
                row.append(sum(cast) / len(cast))
        M.append(row)
    x = _pc1(M) if lop else [0.0] * n
    pd = [float(v) for v in party_dummy]
    mx, mp = ssum(x) / n, ssum(pd) / n
    if ssum((a - mx) * (b - mp) for a, b in zip(x, pd)) < 0:
        x = [-a for a in x]
    gam = []
    for j in close:
        rows = [i for i in range(n) if _ok(votes[i][j])]
        X = [[1.0, x[i], pd[i]] for i in rows]
        G = [[ssum(r[a] * r[b] for r in X) for b in range(3)] for a in range(3)]
        h = [ssum(r[a] * float(votes[i][j]) for r, i in zip(X, rows)) for a in range(3)]
        gam.append(solve(G, h)[2])
    return RichResult(
        payload={
            "ideal_points": x,
            "party_effect": gam,
            "mean_party_effect": ssum(gam) / len(gam) if gam else float("nan"),
            "close_votes": close,
            "lopsided_votes": lop,
        }
    )


def _sig(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def logit_ideal_points(votes, *, prior_sd: float = 5.0, max_iter: int = 500, tol: float = 1e-10) -> RichResult:
    r"""One-dimensional logistic (Luce choice / random utility) ideal points by penalised joint ML.

    ``P(y_ij = 1) = logistic(alpha_j + beta_j x_i)``; bill parameters have
    ``N(0, prior_sd^2)`` penalties and ideal points ``N(0, 1)``. Alternating
    Newton updates (all bills, then all legislators) are followed by the
    identification ``mean(x) = 0``, ``sd(x) = 1`` (bill parameters rescaled
    to keep the linear predictor) until the largest change is below ``tol``.
    Missing votes (``None``/NaN) are skipped; ``x`` starts from the
    standardised mean-centred vote totals and keeps that orientation.

    References
    ----------
    Clinton, J., Jackman, S. and Rivers, D. (2004). The statistical analysis
    of roll call data. APSR 98, 355-370. Luce, R. D. (1959). Individual
    Choice Behavior. Wiley.

    Examples
    --------
    >>> V = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1]]
    >>> r = logit_ideal_points(V)
    >>> round(sum(r.ideal_points), 10) + 0.0
    0.0
    """
    n, m = len(votes), len(votes[0])
    Y = [[float(v) if _ok(v) else None for v in row] for row in votes]
    cm = [
        ssum(Y[i][j] for i in range(n) if Y[i][j] is not None) / max(1, sum(1 for i in range(n) if Y[i][j] is not None))
        for j in range(m)
    ]
    s0 = [ssum(Y[i][j] - cm[j] for j in range(m) if Y[i][j] is not None) for i in range(n)]
    mu = ssum(s0) / n
    sd = math.sqrt(ssum((v - mu) ** 2 for v in s0) / n) or 1.0
    x = [(v - mu) / sd for v in s0]
    ref = list(x)
    a = [0.0] * m
    b = [1.0] * m
    it = 0
    for _ in range(max_iter):
        it += 1
        old = x + a + b
        for j in range(m):
            rows = [i for i in range(n) if Y[i][j] is not None]
            for _k in range(3):
                g = [-a[j] / prior_sd**2, -b[j] / prior_sd**2]
                H = [[-1 / prior_sd**2, 0.0], [0.0, -1 / prior_sd**2]]
                for i in rows:
                    p = _sig(a[j] + b[j] * x[i])
                    r = Y[i][j] - p
                    w = p * (1 - p)
                    g[0] += r
                    g[1] += r * x[i]
                    H[0][0] -= w
                    H[0][1] -= w * x[i]
                    H[1][1] -= w * x[i] * x[i]
                H[1][0] = H[0][1]
                d = solve(H, g)
                a[j] -= d[0]
                b[j] -= d[1]
        for i in range(n):
            cols = [j for j in range(m) if Y[i][j] is not None]
            for _k in range(3):
                g, h = -x[i], -1.0
                for j in cols:
                    p = _sig(a[j] + b[j] * x[i])
                    g += (Y[i][j] - p) * b[j]
                    h -= p * (1 - p) * b[j] * b[j]
                x[i] -= g / h
        mx = ssum(x) / n
        sx = math.sqrt(ssum((v - mx) ** 2 for v in x) / n)
        if ssum((u - mx) * v for u, v in zip(x, ref)) < 0:
            sx = -sx
        x = [(v - mx) / sx for v in x]
        a = [a[j] + b[j] * mx for j in range(m)]
        b = [b[j] * sx for j in range(m)]
        if max(abs(u - v) for u, v in zip(old, x + a + b)) <= tol:
            break
    return RichResult(payload={"ideal_points": x, "alpha": a, "beta": b, "iterations": it})


def cheatsheet() -> str:
    return "rollcall_cohesion / party_influence / logit_ideal_points -> roll-call cohesion, party influence and ideal points."
