"""Roll-call voting: simulation from a spatial model, exact optimal cutting lines and planes, and
classification error (APRE).

Poole, K. T. (2000). Nonparametric unfolding of binary choice data. Political Analysis 8,
211-237 (optimal classification). Poole, K. T. (2005). Spatial Models of Parliamentary Voting.
Cambridge University Press (APRE).
"""

import math
from itertools import combinations

from ._metaheur import Rand
from ._richresult import RichResult

__all__ = ["optimal_cutting_lines"]


def _errors(P, votes, normal, cut):
    e = 0
    for p, v in zip(P, votes):
        if v is None:
            continue
        s = sum(a * b for a, b in zip(p, normal)) - cut
        e += (1 if s > 0 else 0) != v
    return e


def _best_cut(proj, votes):
    """Cutpoint on a line minimising errors with yea on the high side (either orientation)."""
    pts = sorted((pv, v) for pv, v in zip(proj, votes) if v is not None)
    yea_total = sum(v for _, v in pts)
    best = (len(pts) + 1, 0.0, 1)
    # errors with yea above cut: nays above + yeas below
    ya, na = yea_total, len(pts) - yea_total
    _below_y, _below_n = 0, 0
    cands = (
        [pts[0][0] - 1.0]
        + [0.5 * (pts[i][0] + pts[i + 1][0]) for i in range(len(pts) - 1) if pts[i][0] != pts[i + 1][0]]
        + [pts[-1][0] + 1.0]
    )
    for c in cands:
        by = sum(1 for p, v in pts if p < c and v == 1)
        bn = sum(1 for p, v in pts if p < c and v == 0)
        e_up = (na - bn) + by
        e_dn = (ya - by) + bn
        if e_up < best[0]:
            best = (e_up, c, 1)
        if e_dn < best[0]:
            best = (e_dn, c, -1)
    return best


def optimal_cutting_lines(ideals, votes=None, n_votes=0, beta=5.0, seed=0):
    r"""Simulate roll calls and classify votes by optimal cutting lines.

    Simulation (``n_votes`` > 0): each roll call draws a normal vector uniformly on the sphere and
    a cutting point c ~ U(-0.5, 0.5); legislator i votes yea with probability
    Phi(beta (x_i . n - c)) (Philox draws). Classification: for each roll call the cutting line
    (plane) minimising classification errors is found exactly by enumerating candidate normal
    directions -- every direction in 1D, directions perpendicular to each pair of ideal points in
    2D (between consecutive critical angles the order of projections is fixed) and normals of
    planes through each triple of points in 3D -- with the best cutpoint along each (Poole 2000).
    APRE = sum_j (minority_j - errors_j) / sum_j minority_j (Poole 2005).

    Parameters
    ----------
    ideals : list of points (1 to 3 dimensions)
    votes : n x m matrix of 1 (yea), 0 (nay), None (missing), optional
    n_votes : int
        Roll calls to simulate when ``votes`` is not given.
    beta : float
    seed : int

    Returns
    -------
    RichResult
        Keys: votes, normals, cuts (the fitted cutting lines), errors (per roll call),
        classification (share correct), apre, and when simulated true_normals, true_cuts.

    References
    ----------
    Poole, K. T. (2000). Political Analysis 8, 211-237.
    Poole, K. T. (2005). Spatial Models of Parliamentary Voting.

    Examples
    --------
    >>> r = optimal_cutting_lines([[-1.0], [-0.5], [0.4], [1.0]], votes=[[0], [0], [1], [1]])
    >>> r["errors"], r["apre"]
    ([0], 1.0)
    """
    P = [[float(t)] for t in ideals] if isinstance(ideals[0], (int, float)) else [[float(t) for t in r] for r in ideals]
    n, d = len(P), len(P[0])
    if d > 3:
        raise ValueError("cutting lines are enumerated exactly for 1 to 3 dimensions")
    out = {}
    if votes is None:
        rnd = Rand(seed)
        V = [[0] * int(n_votes) for _ in range(n)]
        tn, tc = [], []
        for j in range(int(n_votes)):
            nv = [rnd.n() for _ in range(d)]
            s = math.sqrt(sum(v * v for v in nv))
            nv = [v / s for v in nv]
            c = rnd.u() - 0.5
            tn.append(nv)
            tc.append(c)
            for i in range(n):
                z = beta * (sum(a * b for a, b in zip(P[i], nv)) - c)
                V[i][j] = 1 if rnd.u() < 0.5 * math.erfc(-z / math.sqrt(2)) else 0
        out.update(true_normals=tn, true_cuts=tc)
    else:
        V = [[None if v is None else int(v) for v in r] for r in votes]
    m = len(V[0])
    if d == 1:
        dirs = [[1.0]]
    elif d == 2:
        dirs = []
        for i, k in combinations(range(n), 2):
            dx, dy = P[k][0] - P[i][0], P[k][1] - P[i][1]
            if dx or dy:
                a = math.atan2(dx, -dy)
                for eps in (-1e-7, 1e-7):
                    dirs.append([math.cos(a + eps), math.sin(a + eps)])
        dirs = dirs or [[1.0, 0.0]]
    else:
        dirs = []
        for i, k, l_ in combinations(range(n), 3):
            u = [P[k][q] - P[i][q] for q in range(3)]
            v = [P[l_][q] - P[i][q] for q in range(3)]
            nv = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
            s = math.sqrt(sum(t * t for t in nv))
            if s > 1e-12:
                base = [t / s for t in nv]
                dirs.append(base)
                for q in range(3):  # small rotations resolve points lying on the plane
                    for eps in (-1e-6, 1e-6):
                        w = list(base)
                        w[q] += eps
                        sw = math.sqrt(sum(t * t for t in w))
                        dirs.append([t / sw for t in w])
        dirs = dirs or [[1.0, 0.0, 0.0]]
    normals, cuts, errs, minority = [], [], [], []
    for j in range(m):
        col = [V[i][j] for i in range(n)]
        obs = [v for v in col if v is not None]
        minority.append(min(sum(obs), len(obs) - sum(obs)))
        best = None
        for dv in dirs:
            proj = [sum(a * b for a, b in zip(p, dv)) for p in P]
            e, c, sgn = _best_cut(proj, col)
            if best is None or e < best[0]:
                best = (e, [sgn * t for t in dv], sgn * c)
        errs.append(best[0])
        normals.append(best[1])
        cuts.append(best[2])
    tot = sum(1 for r in V for v in r if v is not None)
    out.update(
        votes=V,
        normals=normals,
        cuts=cuts,
        errors=errs,
        classification=1 - sum(errs) / tot,
        apre=(sum(minority) - sum(errs)) / sum(minority) if sum(minority) else 1.0,
    )
    return RichResult(title="Roll-call analysis", summary_lines=[("APRE", out["apre"])], payload=out)


def cheatsheet():
    return "svroll: roll-call simulation, exact optimal cutting lines/planes and APRE"
