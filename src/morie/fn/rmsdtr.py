"""Best-rotation RMSD between vector sets (Kabsch 1976)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["rmsdtr", "kabsch_rmsd"]


def rmsdtr(P, Q, weights=None):
    """
    Optimal-superposition RMSD by the Kabsch algorithm.

    Kabsch (1976): the orthogonal matrix U minimizing
    E = (1/2) sum_n w_n (U x_n - y_n)^2 (his Eq. 1; translations
    removed by shifting both centroids to the origin) satisfies
    U (S + L) = R with R = sum_n w_n y_n x_n' (Eqs. 7-9).  The
    direct solution diagonalizes R'R: with eigenpairs (mu_k, a_k)
    and b_k = R a_k / sqrt(mu_k), U = sum_k b_k a_k'; the smallest
    eigenvector's sign is flipped when needed so that det(U) = +1,
    which restricts the fit to proper rotations.  The reported RMSD
    is sqrt(sum_n w_n |U x_n - y_n|^2 / sum_n w_n) after optimal
    superposition.

    Sources
    -------
    Kabsch, W. (1976). A solution for the best rotation to relate
    two sets of vectors. *Acta Crystallographica*, A32, 922-923,
    Eqs. 1-9 (local copy fetched-wave3/A solution for the best
    rotation to relate two sets of vectors.pdf).

    Parameters
    ----------
    P, Q : sequences of length-d vectors
        Paired coordinate sets (x_n and y_n; d = 2 or 3 typical).
    weights : sequence of float, optional
        Non-negative pair weights w_n (default all 1).

    Returns
    -------
    RichResult
        Keys: estimate (RMSD), rotation (U as list of rows),
        translation (applied to centered P), det (of U, +1),
        centroids.
    """
    Pv = [[float(v) for v in r] for r in P]
    Qv = [[float(v) for v in r] for r in Q]
    n = len(Pv)
    if n < 3 or len(Qv) != n:
        raise ValueError("need >= 3 paired points")
    d = len(Pv[0])
    if any(len(r) != d for r in Pv + Qv):
        raise ValueError("all points must share one dimension")
    w = [1.0] * n if weights is None else [float(v) for v in weights]
    if len(w) != n or any(v < 0 for v in w) or sum(w) <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    sw = sum(w)
    cp = [sum(w[i] * Pv[i][k] for i in range(n)) / sw for k in range(d)]
    cq = [sum(w[i] * Qv[i][k] for i in range(n)) / sw for k in range(d)]
    X = [[Pv[i][k] - cp[k] for k in range(d)] for i in range(n)]
    Y = [[Qv[i][k] - cq[k] for k in range(d)] for i in range(n)]
    # R_ij = sum_n w_n y_ni x_nj  (Kabsch Eq. 7)
    Rm = [[sum(w[t] * Y[t][i] * X[t][j] for t in range(n))
           for j in range(d)] for i in range(d)]
    RtR = [[sum(Rm[k][i] * Rm[k][j] for k in range(d))
            for j in range(d)] for i in range(d)]
    mu, A = np.linalg.eigh(np.asarray(RtR))
    mu = [float(v) for v in mu]
    A = [[float(A[i][j]) for j in range(d)] for i in range(d)]
    # columns of A are eigenvectors, ascending eigenvalues
    order = sorted(range(d), key=lambda k: -mu[k])
    a_vecs = [[A[i][k] for i in range(d)] for k in order]
    mu_s = [max(mu[k], 0.0) for k in order]
    b_vecs = []
    for k in range(d):
        if mu_s[k] > 1e-24:
            b = [sum(Rm[i][j] * a_vecs[k][j] for j in range(d))
                 / math.sqrt(mu_s[k]) for i in range(d)]
        else:
            # degenerate direction: complete orthonormally
            b = [0.0] * d
        b_vecs.append(b)
    if all(abs(v) < 1e-15 for v in b_vecs[-1]):
        # build the last b as the cross/orthogonal completion (d = 3
        # cross product; d = 2 perpendicular)
        if d == 3:
            u, v = b_vecs[0], b_vecs[1]
            b_vecs[-1] = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2],
                          u[0]*v[1]-u[1]*v[0]]
        elif d == 2:
            b_vecs[-1] = [-b_vecs[0][1], b_vecs[0][0]]
    U = [[sum(b_vecs[k][i] * a_vecs[k][j] for k in range(d))
          for j in range(d)] for i in range(d)]

    def _det(m):
        if len(m) == 2:
            return m[0][0]*m[1][1] - m[0][1]*m[1][0]
        return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))

    if d in (2, 3) and _det(U) < 0:
        # flip the direction of the smallest-eigenvalue pair
        b_vecs[-1] = [-v for v in b_vecs[-1]]
        U = [[sum(b_vecs[k][i] * a_vecs[k][j] for k in range(d))
              for j in range(d)] for i in range(d)]
    sq = 0.0
    for t in range(n):
        ux = [sum(U[i][j] * X[t][j] for j in range(d)) for i in range(d)]
        sq += w[t] * sum((ux[i] - Y[t][i]) ** 2 for i in range(d))
    rmsd = math.sqrt(sq / sw)
    return RichResult(payload={
        "estimate": rmsd,
        "rotation": U,
        "det": _det(U) if d in (2, 3) else None,
        "centroids": {"P": cp, "Q": cq},
        "n": n,
        "method": "Kabsch (1976) optimal superposition RMSD",
    })


# long descriptive alias (stub-era name)
kabsch_rmsd = rmsdtr


def cheatsheet():
    return "rmsdtr: R = sum w y x'; U from R'R eigen; det +1; RMSD after fit"
