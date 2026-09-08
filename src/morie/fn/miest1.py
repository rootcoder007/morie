"""Mutual information by k-nearest-neighbour statistics (KSG 2004)."""

import math

from . import _sci_core as sc
from ._richresult import RichResult

__all__ = ["miest1", "mutual_information_knn"]


def _maxnorm(a, b):
    m = 0.0
    for u, v in zip(a, b):
        d = abs(u - v)
        if d > m:
            m = d
    return m


def miest1(X, Y, k=3, algorithm=1):
    """
    Mutual information from k-nearest-neighbour statistics.

    Kraskov, Stogbauer & Grassberger (2004), the two estimators they
    derive.  Points z_i = (x_i, y_i) live in the joint space under
    the maximum norm ||z - z'|| = max(||x - x'||, ||y - y'||) (their
    Eq. 6).  Let eps(i)/2 be the distance from z_i to its k-th
    neighbour in the joint space, and eps_x(i)/2, eps_y(i)/2 the
    same distance projected into the X and Y subspaces, so that
    eps(i) = max(eps_x(i), eps_y(i)).

    Algorithm 1 (their Eq. 8) counts n_x(i) = #{j != i :
    ||x_i - x_j|| < eps(i)/2} and likewise n_y(i), giving

        I1 = psi(k) - < psi(n_x + 1) + psi(n_y + 1) > + psi(N).

    Algorithm 2 (their Eq. 9) instead counts points with
    ||x_i - x_j|| <= eps_x(i)/2 and ||y_i - y_j|| <= eps_y(i)/2:

        I2 = psi(k) - 1/k - < psi(n_x) + psi(n_y) > + psi(N).

    Both are implemented.  The paper's own comparison is that for the
    same k, Eq. 8 has slightly SMALLER statistical error but LARGER
    systematic error, and that the two agree closely in general;
    algorithm 1 is therefore the default here, with algorithm 2
    available for the cases where the systematic error dominates
    (higher dimension, strong dependence).  Estimates are in nats
    (the paper uses natural logarithms throughout).

    Sources
    -------
    Kraskov, A., Stogbauer, H. & Grassberger, P. (2004).  Estimating
    mutual information.  *Physical Review E* 69, 066138, Eqs. 6, 8
    and 9 (local copy fetched-wave3/Kraskov_2004_MI_kNN.pdf).

    Parameters
    ----------
    X, Y : sequences (n,) or (n, d) matrices
        Paired samples.  Scalars are treated as 1-D.
    k : int
        Number of neighbours (k >= 1; the paper uses k = 1..6).
    algorithm : {1, 2}
        Which KSG estimator (Eq. 8 or Eq. 9).

    Returns
    -------
    RichResult
        Keys: mi (nats), mi_bits, k, algorithm, n.
    """
    def _rows(A):
        out = []
        for v in A:
            if isinstance(v, (list, tuple)):
                out.append([float(t) for t in v])
            else:
                out.append([float(v)])
        return out

    xs = _rows(X)
    ys = _rows(Y)
    n = len(xs)
    if n != len(ys) or n < 2:
        raise ValueError("X and Y must be paired, length >= 2")
    k = int(k)
    if k < 1 or k >= n:
        raise ValueError("need 1 <= k < n")
    if algorithm not in (1, 2):
        raise ValueError("algorithm must be 1 or 2")

    mi_sum = 0.0
    for i in range(n):
        dx = [_maxnorm(xs[i], xs[j]) for j in range(n)]
        dy = [_maxnorm(ys[i], ys[j]) for j in range(n)]
        dz = [max(dx[j], dy[j]) for j in range(n)]
        # k-th neighbour in the joint space, excluding self
        others = sorted(dz[j] for j in range(n) if j != i)
        eps = others[k - 1]                  # this is eps(i)/2
        if algorithm == 1:
            nx = sum(1 for j in range(n) if j != i and dx[j] < eps)
            ny = sum(1 for j in range(n) if j != i and dy[j] < eps)
            mi_sum += sc.digamma(nx + 1) + sc.digamma(ny + 1)
        else:
            # eps_x(i)/2 and eps_y(i)/2 are the half-widths of the
            # smallest rectangle holding all k joint neighbours
            # (their Figs. 1b, 1c), i.e. the LARGEST projected
            # distance over the k nearest neighbours -- not the k-th
            # neighbour's own projection.
            order = sorted((j for j in range(n) if j != i),
                           key=lambda j: (dz[j], j))[:k]
            ex = max(dx[j] for j in order)
            ey = max(dy[j] for j in order)
            nx = sum(1 for j in range(n) if j != i and dx[j] <= ex)
            ny = sum(1 for j in range(n) if j != i and dy[j] <= ey)
            mi_sum += sc.digamma(max(nx, 1)) + sc.digamma(max(ny, 1))
    mean_term = mi_sum / n
    if algorithm == 1:
        mi = sc.digamma(k) - mean_term + sc.digamma(n)
    else:
        mi = sc.digamma(k) - 1.0 / k - mean_term + sc.digamma(n)
    return RichResult(payload={
        "mi": mi,
        "mi_bits": mi / math.log(2.0),
        "k": k,
        "algorithm": algorithm,
        "n": n,
        "method": "KSG mutual information, Eq. %d (Kraskov 2004)"
                  % (8 if algorithm == 1 else 9),
    })


# long descriptive alias (stub-era name)
mutual_information_knn = miest1


def cheatsheet():
    return ("miest1: I1=psi(k)-<psi(nx+1)+psi(ny+1)>+psi(N); "
            "I2=psi(k)-1/k-<psi(nx)+psi(ny)>+psi(N)")

# public names resolved by fn/_lazy_map.json
mi_ksg = miest1
miksg = miest1
