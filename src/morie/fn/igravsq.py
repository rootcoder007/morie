# morie.fn -- function file (rootcoder007/morie)
"""Square gravity (symmetric OD matrix)."""

from __future__ import annotations

import math

from . import _gravity as gv
from ._containers import SpatialResult


def igravsq(flow_matrix, mass, dist_matrix):
    r"""Square gravity (symmetric OD matrix).

    Gravity on a square origin-destination matrix whose origins and
    destinations are the same n places: ``log F_ij = b0 + b1 log(M_i M_j) +
    b2 log d_ij`` by OLS over the off-diagonal pairs with positive flow --
    the symmetric specification with one common mass elasticity (Tinbergen
    1962; Fotheringham and O'Kelly 1989, ch. 2). Also returns the asymmetry
    index ``sum_{i<j} |F_ij - F_ji| / sum_{i<j} (F_ij + F_ji)`` (0 for a
    symmetric matrix).

    Parameters
    ----------
    flow_matrix : array-like, shape (n, n)
        Observed flows.
    mass : array-like, shape (n,)
        Positive masses of the places.
    dist_matrix : array-like, shape (n, n)
        Positive off-diagonal distances.

    Returns
    -------
    SpatialResult
        ``statistic`` is the distance elasticity ``b2``; ``extra`` has
        ``coefficients``, ``se``, ``r2``, ``asymmetry``, ``n_used``.

    References
    ----------
    Tinbergen, J. (1962). *Shaping the World Economy*. Twentieth Century Fund, New York.

    Fotheringham, A. S. and O'Kelly, M. E. (1989). *Spatial Interaction Models: Formulations and
    Applications*. Kluwer, Dordrecht.

    Examples
    --------
    >>> F = [[0, 12, 5, 3], [10, 0, 8, 2], [6, 9, 0, 7], [2, 3, 8, 0]]
    >>> D = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    >>> r = igravsq(F, [10, 8, 12, 5], D)
    >>> round(r.statistic, 10), round(r.extra["asymmetry"], 12)
    (-1.0070710642, 0.093333333333)
    """
    F, M, D = gv.mat(flow_matrix), gv.vec(mass), gv.mat(dist_matrix)
    n = len(M)
    X, y = [], []
    for i in range(n):
        for j in range(n):
            if i != j and F[i][j] > 0:
                X.append([1.0, math.log(M[i] * M[j]), math.log(D[i][j])])
                y.append(math.log(F[i][j]))
    r = gv.ols(X, y)
    num = gv.ssum(abs(F[i][j] - F[j][i]) for i in range(n) for j in range(i + 1, n))
    den = gv.ssum(F[i][j] + F[j][i] for i in range(n) for j in range(i + 1, n))
    return SpatialResult(
        name="igravsq",
        statistic=r["coefficients"][2],
        extra={
            "coefficients": r["coefficients"],
            "se": r["se"],
            "r2": r["r2"],
            "asymmetry": num / den,
            "n_used": len(y),
        },
    )


igravsq_fn = igravsq


def cheatsheet() -> str:
    return "igravsq(F, mass, D) -> symmetric square gravity OLS and asymmetry index"
