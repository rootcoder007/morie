# morie.fn -- tail3 batch (rootcoder007/morie)
"""Eccentricity centrality of a graph vertex.

Source consulted: Hage, P. & Harary, F. (1995). Eccentricity and centrality
in networks.  *Social Networks* 17, 57-63.  The eccentricity of a vertex is
its greatest geodesic distance to any other vertex,

    e(v) = max_u d(v, u)

and the eccentricity centrality (also called the centre or graph centrality)
is its reciprocal, C_E(v) = 1 / e(v).  The vertices of minimum eccentricity
form the centre of the graph; the minimum eccentricity is the radius and the
maximum is the diameter.
"""

from __future__ import annotations

from . import _array_core as np

from . import t3util as _t3
from ._richresult import RichResult

__all__ = ["eccentricity_centrality"]


def eccentricity_centrality(A, node=None):
    """Eccentricity centrality C_E(v) = 1 / max_u d(v, u).

    Parameters
    ----------
    A : array-like
        Square adjacency matrix; non-zero entries are edges.  Distances are
        unweighted geodesics obtained by breadth-first search.
    node : int, optional
        Vertex to report.  When omitted the mean centrality is reported.

    Returns
    -------
    RichResult
        estimate, ecc, centrality (all vertices), radius, diameter, centre,
        n, method.

    References
    ----------
    Hage & Harary (1995), Social Networks 17, 57-63.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = int(A.shape[0])
    eccs = []
    for v in range(n):
        d = _t3.bfsdist(A, v)
        eccs.append(float(np.max(d)))
    e = np.asarray(eccs, dtype=float)
    cent = np.asarray([1.0 / ev if ev > 0.0 else float("inf") for ev in eccs], dtype=float)
    radius = float(np.min(e))
    diameter = float(np.max(e))
    centre = [i for i in range(n) if float(e[i]) == radius]
    if node is None:
        est = float(np.mean(cent))
        ecc = float(np.mean(e))
    else:
        est = float(cent[int(node)])
        ecc = float(e[int(node)])
    return RichResult(
        payload={
            "estimate": est,
            "ecc": ecc,
            "centrality": cent,
            "eccentricity": e,
            "radius": radius,
            "diameter": diameter,
            "centre": centre,
            "n": n,
            "method": "Eccentricity centrality (Hage & Harary 1995)",
        }
    )


# CANONICAL TEST
# >>> # path on 4 vertices: eccentricities 3, 2, 2, 3
# >>> A = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
# >>> r = eccentricity_centrality(A, node=1)
# >>> assert abs(r["ecc"] - 2.0) < 1e-12
# >>> assert abs(r["radius"] - 2.0) < 1e-12 and abs(r["diameter"] - 3.0) < 1e-12


def cheatsheet():
    return "ecccen(A, node): eccentricity centrality, radius, diameter."
