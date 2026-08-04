# morie.fn -- k02 batch (rootcoder007/morie)
"""Number of spanning trees by Kirchhoff's matrix-tree theorem.

Source consulted: Kirchhoff, G. (1847), Ueber die Auflosung der Gleichungen,
auf welche man bei der Untersuchung der linearen Vertheilung galvanischer
Strome gefuhrt wird, *Annalen der Physik* 148(12), 497-508.  With L = D - A
the graph Laplacian, every cofactor of L equals the number of spanning trees,
so deleting any one row and the matching column and taking the determinant
gives the count.  The complete graph K_n returns n^(n-2), Cayley's formula,
which is the canonical test.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sgt_spanning_tree_count"]


def sgt_spanning_tree_count(A, drop=0):
    """Spanning-tree count of an undirected weighted graph.

    Parameters
    ----------
    A : array-like
        Symmetric adjacency (or weight) matrix.
    drop : int, default 0
        Index of the row/column deleted to form the cofactor.

    Returns
    -------
    RichResult
        estimate (count, rounded to the nearest integer), cofactor,
        laplacian, degrees, n, method.
    """
    m = np.atleast_2d(np.asarray(A, dtype=float))
    n = m.shape[0]
    deg = np.sum(m, axis=1)
    lap = np.diag(deg) - m
    keep = [i for i in range(n) if i != int(drop)]
    sub = lap[np.ix_(keep, keep)]
    val = float(np.linalg.det(sub)) if n > 1 else 1.0
    return RichResult(
        payload={
            "estimate": float(np.rint(val)),
            "cofactor": val,
            "laplacian": lap.tolist(),
            "degrees": deg.tolist(),
            "n": int(n),
            "method": "Spanning-tree count by the matrix-tree theorem (Kirchhoff 1847)",
        }
    )


# CANONICAL TEST
# >>> K4 = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
# >>> assert sgt_spanning_tree_count(K4)["estimate"] == 16.0   # Cayley: 4^(4-2)
# >>> P3 = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
# >>> assert sgt_spanning_tree_count(P3)["estimate"] == 1.0    # a path IS its tree


def cheatsheet():
    return "sgtspn(A): number of spanning trees (matrix-tree theorem)."


sgtspanningtreecount = sgt_spanning_tree_count
