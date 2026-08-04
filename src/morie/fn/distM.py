# morie.fn -- function file (rootcoder007/morie)
"""DistMult knowledge-graph score."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["distmult"]


def distmult(triples, dim, E=None, R=None, seed=1):
    """Bilinear triple score with a diagonal relation matrix.

    Restricting the relation matrix to its diagonal drops the parameter
    count from ``d^2`` to ``d`` and makes the score a plain three-way
    inner product.  The cost is exactly the symmetry it cannot escape:
    ``score(h, r, t) = score(t, r, h)`` always, so DistMult cannot
    represent an antisymmetric relation at all.  That limitation is what
    ComplEx was written to remove.

    Formula: ``score = <h, r, t> = sum_k h_k r_k t_k``.

    Parameters
    ----------
    triples : array-like, shape (m, 3)
        Rows ``[head, relation, tail]``, zero-based.
    dim : int
        Embedding dimension.
    E : array-like, optional
        Entity embeddings.
    R : array-like, optional
        Relation embeddings.
    seed : int, default 1
        Seed for the shared generator when embeddings are omitted.

    Returns
    -------
    RichResult
        ``estimate`` (mean score), ``scores``, ``symmetric_gap`` (the
        largest difference between a triple and its reverse, which is
        zero by construction), ``m``, ``dim``.

    References
    ----------
    Yang, B., Yih, W., He, X., Gao, J. & Deng, L. (2015).  Embedding
    entities and relations for learning and inference in knowledge
    bases.  ICLR 2015, equation (3).
    """
    T = [[int(v) for v in row] for row in C.mat(triples)]
    d = int(dim)
    ne = max(max(r[0], r[2]) for r in T) + 1
    nr = max(r[1] for r in T) + 1
    g = C.Lcg(seed)
    Em = C.mat(E) if E is not None else [[g.norm() for _ in range(d)] for _ in range(ne)]
    Rm = C.mat(R) if R is not None else [[g.norm() for _ in range(d)] for _ in range(nr)]
    sc, gap = [], 0.0
    for h, r, t in T:
        s = sum(Em[h][k] * Rm[r][k] * Em[t][k] for k in range(d))
        rev = sum(Em[t][k] * Rm[r][k] * Em[h][k] for k in range(d))
        sc.append(s)
        if abs(s - rev) > gap:
            gap = abs(s - rev)
    return RichResult(payload={
        "estimate": sum(sc) / len(sc), "scores": sc, "symmetric_gap": gap,
        "m": len(sc), "dim": d, "method": "DistMult triple score"})


def cheatsheet():
    return "distM: DistMult knowledge-graph score."
