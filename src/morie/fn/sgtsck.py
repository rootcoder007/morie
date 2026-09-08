# morie.fn -- function file (rootcoder007/morie)
"""Ng-Jordan-Weiss k-way spectral clustering."""

import math

from . import _s03core as core

from ._richresult import RichResult
from .sgtlap2 import _norm_laplacian

__all__ = ["sgt_spectral_clustering_k"]


def _kmeans_det(rows, k, iters=100):
    """Lloyd's algorithm from a deterministic furthest-point seeding.

    Random restarts are what k-means normally needs; they are also what
    would make the two language arms disagree.  Seeding from row 0 and
    then repeatedly taking the row furthest from every chosen centre is
    deterministic, and ties break to the lowest index in both arms.
    """
    n = len(rows)
    p = len(rows[0])
    centres = [list(rows[0])]
    while len(centres) < k:
        best = -1
        bestd = -1.0
        for i in range(n):
            dmin = -1.0
            for c in centres:
                s = 0.0
                for j in range(p):
                    t = rows[i][j] - c[j]
                    s += t * t
                if dmin < 0.0 or s < dmin:
                    dmin = s
            if dmin > bestd + 1e-12:
                bestd = dmin
                best = i
        centres.append(list(rows[best]))
    lab = [0] * n
    for _ in range(iters):
        changed = False
        for i in range(n):
            best = 0
            bestd = -1.0
            for m in range(k):
                s = 0.0
                for j in range(p):
                    t = rows[i][j] - centres[m][j]
                    s += t * t
                if bestd < 0.0 or s < bestd - 1e-12:
                    bestd = s
                    best = m
            if lab[i] != best:
                changed = True
            lab[i] = best
        for m in range(k):
            cnt = 0
            acc = [0.0] * p
            for i in range(n):
                if lab[i] == m:
                    cnt += 1
                    for j in range(p):
                        acc[j] += rows[i][j]
            if cnt > 0:
                centres[m] = [acc[j] / cnt for j in range(p)]
        if not changed:
            break
    return lab, centres


def sgt_spectral_clustering_k(A, k=2):
    """Cluster the nodes of ``A`` into ``k`` groups, Ng-Jordan-Weiss form.

    Take the ``k`` eigenvectors of ``L_sym`` with the smallest
    eigenvalues, normalise each ROW of that n-by-k matrix to unit length,
    and run k-means on the rows.  The row normalisation is the step that
    distinguishes this from plain spectral embedding: it projects nodes
    onto the unit sphere, where well-separated blocks become mutually
    orthogonal directions rather than clusters of different radius.

    Formula: ``L_sym = I - D^-1/2 A D^-1/2``; ``U`` = its ``k`` lowest
    eigenvectors; ``T_ij = U_ij / ||U_i||``; k-means on the rows of T.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Symmetric non-negative weight matrix, positive degrees.
    k : int, default 2
        Number of clusters, ``1 <= k <= n``.

    Returns
    -------
    RichResult
        ``labels`` (0-based, length n), ``eigvecs`` (the row-normalised
        n-by-k matrix T), ``eigvals``, ``k``, ``n``.

    References
    ----------
    Ng, A. Y., Jordan, M. I. & Weiss, Y. (2002).  On spectral
    clustering: analysis and an algorithm.  Advances in Neural
    Information Processing Systems 14, pages 849-856, MIT Press.
    """
    _, L, _, n = _norm_laplacian(A, "sgt_spectral_clustering_k")
    k = int(k)
    if k < 1 or k > n:
        raise ValueError("sgt_spectral_clustering_k: need 1 <= k <= n")
    vals, vecs = core.jacobi(L)
    T = []
    for i in range(n):
        row = [vecs[i][j] for j in range(k)]
        nrm = 0.0
        for v in row:
            nrm += v * v
        nrm = math.sqrt(nrm)
        T.append([v / nrm for v in row] if nrm > 0.0 else row)
    lab, _c = _kmeans_det(T, k)
    return RichResult(payload={
        "labels": lab, "eigvecs": T, "eigvals": [vals[j] for j in range(k)],
        "k": k, "n": n,
        "method": "Ng-Jordan-Weiss k-way spectral clustering"})


def cheatsheet():
    return "sgtsck: Ng-Jordan-Weiss spectral clustering"
