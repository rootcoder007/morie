# morie.fn -- function file (rootcoder007/morie)
"""Spectral graph indices for spatial networks: the Estrada index and communicability, Newman's
modularity matrix and modularity, the Perron-Frobenius eigenpair, the (general) Randic index and
label-propagation communities."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = [
    "estrada_index",
    "communicability",
    "modularity_matrix",
    "perron_frobenius",
    "randic_index",
    "label_propagation",
]


def _adj(A):
    M = [[float(v) for v in row] for row in (A.tolist() if hasattr(A, "tolist") else A)]
    n = len(M)
    if any(len(r) != n for r in M):
        raise ValueError("adjacency must be square")
    return M, n


def _eig(M):
    w, V = np.linalg.eigh(np.asarray(M, dtype=float))
    return [float(v) for v in w.tolist()], [[float(v) for v in r] for r in V.tolist()]


def estrada_index(A) -> float:
    r"""Estrada index ``EE = sum_i exp(lambda_i) = tr exp(A)`` of a symmetric adjacency matrix (Estrada 2000).

    References
    ----------
    Estrada, E. (2000). Characterization of 3D molecular structure.
    *Chemical Physics Letters*, 319(5-6), 713-718.

    Examples
    --------
    >>> round(estrada_index([[0, 1], [1, 0]]), 12) == round(math.exp(1) + math.exp(-1), 12)
    True
    """
    M, _ = _adj(A)
    w, _ = _eig(M)
    return ssum(math.exp(v) for v in w)


def communicability(A) -> RichResult:
    r"""Communicability matrix ``G = exp(A)`` (Estrada and Hatano 2008): weighted count of all walks, longer ones damped by ``1/k!``.

    Computed from the eigen-decomposition of the symmetric adjacency matrix
    ``A = Q diag(lambda) Q'``, ``G = Q diag(exp(lambda)) Q'``; also returns
    each node's subgraph centrality ``G_ii`` and the Estrada index ``tr G``.

    References
    ----------
    Estrada, E. and Hatano, N. (2008). Communicability in complex networks.
    *Physical Review E*, 77(3), 036111.

    Examples
    --------
    >>> r = communicability([[0, 1], [1, 0]])
    >>> round(r.matrix[0][1], 12) == round(math.sinh(1), 12)
    True
    """
    M, n = _adj(A)
    w, V = _eig(M)
    G = [[ssum(V[i][k] * math.exp(w[k]) * V[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
    return RichResult(
        payload={
            "matrix": G,
            "subgraph_centrality": [G[i][i] for i in range(n)],
            "estrada_index": ssum(G[i][i] for i in range(n)),
        }
    )


def modularity_matrix(A, membership=None) -> RichResult:
    r"""Newman's modularity matrix ``B = A - k k' / (2m)`` and, for a partition, modularity ``Q = (1/2m) sum_ij B_ij [c_i = c_j]``.

    ``k`` are the (weighted) degrees and ``m`` the total edge weight. The
    leading eigenvector of ``B`` gives Newman's spectral bisection (signs).

    References
    ----------
    Newman, M. E. J. (2006). Modularity and community structure in
    networks. *PNAS*, 103(23), 8577-8582.

    Examples
    --------
    >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
    >>> round(modularity_matrix(A, [0, 0, 0, 1, 1, 1]).modularity, 6)
    0.357143
    """
    M, n = _adj(A)
    k = [ssum(r) for r in M]
    two_m = ssum(k)
    B = [[M[i][j] - k[i] * k[j] / two_m for j in range(n)] for i in range(n)]
    w, V = _eig(B)
    lead = [V[i][-1] for i in range(n)]
    if lead[max(range(n), key=lambda t: abs(lead[t]))] < 0:
        lead = [-v for v in lead]
    out = {
        "matrix": B,
        "leading_eigenvalue": w[-1],
        "leading_eigenvector": lead,
        "bisection": [1 if v >= 0 else 0 for v in lead],
    }
    if membership is not None:
        c = list(membership)
        out["modularity"] = ssum(B[i][j] for i in range(n) for j in range(n) if c[i] == c[j]) / two_m
    return RichResult(payload=out)


def perron_frobenius(A) -> RichResult:
    r"""Perron-Frobenius eigenpair of a symmetric non-negative matrix: spectral radius and positive eigenvector.

    The eigenvector is scaled to unit Euclidean norm and non-negative sign
    (eigenvector centrality; for a connected graph it is strictly positive);
    ``ratio`` is the spectral gap ``lambda_1 / |lambda_2|``.

    References
    ----------
    Bonacich, P. (1987). Power and centrality: a family of measures.
    *American Journal of Sociology*, 92(5), 1170-1182.

    Examples
    --------
    >>> r = perron_frobenius([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> round(r.eigenvalue, 12), [round(v, 12) for v in r.eigenvector]
    (2.0, [0.57735026919, 0.57735026919, 0.57735026919])
    """
    M, n = _adj(A)
    if any(v < 0 for r in M for v in r):
        raise ValueError("matrix must be non-negative")
    w, V = _eig(M)
    v = [V[i][-1] for i in range(n)]
    if ssum(v) < 0:
        v = [-x for x in v]
    nv = math.sqrt(ssum(x * x for x in v))
    v = [max(0.0, x / nv) if abs(x / nv) < 1e-14 else x / nv for x in v]
    second = max(abs(w[0]), abs(w[-2])) if n > 1 else 0.0
    return RichResult(
        payload={"eigenvalue": w[-1], "eigenvector": v, "ratio": w[-1] / second if second > 0 else math.inf}
    )


def randic_index(A, alpha: float = -0.5) -> float:
    r"""General Randic connectivity index ``R_alpha = sum_{edges uv} (d_u d_v)^alpha`` (Randic 1975: ``alpha = -1/2``).

    Degrees are (weighted) row sums; each undirected edge counted once.

    References
    ----------
    Randic, M. (1975). Characterization of molecular branching. *Journal of
    the American Chemical Society*, 97(23), 6609-6615.

    Examples
    --------
    >>> round(randic_index([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), 12) == round(2 / math.sqrt(2), 12)
    True
    """
    M, n = _adj(A)
    d = [ssum(r) for r in M]
    return ssum((d[i] * d[j]) ** alpha for i in range(n) for j in range(i + 1, n) if M[i][j] != 0)


def label_propagation(A, *, seed: int = 1, max_iter: int = 100) -> RichResult:
    r"""Label-propagation communities (Raghavan, Albert and Kumara 2007), reproducible with Philox draws.

    Every node starts in its own community; in each sweep nodes are visited
    in a random order (Fisher-Yates on Philox stream ``2t``) and adopt the
    label with the largest summed edge weight among their neighbours, ties
    broken by a Philox uniform (stream ``2t + 1``) among the tied labels,
    keeping the current label when it is among them. Stops when no label
    changes. Labels are renumbered 0, 1, ... in order of first appearance;
    also returns the modularity of the partition.

    References
    ----------
    Raghavan, U. N., Albert, R. and Kumara, S. (2007). Near linear time
    algorithm to detect community structures in large-scale networks.
    *Physical Review E*, 76(3), 036106.

    Examples
    --------
    >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
    >>> label_propagation(A).membership
    [0, 0, 0, 1, 1, 1]
    """
    M, n = _adj(A)
    lab = list(range(n))
    it = 0
    for it in range(1, max_iter + 1):
        u = [float(v) for v in random_uniform(n, seed=seed, stream=2 * it)]
        order = list(range(n))
        for t in range(n - 1):
            k = t + int(u[t] * (n - t))
            order[t], order[k] = order[k], order[t]
        tie = [float(v) for v in random_uniform(n, seed=seed, stream=2 * it + 1)]
        changed = False
        for pos, i in enumerate(order):
            score: dict[int, float] = {}
            for j in range(n):
                if M[i][j] != 0 and j != i:
                    score[lab[j]] = score.get(lab[j], 0.0) + M[i][j]
            if not score:
                continue
            best = max(score.values())
            cand = sorted(c for c, s in score.items() if s == best)
            if lab[i] in cand:
                continue
            new = cand[min(len(cand) - 1, int(tie[pos] * len(cand)))]
            lab[i] = new
            changed = True
        if not changed:
            break
    remap: dict[int, int] = {}
    mem = [remap.setdefault(v, len(remap)) for v in lab]
    return RichResult(
        payload={
            "membership": mem,
            "n_communities": len(remap),
            "modularity": modularity_matrix(M, mem)["modularity"],
            "iterations": it,
        }
    )


def cheatsheet() -> str:
    return (
        "estrada_index / communicability / modularity_matrix / perron_frobenius / randic_index / "
        "label_propagation -> spectral graph indices and communities."
    )
