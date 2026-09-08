# morie.fn -- k02 batch (rootcoder007/morie)
"""Degree-corrected stochastic blockmodel objective.

Source consulted: Karrer, B. and Newman, M.E.J. (2011), Stochastic
blockmodels and community structure in networks, *Physical Review E* 83,
016107, equation (16).  Profiling out the block rates in the Poisson
degree-corrected model leaves the objective

    L = sum_{rs} m_rs log( m_rs / (kappa_r kappa_s) )

with m_rs the number of edges between blocks r and s (m_rr counted twice, as
the paper does) and kappa_r the sum of degrees in block r.  Because the
degrees are held fixed, the objective rewards blocks that are denser than
their own degree sequence predicts, which is exactly the failure of the plain
blockmodel that the paper is about: on a graph with a broad degree
distribution the uncorrected model splits by degree instead of by community.
The uncorrected objective is also returned so the two can be compared.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["degree_corrected_sbm"]


def degree_corrected_sbm(A, blocks):
    """Karrer-Newman degree-corrected blockmodel objective.

    Parameters
    ----------
    A : array-like
        Symmetric adjacency matrix.
    blocks : array-like
        Block label per node.

    Returns
    -------
    RichResult
        estimate (degree-corrected objective), uncorrected, m_rs, kappa,
        block_sizes, n_blocks, n_edges, n, method.
    """
    a = np.atleast_2d(np.asarray(A, dtype=float))
    n = a.shape[0]
    lab = list(blocks)
    keys = []
    for c in lab:
        if c not in keys:
            keys.append(c)
    b = len(keys)
    idx = [keys.index(c) for c in lab]
    m = [[0.0] * b for _ in range(b)]
    for i in range(n):
        for j in range(n):
            m[idx[i]][idx[j]] += float(a[i, j])
    deg = [float(t) for t in np.sum(a, axis=1)]
    kappa = [0.0] * b
    for i in range(n):
        kappa[idx[i]] += deg[i]
    sizes = [sum(1 for t in idx if t == r) for r in range(b)]
    ll = 0.0
    for r in range(b):
        for s in range(b):
            if m[r][s] > 0.0 and kappa[r] > 0.0 and kappa[s] > 0.0:
                ll += m[r][s] * float(np.log(m[r][s] / (kappa[r] * kappa[s])))
    tot = float(np.sum(a))
    unc = 0.0
    for r in range(b):
        for s in range(b):
            nn = sizes[r] * sizes[s]
            if m[r][s] > 0.0 and nn > 0:
                unc += m[r][s] * float(np.log(m[r][s] / nn))
    return RichResult(
        payload={
            "estimate": float(ll),
            "uncorrected": float(unc),
            "m_rs": m,
            "kappa": kappa,
            "block_sizes": sizes,
            "n_blocks": int(b),
            "n_edges": float(tot / 2.0),
            "n": int(n),
            "method": "Degree-corrected stochastic blockmodel objective (Karrer & Newman 2011, eq. 16)",
        }
    )


# CANONICAL TEST
# >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
# ...      [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
# >>> good = degree_corrected_sbm(A, [0, 0, 0, 1, 1, 1])
# >>> bad = degree_corrected_sbm(A, [0, 1, 0, 1, 0, 1])
# >>> assert good["estimate"] > bad["estimate"]     # the real split scores higher
# >>> assert abs(sum(good["kappa"]) - 2 * good["n_edges"] * 1.0) < 1e-12


def cheatsheet():
    return "sbmdg2(A, blocks): degree-corrected blockmodel objective."


degreecorrectedsbm = degree_corrected_sbm
