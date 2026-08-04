# morie.fn -- k02 batch (rootcoder007/morie)
"""Stochastic blockmodel: block edge probabilities and log-likelihood.

Source consulted: Holland, P.W., Laskey, K.B. and Leinhardt, S. (1983),
Stochastic blockmodels: first steps, *Social Networks* 5(2), 109-137.  Given
a partition of the nodes into blocks, edges are independent Bernoulli draws
whose probability depends only on the pair of blocks, so the maximum
likelihood estimate is the observed density of each block pair,

    p_rs = e_rs / (n_r n_s)      r != s
    p_rr = 2 e_rr / (n_r (n_r - 1))

with e_rs the number of edges between blocks r and s (e_rr counted once).  The
log-likelihood is the sum over unordered node pairs of the Bernoulli term, so
it is directly comparable across partitions of the same graph.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["stochastic_block_model"]


def stochastic_block_model(A, blocks):
    """Blockmodel probabilities and Bernoulli log-likelihood.

    Parameters
    ----------
    A : array-like
        Symmetric binary adjacency matrix (no self-loops used).
    blocks : array-like
        Block label per node.

    Returns
    -------
    RichResult
        estimate (log-likelihood), probabilities, edge_counts, pair_counts,
        block_sizes, n_blocks, n, method.
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
    e = [[0.0] * b for _ in range(b)]
    npair = [[0.0] * b for _ in range(b)]
    for i in range(n):
        for j in range(i + 1, n):
            r = idx[i]
            s = idx[j]
            e[r][s] += float(a[i, j])
            npair[r][s] += 1.0
            if r != s:
                e[s][r] += float(a[i, j])
                npair[s][r] += 1.0
    p = [[0.0] * b for _ in range(b)]
    for r in range(b):
        for s in range(b):
            p[r][s] = e[r][s] / npair[r][s] if npair[r][s] > 0.0 else 0.0
    ll = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pr = p[idx[i]][idx[j]]
            y = float(a[i, j])
            if 0.0 < pr < 1.0:
                ll += y * float(np.log(pr)) + (1.0 - y) * float(np.log(1.0 - pr))
    sizes = [sum(1 for t in idx if t == r) for r in range(b)]
    return RichResult(
        payload={
            "estimate": float(ll),
            "probabilities": p,
            "edge_counts": e,
            "pair_counts": npair,
            "block_sizes": sizes,
            "n_blocks": int(b),
            "n": int(n),
            "method": "Stochastic blockmodel MLE (Holland, Laskey & Leinhardt 1983)",
        }
    )


# CANONICAL TEST
# >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
# ...      [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
# >>> r = stochastic_block_model(A, [0, 0, 0, 1, 1, 1])
# >>> assert abs(r["probabilities"][0][0] - 1.0) < 1e-15   # both blocks are triangles
# >>> assert abs(r["probabilities"][0][1] - 1.0 / 9.0) < 1e-15
# >>> assert r["block_sizes"] == [3, 3]


def cheatsheet():
    return "sbmest(A, blocks): stochastic blockmodel probabilities and log-likelihood."


stochasticblockmodel = stochastic_block_model
