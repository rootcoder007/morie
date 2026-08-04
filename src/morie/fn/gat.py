# morie.fn -- function file (rootcoder007/morie)
"""Graph attention layer."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["gat"]


def gat(A, X, W, a, alpha_leaky=0.2):
    """Let each node decide how much to weight each neighbour.

    Fixed aggregation -- mean or degree-normalised sum -- treats every
    neighbour alike, which is wrong whenever some edges matter more.
    Attention makes the weight a learned function of the two endpoints,
    and crucially it is computed per edge from node features alone, so
    the layer never needs to see the whole graph and transfers to graphs
    it was not trained on.

    Formula: ``e_ij = LeakyReLU(a' [W h_i || W h_j])``,
    ``alpha_ij = softmax_j(e_ij)`` over the neighbours of ``i``, and
    ``h_i' = sum_j alpha_ij W h_j``.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Adjacency; self-loops are added, as in the paper.
    X : array-like, shape (n, f)
        Node features.
    W : array-like, shape (f, f_out)
        Shared linear map.
    a : array-like, shape (2 f_out,)
        Attention vector.
    alpha_leaky : float, default 0.2
        LeakyReLU negative slope.

    Returns
    -------
    RichResult
        ``H`` (updated features), ``alpha`` (attention matrix),
        ``estimate`` (mean of ``H``), ``n``, ``f_out``.

    References
    ----------
    Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. &
    Bengio, Y. (2018).  Graph attention networks.  ICLR 2018,
    equations (1) to (4).
    """
    Am = C.mat(A)
    Xm = C.mat(X)
    Wm = C.mat(W)
    av = C.vec(a)
    n = len(Am)
    Wh = C.matmul(Xm, Wm)
    fo = len(Wh[0])
    alpha = [[0.0] * n for _ in range(n)]
    for i in range(n):
        nb = [j for j in range(n) if Am[i][j] != 0.0 or i == j]
        e = []
        for j in nb:
            s = sum(av[k] * Wh[i][k] for k in range(fo)) + \
                sum(av[fo + k] * Wh[j][k] for k in range(fo))
            e.append(s if s > 0.0 else alpha_leaky * s)
        sm = S.softmax(e)
        for t, j in enumerate(nb):
            alpha[i][j] = sm[t]
    H = [[sum(alpha[i][j] * Wh[j][k] for j in range(n)) for k in range(fo)]
         for i in range(n)]
    return RichResult(payload={
        "H": H, "alpha": alpha,
        "estimate": sum(sum(row) for row in H) / (n * fo), "n": n, "f_out": fo,
        "method": "Graph attention layer"})


def cheatsheet():
    return "gat: Graph attention layer."
