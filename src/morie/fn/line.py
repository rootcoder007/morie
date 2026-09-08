# morie.fn -- slice s03 (rootcoder007/morie)
"""LINE: large-scale information network embedding.

Source consulted (FETCHED): Tang, J., Qu, M., Wang, M., Zhang, M., Yan,
J. and Mei, Q. (2015).  LINE: large-scale information network embedding.
*WWW* 24, 1067-1077 (arXiv:1503.03578).  First-order proximity, its
equations (1)-(3):

    p_1(v_i, v_j) = 1 / (1 + exp(-u_i' u_j))
    O_1 = d( phat_1(., .), p_1(., .) )
        = - sum_((i,j) in E) w_ij log p_1(v_i, v_j)   + const

and second-order proximity, its equations (4)-(6):

    p_2(v_j | v_i) = exp(u'_j . u_i) / sum_(k=1)^|V| exp(u'_k . u_i)
    O_2 = - sum_((i,j) in E) w_ij log p_2(v_j | v_i)

both being the KL divergence between the empirical and the modelled
proximity, with the constant terms dropped -- exactly as the paper
derives them.

Embeddings are supplied by the caller, or fitted by a fixed number of
full-batch gradient steps at a fixed step size; there is no negative
sampling and no edge sampling, because both would need a generator and
the objectives above are what is being computed.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["line"]


def line(G, dim=2, order=1, U=None, Uc=None, steps=0, lr=0.05):
    """LINE first- or second-order objective, and optionally a fit.

    Parameters
    ----------
    G : 2-D array-like
        Weighted adjacency matrix.
    dim : int
        Embedding dimension.
    order : {1, 2}
        Which proximity to use.
    U : 2-D array-like, optional
        Vertex embeddings; a deterministic low-discrepancy start is used
        when absent.
    Uc : 2-D array-like, optional
        Context embeddings for the second-order objective.
    steps : int
        Full-batch gradient steps to take.
    lr : float
        Step size.

    Returns
    -------
    RichResult with payload:
        estimate : the objective after ``steps`` steps
        O        : same as estimate
        U, Uc    : the embeddings
        O_start  : the objective before any step
    """
    W = k.mat(G)
    n = len(W)
    d = int(dim)
    if U is None:
        U = [[k.vdc(i * d + j, 2) - 0.5 for j in range(d)] for i in range(n)]
    else:
        U = k.mat(U)
    if Uc is None:
        Uc = [[k.vdc(i * d + j, 3) - 0.5 for j in range(d)] for i in range(n)]
    else:
        Uc = k.mat(Uc)

    def obj():
        o = 0.0
        for i in range(n):
            if int(order) == 2:
                logits = []
                for c in range(n):
                    s = 0.0
                    for a in range(d):
                        s += Uc[c][a] * U[i][a]
                    logits.append(s)
                lse = k.logsumexp(logits)
            for j in range(n):
                if W[i][j] == 0.0:
                    continue
                if int(order) == 2:
                    o -= W[i][j] * (logits[j] - lse)
                else:
                    s = 0.0
                    for a in range(d):
                        s += U[i][a] * U[j][a]
                    o -= W[i][j] * math.log(k.sigmoid(s) if k.sigmoid(s) > 1e-300
                                            else 1e-300)
        return o

    o0 = obj()
    for _ in range(int(steps)):
        gU = [[0.0] * d for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if W[i][j] == 0.0 or int(order) != 1:
                    continue
                s = 0.0
                for a in range(d):
                    s += U[i][a] * U[j][a]
                c = W[i][j] * (k.sigmoid(s) - 1.0)
                for a in range(d):
                    gU[i][a] += c * U[j][a]
                    gU[j][a] += c * U[i][a]
        for i in range(n):
            for a in range(d):
                U[i][a] -= float(lr) * gU[i][a]
    o1 = obj()
    return RichResult(
        title="LINE embedding objective",
        summary_lines=[("O", o1), ("order", int(order))],
        payload={
            "estimate": o1,
            "O": o1,
            "O_start": o0,
            "U": U,
            "Uc": Uc,
            "n": n,
            "method": "LINE first/second-order proximity objective (Tang et al. 2015, eqs. 1-6)",
        },
    )


def cheatsheet():
    return "line: LINE embeddings (1st + 2nd order)"
