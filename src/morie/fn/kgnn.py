# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Relational graph convolution (R-GCN).

Schlichtkrull, Kipf, Bloem, van den Berg, Titov and Welling (2018),
"Modeling relational data with graph convolutional networks", ESWC
2018, arXiv:1703.06103, equation (2):

    h_i' = sigma( sum_r sum_{j in N_i^r} (1/c_{i,r}) W_r h_j + W_0 h_i ),

with the normalisation constant c_{i,r} = |N_i^r| as the paper's own
default.  A relation with no neighbours for node i contributes nothing
rather than dividing by zero.  sigma is the rectifier.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["r_gcn"]


def r_gcn(A_r, X, W_r, W0=None):
    """One R-GCN propagation step over a list of per-relation adjacencies."""
    if len(A_r) == 0:
        raise ValueError("r_gcn: no relations supplied")
    if len(A_r) != len(W_r):
        raise ValueError("r_gcn: A_r and W_r have different lengths")
    H = core.mat(X)
    n = len(H)
    if n == 0:
        raise ValueError("r_gcn: X is empty")
    p = len(H[0])
    Ms = []
    Ws = []
    for r in range(len(A_r)):
        M = core.mat(A_r[r])
        if len(M) != n:
            raise ValueError("r_gcn: relation %d has the wrong node count" % r)
        for row in M:
            if len(row) != n:
                raise ValueError("r_gcn: relation %d is not square" % r)
        Wm = core.mat(W_r[r])
        if len(Wm) != p:
            raise ValueError("r_gcn: W_r[%d] must have one row per input feature" % r)
        Ms.append(M)
        Ws.append(Wm)
    q = len(Ws[0][0])
    W0m = [[1.0 if i == j else 0.0 for j in range(q)] for i in range(p)] if W0 is None else core.mat(W0)
    out = [[0.0] * q for _ in range(n)]
    for r in range(len(Ms)):
        M = Ms[r]
        agg = [[0.0] * p for _ in range(n)]
        for i in range(n):
            nb = [j for j in range(n) if M[i][j] != 0.0]
            if not nb:
                continue
            c = float(len(nb))
            for j in nb:
                for cc in range(p):
                    agg[i][cc] += H[j][cc] / c
        prod = core.matmul(agg, Ws[r])
        for i in range(n):
            for cc in range(q):
                out[i][cc] += prod[i][cc]
    self_term = core.matmul(H, W0m)
    Hout = [[core.relu(out[i][cc] + self_term[i][cc]) for cc in range(q)] for i in range(n)]
    flat = [v for row in Hout for v in row]
    return RichResult(
        title="Relational GCN layer",
        summary_lines=[("nodes", n), ("relations", len(Ms))],
        payload={
            "estimate": sum(flat) / len(flat),
            "H": Hout,
            "n": n,
            "relations": len(Ms),
            "method": "h' = relu(sum_r sum_j W_r h_j / |N_i^r| + W_0 h_i), Schlichtkrull et al. (2018) eq. (2)",
        },
    )


def cheatsheet():
    return "kgnn: relational GCN (R-GCN)"


# compact alias per ledger/NAMING.md
rgcn = r_gcn
