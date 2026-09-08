# morie.fn -- tail3 batch (rootcoder007/morie)
"""GraphSAGE embedding generation (forward propagation).

Source consulted: Hamilton, W.L., Ying, R. & Leskovec, J. (2017). Inductive
Representation Learning on Large Graphs.  NIPS 2017, arXiv:1706.02216,
Algorithm 1 and equation (2).  Algorithm 1 is

    h_v^0 = x_v
    for k = 1..K:
        h_{N(v)}^k = AGGREGATE_k({h_u^{k-1}, u in N(v)})
        h_v^k      = sigma( W^k . CONCAT(h_v^{k-1}, h_{N(v)}^k) )
        h_v^k      = h_v^k / ||h_v^k||_2

with the mean aggregator taking the elementwise mean over the neighbourhood.
Equation (2) gives the convolutional variant, which drops the concatenation
and takes the mean over {v} union N(v) instead.

The paper samples a fixed-size neighbourhood; this implementation uses the
full neighbourhood set, which Algorithm 1 itself is stated for and which
keeps the forward pass deterministic.
"""

from __future__ import annotations

from . import _array_core as np

from . import t3util as _t3
from ._richresult import RichResult

__all__ = ["graphsage"]


def _aggregate(A, H, how):
    n = int(A.shape[0])
    d = int(H.shape[1])
    out = [[0.0] * d for _ in range(n)]
    for v in range(n):
        nb = [u for u in range(n) if float(A[v, u]) != 0.0]
        if not nb:
            continue
        for j in range(d):
            vals = [float(H[u, j]) for u in nb]
            if how == "mean":
                out[v][j] = sum(vals) / len(vals)
            elif how == "sum":
                out[v][j] = sum(vals)
            elif how == "max":
                out[v][j] = max(vals)
            else:
                raise ValueError("aggregator must be mean, sum or max")
    return np.asarray(out, dtype=float)


def graphsage(G, X, W=None, aggregator="mean", K=1, convolutional=False):
    """GraphSAGE forward pass over the full neighbourhood.

    Parameters
    ----------
    G : array-like
        Square adjacency matrix; non-zero entries are edges.
    X : array-like
        Node feature matrix, one row per node.
    W : array-like, optional
        Weight matrix.  Shape ``(2d, d)`` for Algorithm 1 (it multiplies the
        concatenation) or ``(d, d)`` when ``convolutional`` is set.  Defaults
        to the matrix that averages the self and neighbourhood halves, which
        makes the layer parameter-free and reproducible.
    aggregator : {"mean", "sum", "max"}
        AGGREGATE_k of Algorithm 1.
    K : int
        Search depth, the number of layers.
    convolutional : bool
        Use equation (2) instead of Algorithm 1 line 5.

    Returns
    -------
    RichResult
        estimate (mean embedding entry), Z (embeddings), frob, n, dim, method.

    References
    ----------
    Hamilton, Ying & Leskovec (2017), arXiv:1706.02216, Algorithm 1, eq. (2).
    """
    A = np.atleast_2d(np.asarray(G, dtype=float))
    H = np.atleast_2d(np.asarray(X, dtype=float))
    n = int(A.shape[0])
    d = int(H.shape[1])
    if W is None:
        if convolutional:
            wm = np.eye(d)
        else:
            rows = []
            for i in range(2 * d):
                row = [0.0] * d
                row[i % d] = 0.5
                rows.append(row)
            wm = np.asarray(rows, dtype=float)
    else:
        wm = np.atleast_2d(np.asarray(W, dtype=float))
    for _ in range(int(K)):
        agg = _aggregate(A, H, aggregator)
        if convolutional:
            comb = []
            for v in range(n):
                nb = [u for u in range(n) if float(A[v, u]) != 0.0] + [v]
                comb.append([sum(float(H[u, j]) for u in nb) / len(nb) for j in range(d)])
            pre = np.asarray(comb, dtype=float) @ wm
        else:
            cat = []
            for v in range(n):
                cat.append([float(H[v, j]) for j in range(d)] + [float(agg[v, j]) for j in range(d)])
            pre = np.asarray(cat, dtype=float) @ wm
        Hn = _t3.relu(pre)
        rows = []
        for v in range(n):
            r = [float(Hn[v, j]) for j in range(int(Hn.shape[1]))]
            nrm = float(np.sqrt(sum(t * t for t in r)))
            rows.append([t / nrm for t in r] if nrm > 0.0 else r)
        H = np.asarray(rows, dtype=float)
        d = int(H.shape[1])
    frob = float(np.sqrt(float(np.sum(H * H))))
    return RichResult(
        payload={
            "estimate": float(np.mean(H)),
            "Z": H,
            "frob": frob,
            "n": n,
            "dim": d,
            "method": "GraphSAGE forward propagation (Hamilton, Ying & Leskovec 2017)",
        }
    )


# CANONICAL TEST
# >>> # every embedding row is L2-normalised, so ||Z||_F = sqrt(n)
# >>> A = [[0, 1], [1, 0]]
# >>> r = graphsage(A, [[1.0, 0.0], [0.0, 1.0]])
# >>> assert abs(r["frob"] - 2.0 ** 0.5) < 1e-12


def cheatsheet():
    return "sage(G, X, W, aggregator, K): GraphSAGE forward pass."
