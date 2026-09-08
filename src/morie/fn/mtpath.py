# morie.fn -- slice s03 (rootcoder007/morie)
"""Meta-path counts on a heterogeneous information network.

Source consulted: Sun, Y., Han, J., Yan, X., Yu, P. S. and Wu, T.
(2011).  PathSim: meta path-based top-k similarity search in
heterogeneous information networks.  *PVLDB* 4(11), 992-1003.  A meta
path P = A_1 -> A_2 -> ... -> A_(l+1) is a sequence of object *types*,
and the number of path instances between two objects following P is the
(i, j) entry of the product of the corresponding type-restricted
adjacency matrices,

    M_P = W_(A1,A2) W_(A2,A3) ... W_(Al,A(l+1))

PathSim itself normalises this into

    s(x, y) = 2 M_P(x, y) / ( M_P(x, x) + M_P(y, y) )

for a symmetric meta path, which is returned as well.  The PVLDB paper
is open access but was not retrievable here; both expressions are quoted
in their standard published form.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["meta_path"]


def meta_path(G, node_types=None, metapath=None):
    """Path-instance counts and PathSim similarities along a meta path.

    Parameters
    ----------
    G : 2-D array-like
        Adjacency matrix of the whole heterogeneous network.
    node_types : array-like
        Type label per node.
    metapath : array-like
        The sequence of types, e.g. ["A", "P", "A"].

    Returns
    -------
    RichResult with payload:
        estimate : total number of path instances
        M        : the count matrix restricted to the endpoint type
        pathsim  : the PathSim matrix (nan when the path is not symmetric)
        counts   : row sums of M
    """
    W = k.mat(G)
    n = len(W)
    ty = [str(t) for t in node_types] if node_types is not None else ["0"] * n
    mp = [str(t) for t in metapath] if metapath is not None else ty[:1] * 2
    M = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for step in range(len(mp) - 1):
        a = mp[step]
        b = mp[step + 1]
        S = [[W[i][j] if (ty[i] == a and ty[j] == b) else 0.0 for j in range(n)]
             for i in range(n)]
        M = k.matmul(M, S)
    ends = [i for i in range(n) if ty[i] == mp[-1]]
    starts = [i for i in range(n) if ty[i] == mp[0]]
    tot = 0.0
    for i in starts:
        for j in ends:
            tot += M[i][j]
    sym = mp[0] == mp[-1]
    ps = [[float("nan")] * n for _ in range(n)]
    if sym:
        for i in starts:
            for j in starts:
                den = M[i][i] + M[j][j]
                ps[i][j] = 2.0 * M[i][j] / den if den > 0.0 else 0.0
    counts = []
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += M[i][j]
        counts.append(s)
    return RichResult(
        title="Meta-path analysis",
        summary_lines=[("path instances", tot)],
        payload={
            "estimate": tot,
            "M": M,
            "pathsim": ps,
            "counts": counts,
            "symmetric": sym,
            "method": "Meta-path instance counts and PathSim (Sun et al. 2011)",
        },
    )


def cheatsheet():
    return "mtpath: Meta-path analysis on heterogeneous network"


metapath = meta_path
