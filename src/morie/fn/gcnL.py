# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Graph convolution with the renormalisation trick.

Kipf and Welling (2017), ICLR 2017, arXiv:1609.02907, equation (8)
together with the layer-wise rule of equation (2):

    H^{l+1} = sigma( Dt^{-1/2} At Dt^{-1/2} H^l W^l ),
    At = A + I_N,     Dt_ii = sum_j At_ij.

Adding the self-loop before normalising is what keeps the eigenvalues
of the propagation operator inside [-1, 1] and stops the repeated
application from exploding.  sigma is the rectifier.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gcn"]


def gcn(A, X, W):
    """One renormalised GCN propagation step."""
    M = core.mat(A)
    n = len(M)
    if n == 0:
        raise ValueError("gcn: adjacency matrix is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("gcn: adjacency matrix must be square")
    H = core.mat(X)
    if len(H) != n:
        raise ValueError("gcn: X must have one row per node")
    Wm = core.mat(W)
    if len(Wm) != len(H[0]):
        raise ValueError("gcn: W must have one row per input feature")
    At = [[M[i][j] + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    d = [sum(At[i]) for i in range(n)]
    s = [0.0 if d[i] <= 0 else d[i] ** -0.5 for i in range(n)]
    An = [[s[i] * At[i][j] * s[j] for j in range(n)] for i in range(n)]
    Z = core.matmul(core.matmul(An, H), Wm)
    Hout = [[core.relu(v) for v in row] for row in Z]
    flat = [v for row in Hout for v in row]
    return RichResult(
        title="Renormalised graph convolutional layer",
        summary_lines=[("nodes", n), ("out_features", len(Hout[0]))],
        payload={
            "estimate": sum(flat) / len(flat),
            "H": Hout,
            "preactivation": Z,
            "n": n,
            "method": "H' = relu(Dt^{-1/2}(A+I)Dt^{-1/2} X W), Kipf & Welling (2017) eq. (8)",
        },
    )
