# morie.fn -- function file (rootcoder007/morie)
"""Graph convolution propagation step."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sgt_graph_neural_propagation"]


def sgt_graph_neural_propagation(A_hat, X, W, activation="relu"):
    """One symmetric-normalised graph convolution.

    The normalisation is the whole design.  A plain adjacency multiply
    lets high-degree nodes dominate and makes the features explode over
    layers; ``D^-1/2 A D^-1/2`` bounds the spectrum at one, so stacking
    layers stays numerically sane.  The self-loop is what stops a node
    from forgetting itself at every step.

    Formula: ``X^(k+1) = sigma(A_hat X^(k) W^(k))`` with
    ``A_hat = D~^-1/2 (A + I) D~^-1/2``.

    Parameters
    ----------
    A_hat : array-like, shape (n, n)
        Adjacency.  Renormalised here, so pass the raw ``A``; passing an
        already-normalised matrix simply normalises a second time.
    X : array-like, shape (n, f)
        Node features.
    W : array-like, shape (f, f_out)
        Layer weights.
    activation : str, default "relu"
        ``relu`` or ``none``.

    Returns
    -------
    RichResult
        ``X_next``, ``estimate`` (its mean), ``A_norm``, ``n``,
        ``f_out``.

    References
    ----------
    Kipf, T. N. & Welling, M. (2017).  Semi-supervised classification
    with graph convolutional networks.  ICLR 2017, equation (2).
    """
    A = C.mat(A_hat)
    n = len(A)
    Ai = [[A[i][j] + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    d = [sum(Ai[i]) for i in range(n)]
    An = [[Ai[i][j] / math.sqrt(d[i] * d[j]) if d[i] > 0 and d[j] > 0 else 0.0
           for j in range(n)] for i in range(n)]
    Z = C.matmul(C.matmul(An, C.mat(X)), C.mat(W))
    if activation == "relu":
        Z = [[v if v > 0.0 else 0.0 for v in row] for row in Z]
    fo = len(Z[0])
    return RichResult(payload={
        "X_next": Z, "estimate": sum(sum(row) for row in Z) / (n * fo),
        "A_norm": An, "n": n, "f_out": fo,
        "method": "Symmetric-normalised graph convolution"})


def cheatsheet():
    return "sgtgrn: Graph convolution propagation step."
