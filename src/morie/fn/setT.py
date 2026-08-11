# morie.fn -- wave 3 slice w5_00 (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Set Transformer attention pooling (MAB / SAB / PMA).

Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S. and Teh, Y. W.
(2019), "Set Transformer: A Framework for Attention-based
Permutation-Invariant Neural Networks", ICML 2019 (PMLR 97),
arXiv:1810.00825. Implemented equations (Section 3.1-3.2, as printed
in the paper PDF):

    MAB(X, Y) = LayerNorm(H + rFF(H)),
        where H = LayerNorm(X + Multihead(X, Y, Y))          (Eq 7)
    SAB(X)    = MAB(X, X)                                    (Eq 8)
    PMA_k(Z)  = MAB(S, rFF(Z))                               (Sec 3.2)

with S a learnable k x d seed matrix: pooling by multihead attention
maps an n-element set to exactly k output vectors. Attention here is
single-head scaled dot-product with explicit projection weights, rFF
is the row-wise map relu(h W1 + b1) W2 + b2, and LayerNorm follows
Ba et al. (2016). Because every row of Z enters only through K and V
of the attention, PMA is PERMUTATION INVARIANT in the set elements --
that theorem (paper Sec 3, Prop 1 context) is the test anchor.

Source: fetched-wave3/lee-etal-2019-set-transformer-arxiv1810.00825.pdf
(Eqs 7-8 and the PMA definition; verified in the PDF text).
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["setT", "set_transformer"]


def _ln(row, eps=1e-5):
    n = len(row)
    mu = sum(row) / n
    var = sum((v - mu) ** 2 for v in row) / n
    s = math.sqrt(var + eps)
    return [(v - mu) / s for v in row]


def _rff(M, W1, b1, W2, b2):
    H = np.asarray(M, dtype=float) @ W1
    R = []
    for row in H:
        r = [max(0.0, float(v) + float(b)) for v, b in zip(row, b1)]
        R.append(r)
    O = np.asarray(R, dtype=float) @ W2
    return [[float(v) + float(b) for v, b in zip(row, b2)] for row in O]


def _attend(X, Y, Wq, Wk, Wv):
    Q = np.asarray(X, dtype=float) @ Wq
    K = np.asarray(Y, dtype=float) @ Wk
    V = np.asarray(Y, dtype=float) @ Wv
    dk = Q.shape[1]
    S = (Q @ K.T) * (1.0 / math.sqrt(dk))
    W = []
    for row in S:
        r = [float(v) for v in row]
        m = max(r)
        e = [math.exp(v - m) for v in r]
        z = sum(e)
        W.append([v / z for v in e])
    O = np.asarray(W, dtype=float) @ V
    return [[float(v) for v in row] for row in O], W


def _mab(X, Y, p):
    A, W = _attend(X, Y, p["Wq"], p["Wk"], p["Wv"])
    H = [_ln([x + a for x, a in zip(xr, ar)])
         for xr, ar in zip(X, A)]
    F = _rff(H, p["W1"], p["b1"], p["W2"], p["b2"])
    O = [_ln([h + f for h, f in zip(hr, fr)])
         for hr, fr in zip(H, F)]
    return O, W


def setT(Z, S, params):
    """PMA_k attention pooling (Lee et al. 2019, arXiv:1810.00825).

    Parameters
    ----------
    Z : array-like, shape (n, d)
        The input set, one element per row.
    S : array-like, shape (k, d)
        Seed matrix (the learnable queries); k is the output size.
    params : dict
        Weights: Wq, Wk, Wv (d x d), W1 (d x d_ff), b1 (d_ff),
        W2 (d_ff x d), b2 (d) -- used for both the pre-pool rFF of Z
        and inside the MAB.

    Returns
    -------
    result : RichResult
        Keys: output (k x d), attention (k x n weights of the pooling
        MAB), k, estimate, n, method.
    """
    Za = np.atleast_2d(np.asarray(Z, dtype=float))
    Sa = np.atleast_2d(np.asarray(S, dtype=float))
    if Za.shape[1] != Sa.shape[1]:
        raise ValueError(
            f"setT: Z width {Za.shape[1]} != seed width {Sa.shape[1]}")
    for name in ("Wq", "Wk", "Wv", "W1", "b1", "W2", "b2"):
        if name not in params:
            raise ValueError(f"setT: params is missing {name}")
    p = {k: np.asarray(v, dtype=float) for k, v in params.items()}
    Zl = [[float(v) for v in row] for row in Za]
    Sl = [[float(v) for v in row] for row in Sa]
    FZ = _rff(Zl, p["W1"], p["b1"], p["W2"], p["b2"])   # rFF(Z)
    O, W = _mab(Sl, FZ, p)                              # MAB(S, rFF(Z))
    return RichResult(payload={
        "output": O,
        "attention": W,
        "k": len(Sl),
        "estimate": float(O[0][0]),
        "n": int(Za.shape[0]),
        "method": "Set Transformer PMA_k(Z) = MAB(S, rFF(Z)) (Lee et al. 2019, Eq 7 + Sec 3.2)",
    })


def set_transformer(X=None, k=None, S=None, params=None):
    """Back-compatible wrapper over :func:`setT` (old stub name).

    The stub took (X, k); a seed matrix and weights are required for
    the real method, so S and params must be supplied.
    """
    if X is None or S is None or params is None:
        raise ValueError("set_transformer: X, S and params are required")
    return setT(X, S, params)


def cheatsheet():
    return "setT: Set Transformer PMA pooling (Lee et al. 2019, arXiv:1810.00825, Eq 7 + Sec 3.2)"
