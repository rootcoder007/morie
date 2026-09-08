# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Deep kernel learning.

Wilson, Hu, Salakhutdinov and Xing (2016), "Deep kernel learning",
AISTATS 2016 (PMLR 51:370-378), arXiv:1511.02222, equation (1):

    k(x, x') -> k( g(x, w), g(x', w) ),

a base kernel applied to the output of a neural network g rather than
to the raw inputs.  The network here is a fixed one-hidden-layer tanh
map whose weights are supplied by the caller (no training loop, so
both language arms follow the same deterministic map); an identity
map recovers the plain RBF GP exactly, which is the identity the
tests check.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["deep_kernel_gp"]


def _apply(A, W1, b1, W2, b2, act):
    out = []
    for row in A:
        h = []
        for j in range(len(W1[0])):
            s = b1[j]
            for i in range(len(row)):
                s += row[i] * W1[i][j]
            h.append(math.tanh(s) if act else s)
        z = []
        for j in range(len(W2[0])):
            s = b2[j]
            for i in range(len(h)):
                s += h[i] * W2[i][j]
            z.append(s)
        out.append(z)
    return out


def _k(A, B, ell, var):
    out = []
    for i in range(len(A)):
        row = []
        for j in range(len(B)):
            s = 0.0
            for c in range(len(A[i])):
                d = A[i][c] - B[j][c]
                s += d * d
            row.append(var * math.exp(-0.5 * s / (ell * ell)))
        out.append(row)
    return out


def deep_kernel_gp(X, y, X_test=None, nn=None, lengthscale=1.0, variance=1.0, noise=0.01):
    """GP regression whose kernel acts on a fixed neural feature map.

    Parameters
    ----------
    nn : dict-like with W1, b1, W2, b2 and optionally tanh (default True).
        None means the identity map, which reduces to the plain RBF GP.
    """
    A = core.mat(X)
    yv = core.vec(y)
    n = len(A)
    if n == 0:
        raise ValueError("deep_kernel_gp: X is empty")
    if len(yv) != n:
        raise ValueError("deep_kernel_gp: X and y have different lengths")
    d = len(A[0])
    Xs = A if X_test is None else core.mat(X_test)
    ell = float(lengthscale)
    var = float(variance)
    s2 = float(noise)
    if ell <= 0 or var <= 0:
        raise ValueError("deep_kernel_gp: lengthscale and variance must be positive")
    if s2 < 0:
        raise ValueError("deep_kernel_gp: noise must be non-negative")
    if nn is None:
        GA, GS = A, Xs
        act = 0
    else:
        g = (lambda k, dv=None: nn.get(k, dv)) if hasattr(nn, "get") else (lambda k, dv=None: getattr(nn, k, dv))
        W1 = core.mat(g("W1"))
        W2 = core.mat(g("W2"))
        b1 = core.vec(g("b1", [0.0] * len(W1[0])))
        b2 = core.vec(g("b2", [0.0] * len(W2[0])))
        act = 1 if g("tanh", True) else 0
        if len(W1) != d:
            raise ValueError("deep_kernel_gp: W1 must have one row per input feature")
        if len(W2) != len(W1[0]):
            raise ValueError("deep_kernel_gp: W2 must have one row per hidden unit")
        GA = _apply(A, W1, b1, W2, b2, act)
        GS = _apply(Xs, W1, b1, W2, b2, act)
    K = _k(GA, GA, ell, var)
    for i in range(n):
        K[i][i] += s2
    alpha = core.cholsolve(K, yv)
    Ks = _k(GS, GA, ell, var)
    mu = [sum(Ks[j][i] * alpha[i] for i in range(n)) for j in range(len(GS))]
    sd = []
    for j in range(len(GS)):
        z = core.cholsolve(K, Ks[j])
        sd.append(max(var - sum(Ks[j][i] * z[i] for i in range(n)), 0.0))
    L = core.chol(K)
    ll = -0.5 * sum(yv[i] * alpha[i] for i in range(n)) - sum(math.log(L[i][i]) for i in range(n)) - 0.5 * n * math.log(2.0 * math.pi)
    return RichResult(
        title="Deep kernel GP",
        summary_lines=[("n", n), ("features", len(GA[0]))],
        payload={
            "estimate": mu[0],
            "mean": mu,
            "variance": sd,
            "loglik": ll,
            "features": len(GA[0]),
            "n": n,
            "method": "k(g(x), g(x')) with a fixed tanh feature map, Wilson et al. (2016) eq. (1)",
        },
    )


def cheatsheet():
    return "gpdkl: deep kernel learning GP"


# compact alias per ledger/NAMING.md
deepkernelgp = deep_kernel_gp
