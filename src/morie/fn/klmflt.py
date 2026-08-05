# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Kalman forward recursion driven by a model specification.

Kalman (1960), Trans. ASME J. Basic Engineering 82(1):35-45,
doi:10.1115/1.3662552.  Identical recursion to the matrix-argument
form, but the system is passed as one object with entries F, H, Q, R
and optionally x0 and P0 -- the shape a fitted model is usually kept
in.  Prediction:

    x_{t|t-1} = F x_{t-1},  P_{t|t-1} = F P F' + Q,

update with gain K_t = P_{t|t-1} H' (H P_{t|t-1} H' + R)^{-1}.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["kalman_filter"]


def _get(model, key, default=None):
    if hasattr(model, "get"):
        v = model.get(key, default)
    else:
        v = getattr(model, key, default)
    if v is None and default is None:
        raise ValueError("kalman_filter: model is missing entry " + key)
    return default if v is None else v


def kalman_filter(y, model):
    """Filtered states and log-likelihood for a packaged state space model."""
    Y = core.mat(y)
    n = len(Y)
    if n == 0:
        raise ValueError("kalman_filter: y is empty")
    m = len(Y[0])
    F = core.mat(_get(model, "F"))
    H = core.mat(_get(model, "H"))
    Q = core.mat(_get(model, "Q"))
    R = core.mat(_get(model, "R"))
    d = len(F)
    if len(F[0]) != d:
        raise ValueError("kalman_filter: F must be square")
    if len(H) != m or len(H[0]) != d:
        raise ValueError("kalman_filter: H must be m x d")
    x = core.vec(_get(model, "x0", [0.0] * d))
    P = core.mat(_get(model, "P0", [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]))
    if len(x) != d or len(P) != d:
        raise ValueError("kalman_filter: x0 and P0 must match the state dimension")
    xs = []
    ll = 0.0
    for t in range(n):
        xpred = core.matvec(F, x)
        FP = core.matmul(F, P)
        Ppred = [[sum(FP[i][k] * F[j][k] for k in range(d)) + Q[i][j] for j in range(d)] for i in range(d)]
        HP = core.matmul(H, Ppred)
        S = [[sum(HP[i][k] * H[j][k] for k in range(d)) + R[i][j] for j in range(m)] for i in range(m)]
        v = [Y[t][i] - sum(H[i][k] * xpred[k] for k in range(d)) for i in range(m)]
        Kcols = []
        for j in range(d):
            e = [HP[i][j] for i in range(m)]
            Kcols.append(core.cholsolve(S, e))
        K = [[Kcols[j][i] for i in range(m)] for j in range(d)]
        x = [xpred[j] + sum(K[j][i] * v[i] for i in range(m)) for j in range(d)]
        KH = core.matmul(K, H)
        P = [[Ppred[i][j] - sum(KH[i][k] * Ppred[k][j] for k in range(d)) for j in range(d)] for i in range(d)]
        sv = core.cholsolve(S, v)
        L = core.chol(S)
        ll += -0.5 * (m * math.log(2.0 * math.pi) + 2.0 * sum(math.log(L[i][i]) for i in range(m)) + sum(v[i] * sv[i] for i in range(m)))
        xs.append(list(x))
    return RichResult(
        title="Kalman filter (model form)",
        summary_lines=[("n", n), ("state dim", d)],
        payload={
            "estimate": xs[-1][0],
            "state": xs,
            "loglik": ll,
            "n": n,
            "method": "forward predict/update recursion, Kalman (1960)",
        },
    )


def cheatsheet():
    return "klmflt: Kalman forward recursion"
