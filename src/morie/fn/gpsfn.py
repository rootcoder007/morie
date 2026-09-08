# morie.fn -- slice s03 (rootcoder007/morie)
"""Sparse GP regression by the FITC approximation.

Source consulted: Snelson, E. and Ghahramani, Z. (2006).  Sparse
Gaussian processes using pseudo-inputs.  *NIPS* 18, 1257-1264, whose
approximation replaces the exact covariance by

    Q = K_nm K_mm^(-1) K_mn + diag( K_nn - K_nm K_mm^(-1) K_mn )

-- the Nystrom low-rank term plus a diagonal correction that restores
the exact marginal variances; the predictive distribution then follows
from the usual Gaussian conditioning with Q + sigma^2 I in place of
K_nn + sigma^2 I.  Quinonero-Candela and Rasmussen (2005), *JMLR* 6,
1939-1959, name this approximation FITC and give the same expression.
Neither was retrievable here as a full text; the covariance is quoted
in its standard published form.

Dropping the diagonal correction gives DTC / the subset-of-regressors
approximation, available as ``kind="dtc"``, so the effect of the
correction is visible rather than assumed.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["gp_sparse_inducing"]


def _rbf(x, y, gamma):
    s = 0.0
    for a in range(len(x)):
        d = x[a] - y[a]
        s += d * d
    return math.exp(-gamma * s)


def _cross(A, B, gamma):
    return [[_rbf(A[i], B[j], gamma) for j in range(len(B))]
            for i in range(len(A))]


def gp_sparse_inducing(X, y, X_test=None, inducing=None, gamma=1.0,
                       sigma2=1e-2, jitter=1e-8, kind="fitc"):
    """FITC (or DTC) sparse GP prediction.

    Returns
    -------
    RichResult with payload:
        estimate : prediction at the first test point
        pred, var
        lam      : the FITC diagonal correction
    """
    Xm = k.mat(X)
    yv = k.vec(y)
    Z = k.mat(inducing) if inducing is not None else Xm
    g = float(gamma)
    n = len(Xm)
    m = len(Z)
    Kmm = _cross(Z, Z, g)
    for i in range(m):
        Kmm[i][i] += float(jitter)
    Knm = _cross(Xm, Z, g)
    # Qnn diagonal, and the FITC correction
    lam = []
    Kinv_rows = []
    for i in range(n):
        w = k.cholsolve(Kmm, Knm[i])
        Kinv_rows.append(w)
        q = 0.0
        for t in range(m):
            q += Knm[i][t] * w[t]
        d = 1.0 - q if kind == "fitc" else 0.0
        lam.append(d if d > 0.0 else 0.0)
    # A = Kmm + Kmn (Lam + s2 I)^-1 Knm ; standard FITC solve
    A = [[Kmm[s][t] for t in range(m)] for s in range(m)]
    rhs = [0.0] * m
    for i in range(n):
        d = lam[i] + float(sigma2)
        for s in range(m):
            for t in range(m):
                A[s][t] += Knm[i][s] * Knm[i][t] / d
            rhs[s] += Knm[i][s] * yv[i] / d
    alpha = k.cholsolve(A, rhs)
    Xt = k.mat(X_test) if X_test is not None else Xm
    Ktm = _cross(Xt, Z, g)
    pred = []
    var = []
    for t in range(len(Xt)):
        s = 0.0
        for a in range(m):
            s += Ktm[t][a] * alpha[a]
        pred.append(s)
        wa = k.cholsolve(A, Ktm[t])
        wb = k.cholsolve(Kmm, Ktm[t])
        qa = 0.0
        qb = 0.0
        for a in range(m):
            qa += Ktm[t][a] * wa[a]
            qb += Ktm[t][a] * wb[a]
        var.append(1.0 - qb + qa)
    return RichResult(
        title="Sparse GP (FITC)",
        summary_lines=[("inducing", m), ("kind", kind)],
        payload={
            "estimate": pred[0] if pred else float("nan"),
            "pred": pred,
            "var": var,
            "lam": lam,
            "alpha": alpha,
            "method": "FITC sparse GP with the diagonal correction (Snelson and Ghahramani 2006)",
        },
    )


def cheatsheet():
    return "gpsfn: Sparse GP via inducing points (FITC)"
