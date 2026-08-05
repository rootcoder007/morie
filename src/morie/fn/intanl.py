# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Treatment x covariate interaction in a marginal structural model.

Hernan and Robins (2020), *Causal Inference: What If*, Chapman &
Hall/CRC, section 12.5 "Effect modification and marginal structural
models", p. 171.  The marginal structural mean model with an effect
modifier V is

    E[Y^a | V] = beta0 + beta1 a + beta2 V a + beta3 V,

and the book's instruction is to "estimate the model parameters by
fitting the linear regression model E[Y|A,V] = theta0 + theta1 A +
theta2 V A + theta3 V via weighted least squares with IP weights W^A or
SW^A".  Additive effect modification is present if beta2 != 0, so
beta2 is the reported estimate.

Because the weights are estimated, the model-based standard error is
anticonservative; the robust (sandwich) standard error is reported, as
the book requires for IP-weighted estimates.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["interaction_analysis"]


def interaction_analysis(y, A, V, H):
    """Effect modification of A by V in an IP-weighted marginal structural model.

    Parameters
    ----------
    y : array-like
        Outcome.
    A : array-like
        Treatment.
    V : array-like
        Effect modifier.
    H : array-like or None
        IP weights W^A (or SW^A).  None or ones give the unweighted fit.
    """
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("interaction_analysis: y is empty")
    a = core.vec(A)
    v = core.vec(V)
    if len(a) != n or len(v) != n:
        raise ValueError("interaction_analysis: y, A and V have different lengths")
    w = core.vec(H) if H is not None else [1.0] * n
    if len(w) != n:
        raise ValueError("interaction_analysis: H and y have different lengths")
    for t in w:
        if t < 0:
            raise ValueError("interaction_analysis: weights must be non-negative")
    Z = [[1.0, a[i], v[i] * a[i], v[i]] for i in range(n)]
    p = 4
    if n <= p:
        raise ValueError("interaction_analysis: need more than four observations")
    ZtWZ = [[0.0] * p for _ in range(p)]
    ZtWy = [0.0] * p
    for i in range(n):
        for r in range(p):
            ZtWy[r] += Z[i][r] * w[i] * yv[i]
            for c in range(p):
                ZtWZ[r][c] += Z[i][r] * w[i] * Z[i][c]
    beta = core.cholsolve(ZtWZ, ZtWy)
    e = [yv[i] - sum(Z[i][r] * beta[r] for r in range(p)) for i in range(n)]
    inv = [core.cholsolve(ZtWZ, [1.0 if r == j else 0.0 for r in range(p)]) for j in range(p)]
    meat = [[0.0] * p for _ in range(p)]
    for i in range(n):
        k = w[i] * w[i] * e[i] * e[i]
        for r in range(p):
            for c in range(p):
                meat[r][c] += k * Z[i][r] * Z[i][c]
    tmp = core.matmul(inv, meat)
    vcv = core.matmul(tmp, inv)
    se = [math.sqrt(vcv[j][j]) for j in range(p)]
    return RichResult(
        title="Treatment x covariate interaction in an MSM",
        summary_lines=[("n", n), ("beta_av", beta[2]), ("se", se[2])],
        payload={
            "estimate": beta[2],
            "beta0": beta[0],
            "beta_a": beta[1],
            "beta_av": beta[2],
            "beta_v": beta[3],
            "se": se[2],
            "se_a": se[1],
            "se0": se[0],
            "se_v": se[3],
            "n": n,
            "method": "E[Y^a|V] = b0 + b1 a + b2 V a + b3 V by IP-weighted least squares, Hernan & Robins (2020) s.12.5",
        },
    )


def cheatsheet():
    return "intanl: Treatment x covariate interaction in MSM"


# compact alias per ledger/NAMING.md
interactionanalysis = interaction_analysis
