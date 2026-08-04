# morie.fn -- slice s03 (rootcoder007/morie)
"""Outcome-only (g-computation) estimation, and its TMLE correction.

Source consulted: Robins, J. (1986).  A new approach to causal inference
in mortality studies with a sustained exposure period.  *Mathematical
Modelling* 7(9-12), 1393-1512, which introduces the g-formula; and
van der Laan, M. J. and Rubin, D. (2006), *The International Journal of
Biostatistics* 2(1), article 11, for the targeting step.  The
outcome-only estimator is the plug-in

    psi^(gcomp) = (1/n) sum_i [ Qbar(1, X_i) - Qbar(0, X_i) ]

which uses no propensity score at all -- so it is consistent when the
outcome regression is right, whatever the treatment mechanism, and that
is exactly the robustness the module's own description claims.  Neither
source was retrievable here as a full text; the g-formula plug-in and
the targeting step are quoted in their standard published form.

The targeted estimate is returned alongside so the two can be compared:
their difference is the entire contribution of the propensity score, and
a large gap is a warning that the outcome model is doing work the data
do not support.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["tmle_outcome_only_regr"]


def tmle_outcome_only_regr(y, D, X=None, alpha=0.05):
    """G-computation ATE, with the TMLE-targeted value for comparison.

    Returns
    -------
    RichResult with payload:
        estimate  : the g-computation ATE
        psi_tmle  : the targeted ATE
        gap       : psi_tmle - estimate
        se        : influence-function SE of the g-computation estimate
    """
    yv = k.vec(y)
    d = k.vec(D)
    n = len(yv)
    Z = k.design(X, n)
    Q = [[1.0, d[i]] + list(Z[i][1:]) for i in range(n)]
    b = k.lstsq(Q, yv)
    q1 = []
    q0 = []
    for i in range(n):
        r1 = [1.0, 1.0] + list(Z[i][1:])
        r0 = [1.0, 0.0] + list(Z[i][1:])
        s1 = 0.0
        s0 = 0.0
        for j in range(len(b)):
            s1 += b[j] * r1[j]
            s0 += b[j] * r0[j]
        q1.append(s1)
        q0.append(s0)
    psi = 0.0
    for i in range(n):
        psi += (q1[i] - q0[i]) / n
    resid = []
    for i in range(n):
        fitted = q1[i] if d[i] > 0.5 else q0[i]
        resid.append(yv[i] - fitted)
    ic = [(q1[i] - q0[i]) - psi for i in range(n)]
    v = 0.0
    for x in ic:
        v += x * x
    se = math.sqrt(v / (n * n)) if n else float("nan")
    tm = k.tmle_ate(yv, d, X)
    z = k.qnorm(1.0 - float(alpha) / 2.0)
    return RichResult(
        title="Outcome-only (g-computation) ATE",
        summary_lines=[("g-comp", psi), ("TMLE", tm["psi"])],
        payload={
            "estimate": psi,
            "psi_gcomp": psi,
            "psi_tmle": tm["psi"],
            "gap": tm["psi"] - psi,
            "se": se,
            "ci_lo": psi - z * se,
            "ci_hi": psi + z * se,
            "rmse_resid": (sum([r * r for r in resid]) / n) ** 0.5 if n else float("nan"),
            "n": n,
            "method": "G-computation plug-in (Robins 1986) with the TMLE-targeted value for comparison",
        },
    )


def cheatsheet():
    return "tmlovo: Outcome-only TMLE -- robust if g misspecified"
