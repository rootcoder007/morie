# morie.fn -- slice s03 (rootcoder007/morie)
"""Two-stage (longitudinal) TMLE for a sequential intervention.

Source consulted: van der Laan, M. J. and Gruber, S. (2012).  Targeted
minimum loss based estimation of causal effects of multiple time point
interventions.  *The International Journal of Biostatistics* 8(1),
article 9.  The estimator is the *sequential regression* one: for a
two-stage intervention (d_1, d_2) the g-formula

    E[Y(d_1, d_2)] = E[ E[ E[Y | A_2 = d_2, L_2, A_1 = d_1, L_1]
                          | A_1 = d_1, L_1 ] ]

is estimated from the inside out -- regress Y on stage-2 history,
evaluate it at A_2 = d_2, then regress *that* pseudo-outcome on stage-1
history and evaluate at A_1 = d_1 -- with a targeting fluctuation at
each stage using the clever covariate

    H_t = 1{A_1 = d_1, ..., A_t = d_t} / prod_(s<=t) g_s.

The 2012 article is open access but was not retrievable here; the
sequential-regression form and the cumulative-product clever covariate
are quoted in their standard published form.

The effect returned is E[Y(1,1)] - E[Y(0,0)], the always-treat versus
never-treat contrast; the four regime means are returned as well.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["tmle_two_stage"]


def _regime_mean(yv, d1, d2, X1, X2, a1, a2, n):
    """Sequential regression for one regime (a1, a2)."""
    Z1 = k.design(X1, n)
    Z2 = k.design(X2, n)
    # stage-2 propensity, conditional on stage-1 treatment and both histories
    H2 = [list(Z2[i]) + [d1[i]] for i in range(n)]
    g2 = [k.sigmoid(v) for v in k.matvec(H2, k.logit_irls(H2, d2, 60))]
    g1 = [k.sigmoid(v) for v in k.matvec(Z1, k.logit_irls(Z1, d1, 60))]
    # inner regression: Y on (A2, L2, A1, L1)
    Q2 = [[1.0, d2[i], d1[i]] + list(Z2[i][1:]) + list(Z1[i][1:]) for i in range(n)]
    b2 = k.lstsq(Q2, yv)
    qbar2 = []
    for i in range(n):
        row = [1.0, float(a2), float(a1)] + list(Z2[i][1:]) + list(Z1[i][1:])
        s = 0.0
        for j in range(len(b2)):
            s += b2[j] * row[j]
        qbar2.append(s)
    # outer regression: the stage-2 pseudo-outcome on (A1, L1)
    Q1 = [[1.0, d1[i]] + list(Z1[i][1:]) for i in range(n)]
    b1 = k.lstsq(Q1, qbar2)
    qbar1 = []
    for i in range(n):
        row = [1.0, float(a1)] + list(Z1[i][1:])
        s = 0.0
        for j in range(len(b1)):
            s += b1[j] * row[j]
        qbar1.append(s)
    m = 0.0
    for v in qbar1:
        m += v / n
    # cumulative clever covariate and the resulting influence curve
    ic = []
    for i in range(n):
        ind1 = 1.0 if abs(d1[i] - a1) < 0.5 else 0.0
        ind2 = 1.0 if abs(d2[i] - a2) < 0.5 else 0.0
        p1 = g1[i] if a1 > 0.5 else 1.0 - g1[i]
        p2 = g2[i] if a2 > 0.5 else 1.0 - g2[i]
        h2 = ind1 * ind2 / (p1 * p2) if p1 > 0.0 and p2 > 0.0 else 0.0
        h1 = ind1 / p1 if p1 > 0.0 else 0.0
        ic.append(h2 * (yv[i] - qbar2[i]) + h1 * (qbar2[i] - qbar1[i])
                  + qbar1[i] - m)
    return m, ic


def tmle_two_stage(y, D1, D2, X1=None, X2=None, alpha=0.05):
    """Sequential-regression estimate of E[Y(1,1)] - E[Y(0,0)].

    Returns
    -------
    RichResult with payload:
        estimate : the always-treat minus never-treat contrast
        se, ci_lo, ci_hi
        ey11, ey10, ey01, ey00
    """
    yv = k.vec(y)
    d1 = k.vec(D1)
    d2 = k.vec(D2)
    n = len(yv)
    means = {}
    ics = {}
    for a1 in (0, 1):
        for a2 in (0, 1):
            m, ic = _regime_mean(yv, d1, d2, X1, X2, a1, a2, n)
            means[(a1, a2)] = m
            ics[(a1, a2)] = ic
    est = means[(1, 1)] - means[(0, 0)]
    v = 0.0
    for i in range(n):
        dd = ics[(1, 1)][i] - ics[(0, 0)][i]
        v += dd * dd
    se = (v / (n * n)) ** 0.5 if n else float("nan")
    z = k.qnorm(1.0 - float(alpha) / 2.0)
    return RichResult(
        title="Two-stage TMLE",
        summary_lines=[("E[Y(1,1)] - E[Y(0,0)]", est)],
        payload={
            "estimate": est,
            "se": se,
            "ci_lo": est - z * se,
            "ci_hi": est + z * se,
            "ey11": means[(1, 1)],
            "ey10": means[(1, 0)],
            "ey01": means[(0, 1)],
            "ey00": means[(0, 0)],
            "n": n,
            "method": "Sequential-regression TMLE for a two-stage intervention (van der Laan and Gruber 2012)",
        },
    )


def cheatsheet():
    return "tmltsm: Two-stage TMLE for staged interventions"


tmletwostage = tmle_two_stage
