# morie.fn -- function file (rootcoder007/morie)
"""Targeted maximum likelihood estimator of the ATE.

van der Laan, M.J. & Rubin, D. (2006).  Targeted maximum likelihood
learning.  Int J Biostat 2(1), Article 11.

The targeting step implemented is the one specified in Gruber, S. &
van der Laan, M.J. (2012), tmle: An R Package for Targeted Maximum
Likelihood Estimation, J Stat Softw 51(13) pp.5-6, fetched and read
in full.  Clever covariates (2)-(3):

    H0*(A, W) = I(A = 0) / g(0 | W),
    H1*(A, W) = I(A = 1) / g(1 | W),

and the fluctuation

    logit(Q1n(A, W)) = logit(Q0n(A, W)) + e0 H0* + e1 H1*,

with e fitted "by a logistic regression of Y on H0*, H1*, with offset
logit(Q0n(A, W))".
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["causal_tmle_targeted"]

_LO = 1e-12


def _logit(p):
    p = min(max(float(p), _LO), 1.0 - _LO)
    return math.log(p / (1.0 - p))


def _expit(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    t = math.exp(z)
    return t / (1.0 + t)


def causal_tmle_targeted(y, T, ps, Q1, Q0, n_iter=100):
    """Targeted ATE: fluctuate the outcome fit along the efficient
    influence curve, then take the g-computation contrast.

    The initial fit Q1, Q0 is whatever the caller's outcome model
    produced; the fluctuation moves it just enough to solve the
    efficient influence curve equation, which is what buys the
    influence-curve standard error and the double robustness.

    ``y`` must lie in [0, 1] -- bound a continuous outcome onto the
    unit interval first, as the package does, so the logistic
    fluctuation stays inside the model space.

    The fluctuation parameter is fitted by Newton-Raphson on the
    weighted logistic score with a fixed iteration count and no
    tolerance-driven early exit, so both language arms follow the
    same arithmetic path.

    Anchor.  When the initial outcome fit already solves the score,
    the fitted epsilon is 0 and the targeted estimate equals the
    plain g-computation estimate ``plugin``; the two are returned
    side by side so that can be seen rather than assumed.

    Returns
    -------
    RichResult with keys estimate (ATE_TMLE), ATE_TMLE, IF, se,
    ci_lower, ci_upper, epsilon, EY1, EY0, plugin, n, method.
    """
    ys = [float(v) for v in y]
    tt = [float(v) for v in T]
    g = [float(v) for v in ps]
    q1 = [float(v) for v in Q1]
    q0 = [float(v) for v in Q0]
    n = len(ys)
    if not (len(tt) == len(g) == len(q1) == len(q0) == n) or n == 0:
        raise ValueError("all inputs must be non-empty and the same length")
    if any(not 0.0 < p < 1.0 for p in g):
        raise ValueError("propensities must lie strictly in (0, 1)")
    if any(v < 0.0 or v > 1.0 for v in ys):
        raise ValueError("y must be bounded in [0, 1] for the fluctuation")
    # clever covariates (2)-(3)
    h1 = [tt[i] / g[i] for i in range(n)]
    h0 = [(1.0 - tt[i]) / (1.0 - g[i]) for i in range(n)]
    qa = [q1[i] if tt[i] == 1.0 else q0[i] for i in range(n)]
    off = [_logit(v) for v in qa]
    e0 = 0.0
    e1 = 0.0
    for _ in range(int(n_iter)):
        s0 = s1 = 0.0
        a00 = a01 = a11 = 0.0
        for i in range(n):
            mu = _expit(off[i] + e0 * h0[i] + e1 * h1[i])
            r = ys[i] - mu
            w = mu * (1.0 - mu)
            s0 += h0[i] * r
            s1 += h1[i] * r
            a00 += w * h0[i] * h0[i]
            a01 += w * h0[i] * h1[i]
            a11 += w * h1[i] * h1[i]
        det = a00 * a11 - a01 * a01
        if abs(det) < 1e-14:
            break
        e0 += (a11 * s0 - a01 * s1) / det
        e1 += (a00 * s1 - a01 * s0) / det
    q1s = [_expit(_logit(q1[i]) + e1 / g[i]) for i in range(n)]
    q0s = [_expit(_logit(q0[i]) + e0 / (1.0 - g[i])) for i in range(n)]
    ey1 = sum(q1s) / n
    ey0 = sum(q0s) / n
    ate = ey1 - ey0
    ic = [h1[i] * (ys[i] - q1s[i]) - h0[i] * (ys[i] - q0s[i])
          + (q1s[i] - q0s[i]) - ate for i in range(n)]
    var = sum(v * v for v in ic) / (n * n)
    se = math.sqrt(var)
    z = 1.959963984540054
    return with_describe_pointer(RichResult(payload={
        "estimate": float(ate), "ATE_TMLE": float(ate), "IF": ic,
        "se": float(se), "ci_lower": float(ate - z * se),
        "ci_upper": float(ate + z * se),
        "epsilon": [float(e0), float(e1)],
        "EY1": float(ey1), "EY0": float(ey0),
        "plugin": float(sum(q1[i] - q0[i] for i in range(n)) / n), "n": n,
        "method": "TMLE of the ATE (van der Laan & Rubin 2006)",
    }), "caustmle")


def cheatsheet():
    return "caustmle: Targeted maximum likelihood estimator of the ATE"


# compact alias per ledger/NAMING.md
tmleate = causal_tmle_targeted
