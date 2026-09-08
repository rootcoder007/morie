# morie.fn -- function file (rootcoder007/morie)
"""G-estimation of a linear structural nested mean model.

Robins, J.M. (1994).  Correcting for non-compliance in randomized
trials using structural nested mean models.  Communications in
Statistics 23:2379-2412.

The linear SNMM posits that removing the treatment received at each
time shifts the mean by a constant per unit,

    E[Y - sum_t psi A_t | H_t] = E[Y(0) | H_t],

so the blipped-down outcome H(psi) = Y - psi sum_t A_t must be
conditionally independent of treatment given history.  G-estimation
solves the estimating equation

    sum_i (A_i - E[A_i | H_i]) (Y_i - psi A_i) = 0,

which for a scalar psi has the closed form

    psi = sum_i (A_i - Abar_i) Y_i / sum_i (A_i - Abar_i) A_i.

No search, no iteration: the linear case is solvable in one step, so
that is what is done here rather than a grid over psi.
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["snm_linear"]


def snm_linear(y, treatment_history, covariate_history=None, time=None,
               propensity=None):
    """Estimate the causal shift psi per unit of cumulative treatment.

    ``treatment_history`` is one row per subject; the cumulative
    treatment sum_t A_t is what the linear blip function acts on.
    ``propensity`` is E[A | H], the fitted expectation of that
    cumulative treatment given history; when omitted the sample mean
    is used, which is correct only when treatment is randomized
    (marginally independent of history).

    Anchor.  Under randomization with the sample mean as the
    propensity, psi coincides exactly with the ordinary least squares
    slope of Y on cumulative treatment -- g-estimation and regression
    agree precisely when there is no confounding to correct.
    ``ols_slope`` is returned so the two can be compared directly.

    Returns
    -------
    RichResult with keys estimate (psi), psi, se, ci_lower, ci_upper,
    ols_slope, residual_treatment, n, method.
    """
    ys = [float(v) for v in y]
    A = [[float(v) for v in row] for row in treatment_history]
    n = len(ys)
    if len(A) != n or n == 0:
        raise ValueError("y and treatment_history must agree and be non-empty")
    a = [sum(row) for row in A]
    if propensity is None:
        abar = sum(a) / n
        ea = [abar] * n
    else:
        ea = [float(v) for v in propensity]
        if len(ea) != n:
            raise ValueError("propensity must have length n")
    r = [a[i] - ea[i] for i in range(n)]
    den = sum(r[i] * a[i] for i in range(n))
    if abs(den) < 1e-12:
        raise ValueError("no residual treatment variation to identify psi")
    psi = sum(r[i] * ys[i] for i in range(n)) / den
    # sandwich standard error of the estimating equation
    u = [r[i] * (ys[i] - psi * a[i]) for i in range(n)]
    meat = sum(v * v for v in u)
    se = math.sqrt(meat) / abs(den) if n > 1 else float("nan")
    ab = sum(a) / n
    yb = sum(ys) / n
    saa = sum((v - ab) ** 2 for v in a)
    ols = (sum((a[i] - ab) * (ys[i] - yb) for i in range(n)) / saa
           if saa > 0 else float("nan"))
    z = 1.959963984540054
    return with_describe_pointer(RichResult(payload={
        "estimate": float(psi), "psi": float(psi), "se": float(se),
        "ci_lower": float(psi - z * se), "ci_upper": float(psi + z * se),
        "ols_slope": float(ols), "residual_treatment": r, "n": n,
        "method": "g-estimation of a linear SNMM (Robins 1994)",
    }), "snmlin")


def cheatsheet():
    return "snmlin: G-estimation of a linear structural nested mean model"


# compact alias per ledger/NAMING.md
snmlinear = snm_linear
