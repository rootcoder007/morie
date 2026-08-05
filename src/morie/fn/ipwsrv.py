# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Inverse probability weighting combined with survey design weights.

DuGoff, Schuler and Stuart (2014), "Generalizing observational study
results: applying propensity score methods to complex surveys", Health
Services Research 49(1):284-303, doi:10.1111/1475-6773.12090.  Their
recommendation is to multiply the propensity-score weight by the survey
design weight, so a unit contributes

    w_i = d_i * ( T_i / pi_i + (1 - T_i) / (1 - pi_i) ),

and to estimate the treatment effect as the difference of the two
weighted (Hajek) group means.  The design weight enters the estimator
and the variance; the effective sample size (Kish) is reported because
the combined weights are typically far more variable than either
component alone.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ipw_with_survey_weights"]


def ipw_with_survey_weights(y, T, weights, propensity):
    """Weighted ATE from combined survey design and propensity weights.

    Parameters
    ----------
    y : array-like
        Outcome.
    T : array-like
        Treatment indicator, 0 or 1.
    weights : array-like or None
        Survey design weights d_i.  None means all ones.
    propensity : array-like
        Estimated propensity scores pi_i, strictly inside (0, 1).
    """
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("ipw_with_survey_weights: y is empty")
    t = core.vec(T)
    pi = core.vec(propensity)
    if len(t) != n or len(pi) != n:
        raise ValueError("ipw_with_survey_weights: y, T and propensity have different lengths")
    d = core.vec(weights) if weights is not None else [1.0] * n
    if len(d) != n:
        raise ValueError("ipw_with_survey_weights: weights and y have different lengths")
    for v in pi:
        if not (0.0 < v < 1.0):
            raise ValueError("ipw_with_survey_weights: propensity must lie strictly in (0, 1)")
    for v in d:
        if v < 0:
            raise ValueError("ipw_with_survey_weights: design weights must be non-negative")
    for v in t:
        if v not in (0.0, 1.0):
            raise ValueError("ipw_with_survey_weights: T must be 0 or 1")
    w = [d[i] * (t[i] / pi[i] + (1.0 - t[i]) / (1.0 - pi[i])) for i in range(n)]
    s1 = sum(w[i] for i in range(n) if t[i] == 1.0)
    s0 = sum(w[i] for i in range(n) if t[i] == 0.0)
    if s1 <= 0 or s0 <= 0:
        raise ValueError("ipw_with_survey_weights: both treatment arms must be non-empty")
    mu1 = sum(w[i] * yv[i] for i in range(n) if t[i] == 1.0) / s1
    mu0 = sum(w[i] * yv[i] for i in range(n) if t[i] == 0.0) / s0
    v1 = sum((w[i] * (yv[i] - mu1)) ** 2 for i in range(n) if t[i] == 1.0) / (s1 * s1)
    v0 = sum((w[i] * (yv[i] - mu0)) ** 2 for i in range(n) if t[i] == 0.0) / (s0 * s0)
    sw = sum(w)
    sw2 = sum(v * v for v in w)
    ess = (sw * sw) / sw2 if sw2 > 0 else 0.0
    return RichResult(
        title="IPW combined with survey weights",
        summary_lines=[("n", n), ("ATE", mu1 - mu0), ("effective n", ess)],
        payload={
            "estimate": mu1 - mu0,
            "mu1": mu1,
            "mu0": mu0,
            "se": math.sqrt(v1 + v0),
            "var1": v1,
            "var0": v0,
            "sum_w": sw,
            "ess": ess,
            "n1": sum(1 for v in t if v == 1.0),
            "n": n,
            "method": "w = d (T/pi + (1-T)/(1-pi)); Hajek difference, DuGoff, Schuler & Stuart (2014)",
        },
    )


def cheatsheet():
    return "ipwsrv: IPW combined with survey weights"


# compact alias per ledger/NAMING.md
ipwsurvey = ipw_with_survey_weights
