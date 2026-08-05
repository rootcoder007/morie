# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Robustness of an inference: percent bias to invalidate, RIR and ITCV.

Frank, Maroulis, Duong and Kelcey (2013), "What would it take to change
an inference? Using Rubin's causal model to interpret the robustness of
causal inferences", Educational Evaluation and Policy Analysis
35(4):437-460, doi:10.3102/0162373713493129, give the replacement-of-
cases index.  With the threshold estimate delta# = t_crit * se,

    percent bias to invalidate = 100 * (1 - delta# / |estimate|),
    RIR = n * (1 - delta# / |estimate|).

Frank (2000), "Impact of a confounding variable on a regression
coefficient", Sociological Methods & Research 29(2):147-194,
doi:10.1177/0049124100029002001, gives the impact threshold for a
confounding variable.  Writing r# = t_crit / sqrt(t_crit^2 + df) for
the correlation corresponding to the significance threshold and
r_xy = t / sqrt(t^2 + df) for the partial correlation of the estimate,

    ITCV = (|r_xy| - r#) / (1 - r#),

and an omitted variable correlated r_cv = sqrt(ITCV) with both the
predictor and the outcome would exactly overturn the inference.

Degrees of freedom follow the reference implementation of these indices
(the konfound R package): df = n - n_covariates - 3.  The published
worked example pkonfound(est = 2, se = 0.4, n = 100, n_covariates = 3)
prints delta# = 0.794, 60.29 percent bias, RIR = 60, r# = 0.201 and
r_cv = 0.568; those five numbers are the anchors used in testing.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401
from ._rrng_core import qt

from ._richresult import RichResult

__all__ = ["konfound"]


def konfound(est, se, n, threshold, n_covariates=0):
    """Percent bias to invalidate, RIR and ITCV for a regression estimate.

    Parameters
    ----------
    est : float
        Unstandardised coefficient.
    se : float
        Its standard error.
    n : int
        Sample size.
    threshold : float
        Two-sided significance level alpha (e.g. 0.05).
    n_covariates : int
        Number of covariates other than the predictor of interest.
    """
    est = float(est)
    se = float(se)
    n = int(n)
    alpha = float(threshold)
    k = int(n_covariates)
    if se <= 0:
        raise ValueError("konfound: se must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("konfound: threshold must be a significance level in (0, 1)")
    df = n - k - 3
    if df <= 0:
        raise ValueError("konfound: not enough degrees of freedom")
    tcrit = qt(1.0 - alpha / 2.0, df)
    delta = tcrit * se
    t = est / se
    if est == 0.0:
        raise ValueError("konfound: estimate is exactly zero")
    frac = 1.0 - delta / abs(est)
    significant = abs(t) > tcrit
    rxy = t / math.sqrt(t * t + df)
    rcrit = tcrit / math.sqrt(tcrit * tcrit + df)
    itcv = (abs(rxy) - rcrit) / (1.0 - rcrit)
    rcv = math.sqrt(itcv) if itcv >= 0 else -math.sqrt(-itcv)
    return RichResult(
        title="Robustness of an inference (konfound)",
        summary_lines=[
            ("percent bias to invalidate", 100.0 * frac),
            ("RIR", n * frac),
            ("ITCV", itcv),
        ],
        payload={
            "estimate": 100.0 * frac,
            "pct_bias": 100.0 * frac,
            "rir": n * frac,
            "rir_cases": float(math.floor(n * frac)) if frac >= 0 else float(math.ceil(n * frac)),
            "delta_threshold": delta,
            "t": t,
            "t_crit": tcrit,
            "r_xy": rxy,
            "r_crit": rcrit,
            "itcv": itcv,
            "r_cv": rcv,
            "impact": rcv * rcv if itcv >= 0 else -(rcv * rcv),
            "df": df,
            "significant": 1.0 if significant else 0.0,
            "n": n,
            "method": "Frank et al (2013) percent bias / RIR and Frank (2000) ITCV",
        },
    )


def cheatsheet():
    return "konfound: Konfound robustness % bias to invalidate"
