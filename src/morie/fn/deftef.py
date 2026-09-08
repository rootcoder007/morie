# morie.fn -- function file (rootcoder007/morie)
"""Kish design effect from a pair of variances."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["design_effect"]


def design_effect(design_var, srs_var):
    """
    Design effect DEFF from the two variances

    Formula: DEFF = Var_design / Var_SRS

    DEFT = sqrt(DEFF) is the standard-error inflation factor, and
    n_eff = n / DEFF the effective sample size.  Both arguments may be
    vectors, in which case the ratio is taken element by element.

    Parameters
    ----------
    design_var : array-like
        Variance of the estimator under the realised complex design.
    srs_var : array-like
        Variance under simple random sampling of the same size.

    Returns
    -------
    result : dict
        Keys: estimate (DEFF), deff, deft, n.

    References
    ----------
    Kish (1965), Survey Sampling, Wiley, section 8.2.
    """
    d = core.vec(design_var)
    s = core.vec(srs_var)
    if not d or not s:
        raise ValueError("empty input: both variances are required")
    if len(d) != len(s) and len(d) != 1 and len(s) != 1:
        raise ValueError("design_var and srs_var must have the same length")
    m = max(len(d), len(s))
    if len(d) == 1:
        d = d * m
    if len(s) == 1:
        s = s * m
    for v in s:
        if not (v > 0.0):
            raise ValueError("srs_var must be strictly positive")
    deff = [d[i] / s[i] for i in range(m)]
    deft = [math.sqrt(v) if v >= 0.0 else float("nan") for v in deff]
    return RichResult(payload={
        "estimate": deff[0] if m == 1 else sum(deff) / m,
        "deff": deff,
        "deft": deft,
        "n": m,
        "method": "Kish design effect DEFF = Var_design / Var_SRS",
    })


def cheatsheet():
    return "deftef: Kish design effect DEFF = Var_design / Var_SRS"


# compact alias per ledger/NAMING.md
designeffect = design_effect
