# morie.fn -- slice s03 (rootcoder007/morie)
"""Fraction of missing information after multiple imputation.

Source consulted: Rubin, D. B. (1987).  *Multiple Imputation for
Nonresponse in Surveys*, Wiley, section 3.1, and Schafer, J. L. (1997).
*Analysis of Incomplete Multivariate Data*, Chapman and Hall, section
4.3.  With m imputations, within-imputation variance W and
between-imputation variance B,

    T      = W + (1 + 1/m) B                      total variance
    r      = (1 + 1/m) B / W                      relative increase in variance
    lambda = (1 + 1/m) B / T                      fraction of missing information
    nu     = (m - 1) (1 + 1/r)^2                  Rubin's degrees of freedom
    gamma  = (r + 2/(nu + 3)) / (r + 1)           df-adjusted FMI

Neither book was available here as a full text, so the five expressions
are quoted in their standard published form; they are reproduced
identically in Rubin (1987) 3.1.10 and Schafer (1997) 4.3.  ``lambda``
is the quantity the module's own formula line names, so it is what is
returned as ``estimate``; ``gamma`` is returned alongside because it is
the small-m correction that most software reports.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = ["mi_fmi"]


def mi_fmi(between, within, m):
    """Fraction of missing information from the MI variance components.

    Parameters
    ----------
    between : float
        Between-imputation variance B.
    within : float
        Within-imputation variance W.
    m : int
        Number of imputations.

    Returns
    -------
    RichResult with payload:
        estimate : lambda = (1 + 1/m) B / T
        gamma    : (r + 2/(nu + 3)) / (r + 1)
        total    : T = W + (1 + 1/m) B
        r        : relative increase in variance
        df       : Rubin's nu
    """
    b = float(between)
    w = float(within)
    mm = float(m)
    fac = 1.0 + 1.0 / mm if mm > 0.0 else float("nan")
    total = w + fac * b
    lam = (fac * b) / total if total != 0.0 else float("nan")
    r = (fac * b) / w if w != 0.0 else float("inf")
    if r == float("inf") or mm <= 1.0:
        nu = float("inf")
        gamma = lam
    else:
        nu = (mm - 1.0) * (1.0 + 1.0 / r) ** 2 if r > 0.0 else float("inf")
        gamma = (r + 2.0 / (nu + 3.0)) / (r + 1.0)
    return RichResult(
        title="Fraction of missing information",
        summary_lines=[("lambda", lam), ("gamma", gamma)],
        payload={
            "estimate": lam,
            "gamma": gamma,
            "total": total,
            "r": r,
            "df": nu,
            "between": b,
            "within": w,
            "m": mm,
            "method": "Fraction of missing information after multiple imputation",
        },
    )


def cheatsheet():
    return "miefa1: Fraction of missing information"


mifmi = mi_fmi
